"""Two-stage hyperparameter search for the Mobilise-D longitudinal MTL model.

Stage 1:
    Tune lambda_fl, lambda_l, and lambda_g
    with lambda_r = 0 and lambda_a = 0.

Stage 2:
    Fix the best Stage-1 parameters and tune
    lambda_r and lambda_a.

Participant-level split:
    70% training
    10% validation
    20% test

Evaluation:
    Overall performance for each clinical prediction objective:
        - nMSE (lower is better)
        - wR   (higher is better)

    Visit-specific performance:
        - rMSE (lower is better)

The validation set is used exclusively for hyperparameter selection.
The test set is used exclusively for final performance reporting.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from demmo.data import (
    OBJECTIVES,
    VISITS,
    load_mobilise_data,
    participant_train_test_split,
)

from demmo.optimizer import CrossDiseaseLongitudinalMTL

RELATION_LABELS = (
    "PD H&Y",
    "PD MDS-UPDRS III",
    "MS EDSS",
    "PFF SPPB impairment",
)


# ============================================================
# Utility functions
# ============================================================


def parse_grid(text: str) -> list[float]:
    """Parse comma-separated hyperparameter values."""

    values = [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]

    if not values:
        raise ValueError(
            "Hyperparameter grid cannot be empty."
        )

    return values


def parameter_name(
        lambda_fl: float,
        lambda_l: float,
        lambda_g: float,
        lambda_r: float,
        lambda_a: float,
) -> str:
    """Create a folder name from the five hyperparameters."""

    return (
        f"fl_{lambda_fl:g}"
        f"_l_{lambda_l:g}"
        f"_g_{lambda_g:g}"
        f"_r_{lambda_r:g}"
        f"_a_{lambda_a:g}"
    )


# ============================================================
# Participant-level 70:10:20 split
# ============================================================


def participant_train_validation_test_split(
        data,
        *,
        random_seed: int = 42,
) -> tuple[
    dict[str, set],
    dict[str, set],
    dict[str, set],
]:
    """Create participant-level 70:10:20 splits.

    Step 1:
        80% development
        20% test

    Step 2:
        Split 12.5% of the development participants into
        validation.

    Therefore:

        train      = 0.80 * 0.875 = 0.70
        validation = 0.80 * 0.125 = 0.10
        test       = 0.20
    """

    development_ids, test_ids = (
        participant_train_test_split(
            data,
            test_fraction=0.20,
            random_seed=random_seed,
        )
    )

    train_ids: dict[str, set] = {}
    validation_ids: dict[str, set] = {}

    validation_fraction_of_development = 0.125

    for disease_index, disease in enumerate(
            sorted(development_ids)
    ):

        participants = np.asarray(
            sorted(
                development_ids[disease]
            ),
            dtype=object,
        )

        if len(participants) < 2:
            raise ValueError(
                f"Not enough participants in {disease} "
                "to construct train/validation splits."
            )

        rng = np.random.default_rng(
            random_seed
            + 1000
            + disease_index
        )

        permutation = rng.permutation(
            len(participants)
        )

        n_validation = int(
            round(
                validation_fraction_of_development
                * len(participants)
            )
        )

        n_validation = max(
            1,
            min(
                n_validation,
                len(participants) - 1,
            ),
        )

        validation_indices = permutation[
            :n_validation
        ]

        training_indices = permutation[
            n_validation:
        ]

        validation_ids[disease] = set(
            participants[
                validation_indices
            ].tolist()
        )

        train_ids[disease] = set(
            participants[
                training_indices
            ].tolist()
        )

    return (
        train_ids,
        validation_ids,
        test_ids,
    )


# ============================================================
# Standardisation
# ============================================================


def fit_feature_scalers(
        data,
        participant_ids: dict[str, set],
) -> dict[
    str,
    tuple[np.ndarray, np.ndarray],
]:
    """Fit disease-specific DMO scalers pooled across visits."""

    feature_scalers = {}

    for disease, frame in (
            data.cohort_frames.items()
    ):

        if disease not in participant_ids:
            continue

        training = frame[
            frame["participantid"].isin(
                participant_ids[disease]
            )
        ]

        if training.empty:
            raise ValueError(
                f"No training observations "
                f"available for {disease}."
            )

        values = training.loc[
            :,
            data.dmo_columns,
        ].to_numpy(
            dtype=float
        )

        mean = values.mean(
            axis=0
        )

        scale = values.std(
            axis=0,
            ddof=0,
        )

        scale[
            scale == 0.0
            ] = 1.0

        feature_scalers[disease] = (
            mean,
            scale,
        )

    return feature_scalers


def fit_outcome_scalers(
        data,
        participant_ids: dict[str, set],
) -> dict[
    str,
    tuple[float, float],
]:
    """Fit outcome scalers pooled across visits."""

    outcome_scalers = {}

    for spec in OBJECTIVES:

        frame = data.objective_frames[
            spec.name
        ]

        training = frame[
            frame["participantid"].isin(
                participant_ids[
                    spec.disease
                ]
            )
        ]

        if training.empty:
            raise ValueError(
                f"No training observations "
                f"available for {spec.name}."
            )

        target_mean = float(
            training[
                "target"
            ].mean()
        )

        target_scale = float(
            training[
                "target"
            ].std(
                ddof=0
            )
        )

        if target_scale == 0.0:
            target_scale = 1.0

        outcome_scalers[
            spec.name
        ] = (
            target_mean,
            target_scale,
        )

    return outcome_scalers


# ============================================================
# Construct longitudinal data
# ============================================================


def build_standardised_longitudinal_data(
        data,
        participant_ids: dict[str, set],
        feature_scalers,
        outcome_scalers,
) -> tuple[
    list[list[np.ndarray]],
    list[list[np.ndarray]],
]:
    """Construct standardised X and y for every objective/visit."""

    all_x = []
    all_y = []

    for spec in OBJECTIVES:

        frame = data.objective_frames[
            spec.name
        ]

        (
            feature_mean,
            feature_scale,
        ) = feature_scalers[
            spec.disease
        ]

        (
            target_mean,
            target_scale,
        ) = outcome_scalers[
            spec.name
        ]

        objective_x = []
        objective_y = []

        for visit in VISITS:

            visit_frame = frame[
                (
                        frame[
                            "visit.number"
                        ]
                        == visit
                )
                &
                (
                    frame[
                        "participantid"
                    ].isin(
                        participant_ids[
                            spec.disease
                        ]
                    )
                )
                ].sort_values(
                "participantid"
            )

            if visit_frame.empty:
                raise ValueError(
                    f"No observations for "
                    f"{spec.name} at {visit}."
                )

            features = visit_frame.loc[
                :,
                data.dmo_columns,
            ].to_numpy(
                dtype=float
            )

            targets = visit_frame[
                "target"
            ].to_numpy(
                dtype=float
            )

            x = (
                        features
                        - feature_mean
                ) / feature_scale

            y = (
                        targets
                        - target_mean
                ) / target_scale

            objective_x.append(
                x
            )

            objective_y.append(
                y
            )

        all_x.append(
            objective_x
        )

        all_y.append(
            objective_y
        )

    return (
        all_x,
        all_y,
    )


# ============================================================
# Evaluation metrics
# ============================================================


def rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:
    """Root Mean Squared Error."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                        y_true
                        - y_pred
                ) ** 2
            )
        )
    )


def pearson_correlation(
        y_true: np.ndarray,
        y_pred: np.ndarray,
) -> float:
    """Pearson correlation."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if len(y_true) < 2:
        return np.nan

    true_std = float(
        np.std(
            y_true,
            ddof=0,
        )
    )

    pred_std = float(
        np.std(
            y_pred,
            ddof=0,
        )
    )

    if (
            true_std == 0.0
            or pred_std == 0.0
    ):
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


def overall_metrics_from_visits(
        visit_data: list[
            tuple[
                np.ndarray,
                np.ndarray,
            ]
        ],
) -> dict[str, float]:
    """Calculate overall nMSE and weighted R-value across visits.

    Definitions:

        nMSE =
            [
                sum_t
                ||y_t - yhat_t||_2^2
                / variance(y_t)
            ]
            /
            [
                sum_t n_t
            ]

        wR =
            [
                sum_t
                Corr(y_t, yhat_t) * n_t
            ]
            /
            [
                sum_t n_t
            ]
    """

    nmse_numerator = 0.0
    wr_numerator = 0.0
    total_n = 0

    wr_is_valid = True

    for (
            y_true,
            y_pred,
    ) in visit_data:

        y_true = np.asarray(
            y_true,
            dtype=float,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float,
        )

        n = len(
            y_true
        )

        if n == 0:
            continue

        errors = (
                y_true
                - y_pred
        )

        squared_error = float(
            errors
            @ errors
        )

        variance = float(
            np.var(
                y_true,
                ddof=0,
            )
        )

        if variance <= 0.0:
            raise ValueError(
                "Cannot calculate nMSE because "
                "the outcome variance is zero "
                "for at least one visit."
            )

        nmse_numerator += (
                squared_error
                / variance
        )

        corr = pearson_correlation(
            y_true,
            y_pred,
        )

        if np.isnan(
                corr
        ):
            wr_is_valid = False
        else:
            wr_numerator += (
                    corr
                    * n
            )

        total_n += n

    if total_n == 0:
        raise ValueError(
            "No observations available "
            "for metric calculation."
        )

    nmse = (
            nmse_numerator
            / total_n
    )

    if wr_is_valid:
        wr = (
                wr_numerator
                / total_n
        )
    else:
        wr = np.nan

    return {
        "n":
            int(
                total_n
            ),

        "nMSE":
            float(
                nmse
            ),

        "wR":
            float(
                wr
            ),
    }


# ============================================================
# Validation evaluation for hyperparameter selection
# ============================================================


def validation_score(
        model: CrossDiseaseLongitudinalMTL,
        x_validation: list[list[np.ndarray]],
        y_validation: list[list[np.ndarray]],
) -> tuple[
    float,
    float,
    list[dict[str, object]],
]:
    """Calculate validation nMSE and wR.

    Each clinical prediction objective receives its own
    overall nMSE and wR across the five visits.

    Hyperparameter ranking:
        1. Mean nMSE across objectives, lower is better.
        2. Mean wR across objectives, higher is better.
    """

    rows = []

    nmse_values = []
    wr_values = []

    for objective_index, spec in enumerate(
            OBJECTIVES
    ):

        visit_data = []

        for visit_index, visit in enumerate(
                VISITS
        ):
            x = x_validation[
                objective_index
            ][
                visit_index
            ]

            y_true = y_validation[
                objective_index
            ][
                visit_index
            ]

            y_pred = (
                    x
                    @ model.coef_[
                        objective_index,
                        :,
                        visit_index,
                    ]
            )

            visit_data.append(
                (
                    y_true,
                    y_pred,
                )
            )

        result = (
            overall_metrics_from_visits(
                visit_data
            )
        )

        nmse_values.append(
            result[
                "nMSE"
            ]
        )

        wr_values.append(
            result[
                "wR"
            ]
        )

        rows.append(
            {
                "objective":
                    spec.name,

                "disease":
                    spec.disease,

                "n":
                    result[
                        "n"
                    ],

                "nMSE":
                    result[
                        "nMSE"
                    ],

                "wR":
                    result[
                        "wR"
                    ],
            }
        )

    mean_nmse = float(
        np.mean(
            nmse_values
        )
    )

    valid_wr = [
        value
        for value in wr_values
        if not np.isnan(
            value
        )
    ]

    mean_wr = (
        float(
            np.mean(
                valid_wr
            )
        )
        if valid_wr
        else np.nan
    )

    return (
        mean_nmse,
        mean_wr,
        rows,
    )


# ============================================================
# Model fitting
# ============================================================


def fit_candidate(
        x_train,
        y_train,
        *,
        lambda_fl: float,
        lambda_l: float,
        lambda_g: float,
        lambda_r: float,
        lambda_a: float,
        verbose: bool,
) -> CrossDiseaseLongitudinalMTL:
    """Fit one hyperparameter configuration."""

    model = (
        CrossDiseaseLongitudinalMTL(
            lambda_fl=lambda_fl,
            lambda_l=lambda_l,
            lambda_g=lambda_g,
            lambda_r=lambda_r,
            lambda_a=lambda_a,
            verbose=verbose,
        )
        .fit(
            x_train,
            y_train,
        )
    )

    return model


# ============================================================
# Stage 1 hyperparameter search
# ============================================================


def stage1_search(
        x_train,
        y_train,
        x_validation,
        y_validation,
        *,
        fl_grid: list[float],
        l_grid: list[float],
        g_grid: list[float],
        output_dir: Path,
        verbose: bool,
) -> tuple[
    float,
    float,
    float,
    pd.DataFrame,
]:
    """Tune FL, Lasso and Group-Lasso penalties.

    Relation-learning block is disabled:

        lambda_r = 0
        lambda_a = 0
    """

    combinations = list(
        itertools.product(
            fl_grid,
            l_grid,
            g_grid,
        )
    )

    total = len(
        combinations
    )

    results = []

    print(
        "\n"
        "============================================================"
    )

    print(
        "STAGE 1: Within-objective longitudinal regularisation"
    )

    print(
        "lambda_r = 0, lambda_a = 0"
    )

    print(
        f"Number of combinations: {total}"
    )

    print(
        "============================================================\n"
    )

    for index, (
            lambda_fl,
            lambda_l,
            lambda_g,
    ) in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"[Stage 1: {index}/{total}] "
            f"FL={lambda_fl:g}, "
            f"L={lambda_l:g}, "
            f"G={lambda_g:g}"
        )

        model = fit_candidate(
            x_train,
            y_train,

            lambda_fl=
            lambda_fl,

            lambda_l=
            lambda_l,

            lambda_g=
            lambda_g,

            lambda_r=
            0.0,

            lambda_a=
            0.0,

            verbose=
            verbose,
        )

        (
            validation_nmse,
            validation_wr,
            objective_rows,
        ) = validation_score(
            model,
            x_validation,
            y_validation,
        )

        result_row = {
            "lambda_fl":
                lambda_fl,

            "lambda_l":
                lambda_l,

            "lambda_g":
                lambda_g,

            "lambda_r":
                0.0,

            "lambda_a":
                0.0,

            "validation_mean_nMSE":
                validation_nmse,

            "validation_mean_wR":
                validation_wr,

            "outer_iterations":
                model.n_outer_iterations_,

            "selected_coefficients":
                int(
                    np.count_nonzero(
                        model.coef_
                    )
                ),

            "selected_objective_dmos":
                int(
                    np.count_nonzero(
                        model.selected_dmos()
                    )
                ),
        }

        # Save objective-specific validation performance
        # for this hyperparameter configuration.
        for objective_result in objective_rows:
            objective_name = str(
                objective_result[
                    "objective"
                ]
            )

            result_row[
                f"{objective_name}_nMSE"
            ] = objective_result[
                "nMSE"
            ]

            result_row[
                f"{objective_name}_wR"
            ] = objective_result[
                "wR"
            ]

        results.append(
            result_row
        )

        print(
            f"    Validation mean nMSE = "
            f"{validation_nmse:.6f}"
        )

        print(
            f"    Validation mean wR   = "
            f"{validation_wr:.6f}"
        )

    results_frame = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "validation_mean_nMSE",
                "validation_mean_wR",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    results_frame.to_csv(
        output_dir
        / "stage1_search.csv",
        index=False,
    )

    best = results_frame.iloc[
        0
    ]

    best_fl = float(
        best[
            "lambda_fl"
        ]
    )

    best_l = float(
        best[
            "lambda_l"
        ]
    )

    best_g = float(
        best[
            "lambda_g"
        ]
    )

    print(
        "\n"
        "---------------- STAGE 1 BEST ----------------"
    )

    print(
        f"lambda_fl = {best_fl:g}"
    )

    print(
        f"lambda_l  = {best_l:g}"
    )

    print(
        f"lambda_g  = {best_g:g}"
    )

    print(
        "lambda_r  = 0"
    )

    print(
        "lambda_a  = 0"
    )

    print(
        "Validation mean nMSE = "
        f"{best['validation_mean_nMSE']:.6f}"
    )

    print(
        "Validation mean wR   = "
        f"{best['validation_mean_wR']:.6f}"
    )

    print(
        "------------------------------------------------\n"
    )

    return (
        best_fl,
        best_l,
        best_g,
        results_frame,
    )


# ============================================================
# Stage 2 hyperparameter search
# ============================================================


def stage2_search(
        x_train,
        y_train,
        x_validation,
        y_validation,
        *,
        best_fl: float,
        best_l: float,
        best_g: float,
        r_grid: list[float],
        a_grid: list[float],
        output_dir: Path,
        verbose: bool,
) -> tuple[
    float,
    float,
    pd.DataFrame,
]:
    """Tune relation-learning hyperparameters."""

    combinations = list(
        itertools.product(
            r_grid,
            a_grid,
        )
    )

    total = len(
        combinations
    )

    results = []

    print(
        "\n"
        "============================================================"
    )

    print(
        "STAGE 2: Cross-objective relation learning"
    )

    print(
        "Fixed Stage-1 parameters:"
    )

    print(
        f"lambda_fl = {best_fl:g}"
    )

    print(
        f"lambda_l  = {best_l:g}"
    )

    print(
        f"lambda_g  = {best_g:g}"
    )

    print(
        f"Number of combinations: {total}"
    )

    print(
        "============================================================\n"
    )

    for index, (
            lambda_r,
            lambda_a,
    ) in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"[Stage 2: {index}/{total}] "
            f"R={lambda_r:g}, "
            f"A={lambda_a:g}"
        )

        model = fit_candidate(
            x_train,
            y_train,

            lambda_fl=
            best_fl,

            lambda_l=
            best_l,

            lambda_g=
            best_g,

            lambda_r=
            lambda_r,

            lambda_a=
            lambda_a,

            verbose=
            verbose,
        )

        (
            validation_nmse,
            validation_wr,
            objective_rows,
        ) = validation_score(
            model,
            x_validation,
            y_validation,
        )

        result_row = {
            "lambda_fl":
                best_fl,

            "lambda_l":
                best_l,

            "lambda_g":
                best_g,

            "lambda_r":
                lambda_r,

            "lambda_a":
                lambda_a,

            "validation_mean_nMSE":
                validation_nmse,

            "validation_mean_wR":
                validation_wr,

            "outer_iterations":
                model.n_outer_iterations_,

            "selected_coefficients":
                int(
                    np.count_nonzero(
                        model.coef_
                    )
                ),

            "selected_objective_dmos":
                int(
                    np.count_nonzero(
                        model.selected_dmos()
                    )
                ),
        }

        for objective_result in objective_rows:
            objective_name = str(
                objective_result[
                    "objective"
                ]
            )

            result_row[
                f"{objective_name}_nMSE"
            ] = objective_result[
                "nMSE"
            ]

            result_row[
                f"{objective_name}_wR"
            ] = objective_result[
                "wR"
            ]

        results.append(
            result_row
        )

        print(
            f"    Validation mean nMSE = "
            f"{validation_nmse:.6f}"
        )

        print(
            f"    Validation mean wR   = "
            f"{validation_wr:.6f}"
        )

    results_frame = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "validation_mean_nMSE",
                "validation_mean_wR",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    results_frame.to_csv(
        output_dir
        / "stage2_search.csv",
        index=False,
    )

    best = results_frame.iloc[
        0
    ]

    best_r = float(
        best[
            "lambda_r"
        ]
    )

    best_a = float(
        best[
            "lambda_a"
        ]
    )

    print(
        "\n"
        "---------------- STAGE 2 BEST ----------------"
    )

    print(
        f"lambda_r = {best_r:g}"
    )

    print(
        f"lambda_a = {best_a:g}"
    )

    print(
        "Validation mean nMSE = "
        f"{best['validation_mean_nMSE']:.6f}"
    )

    print(
        "Validation mean wR   = "
        f"{best['validation_mean_wR']:.6f}"
    )

    print(
        "------------------------------------------------\n"
    )

    return (
        best_r,
        best_a,
        results_frame,
    )


# ============================================================
# Final test predictions
# ============================================================


def make_test_predictions(
        data,
        model,
        test_ids,
        feature_scalers,
        outcome_scalers,
) -> pd.DataFrame:
    """Generate predictions on the held-out test set."""

    prediction_frames = []

    for objective_index, spec in enumerate(
            OBJECTIVES
    ):

        frame = data.objective_frames[
            spec.name
        ]

        (
            feature_mean,
            feature_scale,
        ) = feature_scalers[
            spec.disease
        ]

        (
            target_mean,
            target_scale,
        ) = outcome_scalers[
            spec.name
        ]

        for visit_index, visit in enumerate(
                VISITS
        ):

            visit_test = frame[
                (
                        frame[
                            "visit.number"
                        ]
                        == visit
                )
                &
                (
                    frame[
                        "participantid"
                    ].isin(
                        test_ids[
                            spec.disease
                        ]
                    )
                )
                ].sort_values(
                "participantid"
            )

            if visit_test.empty:
                raise ValueError(
                    f"No test observations for "
                    f"{spec.name} at {visit}."
                )

            features = visit_test.loc[
                :,
                data.dmo_columns,
            ].to_numpy(
                dtype=float
            )

            x_test = (
                             features
                             - feature_mean
                     ) / feature_scale

            prediction_standardised = (
                    x_test
                    @ model.coef_[
                        objective_index,
                        :,
                        visit_index,
                    ]
            )

            prediction = (
                    prediction_standardised
                    * target_scale
                    + target_mean
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "objective":
                            spec.name,

                        "disease":
                            spec.disease,

                        "visit":
                            visit,

                        "participantid":
                            visit_test[
                                "participantid"
                            ].to_numpy(),

                        "y_true":
                            visit_test[
                                "target"
                            ].to_numpy(
                                dtype=float
                            ),

                        "y_pred":
                            prediction,
                    }
                )
            )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


# ============================================================
# Visit-specific rMSE
# ============================================================


def performance_by_visit(
        predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rMSE for each objective at each visit."""

    rows = []

    grouped = predictions.groupby(
        [
            "objective",
            "disease",
            "visit",
        ],
        sort=False,
    )

    for (
            objective,
            disease,
            visit,
    ), group in grouped:
        y_true = group[
            "y_true"
        ].to_numpy(
            dtype=float
        )

        y_pred = group[
            "y_pred"
        ].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "objective":
                    objective,

                "disease":
                    disease,

                "visit":
                    visit,

                "n":
                    len(group),

                "rMSE":
                    rmse(
                        y_true,
                        y_pred,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Overall nMSE and wR
# ============================================================


def performance_overall(
        predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate overall nMSE and wR across all visits.

    One overall result is produced for each clinical
    prediction objective.
    """

    rows = []

    grouped = predictions.groupby(
        [
            "objective",
            "disease",
        ],
        sort=False,
    )

    for (
            objective,
            disease,
    ), objective_frame in grouped:

        visit_data = []

        for visit in VISITS:

            visit_frame = objective_frame[
                objective_frame[
                    "visit"
                ]
                == visit
                ]

            if visit_frame.empty:
                continue

            y_true = visit_frame[
                "y_true"
            ].to_numpy(
                dtype=float
            )

            y_pred = visit_frame[
                "y_pred"
            ].to_numpy(
                dtype=float
            )

            visit_data.append(
                (
                    y_true,
                    y_pred,
                )
            )

        result = (
            overall_metrics_from_visits(
                visit_data
            )
        )

        rows.append(
            {
                "objective":
                    objective,

                "disease":
                    disease,

                "n":
                    result[
                        "n"
                    ],

                "nMSE":
                    result[
                        "nMSE"
                    ],

                "wR":
                    result[
                        "wR"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


def performance_global(
        per_objective: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise performance across all prediction objectives.

    The primary global metrics are weighted by the number of
    held-out observations available for each objective. Macro
    means are also reported so every objective contributes
    equally, irrespective of its sample size.
    """

    required_columns = {
        "objective",
        "n",
        "nMSE",
        "wR",
    }

    missing_columns = (
            required_columns
            - set(per_objective.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns for global performance: "
            + ", ".join(sorted(missing_columns))
        )

    if per_objective.empty:
        raise ValueError(
            "No objective-level results available "
            "for global performance calculation."
        )

    sample_counts = per_objective["n"].to_numpy(
        dtype=float,
    )
    nmse_values = per_objective["nMSE"].to_numpy(
        dtype=float,
    )
    wr_values = per_objective["wR"].to_numpy(
        dtype=float,
    )

    if (
            np.any(~np.isfinite(sample_counts))
            or np.any(sample_counts <= 0.0)
    ):
        raise ValueError(
            "Objective sample counts must be finite "
            "and strictly positive."
        )

    if np.any(~np.isfinite(nmse_values)):
        raise ValueError(
            "All objective nMSE values must be finite."
        )

    valid_wr = np.isfinite(wr_values)

    weighted_nmse = float(
        np.average(
            nmse_values,
            weights=sample_counts,
        )
    )

    weighted_wr = (
        float(
            np.average(
                wr_values[valid_wr],
                weights=sample_counts[valid_wr],
            )
        )
        if np.any(valid_wr)
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "aggregation": "all_objectives",
                "n_objectives": int(len(per_objective)),
                "n": int(sample_counts.sum()),
                "nMSE": weighted_nmse,
                "wR": weighted_wr,
                "macro_nMSE": float(np.mean(nmse_values)),
                "macro_wR": (
                    float(np.mean(wr_values[valid_wr]))
                    if np.any(valid_wr)
                    else np.nan
                ),
            }
        ]
    )


# ============================================================
# Relation matrix visualisation
# ============================================================


def save_relation_matrix_visualisation(
        relation: np.ndarray,
        output_dir: Path,
) -> None:
    """Save relation matrix as CSV, PNG and PDF."""

    matrix = np.asarray(
        relation,
        dtype=float,
    )

    expected_shape = (
        len(RELATION_LABELS),
        len(RELATION_LABELS),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Relation matrix has shape "
            f"{matrix.shape}; "
            f"expected {expected_shape}."
        )

    relation_frame = pd.DataFrame(
        matrix,
        index=RELATION_LABELS,
        columns=RELATION_LABELS,
    )

    relation_frame.to_csv(
        output_dir
        / "relation_matrix.csv"
    )

    try:

        import matplotlib

        matplotlib.use(
            "Agg"
        )

        import matplotlib.pyplot as plt

    except ImportError as error:

        raise ImportError(
            "matplotlib is required "
            "for relation-matrix visualisation."
        ) from error

    colour_limit = float(
        np.max(
            np.abs(
                matrix
            )
        )
    )

    if colour_limit == 0.0:
        colour_limit = 1.0

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            6.5,
        ),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-colour_limit,
        vmax=colour_limit,
    )

    axis.set_xticks(
        np.arange(
            len(
                RELATION_LABELS
            )
        ),
        RELATION_LABELS,
    )

    axis.set_yticks(
        np.arange(
            len(
                RELATION_LABELS
            )
        ),
        RELATION_LABELS,
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    axis.set_title(
        "Learned Cross-Objective Relation Matrix"
    )

    for row in range(
            matrix.shape[0]
    ):

        for column in range(
                matrix.shape[1]
        ):
            text_colour = (
                "white"
                if abs(
                    matrix[
                        row,
                        column,
                    ]
                )
                   >= 0.55
                   * colour_limit
                else "black"
            )

            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=10,
            )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.82,
    )

    colour_bar.set_label(
        "Learned relation weight"
    )

    figure.savefig(
        output_dir
        / "relation_matrix.png",
        dpi=300,
    )

    figure.savefig(
        output_dir
        / "relation_matrix.pdf"
    )

    plt.close(
        figure
    )


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
        dataset_root: Path,
        output_dir: Path,
        *,
        random_seed: int,
        fl_grid: list[float],
        l_grid: list[float],
        g_grid: list[float],
        r_grid: list[float],
        a_grid: list[float],
        verbose: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_mobilise_data(
        dataset_root
    )

    # ========================================================
    # Participant-level 70:10:20 split
    # ========================================================

    (
        train_ids,
        validation_ids,
        test_ids,
    ) = participant_train_validation_test_split(
        data,
        random_seed=random_seed,
    )

    print(
        "\nParticipant-level split:"
    )

    for disease in sorted(
            train_ids
    ):
        n_train = len(
            train_ids[
                disease
            ]
        )

        n_validation = len(
            validation_ids[
                disease
            ]
        )

        n_test = len(
            test_ids[
                disease
            ]
        )

        n_total = (
                n_train
                + n_validation
                + n_test
        )

        print(
            f"{disease}: "
            f"train={n_train} "
            f"({n_train / n_total:.1%}), "
            f"validation={n_validation} "
            f"({n_validation / n_total:.1%}), "
            f"test={n_test} "
            f"({n_test / n_total:.1%})"
        )

    # ========================================================
    # Standardisation
    #
    # Only the 70% training participants are used to estimate
    # scaling parameters.
    # ========================================================

    feature_scalers = (
        fit_feature_scalers(
            data,
            train_ids,
        )
    )

    outcome_scalers = (
        fit_outcome_scalers(
            data,
            train_ids,
        )
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    x_train, y_train = (
        build_standardised_longitudinal_data(
            data,
            train_ids,
            feature_scalers,
            outcome_scalers,
        )
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    x_validation, y_validation = (
        build_standardised_longitudinal_data(
            data,
            validation_ids,
            feature_scalers,
            outcome_scalers,
        )
    )

    # ========================================================
    # Stage 1
    # ========================================================

    (
        best_fl,
        best_l,
        best_g,
        stage1_results,
    ) = stage1_search(
        x_train,
        y_train,
        x_validation,
        y_validation,

        fl_grid=
        fl_grid,

        l_grid=
        l_grid,

        g_grid=
        g_grid,

        output_dir=
        output_dir,

        verbose=
        verbose,
    )

    # ========================================================
    # Stage 2
    # ========================================================

    (
        best_r,
        best_a,
        stage2_results,
    ) = stage2_search(
        x_train,
        y_train,
        x_validation,
        y_validation,

        best_fl=
        best_fl,

        best_l=
        best_l,

        best_g=
        best_g,

        r_grid=
        r_grid,

        a_grid=
        a_grid,

        output_dir=
        output_dir,

        verbose=
        verbose,
    )

    # ========================================================
    # Best hyperparameters
    # ========================================================

    best_hyperparameters = {

        "lambda_fl":
            best_fl,

        "lambda_l":
            best_l,

        "lambda_g":
            best_g,

        "lambda_r":
            best_r,

        "lambda_a":
            best_a,
    }

    (
            output_dir
            / "best_hyperparameters.json"
    ).write_text(
        json.dumps(
            best_hyperparameters,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "FINAL SELECTED HYPERPARAMETERS"
    )

    print(
        "============================================================"
    )

    for key, value in (
            best_hyperparameters.items()
    ):
        print(
            f"{key:10s} = {value:g}"
        )

    # ========================================================
    # Final model
    #
    # Train using only the 70% training participants.
    # ========================================================

    final_model = fit_candidate(
        x_train,
        y_train,

        lambda_fl=
        best_fl,

        lambda_l=
        best_l,

        lambda_g=
        best_g,

        lambda_r=
        best_r,

        lambda_a=
        best_a,

        verbose=
        verbose,
    )

    run_name = parameter_name(
        best_fl,
        best_l,
        best_g,
        best_r,
        best_a,
    )

    final_output_dir = (
            output_dir
            / "final_model"
            / run_name
    )

    final_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Held-out test evaluation
    # ========================================================

    predictions = (
        make_test_predictions(
            data,
            final_model,
            test_ids,
            feature_scalers,
            outcome_scalers,
        )
    )

    # Visit-specific rMSE
    by_visit = performance_by_visit(
        predictions
    )

    # Overall nMSE and wR
    overall = performance_overall(
        predictions
    )

    # Global sample-weighted and macro performance across
    # all four clinical prediction objectives.
    global_overall = performance_global(
        overall
    )

    predictions.to_csv(
        final_output_dir
        / "test_predictions.csv",
        index=False,
    )

    by_visit.to_csv(
        final_output_dir
        / "performance_by_visit.csv",
        index=False,
    )

    overall.to_csv(
        final_output_dir
        / "performance_overall.csv",
        index=False,
    )

    global_overall.to_csv(
        final_output_dir
        / "performance_global.csv",
        index=False,
    )

    # ========================================================
    # Relation matrix
    # ========================================================

    save_relation_matrix_visualisation(
        final_model.relation_,
        final_output_dir,
    )

    # ========================================================
    # Participant split
    # ========================================================

    split_rows = []

    for disease in sorted(
            train_ids
    ):

        for participant in sorted(
                train_ids[
                    disease
                ]
        ):
            split_rows.append(
                {
                    "disease":
                        disease,

                    "participantid":
                        participant,

                    "split":
                        "train",
                }
            )

        for participant in sorted(
                validation_ids[
                    disease
                ]
        ):
            split_rows.append(
                {
                    "disease":
                        disease,

                    "participantid":
                        participant,

                    "split":
                        "validation",
                }
            )

        for participant in sorted(
                test_ids[
                    disease
                ]
        ):
            split_rows.append(
                {
                    "disease":
                        disease,

                    "participantid":
                        participant,

                    "split":
                        "test",
                }
            )

    pd.DataFrame(
        split_rows
    ).to_csv(
        output_dir
        / "participant_splits.csv",
        index=False,
    )

    # ========================================================
    # Save fitted model
    # ========================================================

    np.savez_compressed(
        final_output_dir
        / "fitted_model.npz",

        coefficients=
        final_model.coef_,

        relation=
        final_model.relation_,

        objective_history=
        np.asarray(
            final_model.objective_history_
        ),

        dmo_columns=
        np.asarray(
            data.dmo_columns
        ),

        objective_names=
        np.asarray(
            [
                spec.name
                for spec
                in OBJECTIVES
            ]
        ),

        visits=
        np.asarray(
            VISITS
        ),

        feature_means=
        np.stack(
            [
                feature_scalers[
                    disease
                ][0]
                for disease
                in (
                "PD",
                "MS",
                "PFF",
            )
            ]
        ),

        feature_scales=
        np.stack(
            [
                feature_scalers[
                    disease
                ][1]
                for disease
                in (
                "PD",
                "MS",
                "PFF",
            )
            ]
        ),

        scaler_diseases=
        np.asarray(
            (
                "PD",
                "MS",
                "PFF",
            )
        ),

        outcome_means=
        np.asarray(
            [
                outcome_scalers[
                    spec.name
                ][0]
                for spec
                in OBJECTIVES
            ]
        ),

        outcome_scales=
        np.asarray(
            [
                outcome_scalers[
                    spec.name
                ][1]
                for spec
                in OBJECTIVES
            ]
        ),
    )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {

        "random_seed":
            random_seed,

        "participant_split":
            (
                "participant-level 70:10:20 "
                "train/validation/test within each disease"
            ),

        "training_fraction":
            0.70,

        "validation_fraction":
            0.10,

        "test_fraction":
            0.20,

        "hyperparameter_selection":
            (
                "hyperparameters selected exclusively "
                "using the validation set"
            ),

        "hyperparameter_selection_metric":
            (
                "mean validation nMSE across clinical "
                "prediction objectives; mean validation "
                "wR used as secondary ranking criterion"
            ),

        "evaluation_metrics": {

            "overall":
                (
                    "nMSE and weighted Pearson correlation "
                    "(wR) across all visits for each "
                    "clinical prediction objective"
                ),

            "visit_specific":
                (
                    "rMSE for each clinical prediction "
                    "objective at each visit"
                ),

            "global":
                (
                    "sample-count-weighted nMSE and wR "
                    "across all four prediction objectives; "
                    "unweighted macro means are also reported"
                ),
        },

        "global_test_performance":
            global_overall.iloc[0].to_dict(),

        "final_evaluation":
            (
                "performance reported exclusively "
                "on the held-out test set"
            ),

        "standardisation":
            (
                "feature and outcome standardisation "
                "estimated exclusively from the "
                "70% training set"
            ),

        "stage1": {

            "description":
                (
                    "Tune lambda_fl, lambda_l and lambda_g "
                    "with lambda_r=lambda_a=0."
                ),

            "best_lambda_fl":
                best_fl,

            "best_lambda_l":
                best_l,

            "best_lambda_g":
                best_g,
        },

        "stage2": {

            "description":
                (
                    "Fix Stage-1 parameters and tune "
                    "lambda_r and lambda_a."
                ),

            "best_lambda_r":
                best_r,

            "best_lambda_a":
                best_a,
        },

        "final_hyperparameters":
            best_hyperparameters,

        "train_participants": {

            disease:
                len(
                    train_ids[
                        disease
                    ]
                )

            for disease in train_ids
        },

        "validation_participants": {

            disease:
                len(
                    validation_ids[
                        disease
                    ]
                )

            for disease
            in validation_ids
        },

        "test_participants": {

            disease:
                len(
                    test_ids[
                        disease
                    ]
                )

            for disease
            in test_ids
        },

        "outer_iterations":
            final_model.n_outer_iterations_,

        "selected_coefficients":
            int(
                np.count_nonzero(
                    final_model.coef_
                )
            ),

        "selected_objective_dmos":
            int(
                np.count_nonzero(
                    final_model.selected_dmos()
                )
            ),
    }

    (
            final_output_dir
            / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Console output
    # ========================================================

    print(
        "\nOverall held-out TEST performance "
        "(nMSE ↓, wR ↑):"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\nGlobal held-out TEST performance across all "
        "prediction objectives (nMSE ↓, wR ↑):"
    )

    print(
        global_overall.to_string(
            index=False
        )
    )

    print(
        "\nVisit-specific held-out TEST performance "
        "(rMSE ↓):"
    )

    print(
        by_visit.to_string(
            index=False
        )
    )

    print(
        "\nLearned relation matrix:"
    )

    print(
        pd.DataFrame(
            final_model.relation_,
            index=RELATION_LABELS,
            columns=RELATION_LABELS,
        )
        .round(3)
        .to_string()
    )

    print(
        "\nStage-1 search saved to:"
    )

    print(
        output_dir
        / "stage1_search.csv"
    )

    print(
        "\nStage-2 search saved to:"
    )

    print(
        output_dir
        / "stage2_search.csv"
    )

    print(
        "\nFinal results written to:"
    )

    print(
        final_output_dir
    )


# ============================================================
# Command-line interface
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    project_root = (
        Path(
            __file__
        )
        .resolve()
        .parent
    )

    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=project_root,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/two_stage_search"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Stage-1 hyperparameter grids
    # --------------------------------------------------------

    parser.add_argument(
        "--fl-grid",
        type=str,
        default=(
            "0,0.001,0.01,0.05,0.1"
        ),
    )

    parser.add_argument(
        "--l-grid",
        type=str,
        default=(
            "0,0.001,0.01,0.05,0.1"
        ),
    )

    parser.add_argument(
        "--g-grid",
        type=str,
        default=(
            "0,0.001,0.01,0.05,0.1"
        ),
    )

    # --------------------------------------------------------
    # Stage-2 hyperparameter grids
    # --------------------------------------------------------

    parser.add_argument(
        "--r-grid",
        type=str,
        default=(
            "0.001,0.01,0.05,0.1"
        ),
    )

    parser.add_argument(
        "--a-grid",
        type=str,
        default=(
            "0.001,0.01,0.1,1,10"
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = parser.parse_args()

    run_experiment(
        args.dataset_root,
        args.output_dir,

        random_seed=
        args.seed,

        fl_grid=
        parse_grid(
            args.fl_grid
        ),

        l_grid=
        parse_grid(
            args.l_grid
        ),

        g_grid=
        parse_grid(
            args.g_grid
        ),

        r_grid=
        parse_grid(
            args.r_grid
        ),

        a_grid=
        parse_grid(
            args.a_grid
        ),

        verbose=
        not args.quiet,
    )


if __name__ == "__main__":
    main()
