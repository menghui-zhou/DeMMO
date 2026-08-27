"""One-at-a-time sensitivity analysis for the five model penalties."""

from __future__ import annotations


import time
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (
    OBJECTIVES,
    VISITS,
    load_mobilise_data,
    participant_train_test_split,
)
from optimizer import CrossDiseaseLongitudinalMTL


PARAMETERS = (
    "lambda_fl",
    "lambda_l",
    "lambda_g",
    "lambda_r",
    "lambda_a",
)

GRID = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
)

BASE_VALUE = 1e-3


def _split_development_ids(
    development_ids: dict[str, set[str]],
    *,
    validation_fraction: float,
    random_seed: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    rng = np.random.default_rng(random_seed)

    fit_ids: dict[str, set[str]] = {}
    validation_ids: dict[str, set[str]] = {}

    for disease, participants in development_ids.items():
        ids = np.asarray(sorted(participants), dtype=str)
        shuffled = rng.permutation(ids)

        n_validation = max(
            1,
            int(np.ceil(validation_fraction * ids.size)),
        )

        validation_ids[disease] = set(
            shuffled[:n_validation]
        )
        fit_ids[disease] = set(
            shuffled[n_validation:]
        )

    return fit_ids, validation_ids


def _prepare_arrays(
    dataset_root: Path,
    random_seed: int,
) -> dict[str, object]:
    data = load_mobilise_data(dataset_root)

    development_ids, held_out_test_ids = (
        participant_train_test_split(
            data,
            test_fraction=0.2,
            random_seed=random_seed,
        )
    )

    fit_ids, validation_ids = _split_development_ids(
        development_ids,
        validation_fraction=0.2,
        random_seed=random_seed + 1,
    )

    feature_scalers: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ] = {}

    for disease, frame in data.cohort_frames.items():
        fit_frame = frame[
            frame["participantid"].isin(
                fit_ids[disease]
            )
        ]

        values = fit_frame.loc[
            :,
            data.dmo_columns,
        ].to_numpy(float)

        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale[scale == 0.0] = 1.0

        feature_scalers[disease] = (
            mean,
            scale,
        )

    x_fit: list[list[np.ndarray]] = []
    y_fit: list[list[np.ndarray]] = []

    x_validation: list[list[np.ndarray]] = []
    y_validation: list[list[np.ndarray]] = []

    counts: list[dict[str, object]] = []

    for spec in OBJECTIVES:
        frame = data.objective_frames[
            spec.name
        ]

        fit_frame = frame[
            frame["participantid"].isin(
                fit_ids[spec.disease]
            )
        ]

        target_mean = float(
            fit_frame["target"].mean()
        )

        target_scale = float(
            fit_frame["target"].std(ddof=0)
        )

        if target_scale == 0.0:
            target_scale = 1.0

        feature_mean, feature_scale = (
            feature_scalers[spec.disease]
        )

        objective_x_fit: list[np.ndarray] = []
        objective_y_fit: list[np.ndarray] = []

        objective_x_validation: list[
            np.ndarray
        ] = []

        objective_y_validation: list[
            np.ndarray
        ] = []

        for visit in VISITS:
            visit_frame = frame[
                frame["visit.number"] == visit
            ]

            visit_fit = visit_frame[
                visit_frame[
                    "participantid"
                ].isin(
                    fit_ids[spec.disease]
                )
            ]

            visit_validation = visit_frame[
                visit_frame[
                    "participantid"
                ].isin(
                    validation_ids[
                        spec.disease
                    ]
                )
            ]

            if (
                visit_fit.empty
                or visit_validation.empty
            ):
                raise ValueError(
                    "empty fit/validation visit "
                    f"for {spec.name} at {visit}"
                )

            x_current_fit = visit_fit.loc[
                :,
                data.dmo_columns,
            ].to_numpy(float)

            x_current_validation = (
                visit_validation.loc[
                    :,
                    data.dmo_columns,
                ].to_numpy(float)
            )

            objective_x_fit.append(
                (
                    x_current_fit
                    - feature_mean
                )
                / feature_scale
            )

            objective_x_validation.append(
                (
                    x_current_validation
                    - feature_mean
                )
                / feature_scale
            )

            objective_y_fit.append(
                (
                    visit_fit[
                        "target"
                    ].to_numpy(float)
                    - target_mean
                )
                / target_scale
            )

            objective_y_validation.append(
                (
                    visit_validation[
                        "target"
                    ].to_numpy(float)
                    - target_mean
                )
                / target_scale
            )

            counts.append(
                {
                    "objective": spec.name,
                    "disease": spec.disease,
                    "visit": visit,
                    "fit_rows": len(
                        visit_fit
                    ),
                    "validation_rows": len(
                        visit_validation
                    ),
                }
            )

        x_fit.append(objective_x_fit)
        y_fit.append(objective_y_fit)

        x_validation.append(
            objective_x_validation
        )
        y_validation.append(
            objective_y_validation
        )

    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "counts": counts,
        "fit_participants": {
            key: len(value)
            for key, value in fit_ids.items()
        },
        "validation_participants": {
            key: len(value)
            for key, value
            in validation_ids.items()
        },
        "held_out_test_participants": {
            key: len(value)
            for key, value
            in held_out_test_ids.items()
        },
    }


def _prediction_loss(
    model: CrossDiseaseLongitudinalMTL,
    x: list[list[np.ndarray]],
    y: list[list[np.ndarray]],
) -> tuple[float, list[float]]:
    predictions = model.predict(x)

    task_mse: list[float] = []
    objective_mse: list[float] = []

    for (
        objective_predictions,
        objective_targets,
    ) in zip(predictions, y):
        current: list[float] = []

        for prediction, target in zip(
            objective_predictions,
            objective_targets,
        ):
            mse = float(
                np.mean(
                    (prediction - target) ** 2
                )
            )

            task_mse.append(mse)
            current.append(mse)

        objective_mse.append(
            float(np.mean(current))
        )

    return (
        float(np.mean(task_mse)),
        objective_mse,
    )


def _fit_setting(
    payload: tuple[
        dict[str, object],
        str,
        float,
    ],
) -> dict[str, object]:
    prepared, varied_parameter, value = (
        payload
    )

    penalties = {
        parameter: BASE_VALUE
        for parameter in PARAMETERS
    }

    if varied_parameter != "__baseline__":
        penalties[varied_parameter] = value

    model = CrossDiseaseLongitudinalMTL(
        **penalties,
        max_outer_iter=50,
        max_inner_iter=500,
        verbose=False,
    ).fit(
        prepared["x_fit"],
        prepared["y_fit"],
    )

    fit_mse, _ = _prediction_loss(
        model,
        prepared["x_fit"],
        prepared["y_fit"],
    )

    validation_mse, objective_mse = (
        _prediction_loss(
            model,
            prepared["x_validation"],
            prepared["y_validation"],
        )
    )

    row: dict[str, object] = {
        "parameter": varied_parameter,
        "value": value,
        "fit_mse": fit_mse,
        "validation_mse": validation_mse,
        "regularised_training_objective":
            model.objective_history_[-1],
        "outer_iterations":
            model.n_outer_iterations_,
        "nonzero_coefficients": int(
            np.count_nonzero(model.coef_)
        ),
        "selected_objective_dmos": int(
            np.count_nonzero(
                model.selected_dmos()
            )
        ),
        "relation_frobenius_norm": float(
            np.linalg.norm(
                model.relation_
            )
        ),
    }

    for spec, mse in zip(
        OBJECTIVES,
        objective_mse,
    ):
        row[
            f"validation_mse__{spec.name}"
        ] = mse

    return row


def _summarise(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for parameter, group in results.groupby(
        "parameter",
        sort=False,
    ):
        ordered = group.sort_values("value")

        best = ordered.loc[
            ordered[
                "validation_mse"
            ].idxmin()
        ]

        baseline = ordered[
            np.isclose(
                ordered["value"],
                BASE_VALUE,
            )
        ].iloc[0]

        log_values = np.log(ordered["value"].to_numpy(float))
        log_losses = np.log(ordered["validation_mse"].to_numpy(float))
        interval_sensitivity = np.abs(np.diff(log_losses) / np.diff(log_values))

        minimum = float(
            ordered[
                "validation_mse"
            ].min()
        )

        maximum = float(
            ordered[
                "validation_mse"
            ].max()
        )

        rows.append(
            {
                "parameter": parameter,
                "best_value": float(
                    best["value"]
                ),
                "best_validation_mse":
                    float(
                        best[
                            "validation_mse"
                        ]
                    ),
                "baseline_validation_mse":
                    float(
                        baseline[
                            "validation_mse"
                        ]
                    ),
                "relative_loss_range_percent":
                    100.0
                    * (maximum - minimum)
                    / minimum,
                "mean_log_sensitivity": float(np.mean(interval_sensitivity)),
                "maximum_log_sensitivity": float(np.max(interval_sensitivity)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "relative_loss_range_percent",
        ascending=False,
    )


PARAMETER_LABELS = {
    "lambda_fl": r"$\lambda_{\mathrm{FL}}$",
    "lambda_l": r"$\lambda_{\mathrm{L}}$",
    "lambda_g": r"$\lambda_{\mathrm{G}}$",
    "lambda_r": r"$\lambda_{\mathrm{R}}$",
    "lambda_a": r"$\lambda_{\mathrm{A}}$",
}


def _plot_sensitivity(results: pd.DataFrame, output_dir: Path) -> None:
    """Create one paper-ready validation-loss figure per hyperparameter."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for parameter in PARAMETERS:
        selected = results[results["parameter"] == parameter].sort_values("value")
        x = selected["value"].to_numpy(float)
        y = selected["validation_mse"].to_numpy(float)
        fig, axis = plt.subplots(figsize=(5.2, 3.8))
        axis.plot(x, y, color="#0072B2", linewidth=2.2, marker="o", markersize=6)
        baseline_index = int(np.flatnonzero(np.isclose(x, BASE_VALUE))[0])
        axis.scatter(
            [x[baseline_index]], [y[baseline_index]], color="#D62728",
            marker="*", s=150, zorder=5, label="Reference setting",
        )
        axis.set_xscale("log")
        axis.set_xticks(GRID, ["0.001", "0.01", "0.1", "1"])
        axis.set_xlabel(PARAMETER_LABELS[parameter], fontsize=11)
        axis.set_ylabel("Validation MSE", fontsize=11)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        stem = f"sensitivity_{parameter}"
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)



def run_sensitivity(
    dataset_root: Path,
    output_dir: Path,
    *,
    random_seed: int,
    workers: int,
) -> None:
    start_time = time.perf_counter()

    print("=" * 80, flush=True)
    print("Starting hyperparameter sensitivity analysis", flush=True)
    print(f"Dataset root: {dataset_root.resolve()}", flush=True)
    print(f"Output directory: {output_dir.resolve()}", flush=True)
    print(f"Random seed: {random_seed}", flush=True)
    print(f"Number of workers: {workers}", flush=True)
    print("=" * 80, flush=True)

    print(
        "\n[Step 1] Preparing training and validation data...",
        flush=True,
    )

    prepared = _prepare_arrays(
        dataset_root,
        random_seed,
    )

    print(
        "Data preparation completed.",
        flush=True,
    )

    print(
        f"Fit participants: "
        f"{prepared['fit_participants']}",
        flush=True,
    )

    print(
        f"Validation participants: "
        f"{prepared['validation_participants']}",
        flush=True,
    )

    print(
        f"Held-out test participants: "
        f"{prepared['held_out_test_participants']}",
        flush=True,
    )

    # The all-0.001 reference point is shared by all five curves. Train it
    # once, then reuse it when constructing the 20 displayed points.
    settings = [(prepared, "__baseline__", BASE_VALUE)] + [
        (prepared, parameter, value)
        for parameter in PARAMETERS
        for value in GRID
        if not np.isclose(value, BASE_VALUE)
    ]

    total_settings = len(settings)

    print(
        f"\n[Step 2] Starting {total_settings} unique fits "
        f"for 20 displayed sensitivity points...",
        flush=True,
    )

    print(
        f"Each experiment changes one parameter; "
        f"the remaining parameters are fixed at "
        f"{BASE_VALUE}.",
        flush=True,
    )

    print(
        "-" * 80,
        flush=True,
    )

    def report_progress(
        completed: int,
        result: dict[str, object],
    ) -> None:
        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[{completed:02d}/{total_settings}] "
            f"{result['parameter']}="
            f"{float(result['value']):.5g} | "
            f"fit MSE="
            f"{float(result['fit_mse']):.6f} | "
            f"validation MSE="
            f"{float(result['validation_mse']):.6f} | "
            f"selected DMOs="
            f"{int(result['selected_objective_dmos'])}/96 | "
            f"nonzero coefficients="
            f"{int(result['nonzero_coefficients'])} | "
            f"outer iterations="
            f"{int(result['outer_iterations'])} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    rows: list[dict[str, object]] = []

    if workers == 1:
        for completed, setting in enumerate(
            settings,
            start=1,
        ):
            result = _fit_setting(setting)
            rows.append(result)

            report_progress(
                completed,
                result,
            )

    else:
        with ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            future_to_setting = {
                executor.submit(
                    _fit_setting,
                    setting,
                ): (
                    setting[1],
                    setting[2],
                )
                for setting in settings
            }

            for completed, future in enumerate(
                as_completed(
                    future_to_setting
                ),
                start=1,
            ):
                parameter, value = (
                    future_to_setting[future]
                )

                try:
                    result = future.result()

                except Exception as error:
                    print(
                        f"\nExperiment failed: "
                        f"{parameter}={value}",
                        flush=True,
                    )

                    print(
                        f"Error: {error}",
                        flush=True,
                    )

                    raise

                rows.append(result)

                report_progress(
                    completed,
                    result,
                )

    print(
        "-" * 80,
        flush=True,
    )

    print(
        "\n[Step 3] Summarising and saving results...",
        flush=True,
    )

    raw_results = pd.DataFrame(rows)
    reference = raw_results[raw_results["parameter"] == "__baseline__"].iloc[0]
    expanded_rows = raw_results[raw_results["parameter"] != "__baseline__"].to_dict("records")
    for parameter in PARAMETERS:
        baseline_row = reference.to_dict()
        baseline_row["parameter"] = parameter
        baseline_row["value"] = BASE_VALUE
        expanded_rows.append(baseline_row)
    results = pd.DataFrame(expanded_rows).sort_values(["parameter", "value"])

    summary = _summarise(results)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save all hyperparameter results.
    combined_results_path = (
        output_dir
        / "sensitivity_results.csv"
    )

    results.to_csv(
        combined_results_path,
        index=False,
    )

    print(
        f"Saved combined results: "
        f"{combined_results_path}",
        flush=True,
    )

    # Save each hyperparameter separately.
    for parameter in PARAMETERS:
        parameter_results = results[
            results["parameter"]
            == parameter
        ].sort_values("value")

        parameter_output_path = (
            output_dir
            / (
                "sensitivity_results_"
                f"{parameter}.csv"
            )
        )

        parameter_results.to_csv(
            parameter_output_path,
            index=False,
        )

        print(
            f"Saved {parameter}: "
            f"{parameter_output_path}",
            flush=True,
        )

    summary_path = (
        output_dir
        / "sensitivity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    _plot_sensitivity(results, output_dir)

    print(
        f"Saved sensitivity summary: "
        f"{summary_path}",
        flush=True,
    )

    sample_counts_path = (
        output_dir
        / "sensitivity_sample_counts.csv"
    )

    pd.DataFrame(
        prepared["counts"]
    ).to_csv(
        sample_counts_path,
        index=False,
    )

    print(
        f"Saved sample counts: "
        f"{sample_counts_path}",
        flush=True,
    )

    metadata = {
        "design": (
            "one-at-a-time logarithmic "
            "sensitivity analysis"
        ),
        "baseline_value": BASE_VALUE,
        "grid": GRID,
        "unique_model_fits": total_settings,
        "displayed_sensitivity_points": len(PARAMETERS) * len(GRID),
        "shared_reference_fit": True,
        "selection_metric": (
            "mean validation MSE across "
            "20 objective-visit tasks after "
            "training-only target "
            "standardisation"
        ),
        "fit_participants":
            prepared[
                "fit_participants"
            ],
        "validation_participants":
            prepared[
                "validation_participants"
            ],
        "held_out_test_participants":
            prepared[
                "held_out_test_participants"
            ],
        "random_seed": random_seed,
        "test_set_used": False,
    }

    metadata_path = (
        output_dir
        / "sensitivity_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved experiment metadata: "
        f"{metadata_path}",
        flush=True,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        f"All {total_settings} experiments "
        f"completed successfully.",
        flush=True,
    )

    print(
        f"Total running time: "
        f"{elapsed:.1f} seconds "
        f"({elapsed / 60.0:.2f} minutes)",
        flush=True,
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\nSensitivity summary:",
        flush=True,
    )

    print(
        summary.to_string(
            index=False
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    project_root = (
        Path(__file__).resolve().parent
    )

    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=project_root,
        help=(
            "Project directory containing "
            "the outcome folder"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/"
            "hyperparameter_sensitivity"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    run_sensitivity(
        args.dataset_root,
        args.output_dir,
        random_seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
