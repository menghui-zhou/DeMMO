"""Loading and leakage-free preprocessing for the Mobilise-D experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VISITS = ("t1", "t2", "t3", "t4", "t5")

HOEHN_YAHR_MAP = {
    "Asymptomatic": 0.0,
    "Unilateral involvement only": 1.0,
    "Bilateral involvement without impairment of balance": 2.0,
    (
        "Mild to moderate involvement, some postural instability but "
        "physically independent, needs assistance to recover from pull test"
    ): 3.0,
    "Severe disability, still able to walk or stand unassisted": 4.0,
    "Wheelchair bound or bedridden unless aided": 5.0,
}


@dataclass(frozen=True)
class ObjectiveSpec:
    """Definition of one continuous clinical prediction objective."""

    name: str
    disease: str
    source_column: str


OBJECTIVES = (
    ObjectiveSpec("PD_Hoehn_Yahr", "PD", "hyscr01l"),
    ObjectiveSpec("PD_MDS_UPDRS_III", "PD", "mdsscore3"),
    ObjectiveSpec("MS_EDSS", "MS", "edfscr1l"),
    ObjectiveSpec("PFF_SPPB_impairment", "PFF", "total_sppb_score"),
)


@dataclass
class MobiliseData:
    """Complete weekly DMO records and valid rows for each outcome."""

    dmo_columns: tuple[str, ...]
    cohort_frames: dict[str, pd.DataFrame]
    objective_frames: dict[str, pd.DataFrame]


def _target_values(series: pd.Series, objective: ObjectiveSpec) -> pd.Series:
    if objective.name == "PD_Hoehn_Yahr":
        return series.map(HOEHN_YAHR_MAP).astype(float)

    values = pd.to_numeric(series, errors="coerce").astype(float)
    if objective.name == "PD_MDS_UPDRS_III":
        values = values.where(values >= 0.0)
    elif objective.name == "PFF_SPPB_impairment":
        values = (12.0 - values).where(values.between(0.0, 12.0))
    return values


def load_mobilise_data(dataset_root: str | Path) -> MobiliseData:
    """Load the PD, MS, and PFF weekly data and retain complete DMO rows.

    COPD is intentionally excluded because T2 and T4 are unavailable.  No
    missing-value imputation is performed: every retained row has all 24
    weekly DMO values and a valid target for its prediction objective.
    """

    root = Path(dataset_root).expanduser().resolve()
    outcome_dir = root / "outcome"
    if not outcome_dir.is_dir():
        raise FileNotFoundError(f"outcome directory not found: {outcome_dir}")

    pd_path = outcome_dir / "PD_dataset.csv"
    header = pd.read_csv(pd_path, nrows=0).columns
    dmo_columns = tuple(
        column
        for column in header
        if column.endswith("_w") and column != "n_days_w"
    )
    if len(dmo_columns) != 24:
        raise ValueError(
            f"expected 24 weekly DMO columns, found {len(dmo_columns)}"
        )

    specs_by_disease: dict[str, list[ObjectiveSpec]] = {}
    for spec in OBJECTIVES:
        specs_by_disease.setdefault(spec.disease, []).append(spec)

    cohort_frames: dict[str, pd.DataFrame] = {}
    objective_frames: dict[str, pd.DataFrame] = {}
    for disease, specs in specs_by_disease.items():
        path = outcome_dir / f"{disease}_dataset.csv"
        columns = [
            "participantid",
            "visit.number",
            *dmo_columns,
            *(spec.source_column for spec in specs),
        ]
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
        frame["participantid"] = frame["participantid"].astype(str)
        frame["visit.number"] = frame["visit.number"].str.lower()
        for column in dmo_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame[frame["visit.number"].isin(VISITS)].copy()
        complete_dmos = frame.dropna(subset=list(dmo_columns)).copy()
        if complete_dmos.duplicated(["participantid", "visit.number"]).any():
            raise ValueError(
                f"{disease} contains duplicate participant-visit records"
            )
        cohort_frames[disease] = complete_dmos[
            ["participantid", "visit.number", *dmo_columns]
        ].copy()

        for spec in specs:
            objective_frame = complete_dmos[
                ["participantid", "visit.number", *dmo_columns]
            ].copy()
            objective_frame["target"] = _target_values(
                complete_dmos[spec.source_column], spec
            )
            objective_frame = objective_frame.dropna(subset=["target"])
            objective_frames[spec.name] = objective_frame.reset_index(drop=True)

    return MobiliseData(
        dmo_columns=dmo_columns,
        cohort_frames=cohort_frames,
        objective_frames=objective_frames,
    )


def participant_train_test_split(
    data: MobiliseData,
    *,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Split each disease cohort by participant, preserving all visits."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")

    rng = np.random.default_rng(random_seed)
    train_ids: dict[str, set[str]] = {}
    test_ids: dict[str, set[str]] = {}
    for disease, frame in data.cohort_frames.items():
        ids = np.asarray(sorted(frame["participantid"].unique()), dtype=str)
        shuffled = rng.permutation(ids)
        n_test = max(1, int(np.ceil(test_fraction * shuffled.size)))
        test_ids[disease] = set(shuffled[:n_test])
        train_ids[disease] = set(shuffled[n_test:])
        if train_ids[disease] & test_ids[disease]:
            raise RuntimeError("participant leakage detected in data split")
    return train_ids, test_ids
