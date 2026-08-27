"""Run balanced hyperparameter longitudinal stability selection.

The parameter schedule is fixed before model fitting and is independent of
validation or test performance. Each randomly selected hyperparameter setting
is fitted on the same number of independent participant-level half-samples.
Progress is checkpointed after every fit.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_demmo_two_stage import (
    build_standardised_longitudinal_data,
    fit_candidate,
    fit_feature_scalers,
    fit_outcome_scalers,
    load_mobilise_data,
    participant_train_validation_test_split,
)
from demmo.data import OBJECTIVES, VISITS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_FL_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_L_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_G_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_R_GRID = (0.0, 0.001, 0.01, 0.1)
DEFAULT_A_GRID = (0.0, 0.001, 0.01, 0.1)


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


def random_parameter_schedule(
    iterations: int,
    random_seed: int,
) -> list[dict[str, float]]:
    """Randomly draw unique combinations from the full grid before fitting."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    spaces = {
        "lambda_fl": DEFAULT_FL_GRID,
        "lambda_l": DEFAULT_L_GRID,
        "lambda_g": DEFAULT_G_GRID,
        "lambda_r": DEFAULT_R_GRID,
        "lambda_a": DEFAULT_A_GRID,
    }
    names = tuple(spaces)
    combinations = [
        values
        for values in product(*(spaces[name] for name in names))
        if (
            (values[names.index("lambda_r")] == 0.0
             and values[names.index("lambda_a")] == 0.0)
            or
            (values[names.index("lambda_r")] > 0.0
             and values[names.index("lambda_a")] > 0.0)
        )
    ]
    if iterations > len(combinations):
        raise ValueError("iterations exceed the number of unique grid combinations")
    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(len(combinations), size=iterations, replace=False)
    return [
        {name: float(value) for name, value in zip(names, combinations[index])}
        for index in selected_indices
    ]


def half_sample_participants(
    development_ids: dict[str, set[str]],
    random_seed: int,
) -> dict[str, set[str]]:
    rng = np.random.default_rng(random_seed)
    sampled: dict[str, set[str]] = {}
    for disease in sorted(development_ids):
        identifiers = np.asarray(sorted(development_ids[disease]), dtype=object)
        sample_size = max(1, identifiers.size // 2)
        sampled[disease] = set(
            rng.choice(identifiers, size=sample_size, replace=False).tolist()
        )
    return sampled


def save_final_outputs(
    counts: np.ndarray,
    n_iterations: int,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
) -> None:
    probabilities = counts.astype(float) / n_iterations
    rows: list[dict[str, object]] = []
    stable_rows: list[dict[str, object]] = []
    for objective_index, spec in enumerate(OBJECTIVES):
        objective_scores = probabilities[objective_index]
        for feature_index, feature in enumerate(feature_names):
            for visit_index, visit in enumerate(VISITS):
                rows.append(
                    {
                        "objective": spec.name,
                        "disease": spec.disease,
                        "feature": feature,
                        "visit": visit,
                        "selection_count": int(
                            counts[objective_index, feature_index, visit_index]
                        ),
                        "n_iterations": n_iterations,
                        "selection_probability": float(
                            objective_scores[feature_index, visit_index]
                        ),
                    }
                )
        selected_indices: set[int] = set()
        effective_top_k = min(top_k, len(feature_names))
        for visit_index, visit in enumerate(VISITS):
            ranking = np.argsort(-objective_scores[:, visit_index], kind="stable")
            for rank, feature_index in enumerate(ranking[:effective_top_k], start=1):
                selected_indices.add(int(feature_index))
                stable_rows.append(
                    {
                        "objective": spec.name,
                        "disease": spec.disease,
                        "visit": visit,
                        "rank": rank,
                        "feature": feature_names[feature_index],
                        "selection_probability": float(
                            objective_scores[feature_index, visit_index]
                        ),
                    }
                )

        ordered_indices = sorted(
            selected_indices,
            key=lambda index: (-float(np.max(objective_scores[index])), feature_names[index]),
        )
        heatmap = objective_scores[ordered_indices]
        height = max(5.0, 0.32 * len(ordered_indices) + 1.8)
        figure, axis = plt.subplots(figsize=(7.2, height))
        image = axis.imshow(heatmap, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(VISITS)), labels=[visit.upper() for visit in VISITS])
        axis.set_yticks(
            np.arange(len(ordered_indices)),
            labels=[feature_names[index] for index in ordered_indices],
        )
        axis.set_title(f"Longitudinal Stability Selection: {spec.name}")
        for row, feature_index in enumerate(ordered_indices):
            for column in range(len(VISITS)):
                value = objective_scores[feature_index, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.6 else "black",
                    fontsize=8,
                )
        colourbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
        colourbar.set_label("Selection probability")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"longitudinal_stability_{spec.name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)

    pd.DataFrame(rows).to_csv(
        output_dir / "longitudinal_stability_scores.csv", index=False
    )
    pd.DataFrame(stable_rows).to_csv(
        output_dir / "longitudinal_stable_features_top20.csv", index=False
    )


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_mobilise_data(dataset_root)
    feature_names = [str(name) for name in data.dmo_columns]

    configurations = {}
    for seed in args.seeds:
        train_ids, _, _ = participant_train_validation_test_split(
            data, random_seed=seed
        )
        configurations[seed] = {
            "training_ids": train_ids,
        }

    hyperparameter_settings = random_parameter_schedule(
        args.hyperparameter_settings, args.hyperparameter_seed
    )
    parameter_schedule = [
        {
            "hyperparameter_set": setting_index + 1,
            "repetition": repetition + 1,
            **parameters,
        }
        for setting_index, parameters in enumerate(hyperparameter_settings)
        for repetition in range(args.repeats_per_setting)
    ]
    total_iterations = len(parameter_schedule)
    pd.DataFrame(parameter_schedule).assign(
        iteration=np.arange(1, total_iterations + 1)
    ).loc[:, [
        "iteration", "hyperparameter_set", "repetition", "lambda_fl",
        "lambda_l", "lambda_g", "lambda_r", "lambda_a"
    ]].to_csv(output_dir / "stability_hyperparameter_schedule.csv", index=False)

    shape = (len(OBJECTIVES), len(feature_names), len(VISITS))
    checkpoint_path = output_dir / "stability_checkpoint.npz"
    if checkpoint_path.is_file() and not args.force:
        checkpoint = np.load(checkpoint_path)
        counts = checkpoint["counts"]
        completed = int(checkpoint["completed"])
        if counts.shape != shape:
            raise ValueError("checkpoint shape does not match current dataset")
        if completed > total_iterations:
            raise ValueError(
                "checkpoint contains more iterations than requested; "
                "use the previous iteration count or --force"
            )
        print(f"Resuming from iteration {completed + 1}/{total_iterations}", flush=True)
    else:
        counts = np.zeros(shape, dtype=np.int64)
        completed = 0

    metadata_path = output_dir / "stability_iterations.csv"
    metadata_rows = (
        pd.read_csv(metadata_path).to_dict("records")
        if metadata_path.is_file() and completed > 0 and not args.force
        else []
    )

    for iteration in range(completed, total_iterations):
        parent_seed = args.seeds[iteration % len(args.seeds)]
        subsample_seed = args.subsample_seed + iteration
        configuration = configurations[parent_seed]
        schedule_entry = parameter_schedule[iteration]
        parameters = {
            name: schedule_entry[name]
            for name in ("lambda_fl", "lambda_l", "lambda_g", "lambda_r", "lambda_a")
        }
        sampled_ids = half_sample_participants(
            configuration["training_ids"], subsample_seed
        )
        print(
            f"\n[Stability {iteration + 1}/{total_iterations}] "
            f"set={schedule_entry['hyperparameter_set']}/{args.hyperparameter_settings}, "
            f"repeat={schedule_entry['repetition']}/{args.repeats_per_setting}, "
            f"parent_seed={parent_seed}, subsample_seed={subsample_seed}, "
            + ", ".join(f"{name}={value:g}" for name, value in parameters.items()),
            flush=True,
        )
        started = time.perf_counter()
        feature_scalers = fit_feature_scalers(data, sampled_ids)
        outcome_scalers = fit_outcome_scalers(data, sampled_ids)
        x_sample, y_sample = build_standardised_longitudinal_data(
            data, sampled_ids, feature_scalers, outcome_scalers
        )
        model = fit_candidate(
            x_sample,
            y_sample,
            verbose=False,
            **parameters,
        )
        selected = np.abs(model.coef_) > args.coefficient_threshold
        counts += selected.astype(np.int64)
        metadata_rows.append(
            {
                "iteration": iteration + 1,
                "hyperparameter_set": schedule_entry["hyperparameter_set"],
                "repetition": schedule_entry["repetition"],
                "parent_seed": parent_seed,
                "subsample_seed": subsample_seed,
                "elapsed_seconds": time.perf_counter() - started,
                **parameters,
                **{
                    f"n_{disease}": len(sampled_ids[disease])
                    for disease in sorted(sampled_ids)
                },
            }
        )
        np.savez_compressed(
            checkpoint_path,
            counts=counts,
            completed=np.asarray(iteration + 1),
        )
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        print(
            f"Completed in {metadata_rows[-1]['elapsed_seconds'] / 60.0:.1f} minutes",
            flush=True,
        )

    save_final_outputs(
        counts,
        total_iterations,
        feature_names,
        output_dir,
        args.top_k,
    )
    settings = {
        "iterations": total_iterations,
        "hyperparameter_settings": args.hyperparameter_settings,
        "repeats_per_setting": args.repeats_per_setting,
        "matched_parent_seeds": list(args.seeds),
        "procedure": "balanced hyperparameter longitudinal stability selection",
        "hyperparameter_selection": (
            f"{args.hyperparameter_settings} random unique combinations drawn before "
            f"fitting, each repeated {args.repeats_per_setting} times; no performance selection"
        ),
        "hyperparameter_seed": args.hyperparameter_seed,
        "hyperparameter_spaces": {
            "lambda_fl": list(DEFAULT_FL_GRID),
            "lambda_l": list(DEFAULT_L_GRID),
            "lambda_g": list(DEFAULT_G_GRID),
            "lambda_r": list(DEFAULT_R_GRID),
            "lambda_a": list(DEFAULT_A_GRID),
        },
        "half_sample_without_replacement": True,
        "sampling_pool": "training participants only",
        "coefficient_threshold": args.coefficient_threshold,
        "top_k_per_timepoint": min(args.top_k, len(feature_names)),
    }
    (output_dir / "stability_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nStability selection completed. Results: {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "results" / "five_runs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "five_runs" / "B0_Our_method"
        / "hyperparameter_longitudinal_stability_selection",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--hyperparameter-settings", type=int, default=10)
    parser.add_argument("--repeats-per-setting", type=int, default=10)
    parser.add_argument("--subsample-seed", type=int, default=100_000)
    parser.add_argument("--hyperparameter-seed", type=int, default=2026)
    parser.add_argument("--coefficient-threshold", type=float, default=1e-8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("\nStopped by user. Progress is checkpointed.", flush=True)
        raise SystemExit(130)
