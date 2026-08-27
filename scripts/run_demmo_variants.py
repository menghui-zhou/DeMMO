"""Run two-stage selection for the non-convex DeMMO variants.

DeMMO-var1 and DeMMO-var2 are initialised from the convex DeMMO solution.
Their extra majorisation loop reduces shrinkage bias but increases training
cost. The script uses the same participant-level split as the main experiment.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from demmo.data import load_mobilise_data
from demmo.optimizer_nonconvex import NonConvexDeMMO
from run_demmo_two_stage import (
    build_standardised_longitudinal_data,
    fit_candidate,
    fit_feature_scalers,
    fit_outcome_scalers,
    make_test_predictions,
    participant_train_validation_test_split,
    performance_by_visit,
    performance_global,
    performance_overall,
    validation_score,
)


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("grid cannot be empty")
    return values


def fit_variant(variant, x, y, parameters, initial_model):
    return NonConvexDeMMO(
        variant=variant,
        lambda_nc=parameters["lambda_nc"],
        beta=parameters.get("beta", 1.0),
        lambda_fl=parameters.get("lambda_fl", 0.0),
        lambda_r=parameters.get("lambda_r", 0.0),
        lambda_a=parameters.get("lambda_a", 0.0),
        max_outer_iter=12,
        max_inner_iter=300,
        max_dc_iter=15,
    ).fit(
        x,
        y,
        initial_coefficients=initial_model.coef_,
        initial_relation=(initial_model.relation_
                          if parameters.get("lambda_r", 0.0) > 0
                          else np.zeros_like(initial_model.relation_)),
    )


def select(parameters, variant, x_train, y_train, x_validation, y_validation,
           initial_model, output_path):
    rows = []
    for candidate in parameters:
        model = fit_variant(variant, x_train, y_train, candidate, initial_model)
        nmse, wr, _ = validation_score(model, x_validation, y_validation)
        rows.append({**candidate, "validation_mean_nMSE": nmse,
                     "validation_mean_wR": wr,
                     "dc_iterations": model.n_dc_iterations_})
        pd.DataFrame(rows).to_csv(output_path, index=False)
    table = pd.DataFrame(rows).sort_values(
        ["validation_mean_nMSE", "validation_mean_wR"],
        ascending=[True, False], kind="stable")
    return table.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("var1", "var2"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", default="0.001,0.01,0.1,1")
    parser.add_argument("--relation-grid", default="0.001,0.01,0.1")
    parser.add_argument(
        "--convex-parameters", nargs=5, type=float, required=True,
        metavar=("FL", "L", "G", "R", "A"),
        help="DeMMO parameters selected on the same split",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_mobilise_data(args.dataset_root)
    train_ids, validation_ids, test_ids = participant_train_validation_test_split(
        data, random_seed=args.seed)
    feature_scalers = fit_feature_scalers(data, train_ids)
    outcome_scalers = fit_outcome_scalers(data, train_ids)
    x_train, y_train = build_standardised_longitudinal_data(
        data, train_ids, feature_scalers, outcome_scalers)
    x_validation, y_validation = build_standardised_longitudinal_data(
        data, validation_ids, feature_scalers, outcome_scalers)

    fl, lasso, group, relation, relation_penalty = args.convex_parameters
    initial = fit_candidate(
        x_train, y_train, lambda_fl=fl, lambda_l=lasso, lambda_g=group,
        lambda_r=relation, lambda_a=relation_penalty, verbose=False)
    grid = parse_grid(args.grid)
    if args.variant == "var1":
        stage1 = [{"lambda_fl": x, "lambda_nc": y, "beta": 1.0,
                   "lambda_r": 0.0, "lambda_a": 0.0}
                  for x, y in itertools.product(grid, grid)]
    else:
        stage1 = [{"lambda_fl": 0.0, "lambda_nc": x, "beta": y,
                   "lambda_r": 0.0, "lambda_a": 0.0}
                  for x, y in itertools.product(grid, grid)]
    best1 = select(stage1, args.variant, x_train, y_train, x_validation,
                   y_validation, initial, args.output_dir / "stage1_search.csv")

    relation_grid = parse_grid(args.relation_grid)
    structural = {key: float(best1[key])
                  for key in ("lambda_fl", "lambda_nc", "beta")}
    stage2 = [{**structural, "lambda_r": 0.0, "lambda_a": 0.0}] + [
        {**structural, "lambda_r": r, "lambda_a": a}
        for r, a in itertools.product(relation_grid, relation_grid)]
    best2 = select(stage2, args.variant, x_train, y_train, x_validation,
                   y_validation, initial, args.output_dir / "stage2_search.csv")
    selected = {key: float(best2[key]) for key in
                ("lambda_fl", "lambda_nc", "beta", "lambda_r", "lambda_a")}
    pd.DataFrame([selected]).to_csv(
        args.output_dir / "selected_hyperparameters.csv", index=False)

    development_ids = {disease: train_ids[disease] | validation_ids[disease]
                       for disease in train_ids}
    feature_scalers = fit_feature_scalers(data, development_ids)
    outcome_scalers = fit_outcome_scalers(data, development_ids)
    x_development, y_development = build_standardised_longitudinal_data(
        data, development_ids, feature_scalers, outcome_scalers)
    initial = fit_candidate(
        x_development, y_development, lambda_fl=fl, lambda_l=lasso,
        lambda_g=group, lambda_r=relation, lambda_a=relation_penalty,
        verbose=False)
    model = fit_variant(args.variant, x_development, y_development, selected, initial)
    predictions = make_test_predictions(
        data, model, test_ids, feature_scalers, outcome_scalers)
    by_visit = performance_by_visit(predictions)
    by_target = performance_overall(predictions)
    by_visit.to_csv(args.output_dir / "performance_by_timepoint.csv", index=False)
    by_target.to_csv(args.output_dir / "performance_by_target.csv", index=False)
    performance_global(by_target).to_csv(
        args.output_dir / "performance_global.csv", index=False)


if __name__ == "__main__":
    main()
