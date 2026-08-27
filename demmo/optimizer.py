"""Optimisation for longitudinal, multi-outcome DMO regression.

This module implements the alternating optimisation procedure described in
the manuscript.  It is an independent Python implementation of the convex
fused sparse-group proximal decomposition used by Zhou et al. (KDD 2012):

    1. solve a one-dimensional fused-Lasso signal approximation problem;
    2. apply row-wise group shrinkage.

The relation matrix is constrained to be symmetric with a zero diagonal and
is updated by solving a small ridge-regularised linear system over its unique
upper-triangular entries.

The optimiser expects complete, finite, and already standardised arrays.  It
does not impute missing values or fit intercepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    """Apply element-wise soft thresholding."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _difference(values: FloatArray) -> FloatArray:
    """Apply the first-order difference operator R x = x[:-1] - x[1:]."""

    return values[:-1] - values[1:]


def _difference_transpose(dual: FloatArray, output_size: int) -> FloatArray:
    """Apply the transpose of the first-order difference operator."""

    result = np.zeros(output_size, dtype=np.float64)
    if output_size <= 1:
        return result
    result[0] = dual[0]
    result[-1] = -dual[-1]
    if output_size > 2:
        result[1:-1] = dual[1:] - dual[:-1]
    return result


def tv1d_prox(
    values: ArrayLike,
    weight: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 2_000,
) -> FloatArray:
    r"""Proximal operator of one-dimensional total variation.

    Solves

    .. math::

        \arg\min_x \frac{1}{2}\|x-v\|_2^2
        + \lambda \sum_t |x_t-x_{t+1}|.

    The implementation solves the box-constrained dual problem using an
    accelerated projected-gradient iteration.  For Mobilise-D, each call has
    length five, so the dual problem contains only four variables.
    """

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must contain only finite numbers")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if vector.size <= 1 or weight == 0:
        return vector.copy()

    # The gradient Lipschitz constant of the dual quadratic is ||R R^T||_2,
    # which is strictly smaller than four for a finite one-dimensional chain.
    step_size = 0.25
    dual = np.zeros(vector.size - 1, dtype=np.float64)
    extrapolated = dual.copy()
    momentum = 1.0
    previous_primal = vector.copy()

    for _ in range(max_iter):
        rt_dual = _difference_transpose(extrapolated, vector.size)
        gradient = _difference(rt_dual - vector)
        next_dual = np.clip(
            extrapolated - step_size * gradient,
            -weight,
            weight,
        )
        primal = vector - _difference_transpose(next_dual, vector.size)

        primal_change = np.linalg.norm(primal - previous_primal)
        primal_scale = max(1.0, np.linalg.norm(primal))
        if primal_change <= tol * primal_scale:
            return primal

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = next_dual + (
            (momentum - 1.0) / next_momentum
        ) * (next_dual - dual)
        dual = next_dual
        momentum = next_momentum
        previous_primal = primal

    return vector - _difference_transpose(dual, vector.size)


def fused_lasso_signal_prox(
    values: ArrayLike,
    l1_weight: float,
    fused_weight: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 2_000,
) -> FloatArray:
    r"""Proximal operator for element-wise and fused Lasso on a chain.

    Solves

    .. math::

        \arg\min_x \frac{1}{2}\|x-v\|_2^2
        + \lambda_1\|x\|_1 + \lambda_{\mathrm{FL}}\|Rx\|_1.

    For a one-dimensional chain, the solution is obtained by total-variation
    denoising followed by element-wise soft thresholding.
    """

    if l1_weight < 0 or fused_weight < 0:
        raise ValueError("regularisation weights must be non-negative")
    tv_solution = tv1d_prox(
        values,
        fused_weight,
        tol=tol,
        max_iter=max_iter,
    )
    return _soft_threshold(tv_solution, l1_weight)


def composite_fused_sparse_group_prox(
    values: ArrayLike,
    l1_weight: float,
    fused_weight: float,
    group_weight: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 2_000,
) -> FloatArray:
    r"""Apply the exact cFSGL proximal decomposition row-wise.

    Parameters
    ----------
    values:
        Array with shape ``(n_objectives, n_features, n_visits)``.
    l1_weight, fused_weight, group_weight:
        Proximal-scale regularisation weights, i.e. the model penalties
        divided by the current Lipschitz estimate.

    Returns
    -------
    numpy.ndarray
        The proximal solution with the same shape as ``values``.
    """

    tensor = np.asarray(values, dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(
            "values must have shape (n_objectives, n_features, n_visits)"
        )
    if min(l1_weight, fused_weight, group_weight) < 0:
        raise ValueError("regularisation weights must be non-negative")

    result = np.empty_like(tensor)
    for objective in range(tensor.shape[0]):
        for feature in range(tensor.shape[1]):
            fused = fused_lasso_signal_prox(
                tensor[objective, feature],
                l1_weight,
                fused_weight,
                tol=tol,
                max_iter=max_iter,
            )
            norm = np.linalg.norm(fused)
            if norm <= group_weight or norm == 0.0:
                result[objective, feature] = 0.0
            else:
                result[objective, feature] = (
                    1.0 - group_weight / norm
                ) * fused
    return result


@dataclass(frozen=True)
class _ValidatedData:
    """Internal immutable representation of the nested task/visit arrays."""

    x: tuple[tuple[FloatArray, ...], ...]
    y: tuple[tuple[FloatArray, ...], ...]
    n_objectives: int
    n_visits: int
    n_features: int


class CrossDiseaseLongitudinalMTL:
    """Longitudinal multi-objective DMO regression with relation learning.

    Parameters correspond directly to the manuscript objective.  ``lambda_a``
    must be positive whenever relation learning is enabled, ensuring a unique
    symmetric relation-matrix update.
    """

    def __init__(
        self,
        *,
        lambda_fl: float = 0.1,
        lambda_l: float = 0.1,
        lambda_g: float = 0.1,
        lambda_r: float = 0.1,
        lambda_a: float = 0.1,
        max_outer_iter: int = 50,
        max_inner_iter: int = 500,
        outer_tol: float = 1e-6,
        inner_tol: float = 1e-7,
        initial_lipschitz: float = 1.0,
        backtracking_factor: float = 2.0,
        tv_tol: float = 1e-10,
        tv_max_iter: int = 2_000,
        verbose: bool = False,
    ) -> None:
        penalties = {
            "lambda_fl": lambda_fl,
            "lambda_l": lambda_l,
            "lambda_g": lambda_g,
            "lambda_r": lambda_r,
            "lambda_a": lambda_a,
        }
        for name, value in penalties.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if lambda_r > 0 and lambda_a <= 0:
            raise ValueError("lambda_a must be positive when lambda_r > 0")
        if max_outer_iter <= 0 or max_inner_iter <= 0:
            raise ValueError("iteration limits must be positive")
        if outer_tol <= 0 or inner_tol <= 0 or tv_tol <= 0:
            raise ValueError("tolerances must be positive")
        if initial_lipschitz <= 0:
            raise ValueError("initial_lipschitz must be positive")
        if backtracking_factor <= 1:
            raise ValueError("backtracking_factor must be greater than one")

        self.lambda_fl = float(lambda_fl)
        self.lambda_l = float(lambda_l)
        self.lambda_g = float(lambda_g)
        self.lambda_r = float(lambda_r)
        self.lambda_a = float(lambda_a)
        self.max_outer_iter = int(max_outer_iter)
        self.max_inner_iter = int(max_inner_iter)
        self.outer_tol = float(outer_tol)
        self.inner_tol = float(inner_tol)
        self.initial_lipschitz = float(initial_lipschitz)
        self.backtracking_factor = float(backtracking_factor)
        self.tv_tol = float(tv_tol)
        self.tv_max_iter = int(tv_max_iter)
        self.verbose = bool(verbose)

    def fit(
        self,
        x: Sequence[Sequence[ArrayLike]],
        y: Sequence[Sequence[ArrayLike]],
    ) -> "CrossDiseaseLongitudinalMTL":
        """Fit the alternating longitudinal multi-task model.

        ``x[m][t]`` must have shape ``(n_mt, p)`` and ``y[m][t]`` must
        have shape ``(n_mt,)``.  All arrays must be finite complete cases.
        """

        data = self._validate_training_data(x, y)
        self.n_objectives_ = data.n_objectives
        self.n_visits_ = data.n_visits
        self.n_features_in_ = data.n_features
        self.edge_pairs_ = tuple(
            (left, right)
            for left in range(data.n_objectives)
            for right in range(left + 1, data.n_objectives)
        )

        coefficients = np.zeros(
            (data.n_objectives, data.n_features, data.n_visits),
            dtype=np.float64,
        )
        relation = np.zeros(
            (data.n_objectives, data.n_objectives),
            dtype=np.float64,
        )
        lipschitz = self.initial_lipschitz
        initial_objective = self._objective(data, coefficients, relation)
        self.objective_history_ = [initial_objective]
        self.inner_iterations_ = []

        for outer_iteration in range(self.max_outer_iter):
            coefficients, lipschitz, inner_iterations = self._update_coefficients(
                data,
                coefficients,
                relation,
                lipschitz,
            )
            relation = self._update_relation(coefficients)
            objective = self._objective(data, coefficients, relation)
            self.objective_history_.append(objective)
            self.inner_iterations_.append(inner_iterations)

            previous = self.objective_history_[-2]
            relative_change = abs(objective - previous) / max(
                1.0,
                abs(previous),
            )
            if self.verbose:
                print(
                    f"outer={outer_iteration + 1:03d} "
                    f"objective={objective:.10g} "
                    f"relative_change={relative_change:.3e} "
                    f"inner_iterations={inner_iterations}"
                )
            if relative_change <= self.outer_tol:
                break

        self.coef_ = coefficients
        self.relation_ = relation
        self.lipschitz_ = lipschitz
        self.n_outer_iterations_ = len(self.objective_history_) - 1
        return self

    def predict(
        self,
        x: Sequence[Sequence[ArrayLike]],
    ) -> list[list[FloatArray]]:
        """Predict each clinical objective at each visit."""

        self._check_is_fitted()
        if len(x) != self.n_objectives_:
            raise ValueError("x has a different number of objectives")

        predictions: list[list[FloatArray]] = []
        for objective, visits in enumerate(x):
            if len(visits) != self.n_visits_:
                raise ValueError("x has a different number of visits")
            objective_predictions: list[FloatArray] = []
            for visit, values in enumerate(visits):
                matrix = np.asarray(values, dtype=np.float64)
                if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
                    raise ValueError(
                        "each predictor matrix must have shape (n_samples, p)"
                    )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("predictor matrices must be finite")
                objective_predictions.append(
                    matrix @ self.coef_[objective, :, visit]
                )
            predictions.append(objective_predictions)
        return predictions

    def selected_dmos(
        self,
        threshold: float = 1e-8,
    ) -> NDArray[np.bool_]:
        """Return a Boolean objective-by-feature selection mask."""

        self._check_is_fitted()
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        return np.linalg.norm(self.coef_, axis=2) > threshold

    def _validate_training_data(
        self,
        x: Sequence[Sequence[ArrayLike]],
        y: Sequence[Sequence[ArrayLike]],
    ) -> _ValidatedData:
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("x and y must contain the same non-zero objectives")

        converted_x: list[tuple[FloatArray, ...]] = []
        converted_y: list[tuple[FloatArray, ...]] = []
        n_visits: int | None = None
        n_features: int | None = None

        for objective, (x_visits, y_visits) in enumerate(zip(x, y)):
            if len(x_visits) == 0 or len(x_visits) != len(y_visits):
                raise ValueError(
                    f"objective {objective} has inconsistent visit arrays"
                )
            if n_visits is None:
                n_visits = len(x_visits)
            elif len(x_visits) != n_visits:
                raise ValueError("all objectives must contain the same visits")

            current_x: list[FloatArray] = []
            current_y: list[FloatArray] = []
            for visit, (x_values, y_values) in enumerate(
                zip(x_visits, y_visits)
            ):
                matrix = np.asarray(x_values, dtype=np.float64)
                target = np.asarray(y_values, dtype=np.float64)
                if matrix.ndim != 2:
                    raise ValueError(
                        f"x[{objective}][{visit}] must be two-dimensional"
                    )
                if target.ndim != 1:
                    raise ValueError(
                        f"y[{objective}][{visit}] must be one-dimensional"
                    )
                if matrix.shape[0] != target.shape[0] or matrix.shape[0] == 0:
                    raise ValueError(
                        f"objective {objective}, visit {visit} has invalid sample counts"
                    )
                if n_features is None:
                    n_features = matrix.shape[1]
                elif matrix.shape[1] != n_features:
                    raise ValueError("all predictor matrices must share p features")
                if not np.all(np.isfinite(matrix)) or not np.all(
                    np.isfinite(target)
                ):
                    raise ValueError(
                        "training data must be complete and contain only finite values"
                    )
                current_x.append(np.ascontiguousarray(matrix))
                current_y.append(np.ascontiguousarray(target))
            converted_x.append(tuple(current_x))
            converted_y.append(tuple(current_y))

        assert n_visits is not None
        assert n_features is not None
        return _ValidatedData(
            x=tuple(converted_x),
            y=tuple(converted_y),
            n_objectives=len(converted_x),
            n_visits=n_visits,
            n_features=n_features,
        )

    def _coefficient_matrix_to_u(self, coefficients: FloatArray) -> FloatArray:
        return coefficients.reshape(coefficients.shape[0], -1).T

    def _u_gradient_to_coefficients(
        self,
        gradient: FloatArray,
        coefficient_shape: tuple[int, ...],
    ) -> FloatArray:
        return gradient.T.reshape(coefficient_shape)

    def _smooth_value(
        self,
        data: _ValidatedData,
        coefficients: FloatArray,
        relation: FloatArray,
    ) -> float:
        value = 0.0
        for objective in range(data.n_objectives):
            for visit in range(data.n_visits):
                residual = (
                    data.x[objective][visit]
                    @ coefficients[objective, :, visit]
                    - data.y[objective][visit]
                )
                value += 0.5 * float(residual @ residual) / residual.size

        if self.lambda_r > 0:
            u_matrix = self._coefficient_matrix_to_u(coefficients)
            relation_residual = u_matrix - u_matrix @ relation
            value += 0.5 * self.lambda_r * float(
                np.sum(relation_residual**2)
            )
        return value

    def _smooth_gradient(
        self,
        data: _ValidatedData,
        coefficients: FloatArray,
        relation: FloatArray,
    ) -> FloatArray:
        gradient = np.zeros_like(coefficients)
        for objective in range(data.n_objectives):
            for visit in range(data.n_visits):
                matrix = data.x[objective][visit]
                residual = (
                    matrix @ coefficients[objective, :, visit]
                    - data.y[objective][visit]
                )
                gradient[objective, :, visit] = (
                    matrix.T @ residual / residual.size
                )

        if self.lambda_r > 0:
            u_matrix = self._coefficient_matrix_to_u(coefficients)
            identity_minus_relation = np.eye(data.n_objectives) - relation
            relation_gradient = (
                self.lambda_r
                * u_matrix
                @ identity_minus_relation
                @ identity_minus_relation.T
            )
            gradient += self._u_gradient_to_coefficients(
                relation_gradient,
                coefficients.shape,
            )
        return gradient

    def _nonsmooth_value(self, coefficients: FloatArray) -> float:
        lasso = self.lambda_l * float(np.sum(np.abs(coefficients)))
        fused = self.lambda_fl * float(
            np.sum(np.abs(np.diff(coefficients, axis=2)))
        )
        group = self.lambda_g * float(
            np.sum(np.linalg.norm(coefficients, axis=2))
        )
        return lasso + fused + group

    def _objective(
        self,
        data: _ValidatedData,
        coefficients: FloatArray,
        relation: FloatArray,
    ) -> float:
        return (
            self._smooth_value(data, coefficients, relation)
            + self._nonsmooth_value(coefficients)
            + 0.5 * self.lambda_a * float(np.sum(relation**2))
        )

    def _update_coefficients(
        self,
        data: _ValidatedData,
        initial: FloatArray,
        relation: FloatArray,
        initial_lipschitz: float,
    ) -> tuple[FloatArray, float, int]:
        coefficients = initial.copy()
        extrapolated = coefficients.copy()
        momentum = 1.0
        lipschitz = max(initial_lipschitz, np.finfo(float).eps)

        for iteration in range(1, self.max_inner_iter + 1):
            gradient = self._smooth_gradient(data, extrapolated, relation)
            smooth_at_extrapolated = self._smooth_value(
                data,
                extrapolated,
                relation,
            )

            for _ in range(100):
                gradient_point = extrapolated - gradient / lipschitz
                candidate = composite_fused_sparse_group_prox(
                    gradient_point,
                    self.lambda_l / lipschitz,
                    self.lambda_fl / lipschitz,
                    self.lambda_g / lipschitz,
                    tol=self.tv_tol,
                    max_iter=self.tv_max_iter,
                )
                difference = candidate - extrapolated
                quadratic_bound = (
                    smooth_at_extrapolated
                    + float(np.sum(gradient * difference))
                    + 0.5 * lipschitz * float(np.sum(difference**2))
                )
                candidate_smooth = self._smooth_value(
                    data,
                    candidate,
                    relation,
                )
                if candidate_smooth <= quadratic_bound + 1e-12:
                    break
                lipschitz *= self.backtracking_factor
            else:
                raise RuntimeError("backtracking line search did not converge")

            relative_step = np.linalg.norm(candidate - coefficients) / max(
                1.0,
                np.linalg.norm(coefficients),
            )
            next_momentum = 0.5 * (
                1.0 + np.sqrt(1.0 + 4.0 * momentum**2)
            )
            next_extrapolated = candidate + (
                (momentum - 1.0) / next_momentum
            ) * (candidate - coefficients)

            coefficients = candidate
            extrapolated = next_extrapolated
            momentum = next_momentum
            if relative_step <= self.inner_tol:
                return coefficients, lipschitz, iteration

        return coefficients, lipschitz, self.max_inner_iter

    def _update_relation(self, coefficients: FloatArray) -> FloatArray:
        n_objectives = coefficients.shape[0]
        if self.lambda_r == 0 or n_objectives == 1:
            return np.zeros((n_objectives, n_objectives), dtype=np.float64)

        u_matrix = self._coefficient_matrix_to_u(coefficients)
        target = u_matrix.reshape(-1)
        edge_columns: list[FloatArray] = []
        for left, right in self.edge_pairs_:
            edge_matrix = np.zeros(
                (n_objectives, n_objectives),
                dtype=np.float64,
            )
            edge_matrix[left, right] = 1.0
            edge_matrix[right, left] = 1.0
            edge_columns.append((u_matrix @ edge_matrix).reshape(-1))

        design = np.column_stack(edge_columns)
        hessian = (
            self.lambda_r * design.T @ design
            + 2.0 * self.lambda_a * np.eye(len(self.edge_pairs_))
        )
        right_hand_side = self.lambda_r * design.T @ target
        edge_weights = np.linalg.solve(hessian, right_hand_side)

        relation = np.zeros(
            (n_objectives, n_objectives),
            dtype=np.float64,
        )
        for weight, (left, right) in zip(edge_weights, self.edge_pairs_):
            relation[left, right] = weight
            relation[right, left] = weight
        return relation

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before this operation")
