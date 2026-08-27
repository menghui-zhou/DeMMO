"""Non-convex DeMMO variants solved by iterative reweighting."""

from __future__ import annotations

import numpy as np

from .optimizer import (
    CrossDiseaseLongitudinalMTL,
    FloatArray,
    _ValidatedData,
    fused_lasso_signal_prox,
)


class NonConvexDeMMO(CrossDiseaseLongitudinalMTL):
    """DeMMO-var1 or DeMMO-var2 with reweighted convex updates."""

    def __init__(self, *, variant: str, lambda_nc: float, beta: float = 1.0,
                 lambda_fl: float = 0.0, lambda_r: float = 0.0,
                 lambda_a: float = 0.0, max_dc_iter: int = 20,
                 dc_tol: float = 1e-5, epsilon: float = 1e-6, **kwargs) -> None:
        if variant not in {"var1", "var2"}:
            raise ValueError("variant must be 'var1' or 'var2'")
        if min(lambda_nc, beta, lambda_fl) < 0:
            raise ValueError("non-convex parameters must be non-negative")
        if max_dc_iter < 1 or dc_tol <= 0 or epsilon <= 0:
            raise ValueError("invalid non-convex optimisation settings")
        super().__init__(lambda_fl=0.0, lambda_l=0.0, lambda_g=0.0,
                         lambda_r=lambda_r, lambda_a=lambda_a, **kwargs)
        self.variant = variant
        self.lambda_nc = float(lambda_nc)
        self.beta = float(beta)
        self.nc_lambda_fl = float(lambda_fl)
        self.max_dc_iter = int(max_dc_iter)
        self.dc_tol = float(dc_tol)
        self.epsilon = float(epsilon)

    def fit(self, x, y, *, initial_coefficients=None, initial_relation=None):
        data = self._validate_training_data(x, y)
        self.n_objectives_, self.n_visits_ = data.n_objectives, data.n_visits
        self.n_features_in_ = data.n_features
        self.edge_pairs_ = tuple(
            (left, right) for left in range(data.n_objectives)
            for right in range(left + 1, data.n_objectives)
        )
        shape = (data.n_objectives, data.n_features, data.n_visits)
        coefficients = (np.asarray(initial_coefficients, dtype=np.float64).copy()
                        if initial_coefficients is not None
                        else np.full(shape, 1e-3, dtype=np.float64))
        if coefficients.shape != shape:
            raise ValueError("initial coefficient shape is invalid")
        relation_shape = (data.n_objectives, data.n_objectives)
        relation = (np.asarray(initial_relation, dtype=np.float64).copy()
                    if initial_relation is not None else np.zeros(relation_shape))
        if relation.shape != relation_shape:
            raise ValueError("initial relation shape is invalid")
        if self.lambda_r == 0:
            relation.fill(0.0)

        lipschitz = self.initial_lipschitz
        self.objective_history_ = [self._objective(data, coefficients, relation)]
        self.inner_iterations_ = []
        for dc_iteration in range(1, self.max_dc_iter + 1):
            self._set_reweighting(coefficients)
            previous_coefficients, previous_relation = coefficients.copy(), relation.copy()
            for _ in range(self.max_outer_iter):
                coefficients, lipschitz, inner_iterations = self._update_coefficients(
                    data, coefficients, relation, lipschitz)
                relation = self._update_relation(coefficients)
                self.inner_iterations_.append(inner_iterations)
                block_change = (
                    np.linalg.norm(coefficients - previous_coefficients)
                    + np.linalg.norm(relation - previous_relation)
                ) / max(1.0, np.linalg.norm(previous_coefficients)
                        + np.linalg.norm(previous_relation))
                previous_coefficients, previous_relation = coefficients.copy(), relation.copy()
                if block_change <= self.outer_tol:
                    break
            objective = self._objective(data, coefficients, relation)
            self.objective_history_.append(objective)
            relative = abs(objective - self.objective_history_[-2]) / max(
                1.0, abs(self.objective_history_[-2]))
            if self.verbose:
                print(f"dc={dc_iteration:02d} objective={objective:.10g} "
                      f"relative_change={relative:.3e}", flush=True)
            if relative <= self.dc_tol:
                break

        self.coef_, self.relation_, self.lipschitz_ = coefficients, relation, lipschitz
        self.n_outer_iterations_ = len(self.objective_history_) - 1
        self.n_dc_iterations_ = self.n_outer_iterations_
        return self

    def _set_reweighting(self, coefficients: FloatArray) -> None:
        row_l1 = np.sum(np.abs(coefficients), axis=2)
        if self.variant == "var1":
            weights = 1.0 / np.sqrt(row_l1 + self.epsilon)
            self._row_l1_penalty = 0.5 * self.lambda_nc * weights
            self._row_fused_penalty = np.full_like(weights, self.nc_lambda_fl)
        else:
            temporal = np.sum(np.abs(np.diff(coefficients, axis=2)), axis=2)
            weights = 1.0 / np.sqrt(temporal + self.beta * row_l1 + self.epsilon)
            self._row_l1_penalty = 0.5 * self.lambda_nc * self.beta * weights
            self._row_fused_penalty = 0.5 * self.lambda_nc * weights

    def _nonsmooth_value(self, coefficients: FloatArray) -> float:
        row_l1 = np.sum(np.abs(coefficients), axis=2)
        if self.variant == "var1":
            return (self.lambda_nc * float(np.sum(np.sqrt(row_l1)))
                    + self.nc_lambda_fl
                    * float(np.sum(np.abs(np.diff(coefficients, axis=2)))))
        temporal = np.sum(np.abs(np.diff(coefficients, axis=2)), axis=2)
        return self.lambda_nc * float(np.sum(np.sqrt(temporal + self.beta * row_l1)))

    def _update_coefficients(self, data: _ValidatedData, initial: FloatArray,
                             relation: FloatArray, initial_lipschitz: float):
        coefficients, extrapolated = initial.copy(), initial.copy()
        momentum = 1.0
        lipschitz = max(initial_lipschitz, np.finfo(float).eps)
        for iteration in range(1, self.max_inner_iter + 1):
            gradient = self._smooth_gradient(data, extrapolated, relation)
            smooth_at = self._smooth_value(data, extrapolated, relation)
            for _ in range(100):
                gradient_point = extrapolated - gradient / lipschitz
                candidate = np.empty_like(gradient_point)
                for objective in range(data.n_objectives):
                    for feature in range(data.n_features):
                        candidate[objective, feature] = fused_lasso_signal_prox(
                            gradient_point[objective, feature],
                            self._row_l1_penalty[objective, feature] / lipschitz,
                            self._row_fused_penalty[objective, feature] / lipschitz,
                            tol=self.tv_tol, max_iter=self.tv_max_iter)
                difference = candidate - extrapolated
                bound = (smooth_at + float(np.sum(gradient * difference))
                         + 0.5 * lipschitz * float(np.sum(difference ** 2)))
                if self._smooth_value(data, candidate, relation) <= bound + 1e-12:
                    break
                lipschitz *= self.backtracking_factor
            else:
                raise RuntimeError("backtracking line search did not converge")
            relative_step = np.linalg.norm(candidate - coefficients) / max(
                1.0, np.linalg.norm(coefficients))
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2))
            extrapolated = candidate + ((momentum - 1.0) / next_momentum) * (
                candidate - coefficients)
            coefficients, momentum = candidate, next_momentum
            if relative_step <= self.inner_tol:
                return coefficients, lipschitz, iteration
        return coefficients, lipschitz, self.max_inner_iter
