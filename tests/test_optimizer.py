"""Numerical tests for the longitudinal multi-task optimiser."""

from __future__ import annotations

import unittest

import numpy as np

from demmo import (
    CrossDiseaseLongitudinalMTL,
    NonConvexDeMMO,
    composite_fused_sparse_group_prox,
    fused_lasso_signal_prox,
    tv1d_prox,
)


class ProximalOperatorTests(unittest.TestCase):
    def test_tv_prox_two_point_closed_form(self) -> None:
        np.testing.assert_allclose(
            tv1d_prox(np.array([0.0, 10.0]), 2.0),
            np.array([2.0, 8.0]),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            tv1d_prox(np.array([0.0, 10.0]), 5.0),
            np.array([5.0, 5.0]),
            atol=1e-8,
        )

    def test_fused_lasso_signal_prox(self) -> None:
        np.testing.assert_allclose(
            fused_lasso_signal_prox(
                np.array([0.0, 10.0]),
                l1_weight=1.0,
                fused_weight=2.0,
            ),
            np.array([1.0, 7.0]),
            atol=1e-8,
        )

    def test_composite_identity_without_penalties(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=(4, 3, 5))
        result = composite_fused_sparse_group_prox(values, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(result, values)

    def test_composite_group_shrinkage(self) -> None:
        values = np.array([[[3.0, 4.0]]])
        result = composite_fused_sparse_group_prox(
            values,
            l1_weight=0.0,
            fused_weight=0.0,
            group_weight=2.0,
        )
        np.testing.assert_allclose(result, np.array([[[1.8, 2.4]]]))


class AlternatingOptimiserTests(unittest.TestCase):
    @staticmethod
    def _synthetic_data() -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
        rng = np.random.default_rng(19)
        n_objectives = 4
        n_visits = 5
        n_features = 6

        base = np.array(
            [
                [1.2, 1.2, 1.2, 0.7, 0.7],
                [0.0, 0.0, -0.8, -0.8, -0.8],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        true_coefficients = np.stack(
            [
                base,
                0.85 * base,
                -0.55 * base,
                0.35 * base,
            ]
        )

        x: list[list[np.ndarray]] = []
        y: list[list[np.ndarray]] = []
        for objective in range(n_objectives):
            x_visits: list[np.ndarray] = []
            y_visits: list[np.ndarray] = []
            for visit in range(n_visits):
                # Deliberately vary n across objectives and visits.
                sample_count = 32 + 3 * objective - 2 * visit
                matrix = rng.normal(size=(sample_count, n_features))
                target = (
                    matrix @ true_coefficients[objective, :, visit]
                    + 0.08 * rng.normal(size=sample_count)
                )
                x_visits.append(matrix)
                y_visits.append(target)
            x.append(x_visits)
            y.append(y_visits)
        return x, y

    def test_fit_shapes_constraints_and_objective(self) -> None:
        x, y = self._synthetic_data()
        model = CrossDiseaseLongitudinalMTL(
            lambda_fl=0.02,
            lambda_l=0.01,
            lambda_g=0.02,
            lambda_r=0.05,
            lambda_a=0.1,
            max_outer_iter=20,
            max_inner_iter=500,
            outer_tol=1e-7,
            inner_tol=1e-8,
        ).fit(x, y)

        self.assertEqual(model.coef_.shape, (4, 6, 5))
        self.assertEqual(model.relation_.shape, (4, 4))
        np.testing.assert_allclose(model.relation_, model.relation_.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(model.relation_), 0.0, atol=1e-12)
        self.assertLessEqual(
            model.objective_history_[-1],
            model.objective_history_[0] + 1e-8,
        )
        self.assertTrue(
            all(
                current <= previous + 1e-8
                for previous, current in zip(
                    model.objective_history_,
                    model.objective_history_[1:],
                )
            )
        )
        self.assertTrue(np.all(np.isfinite(model.coef_)))
        self.assertTrue(np.all(np.isfinite(model.relation_)))

        predictions = model.predict(x)
        self.assertEqual(len(predictions), 4)
        self.assertEqual(len(predictions[0]), 5)
        self.assertEqual(predictions[2][3].shape, y[2][3].shape)

    def test_no_relation_penalty_returns_zero_relation(self) -> None:
        x, y = self._synthetic_data()
        model = CrossDiseaseLongitudinalMTL(
            lambda_fl=0.01,
            lambda_l=0.01,
            lambda_g=0.01,
            lambda_r=0.0,
            lambda_a=0.0,
            max_outer_iter=3,
            max_inner_iter=100,
        ).fit(x, y)
        np.testing.assert_array_equal(model.relation_, np.zeros((4, 4)))

    def test_missing_values_are_rejected(self) -> None:
        x, y = self._synthetic_data()
        x[0][0] = x[0][0].copy()
        x[0][0][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "complete"):
            CrossDiseaseLongitudinalMTL(max_outer_iter=1).fit(x, y)

    def test_nonconvex_variants_fit(self) -> None:
        x, y = self._synthetic_data()
        initial = CrossDiseaseLongitudinalMTL(
            lambda_fl=0.01,
            lambda_l=0.01,
            lambda_g=0.01,
            max_outer_iter=2,
            max_inner_iter=60,
        ).fit(x, y)
        for variant in ("var1", "var2"):
            model = NonConvexDeMMO(
                variant=variant,
                lambda_nc=0.01,
                lambda_fl=0.01,
                beta=0.1,
                max_dc_iter=2,
                max_outer_iter=2,
                max_inner_iter=60,
            ).fit(x, y, initial_coefficients=initial.coef_)
            self.assertEqual(model.coef_.shape, (4, 6, 5))
            self.assertTrue(np.all(np.isfinite(model.coef_)))
            np.testing.assert_array_equal(model.relation_, np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
