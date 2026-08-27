"""Structured multi-task learning for longitudinal Mobilise-D analyses."""

from .optimizer import (
    CrossDiseaseLongitudinalMTL,
    composite_fused_sparse_group_prox,
    fused_lasso_signal_prox,
    tv1d_prox,
)
from .optimizer_nonconvex import NonConvexDeMMO

__all__ = [
    "CrossDiseaseLongitudinalMTL",
    "composite_fused_sparse_group_prox",
    "fused_lasso_signal_prox",
    "tv1d_prox",
    "NonConvexDeMMO",
]
