"""
Numerical drift and multi-precision verification suite for MLIR-TensorOpt.
"""

from .divergence_checker import (
    PrecisionFormat,
    NumericalMetrics,
    compute_numerical_divergence,
    verify_precision_suite,
    format_divergence_table,
)

__all__ = [
    "PrecisionFormat",
    "NumericalMetrics",
    "compute_numerical_divergence",
    "verify_precision_suite",
    "format_divergence_table",
]
