"""
Unit tests for numerical divergence validation suite.
"""

import pytest
import torch
from mlir_tensoropt.verification.divergence_checker import (
    PrecisionFormat,
    compute_numerical_divergence,
    verify_precision_suite,
    format_divergence_table,
)

def test_identical_tensors_pass():
    """Verify identical tensors have 0 difference and pass FP32 tolerances."""
    ref = torch.randn(10, 10, dtype=torch.float32)
    metrics = compute_numerical_divergence(ref, ref, PrecisionFormat.FP32)
    assert metrics.passed is True
    assert metrics.max_absolute_diff == 0.0
    assert metrics.rmse == 0.0
    assert abs(metrics.cosine_similarity - 1.0) < 1e-6


def test_bf16_tolerances():
    """Verify BF16 quantized tensors respect BF16 tolerances."""
    ref = torch.randn(20, 20, dtype=torch.float32)
    bf16_val = ref.to(torch.bfloat16).to(torch.float32)
    metrics = compute_numerical_divergence(bf16_val, ref, PrecisionFormat.BF16)
    assert metrics.passed is True
    assert metrics.max_absolute_diff < 1e-1
    assert metrics.cosine_similarity > 0.999


def test_full_precision_suite_and_formatting():
    """Verify complete suite generation and markdown table rendering."""
    ref = torch.randn(16, 16, dtype=torch.float32)
    suite = verify_precision_suite(ref)
    assert len(suite) == 4
    table_str = format_divergence_table(suite)
    assert "FP32" in table_str
    assert "BF16" in table_str
    assert "FP8_E4M3" in table_str
    assert "FP8_E5M2" in table_str
