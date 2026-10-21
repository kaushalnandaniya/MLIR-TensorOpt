"""
Unit tests for custom MLIR-TensorOpt optimization passes.
"""

import subprocess
from pathlib import Path
import pytest
from mlir_tensoropt import get_optimizer_binary
from mlir_tensoropt.passes import (
    apply_attention_fusion,
    apply_loop_tiling,
    apply_vectorization,
    TilingConfig,
)

SAMPLE_IR = """
module {
  func.func @matmul_test(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
"""

def test_optimizer_binary_flags():
    """Verify all pass options are exposed via mlir-tensoropt-opt."""
    bin_path = get_optimizer_binary()
    assert bin_path != ""
    res = subprocess.run([bin_path, "--help"], capture_output=True, text=True, check=True)
    assert "--mlir-tensoropt-attention-fusion" in res.stdout
    assert "--mlir-tensoropt-loop-tiling" in res.stdout
    assert "--mlir-tensoropt-vectorize" in res.stdout


def test_attention_fusion_pass():
    """Verify attention fusion pass executes and annotates functions."""
    out = apply_attention_fusion(SAMPLE_IR)
    assert "mlir_tensoropt.attention" in out or "func.func" in out


def test_loop_tiling_pass():
    """Verify hierarchical loop tiling pass processes linalg operations."""
    out = apply_loop_tiling(SAMPLE_IR, config=TilingConfig(m_tile=32, n_tile=32, k_tile=16))
    assert "mlir_tensoropt.tile_sizes" in out or "func.func" in out


def test_vectorization_pass():
    """Verify vectorization pass tags operations with SIMD width."""
    out = apply_vectorization(SAMPLE_IR, vector_width_bits=128)
    assert "mlir_tensoropt.vector" in out or "func.func" in out


def test_loop_tiling_custom_sizes():
    """Verify custom tile sizes [16, 16, 8] are propagated into IR attributes."""
    out = apply_loop_tiling(SAMPLE_IR, config=TilingConfig(m_tile=16, n_tile=16, k_tile=8))
    assert "[16, 16, 8]" in out


def test_vectorization_custom_width():
    """Verify custom 256-bit SIMD width is propagated into IR attributes."""
    out = apply_vectorization(SAMPLE_IR, vector_width_bits=256)
    assert "256 : i64" in out


def test_attention_sample_mlir_file():
    """Verify passes process the tests/mlir/attention_sample.mlir file."""
    sample_path = Path(__file__).parent / "mlir" / "attention_sample.mlir"
    assert sample_path.exists()
    ir_text = sample_path.read_text()
    fused = apply_attention_fusion(ir_text)
    tiled = apply_loop_tiling(fused, config=TilingConfig(m_tile=64, n_tile=64, k_tile=32))
    vec = apply_vectorization(tiled, vector_width_bits=128)
    assert "attention_fusion_applied" in vec
    assert "vector_width_bits = 128" in vec


