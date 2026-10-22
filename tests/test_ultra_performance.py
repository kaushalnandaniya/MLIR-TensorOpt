"""
Unit tests for Ultra-Performance passes, Native SIMD C++ engine, and Apple Metal GPU.
"""

import pytest
import torch
from mlir_tensoropt.passes import (
    apply_algebraic_simplification,
    apply_flash_attention_tiling,
    FlashAttentionConfig,
)
from mlir_tensoropt.execution.native_engine import NativeExecutionEngine
from mlir_tensoropt.execution.metal_engine import MetalExecutionEngine
from mlir_tensoropt.frontend.models import MultiHeadAttentionBlock
from mlir_tensoropt.frontend.export import export_torch_to_mlir


def test_algebraic_simplification_pass():
    model = MultiHeadAttentionBlock(embed_dim=128, num_heads=4)
    x = torch.randn(1, 64, 128)
    mlir_module = export_torch_to_mlir(model, x, output_type="linalg-on-tensors")
    simplified_ir = apply_algebraic_simplification(mlir_module)
    assert "func.func" in simplified_ir
    assert len(simplified_ir) > 0


def test_flash_attention_tiling_pass():
    model = MultiHeadAttentionBlock(embed_dim=128, num_heads=4)
    x = torch.randn(1, 64, 128)
    mlir_module = export_torch_to_mlir(model, x, output_type="linalg-on-tensors")
    flash_ir = apply_flash_attention_tiling(
        mlir_module,
        config=FlashAttentionConfig(block_size_q=32, block_size_kv=32),
    )
    assert "func.func" in flash_ir
    assert len(flash_ir) > 0


def test_native_execution_engine_compilation():
    success = NativeExecutionEngine.compile_library()
    assert success is True


def test_native_fused_attention_numerical_correctness():
    B, H, S, D = 1, 4, 32, 32
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)

    native_out = NativeExecutionEngine.run_fused_attention(q, k, v)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    max_diff = (native_out - ref_out).abs().max().item()
    assert max_diff < 1e-4, f"Native attention diff too high: {max_diff}"


def test_native_tiled_gemm_numerical_correctness():
    a = torch.randn(64, 32, dtype=torch.float32)
    b = torch.randn(32, 64, dtype=torch.float32)

    native_out = NativeExecutionEngine.run_tiled_gemm(a, b)
    ref_out = torch.matmul(a, b)

    max_diff = (native_out - ref_out).abs().max().item()
    assert max_diff < 1e-4, f"Native GEMM diff too high: {max_diff}"


def test_metal_execution_engine_interface():
    # If on Apple Silicon Mac, test Metal execution; on Linux CI, verify graceful fallback
    available = MetalExecutionEngine.is_available()
    device = MetalExecutionEngine.get_device()
    assert str(device) in ["mps", "cpu"]

    model = MultiHeadAttentionBlock(embed_dim=64, num_heads=2)
    x = torch.randn(1, 16, 64)
    out = MetalExecutionEngine.execute_attention(model, x, return_cpu=True)
    assert out.shape == (1, 16, 64)
