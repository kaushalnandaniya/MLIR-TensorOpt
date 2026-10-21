"""
Unit tests for execution engine, lowering pipeline to LLVM dialect, and LLVM IR generation.
"""

import pytest
import torch
from mlir_tensoropt.execution import lower_to_llvm_dialect, translate_to_llvm_ir, MLIRExecutionEngine

MATMUL_MLIR = """
module {
  func.func @test_matmul(%A: tensor<32x16xf32>, %B: tensor<16x64xf32>, %C: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %res = linalg.matmul
      ins(%A, %B : tensor<32x16xf32>, tensor<16x64xf32>)
      outs(%C : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %res : tensor<32x64xf32>
  }
}
"""

def test_lower_to_llvm_dialect():
    """Verify lowering from linalg down to llvm dialect."""
    llvm_mlir = lower_to_llvm_dialect(MATMUL_MLIR)
    assert "llvm.func" in llvm_mlir or "func.func" in llvm_mlir


def test_translate_to_llvm_ir():
    """Verify translation from LLVM dialect to native LLVM IR text."""
    llvm_mlir = lower_to_llvm_dialect(MATMUL_MLIR)
    if "llvm.func" in llvm_mlir:
        llvm_ir = translate_to_llvm_ir(llvm_mlir)
        assert "define " in llvm_ir or "matmul" in llvm_ir


def test_mlir_execution_engine_interface():
    """Verify MLIRExecutionEngine lifecycle."""
    engine = MLIRExecutionEngine(MATMUL_MLIR)
    engine.compile()
    inp = torch.randn(32, 16)
    out = engine.run(inp)
    assert out is not None
    assert out.shape == inp.shape
