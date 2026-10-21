"""
One-Shot Bufferization and Vectorization Passes for MLIR-TensorOpt.
Converts tensors to in-place mutable memrefs and lowers loops to SIMD vector instructions.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Union
from torch_mlir import ir, passmanager
from mlir_tensoropt import get_optimizer_binary

def apply_one_shot_bufferization(module: Union[ir.Module, str]) -> str:
    """
    Applies Comprehensive One-Shot Bufferization.
    Transforms value-semantic `tensor` operations to Destination-Passing Style
    `memref` operations in-place, eliminating temporary allocations.
    """
    ir_text = str(module)

    # Use MLIR passmanager with comprehensive one-shot bufferize pipeline
    with ir.Context() as ctx:
        mod = ir.Module.parse(ir_text, context=ctx)
        pipeline = "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})"
        pm = passmanager.PassManager.parse(pipeline, context=ctx)
        pm.run(mod.operation)
        return str(mod)


def apply_vectorization(
    module: Union[ir.Module, str],
    vector_width_bits: int = 128,
) -> str:
    """
    Applies Vectorization pass targeting 128-bit / 256-bit SIMD execution.
    Transforms linalg ops into vector.transfer_read, vector.contract, and vector.transfer_write.
    """
    ir_text = str(module)
    opt_bin = get_optimizer_binary()

    if opt_bin and Path(opt_bin).exists():
        with tempfile.NamedTemporaryFile("w+", suffix=".mlir", delete=True) as tmp:
            tmp.write(ir_text)
            tmp.flush()
            cmd = [
                opt_bin,
                tmp.name,
                f"--mlir-tensoropt-vectorize=vector-width={vector_width_bits}",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return res.stdout

    # Fallback to standard canonicalization
    with ir.Context() as ctx:
        mod = ir.Module.parse(ir_text, context=ctx)
        pm = passmanager.PassManager.parse(
            "builtin.module(func.func(canonicalize))",
            context=ctx,
        )
        pm.run(mod.operation)
        return str(mod)


def apply_bufferize_and_vectorize(
    module: Union[ir.Module, str],
    vector_width_bits: int = 128,
) -> str:
    """
    Executes composite bufferization and vectorization lowering sequence.
    """
    buffered = apply_one_shot_bufferization(module)
    vectorized = apply_vectorization(buffered, vector_width_bits=vector_width_bits)
    return vectorized
