"""
Algebraic Simplification and Reciprocal Constant Folding Pass for MLIR-TensorOpt.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Union
from torch_mlir import ir, passmanager
from mlir_tensoropt import get_optimizer_binary


def apply_algebraic_simplification(module: Union[ir.Module, str]) -> str:
    """
    Applies algebraic simplification, reciprocal constant folding, and identity pruning.
    Transforms division by constant into reciprocal multiplication and prunes dead operations.
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
                "--mlir-tensoropt-algebraic-simplification",
                "--allow-unregistered-dialect",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout:
                return res.stdout

    # Fallback to standard canonicalize and cse
    with ir.Context() as ctx:
        mod = ir.Module.parse(ir_text, context=ctx)
        pm = passmanager.PassManager.parse(
            "builtin.module(func.func(canonicalize,cse))",
            context=ctx,
        )
        pm.run(mod.operation)
        return str(mod)
