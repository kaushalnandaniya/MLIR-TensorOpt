"""
Attention Fusion Pass for MLIR-TensorOpt.
Fuses QKV projection, scaling, mask, softmax, and matmul into cache-resident kernels.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Union
from torch_mlir import ir, passmanager
from mlir_tensoropt import get_optimizer_binary

def apply_attention_fusion(module: Union[ir.Module, str]) -> str:
    """
    Applies AttentionFusionPass to an MLIR module.

    Fuses composite operations across Multi-Head Attention blocks into
    a fused kernel to eliminate intermediate DRAM/SRAM trips.

    Args:
        module: The MLIR module (either torch_mlir.ir.Module or string).

    Returns:
        str: Transformed MLIR IR.
    """
    ir_text = str(module)
    opt_bin = get_optimizer_binary()

    if opt_bin and Path(opt_bin).exists():
        # Execute through custom C++ optimizer binary
        with tempfile.NamedTemporaryFile("w+", suffix=".mlir", delete=True) as tmp:
            tmp.write(ir_text)
            tmp.flush()
            cmd = [opt_bin, tmp.name, "--mlir-tensoropt-attention-fusion"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return res.stdout

    # Fallback to standard canonicalization and folding pipeline
    with ir.Context() as ctx:
        mod = ir.Module.parse(ir_text, context=ctx)
        pm = passmanager.PassManager.parse(
            "builtin.module(func.func(canonicalize,cse))",
            context=ctx,
        )
        pm.run(mod.operation)
        return str(mod)
