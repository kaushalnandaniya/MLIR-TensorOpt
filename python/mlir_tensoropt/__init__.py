"""
MLIR-TensorOpt: Deep Learning Compiler & Lowering Pipeline.

An end-to-end MLIR compiler and code-generation pipeline bringing high-level
PyTorch neural operators down to optimized hardware execution through
custom rewrite patterns, operator fusion, loop tiling, and numerical drift validation.
"""

__version__ = "0.1.0"
__author__ = "Kaushal Nandaniya"

import sys
from pathlib import Path

# Add project root to path for tool discovery
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BUILD_BIN = ROOT_DIR / "build" / "bin" / "mlir-tensoropt-opt"

def get_optimizer_binary() -> str:
    """Returns the absolute path to the compiled C++ optimizer tool if present."""
    if BUILD_BIN.exists():
        return str(BUILD_BIN)
    return ""
