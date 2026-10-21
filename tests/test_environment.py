"""
Test environment configuration, dependencies, and toolchain integrity.
"""

import shutil
from pathlib import Path
import pytest
import torch
import torch_mlir
from mlir_tensoropt import get_optimizer_binary

def test_python_packages_installed():
    """Verify core python libraries are functional."""
    assert torch.__version__ is not None
    from torch_mlir import ir, passmanager, execution_engine
    assert ir is not None
    assert passmanager is not None
    assert execution_engine is not None


def test_optimizer_binary_exists():
    """Verify compiled C++ mlir-tensoropt-opt optimizer is available."""
    bin_path = get_optimizer_binary()
    assert bin_path != "", "Optimizer binary path should not be empty"
    path = Path(bin_path)
    assert path.exists(), f"Binary {path} does not exist. Run cmake --build build"
    assert path.is_file()


def test_mlir_context_creation():
    """Verify MLIR Context can be initialized and load standard dialects."""
    with torch_mlir.ir.Context() as ctx:
        assert ctx is not None
        # Verify dialects can be loaded
        mod = torch_mlir.ir.Module.parse("module {}", context=ctx)
        assert str(mod).strip() == "module {\n}"
