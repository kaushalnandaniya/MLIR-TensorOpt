"""
Export PyTorch models to MLIR representations (Torch and Linalg-on-Tensors dialects).
"""

from typing import Any, Tuple, Union
import torch
import torch.nn as nn
from torch_mlir import fx, ir
from .models import MultiHeadAttentionBlock, FeedForwardBlock, TransformerBlock

def export_torch_to_mlir(
    model: nn.Module,
    *args: torch.Tensor,
    output_type: str = "linalg-on-tensors",
) -> ir.Module:
    """
    Traces and converts a PyTorch model into an MLIR Module.

    Args:
        model: PyTorch nn.Module to export.
        *args: Example input tensors for tracing.
        output_type: Target dialect representation.
                     Options: "linalg-on-tensors" (default), "torch", "tosa".

    Returns:
        ir.Module: MLIR module in the requested dialect.
    """
    model.eval()

    output_type_map = {
        "linalg-on-tensors": fx.OutputType.LINALG_ON_TENSORS,
        "torch": fx.OutputType.TORCH,
        "tosa": fx.OutputType.TOSA,
    }

    if output_type not in output_type_map:
        raise ValueError(
            f"Unsupported output_type: '{output_type}'. Must be one of {list(output_type_map.keys())}"
        )

    target_output_type = output_type_map[output_type]

    with torch.no_grad():
        module = fx.export_and_import(
            model,
            *args,
            output_type=target_output_type,
        )

    return module


def export_attention_to_mlir(
    batch_size: int = 1,
    seq_len: int = 64,
    embed_dim: int = 128,
    num_heads: int = 4,
    output_type: str = "linalg-on-tensors",
) -> Tuple[ir.Module, MultiHeadAttentionBlock, torch.Tensor]:
    """
    Exports a reference MultiHeadAttention block to MLIR.

    Returns:
        (mlir_module, pytorch_model, sample_input)
    """
    model = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
    sample_input = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    module = export_torch_to_mlir(model, sample_input, output_type=output_type)
    return module, model, sample_input


def export_mlp_to_mlir(
    batch_size: int = 1,
    seq_len: int = 64,
    embed_dim: int = 128,
    hidden_dim: int = 512,
    output_type: str = "linalg-on-tensors",
) -> Tuple[ir.Module, FeedForwardBlock, torch.Tensor]:
    """
    Exports a reference FeedForward MLP block to MLIR.

    Returns:
        (mlir_module, pytorch_model, sample_input)
    """
    model = FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim)
    sample_input = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    module = export_torch_to_mlir(model, sample_input, output_type=output_type)
    return module, model, sample_input


def save_mlir_to_file(module: ir.Module, output_path: str) -> None:
    """Writes an MLIR module to text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(module))
