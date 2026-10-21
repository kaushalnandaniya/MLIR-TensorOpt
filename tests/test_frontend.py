"""
Unit tests for PyTorch model definitions and MLIR frontend export.
"""

import pytest
import torch
from mlir_tensoropt.frontend.models import (
    MultiHeadAttentionBlock,
    FeedForwardBlock,
    TransformerBlock,
)
from mlir_tensoropt.frontend.export import export_torch_to_mlir

def test_multi_head_attention_eager():
    """Verify PyTorch forward pass of MultiHeadAttentionBlock."""
    batch_size, seq_len, embed_dim = 2, 32, 64
    model = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=4)
    model.eval()

    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_feed_forward_eager():
    """Verify PyTorch forward pass of FeedForwardBlock."""
    batch_size, seq_len, embed_dim = 2, 32, 64
    model = FeedForwardBlock(embed_dim=embed_dim, hidden_dim=256)
    model.eval()

    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_transformer_block_eager():
    """Verify PyTorch forward pass of TransformerBlock."""
    batch_size, seq_len, embed_dim = 2, 32, 64
    model = TransformerBlock(embed_dim=embed_dim, num_heads=4, hidden_dim=256)
    model.eval()

    x = torch.randn(batch_size, seq_len, embed_dim)
    out = model(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_export_to_torch_dialect():
    """Verify export to MLIR Torch dialect."""
    class SimpleModule(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    model = SimpleModule()
    x = torch.randn(4, 4)
    mod = export_torch_to_mlir(model, x, output_type="torch")
    ir_str = str(mod)
    assert "torch." in ir_str


def test_export_to_linalg_on_tensors():
    """Verify export to MLIR Linalg on Tensors dialect."""
    class MatmulModule(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    model = MatmulModule()
    a = torch.randn(16, 32)
    b = torch.randn(32, 64)
    mod = export_torch_to_mlir(model, a, b, output_type="linalg-on-tensors")
    ir_str = str(mod)
    assert "linalg.matmul" in ir_str or "linalg." in ir_str
