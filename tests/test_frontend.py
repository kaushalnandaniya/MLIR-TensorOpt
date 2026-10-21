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


def test_batched_gemm_eager_and_export():
    """Verify BatchedGEMMBlock eager execution and export to MLIR."""
    from mlir_tensoropt.frontend.models import BatchedGEMMBlock
    gemm = BatchedGEMMBlock(in_features=32, out_features=64)
    gemm.eval()
    x = torch.randn(4, 32)
    out = gemm(x)
    assert out.shape == (4, 64)
    mod = export_torch_to_mlir(gemm, x, output_type="linalg-on-tensors")
    assert "linalg." in str(mod)


def test_causal_mask_attention():
    """Verify MultiHeadAttention with additive causal masking."""
    from mlir_tensoropt.frontend.models import create_causal_mask
    seq_len = 16
    mask = create_causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)
    assert mask[0, 1] == float("-inf")
    assert mask[1, 0] == 0.0

    model = MultiHeadAttentionBlock(embed_dim=64, num_heads=2)
    model.eval()
    x = torch.randn(1, seq_len, 64)
    out = model(x, mask=mask)
    assert out.shape == (1, seq_len, 64)

