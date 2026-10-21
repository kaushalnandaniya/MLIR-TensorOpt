"""
Reference PyTorch models for MLIR-TensorOpt compilation pipeline.
Includes MultiHeadAttentionBlock, FeedForwardBlock, and TransformerBlock.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    """
    Scaled Dot-Product Multi-Head Attention block with fused QKV projection,
    scaling, causal masking, and output projection.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        mask: optional [seq_len, seq_len] causal mask
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Matmul Q @ K^T and scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask

        # Softmax over sequence dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Matmul Attention @ V
        out = torch.matmul(attn_weights, v)

        # Recombine heads [batch, seq_len, embed_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardBlock(nn.Module):
    """
    Two-layer MLP block with LayerNorm and activation (GELU).
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder layer combining MultiHeadAttention and FeedForward blocks.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = self.mlp(x)
        return x


class BatchedGEMMBlock(nn.Module):
    """
    Batched Matrix Multiplication block with optional bias and ReLU/GELU activation.
    Ideal for isolated L1 cache locality and loop tiling optimization studies.
    """
    def __init__(self, in_features: int = 64, out_features: int = 256, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generates an additive upper-triangular causal attention mask filled with -inf.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask

