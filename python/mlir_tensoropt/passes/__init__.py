"""
Optimization and transformation passes for MLIR-TensorOpt.
"""

from .algebraic_simplification import apply_algebraic_simplification
from .attention_fusion import apply_attention_fusion
from .flash_attention import apply_flash_attention_tiling, FlashAttentionConfig
from .loop_tiling import apply_loop_tiling, TilingConfig
from .bufferize_and_vectorize import (
    apply_one_shot_bufferization,
    apply_vectorization,
    apply_bufferize_and_vectorize,
)

__all__ = [
    "apply_algebraic_simplification",
    "apply_attention_fusion",
    "apply_flash_attention_tiling",
    "FlashAttentionConfig",
    "apply_loop_tiling",
    "TilingConfig",
    "apply_one_shot_bufferization",
    "apply_vectorization",
    "apply_bufferize_and_vectorize",
]
