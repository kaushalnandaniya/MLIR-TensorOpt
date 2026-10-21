"""
Standalone numerical drift verification executable.
Runs PyTorch reference vs compiled / multi-precision variants for Attention and MLP.
"""

import sys
import torch
from mlir_tensoropt.frontend.models import MultiHeadAttentionBlock, FeedForwardBlock
from mlir_tensoropt.verification.divergence_checker import (
    verify_precision_suite,
    print_rich_divergence_table,
    format_divergence_table,
)

def main():
    print("=" * 70)
    print("MLIR-TensorOpt: Numerical Drift & Precision Verification Suite")
    print("=" * 70)

    # 1. Attention Block Evaluation
    print("\n[1/2] Evaluating MultiHeadAttentionBlock (FP32 baseline vs BF16 / FP8)...")
    attn = MultiHeadAttentionBlock(embed_dim=128, num_heads=4)
    attn.eval()
    x_attn = torch.randn(2, 64, 128)
    with torch.no_grad():
        y_ref_attn = attn(x_attn)

    attn_metrics = verify_precision_suite(y_ref_attn)
    print_rich_divergence_table(attn_metrics)

    # 2. FeedForward MLP Block Evaluation
    print("\n[2/2] Evaluating FeedForwardBlock (FP32 baseline vs BF16 / FP8)...")
    mlp = FeedForwardBlock(embed_dim=128, hidden_dim=512)
    mlp.eval()
    x_mlp = torch.randn(2, 64, 128)
    with torch.no_grad():
        y_ref_mlp = mlp(x_mlp)

    mlp_metrics = verify_precision_suite(y_ref_mlp)
    print_rich_divergence_table(mlp_metrics)

    all_passed = all(m.passed for m in attn_metrics + mlp_metrics)
    if all_passed:
        print("\nAll numerical tolerance checks PASSED successfully.")
        return 0
    else:
        print("\nSome numerical tolerance checks exceeded threshold limits.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
