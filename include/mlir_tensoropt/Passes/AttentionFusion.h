//===-- AttentionFusion.h - Attention Fusion Pass Definition ----*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Implements custom pattern rewrites to fuse Multi-Head Attention subgraphs:
// Linear(QKV) -> Reshape -> MatMul(Q, K) -> Div(sqrt(d)) -> Mask -> Softmax -> MatMul(Attn, V)
// into cache-resident fused execution kernels.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensoropt {

/// Creates an instance of the AttentionFusion pass.
std::unique_ptr<Pass> createAttentionFusionPass();

/// Registers the AttentionFusion pass in the global MLIR pass registry.
void registerAttentionFusionPass();

} // namespace tensoropt
} // namespace mlir
