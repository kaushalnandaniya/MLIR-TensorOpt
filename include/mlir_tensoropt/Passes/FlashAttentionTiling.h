//===-- FlashAttentionTiling.h - Block-wise Online Softmax Tiling -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Defines the FlashAttentionTilingPass for block-wise attention decomposition
// and online softmax statistics (FlashAttention algorithm) to restrict memory
// footprint to on-chip SRAM.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensoropt {

struct FlashAttentionOptions {
  unsigned blockSizeQ = 64;
  unsigned blockSizeKV = 64;
};

/// Creates an instance of the FlashAttentionTilingPass.
std::unique_ptr<mlir::Pass> createFlashAttentionTilingPass();
std::unique_ptr<mlir::Pass> createFlashAttentionTilingPass(const FlashAttentionOptions &options);

/// Registers the FlashAttentionTilingPass with the MLIR pass registry.
void registerFlashAttentionTilingPass();

} // namespace tensoropt
} // namespace mlir
