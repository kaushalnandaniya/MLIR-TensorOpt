//===-- LoopTiling.h - Cache & SRAM Hierarchical Loop Tiling Pass -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Implements hierarchical loop tiling on linalg/scf operations to partition
// large matrix multiplications and convolutions to strictly fit inside L1 cache
// and accelerator on-chip SRAM partitions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <vector>

namespace mlir {
namespace tensoropt {

struct LoopTilingOptions {
  std::vector<int64_t> tileSizes = {64, 64, 32};
  bool unrollInnerLoops = true;
};

/// Creates an instance of the LoopTiling pass with default options.
std::unique_ptr<Pass> createLoopTilingPass();

/// Creates an instance of the LoopTiling pass with specified options.
std::unique_ptr<Pass> createLoopTilingPass(const LoopTilingOptions &options);

/// Registers the LoopTiling pass in the global MLIR pass registry.
void registerLoopTilingPass();

} // namespace tensoropt
} // namespace mlir
