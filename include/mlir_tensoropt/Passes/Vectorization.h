//===-- Vectorization.h - Vectorization and SIMD Pass -----------*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Lowers structured linalg.generic and compute operations into native
// hardware SIMD vector operations (vector.transfer_read, vector.contract,
// vector.transfer_write) targeting AVX-512, AVX2, and ARM NEON units.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensoropt {

struct VectorizationOptions {
  int64_t vectorWidthBits = 128;
  bool enableMasking = true;
};

/// Creates an instance of the Vectorization pass with default options.
std::unique_ptr<Pass> createVectorizationPass();

/// Creates an instance of the Vectorization pass with specified options.
std::unique_ptr<Pass> createVectorizationPass(const VectorizationOptions &options);

/// Registers the Vectorization pass in the global MLIR pass registry.
void registerVectorizationPass();

} // namespace tensoropt
} // namespace mlir
