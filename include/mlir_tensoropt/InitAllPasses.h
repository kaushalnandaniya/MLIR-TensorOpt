//===-- InitAllPasses.h - MLIR-TensorOpt All Passes Initializer --*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Provides a unified entry point to register all MLIR-TensorOpt passes into
// the MLIR pass infrastructure.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir_tensoropt/Passes/AlgebraicSimplification.h"
#include "mlir_tensoropt/Passes/AttentionFusion.h"
#include "mlir_tensoropt/Passes/FlashAttentionTiling.h"
#include "mlir_tensoropt/Passes/LoopTiling.h"
#include "mlir_tensoropt/Passes/Vectorization.h"

namespace mlir {
namespace tensoropt {

/// Registers all MLIR-TensorOpt optimization and lowering passes.
inline void registerAllPasses() {
  registerAlgebraicSimplificationPass();
  registerAttentionFusionPass();
  registerFlashAttentionTilingPass();
  registerLoopTilingPass();
  registerVectorizationPass();
}

} // namespace tensoropt
} // namespace mlir
