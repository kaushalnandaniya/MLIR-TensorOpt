//===-- AlgebraicSimplification.h - Constant Folding & Simplification -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Defines the AlgebraicSimplificationPass for constant folding scaling factors,
// dead code elimination, and algebraic identity rewrites.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensoropt {

/// Creates an instance of the AlgebraicSimplificationPass.
std::unique_ptr<mlir::Pass> createAlgebraicSimplificationPass();

/// Registers the AlgebraicSimplificationPass with the MLIR pass registry.
void registerAlgebraicSimplificationPass();

} // namespace tensoropt
} // namespace mlir
