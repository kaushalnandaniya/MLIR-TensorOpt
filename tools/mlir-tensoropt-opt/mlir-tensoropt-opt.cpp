//===-- mlir-tensoropt-opt.cpp - MLIR-TensorOpt Optimizer Driver -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Standalone MLIR optimizer tool incorporating custom MLIR-TensorOpt passes:
//   - AttentionFusionPass (--mlir-tensoropt-attention-fusion)
//   - LoopTilingPass (--mlir-tensoropt-loop-tiling)
//   - VectorizationPass (--mlir-tensoropt-vectorize)
//
//===----------------------------------------------------------------------===//

#include "mlir_tensoropt/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Register all upstream MLIR passes
  mlir::registerAllPasses();

  // Register all MLIR-TensorOpt custom optimization passes
  mlir::tensoropt::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR-TensorOpt Optimizer Driver\n", registry));
}
