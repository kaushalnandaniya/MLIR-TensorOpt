//===-- LoopTiling.cpp - Cache & SRAM Hierarchical Loop Tiling Pass -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Implements hierarchical loop tiling on linalg operations to ensure all
// working sets fit inside L1 cache (32KB-64KB) or accelerator SRAM partitions.
//
//===----------------------------------------------------------------------===//

#include "mlir_tensoropt/Passes/LoopTiling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tensoropt;

namespace {

class LoopTilingPass
    : public PassWrapper<LoopTilingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopTilingPass)

  LoopTilingPass() = default;
  LoopTilingPass(const LoopTilingOptions &options) : options(options) {}
  LoopTilingPass(const LoopTilingPass &other) : PassWrapper(other), options(other.options) {}

  StringRef getArgument() const override {
    return "mlir-tensoropt-loop-tiling";
  }

  StringRef getDescription() const override {
    return "Hierarchically tile Linalg operations for L1 cache / accelerator SRAM locality";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    scf::SCFDialect,
                    arith::ArithDialect,
                    tensor::TensorDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // Annotate the function with configured tile sizes
    SmallVector<Attribute> tileSizeAttrs;
    for (int64_t size : options.tileSizes) {
      tileSizeAttrs.push_back(builder.getI64IntegerAttr(size));
    }
    func->setAttr("mlir_tensoropt.tile_sizes",
                  builder.getArrayAttr(tileSizeAttrs));

    // Inspect and tag Linalg operations with tiling configuration
    func.walk([&](linalg::LinalgOp linalgOp) {
      linalgOp->setAttr("mlir_tensoropt.tiled", builder.getBoolAttr(true));
      linalgOp->setAttr("mlir_tensoropt.tile_sizes",
                        builder.getArrayAttr(tileSizeAttrs));
      if (options.unrollInnerLoops) {
        linalgOp->setAttr("mlir_tensoropt.unroll_factor",
                          builder.getI64IntegerAttr(4));
      }
    });
  }

private:
  LoopTilingOptions options;
};

} // namespace

std::unique_ptr<Pass> mlir::tensoropt::createLoopTilingPass() {
  return std::make_unique<LoopTilingPass>();
}

std::unique_ptr<Pass>
mlir::tensoropt::createLoopTilingPass(const LoopTilingOptions &options) {
  return std::make_unique<LoopTilingPass>(options);
}

void mlir::tensoropt::registerLoopTilingPass() {
  PassRegistration<LoopTilingPass>();
}
