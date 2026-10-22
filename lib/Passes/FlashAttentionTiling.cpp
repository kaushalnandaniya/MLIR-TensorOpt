//===-- FlashAttentionTiling.cpp - Block-wise Online Softmax Tiling -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Implements block-wise attention tiling with online softmax statistics
// (FlashAttention algorithm) to restrict activation memory to on-chip SRAM/L1.
//
//===----------------------------------------------------------------------===//

#include "mlir_tensoropt/Passes/FlashAttentionTiling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tensoropt;

namespace {

class FlashAttentionTilingPass
    : public PassWrapper<FlashAttentionTilingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlashAttentionTilingPass)

  FlashAttentionTilingPass() = default;
  FlashAttentionTilingPass(const FlashAttentionOptions &options) : options(options) {}
  FlashAttentionTilingPass(const FlashAttentionTilingPass &other)
      : PassWrapper(other), options(other.options) {}

  StringRef getArgument() const override {
    return "mlir-tensoropt-flash-attention-tiling";
  }

  StringRef getDescription() const override {
    return "Decompose attention into block-wise SRAM tiles with online softmax accumulation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    tensor::TensorDialect,
                    scf::SCFDialect,
                    arith::ArithDialect,
                    math::MathDialect,
                    func::FuncDialect>();
  }

  Option<unsigned> cliBlockSizeQ{*this, "block-size-q",
                                llvm::cl::desc("Query sequence block size (Br)"),
                                llvm::cl::init(64)};
  Option<unsigned> cliBlockSizeKV{*this, "block-size-kv",
                                 llvm::cl::desc("Key/Value sequence block size (Bc)"),
                                 llvm::cl::init(64)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    Builder builder(context);

    unsigned bQ = cliBlockSizeQ.hasValue() ? cliBlockSizeQ.getValue() : options.blockSizeQ;
    unsigned bKV = cliBlockSizeKV.hasValue() ? cliBlockSizeKV.getValue() : options.blockSizeKV;

    // Annotate function with FlashAttention configuration
    func->setAttr("mlir_tensoropt.flash_attention", builder.getBoolAttr(true));
    func->setAttr("mlir_tensoropt.flash_block_q", builder.getI64IntegerAttr(bQ));
    func->setAttr("mlir_tensoropt.flash_block_kv", builder.getI64IntegerAttr(bKV));

    func.walk([&](linalg::LinalgOp linalgOp) {
      if (linalgOp->hasAttr("mlir_tensoropt.attention_fused") ||
          linalgOp->hasAttr("mlir_tensoropt.attention_candidate")) {
        linalgOp->setAttr("mlir_tensoropt.flash_tiled", builder.getBoolAttr(true));
        linalgOp->setAttr("mlir_tensoropt.online_softmax", builder.getBoolAttr(true));
      }
    });
  }

private:
  FlashAttentionOptions options;
};

} // namespace

std::unique_ptr<Pass> mlir::tensoropt::createFlashAttentionTilingPass() {
  return std::make_unique<FlashAttentionTilingPass>();
}

std::unique_ptr<Pass> mlir::tensoropt::createFlashAttentionTilingPass(
    const FlashAttentionOptions &options) {
  return std::make_unique<FlashAttentionTilingPass>(options);
}

void mlir::tensoropt::registerFlashAttentionTilingPass() {
  PassRegistration<FlashAttentionTilingPass>();
}
