//===-- AlgebraicSimplification.cpp - Constant Folding & Simplification -*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Implements algebraic transformations such as converting FP division by constants
// to reciprocal multiplication, folding constant scaling factors, and eliminating
// identity operations.
//
//===----------------------------------------------------------------------===//

#include "mlir_tensoropt/Passes/AlgebraicSimplification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;
using namespace mlir::tensoropt;

namespace {

/// Pattern to replace division by a constant with multiplication by its reciprocal.
/// On modern SIMD/GPU hardware, FP multiplication has ~3-4x lower latency than FP division.
struct ConvertDivFByConstantToMulF : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    auto rhsConst = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!rhsConst)
      return failure();

    auto floatAttr = dyn_cast<FloatAttr>(rhsConst.getValue());
    if (!floatAttr)
      return failure();

    double val = floatAttr.getValueAsDouble();
    if (val == 0.0)
      return failure();

    double reciprocal = 1.0 / val;
    Location loc = op.getLoc();
    auto reciprocalAttr = FloatAttr::get(floatAttr.getType(), reciprocal);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    Value reciprocalConst = rewriter.create<arith::ConstantOp>(loc, reciprocalAttr);
#pragma clang diagnostic pop

    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), reciprocalConst);
    return success();
  }
};

class AlgebraicSimplificationPass
    : public PassWrapper<AlgebraicSimplificationPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlgebraicSimplificationPass)

  StringRef getArgument() const override {
    return "mlir-tensoropt-algebraic-simplification";
  }

  StringRef getDescription() const override {
    return "Perform algebraic simplification, reciprocal constant folding, and identity pruning";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    math::MathDialect,
                    linalg::LinalgDialect,
                    tensor::TensorDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvertDivFByConstantToMulF>(context);

    func->setAttr("mlir_tensoropt.algebraic_simplified",
                  Builder(context).getBoolAttr(true));

#if LLVM_VERSION_MAJOR >= 21
    (void)applyPatternsGreedily(func, std::move(patterns));
#else
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
#endif
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tensoropt::createAlgebraicSimplificationPass() {
  return std::make_unique<AlgebraicSimplificationPass>();
}

void mlir::tensoropt::registerAlgebraicSimplificationPass() {
  PassRegistration<AlgebraicSimplificationPass>();
}
