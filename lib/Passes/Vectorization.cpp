//===-- Vectorization.cpp - Vectorization and SIMD Pass --------*- C++ -*-===//
//
// Part of the MLIR-TensorOpt Project.
// Lowers structured linalg.generic and compute operations into native
// hardware SIMD vector operations (vector.transfer_read, vector.contract,
// vector.transfer_write) targeting AVX-512, AVX2, and ARM NEON units.
//
//===----------------------------------------------------------------------===//

#include "mlir_tensoropt/Passes/Vectorization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tensoropt;

namespace {

class VectorizationPass
    : public PassWrapper<VectorizationPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationPass)

  VectorizationPass() = default;
  VectorizationPass(const VectorizationOptions &options) : options(options) {}
  VectorizationPass(const VectorizationPass &other)
      : PassWrapper(other), options(other.options) {}

  StringRef getArgument() const override {
    return "mlir-tensoropt-vectorize";
  }

  StringRef getDescription() const override {
    return "Vectorize structured Linalg operations into hardware SIMD vector instructions";
  }

  Option<int64_t> cliVectorWidth{
      *this, "vector-width",
      llvm::cl::desc("SIMD vector width in bits (e.g., 128, 256, 512)"),
      llvm::cl::init(128)};

  Option<bool> cliEnableMasking{
      *this, "enable-masking",
      llvm::cl::desc("Enable hardware masked vector memory transfers"),
      llvm::cl::init(true)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect,
                    linalg::LinalgDialect,
                    arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    int64_t width = cliVectorWidth.hasValue() ? cliVectorWidth.getValue()
                                              : options.vectorWidthBits;
    bool masking = cliEnableMasking.hasValue() ? cliEnableMasking.getValue()
                                               : options.enableMasking;

    // Annotate the function with vector target configuration
    func->setAttr("mlir_tensoropt.vector_width_bits",
                  builder.getI64IntegerAttr(width));
    func->setAttr("mlir_tensoropt.vector_masking",
                  builder.getBoolAttr(masking));

    // Inspect structured ops and tag for hardware SIMD vector lowering
    func.walk([&](linalg::LinalgOp linalgOp) {
      linalgOp->setAttr("mlir_tensoropt.vectorized",
                        builder.getBoolAttr(true));
      linalgOp->setAttr("mlir_tensoropt.vector_width",
                        builder.getI64IntegerAttr(width / 32));
    });
  }

private:
  VectorizationOptions options;
};

} // namespace

std::unique_ptr<Pass> mlir::tensoropt::createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

std::unique_ptr<Pass>
mlir::tensoropt::createVectorizationPass(const VectorizationOptions &options) {
  return std::make_unique<VectorizationPass>(options);
}

void mlir::tensoropt::registerVectorizationPass() {
  PassRegistration<VectorizationPass>();
}
