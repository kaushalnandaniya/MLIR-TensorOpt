add_mlir_library(MLIRTensorOptPasses
  AttentionFusion.cpp
  LoopTiling.cpp
  Vectorization.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/mlir_tensoropt/Passes

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRTransformUtils
  MLIRTransforms
)
