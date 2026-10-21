add_llvm_executable(mlir-tensoropt-opt
  mlir-tensoropt-opt.cpp
)

target_link_libraries(mlir-tensoropt-opt
  PRIVATE
  MLIRTensorOptPasses
  MLIRMlirOptMain
  MLIR
)
