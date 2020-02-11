add_llvm_library(MLIRTapir
  IR/TapirDialect.cpp
  TapirOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tapir
  )

add_dependencies(MLIRTapir
    MLIRTapirOpsIncGen
    MLIREDSC
    MLIRStandardOps
    LLVMSupport
)

target_link_libraries(MLIRTapir
  MLIRIR
  MLIREDSC
  MLIRStandardOps
  LLVMSupport
)
