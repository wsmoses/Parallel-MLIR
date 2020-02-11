//===- TapirDialect.h - MLIR Dialect for Task Parallelism ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Tapir dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TAPIR_TAPIRDIALECT_H_
#define MLIR_DIALECT_TAPIR_TAPIRDIALECT_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/LoopLikeInterface.h"

namespace mlir {
namespace tapir {

#define GET_OP_CLASSES
#include "mlir/Dialect/Tapir/TapirOps.h.inc"

class TapirDialect : public Dialect {
public:
  explicit TapirDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tapir"; }
};

} // namespace tapir
} // namespace mlir

#endif // MLIR_DIALECT_TAPIR_TAPIRDIALECT_H_
