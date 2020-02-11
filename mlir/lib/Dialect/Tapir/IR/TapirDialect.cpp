//===- TapirDialect.cpp - MLIR Dialect for Tapir implementation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Tapir dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tapir/TapirDialect.h"
#include "mlir/IR/OpImplementation.h"


#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

using namespace mlir;
using namespace mlir::tapir;


//===----------------------------------------------------------------------===//
// LoopOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

struct LoopSideEffectsInterface : public SideEffectsDialectInterface {
  using SideEffectsDialectInterface::SideEffectsDialectInterface;

  SideEffecting isSideEffecting(Operation *op) const override {
    if (isa<PForOp>(op)) {
      return Recursive;
    }
    return SideEffectsDialectInterface::isSideEffecting(op);
  };
};

} // namespace

TapirDialect::TapirDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tapir/TapirOps.cpp.inc"
      >();
}

namespace mlir {
namespace tapir {

static LogicalResult verify(PForOp op);
static void print(OpAsmPrinter &p, PForOp op);
static ParseResult parsePForOp(OpAsmParser &parser, OperationState &result);

static LogicalResult verify(SyncRegionOp op);
static void print(OpAsmPrinter &p, SyncRegionOp op);
static ParseResult parseSyncRegionOp(OpAsmParser &parser, OperationState &result);

static LogicalResult verify(DetachOp op);
static void print(OpAsmPrinter &p, DetachOp op);
static ParseResult parseDetachOp(OpAsmParser &parser, OperationState &result);

#define GET_OP_CLASSES
#include "mlir/Dialect/Tapir/TapirOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PForOp
//===----------------------------------------------------------------------===//

void PForOp::build(Builder *builder, OperationState &result, Value lb, Value ub, Value step) {
  result.addOperands({lb, ub, step});
  Region *bodyRegion = result.addRegion();
  PForOp::ensureTerminator(*bodyRegion, *builder, result.location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

static LogicalResult verify(PForOp op) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(op.step().getDefiningOp()))
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (body->getNumArguments() != 1 || !body->getArgument(0).getType().isIndex())
    return op.emitOpError("expected body to have a single index argument for "
                          "the induction variable");
  return success();
}

static void print(OpAsmPrinter &p, PForOp op) {
  p << op.getOperationName() << " " << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parsePForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable, indexType))
    return failure();

  PForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Region &PForOp::getLoopBody() { return region(); }

bool PForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult PForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(this->getOperation());
  return success();
}


//===----------------------------------------------------------------------===//
// SyncRegionOp
//===----------------------------------------------------------------------===//

void SyncRegionOp::build(Builder *builder, OperationState &result) {
  //Region *bodyRegion = result.addRegion();
  //SyncRegionOp::ensureTerminator(*bodyRegion, *builder, result.location);
  //bodyRegion->front().addArgument(builder->getIndexType());
}

static LogicalResult verify(SyncRegionOp op) {
  return success();
}

static void print(OpAsmPrinter &p, SyncRegionOp op) {
  p << op.getOperationName();
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseSyncRegionOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, {}))
    return failure();

  //PForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}


//===----------------------------------------------------------------------===//
// DetachOp
//===----------------------------------------------------------------------===//

void DetachOp::build(Builder *builder, OperationState &result) {
  //Region *bodyRegion = result.addRegion();
  //SyncRegionOp::ensureTerminator(*bodyRegion, *builder, result.location);
  //bodyRegion->front().addArgument(builder->getIndexType());
}

static LogicalResult verify(DetachOp op) {
  return success();
}

static void print(OpAsmPrinter &p, DetachOp op) {
  p << op.getOperationName();
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseDetachOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, {}))
    return failure();

  //PForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

} // namespace parallel
} // namespace mlir
