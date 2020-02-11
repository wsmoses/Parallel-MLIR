//===-- TapirOps.td - Tapir Parallel dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the Tapir dialect.
//
//===----------------------------------------------------------------------===//


#ifndef TAPIR_OPS
#define TAPIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

def Tapir_Dialect : Dialect {
  let name = "tapir";
}

class Tapir_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<Tapir_Dialect, mnemonic, traits> ;

def SyncRegionOp : Tapir_Op<"syncregion"> {
  let summary = "sync region";
  let description = [{TODO}];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
  }];
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result">
  ];

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def DetachOp : Tapir_Op<"detach"> {
  let summary = "detach";
  let description = [{TODO}];

  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
  }];
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result">
  ];

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ReattachOp : Tapir_Op<"reattach", [HasParent<"DetachOp">, Terminator]> {
  let summary = "reattach construct";
  let description = [{TODO}];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

//should check that syncregion is in parent list
def SyncOp : Tapir_Op<"sync"> {
  let summary = "sync construct";
  let description = [{TODO}];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

def SyncDoneOp : Tapir_Op<"syncdone", [HasParent<"SyncRegionOp">, Terminator]> {
  let summary = "syncdone construct";
  let description = [{TODO}];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

def PForOp : Tapir_Op<"pfor",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "parallel for operation";
  let description = [{TODO}];

  let arguments = (ins Index:$lowerBound, Index:$upperBound, Index:$step);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value lowerBound, Value upperBound, "
              "Value step">
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }
  }];

  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def TerminatorOp : Tapir_Op<"terminator", [Terminator]> {
  let summary = "cf terminator operation";
  let description = [{
    "loop.terminator" is a special terminator operation for blocks inside
    loops. It terminates the region. This operation does _not_ have a custom
    syntax. However, `std` control operations omit the terminator in their
    custom syntax for brevity.

    ```mlir
       loop.terminator
    ```
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

def BarrierOp : Tapir_Op<"barrier", [HasParent<"PForOp">]> {
  let summary = "barrier construct";
  let description = [{
    The barrier construct specifies an explicit barrier at the point at which
    the construct appears.
  }];


  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

#endif // TAPIR_OPS
