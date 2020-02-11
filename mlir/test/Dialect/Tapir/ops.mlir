// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @parallel_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  tapir.pfor %i0 = %arg0 to %arg1 step %arg2 {
    tapir.pfor %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      tapir.pfor %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}

// CHECK-LABEL: func @parallel_for(
//  CHECK-NEXT:   tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {


func @parallel_for_barrier(%arg0 : index, %arg1 : index, %arg2 : index) {
  tapir.pfor %i0 = %arg0 to %arg1 step %arg2 {
    tapir.pfor %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      tapir.barrier
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      tapir.pfor %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}

// CHECK-LABEL: func @parallel_for_barrier(
//  CHECK-NEXT:   tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       tapir.barrier
//  CHECK-NEXT:       %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       tapir.pfor %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

func @fib(%arg0 : index) -> i64 {
  %c0 = constant 0 : index
  %result = alloc() : memref<i64>
  tapir.syncregion {
     %x = alloc() : memref<i64>
     tapir.detach {
        %c2 = constant 2 : index
        %argm2 = subi %arg0, %c2: index
        %xr = call @fib(%argm2) : (index) -> i64
        store %xr, %x[] : memref<i64>
        tapir.reattach
     }
     %c1 = constant 1 : index
     %argm1 = subi %arg0, %c1: index
     %y = call @fib(%argm1) : (index) -> i64
     tapir.sync
     %ld = load %x[] : memref<i64>
     %add = addi %ld, %y : i64
     store %add, %result[] : memref<i64>
     tapir.syncdone
  }
  %res = load %result[] : memref<i64>
  return %res : i64
}
