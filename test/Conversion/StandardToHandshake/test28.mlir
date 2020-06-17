// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -create-dataflow %s | FileCheck %s
  func @affine_load(%arg0: index) {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> none {
// CHECK:           %[[VAL_2:.*]]:3 = "handshake.memory"(%[[VAL_3:.*]]#0, %[[VAL_3]]#1, %[[VAL_4:.*]]) {id = 1 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = "handshake.fork"(%[[VAL_2]]#2) {control = false} : (none) -> (none, none)
// CHECK:           %[[VAL_6:.*]]:2 = "handshake.memory"(%[[VAL_7:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xf32>} : (index) -> (f32, none)
// CHECK:           %[[VAL_8:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_9:.*]]:4 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_10:.*]] = "handshake.constant"(%[[VAL_9]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_11:.*]] = "handshake.constant"(%[[VAL_9]]#1) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_12:.*]] = "handshake.constant"(%[[VAL_9]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_13:.*]] = "handshake.branch"(%[[VAL_8]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_14:.*]] = "handshake.branch"(%[[VAL_9]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_15:.*]] = "handshake.branch"(%[[VAL_10]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_16:.*]] = "handshake.branch"(%[[VAL_11]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_17:.*]] = "handshake.branch"(%[[VAL_12]]) {control = false} : (index) -> index
// CHECK:           "handshake.terminator"()[^bb1] : () -> ()
// CHECK:         ^bb1:
// CHECK:           %[[VAL_18:.*]] = "handshake.mux"(%[[VAL_19:.*]]#3, %[[VAL_20:.*]], %[[VAL_16]]) : (index, index, index) -> index
// CHECK:           %[[VAL_21:.*]]:2 = "handshake.fork"(%[[VAL_18]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_22:.*]] = "handshake.mux"(%[[VAL_19]]#2, %[[VAL_23:.*]], %[[VAL_13]]) : (index, index, index) -> index
// CHECK:           %[[VAL_24:.*]] = "handshake.mux"(%[[VAL_19]]#1, %[[VAL_25:.*]], %[[VAL_17]]) : (index, index, index) -> index
// CHECK:           %[[VAL_26:.*]]:2 = "handshake.control_merge"(%[[VAL_27:.*]], %[[VAL_14]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_19]]:4 = "handshake.fork"(%[[VAL_26]]#1) {control = false} : (index) -> (index, index, index, index)
// CHECK:           %[[VAL_28:.*]] = "handshake.mux"(%[[VAL_19]]#0, %[[VAL_29:.*]], %[[VAL_15]]) : (index, index, index) -> index
// CHECK:           %[[VAL_30:.*]]:2 = "handshake.fork"(%[[VAL_28]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_31:.*]] = cmpi "slt", %[[VAL_30]]#1, %[[VAL_21]]#1 : index
// CHECK:           %[[VAL_32:.*]]:5 = "handshake.fork"(%[[VAL_31]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#4, %[[VAL_21]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_34]]) : (index) -> ()
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#3, %[[VAL_22]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_36]]) : (index) -> ()
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#2, %[[VAL_24]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_38]]) : (index) -> ()
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#1, %[[VAL_26]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = "handshake.conditional_branch"(%[[VAL_32]]#0, %[[VAL_30]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_42]]) : (index) -> ()
// CHECK:           "handshake.terminator"()[^bb2, ^bb3] : () -> ()
// CHECK:         ^bb2:
// CHECK:           %[[VAL_43:.*]] = "handshake.merge"(%[[VAL_41]]) : (index) -> index
// CHECK:           %[[VAL_44:.*]]:2 = "handshake.fork"(%[[VAL_43]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_45:.*]] = "handshake.merge"(%[[VAL_35]]) : (index) -> index
// CHECK:           %[[VAL_46:.*]]:2 = "handshake.fork"(%[[VAL_45]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_47:.*]] = "handshake.merge"(%[[VAL_37]]) : (index) -> index
// CHECK:           %[[VAL_48:.*]]:2 = "handshake.fork"(%[[VAL_47]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_49:.*]] = "handshake.merge"(%[[VAL_33]]) : (index) -> index
// CHECK:           %[[VAL_50:.*]]:2 = "handshake.control_merge"(%[[VAL_39]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_51:.*]]:4 = "handshake.fork"(%[[VAL_50]]#0) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_52:.*]]:2 = "handshake.fork"(%[[VAL_51]]#3) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_53:.*]] = "handshake.join"(%[[VAL_52]]#1, %[[VAL_6]]#1, %[[VAL_5]]#1, %[[VAL_2]]#1) {control = true} : (none, none, none, none) -> none
// CHECK:           "handshake.sink"(%[[VAL_50]]#1) : (index) -> ()
// CHECK:           %[[VAL_54:.*]] = addi %[[VAL_44]]#1, %[[VAL_46]]#1 : index
// CHECK:           %[[VAL_55:.*]] = "handshake.constant"(%[[VAL_52]]#0) {value = 7 : index} : (none) -> index
// CHECK:           %[[VAL_56:.*]] = addi %[[VAL_54]], %[[VAL_55]] : index
// CHECK:           %[[VAL_57:.*]]:3 = "handshake.fork"(%[[VAL_56]]) {control = false} : (index) -> (index, index, index)
// CHECK:           %[[VAL_58:.*]], %[[VAL_7]] = "handshake.load"(%[[VAL_57]]#2, %[[VAL_6]]#0, %[[VAL_51]]#2) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_59:.*]] = addi %[[VAL_44]]#0, %[[VAL_48]]#1 : index
// CHECK:           %[[VAL_60:.*]], %[[VAL_4]] = "handshake.load"(%[[VAL_57]]#1, %[[VAL_2]]#0, %[[VAL_51]]#1) : (index, f32, none) -> (f32, index)
// CHECK:           %[[VAL_61:.*]] = addf %[[VAL_58]], %[[VAL_60]] : f32
// CHECK:           %[[VAL_62:.*]] = "handshake.join"(%[[VAL_51]]#0, %[[VAL_5]]#0) {control = true} : (none, none) -> none
// CHECK:           %[[VAL_3]]:2 = "handshake.store"(%[[VAL_61]], %[[VAL_57]]#0, %[[VAL_62]]) : (f32, index, none) -> (f32, index)
// CHECK:           %[[VAL_23]] = "handshake.branch"(%[[VAL_46]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_25]] = "handshake.branch"(%[[VAL_48]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_20]] = "handshake.branch"(%[[VAL_49]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_27]] = "handshake.branch"(%[[VAL_53]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_29]] = "handshake.branch"(%[[VAL_59]]) {control = false} : (index) -> index
// CHECK:           "handshake.terminator"()[^bb1] : () -> ()
// CHECK:         ^bb3:
// CHECK:           %[[VAL_63:.*]]:2 = "handshake.control_merge"(%[[VAL_40]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_63]]#1) : (index) -> ()
// CHECK:           handshake.return %[[VAL_63]]#0 : none
// CHECK:         }
// CHECK:       }
    %0 = alloc() : memref<10xf32>
    %10 = alloc() : memref<10xf32>
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
    %2 = cmpi "slt", %1, %c10 : index
    cond_br %2, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    %3 = addi %1, %arg0 : index
    %c7 = constant 7 : index
    %4 = addi %3, %c7 : index
    %5 = load %0[%4] : memref<10xf32>
    %6 = addi %1, %c1 : index
    %7 = load %10[%4] : memref<10xf32>
    %8 = addf %5, %7 : f32
    store %8, %10[%4] : memref<10xf32>
    br ^bb1(%6 : index)
  ^bb3: // pred: ^bb1
    return
  }
