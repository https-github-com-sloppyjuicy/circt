// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -create-dataflow %s | FileCheck %s
  func @affine_load(%arg0: index) {
// CHECK:       module {

// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: none, ...) -> none {
// CHECK:           %[[VAL_2:.*]]:2 = "handshake.memory"(%[[VAL_3:.*]]) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xf32>} : (index) -> (f32, none)
// CHECK:           %[[VAL_4:.*]] = "handshake.merge"(%[[VAL_0]]) : (index) -> index
// CHECK:           %[[VAL_5:.*]]:4 = "handshake.fork"(%[[VAL_1]]) {control = true} : (none) -> (none, none, none, none)
// CHECK:           %[[VAL_6:.*]] = "handshake.constant"(%[[VAL_5]]#2) {value = 0 : index} : (none) -> index
// CHECK:           %[[VAL_7:.*]] = "handshake.constant"(%[[VAL_5]]#1) {value = 10 : index} : (none) -> index
// CHECK:           %[[VAL_8:.*]] = "handshake.constant"(%[[VAL_5]]#0) {value = 1 : index} : (none) -> index
// CHECK:           %[[VAL_9:.*]] = "handshake.branch"(%[[VAL_4]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_10:.*]] = "handshake.branch"(%[[VAL_5]]#3) {control = true} : (none) -> none
// CHECK:           %[[VAL_11:.*]] = "handshake.branch"(%[[VAL_6]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_12:.*]] = "handshake.branch"(%[[VAL_7]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_13:.*]] = "handshake.branch"(%[[VAL_8]]) {control = false} : (index) -> index
// CHECK:           "handshake.terminator"()[^bb1] : () -> ()
// CHECK:         ^bb1:
// CHECK:           %[[VAL_14:.*]] = "handshake.mux"(%[[VAL_15:.*]]#3, %[[VAL_16:.*]], %[[VAL_12]]) : (index, index, index) -> index
// CHECK:           %[[VAL_17:.*]]:2 = "handshake.fork"(%[[VAL_14]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_18:.*]] = "handshake.mux"(%[[VAL_15]]#2, %[[VAL_19:.*]], %[[VAL_9]]) : (index, index, index) -> index
// CHECK:           %[[VAL_20:.*]] = "handshake.mux"(%[[VAL_15]]#1, %[[VAL_21:.*]], %[[VAL_13]]) : (index, index, index) -> index
// CHECK:           %[[VAL_22:.*]]:2 = "handshake.control_merge"(%[[VAL_23:.*]], %[[VAL_10]]) {control = true} : (none, none) -> (none, index)
// CHECK:           %[[VAL_15]]:4 = "handshake.fork"(%[[VAL_22]]#1) {control = false} : (index) -> (index, index, index, index)
// CHECK:           %[[VAL_24:.*]] = "handshake.mux"(%[[VAL_15]]#0, %[[VAL_25:.*]], %[[VAL_11]]) : (index, index, index) -> index
// CHECK:           %[[VAL_26:.*]]:2 = "handshake.fork"(%[[VAL_24]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_27:.*]] = cmpi "slt", %[[VAL_26]]#1, %[[VAL_17]]#1 : index
// CHECK:           %[[VAL_28:.*]]:5 = "handshake.fork"(%[[VAL_27]]) {control = false} : (i1) -> (i1, i1, i1, i1, i1)
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = "handshake.conditional_branch"(%[[VAL_28]]#4, %[[VAL_17]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_30]]) : (index) -> ()
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = "handshake.conditional_branch"(%[[VAL_28]]#3, %[[VAL_18]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_32]]) : (index) -> ()
// CHECK:           %[[VAL_33:.*]], %[[VAL_34:.*]] = "handshake.conditional_branch"(%[[VAL_28]]#2, %[[VAL_20]]) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_34]]) : (index) -> ()
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = "handshake.conditional_branch"(%[[VAL_28]]#1, %[[VAL_22]]#0) {control = true} : (i1, none) -> (none, none)
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = "handshake.conditional_branch"(%[[VAL_28]]#0, %[[VAL_26]]#0) {control = false} : (i1, index) -> (index, index)
// CHECK:           "handshake.sink"(%[[VAL_38]]) : (index) -> ()
// CHECK:           "handshake.terminator"()[^bb2, ^bb3] : () -> ()
// CHECK:         ^bb2:
// CHECK:           %[[VAL_39:.*]] = "handshake.merge"(%[[VAL_37]]) : (index) -> index
// CHECK:           %[[VAL_40:.*]]:2 = "handshake.fork"(%[[VAL_39]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_41:.*]] = "handshake.merge"(%[[VAL_31]]) : (index) -> index
// CHECK:           %[[VAL_42:.*]]:2 = "handshake.fork"(%[[VAL_41]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_43:.*]] = "handshake.merge"(%[[VAL_33]]) : (index) -> index
// CHECK:           %[[VAL_44:.*]]:2 = "handshake.fork"(%[[VAL_43]]) {control = false} : (index) -> (index, index)
// CHECK:           %[[VAL_45:.*]] = "handshake.merge"(%[[VAL_29]]) : (index) -> index
// CHECK:           %[[VAL_46:.*]]:2 = "handshake.control_merge"(%[[VAL_35]]) {control = true} : (none) -> (none, index)
// CHECK:           %[[VAL_47:.*]]:2 = "handshake.fork"(%[[VAL_46]]#0) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_48:.*]]:2 = "handshake.fork"(%[[VAL_47]]#1) {control = true} : (none) -> (none, none)
// CHECK:           %[[VAL_49:.*]] = "handshake.join"(%[[VAL_48]]#1, %[[VAL_2]]#1) {control = true} : (none, none) -> none
// CHECK:           "handshake.sink"(%[[VAL_46]]#1) : (index) -> ()
// CHECK:           %[[VAL_50:.*]] = addi %[[VAL_40]]#1, %[[VAL_42]]#1 : index
// CHECK:           %[[VAL_51:.*]] = "handshake.constant"(%[[VAL_48]]#0) {value = 7 : index} : (none) -> index
// CHECK:           %[[VAL_52:.*]] = addi %[[VAL_50]], %[[VAL_51]] : index
// CHECK:           %[[VAL_53:.*]], %[[VAL_3]] = "handshake.load"(%[[VAL_52]], %[[VAL_2]]#0, %[[VAL_47]]#0) : (index, f32, none) -> (f32, index)
// CHECK:           "handshake.sink"(%[[VAL_53]]) : (f32) -> ()
// CHECK:           %[[VAL_54:.*]] = addi %[[VAL_40]]#0, %[[VAL_44]]#1 : index
// CHECK:           %[[VAL_19]] = "handshake.branch"(%[[VAL_42]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_21]] = "handshake.branch"(%[[VAL_44]]#0) {control = false} : (index) -> index
// CHECK:           %[[VAL_16]] = "handshake.branch"(%[[VAL_45]]) {control = false} : (index) -> index
// CHECK:           %[[VAL_23]] = "handshake.branch"(%[[VAL_49]]) {control = true} : (none) -> none
// CHECK:           %[[VAL_25]] = "handshake.branch"(%[[VAL_54]]) {control = false} : (index) -> index
// CHECK:           "handshake.terminator"()[^bb1] : () -> ()
// CHECK:         ^bb3:
// CHECK:           %[[VAL_55:.*]]:2 = "handshake.control_merge"(%[[VAL_36]]) {control = true} : (none) -> (none, index)
// CHECK:           "handshake.sink"(%[[VAL_55]]#1) : (index) -> ()
// CHECK:           handshake.return %[[VAL_55]]#0 : none
// CHECK:         }
// CHECK:       }
    %0 = alloc() : memref<10xf32>
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
    br ^bb1(%6 : index)
  ^bb3: // pred: ^bb1
    return
  }
