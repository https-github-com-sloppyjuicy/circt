// RUN: circt-opt -pass-pipeline='firrtl.circuit(sifive-gct)' -split-input-file %s | FileCheck %s

firrtl.circuit "InterfaceGroundType" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{description = "description of foo", name = "foo", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}, {name = "bar", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]} {
  firrtl.module @InterfaceGroundType() {
    %a = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo", target = []}]} : !firrtl.uint<2>
    %b = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "bar", target = []}]} : !firrtl.uint<4>
  }
}

// This block is checking that all annotations were removed.
// CHECK-LABEL: firrtl.circuit "InterfaceGroundType"
// CHECK-NOT: annotations
// CHECK-SAME: {

// All annotations are removed from the wires.
// CHECK: firrtl.module @InterfaceGroundType
// CHECK: %a = firrtl.wire
// CHECK-NOT: annotations
// CHECK-SAME: !firrtl.uint<2>
// CHECK: %b = firrtl.wire
// CHECK-NOT: annotations
// CHECK-SAME: !firrtl.uint<4>

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "\0A// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : i2
// CHECK-NEXT: sv.interface.signal @bar : i4

// -----

firrtl.circuit "InterfaceVectorType" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{description = "description of foo", name = "foo", tpe = "sifive.enterprise.grandcentral.AugmentedVectorType"}]}]} {
  firrtl.module @InterfaceVectorType(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %a_0 = firrtl.reg %clock {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo"}]} : (!firrtl.clock) -> !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %a_1 = firrtl.regreset %clock, %reset, %c0_ui1 {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo"}]} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// All annotations are removed from the circuit.
// CHECK-LABEL: firrtl.circuit "InterfaceVectorType"
// CHECK-NOT: annotations
// CHECK-SAME: {

// All annotations are removed from the registers.
// CHECK: firrtl.module @InterfaceVectorType
// CHECK: %a_0 = firrtl.reg
// CHECK-NOT: annotations
// CHECK-SAME: (!firrtl.clock) -> !firrtl.uint<1>
// CHECK: %a_1 = firrtl.regreset
// CHECK-NOT: annotations
// CHECK-SAME: (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "\0A// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : !hw.uarray<2xi1>

// -----

firrtl.circuit "InterfaceBundleType" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Bar", elements = [{name = "b", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}, {name = "a", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}, {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{description = "descripton of Bar", name = "Bar", tpe = "sifive.enterprise.grandcentral.AugmentedBundleType"}]}]}  {
  firrtl.module @InterfaceBundleType() {
    %x = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Bar", name = "a"}]} : !firrtl.uint<1>
    %y = firrtl.wire {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Bar", name = "b"}]} : !firrtl.uint<2>
  }
}

// All annotations are removed from the circuit.
// CHECK-LABEL: firrtl.circuit "InterfaceBundleType"
// CHECK-NOT: annotations
// CHECK-SAME: {

// All annotations are removed from the wires.
// CHECK-LABEL: firrtl.module @InterfaceBundleType
// CHECK: %x = firrtl.wire
// CHECK-NOT: annotations
// CHECK-SAME: !firrtl.uint<1>
// CHECK: %y = firrtl.wire
// CHECK-NOT: annotations
// CHECK-SAME: !firrtl.uint<2>

// CHECK: sv.interface @Bar
// CHECK-NEXT: sv.interface.signal @b : i2
// CHECK-NEXT: sv.interface.signal @a : i1

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "\0A// descripton of Bar"
// CHECK-NEXT: Bar Bar();

// -----

firrtl.circuit "InterfaceNode" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{description = "some expression", name = "foo", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]} {
  firrtl.module @InterfaceNode() {
    %a = firrtl.wire : !firrtl.uint<2>
    %notA = firrtl.not %a : (!firrtl.uint<2>) -> !firrtl.uint<2>
    %b = firrtl.node %notA {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo", target = []}]} : !firrtl.uint<2>
  }
}

// All annotations are removed from the circuit.
// CHECK-LABEL: firrtl.circuit "InterfaceNode"
// CHECK-NOT: annotations
// CHECK-SAME: {

// The annotation is removed from the node.
// CHECK: firrtl.node
// CHECK-NOT: annotations
// CHECK: !firrtl.uint<2>

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "\0A// some expression"
// CHECK-NEXT: sv.interface.signal @foo : i2

// -----

firrtl.circuit "InterfacePort" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{description = "description of foo", name = "foo", tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}]} {
  firrtl.module @InterfacePort(in %a : !firrtl.uint<4> {firrtl.annotations = [{a}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", defName = "Foo", name = "foo", target = []}]}) {
  }
}

// All annotations are removed from the circuit.
// CHECK-LABEL: firrtl.circuit "InterfacePort"
// CHECK-NOT: annotations
// CHECK-SAME: {

// The annotations related to Grand Central are removed.
// CHECK: firrtl.module @InterfacePort
// CHECK-SAME: firrtl.annotations = [{a}]

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "\0A// description of foo"
// CHECK-NEXT: sv.interface.signal @foo : i4

// -----

firrtl.circuit "UnsupportedTypes" attributes {annotations = [{class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Foo", elements = [{name = "string", tpe = "sifive.enterprise.grandcentral.AugmentedStringType"}, {name = "boolean", tpe = "sifive.enterprise.grandcentral.AugmentedBooleanType"}, {name = "integer", tpe = "sifive.enterprise.grandcentral.AugmentedIntegerType"}, {name = "double", tpe = "sifive.enterprise.grandcentral.AugmentedDoubleType"}]}]} {
  firrtl.module @UnsupportedTypes() {}
}

// All annotations are removed from the circuit.
// CHECK-LABEL: firrtl.circuit "UnsupportedTypes"
// CHECK-NOT: annotations
// CHECK-SAME: {

// CHECK: sv.interface @Foo
// CHECK-NEXT: sv.verbatim "// string = <unsupported string type>;"
// CHECK-NEXT: sv.verbatim "// boolean = <unsupported boolean type>;"
// CHECK-NEXT: sv.verbatim "// integer = <unsupported integer type>;"
// CHECK-NEXT: sv.verbatim "// double = <unsupported double type>;"
