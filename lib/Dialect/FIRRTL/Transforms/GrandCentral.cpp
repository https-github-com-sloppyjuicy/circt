//===- GrandCentral.cpp - Ingest black box sources --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Implement SiFive's Grand Central transform.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
/// Mutable store of information about an Element in an interface.  This is
/// derived from information stored in the "elements" field of an
/// "AugmentedBundleType".  This is updated as more information is known about
/// an Element.
struct ElementInfo {
  /// Encodes the "tpe" of an element.  This is called "Kind" to avoid
  /// overloading the meeaning of "Type" (which also conflicts with mlir::Type).
  enum Kind {
    Error = -1,
    Ground,
    Vector,
    Bundle,
    String,
    Boolean,
    Integer,
    Double
  };
  /// The "tpe" field indicating if this element of the interface is a ground
  /// type, a vector type, or a bundle type.  Bundle types are nested
  /// interfaces.
  Kind tpe;
  /// A string description that will show up as a comment in the output Verilog.
  StringRef description;
  /// The width of this interface.  This is only non-negative for ground or
  /// vector types.
  int32_t width = -1;
  /// The depth of the interface.  This is one for ground types and greater
  /// than one for vector types.
  uint32_t depth = 0;
  /// Indicate if this element was found in the circuit.
  bool found = false;
  /// Trakcs location information about what was used to build this element.
  SmallVector<Location> locations = SmallVector<Location>();
  /// True if this is a ground or vector type and it was not (statefully) found.
  /// This indicates that an interface element, which is composed of ground and
  /// vector types, found no matching, annotated components in the circuit.
  bool isMissing() { return !found && (tpe == Ground || tpe == Vector); }
};

/// Stores a decoded Grand Central AugmentedField
struct AugmentedField {
  /// The name of the field.
  StringRef name;
  /// An optional descripton that the user provided for the field.  This should
  /// become a comment in the Verilog.
  StringRef description;
  /// The "type" of the field.
  ElementInfo::Kind tpe;
};

/// Stores a decoded Grand Central AugmentedBundleType.
struct AugmentedBundleType {
  /// The name of the interface.
  StringRef defName;
  /// The elements that make up the body of the interface.
  SmallVector<AugmentedField> elements;
};

/// Convert an arbitrary attributes into an optional AugmentedField.  Returns
/// None if the attribute is an invalid AugmentedField.
static Optional<AugmentedField> decodeField(Attribute maybeField) {
  auto field = maybeField.dyn_cast_or_null<DictionaryAttr>();
  if (!field)
    return {};
  auto tpeString = field.getAs<StringAttr>("tpe");
  auto name = field.getAs<StringAttr>("name");
  if (!name || !tpeString)
    return {};
  auto tpe = llvm::StringSwitch<ElementInfo::Kind>(tpeString.getValue())
                 .Case("sifive.enterprise.grandcentral.AugmentedBundleType",
                       ElementInfo::Bundle)
                 .Case("sifive.enterprise.grandcentral.AugmentedVectorType",
                       ElementInfo::Vector)
                 .Case("sifive.enterprise.grandcentral.AugmentedGroundType",
                       ElementInfo::Ground)
                 .Case("sifive.enterprise.grandcentral.AugmentedStringType",
                       ElementInfo::String)
                 .Case("sifive.enterprise.grandcentral.AugmentedBooleanType",
                       ElementInfo::Boolean)
                 .Case("sifive.enterprise.grandcentral.AugmentedIntegerType",
                       ElementInfo::Integer)
                 .Case("sifive.enterprise.grandcentral.AugmentedDoubleType",
                       ElementInfo::Double)
                 .Default(ElementInfo::Error);
  if (tpe == ElementInfo::Error)
    return {};

  StringRef description = {};
  if (auto maybeDescription = field.getAs<StringAttr>("description"))
    description = maybeDescription.getValue();
  return Optional<AugmentedField>({name.getValue(), description, tpe});
};

/// Convert an Annotation into an optional AugmentedBundleType.  Returns None if
/// the annotation is not an AugmentedBundleType.
static Optional<AugmentedBundleType> decodeBundleType(Annotation anno) {
  auto defName = anno.getMember<StringAttr>("defName");
  auto elements = anno.getMember<ArrayAttr>("elements");
  if (!defName || !elements)
    return {};
  AugmentedBundleType bundle(
      {defName.getValue(), SmallVector<AugmentedField>()});
  for (auto element : elements) {
    auto field = decodeField(element);
    if (!field)
      return {};
    bundle.elements.push_back(field.getValue());
  }
  return Optional<AugmentedBundleType>(bundle);
};

/// Remove Grand Central Annotations associated with SystemVerilog interfaces
/// that should emitted.  This pass works in three major phases:
///
/// 1. The circuit's annotations are examnined to figure out _what_ interfaces
///    there are.  This includes information about the name of the interface
///    ("defName") and each of the elements (sv::InterfaceSignalOp) that make up
///    the interface.  However, no information about the _type_ of the elements
///    is known.
///
/// 2. With this, information, walk through the circuit to find scattered
///    information about the types of the interface elements.  Annotations are
///    scattered during FIRRTL parsing to attach all the annotations associated
///    with elements on the right components.
///
/// 3. Add interface ops and populate the elements.
///
/// Grand Central supports three "normal" element types and four "weird" element
/// types.  The normal ones are ground types (SystemVerilog logic), vector types
/// (SystemVerilog unpacked arrays), and nested interface types (another
/// SystemVerilog interface).  The Chisel API provides "weird" elements that
/// include: Boolean, Integer, String, and Double.  The SFC implementation
/// currently drops these, but this pass emits them as commented out strings.
struct GrandCentralPass : public GrandCentralBase<GrandCentralPass> {
  void runOnOperation() override;
};

class GrandCentralVisitor : public FIRRTLVisitor<GrandCentralVisitor> {
public:
  GrandCentralVisitor(MLIRContext *context,
                      llvm::DenseMap<std::pair<StringRef, StringRef>,
                                     ElementInfo> &interfaceMap)
      : context(context), interfaceMap(interfaceMap) {}

private:
  MLIRContext *context;

  /// Mutable store tracking each element in an interface.  This is indexed by a
  /// "defName" -> "name" tuple.
  llvm::DenseMap<std::pair<StringRef, StringRef>, ElementInfo> &interfaceMap;

  /// Helper to handle wires, registers, and nodes.
  void handleRef(Operation *op);

  /// A helper used by handleRef that can also be used to process ports.
  ArrayAttr handleRefLike(Operation *op, AnnotationSet annotations,
                          FIRRTLType type);

  // Helper to handle ports of modules that may have Grand Central annotations.
  ArrayAttr handlePorts(Operation *op);

  // If true, then some error occurred while the visitor was running.  This
  // indicates that pass failure should occur.
  bool failed = false;

public:
  using FIRRTLVisitor<GrandCentralVisitor>::visitDecl;

  /// Visit FModuleOp and FExtModuleOp
  void visitModule(Operation *op);

  /// Visit ops that can make up an interface element.
  void visitDecl(RegOp op) { handleRef(op); }
  void visitDecl(RegResetOp op) { handleRef(op); }
  void visitDecl(WireOp op) { handleRef(op); }
  void visitDecl(NodeOp op) { handleRef(op); }

  /// Process all other ops.  Error if any of these ops contain annotations that
  /// indicate it as being part of an interface.
  void visitUnhandledDecl(Operation *op);

  /// Returns true if an error condition occurred while visiting ops.
  bool hasFailed() { return failed; };
};

} // namespace

void GrandCentralVisitor::visitModule(Operation *op) {
  mlir::function_like_impl::setAllArgAttrDicts(op, handlePorts(op).getValue());

  if (isa<FModuleOp>(op))
    for (auto &stmt : op->getRegion(0).front())
      dispatchVisitor(&stmt);
}

/// Process all other operations.  This will throw an error if the operation
/// contains any annotations that indicates that this should be included in an
/// interface.  Otherwise, this is a valid nop.
void GrandCentralVisitor::visitUnhandledDecl(Operation *op) {
  AnnotationSet annotations(op);
  auto anno = annotations.getAnnotation(
      "sifive.enterprise.grandcentral.AugmentedGroundType");
  if (anno) {
    auto diag =
        op->emitOpError()
        << "is marked as a an interface element, but this op or its ports are "
           "not supposed to be interface elements (Are your annotations "
           "malformed? Is this a missing feature that should be supported?)";
    diag.attachNote()
        << "this annotation marked the op as an interface element: '" << anno
        << "'";
    failed = true;
  }
}

/// Process annotations associated with an operation and having some type.
/// Return the annotations with the processed annotations removed.  If all
/// annotations are removed, this returns an empty ArrayAttr.
ArrayAttr GrandCentralVisitor::handleRefLike(mlir::Operation *op,
                                             AnnotationSet annotations,
                                             FIRRTLType type) {
  if (annotations.empty())
    return ArrayAttr::get(context, {});

  llvm::SmallVector<Attribute> unprocessedAnnos;

  for (auto anno : annotations) {
    if (!anno.isClass("sifive.enterprise.grandcentral.AugmentedGroundType")) {
      unprocessedAnnos.push_back(anno.getDict());
      continue;
    }

    auto defName = anno.getMember<StringAttr>("defName").getValue();
    auto name = anno.getMember<StringAttr>("name").getValue();

    // TODO: This is ignoring situations where the leaves of and interface are
    // not ground types.  This enforces the requirement that this runs after
    // LowerTypes.  However, this could eventually be relaxed.
    if (!type.isGround()) {
      auto diag = op->emitOpError()
                  << "cannot be added to interface '" << defName
                  << "', component '" << name
                  << "' because it is not a ground type. (Got type '" << type
                  << "'.) This will be dropped from the interface. (Did you "
                     "forget to run LowerTypes?)";
      diag.attachNote()
          << "The annotation indicating that this should be added was: '"
          << anno.getDict();
      failed = true;
      unprocessedAnnos.push_back(anno.getDict());
      continue;
    }

    auto &component = interfaceMap[{defName, name}];
    component.found = true;

    switch (component.tpe) {
    case ElementInfo::Vector:
      component.width = type.getBitWidthOrSentinel();
      component.depth++;
      component.locations.push_back(op->getLoc());
      break;
    case ElementInfo::Ground:
      component.width = type.getBitWidthOrSentinel();
      component.locations.push_back(op->getLoc());
      break;
    case ElementInfo::Bundle:
    case ElementInfo::String... ElementInfo::Double:
      break;
    case ElementInfo::Error:
      llvm_unreachable("Shouldn't be here");
      break;
    }
  }

  return ArrayAttr::get(context, unprocessedAnnos);
}

/// Combined logic to handle Wires, Registers, and Nodes because these all use
/// the same approach.
void GrandCentralVisitor::handleRef(mlir::Operation *op) {
  op->setAttr("annotations",
              handleRefLike(op, AnnotationSet(op),
                            op->getResult(0).getType().cast<FIRRTLType>()));
}

/// Remove Grand Central Annotations from ports of modules or external modules.
/// Return argument attributes with annotations removed.
ArrayAttr GrandCentralVisitor::handlePorts(Operation *op) {

  SmallVector<Attribute> newArgAttrs;
  auto ports = getModulePortInfo(op);
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    auto port = ports[i];
    auto argAttr = mlir::function_like_impl::getArgAttrs(op, i);
    if (port.annotations.empty()) {
      newArgAttrs.push_back(DictionaryAttr::get(op->getContext(), argAttr));
      continue;
    }

    auto remainingAnnotations = handleRefLike(op, port.annotations, port.type);

    // Overwrite the annotations value with remaining annotations.
    SmallVector<NamedAttribute> newArgAttr;
    for (auto argIter = argAttr.begin(), argEnd = argAttr.end();
         argIter != argEnd; ++argIter) {
      if (argIter->first != "firrtl.annotations") {
        newArgAttr.push_back(*argIter);
        continue;
      }
      newArgAttr.push_back({argIter->first, remainingAnnotations});
    }
    newArgAttrs.push_back(DictionaryAttr::get(op->getContext(), newArgAttr));
  }

  return ArrayAttr::get(op->getContext(), newArgAttrs);
}

void GrandCentralPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  AnnotationSet annotations(circuitOp);
  if (annotations.empty())
    return;

  auto builder = OpBuilder::atBlockEnd(circuitOp->getBlock());

  // Store a mapping of interface name to InterfaceOp.
  llvm::StringMap<sv::InterfaceOp> interfaces;

  // Discovered interfaces that need to be constructed.
  llvm::DenseMap<std::pair<StringRef, StringRef>, ElementInfo> interfaceMap;

  // Track the order that interfaces should be emitted in.
  llvm::SmallVector<std::pair<StringRef, StringRef>> interfaceKeys;

  // Examine the Circuit's Annotations doing work to remove Grand Central
  // Annotations.  Ignore any unprocesssed annotations and rewrite the Circuit's
  // Annotations with these when done.
  llvm::SmallVector<Attribute> unprocessedAnnos;
  for (auto anno : annotations) {
    if (!anno.isClass("sifive.enterprise.grandcentral.AugmentedBundleType")) {
      unprocessedAnnos.push_back(anno.getDict());
      continue;
    }

    AugmentedBundleType bundle;
    if (auto maybeBundle = decodeBundleType(anno))
      bundle = maybeBundle.getValue();
    else {
      emitError(circuitOp.getLoc(),
                "'firrtl.circuit' op contained an 'AugmentedBundleType' "
                "Annotation which did not conform to the expected format")
              .attachNote()
          << "the problematic 'AugmentedBundleType' is: '" << anno.getDict()
          << "'";
      return signalPassFailure();
    }

    for (auto elt : bundle.elements) {
      std::pair<StringRef, StringRef> key = {bundle.defName, elt.name};
      interfaceMap[key] = {elt.tpe, elt.description};
      interfaceKeys.push_back(key);
    }

    // If the interface already exists, don't create it.
    if (interfaces.count(bundle.defName))
      continue;

    // Create the interface.  This will be populated later.
    interfaces[bundle.defName] =
        builder.create<sv::InterfaceOp>(circuitOp->getLoc(), bundle.defName);
  }

  circuitOp->setAttr("annotations", builder.getArrayAttr(unprocessedAnnos));

  // Walk through the circuit to collect additional information.  If this fails,
  // signal pass failure.
  for (auto &op : circuitOp.getBody()->getOperations()) {
    if (isa<FModuleOp, FExtModuleOp>(op)) {
      GrandCentralVisitor visitor(&getContext(), interfaceMap);
      visitor.visitModule(&op);
      if (visitor.hasFailed())
        return signalPassFailure();
    }
  }

  // Populate interfaces.
  for (auto &a : interfaceKeys) {
    auto defName = a.first;
    auto name = a.second;

    auto &info = interfaceMap[{defName, name}];
    if (info.isMissing()) {
      emitError(circuitOp.getLoc())
          << "'firrtl.circuit' op contained a Grand Central Interface '"
          << defName << "' that had an element '" << name
          << "' which did not have a scattered companion annotation (is there "
             "an invalid target in your annotation file?)";
      continue;
    }

    builder.setInsertionPointToEnd(interfaces[defName].getBodyBlock());

    auto loc = builder.getFusedLoc(info.locations);
    auto description = info.description;
    if (!description.empty())
      builder.create<sv::VerbatimOp>(loc, ("\n// " + description).str());

    switch (info.tpe) {
    case ElementInfo::Bundle:
      // TODO: Change this to actually use an interface type.  This currently
      // does not work because: (1) interfaces don't have a defined way to get
      // their bit width and (2) interfaces have a symbol table that is used to
      // verify internal ops, but this requires looking arbitrarily far upwards
      // to find other symbols.
      builder.create<sv::VerbatimOp>(loc, (name + " " + name + "();").str());
      break;
    case ElementInfo::Vector: {
      auto type = hw::UnpackedArrayType::get(builder.getIntegerType(info.width),
                                             info.depth);
      builder.create<sv::InterfaceSignalOp>(loc, name, type);
      break;
    }
    case ElementInfo::Ground: {
      auto type = builder.getIntegerType(info.width);
      builder.create<sv::InterfaceSignalOp>(loc, name, type);
      break;
    }
    case ElementInfo::String:
      builder.create<sv::VerbatimOp>(
          loc, ("// " + name + " = <unsupported string type>;").str());
      break;
    case ElementInfo::Boolean:
      builder.create<sv::VerbatimOp>(
          loc, ("// " + name + " = <unsupported boolean type>;").str());
      break;
    case ElementInfo::Integer:
      builder.create<sv::VerbatimOp>(
          loc, ("// " + name + " = <unsupported integer type>;").str());
      break;
    case ElementInfo::Double:
      builder.create<sv::VerbatimOp>(
          loc, ("// " + name + " = <unsupported double type>;").str());
      break;
    case ElementInfo::Error:
      llvm_unreachable("Shouldn't be here");
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralPass() {
  return std::make_unique<GrandCentralPass>();
}
