"""Flows: joins, rates, and a compiled JAX vector field."""

from summer4.flows.actualize import (
    EntryEdges,
    ExitEdges,
    FlowEdges,
    TransitionEdges,
    actualize,
)
from summer4.flows.compiled import CompiledModel, FlowModel, euler, numpy_euler
from summer4.flows.edges import EdgeMap, EdgeRoles
from summer4.flows.join import (
    TraitChain,
    TraitMatrix,
    identity_join,
    selector_properties,
    selector_values,
)
from summer4.flows.rates import (
    BinOp,
    Const,
    FieldRef,
    FlowRef,
    Multiply,
    Overwrite,
    Transform,
    as_adjust,
    as_rate,
    derived_refs,
)
from summer4.flows.types import EntryFlow, ExitFlow, TransitionFlow

__all__ = [
    "BinOp",
    "CompiledModel",
    "Const",
    "EdgeMap",
    "EdgeRoles",
    "EntryEdges",
    "EntryFlow",
    "ExitEdges",
    "ExitFlow",
    "FieldRef",
    "FlowEdges",
    "FlowModel",
    "FlowRef",
    "Multiply",
    "Overwrite",
    "TraitChain",
    "TraitMatrix",
    "Transform",
    "TransitionEdges",
    "TransitionFlow",
    "actualize",
    "as_adjust",
    "as_rate",
    "derived_refs",
    "euler",
    "identity_join",
    "numpy_euler",
    "selector_properties",
    "selector_values",
]
