"""Run-start / per-step staging for parameter transforms and rate hoisting."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jax.tree_util import register_pytree_node_class

from summer4.flows.actualize import FlowEdges
from summer4.flows.rates import (
    ArrayConst,
    BinOp,
    Capture,
    Const,
    FieldRef,
    FlowRef,
    GaussianPulse,
    Interp,
    Lookup,
    Multiply,
    Overwrite,
    RateOps,
    Reduce,
    TableInterp,
    Time,
    Transform,
    UnaryOp,
)

PrepareFn = Callable[[Any], Any]
Stage = Literal["run", "step"]


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Prepared:
    """Params (and optional hoisted rate values) fixed for one ``run`` call."""

    params: Any
    hoisted: tuple[Any, ...] = ()

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (self.params, self.hoisted), None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Any, ...]) -> Prepared:
        del aux
        params, hoisted = children
        return cls(params=params, hoisted=hoisted)


@dataclass(frozen=True, slots=True, eq=False)
class HoistEntry:
    """One slotted run-stage rate node (or Interp knot stack part)."""

    node: RateOps
    part: Literal["value", "breakpoints", "values"]


@dataclass(frozen=True, slots=True, eq=False)
class HoistTable:
    """Compile-time map from ``(id(node), part)`` to a slot index in ``Prepared.hoisted``."""

    entries: tuple[HoistEntry, ...]
    slot: Mapping[tuple[int, str], int]


def rate_stage(expr: RateOps, *, params_are_static: bool) -> Stage:
    """Classify a rate expression as run-stage or step-stage."""
    if isinstance(expr, (Const, ArrayConst)):
        return "run"
    if isinstance(expr, FieldRef):
        return "run" if params_are_static else "step"
    if isinstance(expr, (Time, FlowRef, Reduce, Capture)):
        return "step"
    if isinstance(expr, BinOp):
        left = rate_stage(expr.left, params_are_static=params_are_static)
        right = rate_stage(expr.right, params_are_static=params_are_static)
        return "step" if left == "step" or right == "step" else "run"
    if isinstance(expr, UnaryOp):
        return rate_stage(expr.arg, params_are_static=params_are_static)
    if isinstance(expr, Interp):
        parts = (
            *expr.breakpoints,
            *expr.values,
            expr.arg,
        )
        if any(rate_stage(p, params_are_static=params_are_static) == "step" for p in parts):
            return "step"
        return "run"
    if isinstance(expr, GaussianPulse):
        parts = (expr.arg, expr.centre, expr.width, expr.height)
        if any(rate_stage(p, params_are_static=params_are_static) == "step" for p in parts):
            return "step"
        return "run"
    if isinstance(expr, TableInterp):
        # Knot arrays are constants. The stage is whatever the argument reads.
        return rate_stage(expr.arg, params_are_static=params_are_static)
    if isinstance(expr, Lookup):
        table = rate_stage(expr.table, params_are_static=params_are_static)
        index = rate_stage(expr.index, params_are_static=params_are_static)
        return "step" if table == "step" or index == "step" else "run"
    stage_fn = getattr(expr, "__rate_stage__", None)
    if callable(stage_fn):
        result = stage_fn()
        if result not in ("run", "step"):
            raise TypeError(f"__rate_stage__ must return 'run' or 'step', got {result!r}.")
        return result  # type: ignore[no-any-return]
    return "step"


def _is_leaf(expr: RateOps) -> bool:
    return isinstance(expr, (Const, ArrayConst, FieldRef))


def build_hoist_table(roots: Sequence[RateOps], *, params_are_static: bool) -> HoistTable:
    """Walk rate roots and slot run-stage non-leaf subtrees (and Interp knot stacks)."""
    entries: list[HoistEntry] = []
    slot: dict[tuple[int, str], int] = {}

    def add(node: RateOps, part: Literal["value", "breakpoints", "values"]) -> None:
        key = (id(node), part)
        if key in slot:
            return
        slot[key] = len(entries)
        entries.append(HoistEntry(node=node, part=part))

    def walk(node: RateOps) -> None:
        # A table evaluates to a GroupedRate. Hoisting that object would put a
        # non-array in ``Prepared.hoisted``. Walk the argument instead so a
        # parameter-only piece inside it still hoists.
        if isinstance(node, TableInterp):
            walk(node.arg)
            return
        stage = rate_stage(node, params_are_static=params_are_static)
        if stage == "run" and not _is_leaf(node):
            add(node, "value")
            return
        if isinstance(node, Interp):
            if all(
                rate_stage(bp, params_are_static=params_are_static) == "run"
                for bp in node.breakpoints
            ):
                add(node, "breakpoints")
            if all(
                rate_stage(v, params_are_static=params_are_static) == "run" for v in node.values
            ):
                add(node, "values")
            walk(node.arg)
            bp_slotted = (id(node), "breakpoints") in slot
            val_slotted = (id(node), "values") in slot
            if not bp_slotted:
                for bp in node.breakpoints:
                    walk(bp)
            if not val_slotted:
                for v in node.values:
                    walk(v)
            return
        if isinstance(node, BinOp):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, GaussianPulse):
            walk(node.arg)
            walk(node.centre)
            walk(node.width)
            walk(node.height)
            return
        if isinstance(node, UnaryOp):
            walk(node.arg)
            return
        if isinstance(node, Lookup):
            walk(node.index)
            return
        # Capture and unknown custom nodes: do not descend.

    for root in roots:
        walk(root)
    return HoistTable(entries=tuple(entries), slot=slot)


def roots_of(flows: Mapping[str, FlowEdges], order: tuple[str, ...]) -> list[RateOps]:
    """Collect rate roots from every flow's rate and adjustments, in compile order."""
    roots: list[RateOps] = []
    for name in order:
        flow = flows[name]
        roots.append(flow.rate)
        for adj in flow.adjust:
            if isinstance(adj, (Multiply, Overwrite)):
                roots.append(adj.value)
            elif isinstance(adj, Transform):
                roots.extend(adj.args)
    return roots
