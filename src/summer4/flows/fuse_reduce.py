"""Compile-time fusion of weighted sums of scalar ``Reduce`` terms.

A rate such as ``sum(w_k * Reduce(where=sel_k) for k in range(K))`` lowers term
by term: one gather and one sum per ``Reduce``. Mathematically it is one dot
product, ``y[idx] @ w``, where ``idx`` is the union of the compartments any
``sel_k`` selects and ``w[i]`` is the sum of the ``w_k`` whose ``sel_k`` matches
compartment ``idx[i]``. Under reverse-mode differentiation through a fixed-step
``lax.scan`` every gather is transposed into a scatter-add at every step, so the
term-by-term form costs time in proportion to K for no modelling reason.

:func:`fuse_reductions` rewrites the largest subtree that is *linear in scalar
``Reduce`` terms* into one :class:`WeightedReduce`. A subtree is linear when it is
built from:

- ``Reduce(where=...)`` with ``group_by=None``;
- ``+``, ``-`` and unary ``-`` of linear subtrees (other addends are kept as
  they are and added back);
- ``c * linear``, ``linear * c`` and ``linear / c``, where ``c`` is a
  *coefficient*: ``Const`` and ``Param``/``FieldRef`` leaves combined with
  arithmetic and unary operations (parameters only, no state, no time).

It needs at least two ``Reduce`` terms. Anything else is left in place and its
``BinOp``/``UnaryOp``/``Capture``/``Item`` children are rewritten in turn.
Grouped ``Reduce(group_by=...)``, custom nodes (``ForceOfInfection``, ``Defer``)
and their arguments are not touched.

The weight vector does not depend on the state. With constant coefficients it
is folded on the host into an :class:`~summer4.flows.rates.ArrayConst`. With
parameter coefficients it is a :class:`ReduceWeights` node, which is run-stage,
so :func:`~summer4.flows.stages.build_hoist_table` slots it and
:meth:`CompiledModel.prepare` evaluates it once per run.

**Semantics.** Masks come from ``pmap.select``, as for ``Reduce``, so a selector
that matches nothing contributes 0, and a property absent from a compartment
does not match it. Only the union of the selected compartments is gathered, so
an unrelated compartment (even a non-finite one) never enters the sum. A
selector that ``pmap`` rejects leaves its subtree unfused, so the error is
raised where it was before (at the first evaluation).

**Numerics.** The fused form sums ``y[i] * w[i]`` over compartments instead of
``c_k * sum(y[sel_k])`` over terms, and folds the constants of each coefficient
into one number. Results agree to floating-point rounding (a few ulp), not
bit for bit. ``FlowModel.compile(fuse_reductions=False)`` keeps the
term-by-term form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from summer4.flows.rates import (
    ArrayConst,
    BinOp,
    Capture,
    Const,
    FieldRef,
    Item,
    Multiply,
    Overwrite,
    RateOps,
    Reduce,
    UnaryOp,
    _rate_bytes,
    _selector_bytes,
    register_rate_eval,
)

if TYPE_CHECKING:
    from summer4.flows.actualize import FlowEdges
    from summer4.propertymap import PropertyMap
    from summer4.selectors import Selector

#: Fewest ``Reduce`` terms worth fusing. One term is already one gather and one sum.
MIN_TERMS = 2


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class ReduceWeights(RateOps):
    """Per-compartment weights over a fused ``Reduce`` union (run stage).

    The value is ``sum_g factors[g] * numbers[g]``: each group ``g`` is one
    symbolic coefficient factor (a parameter-only expression) times a host
    vector of numbers over the union. ``factors[g] is None`` means the factor 1.
    Groups are summed in order.
    """

    factors: tuple[RateOps | None, ...]
    numbers: tuple[NDArray[np.float64], ...]

    def __rate_stage__(self) -> str:
        return "run"

    def __rate_children__(self) -> tuple[RateOps, ...]:
        return tuple(f for f in self.factors if f is not None)

    def __rate_bytes__(self) -> bytes:
        body = b"reduce-weights"
        for factor, nums in zip(self.factors, self.numbers, strict=True):
            body += b"|" + (b"one" if factor is None else _rate_bytes(factor))
            body += b":" + np.ascontiguousarray(nums, dtype=np.float64).tobytes()
        return body


@dataclass(frozen=True, slots=True, eq=False)
class WeightedReduce(RateOps):
    """``y[index] @ weights``: a weighted scalar sum over a fixed set of compartments.

    ``index`` is ``None`` for every compartment (no gather), a ``slice`` for a
    contiguous run, or an int32 array. ``weights`` evaluates to a vector over
    the selected compartments (or an array whose last axis is that vector).
    """

    weights: RateOps
    index: slice | NDArray[np.int32] | None

    def __rate_stage__(self) -> str:
        return "step"

    def __rate_children__(self) -> tuple[RateOps, ...]:
        return (self.weights,)

    def __rate_bytes__(self) -> bytes:
        idx = self.index
        if idx is None:
            where = b"all"
        elif isinstance(idx, slice):
            where = b"slice" + repr((idx.start, idx.stop)).encode()
        else:
            where = b"idx" + np.ascontiguousarray(idx, dtype=np.int32).tobytes()
        return b"weighted-reduce" + where + _rate_bytes(self.weights)


@register_rate_eval(ReduceWeights)
def _eval_reduce_weights(
    expr: ReduceWeights, *, eval_child: Callable[[RateOps], Any], **_: Any
) -> Any:
    import jax.numpy as jnp

    total: Any = None
    for factor, nums in zip(expr.factors, expr.numbers, strict=True):
        part: Any = jnp.asarray(nums)
        if factor is not None:
            part = jnp.asarray(eval_child(factor))[..., None] * part
        total = part if total is None else total + part
    return total


@register_rate_eval(WeightedReduce)
def _eval_weighted_reduce(
    expr: WeightedReduce, *, eval_child: Callable[[RateOps], Any], y_arr: Any, **_: Any
) -> Any:
    import jax.numpy as jnp

    idx = expr.index
    if idx is None:
        y_sel = y_arr
    elif isinstance(idx, slice):
        y_sel = y_arr[..., idx]
    else:
        # Host indices, all in bounds: "clip" skips take's negative-index wrapping.
        y_sel = jnp.take(y_arr, jnp.asarray(idx), axis=-1, mode="clip")
    w = jnp.asarray(eval_child(expr.weights))
    if w.ndim == 1:
        return y_sel @ w
    return jnp.sum(w * y_sel, axis=-1)


# ---------------------------------------------------------------------------
# Linear decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Coeff:
    """``number * factor`` (``factor`` ``None`` means 1)."""

    number: float
    factor: RateOps | None = None

    def scaled(self, number: float, factor: RateOps | None = None, *, div: bool = False) -> _Coeff:
        new_number = self.number / number if div else self.number * number
        if factor is None:
            return _Coeff(new_number, self.factor)
        op = "div" if div else "mul"
        if self.factor is None:
            new_factor = BinOp("div", Const(1.0), factor) if div else factor
        else:
            new_factor = BinOp(op, self.factor, factor)
        return _Coeff(new_number, new_factor)


@dataclass(frozen=True, slots=True)
class _Linear:
    """``sum(c * Reduce(where=sel)) + sum(rest)``."""

    terms: tuple[tuple[Selector | None, _Coeff], ...]
    rest: tuple[RateOps, ...]

    def scaled(self, number: float, factor: RateOps | None, *, div: bool = False) -> _Linear:
        terms = tuple((sel, c.scaled(number, factor, div=div)) for sel, c in self.terms)
        scale: RateOps = Const(number) if factor is None else factor
        if factor is not None and number != 1.0:
            scale = BinOp("mul", Const(number), factor)
        op = "div" if div else "mul"
        rest = tuple(BinOp(op, r, scale) if div else BinOp(op, scale, r) for r in self.rest)
        return _Linear(terms, rest)

    def negated(self) -> _Linear:
        terms = tuple((sel, _Coeff(-c.number, c.factor)) for sel, c in self.terms)
        rest = tuple(UnaryOp("neg", r) for r in self.rest)
        return _Linear(terms, rest)

    def plus(self, other: _Linear) -> _Linear:
        return _Linear(self.terms + other.terms, self.rest + other.rest)


def _as_coeff(node: RateOps) -> tuple[float, RateOps | None] | None:
    """``node`` as ``(number, factor)`` if it is a parameter-only scalar coefficient."""
    if isinstance(node, Const):
        return float(node.value), None
    if _is_coeff_expr(node):
        return 1.0, node
    return None


def _is_coeff_expr(node: RateOps) -> bool:
    if isinstance(node, (Const, FieldRef)):
        return True
    if isinstance(node, BinOp):
        return _is_coeff_expr(node.left) and _is_coeff_expr(node.right)
    if isinstance(node, UnaryOp):
        return _is_coeff_expr(node.arg)
    return False


def _addend(node: RateOps) -> _Linear:
    """A non-``Reduce`` addend; a literal 0 (as ``sum()`` starts from) is dropped."""
    if isinstance(node, Const) and node.value == 0.0:
        return _Linear((), ())
    return _Linear((), (node,))


def _linear(node: RateOps) -> _Linear | None:
    """Decompose ``node`` as linear in scalar ``Reduce`` terms, or ``None``."""
    if isinstance(node, Reduce):
        if node.group_by is not None:
            return None
        return _Linear(((node.where, _Coeff(1.0)),), ())
    if isinstance(node, UnaryOp) and node.op == "neg":
        inner = _linear(node.arg)
        return None if inner is None else inner.negated()
    if not isinstance(node, BinOp):
        return None
    op = node.op
    if op in ("add", "sub"):
        left = _linear(node.left)
        right = _linear(node.right)
        if left is None and right is None:
            return None
        lhs = left if left is not None else _addend(node.left)
        rhs = right if right is not None else _addend(node.right)
        return lhs.plus(rhs.negated() if op == "sub" else rhs)
    if op == "mul":
        coeff = _as_coeff(node.left)
        if coeff is not None:
            inner = _linear(node.right)
            if inner is not None and inner.terms:
                return inner.scaled(*coeff)
        coeff = _as_coeff(node.right)
        if coeff is not None:
            inner = _linear(node.left)
            if inner is not None and inner.terms:
                return inner.scaled(*coeff)
        return None
    if op == "div":
        coeff = _as_coeff(node.right)
        if coeff is not None:
            inner = _linear(node.left)
            if inner is not None and inner.terms:
                return inner.scaled(*coeff, div=True)
        return None
    return None


# ---------------------------------------------------------------------------
# Building the fused node
# ---------------------------------------------------------------------------


def _union_index(
    selected: list[NDArray[np.int32]], size: int
) -> tuple[slice | NDArray[np.int32] | None, NDArray[np.int32]]:
    union = np.unique(np.concatenate(selected)) if selected else np.zeros(0, np.int32)
    union = union.astype(np.int32, copy=False)
    if union.size == size:
        return None, union
    if union.size and union[-1] - union[0] + 1 == union.size:
        return slice(int(union[0]), int(union[-1]) + 1), union
    return union, union


def _build(lin: _Linear, pmap: PropertyMap) -> RateOps | None:
    """The fused node for ``lin``, or ``None`` when a selector cannot be resolved."""
    selected: list[NDArray[np.int32]] = []
    for sel, _coeff in lin.terms:
        try:
            idx = np.arange(pmap.size, dtype=np.int32) if sel is None else pmap.select(sel)
        except Exception:  # noqa: BLE001 - keep the original error at its original time
            return None
        selected.append(idx)
    index, union = _union_index(selected, pmap.size)
    position = {int(c): i for i, c in enumerate(union)}

    # Group terms by their symbolic factor (by node identity, then digest).
    groups: dict[bytes, tuple[RateOps | None, NDArray[np.float64]]] = {}
    for (_sel, coeff), idx in zip(lin.terms, selected, strict=True):
        key = b"one" if coeff.factor is None else _rate_bytes(coeff.factor)
        if key not in groups:
            groups[key] = (coeff.factor, np.zeros(union.size, dtype=np.float64))
        nums = groups[key][1]
        for c in idx:
            nums[position[int(c)]] += coeff.number
    factors = tuple(f for f, _ in groups.values())
    numbers = tuple(n for _, n in groups.values())
    weights: RateOps
    if len(factors) == 1 and factors[0] is None:
        weights = ArrayConst(numbers[0])
    else:
        weights = ReduceWeights(factors, numbers)
    return WeightedReduce(weights, index)


# ---------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------


class _Rewriter:
    """Rewrites rate trees for one model. Shared subtrees map to one shared result."""

    def __init__(self, pmap: PropertyMap) -> None:
        self.pmap = pmap
        self.cache: dict[int, tuple[RateOps, RateOps]] = {}

    def __call__(self, node: RateOps) -> RateOps:
        hit = self.cache.get(id(node))
        if hit is not None and hit[0] is node:
            return hit[1]
        out = self._rewrite(node)
        self.cache[id(node)] = (node, out)
        return out

    def _rewrite(self, node: RateOps) -> RateOps:
        lin = _linear(node)
        if lin is not None and len(lin.terms) >= MIN_TERMS:
            fused = _build(lin, self.pmap)
            if fused is not None:
                out = fused
                for r in lin.rest:
                    out = BinOp("add", out, self(r))
                return out
        return self._rewrite_children(node)

    def _rewrite_children(self, node: RateOps) -> RateOps:
        if isinstance(node, BinOp):
            left, right = self(node.left), self(node.right)
            if left is node.left and right is node.right:
                return node
            return BinOp(node.op, left, right)
        if isinstance(node, UnaryOp):
            arg = self(node.arg)
            return node if arg is node.arg else UnaryOp(node.op, arg)
        if isinstance(node, Capture):
            inner = self(node.inner)
            return node if inner is node.inner else Capture(node.name, inner)
        if isinstance(node, Item):
            arg = self(node.arg)
            return node if arg is node.arg else replace(node, arg=arg)
        return node


def fuse_reductions(expr: RateOps, pmap: PropertyMap) -> RateOps:
    """Rewrite one rate tree; see the module docstring. Returns ``expr`` when nothing fuses."""
    return _Rewriter(pmap)(expr)


def fuse_flows(flows: Mapping[str, FlowEdges], pmap: PropertyMap) -> dict[str, FlowEdges]:
    """Fuse every flow's rate and ``Multiply``/``Overwrite`` values, sharing subtrees.

    Flows with nothing to fuse are returned as the same objects. ``Transform``
    arguments and split proportions are left as they are.
    """
    rewrite = _Rewriter(pmap)
    out: dict[str, FlowEdges] = {}
    for name, flow in flows.items():
        rate = rewrite(flow.rate)
        adjust = tuple(_fuse_adjustment(adj, rewrite) for adj in flow.adjust)
        changed = rate is not flow.rate or any(
            a is not b for a, b in zip(adjust, flow.adjust, strict=True)
        )
        out[name] = replace(flow, rate=rate, adjust=adjust) if changed else flow
    return out


def _fuse_adjustment(adj: Any, rewrite: _Rewriter) -> Any:
    if isinstance(adj, (Multiply, Overwrite)):
        value = rewrite(adj.value)
        if value is adj.value:
            return adj
        return type(adj)(value, where=adj.where, precedence=adj.precedence)
    return adj


__all__ = ["ReduceWeights", "WeightedReduce", "fuse_flows", "fuse_reductions"]

_ = _selector_bytes  # imported for symmetry with rates; selectors digest via the masks
