"""Pointwise operators shared by rate trees and saved outputs.

:class:`~summer4.flows.rates.UnaryOp` and :class:`~summer4.flows.rates.BinOp`
are the symbolic form. Evaluating them, and applying the same operation to an
already-evaluated value (a JAX array, a :class:`~summer4.flows.compiled.GroupedRate`,
or the :class:`~summer4.jax.propertydata.PropertyData` inside a
:class:`~summer4.results.output.Output`), both go through :func:`apply_unary` and
:func:`apply_binary`.

Wrapper policy stays with the wrapper. A ``GroupedRate`` combines only with the
same grouping; a ``PropertyData`` only with an equal map. Neither broadcasts
across names. Name-aligned ``Output``-to-``Output`` broadcasting is a later step;
what this module provides is the numeric kernel that step will call after the
indexes are aligned, and the kernel a parameter-only rate expression uses when
:func:`~summer4.flows.compiled.eval_closed` turns it into an array an ``Output``
can scale by.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import jax.numpy as jnp

# NumPy ufunc names that are not summer4's existing op strings. ``_rate_bytes``
# encodes the op string, so ``np.multiply`` and ``*`` must share one spelling
# or two identical computations compile twice.
UFUNC_ALIASES: dict[str, str] = {
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "true_divide": "div",
    "power": "pow",
    "float_power": "pow",
    "negative": "neg",
    "absolute": "abs",
    "fabs": "abs",
}

# summer4 spellings that are not ``jax.numpy`` attribute names.
_JNP_NAMES: dict[str, str] = {
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "pow": "power",
    "neg": "negative",
    "abs": "abs",
}

# Not elementwise rates. Masks belong to ``Selector``; ``matmul`` is
# ``GroupedRate.__matmul__``.
DENY_OPS: frozenset[str] = frozenset(
    {
        "matmul",
        "greater",
        "less",
        "equal",
        "not_equal",
        "greater_equal",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_not",
        "isnan",
        "isinf",
        "isfinite",
        "signbit",
    }
)

UNARY_OPS: dict[str, Callable[[Any], Any]] = {
    "neg": jnp.negative,
    "exp": jnp.exp,
    "log": jnp.log,
    "abs": jnp.abs,
    "tanh": jnp.tanh,
    "sqrt": jnp.sqrt,
    "floor": jnp.floor,
}

BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "add": jnp.add,
    "sub": jnp.subtract,
    "mul": jnp.multiply,
    "div": jnp.divide,
    "pow": jnp.power,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
}

# Ops that are neither in the two seed dicts (``sin``, ``cos``, …).
_OP_IMPL: dict[str, Callable[..., Any]] = {}

DispatchMode = Literal["symbolic", "output", "value"]


def canonical_op(name: str) -> str:
    """Map a NumPy ufunc name onto summer4's existing op string, if one exists."""
    return UFUNC_ALIASES.get(name, name)


def resolve_op(name: str) -> Callable[..., Any]:
    """Return the ``jax.numpy`` implementation of a canonical op name.

    Raises:
        ValueError: ``name`` is in :data:`DENY_OPS`, or ``jax.numpy`` has no
            such function. The message names the offending op.
    """
    canonical = canonical_op(name)
    if name in DENY_OPS or canonical in DENY_OPS:
        raise ValueError(
            f"Op {name!r} cannot be a rate. Boolean and integer masks belong to "
            "Selector, not rate arithmetic, and 'matmul' is reserved for "
            f"GroupedRate's @ operator. Refused: {', '.join(sorted(DENY_OPS))}."
        )
    cached = UNARY_OPS.get(canonical) or BINARY_OPS.get(canonical) or _OP_IMPL.get(canonical)
    if cached is not None:
        return cached
    attr = _JNP_NAMES.get(canonical, canonical)
    fn = getattr(jnp, attr, None)
    if not callable(fn):
        raise ValueError(
            f"Op {name!r} has no jax.numpy equivalent. "
            "Use a NumPy ufunc name (for example 'sin'), or a summer4 spelling "
            "such as 'mul', 'sub', 'div', 'pow', 'neg' or 'abs'."
        )
    impl = cast(Callable[..., Any], fn)
    _OP_IMPL[canonical] = impl
    return impl


def apply_unary(op: str, value: Any) -> Any:
    """Apply ``op`` pointwise, preserving a ``GroupedRate`` or ``PropertyData``."""
    fn = UNARY_OPS.get(op)
    if fn is None:
        fn = resolve_op(op)
        UNARY_OPS[canonical_op(op)] = fn
    return _map_unary(fn, value)


def apply_binary(op: str, left: Any, right: Any) -> Any:
    """Apply ``op`` pointwise, preserving a ``GroupedRate`` or ``PropertyData``."""
    fn = BINARY_OPS.get(op)
    if fn is None:
        fn = resolve_op(op)
        BINARY_OPS[canonical_op(op)] = fn
    return _map_binary(fn, left, right)


def operands_are_mixed(inputs: tuple[Any, ...]) -> bool:
    """True when a rate-tree node is combined with an evaluated wrapper."""
    from summer4.flows.compiled import GroupedRate
    from summer4.flows.rates import RateOps
    from summer4.jax.propertydata import PropertyData
    from summer4.results.output import Output

    symbolic = any(isinstance(value, RateOps) for value in inputs)
    evaluated = any(isinstance(value, (Output, GroupedRate, PropertyData)) for value in inputs)
    return symbolic and evaluated


def _apply_mode(mode: DispatchMode, canonical: str, inputs: tuple[Any, ...]) -> Any:
    if operands_are_mixed(inputs):
        from summer4.flows.rates import _MIXED_EXPR

        raise TypeError(_MIXED_EXPR)
    if mode == "symbolic":
        from summer4.flows.rates import BinOp, UnaryOp, as_rate

        if len(inputs) == 1:
            return UnaryOp(canonical, as_rate(inputs[0]))
        return BinOp(canonical, as_rate(inputs[0]), as_rate(inputs[1]))
    if mode == "output":
        from summer4.results.output import Output

        if len(inputs) == 1:
            owner = inputs[0]
            if not isinstance(owner, Output):
                raise TypeError("Output dispatch expected an Output.")
            return owner._map_unary(canonical)
        return Output._combine(inputs[0], inputs[1], canonical)
    if len(inputs) == 1:
        return apply_unary(canonical, inputs[0])
    return apply_binary(canonical, inputs[0], inputs[1])


def dispatch_ufunc(
    ufunc: Any,
    method: str,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    mode: DispatchMode,
) -> Any:
    """Shared ``__array_ufunc__`` body.

    ``mode`` is ``symbolic`` for a rate tree (build a node), ``output`` for an
    :class:`~summer4.results.output.Output`, and ``value`` for a
    ``GroupedRate`` or ``PropertyData``.
    """
    if method != "__call__" or kwargs or len(inputs) not in (1, 2):
        return NotImplemented
    raw = getattr(ufunc, "__name__", None)
    if not isinstance(raw, str):
        return NotImplemented
    # Validate before building so a bad op fails at construction, not at trace.
    canonical = canonical_op(raw)
    resolve_op(raw)
    return _apply_mode(mode, canonical, inputs)


def dispatch_array_function(
    func: Any,
    types: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    mode: DispatchMode,
) -> Any:
    """Shared ``__array_function__`` body. Five functions; everything else refuses.

    An open implementation would claim reductions and reshapes that have no
    meaning on a rate. ``np.maximum`` and the other three ufuncs normally
    arrive via :func:`dispatch_ufunc`; they are listed here so a direct
    ``__array_function__`` call still canonicalises.
    """
    del types
    import numpy as np

    if func is np.clip:
        if kwargs.get("out") is not None:
            return NotImplemented
        from summer4.flows.rates import clip

        expr = args[0] if args else kwargs["a"]
        lo = args[1] if len(args) >= 2 else kwargs.get("a_min")
        hi = args[2] if len(args) >= 3 else kwargs.get("a_max")
        return clip(expr, lo, hi)
    names = {
        np.maximum: "maximum",
        np.minimum: "minimum",
        np.power: "power",
        np.absolute: "absolute",
    }
    raw = names.get(func)
    if raw is None or kwargs:
        return NotImplemented
    if len(args) not in (1, 2):
        return NotImplemented
    canonical = canonical_op(raw)
    resolve_op(raw)
    return _apply_mode(mode, canonical, args)


def _map_unary(fn: Callable[[Any], Any], value: Any) -> Any:
    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData

    if isinstance(value, GroupedRate):
        return GroupedRate(data=fn(value.data), properties=value.properties)
    if isinstance(value, PropertyData):
        return PropertyData(value.pmap, fn(value.data))
    return fn(value)


def _map_binary(fn: Callable[[Any, Any], Any], left: Any, right: Any) -> Any:
    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData, _require_same_map

    left_grouped = isinstance(left, GroupedRate)
    right_grouped = isinstance(right, GroupedRate)
    left_pdata = isinstance(left, PropertyData)
    right_pdata = isinstance(right, PropertyData)
    if (left_grouped or right_grouped) and (left_pdata or right_pdata):
        raise TypeError("Cannot combine GroupedRate with PropertyData.")
    if left_grouped and right_grouped:
        left._require_same_grouping(right)
        return GroupedRate(data=fn(left.data, right.data), properties=left.properties)
    if left_grouped:
        return GroupedRate(data=fn(left.data, right), properties=left.properties)
    if right_grouped:
        return GroupedRate(data=fn(left, right.data), properties=right.properties)
    if left_pdata and right_pdata:
        _require_same_map(left.pmap, right.pmap)
        return PropertyData(left.pmap, fn(left.data, right.data))
    if left_pdata:
        return PropertyData(left.pmap, fn(left.data, right))
    if right_pdata:
        return PropertyData(right.pmap, fn(left, right.data))
    return fn(left, right)
