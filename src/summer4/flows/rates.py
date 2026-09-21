"""Lazy rate expressions, flow references, and adjustments."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NamedTuple, cast, get_type_hints

import numpy as np

from summer4.enums import coerce_strenum
from summer4.properties import Property
from summer4.selectors import (
    Absent,
    And,
    Dest,
    Everything,
    IsIn,
    Not,
    Nothing,
    Or,
    Present,
    Selector,
    Source,
)

type FlowReduce = Literal["identity", "sum"] | tuple[Literal["sum_over"], str]
type Adjustment = Multiply | Overwrite | Transform
type AdjustSpec = Sequence[object] | None


class InterpKind(StrEnum):
    """Interpolation shapes for :class:`Interp` and :class:`TableInterp`.

    Prefer these members at call sites. Bare strings such as ``"linear"`` are
    still accepted and coerced.
    """

    LINEAR = "linear"
    SIGMOIDAL = "sigmoidal"
    STEP = "step"


type InterpKindArg = InterpKind | str

# Extension point for rate nodes defined outside ``summer4.flows`` (e.g. epi).
_RATE_EVALUATORS: dict[type, Callable[..., Any]] = {}


def register_rate_eval[T](cls: type[T]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register an evaluator for a :class:`RateOps` subclass defined elsewhere.

    ``cls`` **must** define ``__rate_bytes__(self) -> bytes``, returning a stable
    encoding of every field that changes the node's numerics. The digest built
    from it is both :meth:`CompiledModel.__hash__` and :meth:`__eq__`, and the
    model is a static argument to :func:`jax.jit`: two models whose custom nodes
    encode to the same bytes share a compiled program, so a node that omits a
    field silently returns another model's numbers. Registration therefore
    rejects a class without the dunder, at import time rather than at run time.

    Raises:
        TypeError: if ``cls`` does not define a callable ``__rate_bytes__``.
    """
    if not callable(getattr(cls, "__rate_bytes__", None)):
        raise TypeError(
            f"{cls.__name__} cannot be registered as a rate node: it must define "
            "__rate_bytes__(self) -> bytes, a stable encoding of every field that "
            "affects the node's value. Without it two models differing only in "
            f"{cls.__name__}'s fields would compare equal and share a jax.jit "
            "cache entry."
        )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _RATE_EVALUATORS[cls] = fn
        return fn

    return decorator


class RateOps:
    """Arithmetic mixin for rate expression nodes."""

    def __add__(self, other: object) -> BinOp:
        return BinOp("add", as_rate(self), as_rate(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("add", as_rate(other), as_rate(self))

    def __sub__(self, other: object) -> BinOp:
        return BinOp("sub", as_rate(self), as_rate(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("sub", as_rate(other), as_rate(self))

    def __mul__(self, other: object) -> BinOp:
        return BinOp("mul", as_rate(self), as_rate(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("mul", as_rate(other), as_rate(self))

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("div", as_rate(self), as_rate(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("div", as_rate(other), as_rate(self))

    def __pow__(self, other: object) -> BinOp:
        return BinOp("pow", as_rate(self), as_rate(other))

    def __rpow__(self, other: object) -> BinOp:
        return BinOp("pow", as_rate(other), as_rate(self))

    def __neg__(self) -> UnaryOp:
        return UnaryOp("neg", as_rate(self))

    def __abs__(self) -> UnaryOp:
        return UnaryOp("abs", as_rate(self))

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from summer4.flows.algebra import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs, mode="symbolic")

    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from summer4.flows.algebra import dispatch_array_function

        return dispatch_array_function(func, types, args, kwargs, mode="symbolic")


@dataclass(frozen=True, slots=True)
class Const(RateOps):
    """Literal scalar rate."""

    value: float


@dataclass(frozen=True, slots=True)
class Time(RateOps):
    """The current model time, usable anywhere a rate expression is.

    Field-less on purpose: every instance compares and hashes equal, so two
    models built from separately-constructed ``Time()`` nodes share a jit
    cache entry. Offset with arithmetic (``Time() - t0``), not a field.
    """


@dataclass(frozen=True, slots=True)
class Interp(RateOps):
    """Interpolation between knots, evaluated at a rate expression.

    ``breakpoints`` and ``values`` are rate expressions (length fixed at
    construct time) so knot times and heights may calibrate via
    :class:`FieldRef`. Outside the breakpoint range,
    :func:`jax.numpy.interp` (and the step / sigmoidal evaluators) clamp to
    the end values — they do not extrapolate. Evaluated breakpoints must
    remain strictly increasing; they are not sorted at runtime.
    """

    kind: InterpKindArg
    breakpoints: tuple[RateOps, ...]
    values: tuple[RateOps, ...]
    arg: RateOps
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            coerce_strenum(InterpKind, self.kind, what="Interp kind"),
        )


@dataclass(frozen=True, slots=True)
class GaussianPulse(RateOps):
    """``height * exp(-0.5 * ((arg - centre) / width)**2)``."""

    arg: RateOps
    centre: RateOps
    width: RateOps
    height: RateOps


@dataclass(frozen=True, slots=True)
class FieldRef(RateOps):
    """Lazy path into the runtime derived-param struct."""

    path: tuple[str, ...]

    def __getattr__(self, name: str) -> FieldRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldRef((*self.path, name))


def Param(name: str) -> FieldRef:
    """Named parameter: thin alias for ``FieldRef((name,))``.

    Resolves against a plain dict or NamedTuple params the same way a
    one-element :class:`FieldRef` path already does.
    """
    return FieldRef((name,))


@dataclass(frozen=True, slots=True)
class FlowRef(RateOps):
    """Reference to another flow's already-computed contribution."""

    name: str
    reduce: FlowReduce = "identity"

    def sum(self) -> FlowRef:
        """Return this flow's contribution reduced to a scalar."""
        return FlowRef(self.name, reduce="sum")

    def sum_over(self, prop: Property | str) -> FlowRef:
        """Reduce this flow's contribution onto one property's traits."""
        name = prop.name if isinstance(prop, Property) else prop
        return FlowRef(self.name, reduce=("sum_over", name))


@dataclass(frozen=True, slots=True)
class UnaryOp(RateOps):
    """Pointwise unary operation on one rate expression."""

    op: str
    arg: RateOps

    def __post_init__(self) -> None:
        from summer4.flows.algebra import canonical_op, resolve_op

        canonical = canonical_op(self.op)
        resolve_op(canonical)
        if canonical != self.op:
            object.__setattr__(self, "op", canonical)


@dataclass(frozen=True, slots=True)
class BinOp(RateOps):
    """Binary arithmetic on two rate expressions."""

    op: str
    left: RateOps
    right: RateOps

    def __post_init__(self) -> None:
        from summer4.flows.algebra import canonical_op, resolve_op

        canonical = canonical_op(self.op)
        resolve_op(canonical)
        if canonical != self.op:
            object.__setattr__(self, "op", canonical)


@dataclass(frozen=True, slots=True)
class Reduce(RateOps):
    """Reduce compartment state over a grouping; ``where`` means KEEP.

    ``Reduce(where=sel, sum_over=prop)`` sums only compartments matching
    ``sel``. That is the opposite polarity of :meth:`PropertyData.where`,
    which *replaces* matches — use :meth:`PropertyData.keep` for the same
    keep-shaped reading on a :class:`PropertyData`.
    """

    sum_over: Property | str
    where: Selector | None = None


@dataclass(frozen=True, slots=True)
class Capture(RateOps):
    """Evaluate ``inner`` (typically a :class:`GroupedRate`) and stash it by name.

    Saved via :class:`~summer4.results.plan.GroupedOutput` so a force of
    infection (or any other grouped quantity) is inspectable as a
    properly-dimensioned trace without re-slicing a broadcast array.
    """

    name: str
    inner: RateOps


@dataclass(frozen=True, slots=True)
class ArrayConst(RateOps):
    """Literal array rate (e.g. a static mixing matrix)."""

    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", np.asarray(self.value, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class TableInterp(RateOps):
    """Interpolate every column of a ``(T, K)`` table at one argument.

    ``times`` and ``values`` are literal arrays (:class:`ArrayConst`), not one
    node per knot, so the traced program does not grow with ``T`` or ``K``.
    The result is a :class:`~summer4.flows.compiled.GroupedRate` over ``over``,
    in trait order. Outside the time range the value clamps to the end column
    — it does not extrapolate.

    For ``kind=InterpKind.STEP``, ``values`` has one more row than ``times``: the
    leading row is the value before the first time, matching
    :func:`~summer4.timevarying.step`. :meth:`summer4.data.TableData.interp`
    prepends that row. ``linear`` and ``sigmoidal`` have one row per time.
    """

    kind: InterpKindArg
    times: ArrayConst
    values: ArrayConst
    over: Property
    arg: RateOps = Time()
    sharpness: float = 1.0

    def __post_init__(self) -> None:
        resolved = coerce_strenum(InterpKind, self.kind, what="TableInterp kind")
        object.__setattr__(self, "kind", resolved)
        times = np.asarray(self.times.value, dtype=np.float64).reshape(-1)
        values = np.asarray(self.values.value, dtype=np.float64)
        if times.size < 1:
            raise ValueError("TableInterp requires at least one time.")
        if values.ndim != 2:
            raise ValueError(
                f"TableInterp values must have shape (n_times, n_traits), got {values.shape}."
            )
        n_traits = len(self.over.traits)
        if values.shape[1] != n_traits:
            raise ValueError(
                f"TableInterp has {values.shape[1]} columns but property {self.over.name!r} "
                f"has traits {self.over.traits}."
            )
        if resolved is InterpKind.STEP:
            if values.shape[0] != times.size + 1:
                raise ValueError(
                    "step TableInterp requires len(values) == len(times) + 1; "
                    f"got {values.shape[0]} vs {times.size} + 1."
                )
        else:
            if times.size < 2:
                raise ValueError(f"{resolved.value} TableInterp requires at least two times.")
            if values.shape[0] != times.size:
                raise ValueError(
                    f"{resolved.value} TableInterp requires one value row per time; "
                    f"got {values.shape[0]} rows and {times.size} times."
                )
        if times.size >= 2 and bool(np.any(np.diff(times) <= 0)):
            raise ValueError("TableInterp times must be strictly increasing.")
        object.__setattr__(self, "sharpness", float(self.sharpness))


@dataclass(frozen=True, slots=True)
class Lookup(RateOps):
    """Gather one row of an array carried in the parameters.

    ``index`` is typically ``floor(Time() - year0)``. With ``clamp=True``
    (the default) the index is clipped into ``[0, n_rows - 1]`` before
    :func:`jax.numpy.take`, so a time outside the table holds the end row.
    A ``(n_years, K, K)`` table therefore yields a ``(K, K)`` matrix, which
    is what :class:`~summer4.epi.mixing.MixingMatrix` evaluates.
    """

    table: FieldRef
    index: RateOps
    clamp: bool = True

    def __init__(self, table: FieldRef, index: object, *, clamp: bool = True) -> None:
        if not isinstance(table, FieldRef):
            raise TypeError(
                "Lookup table must be a FieldRef (Param(...) or FieldRef(...)), "
                f"got {type(table).__name__}."
            )
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "index", as_rate(index))
        object.__setattr__(self, "clamp", bool(clamp))


@dataclass(frozen=True, slots=True)
class Multiply:
    """Multiply the previous aligned rate by ``value`` (optional ``where`` mask)."""

    value: RateOps
    where: Selector | None = None
    precedence: int | None = None

    def __init__(
        self,
        value: object,
        where: Selector | None = None,
        precedence: int | None = None,
    ) -> None:
        object.__setattr__(self, "value", as_rate(value))
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "precedence", precedence)


@dataclass(frozen=True, slots=True)
class Overwrite:
    """Replace the previous aligned rate with ``value`` (optional ``where`` mask)."""

    value: RateOps
    where: Selector | None = None
    precedence: int | None = None

    def __init__(
        self,
        value: object,
        where: Selector | None = None,
        precedence: int | None = None,
    ) -> None:
        object.__setattr__(self, "value", as_rate(value))
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "precedence", precedence)


@dataclass(frozen=True, slots=True)
class Transform:
    """Call ``fn(prev, *args)`` on the previous aligned rate."""

    fn: Callable[..., Any]
    args: tuple[RateOps, ...]
    where: Selector | None = None
    precedence: int | None = None

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: object,
        where: Selector | None = None,
        precedence: int | None = None,
    ) -> None:
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", tuple(as_rate(arg) for arg in args))
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "precedence", precedence)


def adjustment_level(adj: Adjustment) -> int:
    """Return the evaluation level for ``adj`` (``precedence`` or kind default)."""
    if adj.precedence is not None:
        return adj.precedence
    if isinstance(adj, Overwrite):
        return 0
    if isinstance(adj, Multiply):
        return 1
    return 2


def canonical_adjustments(adjust: tuple[Adjustment, ...]) -> tuple[Adjustment, ...]:
    """Stable-sort adjustments by ``(level, declaration index)``."""
    order = sorted(range(len(adjust)), key=lambda i: (adjustment_level(adjust[i]), i))
    return tuple(adjust[i] for i in order)


def _is_namedtuple_class(schema: type) -> bool:
    return (
        isinstance(schema, type)
        and issubclass(schema, tuple)
        and isinstance(getattr(schema, "_fields", None), tuple)
    )


def _field_annotations(schema: type) -> dict[str, Any]:
    try:
        return get_type_hints(schema)
    except (NameError, TypeError, AttributeError):
        return dict(getattr(schema, "__annotations__", {}))


def _nested_schema(schema: type, field: str) -> type | None:
    """Return the NamedTuple class annotated on ``field``, if any."""
    raw: object = _field_annotations(schema).get(field)
    if raw is None:
        raw = getattr(schema, "__annotations__", {}).get(field)
    if isinstance(raw, str):
        module = sys.modules.get(getattr(schema, "__module__", ""))
        if module is not None:
            raw = getattr(module, raw, raw)
    if isinstance(raw, type) and _is_namedtuple_class(raw):
        return raw
    return None


def _field_ref_tree(schema: type, prefix: tuple[str, ...]) -> Any:
    values: list[Any] = []
    fields = cast(tuple[str, ...], cast(Any, schema)._fields)
    for name in fields:
        path = (*prefix, name)
        nested = _nested_schema(schema, name)
        if nested is not None:
            values.append(_field_ref_tree(nested, path))
        else:
            values.append(FieldRef(path))
    return schema(*values)


def derived_refs[T: NamedTuple](schema: type[T]) -> T:
    """Build a NamedTuple of :class:`FieldRef`s named after ``schema`` fields.

    Nested NamedTuple field annotations become nested ref trees so a bundle
    like ``D.migration.baseline`` is one path. The return is annotated as
    ``schema`` so IDEs complete only those fields. Runtime values are
    ``FieldRef`` paths (or nested schemas of them), not the declared types.
    """
    if not _is_namedtuple_class(schema):
        raise TypeError(f"derived_refs expects a NamedTuple class, got {schema!r}.")
    return cast(T, _field_ref_tree(schema, ()))


def as_rate(value: object) -> RateOps:
    """Coerce a scalar, array, or rate node into a :class:`RateOps` expression."""
    if isinstance(value, RateOps):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return Const(float(value))
        return ArrayConst(value)
    raise TypeError(f"Cannot use {type(value).__name__} as a flow rate.")


def _is_value(value: object) -> bool:
    """True when ``value`` is already evaluated, not a rate-tree node.

    Python scalars stay symbolic (``tanh(0.5)`` is a node). Arrays, grouped
    rates, property data and traces are applied immediately, which is how a
    parameter array inside a jitted loss meets the same operator a rate tree
    uses.
    """
    if value is None or isinstance(value, RateOps):
        return False
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return False
    import jax

    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData
    from summer4.results.trace import Trace

    return isinstance(value, (Trace, GroupedRate, PropertyData, np.ndarray, jax.Array))


_MIXED_EXPR = (
    "Cannot combine an evaluated value with an unevaluated rate expression. "
    "Evaluate parameter-only expressions with eval_closed(expr, params)."
)


def _apply_value_unary(op: str, arg: object) -> Any:
    from summer4.flows.algebra import apply_unary
    from summer4.results.trace import Trace

    if isinstance(arg, Trace):
        return arg._map_unary(op)
    return apply_unary(op, arg)


def _apply_value_binary(op: str, left: object, right: object) -> Any:
    from summer4.flows.algebra import apply_binary
    from summer4.results.trace import Trace

    if isinstance(left, RateOps) or isinstance(right, RateOps):
        raise TypeError(_MIXED_EXPR)
    if isinstance(left, Trace) or isinstance(right, Trace):
        return Trace._combine(left, right, op)
    return apply_binary(op, left, right)


def _unary(op: str, arg: object) -> Any:
    if _is_value(arg):
        return _apply_value_unary(op, arg)
    return UnaryOp(op, as_rate(arg))


def _binary(op: str, left: object, right: object) -> Any:
    if _is_value(left) or _is_value(right):
        return _apply_value_binary(op, left, right)
    return BinOp(op, as_rate(left), as_rate(right))


def exp(arg: object) -> Any:
    """Exponential. A rate node, or the value if ``arg`` is already evaluated."""
    return _unary("exp", arg)


def log(arg: object) -> Any:
    """Natural log. A rate node, or the value if ``arg`` is already evaluated."""
    return _unary("log", arg)


def tanh(arg: object) -> Any:
    """Hyperbolic tangent. A rate node, or the value if ``arg`` is already evaluated."""
    return _unary("tanh", arg)


def sqrt(arg: object) -> Any:
    """Square root. A rate node, or the value if ``arg`` is already evaluated."""
    return _unary("sqrt", arg)


def floor(arg: object) -> Any:
    """Round toward -inf. A rate node, or the value if ``arg`` is already evaluated.

    The derivative is zero, so ``floor`` does not stop ``jax.grad`` of a larger
    expression.
    """
    return _unary("floor", arg)


def maximum(left: object, right: object) -> Any:
    """Pointwise maximum. A rate node, unless either side is already evaluated."""
    return _binary("maximum", left, right)


def minimum(left: object, right: object) -> Any:
    """Pointwise minimum. A rate node, unless either side is already evaluated."""
    return _binary("minimum", left, right)


def clip(x: object, lo: object | None = None, hi: object | None = None) -> Any:
    """Clamp ``x`` to ``[lo, hi]`` via ``maximum`` / ``minimum``.

    Either bound may be omitted. Both omitted is an error. Bounds and ``x``
    follow the same split as :func:`maximum`: rate expressions stay symbolic,
    evaluated values (including a :class:`~summer4.results.trace.Trace`) are
    applied now. Mixing the two raises; evaluate the expression with
    :func:`~summer4.flows.compiled.eval_closed` first.
    """
    if lo is None and hi is None:
        raise ValueError("clip requires a lower bound, an upper bound, or both.")
    if _is_value(x) or _is_value(lo) or _is_value(hi):
        if isinstance(x, RateOps) or isinstance(lo, RateOps) or isinstance(hi, RateOps):
            raise TypeError(_MIXED_EXPR)
        out = x
        if lo is not None:
            out = _apply_value_binary("maximum", out, lo)
        if hi is not None:
            out = _apply_value_binary("minimum", out, hi)
        return out
    expr = as_rate(x)
    if lo is not None:
        expr = BinOp("maximum", expr, as_rate(lo))
    if hi is not None:
        expr = BinOp("minimum", expr, as_rate(hi))
    return expr


def as_adjust(value: object) -> Adjustment:
    """Coerce a bare rate or explicit adjustment into an :class:`Adjustment`."""
    if isinstance(value, (Multiply, Overwrite, Transform)):
        return value
    return Multiply(as_rate(value))


def _normalize_adjust(adjust: AdjustSpec) -> tuple[Adjustment, ...]:
    if not adjust:
        return ()
    return tuple(as_adjust(item) for item in adjust)


def _flow_refs(expr: RateOps) -> set[str]:
    match expr:
        case FlowRef(name=name):
            return {name}
        case BinOp(left=left, right=right):
            return _flow_refs(left) | _flow_refs(right)
        case UnaryOp(arg=arg):
            return _flow_refs(arg)
        case Interp(breakpoints=breakpoints, values=values, arg=arg):
            refs: set[str] = set()
            for bp in breakpoints:
                refs |= _flow_refs(bp)
            for value in values:
                refs |= _flow_refs(value)
            return refs | _flow_refs(arg)
        case GaussianPulse(arg=arg, centre=centre, width=width, height=height):
            return _flow_refs(arg) | _flow_refs(centre) | _flow_refs(width) | _flow_refs(height)
        case Time() | Const() | FieldRef() | Reduce() | ArrayConst():
            return set()
        case TableInterp(arg=arg):
            return _flow_refs(arg)
        case Lookup(index=index):
            return _flow_refs(index)
        case Capture(inner=inner):
            return _flow_refs(inner)
        case _:
            custom = getattr(expr, "__flow_refs__", None)
            if callable(custom):
                return set(custom())
            return set()


def _adjust_flow_refs(adj: Adjustment) -> set[str]:
    if isinstance(adj, Transform):
        refs: set[str] = set()
        for arg in adj.args:
            refs |= _flow_refs(arg)
        return refs
    return _flow_refs(adj.value)


def _flow_rate_refs(rate: RateOps, adjust: Sequence[Adjustment]) -> set[str]:
    refs = _flow_refs(rate)
    for adj in adjust:
        refs |= _adjust_flow_refs(adj)
    return refs


def _field_paths(expr: RateOps) -> set[tuple[str, ...]]:
    match expr:
        case FieldRef(path=path):
            return {path}
        case BinOp(left=left, right=right):
            return _field_paths(left) | _field_paths(right)
        case UnaryOp(arg=arg):
            return _field_paths(arg)
        case Interp(breakpoints=breakpoints, values=values, arg=arg):
            paths: set[tuple[str, ...]] = set()
            for bp in breakpoints:
                paths |= _field_paths(bp)
            for value in values:
                paths |= _field_paths(value)
            return paths | _field_paths(arg)
        case GaussianPulse(arg=arg, centre=centre, width=width, height=height):
            return (
                _field_paths(arg)
                | _field_paths(centre)
                | _field_paths(width)
                | _field_paths(height)
            )
        case Time() | Const() | FlowRef() | Reduce() | ArrayConst():
            return set()
        case TableInterp(arg=arg):
            return _field_paths(arg)
        case Lookup(table=table, index=index):
            return _field_paths(table) | _field_paths(index)
        case Capture(inner=inner):
            return _field_paths(inner)
        case _:
            custom = getattr(expr, "__field_paths__", None)
            if callable(custom):
                return set(custom())
            return set()


def _adjust_field_paths(adj: Adjustment) -> set[tuple[str, ...]]:
    if isinstance(adj, Transform):
        paths: set[tuple[str, ...]] = set()
        for arg in adj.args:
            paths |= _field_paths(arg)
        return paths
    return _field_paths(adj.value)


def _lookup_path(root: object, path: tuple[str, ...]) -> object:
    current: object = root
    for name in path:
        if isinstance(current, Mapping):
            try:
                current = current[name]
            except KeyError as exc:
                raise KeyError(f"Derived struct has no field {path!r}.") from exc
        else:
            try:
                current = getattr(current, name)
            except AttributeError as exc:
                raise AttributeError(f"Derived struct has no field {path!r}.") from exc
    return current


def _schema_paths(schema: type, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """List every leaf path on a NamedTuple schema (including nested bundles)."""
    if not _is_namedtuple_class(schema):
        return []
    out: list[tuple[str, ...]] = []
    fields = cast(tuple[str, ...], cast(Any, schema)._fields)
    for name in fields:
        path = (*prefix, name)
        nested = _nested_schema(schema, name)
        if nested is not None:
            out.extend(_schema_paths(nested, path))
        else:
            out.append(path)
    return out


def derived_return_schema(derived_fn: Callable[..., Any] | None) -> type | None:
    """Return the NamedTuple annotation on ``derived_fn``'s return, if any.

    Unannotated hooks skip validation. Annotations that cannot be resolved
    (e.g. a class local to the defining function under
    ``from __future__ import annotations``) are treated as absent.
    """
    if derived_fn is None:
        return None
    raw = getattr(derived_fn, "__annotations__", {}).get("return")
    if isinstance(raw, type) and _is_namedtuple_class(raw):
        return raw
    try:
        hints = get_type_hints(derived_fn)
    except (NameError, TypeError, AttributeError):
        hints = {}
    ret = hints.get("return")
    if isinstance(ret, type) and _is_namedtuple_class(ret):
        return ret
    if isinstance(raw, str):
        candidate = derived_fn.__globals__.get(raw)
        if isinstance(candidate, type) and _is_namedtuple_class(candidate):
            return candidate
    return None


def validate_computed_path(path: tuple[str, ...], schema: type) -> None:
    """Raise if ``path`` is not a leaf on ``schema``, listing available paths."""
    available = _schema_paths(schema)
    if path in available:
        return
    pretty = ", ".join(".".join(p) for p in available) or "(none)"
    raise ValueError(
        f"ComputedValue path {'.'.join(path)!r} is not a field of the derived "
        f"schema {schema.__name__}. Available paths: {pretty}."
    )


def _selector_bytes(sel: Selector) -> bytes:
    """Stable encoding of a selector AST for digests / jit cache keys."""
    from summer4.properties import Trait

    match sel:
        case Trait(property=prop, name=name):
            return b"trait" + prop.encode() + b"\0" + name.encode()
        case IsIn(property=prop, names=names):
            return b"isin" + prop.encode() + b"\0" + repr(names).encode()
        case Present(property=prop):
            return b"present" + prop.encode()
        case Absent(property=prop):
            return b"absent" + prop.encode()
        case Everything():
            return b"everything"
        case Nothing():
            return b"nothing"
        case And(left=left, right=right):
            return b"and" + _selector_bytes(left) + _selector_bytes(right)
        case Or(left=left, right=right):
            return b"or" + _selector_bytes(left) + _selector_bytes(right)
        case Not(inner=inner):
            return b"not" + _selector_bytes(inner)
        case Source(inner=inner):
            return b"source" + _selector_bytes(inner)
        case Dest(inner=inner):
            return b"dest" + _selector_bytes(inner)
        case _:
            raise TypeError(f"Unsupported selector {type(sel).__name__}.")


def _rate_bytes(expr: RateOps) -> bytes:
    match expr:
        case Const(value=value):
            return b"const" + np.float64(value).tobytes()
        case Time():
            return b"time"
        case FieldRef(path=path):
            return b"field" + repr(path).encode()
        case FlowRef(name=name, reduce=reduce):
            return b"flow" + name.encode() + b"|" + repr(reduce).encode()
        case BinOp(op=op, left=left, right=right):
            return b"binop:" + op.encode() + _rate_bytes(left) + _rate_bytes(right)
        case UnaryOp(op=op, arg=arg):
            return b"unary:" + op.encode() + _rate_bytes(arg)
        case Interp(
            kind=kind,
            breakpoints=breakpoints,
            values=values,
            arg=arg,
            sharpness=sharpness,
        ):
            return (
                b"interp"
                + str(kind).encode()
                + np.float64(sharpness).tobytes()
                + b"".join(_rate_bytes(bp) for bp in breakpoints)
                + b"".join(_rate_bytes(v) for v in values)
                + _rate_bytes(arg)
            )
        case GaussianPulse(arg=arg, centre=centre, width=width, height=height):
            return (
                b"gpulse"
                + _rate_bytes(arg)
                + _rate_bytes(centre)
                + _rate_bytes(width)
                + _rate_bytes(height)
            )
        case Reduce(sum_over=sum_over, where=where):
            prop_name = sum_over.name if isinstance(sum_over, Property) else sum_over
            body = b"reduce" + prop_name.encode()
            if where is None:
                return body + b"nowhere"
            return body + _selector_bytes(where)
        case Capture(name=name, inner=inner):
            return b"capture" + name.encode() + _rate_bytes(inner)
        case ArrayConst(value=value):
            return b"array" + np.ascontiguousarray(value, dtype=np.float64).tobytes()
        case TableInterp(
            kind=kind,
            times=times,
            values=values,
            over=over,
            arg=arg,
            sharpness=sharpness,
        ):
            return (
                b"tableinterp"
                + str(kind).encode()
                + np.float64(sharpness).tobytes()
                + over.name.encode()
                + repr(over.traits).encode()
                + _rate_bytes(times)
                + _rate_bytes(values)
                + _rate_bytes(arg)
            )
        case Lookup(table=table, index=index, clamp=clamp):
            flag = b"1" if clamp else b"0"
            return b"lookup" + flag + _rate_bytes(table) + _rate_bytes(index)
        case _:
            # Extension nodes (e.g. epi) supply their own stable encoding. There
            # is no safe fallback: digesting by class name alone would make two
            # models differing only in this node's fields compare equal, hash
            # equal, and share a ``jax.jit`` cache entry.
            custom = getattr(expr, "__rate_bytes__", None)
            if callable(custom):
                return bytes(custom())
            raise TypeError(
                f"Rate node {type(expr).__name__} does not define "
                "__rate_bytes__(self) -> bytes, so it cannot be digested for the "
                "jax.jit cache key. Implement it, encoding every field that "
                "affects the node's value."
            )


def _adjust_bytes(adj: Adjustment) -> bytes:
    if isinstance(adj, Transform):
        return b"tf" + str(id(adj.fn)).encode() + b"".join(_rate_bytes(arg) for arg in adj.args)
    kind = b"ow" if isinstance(adj, Overwrite) else b"mul"
    return kind + _rate_bytes(adj.value)
