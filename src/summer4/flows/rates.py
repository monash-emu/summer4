"""Lazy rate expressions, flow references, and adjustments."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast, get_type_hints

import numpy as np

from summer4.properties import Property
from summer4.selectors import Selector

type FlowReduce = Literal["identity", "sum"] | tuple[Literal["sum_over"], str]
type Adjustment = Multiply | Overwrite | Transform
type AdjustSpec = Sequence[object] | None


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


@dataclass(frozen=True, slots=True)
class Const(RateOps):
    """Literal scalar rate."""

    value: float


@dataclass(frozen=True, slots=True)
class FieldRef(RateOps):
    """Lazy path into the runtime derived-param struct."""

    path: tuple[str, ...]

    def __getattr__(self, name: str) -> FieldRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldRef((*self.path, name))


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
class BinOp(RateOps):
    """Binary arithmetic on two rate expressions."""

    op: Literal["add", "sub", "mul", "div"]
    left: RateOps
    right: RateOps


@dataclass(frozen=True, slots=True)
class Multiply:
    """Multiply the previous aligned rate by ``value`` (optional ``where`` mask)."""

    value: RateOps
    where: Selector | None = None

    def __init__(self, value: object, where: Selector | None = None) -> None:
        object.__setattr__(self, "value", as_rate(value))
        object.__setattr__(self, "where", where)


@dataclass(frozen=True, slots=True)
class Overwrite:
    """Replace the previous aligned rate with ``value`` (optional ``where`` mask)."""

    value: RateOps
    where: Selector | None = None

    def __init__(self, value: object, where: Selector | None = None) -> None:
        object.__setattr__(self, "value", as_rate(value))
        object.__setattr__(self, "where", where)


@dataclass(frozen=True, slots=True)
class Transform:
    """Call ``fn(prev, *args)`` on the previous aligned rate."""

    fn: Callable[..., Any]
    args: tuple[RateOps, ...]
    where: Selector | None = None

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: object,
        where: Selector | None = None,
    ) -> None:
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", tuple(as_rate(arg) for arg in args))
        object.__setattr__(self, "where", where)


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
    """Coerce a scalar or rate node into a :class:`RateOps` expression."""
    if isinstance(value, RateOps):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    raise TypeError(f"Cannot use {type(value).__name__} as a flow rate.")


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
        case _:
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
        case _:
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


def _rate_bytes(expr: RateOps) -> bytes:
    match expr:
        case Const(value=value):
            return b"const" + np.float64(value).tobytes()
        case FieldRef(path=path):
            return b"field" + repr(path).encode()
        case FlowRef(name=name, reduce=reduce):
            return b"flow" + name.encode() + b"|" + repr(reduce).encode()
        case BinOp(op=op, left=left, right=right):
            return b"binop" + op.encode() + _rate_bytes(left) + _rate_bytes(right)
        case _:
            return type(expr).__name__.encode()


def _adjust_bytes(adj: Adjustment) -> bytes:
    if isinstance(adj, Transform):
        return b"tf" + str(id(adj.fn)).encode() + b"".join(_rate_bytes(arg) for arg in adj.args)
    kind = b"ow" if isinstance(adj, Overwrite) else b"mul"
    return kind + _rate_bytes(adj.value)
