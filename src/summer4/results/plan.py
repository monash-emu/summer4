"""Save plans: what to keep from a solve."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from summer4.flows.rates import _callable_key
from summer4.properties import Property
from summer4.selectors import Selector

if TYPE_CHECKING:
    from summer4.results.outputset import OutputExpr

type Side = Literal["source", "dest"]
type Quantity = Compartments | FlowMass | ComputedValue | SaveFn | GroupedOutput


class _OutputExprChain:
    """Record post-ops on a save spec for :class:`~summer4.results.outputset.OutputSet`.

    ``Compartments().sum_by()`` does not reduce at save time. The save is the
    quantity; :meth:`~summer4.results.outputset.OutputSet.evaluate` applies the
    chain. Arithmetic on the returned expression is the same idea.
    """

    __slots__ = ()

    def _as_output_expr(self) -> OutputExpr:
        from summer4.results.outputset import OutputExpr

        return OutputExpr.leaf(self)

    def select(self, sel: Selector | np.ndarray) -> OutputExpr:
        return self._as_output_expr().select(sel)

    def sum_by(self, *props: Property | str, side: Side | None = None) -> OutputExpr:
        return self._as_output_expr().sum_by(*props, side=side)

    def total(self) -> OutputExpr:
        """Deprecated spelling of :meth:`sum_by` with no properties."""
        from summer4._deprecation import warn_renamed

        warn_renamed("total()", "sum_by()")
        return self._as_output_expr().sum_by()

    def rolling(
        self,
        window: int,
        *,
        how: Any = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ) -> OutputExpr:
        return self._as_output_expr().rolling(
            window, how=how, center=center, min_periods=min_periods
        )

    def cumulative(self, *, start: Any = None, end: Any = None) -> OutputExpr:
        return self._as_output_expr().cumulative(start=start, end=end)

    def midpoint(self) -> OutputExpr:
        return self._as_output_expr().midpoint()

    def integrate_intervals(
        self, method: Literal["trapezoid", "simpson"] = "trapezoid"
    ) -> OutputExpr:
        return self._as_output_expr().integrate_intervals(method)

    def integrate(self, method: Literal["trapezoid", "simpson"] = "trapezoid") -> OutputExpr:
        return self._as_output_expr().integrate(method)

    def at(self, when: Any) -> OutputExpr:
        return self._as_output_expr().at(when)

    def at_times(self, ts: Any) -> OutputExpr:
        return self._as_output_expr().at_times(ts)

    def array(self) -> OutputExpr:
        return self._as_output_expr().array()

    def scalar(self) -> OutputExpr:
        return self._as_output_expr().scalar()

    def with_values(self, values: object) -> OutputExpr:
        return self._as_output_expr().with_values(values)


@dataclass(frozen=True, slots=True, init=False)
class Compartments(_OutputExprChain):
    """Save compartment densities (optionally filtered / reduced at save time).

    ``group_by=age`` saves one value per age instead of every compartment.
    ``sum_over=`` is the deprecated spelling of ``group_by=``.
    """

    where: Selector | None = None
    group_by: Property | None = None

    def __init__(
        self,
        where: Selector | None = None,
        group_by: Property | None = None,
        *,
        sum_over: Property | None = None,
    ) -> None:
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "group_by", _renamed_group_by("Compartments", group_by, sum_over))


def _renamed_group_by(owner: str, group_by: Any, sum_over: Any) -> Any:
    if sum_over is None:
        return group_by
    if group_by is not None:
        raise TypeError(f"{owner}: pass group_by= only (sum_over= is its deprecated spelling).")
    from summer4._deprecation import warn_renamed

    warn_renamed(f"{owner}(sum_over=...)", f"{owner}(group_by=...)", stacklevel=4)
    return sum_over


def flow_names(flow: str | tuple[str, ...]) -> tuple[str, ...]:
    """Return the flow names a :class:`FlowMass` reads, in declaration order."""
    names = (flow,) if isinstance(flow, str) else tuple(flow)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"FlowMass.flow must name at least one flow, got {flow!r}.")
    if len(names) != len(set(names)):
        raise ValueError(f"FlowMass.flow repeats a name: {names}.")
    return names


@dataclass(frozen=True, slots=True, init=False)
class FlowMass(_OutputExprChain):
    """Per-edge mass. Reduce *here* so the buffer is ``(n_saves, n_kept)``.

    ``flow`` is one name or several. Several flows are summed at each save
    after the same ``where`` / ``group_by``. They must share those selected
    dims (the same :class:`~summer4.propertymap.PropertyMap`); otherwise
    evaluation raises and names the flows.

    ``group_by=(age, "source")`` saves one value per source age instead of every
    edge; ``sum_over=`` is its deprecated spelling. A 200k-edge flow over 3650
    days is ~5.8 GB dense and ~470 kB grouped by age. Post-hoc edge queries use
    ``Output.sum_by(..., side=)``, ``integrate_intervals``, and ``integrate``.
    """

    flow: str | tuple[str, ...]
    where: Selector | NDArray[np.bool_] | None = None
    group_by: tuple[Property, Side] | None = None

    def __init__(
        self,
        flow: str | tuple[str, ...],
        where: Selector | NDArray[np.bool_] | None = None,
        group_by: tuple[Property, Side] | None = None,
        *,
        sum_over: tuple[Property, Side] | None = None,
    ) -> None:
        names = flow_names(flow)
        stored: str | tuple[str, ...] = names[0] if len(names) == 1 else names
        object.__setattr__(self, "flow", stored)
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "group_by", _renamed_group_by("FlowMass", group_by, sum_over))


@dataclass(frozen=True, slots=True)
class ComputedValue(_OutputExprChain):
    """Capture a path from the derived-param struct produced by ``derived_fn``."""

    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupedOutput(_OutputExprChain):
    """Save a :class:`~summer4.flows.rates.Capture`d :class:`GroupedRate` by name.

    The resulting output is a :class:`~summer4.jax.propertydata.PropertyData`
    over ``PropertyMap.from_property(group)``, so ``dims`` carry the grouping
    property name (e.g. ``("time", "age")``).
    """

    name: str


@dataclass(frozen=True, slots=True)
class SaveFn(_OutputExprChain):
    """Escape hatch: ``fn(ctx: SaveContext) -> array``.

    ``fn`` hashes by identity, so a lambda defined inside a loop retraces every
    call (same trap as ``derived_fn`` / ``Transform.fn``).
    """

    fn: Callable[[Any], Any]
    reads: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class SaveRequest:
    """One named output in a :class:`SavePlan`."""

    what: Quantity
    ts: NDArray[np.float64] | None = None  # None -> the plan's default grid


def _quantity_bytes(what: Quantity) -> bytes:
    match what:
        case Compartments(where=where, group_by=group_by):
            return b"comp" + repr(where).encode() + repr(group_by).encode()
        case FlowMass(flow=flow, where=where, group_by=group_by):
            names = b"\0".join(name.encode() for name in flow_names(flow))
            return b"flow" + names + repr(where).encode() + repr(group_by).encode()
        case ComputedValue(path=path):
            return b"cv" + repr(path).encode()
        case GroupedOutput(name=name):
            return b"grouped" + name.encode()
        case SaveFn(fn=fn, reads=reads):
            return b"fn" + _callable_key(fn).encode() + repr(sorted(reads)).encode()
        case _:
            raise TypeError(f"Unknown quantity {type(what).__name__}.")


def _plan_digest(
    requests: Mapping[str, SaveRequest],
    ts: NDArray[np.float64] | None,
    dense: bool,
    solver_stats: bool,
) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    for key in sorted(requests):
        req = requests[key]
        hasher.update(key.encode())
        hasher.update(_quantity_bytes(req.what))
        if req.ts is not None:
            hasher.update(np.ascontiguousarray(req.ts, dtype=np.float64).tobytes())
        else:
            hasher.update(b"nots")
    if ts is not None:
        hasher.update(np.ascontiguousarray(ts, dtype=np.float64).tobytes())
    else:
        hasher.update(b"nots")
    hasher.update(b"d1" if dense else b"d0")
    hasher.update(b"s1" if solver_stats else b"s0")
    return hasher.digest()


@dataclass(frozen=True, slots=True, eq=False)
class SavePlan:
    """What to keep from a solve.

    An empty :attr:`requests` mapping is the ``EVERYTHING`` sentinel:
    :meth:`CompiledModel.expand` fills compartments, every flow, and every
    computed path. There is exactly one code path below ``expand``.
    """

    requests: Mapping[str, SaveRequest] = field(default_factory=dict)
    ts: NDArray[np.float64] | None = None
    dense: bool = False
    solver_stats: bool = True
    _digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_digest",
            _plan_digest(self.requests, self.ts, self.dense, self.solver_stats),
        )

    def __hash__(self) -> int:
        return hash(self._digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SavePlan):
            return NotImplemented
        return self._digest == other._digest

    def flow_reads(self) -> frozenset[str]:
        """Return flow names this plan materialises (directly or via SaveFn.reads)."""
        names: set[str] = set()
        for req in self.requests.values():
            match req.what:
                case FlowMass(flow=flow):
                    names.update(flow_names(flow))
                case SaveFn(reads=reads):
                    names |= set(reads)
                case _:
                    pass
        return frozenset(names)


EVERYTHING = SavePlan()


@dataclass(frozen=True, slots=True)
class OutputShape:
    """Shape and byte footprint of one saved quantity."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class PlanDescription:
    """Result of :meth:`CompiledModel.describe` — shapes without solving."""

    outputs: tuple[OutputShape, ...]
    n_saves: int
    total_nbytes: int

    def __str__(self) -> str:
        lines = [f"PlanDescription(n_saves={self.n_saves}, total={self.total_nbytes:,} B)"]
        for out in self.outputs:
            lines.append(f"  {out.key}: shape={out.shape} dtype={out.dtype} ({out.nbytes:,} B)")
        return "\n".join(lines)
