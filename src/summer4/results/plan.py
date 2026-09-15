"""Save plans: what to keep from a solve."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from summer4.properties import Property
from summer4.selectors import Selector

type Side = Literal["source", "dest"]
type Quantity = Compartments | FlowMass | ComputedValue | SaveFn


@dataclass(frozen=True, slots=True)
class Compartments:
    """Save compartment densities (optionally filtered / reduced at save time)."""

    where: Selector | None = None
    sum_over: Property | None = None


@dataclass(frozen=True, slots=True)
class FlowMass:
    """Per-edge mass. Reduce *here* so the buffer is ``(n_saves, n_kept)``.

    A 200k-edge flow over 3650 days is ~5.8 GB dense and ~470 kB when summed
    over age. Post-hoc edge queries use ``Trace.sum_over(..., side=)``,
    ``integrate``, and ``incidence``.
    """

    flow: str
    where: Selector | NDArray[np.bool_] | None = None
    sum_over: tuple[Property, Side] | None = None


@dataclass(frozen=True, slots=True)
class ComputedValue:
    """Capture a path from the derived-param struct produced by ``derived_fn``."""

    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SaveFn:
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
        case Compartments(where=where, sum_over=sum_over):
            return b"comp" + repr(where).encode() + repr(sum_over).encode()
        case FlowMass(flow=flow, where=where, sum_over=sum_over):
            return b"flow" + flow.encode() + repr(where).encode() + repr(sum_over).encode()
        case ComputedValue(path=path):
            return b"cv" + repr(path).encode()
        case SaveFn(fn=fn, reads=reads):
            return b"fn" + str(id(fn)).encode() + repr(sorted(reads)).encode()
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
                    names.add(flow)
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
