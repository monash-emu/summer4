"""SolverInfo and Result container."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from jax.tree_util import register_pytree_node_class

from summer4.results.trace import Trace
from summer4.time import TimeAxis


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Typed solver metadata — not a free-form extras dict."""

    solver: str | None = None
    num_steps: int | None = None
    num_accepted_steps: int | None = None
    num_rejected_steps: int | None = None
    result_code: int | None = None
    max_steps: int | None = None
    dense: bool | None = None


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Result:
    """Flat mapping of named :class:`Trace`s from one solve.

    Keys are exactly the names in the :class:`~summer4.results.plan.SavePlan`;
    there is no privileged ``.compartments`` / ``.flows`` namespace.
    Deliberately does not carry ``params`` or ``model``.
    """

    times: TimeAxis
    traces: Mapping[str, Trace]
    solver: SolverInfo | None = None

    def __getitem__(self, key: str) -> Trace:
        try:
            return self.traces[key]
        except KeyError:
            raise KeyError(f"Unknown result key {key!r}. Known: {list(self.traces)}") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.traces

    def __iter__(self) -> Iterator[str]:
        return iter(self.traces)

    def keys(self) -> Any:
        return self.traces.keys()

    def with_params(self, params: object) -> ResultWithParams:
        """Host-side provenance wrapper; does not enter the pytree."""
        return ResultWithParams(result=self, params=params)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Any, ...], Any]:
        keys = tuple(self.traces.keys())
        dims = tuple(self.traces[k].dims for k in keys)
        flat_children: list[Any] = [self.times.values]
        pmaps: list[Any] = []
        for key in keys:
            tr = self.traces[key]
            from summer4.jax.propertydata import PropertyData

            if isinstance(tr.values, PropertyData):
                flat_children.append(tr.values.data)
                pmaps.append(tr.values.pmap)
            else:
                flat_children.append(tr.values)
                pmaps.append(None)
        aux = (self.times.epoch, self.times.kind, keys, dims, tuple(pmaps), self.solver)
        return tuple(flat_children), aux

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[Any, ...]) -> Result:
        from summer4.jax.propertydata import PropertyData

        epoch, kind, keys, dims, pmaps, solver = aux
        time_values, *vals = children
        times = TimeAxis(values=time_values, epoch=epoch, kind=kind)
        traces: dict[str, Trace] = {}
        for key, dim, pmap, val in zip(keys, dims, pmaps, vals, strict=True):
            values = PropertyData(pmap, val) if pmap is not None else val
            traces[key] = Trace(times=times, values=values, dims=dim)
        return cls(times=times, traces=traces, solver=solver)


@dataclass(frozen=True, slots=True)
class ResultWithParams:
    """Host-side provenance: a :class:`Result` plus the params used to produce it."""

    result: Result
    params: object
