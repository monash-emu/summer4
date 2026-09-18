"""SolverInfo and Result container."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from jax.tree_util import register_pytree_node_class

from summer4.results.trace import Trace
from summer4.time import TimeAxis


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Typed solver metadata — not a free-form extras dict.

    Numeric stats are pytree children so they may be tracers under ``jax.jit``.
    ``solver`` / ``dense`` / ``max_steps`` stay static aux.
    """

    solver: str | None = None
    num_steps: int | None = None
    num_accepted_steps: int | None = None
    num_rejected_steps: int | None = None
    result_code: int | None = None
    max_steps: int | None = None
    dense: bool | None = None

    @property
    def message(self) -> str:
        """Human-readable status from ``result_code`` (host-side; not traced)."""
        if self.result_code is None:
            return ""
        code = int(self.result_code)
        try:
            from diffrax import RESULTS

            messages = RESULTS._index_to_message
            if 0 <= code < len(messages):
                msg = messages[code]
                return msg if msg else "successful"
        except Exception:
            pass
        return f"result_code={code}"

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        children = (
            self.num_steps,
            self.num_accepted_steps,
            self.num_rejected_steps,
            self.result_code,
        )
        aux = (self.solver, self.max_steps, self.dense)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[Any, ...]) -> SolverInfo:
        solver, max_steps, dense = aux
        num_steps, num_accepted, num_rejected, result_code = children
        return cls(
            solver=solver,
            num_steps=num_steps,
            num_accepted_steps=num_accepted,
            num_rejected_steps=num_rejected,
            result_code=result_code,
            max_steps=max_steps,
            dense=dense,
        )


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Result:
    """Flat mapping of named :class:`~summer4.results.trace.Trace` objects from one solve.

    Keys are exactly the names in the :class:`~summer4.results.plan.SavePlan`;
    there is no privileged ``.compartments`` / ``.flows`` namespace.
    Deliberately does not carry ``params`` or ``model``.

    ``dense`` holds the solver's interpolation when the plan requested it.
    Calling :meth:`evaluate` requires a finite ``max_steps`` at solve time and
    allocates that many interpolation coefficients — strictly heavier than any
    save grid based on ``ts`` alone.
    """

    times: TimeAxis
    traces: Mapping[str, Trace]
    solver: SolverInfo | None = None
    dense: Any | None = None
    _state_pmap: Any = field(default=None, repr=False, compare=False)

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

    def evaluate(self, t: object) -> Any:
        """Evaluate dense state at time ``t`` as :class:`~summer4.jax.PropertyData`.

        Raises if the plan was not solved with ``dense=True``.
        """
        if self.dense is None:
            raise ValueError(
                "Result has no dense output. Re-run with SavePlan(..., dense=True) "
                "and a finite max_steps (dense allocates max_steps interpolation "
                "coefficients)."
            )
        if self._state_pmap is None:
            raise ValueError("Result.dense is set but the state PropertyMap is missing.")
        from summer4.jax.propertydata import PropertyData

        y = self.dense.evaluate(t)
        return PropertyData(self._state_pmap, y)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Any, ...], Any]:
        keys = tuple(self.traces.keys())
        dims = tuple(self.traces[k].dims for k in keys)
        times_list = [self.traces[k].times.values for k in keys]
        epochs = tuple(self.traces[k].times.epoch for k in keys)
        kinds = tuple(self.traces[k].times.kind for k in keys)
        flat_children: list[Any] = [self.times.values, self.solver, self.dense]
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
        flat_children.extend(times_list)
        aux = (
            self.times.epoch,
            self.times.kind,
            keys,
            dims,
            tuple(pmaps),
            epochs,
            kinds,
            self._state_pmap,
        )
        return tuple(flat_children), aux

    @classmethod
    def tree_unflatten(cls, aux: Any, children: tuple[Any, ...]) -> Result:
        from summer4.jax.propertydata import PropertyData

        epoch, kind, keys, dims, pmaps, epochs, kinds, state_pmap = aux
        time_values, solver, dense, *rest = children
        n = len(keys)
        vals = rest[:n]
        trace_times = rest[n:]
        times = TimeAxis(values=time_values, epoch=epoch, kind=kind)
        traces: dict[str, Trace] = {}
        for key, dim, pmap, val, tvals, ep, kd in zip(
            keys, dims, pmaps, vals, trace_times, epochs, kinds, strict=True
        ):
            values = PropertyData(pmap, val) if pmap is not None else val
            traces[key] = Trace(
                times=TimeAxis(values=tvals, epoch=ep, kind=kd),
                values=values,
                dims=dim,
            )
        return cls(times=times, traces=traces, solver=solver, dense=dense, _state_pmap=state_pmap)


@dataclass(frozen=True, slots=True)
class ResultWithParams:
    """Host-side provenance: a :class:`Result` plus the params used to produce it."""

    result: Result
    params: object
