"""SolverInfo and Result container."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from jax.tree_util import register_pytree_node_class

from summer4.results.output import (
    Output,
    _as_array,
    _column_labels,
    _time_axis_index,
)
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
    """Flat mapping of named :class:`~summer4.results.output.Output` objects from one solve.

    Keys are exactly the names in the :class:`~summer4.results.plan.SavePlan`;
    there is no privileged ``.compartments`` / ``.flows`` namespace.
    Deliberately does not carry ``params`` or ``model``.

    ``dense`` holds the solver's interpolation when the plan requested it.
    Calling :meth:`evaluate` requires a finite ``max_steps`` at solve time and
    allocates that many interpolation coefficients — strictly heavier than any
    save grid based on ``ts`` alone.
    """

    times: TimeAxis
    outputs: Mapping[str, Output]
    solver: SolverInfo | None = None
    dense: Any | None = None
    _state_pmap: Any = field(default=None, repr=False, compare=False)

    def __getitem__(self, key: str) -> Output:
        try:
            return self.outputs[key]
        except KeyError:
            raise KeyError(f"Unknown result key {key!r}. Known: {list(self.outputs)}") from None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.outputs

    def __iter__(self) -> Iterator[str]:
        return iter(self.outputs)

    def keys(self) -> Any:
        return self.outputs.keys()

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

    def to_frame(
        self,
        names: Sequence[str] | None = None,
        *,
        shape: Literal["wide", "long"] = "wide",
        backend: Literal["pandas", "polars"] = "polars",
    ) -> object:
        """Stack named outputs into one frame. Host-side; not traceable.

        ``wide`` has one time column (the pandas index) and one column per
        scalar output. An output with several aligned values becomes
        ``name[label]`` columns. Every included output must share the same
        times.

        ``long`` has ``time``, ``output`` and ``value``, plus one column per
        property on the aligned map (an age band, a compartment state). Outputs
        may use different times. A polars frame writes parquet with
        ``.write_parquet``; that needs the ``frames`` extra. ``backend="pandas"``
        needs the ``pandas`` extra, and uses a ``DatetimeIndex`` when the time
        axis has an epoch.
        """
        if shape not in ("wide", "long"):
            raise ValueError(f"shape must be 'wide' or 'long', got {shape!r}.")
        if backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got {backend!r}.")
        chosen = _frame_names(self, names)
        outputs = [(name, self.outputs[name]) for name in chosen]
        for name, output in outputs:
            _require_host_output(name, output)
        if shape == "wide":
            return _wide_frame(outputs, backend)
        return _long_frame(outputs, backend)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Any, ...], Any]:
        keys = tuple(self.outputs.keys())
        dims = tuple(self.outputs[k].dims for k in keys)
        times_list = [self.outputs[k].times.values for k in keys]
        epochs = tuple(self.outputs[k].times.epoch for k in keys)
        kinds = tuple(self.outputs[k].times.kind for k in keys)
        flat_children: list[Any] = [self.times.values, self.solver, self.dense]
        pmaps: list[Any] = []
        for key in keys:
            tr = self.outputs[key]
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
        outputs: dict[str, Output] = {}
        for key, dim, pmap, val, tvals, ep, kd in zip(
            keys, dims, pmaps, vals, trace_times, epochs, kinds, strict=True
        ):
            values = PropertyData(pmap, val) if pmap is not None else val
            outputs[key] = Output(
                times=TimeAxis(values=tvals, epoch=ep, kind=kd),
                values=values,
                dims=dim,
            )
        return cls(times=times, outputs=outputs, solver=solver, dense=dense, _state_pmap=state_pmap)


@dataclass(frozen=True, slots=True)
class ResultWithParams:
    """Host-side provenance: a :class:`Result` plus the params used to produce it."""

    result: Result
    params: object


def _frame_names(result: Result, names: Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(result.keys())
    missing = [name for name in names if name not in result]
    if missing:
        raise KeyError(f"Result has no output {missing[0]!r}. Known: {list(result.keys())}.")
    return tuple(names)


def _is_traced_value(value: object) -> bool:
    """True for a JAX tracer, not for a concrete jax array (those have ``aval`` too)."""
    return type(value).__name__ in {"DynamicJaxprTracer", "Tracer"}


def _require_host_output(name: str, output: Output) -> None:
    if _is_traced_value(output.times.values) or _is_traced_value(_as_array(output.values)):
        raise TypeError(
            f"Result.to_frame cannot format {name!r} while its values are traced. "
            "Call it outside jit."
        )


def _require_polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "Result.to_frame(backend='polars') requires the frames extra: "
            "pip install 'summer4[frames]'."
        ) from exc
    return pl


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Result.to_frame(backend='pandas') requires the pandas extra: "
            "pip install 'summer4[pandas]'."
        ) from exc
    return pd


def _time_major(output: Output) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(_as_array(output.values))
    times = np.asarray(output.times.values, dtype=np.float64).reshape(-1)
    if "time" in output.dims:
        t_ax = _time_axis_index(output.dims)
        if values.ndim == 0:
            values = values.reshape(1)
        if t_ax != 0:
            values = np.moveaxis(values, t_ax, 0)
        if values.shape[0] != times.shape[0]:
            raise ValueError(
                f"Time axis length {values.shape[0]} does not match {times.shape[0]} save times."
            )
        return times, values
    flat = np.asarray(values).reshape(1, -1) if np.asarray(values).size else np.zeros((1, 0))
    return times[:1], flat


def _aligned_labels(output: Output, n_cols: int) -> tuple[str, ...] | None:
    pmap = output._pmap()
    if pmap is None or n_cols != pmap.size:
        return None
    return _column_labels(pmap, output.dims)


def _wide_columns(name: str, output: Output, values: np.ndarray) -> dict[str, np.ndarray]:
    if values.ndim <= 1:
        column = values.reshape(-1) if values.ndim == 1 else values.reshape(1)
        return {name: column}
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[1] == 1:
        return {name: flat[:, 0]}
    labels = _aligned_labels(output, flat.shape[1])
    columns: dict[str, np.ndarray] = {}
    for j in range(flat.shape[1]):
        label = labels[j] if labels is not None else f"v{j}"
        columns[f"{name}[{label}]"] = flat[:, j]
    return columns


def _epoch_of(outputs: list[tuple[str, Output]]) -> Any:
    epochs = {output.times.epoch for _name, output in outputs}
    if len(epochs) != 1:
        return None
    return next(iter(epochs))


def _wide_frame(outputs: list[tuple[str, Output]], backend: Literal["pandas", "polars"]) -> object:
    if not outputs:
        columns: dict[str, Any] = {"time": np.zeros(0, dtype=np.float64)}
        times = columns["time"]
    else:
        first_name, first = outputs[0]
        times, values = _time_major(first)
        columns = {"time": times}
        columns.update(_wide_columns(first_name, first, values))
        for name, output in outputs[1:]:
            other_times, other_values = _time_major(output)
            if other_times.shape != times.shape or not np.array_equal(other_times, times):
                raise ValueError(
                    f"Output {name!r} is not on the same times as {first_name!r}. "
                    "A wide frame needs one time axis; use shape='long' for mixed grids."
                )
            extra = _wide_columns(name, output, other_values)
            overlap = set(extra).intersection(columns)
            if overlap:
                raise ValueError(f"Wide frame columns collide: {sorted(overlap)}.")
            columns.update(extra)
    epoch = _epoch_of(outputs)
    if backend == "polars":
        pl = _require_polars()
        return pl.DataFrame(columns)
    pd = _require_pandas()
    frame = pd.DataFrame(columns)
    if epoch is not None and len(times):
        frame["time"] = epoch.from_model(np.asarray(times, dtype=np.float64))
    return frame.set_index("time")


def _long_rows(name: str, output: Output) -> list[dict[str, Any]]:
    times, values = _time_major(output)
    flat = values.reshape(-1, 1) if values.ndim <= 1 else values.reshape(values.shape[0], -1)
    pmap = output._pmap()
    dicts: list[dict[str, str]] | None = None
    if pmap is not None and flat.shape[1] == pmap.size:
        dicts = pmap.to_dicts()
    labels = _aligned_labels(output, int(flat.shape[1]))
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(times):
        for j in range(flat.shape[1]):
            row: dict[str, Any] = {"time": float(t), "output": name, "value": float(flat[i, j])}
            if dicts is not None:
                row.update(dicts[j])
            elif labels is not None:
                row["label"] = labels[j]
            elif flat.shape[1] > 1:
                row["index"] = j
            rows.append(row)
    return rows


def _long_frame(outputs: list[tuple[str, Output]], backend: Literal["pandas", "polars"]) -> object:
    rows: list[dict[str, Any]] = []
    for name, output in outputs:
        rows.extend(_long_rows(name, output))
    epoch = _epoch_of(outputs)
    if backend == "polars":
        pl = _require_polars()
        if not rows:
            return pl.DataFrame({"time": [], "output": [], "value": []})
        return pl.DataFrame(rows)
    pd = _require_pandas()
    frame = pd.DataFrame(rows if rows else {"time": [], "output": [], "value": []})
    if epoch is not None and len(frame):
        frame["time"] = epoch.from_model(np.asarray(frame["time"], dtype=np.float64))
    return frame
