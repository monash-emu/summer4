"""Batched posterior scenario runs, quantile frames, and averted differences.

``BayesianModel.posterior_runs`` normalises draws to a constrained site dict
with leading axis ``n``, evaluates each scenario in memory-bounded chunks, and
returns a :class:`PosteriorRuns` that keeps per-draw series for spaghetti plots
(step 28) as well as Kiribati-shaped quantile / difference frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from summer4.flows.compiled import CompiledModel
from summer4.results.outputset import OutputSet
from summer4.results.plan import SavePlan


@dataclass(frozen=True, slots=True)
class Scenario:
    """One forward-run variant relative to a :class:`BayesianModel`.

    ``params`` are merged on top of ``BayesianModel.merge_params(draw)``.
    ``model`` swaps the compiled graph (for example screening flows added).
    ``outputs`` swaps the named output DAG; when omitted, the BayesianModel's
    outputs (or raw save keys) are used.
    """

    params: Mapping[str, Any] | None = None
    model: CompiledModel | None = None
    outputs: OutputSet | None = None


@dataclass(frozen=True, slots=True)
class PosteriorRuns:
    """Per-draw scenario outputs with Kiribati quantile / difference summaries.

    ``samples(scenario, output)`` returns shape ``(n, t)``. Quantile frames use
    a time index and MultiIndex columns ``(output, quantile)`` with quantile
    labels as strings. Difference frames use a quantile index and ``X`` /
    ``X_relative`` columns for each logical name in ``outputs=``.
    """

    times: np.ndarray
    # scenario -> output_name -> (n, t)
    series: dict[str, dict[str, np.ndarray]]
    n: int
    scenario_names: tuple[str, ...]
    output_names: tuple[str, ...]
    draws: dict[str, np.ndarray] = field(repr=False)

    def samples(self, scenario: str, output: str) -> np.ndarray:
        """Per-draw trajectory for one named output; shape ``(n, t)``."""
        if scenario not in self.series:
            raise KeyError(f"Unknown scenario {scenario!r}. Known: {list(self.series)}.")
        by_out = self.series[scenario]
        if output not in by_out:
            raise KeyError(
                f"Unknown output {output!r} for scenario {scenario!r}. " f"Known: {list(by_out)}."
            )
        return np.asarray(by_out[output])

    def quantiles(
        self,
        q: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
    ) -> dict[str, Any]:
        """Kiribati-shaped uncertainty frames: time index, ``(output, q)`` columns."""
        import pandas as pd  # type: ignore[import-untyped]

        q_arr = np.asarray(list(q), dtype=float)
        q_labels = tuple(str(float(v)) for v in q_arr)
        out: dict[str, Any] = {}
        for scen in self.scenario_names:
            blocks: dict[tuple[str, str], np.ndarray] = {}
            for name in self.output_names:
                traj = self.samples(scen, name)  # (n, t)
                qs = np.quantile(traj, q_arr, axis=0)  # (n_q, t)
                for i, label in enumerate(q_labels):
                    blocks[(name, label)] = qs[i]
            frame = pd.DataFrame(blocks, index=np.asarray(self.times, dtype=float))
            frame.index.name = "time"
            frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["output", "quantile"])
            out[scen] = frame
        return out

    def differences(
        self,
        ref: str,
        outputs: Mapping[str, str],
        *,
        at: float,
        relative: bool = True,
        q: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
    ) -> dict[str, Any]:
        """Averted-burden quantiles vs ``ref`` at time ``at``.

        ``outputs`` maps a result name (column stem ``X``) to a stored output
        name. Absolute difference is scenario − ref; relative is that divided
        by the ref draw. The reference scenario is omitted from the result.
        """
        import pandas as pd

        if ref not in self.series:
            raise KeyError(f"Unknown ref scenario {ref!r}. Known: {list(self.series)}.")
        t_idx = _nearest_time_index(self.times, float(at))
        q_arr = np.asarray(list(q), dtype=float)
        q_labels = [str(float(v)) for v in q_arr]
        out: dict[str, Any] = {}
        for scen in self.scenario_names:
            if scen == ref:
                continue
            cols: dict[str, np.ndarray] = {}
            for result_name, src_name in outputs.items():
                scen_v = self.samples(scen, src_name)[:, t_idx]
                ref_v = self.samples(ref, src_name)[:, t_idx]
                diff = scen_v - ref_v
                cols[result_name] = np.quantile(diff, q_arr, axis=0)
                if relative:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.where(ref_v == 0.0, np.nan, diff / ref_v)
                    cols[f"{result_name}_relative"] = np.quantile(rel, q_arr, axis=0)
            frame = pd.DataFrame(cols, index=q_labels)
            frame.index.name = "quantile"
            out[scen] = frame
        return out


def _nearest_time_index(times: np.ndarray, at: float) -> int:
    arr = np.asarray(times, dtype=float)
    return int(np.argmin(np.abs(arr - float(at))))


def _series_1d(output: Any) -> Any:
    """Flatten one Output to a 1-d time series (JAX or NumPy)."""
    import jax.numpy as jnp

    vals = output.values
    data = vals.data if hasattr(vals, "data") else vals
    arr = jnp.asarray(data)
    if arr.ndim == 0:
        return arr.reshape((1,))
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2 and int(np.prod(arr.shape[1:])) == 1:
        return arr.reshape((arr.shape[0],))
    raise ValueError(
        "posterior_runs requires scalar time series outputs; got shape "
        f"{tuple(arr.shape)}. Use OutputSet entries with .total() (or similar)."
    )


def _as_draw_dict(draws: Any, *, n: int | None, burn_in: int, seed: int) -> dict[str, np.ndarray]:
    """Normalise InferenceData / Candidates-like / Mapping → constrained site arrays."""
    if draws is None:
        raise TypeError("posterior_runs requires draws (InferenceData, Mapping, or Candidates).")

    # Candidates (step 24+) expose .params with leading axis n.
    params_attr = getattr(draws, "params", None)
    if (
        params_attr is not None
        and isinstance(params_attr, Mapping)
        and not hasattr(draws, "posterior")
    ):
        return _subsample_mapping(dict(params_attr), n=n, seed=seed)

    posterior = getattr(draws, "posterior", None)
    if posterior is not None:
        return _draws_from_idata(posterior, n=n, burn_in=burn_in, seed=seed)

    if isinstance(draws, Mapping):
        return _subsample_mapping(dict(draws), n=n, seed=seed)

    raise TypeError(
        "posterior_runs draws must be an arviz.InferenceData, a Mapping of "
        f"site arrays, or an object with .params; got {type(draws).__name__}."
    )


def _draws_from_idata(
    posterior: Any,
    *,
    n: int | None,
    burn_in: int,
    seed: int,
) -> dict[str, np.ndarray]:
    sites = list(posterior.data_vars)
    if not sites:
        raise ValueError("InferenceData.posterior has no data variables.")
    out: dict[str, np.ndarray] = {}
    length: int | None = None
    for name in sites:
        arr = np.asarray(posterior[name].values)
        # (chain, draw, ...) → drop burn_in along draw, then flatten leading two.
        if arr.ndim < 2:
            raise ValueError(f"Posterior site {name!r} must have chain and draw axes.")
        if burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {burn_in}.")
        if burn_in >= arr.shape[1]:
            raise ValueError(
                f"burn_in={burn_in} leaves no samples for site {name!r} "
                f"(draw axis length {arr.shape[1]})."
            )
        trimmed = arr[:, burn_in:, ...]
        flat = trimmed.reshape(-1, *trimmed.shape[2:])
        if flat.ndim > 1 and flat.shape[1:] != ():
            # Scalar sites only for merge_params today.
            flat = flat.reshape(flat.shape[0], -1)
            if flat.shape[1] != 1:
                raise ValueError(
                    f"Posterior site {name!r} is not scalar (shape {arr.shape}); "
                    "posterior_runs expects scalar parameter sites."
                )
            flat = flat[:, 0]
        out[name] = np.asarray(flat, dtype=float)
        if length is None:
            length = out[name].shape[0]
        elif out[name].shape[0] != length:
            raise ValueError("Posterior sites have inconsistent draw counts.")
    assert length is not None
    return _subsample_mapping(out, n=n, seed=seed)


def _subsample_mapping(
    draws: Mapping[str, Any],
    *,
    n: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    if not draws:
        raise ValueError("draws mapping is empty.")
    arrays = {k: np.asarray(v, dtype=float) for k, v in draws.items()}
    lengths = {k: (a.shape[0] if a.ndim >= 1 else 1) for k, a in arrays.items()}
    # Promote scalars to length-1.
    for k, a in list(arrays.items()):
        if a.ndim == 0:
            arrays[k] = a.reshape((1,))
            lengths[k] = 1
    n_all = next(iter(lengths.values()))
    if any(n_all != L for L in lengths.values()):
        raise ValueError(f"Draw arrays have inconsistent leading lengths: {lengths}.")
    if n is None or int(n) >= n_all:
        return arrays
    n_keep = int(n)
    if n_keep <= 0:
        raise ValueError(f"n must be positive, got {n_keep}.")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_all, size=n_keep, replace=False)
    idx.sort()
    return {k: a[idx] for k, a in arrays.items()}


def _build_save_plan(
    targets_plan: SavePlan,
    outputs: OutputSet | None,
) -> SavePlan:
    # Prefer the OutputSet alone so posterior trajectories use the solver's
    # dense save grid — targets.plan would pin shared leaves to observation times.
    if outputs is not None:
        return outputs.plan(SavePlan())
    return targets_plan


def _output_names_for(
    bm: Any,
    outputs: OutputSet | None,
    probe_result: Any,
) -> tuple[str, ...]:
    if outputs is not None:
        return tuple(outputs.keys())
    if bm.outputs is not None:
        return tuple(bm.outputs.keys())
    # Fall back to every key on the probe result (targets + leaves).
    return tuple(str(k) for k in probe_result)


def _run_scenario_draws(
    *,
    compiled: CompiledModel,
    outputs: OutputSet | None,
    run_kwargs: dict[str, Any],
    y0_fn: Any,
    merge_fn: Any,
    save_plan: SavePlan,
    draw_dict: dict[str, np.ndarray],
    names: tuple[str, ...],
    batch: int,
) -> np.ndarray:
    """Return stacked draws as ``(n, n_out, t)`` for one scenario."""
    import jax
    import jax.numpy as jnp

    n_draws = next(iter(draw_dict.values())).shape[0]
    names_local = names
    outputs_local = outputs
    compiled_local = compiled
    merge_local = merge_fn
    y0_local = y0_fn
    save_local = save_plan
    kwargs_local = run_kwargs

    def run_draw(draw_i: dict[str, Any]) -> Any:
        params = merge_local(draw_i)
        y0 = y0_local(params)
        result = compiled_local.run(params, y0, save=save_local, **kwargs_local)
        scored = result
        if outputs_local is not None:
            scored = outputs_local.evaluate(result, params)
        cols = tuple(_series_1d(scored[name]) for name in names_local)
        return jnp.stack(cols, axis=0)

    run_chunk = jax.jit(jax.vmap(run_draw, in_axes=({k: 0 for k in draw_dict},)))
    chunks: list[np.ndarray] = []
    for start in range(0, n_draws, batch):
        stop = min(start + batch, n_draws)
        chunk = {k: jnp.asarray(v[start:stop]) for k, v in draw_dict.items()}
        chunks.append(np.asarray(run_chunk(chunk)))
    return np.concatenate(chunks, axis=0)


def run_posterior(
    bm: Any,
    draws: Any,
    *,
    n: int | None = 1000,
    burn_in: int = 0,
    seed: int = 0,
    scenarios: Mapping[str, Scenario | None] | None = None,
    batch_size: int = 64,
) -> PosteriorRuns:
    """Evaluate constrained draws under named scenarios; return :class:`PosteriorRuns`."""
    draw_dict = _as_draw_dict(draws, n=n, burn_in=int(burn_in), seed=int(seed))
    scen_map: dict[str, Scenario | None]
    if scenarios is None:
        scen_map = {"baseline": None}
    else:
        if not scenarios:
            raise ValueError("scenarios must be a non-empty mapping.")
        scen_map = dict(scenarios)

    batch = max(1, int(batch_size))
    series: dict[str, dict[str, np.ndarray]] = {}
    times_out: np.ndarray | None = None
    output_names: tuple[str, ...] | None = None

    for scen_name, scen in scen_map.items():
        compiled, outputs, run_kwargs, y0_fn, merge_fn, save_plan = _scenario_bindings(bm, scen)
        probe_params = merge_fn({k: float(v[0]) for k, v in draw_dict.items()})
        probe_y0 = y0_fn(probe_params)
        probe = compiled.run(probe_params, probe_y0, save=save_plan, **run_kwargs)
        if outputs is not None:
            probe = outputs.evaluate(probe, probe_params)
        names = _output_names_for(bm, outputs, probe)
        if not names:
            raise ValueError(
                "posterior_runs found no outputs to record; pass outputs= on "
                "BayesianModel or Scenario, or ensure the save plan has keys."
            )
        if output_names is None:
            output_names = names
        elif names != output_names:
            raise ValueError(
                f"Scenario {scen_name!r} output names {names} differ from "
                f"{output_names}; every scenario must share the same output set."
            )
        t_probe = np.asarray(probe[names[0]].times.values, dtype=float)
        if times_out is None:
            times_out = t_probe
        elif not np.allclose(times_out, t_probe):
            raise ValueError(f"Scenario {scen_name!r} times differ from the first scenario.")

        all_draws = _run_scenario_draws(
            compiled=compiled,
            outputs=outputs,
            run_kwargs=run_kwargs,
            y0_fn=y0_fn,
            merge_fn=merge_fn,
            save_plan=save_plan,
            draw_dict=draw_dict,
            names=names,
            batch=batch,
        )
        series[scen_name] = {name: all_draws[:, i, :] for i, name in enumerate(names)}

    assert times_out is not None and output_names is not None
    return PosteriorRuns(
        times=times_out,
        series=series,
        n=next(iter(draw_dict.values())).shape[0],
        scenario_names=tuple(scen_map.keys()),
        output_names=output_names,
        draws={k: np.asarray(v) for k, v in draw_dict.items()},
    )


def _scenario_bindings(
    bm: Any,
    scen: Scenario | None,
) -> tuple[
    CompiledModel,
    OutputSet | None,
    dict[str, Any],
    Any,
    Any,
    SavePlan,
]:
    compiled = bm.compiled if scen is None or scen.model is None else scen.model
    if not isinstance(compiled, CompiledModel):
        raise TypeError(f"scenario model must be a CompiledModel, got {type(compiled).__name__}.")
    outputs = bm.outputs if scen is None or scen.outputs is None else scen.outputs
    overrides = {} if scen is None or scen.params is None else dict(scen.params)

    def merge_fn(draw: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(bm.merge_params(draw))
        params.update(overrides)
        return params

    # y0 resolution mirrors BayesianModel._resolve_y0 but with the scenario model.
    init = bm.init
    fixed_y0 = bm.y0

    def y0_fn(params: Any) -> Any:
        if fixed_y0 is not None:
            return fixed_y0
        if init is not None:
            prepared = compiled.prepare(params)
            return init.compile(compiled.pmap).evaluate(prepared.params)
        return None

    run_kwargs = dict(bm.run_kwargs)
    if "save" in run_kwargs:
        raise ValueError(
            "run_kwargs must not include save=; posterior_runs builds the save "
            "plan from targets and outputs."
        )
    save_plan = _build_save_plan(bm.targets.plan(SavePlan()), outputs)
    return compiled, outputs, run_kwargs, y0_fn, merge_fn, save_plan


__all__ = [
    "PosteriorRuns",
    "Scenario",
    "run_posterior",
]
