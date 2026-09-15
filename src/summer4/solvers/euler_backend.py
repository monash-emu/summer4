"""Fixed-step Euler backend with per-group saves."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from summer4.results.groups import SaveGroup
from summer4.results.result import SolverInfo
from summer4.solvers.base import SolveOutput, SolveSpec


def _is_arithmetic_subgrid(ts: NDArray[np.float64], t0: float, dt: float) -> bool:
    """True when ``ts`` is an arithmetic subgrid of the Euler step grid including t0."""
    if ts.size == 0:
        return False
    if abs(float(ts[0]) - float(t0)) > 1e-9 * max(1.0, abs(t0)):
        return False
    steps = (ts - t0) / dt
    if not np.allclose(steps, np.round(steps), atol=1e-6):
        return False
    rounded = np.round(steps).astype(np.int64)
    if rounded.size >= 2:
        diffs = np.diff(rounded)
        if not np.all(diffs == diffs[0]) or diffs[0] <= 0:
            return False
    return True


def _snapshot_factory(
    model: Any,
    params: object,
    rebox: Any,
    save_fn: Any,
    keep: frozenset[str] | None = None,
) -> Any:
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData

    def observe_arr(t: Any, y: Any) -> Any:
        return model.observe(t, rebox(y), params, keep=keep)

    def snapshot(t: Any, y: Any) -> dict[str, Any]:
        ctx = observe_arr(t, y)
        raw = save_fn(ctx)
        out: dict[str, Any] = {}
        for k, v in raw.items():
            out[k] = v.data if isinstance(v, PropertyData) else jnp.asarray(v)
        return out

    return observe_arr, snapshot


def _dy_array(dy_val: Any) -> Any:
    from summer4.jax.propertydata import PropertyData
    from summer4.jax.state import State

    if isinstance(dy_val, State):
        return dy_val.compartments.data
    if isinstance(dy_val, PropertyData):
        return dy_val.data
    return dy_val


def _euler_group_fast(
    *,
    observe_arr: Any,
    snapshot: Any,
    y_arr: Any,
    t0: float,
    dt: float,
    steps: int,
    ts: NDArray[np.float64],
    init_snap: dict[str, Any],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    stride = int(round((float(ts[1]) - float(ts[0])) / dt)) if ts.size > 1 else max(steps, 1)
    n_saves = int(ts.size)

    def outer_body(
        carry: tuple[Any, Any, Any], unused: Any
    ) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
        del unused
        t, y, k = carry
        snap = snapshot(t, y)

        def step(i: Any, ty: tuple[Any, Any]) -> tuple[Any, Any]:
            del i
            tt, yy = ty
            ctx = observe_arr(tt, yy)
            return tt + dt, yy + dt * _dy_array(ctx.dy)

        do_step = k < (n_saves - 1)

        def do_stride(ty: tuple[Any, Any]) -> tuple[Any, Any]:
            result: tuple[Any, Any] = jax.lax.fori_loop(0, stride, step, ty)
            return result

        t2, y2 = jax.lax.cond(do_step, do_stride, lambda ty: ty, (t, y))
        return (t2, y2, k + 1), snap

    (_tf, _yf, _), snaps = jax.lax.scan(
        outer_body, (jnp.asarray(t0), y_arr, jnp.asarray(0)), xs=None, length=n_saves
    )
    return {k: snaps[k] for k in init_snap}


def _euler_trajectory(
    *,
    observe_arr: Any,
    y_arr: Any,
    t0: float,
    dt: float,
    steps: int,
) -> tuple[Any, NDArray[np.float64]]:
    """Integrate once; return stacked states including y0 and the step grid."""
    import jax
    import jax.numpy as jnp

    def body(carry: tuple[Any, Any], unused: Any) -> tuple[tuple[Any, Any], Any]:
        del unused
        t, y = carry
        ctx = observe_arr(t, y)
        return (t + dt, y + dt * _dy_array(ctx.dy)), y

    (_t_final, y_final), ys = jax.lax.scan(body, (jnp.asarray(t0), y_arr), xs=None, length=steps)
    ys_all = jnp.concatenate([ys, y_final[None, ...]], axis=0)
    step_ts = t0 + dt * np.arange(steps + 1, dtype=np.float64)
    return ys_all, step_ts


def _lerp_snapshot(
    *,
    snapshot: Any,
    ys_all: Any,
    step_ts: NDArray[np.float64],
    ts: NDArray[np.float64],
    init_snap: dict[str, Any],
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from summer4.time import TimeAxis

    axis = TimeAxis(values=step_ts, kind="grid")
    idx, w = axis.weights_for(ts)
    idx_j = jnp.asarray(idx)
    w_j = jnp.asarray(w)
    y_left = ys_all[idx_j[:, 0]]
    y_right = ys_all[idx_j[:, 1]]
    y_at = w_j[:, 0:1] * y_left + w_j[:, 1:2] * y_right
    snaps = jax.vmap(snapshot)(jnp.asarray(ts), y_at)
    return {k: snaps[k] for k in init_snap}


def _filtered_plan(plan: Any, keys: tuple[str, ...]) -> Any:
    from summer4.results.plan import SavePlan

    return SavePlan(
        requests={k: plan.requests[k] for k in keys},
        ts=plan.ts,
        dense=plan.dense,
        solver_stats=plan.solver_stats,
    )


def euler_solve(
    model: Any,
    *,
    y0: object,
    params: object,
    spec: SolveSpec,
    groups: tuple[SaveGroup, ...],
    plan: Any,
    solver_stats: bool,
) -> SolveOutput:
    """Integrate with Euler and evaluate each save group on its own ``ts``."""
    from summer4.jax.state import unpack_state
    from summer4.results.eval import build_save_fn

    if spec.steps is None:
        raise ValueError("Euler backend requires a fixed step count.")
    steps = int(spec.steps)
    dt = float(spec.dt)
    t0 = float(spec.t0)

    y_arr, rebox = unpack_state(y0, model.pmap)
    saved: dict[str, Any] = {}
    ys_all: Any | None = None
    step_ts: NDArray[np.float64] | None = None

    for group in groups:
        group_plan = _filtered_plan(plan, group.keys)
        save_fn = build_save_fn(group_plan, pmap=model.pmap, edge_maps=dict(model.edge_maps))
        keep = group_plan.flow_reads()
        observe_arr, snapshot = _snapshot_factory(model, params, rebox, save_fn, keep=keep)
        init_snap = snapshot(np.asarray(t0), y_arr)

        if _is_arithmetic_subgrid(group.ts, t0, dt):
            part = _euler_group_fast(
                observe_arr=observe_arr,
                snapshot=snapshot,
                y_arr=y_arr,
                t0=t0,
                dt=dt,
                steps=steps,
                ts=group.ts,
                init_snap=init_snap,
            )
        else:
            if ys_all is None:
                ys_all, step_ts = _euler_trajectory(
                    observe_arr=observe_arr,
                    y_arr=y_arr,
                    t0=t0,
                    dt=dt,
                    steps=steps,
                )
            assert step_ts is not None
            part = _lerp_snapshot(
                snapshot=snapshot,
                ys_all=ys_all,
                step_ts=step_ts,
                ts=group.ts,
                init_snap=init_snap,
            )
        saved.update(part)

    stats = SolverInfo(solver="euler", num_steps=steps, dense=False) if solver_stats else None
    return SolveOutput(saved=saved, stats=stats, dense=None)
