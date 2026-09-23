"""TB-scale synthetic model benchmarks (Kiribati shape, no real data).

Marked ``slow`` so push CI stays fast; ``pixi run test`` / ``test-all`` collect
this file. Numbers for the baseline table live in ``benchmarks/README.md``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from benchmarks.tb_scale import T0, T1, TbScaleModel, build_tb_scale_model, measure_jaxpr_eqns

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model() -> TbScaleModel:
    return build_tb_scale_model()


def test_tb_scale_shape(model: TbScaleModel) -> None:
    assert model.n_compartments == 160
    assert 30 <= model.n_flows <= 50
    assert 140 <= model.n_outputs <= 220
    assert model.save.ts is not None
    assert len(model.save.ts) == int(T1 - T0) + 1


def _wall(fn: Any, *, repeats: int = 2) -> float:
    """Median of ``repeats`` after one warm-up call.

    Always ``jax.block_until_ready`` on the result so async dispatch is not
    counted as wall time. Median (not best-of) so a single lucky sample does
    not flip fused vs looped A/B.
    """
    out = fn()
    jax.block_until_ready(out)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - start)
    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2:
        return samples[mid]
    return 0.5 * (samples[mid - 1] + samples[mid])


def _run_kwargs(model: TbScaleModel, solver: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "t0": T0,
        "t1": T1,
        "dt": 1.0,
        "save": model.save,
        "solver": solver,
    }
    if solver != "euler":
        # Default max_steps=4096 is known-short for stiff 185-year runs (step 12).
        kwargs["max_steps"] = 100_000
    return kwargs


def test_tb_scale_compile_jaxpr_and_solvers(model: TbScaleModel) -> None:
    """Record compile time, jaxpr sizes, wall time, and vmap×64 for euler/dopri5."""
    compile_s = _wall(lambda: build_tb_scale_model(), repeats=2)

    y0 = model.compiled.initial_state(model.compiled.prepare(model.params))
    prepared = model.compiled.prepare(model.params)
    vf_eqns = measure_jaxpr_eqns(model.compiled.vector_field, T0, y0, prepared)

    def run_with(solver: str, contact: object) -> Any:
        params = dict(model.params)
        params["contact_rate"] = contact
        result = model.compiled.run(params, **_run_kwargs(model, solver))
        return result["population"].values.data

    run_eqns = {
        "euler": measure_jaxpr_eqns(lambda c: run_with("euler", c), 0.35),
        "dopri5": measure_jaxpr_eqns(lambda c: run_with("dopri5", c), 0.35),
    }

    contacts = jnp.linspace(0.30, 0.40, 64)
    rows: dict[str, dict[str, float | int]] = {}
    for solver in ("euler", "dopri5"):

        def _one_run(s: str = solver) -> Any:
            return run_with(s, 0.35)

        wall_s = _wall(_one_run)

        def _batch(c: object, s: str = solver) -> Any:
            return run_with(s, c)

        vmap_fn = jax.vmap(_batch)

        def _vmap_once(fn: Any = vmap_fn) -> Any:
            return fn(contacts)

        vmap_s = _wall(_vmap_once)
        stacked = vmap_fn(contacts)
        assert stacked.shape == (64, int(T1 - T0) + 1, model.n_compartments)
        assert bool(np.isfinite(np.asarray(stacked)).all())

        one = model.compiled.run(model.params, **_run_kwargs(model, solver))

        def _eval_once(result: Any = one) -> Any:
            return model.outputs.evaluate(result, model.params)

        eval_s = _wall(_eval_once)
        named = model.outputs.evaluate(one, model.params)
        assert len(named.outputs) == model.n_outputs
        pop = np.asarray(named["population"].values)
        assert pop.shape == (int(T1 - T0) + 1,)
        assert bool(np.isfinite(pop).all())

        rows[solver] = {
            "compile_s": compile_s,
            "vf_eqns": vf_eqns,
            "run_eqns": run_eqns[solver],
            "wall_s": wall_s,
            "vmap64_s": vmap_s,
            "evaluate_s": eval_s,
        }

    assert rows["euler"]["wall_s"] > 0.0
    assert rows["dopri5"]["wall_s"] > 0.0
    assert vf_eqns > 100

    print("\n# TB-scale baseline (this machine)\n")
    print(
        "| Solver | Build+compile (s) | VF eqns | Run jaxpr eqns | "
        "Wall / run (s) | vmap×64 (s) | OutputSet evaluate (s) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for solver, row in rows.items():
        print(
            f"| `{solver}` | {row['compile_s']:.3f} | {row['vf_eqns']} | "
            f"{row['run_eqns']} | {row['wall_s']:.3f} | {row['vmap64_s']:.3f} | "
            f"{row['evaluate_s']:.3f} |"
        )
    print(
        f"\nModel: {model.n_compartments} compartments, {model.n_flows} flows, "
        f"{model.n_outputs} outputs, saves {T0:.0f}–{T1:.0f} yearly.\n"
    )


def _count_vf_ops(cm: Any, t0: float, y0: Any, prepared: Any) -> tuple[int, int, int]:
    """Return ``(eqns, scatter-add, gather)`` for ``cm.vector_field``."""
    jaxpr = jax.make_jaxpr(cm.vector_field)(t0, y0, prepared)
    names = [e.primitive.name for e in jaxpr.jaxpr.eqns]
    return len(names), names.count("scatter-add"), names.count("gather")


def _sir_fuse_models() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Small multi-flow SIR used in fuse unit tests / notebook."""
    from summer4 import (
        Compartments,
        EntryFlow,
        Everything,
        ExitFlow,
        FlowModel,
        Param,
        Property,
        PropertyMap,
        SavePlan,
        SaveRequest,
        TransitionFlow,
    )

    state = Property("state", ("S", "I", "R"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], Param("infection")))
    death = model.add_flow(ExitFlow("death", Everything(), 0.01))
    model.add_flow(EntryFlow("birth", state["S"], death.sum()))
    model.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.05))
    fused = model.compile(fuse_compartment_updates=True)
    looped = model.compile(fuse_compartment_updates=False)
    y0 = jnp.asarray([900.0, 80.0, 20.0])
    save = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    run_kw: dict[str, Any] = {
        "t0": 0.0,
        "t1": 40.0,
        "dt": 1.0,
        "save": save,
        "solver": "dopri5",
        "rtol": 1e-5,
        "atol": 1e-7,
    }
    return fused, looped, y0, run_kw


def _time_modes(
    *,
    name: str,
    build_loss: Any,
    theta0: Any,
    vf_meta: tuple[tuple[str, int, int, int], tuple[str, int, int, int]] | None = None,
) -> None:
    """Print solve / grad / value_and_grad walls for fused vs looped.

    Times each op by alternating fused and looped calls (median of 8) so
    order bias and async dispatch do not dominate the A/B.
    """
    loss_f = build_loss(True)
    loss_l = build_loss(False)
    v_f = float(jax.jit(loss_f)(theta0))
    v_l = float(jax.jit(loss_l)(theta0))
    np.testing.assert_allclose(v_f, v_l, rtol=1e-4, atol=0.0)

    pairs = {
        "solve_s": (jax.jit(loss_f), jax.jit(loss_l)),
        "grad_s": (jax.jit(jax.grad(loss_f)), jax.jit(jax.grad(loss_l))),
        "vag_s": (jax.jit(jax.value_and_grad(loss_f)), jax.jit(jax.value_and_grad(loss_l))),
    }
    for ff, fl in pairs.values():
        jax.block_until_ready(ff(theta0))
        jax.block_until_ready(fl(theta0))

    rows: dict[str, dict[str, Any]] = {
        "True": {"mode": "True"},
        "False": {"mode": "False"},
    }
    if vf_meta is not None:
        for label, eqns, scat, gath in vf_meta:
            rows[label]["vf_eqns"] = eqns
            rows[label]["scat"] = scat
            rows[label]["gath"] = gath

    for key, (ff, fl) in pairs.items():
        samples_f: list[float] = []
        samples_l: list[float] = []
        for i in range(8):
            if i % 2 == 0:
                start = time.perf_counter()
                jax.block_until_ready(ff(theta0))
                samples_f.append(time.perf_counter() - start)
                start = time.perf_counter()
                jax.block_until_ready(fl(theta0))
                samples_l.append(time.perf_counter() - start)
            else:
                start = time.perf_counter()
                jax.block_until_ready(fl(theta0))
                samples_l.append(time.perf_counter() - start)
                start = time.perf_counter()
                jax.block_until_ready(ff(theta0))
                samples_f.append(time.perf_counter() - start)
        samples_f.sort()
        samples_l.sort()
        mid = len(samples_f) // 2
        rows["True"][key] = 0.5 * (samples_f[mid - 1] + samples_f[mid])
        rows["False"][key] = 0.5 * (samples_l[mid - 1] + samples_l[mid])

    ordered = [rows["True"], rows["False"]]
    print(f"\n# fuse_compartment_updates A/B — {name} (this machine)\n")
    print("(median of 8 interleaved samples; jax.block_until_ready)\n")
    if vf_meta is not None:
        print(
            "| fuse_compartment_updates | VF eqns | scatter-add | gather | "
            "solve (s) | grad (s) | value_and_grad (s) |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in ordered:
            print(
                f"| `{row['mode']}` | {row['vf_eqns']} | {row['scat']} | {row['gath']} | "
                f"{row['solve_s']:.4f} | {row['grad_s']:.4f} | {row['vag_s']:.4f} |"
            )
    else:
        print("| fuse_compartment_updates | solve (s) | grad (s) | value_and_grad (s) |")
        print("| --- | ---: | ---: | ---: |")
        for row in ordered:
            print(
                f"| `{row['mode']}` | {row['solve_s']:.4f} | "
                f"{row['grad_s']:.4f} | {row['vag_s']:.4f} |"
            )
    fused_row, looped_row = ordered
    for key, label in (
        ("solve_s", "solve"),
        ("grad_s", "grad"),
        ("vag_s", "value_and_grad"),
    ):
        a, b = fused_row[key], looped_row[key]
        if a > 0:
            print(f"{label} speedup (looped/fused): {b / a:.3f}×")
    print()


def test_fuse_compartment_updates_timing_ab(model: TbScaleModel) -> None:
    """A/B wall times: SIR + TB-scale, each with solve / grad / value_and_grad."""
    # --- SIR (short dopri5, grad w.r.t. infection rate) ---
    fused_sir, looped_sir, y0_sir, sir_kw = _sir_fuse_models()
    y0_sir_arr = np.asarray(y0_sir)
    eq_f, scat_f, gath_f = _count_vf_ops(fused_sir, 0.0, y0_sir_arr, {"infection": 0.1})
    eq_l, scat_l, gath_l = _count_vf_ops(looped_sir, 0.0, y0_sir_arr, {"infection": 0.1})
    scat_sir = (
        ("True", eq_f, scat_f, gath_f),
        ("False", eq_l, scat_l, gath_l),
    )
    assert scat_f < scat_l

    def sir_loss_builder(fuse: bool) -> Any:
        cm = fused_sir if fuse else looped_sir

        def loss(infection: Any) -> Any:
            res = cm.run({"infection": infection}, y0_sir, **sir_kw)
            return jnp.sum(res["compartments"].values.data**2)

        return loss

    _time_modes(
        name="SIR (4 flows, dopri5 t=0..40)",
        build_loss=sir_loss_builder,
        theta0=jnp.asarray(0.1),
        vf_meta=scat_sir,
    )

    # --- TB-scale (full horizon dopri5, grad w.r.t. contact_rate) ---
    looped = build_tb_scale_model(fuse_compartment_updates=False)
    y0_f = model.compiled.initial_state(model.compiled.prepare(model.params))
    prep_f = model.compiled.prepare(model.params)
    y0_l = looped.compiled.initial_state(looped.compiled.prepare(looped.params))
    prep_l = looped.compiled.prepare(looped.params)

    def _dy(cm: Any, y0: Any, prepared: Any) -> np.ndarray:
        out = cm.vector_field(T0, y0, prepared)
        return np.asarray(out.data if hasattr(out, "data") else out)

    np.testing.assert_allclose(
        _dy(model.compiled, y0_f, prep_f),
        _dy(looped.compiled, y0_l, prep_l),
        rtol=1e-10,
        atol=1e-10,
    )

    eq_f, scat_f, gath_f = _count_vf_ops(model.compiled, T0, y0_f, prep_f)
    eq_l, scat_l, gath_l = _count_vf_ops(looped.compiled, T0, y0_l, prep_l)
    scat_tb = (
        ("True", eq_f, scat_f, gath_f),
        ("False", eq_l, scat_l, gath_l),
    )
    assert scat_f < scat_l

    run_kw_f = _run_kwargs(model, "dopri5")
    run_kw_l = _run_kwargs(looped, "dopri5")
    base_params = dict(model.params)

    def tb_loss_builder(fuse: bool) -> Any:
        cm = model.compiled if fuse else looped.compiled
        kw = run_kw_f if fuse else run_kw_l

        def loss(contact: Any) -> Any:
            params = dict(base_params)
            params["contact_rate"] = contact
            res = cm.run(params, **kw)
            return jnp.sum(res["population"].values.data**2)

        return loss

    _time_modes(
        name=(
            f"TB-scale ({model.n_compartments} comps, {model.n_flows} flows, "
            f"dopri5 {T0:.0f}–{T1:.0f})"
        ),
        build_loss=tb_loss_builder,
        theta0=jnp.asarray(0.35),
        vf_meta=scat_tb,
    )
