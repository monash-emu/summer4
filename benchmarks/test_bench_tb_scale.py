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
    """Best of ``repeats`` after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


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
