"""Step R0 benchmark gate for ``plans/runtime-defaults.plan.md``.

Measures, before any default changes:

* ``solver``: Euler vs ``Diffrax(Tsit5, PIDController)`` at two tolerances, on the guide-01 SIR,
  the guide-09 Lotka-Volterra and the tb-scale model: wall seconds per ``run()`` after compile,
  error against a ``dopri5`` reference at 1e-12, solver step counts, and an SIR ``jax.grad``;
* ``devices``: one SIR ``run()`` and one tb-scale ``run()`` with the host device count at 1 and at
  the core count, each in its own process (the count is fixed when JAX initialises);
* ``chains``: the guide-08 calibration with 4 chains, ``"vectorized"`` vs ``"parallel"``, with the
  device count at the core count.

Usage::

    pixi run python benchmarks/runtime_defaults.py --part all \
        --out benchmarks/results/runtime_defaults.json
    pixi run python benchmarks/runtime_defaults.py --part all --smoke --out /tmp/rd_smoke.json

``--part devices-child`` / ``chains-child`` are internal: they run inside a subprocess whose
``XLA_FLAGS`` the parent set, and print one JSON object on their last stdout line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The repo root must be importable when run as a script (benchmarks/ is not a package here).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import summer4  # noqa: E402,F401  (sets float64 before JAX is used)
from summer4 import (  # noqa: E402
    Compartments,
    EntryFlow,
    ExitFlow,
    FlowMass,
    FlowModel,
    FlowRef,
    OutputSet,
    Param,
    Property,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TransitionFlow,
)
from summer4.epi import BayesianModel, ForceOfInfection, Poisson, Uniform  # noqa: E402

SOLVER_LABELS: tuple[str, ...] = ("euler", "default", "tsit5_loose")
MODELS: tuple[str, ...] = ("sir", "lotka_volterra", "tb_scale")


# --------------------------------------------------------------------------------------
# Models (copied from docs/guide notebooks 01, 08 and 09; the tb-scale builder is reused)
# --------------------------------------------------------------------------------------


class Case:
    """A compiled model plus the arguments of one ``run()`` and its save plan."""

    def __init__(
        self,
        name: str,
        compiled: Any,
        params: dict[str, Any],
        run_kwargs: dict[str, Any],
        plan: SavePlan,
        non_euler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.compiled = compiled
        self.params = params
        self.run_kwargs = run_kwargs
        self.plan = plan
        self.non_euler_kwargs = non_euler_kwargs or {}

    def run(self, solver: Any, **extra: Any) -> Any:
        kwargs = dict(self.run_kwargs)
        if not (isinstance(solver, str) and solver == "euler"):
            kwargs.update(self.non_euler_kwargs)
        kwargs.update(extra)
        return self.compiled.run(self.params, save=self.plan, solver=solver, **kwargs)


def build_sir() -> Case:
    """Guide 01: S -> I -> R, dt 0.1 over 100 days."""
    state = Property("state", ("S", "I", "R"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            ForceOfInfection(
                "infection", infectious=state["I"], contact_rate=Param("contact_rate")
            ),
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery_rate")))
    model.set_initial_population({state["S"]: 990.0, state["I"]: 10.0})
    plan = SavePlan(
        requests={"compartments": SaveRequest(Compartments())}, ts=np.linspace(0.0, 100.0, 101)
    )
    return Case(
        "sir",
        model.compile(),
        {"contact_rate": 0.5, "recovery_rate": 0.2},
        {"t0": 0.0, "t1": 100.0, "dt": 0.1},
        plan,
    )


def build_lotka_volterra() -> Case:
    """Guide 09: prey and predators, with ``Reduce(where=...)`` head counts."""
    species = Property("species", ("prey", "predator"))
    model = FlowModel(PropertyMap.from_property(species))
    model.add_flow(
        EntryFlow("prey_birth", species["prey"], Param("alpha") * Reduce(where=species["prey"]))
    )
    model.add_flow(ExitFlow("prey_death", species["prey"], Param("mu")))
    model.add_flow(
        ExitFlow("predation", species["prey"], Param("a") * Reduce(where=species["predator"]))
    )
    model.add_flow(
        EntryFlow("predator_birth", species["predator"], Param("e") * FlowRef("predation").sum_by())
    )
    model.add_flow(ExitFlow("predator_death", species["predator"], Param("d")))
    model.set_initial_population({species["prey"]: 40.0, species["predator"]: 15.0})
    plan = SavePlan(
        requests={"compartments": SaveRequest(Compartments())}, ts=np.linspace(0.0, 100.0, 401)
    )
    return Case(
        "lotka_volterra",
        model.compile(),
        {"alpha": 0.6, "mu": 0.1, "a": 0.02, "e": 0.5, "d": 0.25},
        {"t0": 0.0, "t1": 100.0, "dt": 0.1},
        plan,
    )


def build_tb() -> Case:
    """The tb-scale benchmark model: 160 compartments, yearly saves 1850-2035, dt 1."""
    from tb_scale import SAVE_TIMES, T0, T1, build_tb_scale_model

    tb = build_tb_scale_model()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())}, ts=SAVE_TIMES)
    return Case(
        "tb_scale",
        tb.compiled,
        dict(tb.params),
        {"t0": T0, "t1": T1, "dt": 1.0},
        plan,
        non_euler_kwargs={"max_steps": 100_000},
    )


def build_case(name: str) -> Case:
    builders: dict[str, Callable[[], Case]] = {
        "sir": build_sir,
        "lotka_volterra": build_lotka_volterra,
        "tb_scale": build_tb,
    }
    return builders[name]()


# --------------------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------------------


def solver_configs() -> dict[str, Any]:
    import diffrax

    from summer4.solvers import Diffrax

    return {
        "euler": "euler",
        "default": Diffrax(
            diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9)
        ),
        "tsit5_loose": Diffrax(
            diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7)
        ),
    }


def _block(value: Any) -> Any:
    import jax

    return jax.block_until_ready(value)


def compartments_of(result: Any) -> np.ndarray:
    return np.asarray(result["compartments"].values.data, dtype=np.float64)


def time_calls(fn: Callable[[], Any], repeats: int) -> dict[str, float]:
    """Warm up once (compile), then time ``repeats`` calls; return median and spread."""
    t0 = time.perf_counter()
    _block(fn())
    first = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(fn())
        samples.append(time.perf_counter() - t0)
    return {
        "first_call_s": first,
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "repeats": float(repeats),
    }


def step_counts(result: Any) -> dict[str, int | None]:
    info = getattr(result, "solver", None)
    if info is None:
        return {"num_steps": None, "num_accepted_steps": None, "num_rejected_steps": None}
    out: dict[str, int | None] = {}
    for key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
        value = getattr(info, key, None)
        out[key] = None if value is None else int(np.asarray(value))
    return out


def _reference(case: Case) -> np.ndarray:
    """dopri5 at 1e-12 on the case's save grid."""
    kwargs = {"rtol": 1e-12, "atol": 1e-12}
    kwargs["max_steps"] = 1_000_000
    return compartments_of(case.run("dopri5", **kwargs))


# --------------------------------------------------------------------------------------
# Part: solver
# --------------------------------------------------------------------------------------


def sir_grad_fn(case: Case, solver: Any, truth: np.ndarray) -> Callable[[Any], Any]:
    """jit(grad(sum of squared prevalence error)) with respect to the contact rate."""
    import jax
    import jax.numpy as jnp

    truth_j = jnp.asarray(truth)

    def loss(contact_rate: Any) -> Any:
        params = dict(case.params)
        params["contact_rate"] = contact_rate
        kwargs = dict(case.run_kwargs)
        res = case.compiled.run(params, save=case.plan, solver=solver, **kwargs)
        infectious = res["compartments"].values.data[:, 1]
        return jnp.sum((infectious - truth_j) ** 2) / truth_j.size

    return jax.jit(jax.grad(loss))


def part_solver(smoke: bool) -> dict[str, Any]:
    repeats = 2 if smoke else 7
    configs = solver_configs()
    out: dict[str, Any] = {"models": {}, "grad": {}}
    names = ("sir",) if smoke else MODELS
    if smoke:
        names = ("sir", "lotka_volterra")
    for name in names:
        case = build_case(name)
        ref = _reference(case)
        scale = np.maximum(np.abs(ref), 1e-12)
        rows: dict[str, Any] = {}
        for label in SOLVER_LABELS:
            solver = configs[label]
            timing = time_calls(lambda c=case, s=solver: c.run(s), repeats)
            result = case.run(solver)
            comp = compartments_of(result)
            row: dict[str, Any] = dict(timing)
            row["max_abs_error"] = float(np.max(np.abs(comp - ref)))
            row["max_rel_error"] = float(np.max(np.abs(comp - ref) / scale))
            row["ok"] = None
            info = getattr(result, "solver", None)
            if info is not None:
                try:
                    row["ok"] = bool(np.asarray(info.ok))
                except Exception:  # pragma: no cover - defensive
                    row["ok"] = None
            row.update(step_counts(result))
            rows[label] = row
        out["models"][name] = rows
        if name == "sir":
            truth = ref[:, 1]
            grad_rows: dict[str, Any] = {}
            for label in ("euler", "default"):
                fn = sir_grad_fn(case, configs[label], truth)
                grad_rows[label] = time_calls(lambda f=fn: f(0.45), repeats)
            out["grad"] = grad_rows
    return out


# --------------------------------------------------------------------------------------
# Part: devices (children)
# --------------------------------------------------------------------------------------


def child_env(device_count: int) -> dict[str, str]:
    env = dict(os.environ)
    flag = f"--xla_force_host_platform_device_count={device_count}"
    existing = " ".join(
        tok
        for tok in env.get("XLA_FLAGS", "").split()
        if not tok.startswith("--xla_force_host_platform_device_count")
    )
    env["XLA_FLAGS"] = f"{existing} {flag}".strip()
    env["SUMMER4_HOST_DEVICES"] = "0"  # a later runtime-defaults step must not override us
    return env


def run_child(part: str, device_count: int, smoke: bool) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--part", part]
    if smoke:
        cmd.append("--smoke")
    proc = subprocess.run(
        cmd, env=child_env(device_count), capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{part} child failed ({device_count} devices):\n{proc.stderr[-3000:]}")
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(last)  # type: ignore[no-any-return]


def child_devices(smoke: bool) -> dict[str, Any]:
    import jax

    repeats = 2 if smoke else 7
    configs = solver_configs()
    out: dict[str, Any] = {"device_count": jax.device_count(), "models": {}}
    names = ("sir",) if smoke else ("sir", "tb_scale")
    for name in names:
        case = build_case(name)
        rows = {}
        for label in ("euler", "default"):
            rows[label] = time_calls(lambda c=case, s=configs[label]: c.run(s), repeats)
        out["models"][name] = rows
    return out


def part_devices(smoke: bool) -> dict[str, Any]:
    ncpu = os.cpu_count() or 1
    single = run_child("devices-child", 1, smoke)
    forced = run_child("devices-child", ncpu, smoke)
    ratios: dict[str, dict[str, float]] = {}
    for name, rows in single["models"].items():
        ratios[name] = {
            label: forced["models"][name][label]["median_s"] / rows[label]["median_s"]
            for label in rows
        }
    return {"single": single, "forced": forced, "ratio_forced_over_single": ratios}


# --------------------------------------------------------------------------------------
# Part: chains (child)
# --------------------------------------------------------------------------------------


def build_calibration() -> BayesianModel:
    """Guide 08: SIR, one parameter, Poisson target on daily new infections."""
    state = Property("state", ("S", "I", "R"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            ForceOfInfection("foi", infectious=state["I"], contact_rate=Param("contact_rate")),
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    model.set_initial_population({state["S"]: 9990.0, state["I"]: 10.0})
    cm = model.compile()
    run_kwargs = dict(t0=0.0, t1=60.0, dt=1.0)
    outputs = OutputSet()
    outputs["new_infections"] = FlowMass("infection").sum_by()
    true_params = {"contact_rate": 0.5}
    truth = outputs.evaluate(cm.run(true_params, save=outputs.plan(), **run_kwargs), true_params)[
        "new_infections"
    ]
    obs_days = np.arange(5.0, 41.0, 5.0)
    expected = np.asarray(truth.at_times(obs_days).values).reshape(-1)
    observed = np.random.default_rng(1).poisson(expected).astype(float)
    target = Target(
        key="new_infections", times=obs_days, values=observed, likelihood=Poisson("sum")
    )
    return BayesianModel(
        cm,
        {},
        [Uniform("contact_rate", 0.1, 1.5)],
        TargetSet((target,)),
        outputs=outputs,
        run_kwargs=run_kwargs,
    )


def diagnostics(run: Any) -> dict[str, float | None]:
    try:
        import arviz as az

        summ = az.summary(run.idata)
        return {
            "max_r_hat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()),
        }
    except Exception:  # pragma: no cover - diagnostics are optional
        return {"max_r_hat": None, "min_ess_bulk": None}


def child_chains(smoke: bool) -> dict[str, Any]:
    import jax

    warmup, samples = (50, 50) if smoke else (500, 500)
    bm = build_calibration()
    out: dict[str, Any] = {
        "device_count": jax.device_count(),
        "num_chains": 4,
        "num_warmup": warmup,
        "num_samples": samples,
        "methods": {},
    }
    for method in ("vectorized", "parallel"):
        row: dict[str, Any] = {}
        for tag, seed in (("first_call_s", 0), ("second_call_s", 1)):
            t0 = time.perf_counter()
            run = bm.sample(
                num_warmup=warmup,
                num_samples=samples,
                num_chains=4,
                seed=seed,
                chain_method=method,
            )
            _block(run.idata)
            row[tag] = time.perf_counter() - t0
        row.update(diagnostics(run))
        out["methods"][method] = row
    return out


def part_chains(smoke: bool) -> dict[str, Any]:
    return run_child("chains-child", os.cpu_count() or 1, smoke)


# --------------------------------------------------------------------------------------
# Gate and reporting
# --------------------------------------------------------------------------------------


def _criterion(name: str, value: float | None, limit: float, passed: bool | None) -> dict[str, Any]:
    return {"name": name, "value": value, "limit": limit, "pass": passed}


def evaluate_gate(results: dict[str, Any]) -> dict[str, Any]:
    """The plan's R0 gate; ``pass`` True means the criterion did not trip."""
    criteria: list[dict[str, Any]] = []
    solver = results.get("solver")
    if solver:
        for name, rows in solver["models"].items():
            ratio = rows["default"]["median_s"] / rows["euler"]["median_s"]
            criteria.append(
                _criterion(f"solver_default_over_euler[{name}]", ratio, 3.0, ratio <= 3.0)
            )
        if solver.get("grad"):
            g = solver["grad"]
            ratio = g["default"]["median_s"] / g["euler"]["median_s"]
            criteria.append(_criterion("grad_default_over_euler[sir]", ratio, 3.0, ratio <= 3.0))
    devices = results.get("devices")
    if devices:
        for name, by_label in devices["ratio_forced_over_single"].items():
            for label, ratio in by_label.items():
                criteria.append(
                    _criterion(f"forced_over_single[{name}/{label}]", ratio, 1.10, ratio <= 1.10)
                )
    chains = results.get("chains")
    if chains:
        m = chains["methods"]
        key = "second_call_s"
        par, vec = m["parallel"][key], m["vectorized"][key]
        needs = chains["device_count"] >= chains["num_chains"]
        criteria.append(
            _criterion(
                "parallel_over_vectorized_wall",
                par / vec,
                1.0,
                (par < vec) if needs else None,
            )
        )
    verdicts = [c["pass"] for c in criteria if c["pass"] is not None]
    return {"criteria": criteria, "all_pass": all(verdicts) if verdicts else None}


def print_tables(results: dict[str, Any]) -> None:
    solver = results.get("solver")
    if solver:
        print("\nSOLVER  (median s/run | max rel err | steps)")
        for name, rows in solver["models"].items():
            for label, r in rows.items():
                print(
                    f"  {name:15s} {label:12s} {r['median_s']:10.4f}  {r['max_rel_error']:10.2e}"
                    f"  steps={r['num_steps']}"
                )
        for label, r in solver.get("grad", {}).items():
            print(f"  {'sir grad':15s} {label:12s} {r['median_s']:10.4f}")
    devices = results.get("devices")
    if devices:
        print("\nDEVICES (forced / single median)")
        for name, by_label in devices["ratio_forced_over_single"].items():
            for label, ratio in by_label.items():
                print(f"  {name:15s} {label:12s} {ratio:8.3f}")
    chains = results.get("chains")
    if chains:
        print(f"\nCHAINS ({chains['device_count']} devices, {chains['num_chains']} chains)")
        for method, r in chains["methods"].items():
            print(
                f"  {method:11s} first {r['first_call_s']:8.2f}s second {r['second_call_s']:8.2f}s"
                f"  rhat={r['max_r_hat']} ess={r['min_ess_bulk']}"
            )
    gate = results.get("gate")
    if gate:
        print("\nGATE")
        for c in gate["criteria"]:
            value = "n/a" if c["value"] is None else f"{c['value']:.3f}"
            print(f"  {c['name']:50s} {value:>8s} (limit {c['limit']})  pass={c['pass']}")
        print(f"  all_pass={gate['all_pass']}")


def environment() -> dict[str, Any]:
    import jax
    import jaxlib

    versions: dict[str, str] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__}
    for mod in ("diffrax", "numpyro", "equinox", "arviz", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            versions[mod] = "unavailable"
    return {
        **versions,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": jax.default_backend(),
        "device_count_parent": jax.device_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--part",
        choices=["solver", "devices", "chains", "all", "devices-child", "chains-child"],
        default="all",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, for a quick check")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args()

    if args.part == "devices-child":
        print(json.dumps(child_devices(args.smoke)))
        return
    if args.part == "chains-child":
        print(json.dumps(child_chains(args.smoke)))
        return

    results: dict[str, Any] = {"smoke": args.smoke, "environment": environment()}
    if args.part in ("solver", "all"):
        results["solver"] = part_solver(args.smoke)
    if args.part in ("devices", "all"):
        results["devices"] = part_devices(args.smoke)
    if args.part in ("chains", "all"):
        results["chains"] = part_chains(args.smoke)
    results["gate"] = evaluate_gate(results)

    print_tables(results)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
