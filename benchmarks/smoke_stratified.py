"""Smoke the stratified summer4 ports at the spec's smoke step count.

``age_mix``, ``age_mix_tv``, and ``stress``. Both Diffrax solvers, one warm
call each. Does not run 2_000 or 8_000 steps.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import diffrax
import jax

from diffrax_rk4 import ClassicalRK4
from harness import time_run
from models import (
    EXPECTED_COMPARTMENTS,
    STRATIFIED_NAMES,
    TV_MIXING_PARAM,
    build_model,
    fixed_step_run_kwargs,
    flow_series,
    load_spec,
    reduce_result,
)

SOLVERS: tuple[tuple[str, Any], ...] = (
    ("diffrax-euler", diffrax.Euler()),
    ("diffrax-rk4", ClassicalRK4()),
)


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return type(value).__name__
    name = getattr(dtype, "name", None)
    return str(name if name is not None else dtype)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _require_float64(label: str, value: Any) -> None:
    name = _dtype_name(value)
    if name != "float64":
        _fail(f"{label} dtype is {name}, expected float64")


def main() -> None:
    """Smoke every stratified model on both Diffrax fixed-step solvers."""
    spec = load_spec()
    steps = int(spec["smoke_steps"])
    if steps not in spec["step_counts"]:
        _fail(f"smoke_steps {steps} is not in step_counts {spec['step_counts']}")
    flow_names = ("infection", "recovery")
    rows = steps + 1

    print(f"jax {jax.__version__}", flush=True)
    print(f"jax_enable_x64 {jax.config.jax_enable_x64}", flush=True)
    print(f"smoke_steps {steps} rows {rows}", flush=True)

    for name in STRATIFIED_NAMES:
        for solver_label, solver in SOLVERS:
            label = f"{name} {solver_label}"
            seen: dict[str, Any] = {}

            def build(model_name: str = name) -> Any:
                return build_model(model_name, spec)

            def call(
                built: Any,
                solver_inst: Any = solver,
                stash: dict[str, Any] = seen,
            ) -> Any:
                kwargs = fixed_step_run_kwargs(built, solver_inst, steps, spec=spec)
                result = built.compiled.run(built.parameters, built.y0, **kwargs)
                stash["result"] = result
                stash["built"] = built
                stash["compartments"] = result["compartments"].values.data
                stash["flows"] = {flow: flow_series(result, flow) for flow in flow_names}
                return reduce_result(result, flow_names)

            built, timed = time_run(build, call, n_warm=1)
            print(
                f"{label} times build={timed.build_s:.3f}s "
                f"compile={timed.compile_s:.3f}s warm={timed.warm_s[0]:.3f}s",
                flush=True,
            )
            print(f"{label} compartments {built.n_compartments}", flush=True)
            if built.n_compartments != EXPECTED_COMPARTMENTS[name]:
                _fail(
                    f"{label} compartment count {built.n_compartments}, "
                    f"expected {EXPECTED_COMPARTMENTS[name]}"
                )
            compartments = seen["compartments"]
            _require_float64(f"{label} compartments", compartments)
            expected_shape = (rows, built.n_compartments)
            if tuple(compartments.shape) != expected_shape:
                _fail(f"{label} compartments shape {compartments.shape}, expected {expected_shape}")
            print(
                f"{label} compartments dtype={_dtype_name(compartments)} "
                f"shape={tuple(compartments.shape)}",
                flush=True,
            )
            for flow_name in flow_names:
                series = seen["flows"][flow_name]
                _require_float64(f"{label} {flow_name}", series)
                if tuple(series.shape) != (rows,):
                    _fail(f"{label} {flow_name} shape {series.shape}, expected {(rows,)}")
                print(
                    f"{label} {flow_name} dtype={_dtype_name(series)} "
                    f"shape={tuple(series.shape)}",
                    flush=True,
                )
            reduced = float(reduce_result(seen["result"], flow_names))
            if not math.isfinite(reduced):
                _fail(f"{label} reduced sum is not finite: {reduced!r}")
            print(f"{label} reduced {reduced!r} finite=True", flush=True)
            if name in ("age_mix_tv", "stress") and solver_label == SOLVERS[0][0]:
                table = built.parameters[TV_MIXING_PARAM]
                print(
                    f"{label} {TV_MIXING_PARAM} shape={tuple(table.shape)} "
                    f"dtype={_dtype_name(table)}",
                    flush=True,
                )
                print(f"{label} infection_flows {built.infection_flows}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
