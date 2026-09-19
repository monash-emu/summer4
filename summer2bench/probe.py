"""Prove float64 fixed-step measurement on one unstratified SIR.

Enables ``jax_enable_x64`` before importing summer2. Twenty steps, ``dt=0.1``,
both ``euler`` and ``rk4``. Flow outputs are requested before ``get_runner``.
"""

from __future__ import annotations

import importlib.metadata
import math
import sys
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

from summer2 import CompartmentalModel  # noqa: E402
from time_run import time_runner  # noqa: E402

PARAMETERS: dict[str, float] = {}
SOLVERS = ("euler", "rk4")
STEPS = 20
DT = 0.1


def build_sir(solver: str) -> Any:
    """Build S/I/R, request both flows, then finalise via ``get_runner``.

    ``solver`` is a backend keyword, not a positional argument.
    """
    model = CompartmentalModel(
        times=(0.0, STEPS * DT),
        compartments=["S", "I", "R"],
        infectious_compartments=["I"],
        timestep=DT,
    )
    model.set_initial_population(distribution={"S": 999_990.0, "I": 10.0})
    model.add_infection_frequency_flow("infection", 0.35, "S", "I")
    model.add_transition_flow("recovery", 0.1, "I", "R")
    model.request_output_for_flow("infection", "infection")
    model.request_output_for_flow("recovery", "recovery")
    return model.get_runner(
        PARAMETERS,
        include_full_outputs=True,
        solver=solver,
    )


def _reduced(runner: Any, seen: dict[str, Any]) -> Any:
    """Sum compartments and both flow series. Stash the arrays for printing."""
    results = runner.function(parameters=PARAMETERS)
    outputs = results["outputs"]
    derived = results["derived_outputs"]
    infection = derived["infection"]
    recovery = derived["recovery"]
    reduced = outputs.sum() + infection.sum() + recovery.sum()
    seen["outputs"] = outputs
    seen["infection"] = infection
    seen["recovery"] = recovery
    seen["reduced"] = reduced
    return reduced


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return type(value).__name__
    name = getattr(dtype, "name", None)
    return str(name if name is not None else dtype)


def _report(solver: str, label: str, value: Any) -> None:
    print(
        f"{solver} {label} dtype={_dtype_name(value)} shape={getattr(value, 'shape', None)}",
        flush=True,
    )


def _require_float64(solver: str, label: str, value: Any) -> None:
    name = _dtype_name(value)
    if name != "float64":
        raise SystemExit(f"{solver} {label} dtype is {name}, expected float64")


def main() -> None:
    """Run both solvers and print dtypes, shapes, and whether the sums are finite."""
    print(f"summerepi2 {importlib.metadata.version('summerepi2')}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    print(f"jax_enable_x64 {jax.config.jax_enable_x64}", flush=True)

    for solver in SOLVERS:
        seen: dict[str, Any] = {}

        def call(runner: Any, box: dict[str, Any] = seen) -> Any:
            return _reduced(runner, box)

        runner, _timed = time_runner(lambda solver=solver: build_sir(solver), call, n_warm=1)
        _report(solver, "compartments", seen["outputs"])
        _report(solver, "infection", seen["infection"])
        _report(solver, "recovery", seen["recovery"])
        _require_float64(solver, "compartments", seen["outputs"])
        _require_float64(solver, "infection", seen["infection"])
        _require_float64(solver, "recovery", seen["recovery"])

        reduced = float(seen["reduced"])
        finite = math.isfinite(reduced)
        print(f"{solver} reduced {reduced!r} finite={finite}", flush=True)
        if not finite:
            raise SystemExit(f"{solver} reduced sum is not finite")

        # Inspection path the handoff records. Not part of the warm timer.
        runner.run(PARAMETERS)
        _report(solver, "model.outputs", runner.model.outputs)
        for flow_name, series in runner.model.derived_outputs.items():
            _report(solver, f"model.derived_outputs[{flow_name}]", series)
            _require_float64(solver, f"model.derived_outputs[{flow_name}]", series)
        _require_float64(solver, "model.outputs", runner.model.outputs)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
