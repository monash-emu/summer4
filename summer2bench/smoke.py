"""Smoke ``sir``, ``sir_adjust``, and ``sir_tv`` at the spec's smoke step count.

Both fixed-step solvers. One warm call each. Does not time the full ladder
and does not build age or stress models.
"""

from __future__ import annotations

import importlib.metadata
import math
import sys
from typing import Any

import jax

from models import (
    MODEL_NAMES,
    host_callback_primitives,
    inspect_multiply_chain,
    assert_knots_clamped,
    load_spec,
    prepare_model,
    runner_for,
)
from time_run import reduce_runner_outputs, time_runner

SOLVERS = ("euler", "rk4")


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


def _placement(runner: Any, flow_name: str) -> str:
    """Where the infection rate sits after ``get_runner`` splits the graph.

    A timestep node that is only ``static_inputs[...]`` is an alias of the
    frozen value, not a second evaluation inside the step.
    """
    flow = next(item for item in runner.model.flows if item.name == flow_name)
    key = flow._graph_key
    parts: list[str] = []
    for label in ("static_cg", "timestep_cg"):
        graph = runner.impl_dict[label]
        if key not in graph.dag.nodes:
            continue
        node = graph[key]
        kind = type(node).__name__
        if kind == "Variable" and "static_inputs" in repr(node):
            kind = "static_inputs"
        parts.append(f"{label}:{kind}")
    if not parts:
        parts.append("absent")
    return f"{key} {', '.join(parts)}"


def _check_saved(
    label: str,
    seen: dict[str, Any],
    flow_names: tuple[str, ...],
    rows: int,
    compartments: int,
) -> None:
    outputs = seen["outputs"]
    _require_float64(f"{label} compartments", outputs)
    if tuple(outputs.shape) != (rows, compartments):
        _fail(f"{label} compartments shape {outputs.shape}, expected {(rows, compartments)}")
    if tuple(seen["derived_keys"]) != flow_names:
        _fail(f"{label} derived keys {seen['derived_keys']}, expected {flow_names}")
    for flow_name in flow_names:
        series = seen["flows"][flow_name]
        _require_float64(f"{label} {flow_name}", series)
        if tuple(series.shape) != (rows,):
            _fail(f"{label} {flow_name} shape {series.shape}, expected {(rows,)}")
    reduced = float(seen["reduced"])
    if not math.isfinite(reduced):
        _fail(f"{label} reduced sum is not finite: {reduced!r}")
    print(
        f"{label} compartments dtype={_dtype_name(outputs)} shape={tuple(outputs.shape)}",
        flush=True,
    )
    for flow_name in flow_names:
        series = seen["flows"][flow_name]
        print(
            f"{label} {flow_name} dtype={_dtype_name(series)} shape={tuple(series.shape)}",
            flush=True,
        )
    print(f"{label} reduced {reduced!r} finite=True", flush=True)


def _check_adjust_chain(
    label: str,
    prepared_contact: Any,
    spec: dict[str, Any],
    runner: Any,
    parameters: dict[str, float],
) -> None:
    chain = inspect_multiply_chain(prepared_contact)
    expected = len(spec["adjustments"])
    if chain.count != expected:
        _fail(f"{label} multiply nodes {chain.count}, expected {expected}")
    print(
        f"{label} multiply_nodes={chain.count} funcs={','.join(chain.functions)}",
        flush=True,
    )
    print(f"{label} breakpoint {chain.breakpoint}", flush=True)
    closed = jax.make_jaxpr(lambda params: runner.function(params))(parameters)
    callbacks = host_callback_primitives(closed)
    if callbacks:
        _fail(f"{label} jaxpr host callbacks {callbacks}")
    print(f"{label} jaxpr host_callback=False", flush=True)


def main() -> None:
    """Smoke every unstratified model on both solvers."""
    spec = load_spec()
    steps = int(spec["smoke_steps"])
    if steps not in spec["step_counts"]:
        _fail(f"smoke_steps {steps} is not in step_counts {spec['step_counts']}")
    flow_names = tuple(str(name) for name in spec["flows"])
    rows = steps + 1
    n_compartments = len(spec["compartments"])

    print(f"summerepi2 {importlib.metadata.version('summerepi2')}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    print(f"jax_enable_x64 {jax.config.jax_enable_x64}", flush=True)
    print(f"smoke_steps {steps} rows {rows}", flush=True)

    for name in MODEL_NAMES:
        for solver in SOLVERS:
            label = f"{name} {solver}"
            seen: dict[str, Any] = {}
            box: dict[str, Any] = {}

            def build(model_name: str = name, solver_name: str = solver) -> Any:
                prepared = prepare_model(model_name, steps, spec)
                box["prepared"] = prepared
                return runner_for(prepared, solver_name)

            def call(runner: Any, stash: dict[str, Any] = seen, held: dict[str, Any] = box) -> Any:
                return reduce_runner_outputs(runner, held["prepared"].parameters, flow_names, stash)

            runner, timed = time_runner(build, call, n_warm=1)
            print(
                f"{label} times build={timed.build_s:.3f}s "
                f"compile={timed.compile_s:.3f}s warm={timed.warm_s[0]:.3f}s",
                flush=True,
            )
            _check_saved(label, seen, flow_names, rows, n_compartments)
            names = [compartment.name for compartment in runner.model.compartments]
            if names != list(spec["compartments"]):
                _fail(f"{label} compartments {names}")
            consumed = sorted(runner.model.get_input_parameters())
            print(f"{label} parameters {consumed}", flush=True)
            print(f"{label} placement {_placement(runner, 'infection')}", flush=True)
            if name == "sir_adjust" and solver == SOLVERS[0]:
                _check_adjust_chain(
                    label,
                    box["prepared"].contact,
                    spec,
                    runner,
                    box["prepared"].parameters,
                )
            if name == "sir_tv" and solver == SOLVERS[0]:
                assert_knots_clamped(box["prepared"].contact, spec)
                print(f"{label} knots clamped", flush=True)
                closed = jax.make_jaxpr(lambda params: runner.function(params))(
                    box["prepared"].parameters
                )
                callbacks = host_callback_primitives(closed)
                if callbacks:
                    _fail(f"{label} jaxpr host callbacks {callbacks}")
                print(f"{label} jaxpr host_callback=False", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
