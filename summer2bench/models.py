"""Summer2 models driven only by ``spec.json``.

Import this module before any other ``summer2`` import. It enables float64
first, then imports summer2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from computegraph.types import Data, Function  # noqa: E402
from summer2 import CompartmentalModel, StrainStratification, Stratification  # noqa: E402
from summer2.functions import get_linear_interpolation_function  # noqa: E402
from summer2.functions.time import get_time_callable  # noqa: E402
from summer2.parameters import Parameter, Time  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent / "spec.json"
MODEL_NAMES = ("sir", "sir_adjust", "sir_tv")
STRATIFIED_NAMES = ("age_mix", "age_mix_tv", "stress")

_HOST_CALLBACKS = (
    "pure_callback",
    "io_callback",
    "debug_callback",
    "ffi_call",
    "host_callback",
)


@dataclass(frozen=True)
class PreparedModel:
    """A model whose flow outputs are already requested. ``get_runner`` is separate."""

    name: str
    model: CompartmentalModel
    parameters: dict[str, float]
    contact: Any
    mixing: Any = None


@dataclass(frozen=True)
class MultiplyChain:
    """What a breakpoint on the adjustment chain actually is."""

    count: int
    functions: tuple[str, ...]
    breakpoint: str


def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the shared numeric spec. Model code must not restate those numbers."""
    spec_path = SPEC_PATH if path is None else path
    with spec_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"{spec_path} must contain a JSON object")
    return loaded


def adjustment_parameter_key(index: int) -> str:
    """Parameter name for one factor in spec order. The index is the list position."""
    return f"adjustment_{index}"


def parameter_values(name: str, spec: dict[str, Any]) -> dict[str, float]:
    """Values ``get_runner`` should be given. Unused keys are dropped by summer2."""
    if name not in MODEL_NAMES and name not in STRATIFIED_NAMES:
        raise KeyError(name)
    values = {"recovery": float(spec["recovery"])}
    if name not in ("sir_tv", "stress"):
        values["contact_rate"] = float(spec["contact_rate"])
    if name in ("sir_adjust", "stress"):
        for index, factor in enumerate(spec["adjustments"]):
            values[adjustment_parameter_key(index)] = float(factor)
    return values


def contact_with_adjustments(spec: dict[str, Any]) -> Function:
    """Left-fold the adjustment list onto the contact rate as graph multiplies.

    The Python loop only builds the chain. Each node is ``jax.numpy.multiply``
    via ``Parameter.__mul__``, not ``adjust.Multiply.get_new_value``.
    """
    rate: Any = Parameter("contact_rate")
    for index, _factor in enumerate(spec["adjustments"]):
        rate = rate * Parameter(adjustment_parameter_key(index))
    if not isinstance(rate, Function):
        raise TypeError("adjustment chain did not build a computegraph Function")
    return rate


def time_varying_contact(spec: dict[str, Any]) -> Function:
    """Clamped linear interpolation of the spec knots. ``x_axis`` stays ``Time``."""
    knots = spec["contact_knots"]
    times = jnp.asarray(knots["times"], dtype=jnp.float64)
    values = jnp.asarray(knots["values"], dtype=jnp.float64)
    return get_linear_interpolation_function(times, values)


def static_mixing_matrix(spec: dict[str, Any]) -> Any:
    """Age mixing as a float64 array. Not a Python callable."""
    return jnp.asarray(spec["static_mixing"], dtype=jnp.float64)


def _select_mixing_matrix(time: Any, stack: Any, bin_width: Any, last_index: Any) -> Any:
    """Index the frozen stack inside a traced step. Clamp outside ``0 .. last_index``."""
    index = jnp.floor(time / bin_width).astype(jnp.int32)
    index = jnp.clip(index, 0, last_index)
    return stack[index]


def time_varying_mixing(spec: dict[str, Any]) -> Function:
    """Stack of spec matrices, indexed by ``floor(t / bin_width)`` on ``Time``.

    The stack is ``Data``, not a host callable. ``set_mixing_matrix`` receives
    this Function.
    """
    stack = jnp.asarray(spec["tv_mixing"], dtype=jnp.float64)
    bin_width = jnp.asarray(spec["tv_mixing_bin_width"], dtype=jnp.float64)
    last_index = jnp.asarray(stack.shape[0] - 1, dtype=jnp.int32)
    return Function(
        _select_mixing_matrix,
        (Time, Data(stack), Data(bin_width), Data(last_index)),
    )


def stress_contact(spec: dict[str, Any]) -> Function:
    """Time-varying contact rate, then the 24 adjustment factors, as graph multiplies."""
    rate: Any = time_varying_contact(spec)
    for index, _factor in enumerate(spec["adjustments"]):
        rate = rate * Parameter(adjustment_parameter_key(index))
    if not isinstance(rate, Function):
        raise TypeError("stress contact did not build a computegraph Function")
    return rate


def prepare_model(name: str, steps: int, spec: dict[str, Any] | None = None) -> PreparedModel:
    """Build one model and request flow outputs. Does not call ``get_runner``."""
    if name not in MODEL_NAMES and name not in STRATIFIED_NAMES:
        raise KeyError(name)
    loaded = load_spec() if spec is None else spec
    _require_layout(loaded)
    model = _base_model(loaded, steps)
    contact = _contact_for(name, loaded)
    compartments = loaded["compartments"]
    model.add_infection_frequency_flow("infection", contact, compartments[0], compartments[1])
    model.add_transition_flow(
        "recovery",
        Parameter("recovery"),
        compartments[1],
        compartments[2],
    )
    mixing = _apply_stratification(name, model, loaded)
    for flow_name in loaded["flows"]:
        model.request_output_for_flow(str(flow_name), str(flow_name))
    return PreparedModel(
        name=name,
        model=model,
        parameters=parameter_values(name, loaded),
        contact=contact,
        mixing=mixing,
    )


def runner_for(prepared: PreparedModel, solver: str) -> Any:
    """Finalise the prepared model. ``solver`` is a backend keyword, not positional."""
    return prepared.model.get_runner(
        prepared.parameters,
        include_full_outputs=True,
        solver=solver,
    )


def traced_mixing_matrix(runner: Any, parameters: dict[str, float], time: float) -> Any:
    """Mixing matrix the timestep graph actually feeds to force of infection."""
    step = runner.impl_dict["one_step"](parameters=parameters, t=time)
    values = step.ts_graph_vals
    if "mixing_matrix" not in values:
        raise KeyError("timestep graph has no mixing_matrix")
    return values["mixing_matrix"]


def assert_mixing_matches_spec(
    runner: Any, parameters: dict[str, float], spec: dict[str, Any], name: str
) -> None:
    """The traced mixing matrix is the spec array, not a new draw."""
    if name == "age_mix":
        actual = traced_mixing_matrix(runner, parameters, float(spec["t0"]))
        expected = jnp.asarray(spec["static_mixing"], dtype=jnp.float64)
        if not jnp.array_equal(actual, expected):
            raise ValueError(f"{name} traced mixing is not spec['static_mixing']")
        return
    if name not in ("age_mix_tv", "stress"):
        raise KeyError(name)
    stack = jnp.asarray(spec["tv_mixing"], dtype=jnp.float64)
    width = float(spec["tv_mixing_bin_width"])
    origin = float(spec["t0"])
    last = int(stack.shape[0]) - 1
    checks = (
        (origin, 0),
        (width, 1),
        (origin - float(spec["dt"]), 0),
        (width * float(stack.shape[0]), last),
    )
    for time_value, index in checks:
        actual = traced_mixing_matrix(runner, parameters, time_value)
        if not jnp.array_equal(actual, stack[index]):
            raise ValueError(f"{name} mixing at t={time_value} is not spec slice {index}")


def inspect_multiply_chain(node: Any) -> MultiplyChain:
    """Require every function in the chain to be ``jnp.multiply``, not a Python callable."""
    root = node.obj if type(node).__name__ == "GraphObjectParameter" else node
    names: list[str] = []
    innermost = ""

    def walk(current: Any) -> None:
        nonlocal innermost
        if not isinstance(current, Function):
            return
        func = current.func
        module = getattr(func, "__module__", "")
        func_name = getattr(func, "__name__", type(func).__name__)
        names.append(f"{module}.{func_name}")
        if func is not jnp.multiply:
            raise ValueError(f"adjustment chain node is not jnp.multiply: {func!r}")
        if all(type(arg).__name__ == "Parameter" for arg in current.args):
            innermost = repr(current)
        for arg in (*current.args, *current.kwargs.values()):
            walk(arg)

    walk(root)
    if not names:
        raise ValueError("adjustment chain has no graph functions")
    if not innermost:
        raise ValueError("adjustment chain has no Parameter * Parameter breakpoint")
    return MultiplyChain(
        count=len(names), functions=tuple(dict.fromkeys(names)), breakpoint=innermost
    )


def assert_knots_clamped(contact: Any, spec: dict[str, Any]) -> None:
    """The interpolator summer2 traces must hold the endpoint values outside the knots."""
    knots = spec["contact_knots"]
    times = knots["times"]
    values = knots["values"]
    dt = float(spec["dt"])
    evaluated = get_time_callable(contact, jit_compile=True)
    checks = (
        (float(times[0]) - dt, float(values[0])),
        (float(times[-1]) + dt, float(values[-1])),
        (float(times[0]), float(values[0])),
        (float(times[-1]), float(values[-1])),
    )
    for time_value, expected in checks:
        actual = float(evaluated(time_value, parameters={}))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"contact clamp at t={time_value} is {actual}, expected {expected}")


def host_callback_primitives(closed: Any) -> tuple[str, ...]:
    """Primitive names in a jaxpr that would call back into Python."""
    found: list[str] = []

    def walk(jaxpr: Any) -> None:
        eqns = getattr(jaxpr, "eqns", None)
        if eqns is None and hasattr(jaxpr, "jaxpr"):
            walk(jaxpr.jaxpr)
            return
        if eqns is None:
            return
        for eqn in eqns:
            name = str(eqn.primitive)
            if any(marker in name for marker in _HOST_CALLBACKS):
                found.append(name)
            for value in eqn.params.values():
                _walk_param(value, walk)

    walk(closed)
    return tuple(found)


def _walk_param(value: Any, walk: Any) -> None:
    if hasattr(value, "eqns") or hasattr(value, "jaxpr"):
        walk(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_param(item, walk)


def _require_layout(spec: dict[str, Any]) -> None:
    if list(spec["compartments"]) != ["S", "I", "R"]:
        raise ValueError(f"unexpected compartments {spec['compartments']}")
    if list(spec["infectious_compartments"]) != ["I"]:
        raise ValueError(f"unexpected infectious compartments {spec['infectious_compartments']}")
    if list(spec["flows"]) != ["infection", "recovery"]:
        raise ValueError(f"unexpected flows {spec['flows']}")


def _base_model(spec: dict[str, Any], steps: int) -> CompartmentalModel:
    t0 = float(spec["t0"])
    dt = float(spec["dt"])
    model = CompartmentalModel(
        times=(t0, t0 + steps * dt),
        compartments=list(spec["compartments"]),
        infectious_compartments=list(spec["infectious_compartments"]),
        timestep=dt,
    )
    distribution = {str(name): float(value) for name, value in spec["initial_population"].items()}
    model.set_initial_population(distribution=distribution)
    return model


def _contact_for(name: str, spec: dict[str, Any]) -> Any:
    if name in ("sir", "age_mix", "age_mix_tv"):
        return Parameter("contact_rate")
    if name == "sir_adjust":
        return contact_with_adjustments(spec)
    if name == "sir_tv":
        return time_varying_contact(spec)
    if name == "stress":
        return stress_contact(spec)
    raise KeyError(name)


def _age_stratification(spec: dict[str, Any], mixing: Any) -> Stratification:
    """Age bands from the spec. Not ``AgeStratification``: that adds ageing flows."""
    strat = Stratification("age", list(spec["age_bands"]), list(spec["compartments"]))
    strat.set_mixing_matrix(mixing)
    return strat


def _apply_stratification(name: str, model: CompartmentalModel, spec: dict[str, Any]) -> Any:
    """Stratify in place. Returns the mixing object passed to summer2, or ``None``."""
    if name in MODEL_NAMES:
        return None
    if name == "age_mix":
        mixing = static_mixing_matrix(spec)
        model.stratify_with(_age_stratification(spec, mixing))
        return mixing
    if name == "age_mix_tv":
        mixing = time_varying_mixing(spec)
        model.stratify_with(_age_stratification(spec, mixing))
        return mixing
    if name == "stress":
        mixing = time_varying_mixing(spec)
        model.stratify_with(_age_stratification(spec, mixing))
        model.stratify_with(
            Stratification("location", list(spec["locations"]), list(spec["compartments"]))
        )
        model.stratify_with(
            StrainStratification("strain", list(spec["strains"]), list(spec["compartments"]))
        )
        return mixing
    raise KeyError(name)
