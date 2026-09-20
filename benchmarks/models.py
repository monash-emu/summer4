"""Summer4 benchmark models driven only by ``summer2bench/spec.json``.

Import this module before the first ``jit``. It enables float64 first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from summer4 import (  # noqa: E402
    Compartments,
    FlowMass,
    FlowModel,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Time,
    TransitionFlow,
)
from summer4.epi import ForceOfInfection, MixingMatrix  # noqa: E402
from summer4.timevarying import linear  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent.parent / "summer2bench" / "spec.json"
UNSTRATIFIED_NAMES = ("sir", "sir_adjust", "sir_tv")


@dataclass(frozen=True)
class BuiltModel:
    """Compiled unstratified model plus the save plan and parameters to run it."""

    name: str
    compiled: Any
    parameters: dict[str, float]
    y0: PropertyData
    state: Property
    pop: Property
    plan: SavePlan
    contact: Any


def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the shared numeric spec. Model code must not restate those numbers."""
    spec_path = SPEC_PATH if path is None else path
    with spec_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"{spec_path} must contain a JSON object")
    return loaded


def adjustment_parameter_key(index: int) -> str:
    """Parameter name for one factor in spec order."""
    return f"adjustment_{index}"


def contact_with_adjustments(spec: dict[str, Any]) -> Any:
    """Left-fold the adjustment list onto the contact rate as rate multiplies."""
    rate: Any = Param("contact_rate")
    for index, _factor in enumerate(spec["adjustments"]):
        rate = rate * Param(adjustment_parameter_key(index))
    return rate


def time_varying_contact(spec: dict[str, Any]) -> Any:
    """Clamped linear interpolation of the spec knots, evaluated at ``Time()``."""
    knots = spec["contact_knots"]
    times = tuple(float(value) for value in knots["times"])
    values = tuple(float(value) for value in knots["values"])
    return linear(Time(), times, values)


def parameter_values(name: str, spec: dict[str, Any]) -> dict[str, float]:
    """Values ``CompiledModel.run`` should be given."""
    if name not in UNSTRATIFIED_NAMES:
        raise KeyError(name)
    values = {"recovery": float(spec["recovery"])}
    if name != "sir_tv":
        values["contact_rate"] = float(spec["contact_rate"])
    if name == "sir_adjust":
        for index, factor in enumerate(spec["adjustments"]):
            values[adjustment_parameter_key(index)] = float(factor)
    return values


def save_plan(pop: Property, flow_names: tuple[str, ...]) -> SavePlan:
    """Compartments plus one flow-total series per named flow.

    ``FlowMass(..., sum_over=(pop, \"source\"))`` reduces every edge of the
    flow to the single ``pop`` group, matching summer2's one series per flow.
    """
    requests: dict[str, SaveRequest] = {
        "compartments": SaveRequest(Compartments()),
    }
    for flow_name in flow_names:
        requests[flow_name] = SaveRequest(
            FlowMass(flow=flow_name, sum_over=(pop, "source")),
        )
    return SavePlan(requests=requests)


def build_model(name: str, spec: dict[str, Any] | None = None) -> BuiltModel:
    """Build and compile one unstratified model. Does not run it."""
    if name not in UNSTRATIFIED_NAMES:
        raise KeyError(name)
    loaded = load_spec() if spec is None else spec
    compartments = list(loaded["compartments"])
    if compartments != ["S", "I", "R"]:
        raise ValueError(f"unexpected compartments {compartments}")
    flow_names = tuple(str(item) for item in loaded["flows"])
    if flow_names != ("infection", "recovery"):
        raise ValueError(f"unexpected flows {flow_names}")

    state = Property("state", tuple(compartments))
    pop = Property("pop", ("all",))
    pmap = PropertyMap.from_property(state).stratify(pop)
    contact = _contact_for(name, loaded)
    mixing = MixingMatrix(pop, np.array([[1.0]], dtype=np.float64), check_reciprocal=False)
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            ForceOfInfection(
                "infection",
                infectious=state["I"],
                group_by=mixing.prop,
                mixing=mixing,
                kind="frequency",
                contact_rate=contact,
            ),
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    model.set_initial_population(
        {
            state[compartment]: float(value)
            for compartment, value in loaded["initial_population"].items()
        }
    )
    compiled = model.compile()
    y0 = compiled.initial_state(compiled.prepare(parameter_values(name, loaded)))
    return BuiltModel(
        name=name,
        compiled=compiled,
        parameters=parameter_values(name, loaded),
        y0=y0,
        state=state,
        pop=pop,
        plan=save_plan(pop, flow_names),
        contact=contact,
    )


def fixed_step_run_kwargs(
    built: BuiltModel,
    solver: Any,
    steps: int,
    *,
    t0: float | None = None,
    dt: float | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keyword arguments that select Diffrax ``ConstantStepSize``.

    Pass a Diffrax solver instance (``diffrax.Euler()`` or ``ClassicalRK4()``),
    ``dt``, ``steps``, and ``max_steps`` at least ``steps``. Do not pass
    ``rtol``, ``atol``, or ``solver=\"euler\"``.
    """
    loaded = load_spec() if spec is None else spec
    return {
        "solver": solver,
        "t0": float(loaded["t0"] if t0 is None else t0),
        "dt": float(loaded["dt"] if dt is None else dt),
        "steps": int(steps),
        "max_steps": int(steps),
        "save": built.plan,
    }


def flow_series(result: Any, flow_name: str) -> Any:
    """One float64 series per flow: time axis length ``steps + 1``."""
    data = jnp.asarray(result[flow_name].values.data)
    if data.ndim == 1:
        return data
    if data.ndim == 2 and data.shape[-1] == 1:
        return data[:, 0]
    return jnp.sum(data, axis=tuple(range(1, data.ndim)))


def reduce_result(result: Any, flow_names: tuple[str, ...]) -> Any:
    """Sum compartments and each named flow series for ``block_until_ready``."""
    total = jnp.asarray(result["compartments"].values.data).sum()
    for flow_name in flow_names:
        total = total + flow_series(result, flow_name).sum()
    return total


def _contact_for(name: str, spec: dict[str, Any]) -> Any:
    if name == "sir":
        return Param("contact_rate")
    if name == "sir_adjust":
        return contact_with_adjustments(spec)
    if name == "sir_tv":
        return time_varying_contact(spec)
    raise KeyError(name)
