"""Summer4 benchmark models driven only by ``summer2bench/spec.json``.

Import this module before the first ``jit``. It enables float64 first.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    Lookup,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SaveFn,
    SavePlan,
    SaveRequest,
    Split,
    Time,
    TransitionFlow,
    floor,
)
from summer4.epi import ForceOfInfection, MixingMatrix  # noqa: E402
from summer4.timevarying import linear  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent.parent / "summer2bench" / "spec.json"
UNSTRATIFIED_NAMES = ("sir", "sir_adjust", "sir_tv")
STRATIFIED_NAMES = ("age_mix", "age_mix_tv", "stress")
MODEL_NAMES = UNSTRATIFIED_NAMES + STRATIFIED_NAMES
EXPECTED_COMPARTMENTS = {
    "sir": 3,
    "sir_adjust": 3,
    "sir_tv": 3,
    "age_mix": 48,
    "age_mix_tv": 48,
    "stress": 3840,
}
TV_MIXING_PARAM = "tv_mixing"


@dataclass(frozen=True)
class BuiltModel:
    """Compiled model plus the save plan and parameters to run it."""

    name: str
    compiled: Any
    parameters: dict[str, Any]
    y0: PropertyData
    state: Property
    plan: SavePlan
    contact: Any
    mixing: Any = None
    n_compartments: int = 0
    infection_flows: tuple[str, ...] = ("infection",)


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


def stress_contact(spec: dict[str, Any]) -> Any:
    """Time-varying contact rate, then the 24 adjustment factors."""
    rate: Any = time_varying_contact(spec)
    for index, _factor in enumerate(spec["adjustments"]):
        rate = rate * Param(adjustment_parameter_key(index))
    return rate


def static_mixing_matrix(age: Property, spec: dict[str, Any]) -> MixingMatrix:
    """Age mixing held as a float64 array baked into the MixingMatrix."""
    matrix = jnp.asarray(spec["static_mixing"], dtype=jnp.float64)
    return MixingMatrix(
        age,
        np.asarray(matrix),
        normalize="none",
        check_reciprocal=False,
    )


def time_varying_mixing_matrix(age: Property, spec: dict[str, Any]) -> MixingMatrix:
    """``Lookup`` of the ``(8, 16, 16)`` stack, indexed inside the jitted step."""
    width = float(spec["tv_mixing_bin_width"])
    return MixingMatrix(
        age,
        Lookup(Param(TV_MIXING_PARAM), floor(Time() / width)),
        normalize="none",
        check_reciprocal=False,
    )


def parameter_values(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Values ``CompiledModel.run`` should be given."""
    if name not in MODEL_NAMES:
        raise KeyError(name)
    values: dict[str, Any] = {"recovery": float(spec["recovery"])}
    if name not in ("sir_tv", "stress"):
        values["contact_rate"] = float(spec["contact_rate"])
    if name in ("sir_adjust", "stress"):
        for index, factor in enumerate(spec["adjustments"]):
            values[adjustment_parameter_key(index)] = float(factor)
    if name in ("age_mix_tv", "stress"):
        values[TV_MIXING_PARAM] = jnp.asarray(spec["tv_mixing"], dtype=jnp.float64)
    return values


def save_plan(
    *,
    infection_flows: tuple[str, ...],
    recovery_flow: str = "recovery",
    sum_over: tuple[Property, str] | None = None,
) -> SavePlan:
    """Compartments plus one infection series and one recovery series.

    A single infection flow uses ``FlowMass``. Several per-strain infection
    flows are summed by a ``SaveFn`` so the saved key stays ``infection``.
    """
    requests: dict[str, SaveRequest] = {
        "compartments": SaveRequest(Compartments()),
    }
    if len(infection_flows) == 1:
        flow_name = infection_flows[0]
        mass: FlowMass | SaveFn
        if sum_over is None:
            mass = FlowMass(flow=flow_name)
        else:
            mass = FlowMass(flow=flow_name, sum_over=sum_over)
        requests["infection"] = SaveRequest(mass)
    else:
        requests["infection"] = SaveRequest(_infection_total_save(infection_flows))
    if sum_over is None:
        requests["recovery"] = SaveRequest(FlowMass(flow=recovery_flow))
    else:
        requests["recovery"] = SaveRequest(
            FlowMass(flow=recovery_flow, sum_over=sum_over),
        )
    return SavePlan(requests=requests)


def build_model(name: str, spec: dict[str, Any] | None = None) -> BuiltModel:
    """Build and compile one model from the shared spec. Does not run it."""
    if name not in MODEL_NAMES:
        raise KeyError(name)
    loaded = load_spec() if spec is None else spec
    _require_layout(loaded)
    if name in UNSTRATIFIED_NAMES:
        return _build_unstratified(name, loaded)
    return _build_stratified(name, loaded)


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
    """One float64 series per logical flow: time axis length ``steps + 1``."""
    values = result[flow_name].values
    data = jnp.asarray(values.data if hasattr(values, "data") else values)
    if data.ndim == 1:
        return data
    if data.ndim == 2 and data.shape[-1] == 1:
        return data[:, 0]
    return jnp.sum(data, axis=tuple(range(1, data.ndim)))


def reduce_result(result: Any, flow_names: tuple[str, ...] = ("infection", "recovery")) -> Any:
    """Sum compartments and each named flow series for ``block_until_ready``."""
    compartments = result["compartments"].values
    comp_data = compartments.data if hasattr(compartments, "data") else compartments
    total = jnp.asarray(comp_data).sum()
    for flow_name in flow_names:
        total = total + flow_series(result, flow_name).sum()
    return total


def _infection_total_save(infection_flows: tuple[str, ...]) -> SaveFn:
    """Sum every per-strain infection edge mass into one scalar per save time."""

    def fn(ctx: Any) -> Any:
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for name in infection_flows:
            total = total + jnp.asarray(ctx.flows[name]).sum()
        return total

    return SaveFn(fn=fn, reads=frozenset(infection_flows))


def _require_layout(spec: dict[str, Any]) -> None:
    if list(spec["compartments"]) != ["S", "I", "R"]:
        raise ValueError(f"unexpected compartments {spec['compartments']}")
    if list(spec["flows"]) != ["infection", "recovery"]:
        raise ValueError(f"unexpected flows {spec['flows']}")


def _equal_split(prop: Property) -> Split:
    n = len(prop.traits)
    return Split(prop, {trait: 1.0 / float(n) for trait in prop.traits})


def _contact_for(name: str, spec: dict[str, Any]) -> Any:
    if name in ("sir", "age_mix", "age_mix_tv"):
        return Param("contact_rate")
    if name == "sir_adjust":
        return contact_with_adjustments(spec)
    if name == "sir_tv":
        return time_varying_contact(spec)
    if name == "stress":
        return stress_contact(spec)
    raise KeyError(name)


def _build_unstratified(name: str, loaded: dict[str, Any]) -> BuiltModel:
    compartments = list(loaded["compartments"])
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
    return _finish(name, model, state, contact, mixing, ("infection",), (pop, "source"), loaded)


def _build_stratified(name: str, loaded: dict[str, Any]) -> BuiltModel:
    compartments = list(loaded["compartments"])
    state = Property("state", tuple(compartments))
    age = Property("age", tuple(str(band) for band in loaded["age_bands"]))
    pmap = PropertyMap.from_property(state).stratify(age)
    splits: list[Split] = [_equal_split(age)]
    strain: Property | None = None
    if name == "stress":
        location = Property("location", tuple(str(item) for item in loaded["locations"]))
        strain = Property("strain", tuple(str(item) for item in loaded["strains"]))
        pmap = pmap.stratify(location).stratify(strain)
        splits.extend([_equal_split(location), _equal_split(strain)])

    contact = _contact_for(name, loaded)
    if name == "age_mix":
        mixing = static_mixing_matrix(age, loaded)
    else:
        mixing = time_varying_mixing_matrix(age, loaded)

    model = FlowModel(pmap)
    infection_flows = _add_infection_flows(model, state, age, mixing, contact, strain)
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    model.set_initial_population(
        {
            state[compartment]: float(value)
            for compartment, value in loaded["initial_population"].items()
        },
        splits=tuple(splits),
    )
    return _finish(name, model, state, contact, mixing, infection_flows, None, loaded)


def _add_infection_flows(
    model: FlowModel,
    state: Property,
    age: Property,
    mixing: MixingMatrix,
    contact: Any,
    strain: Property | None,
) -> tuple[str, ...]:
    if strain is None:
        model.add_flow(
            TransitionFlow(
                "infection",
                state["S"],
                state["I"],
                ForceOfInfection(
                    "infection",
                    infectious=state["I"],
                    group_by=age,
                    mixing=mixing,
                    kind="frequency",
                    contact_rate=contact,
                ),
            )
        )
        return ("infection",)

    fois = ForceOfInfection.per_trait(
        strain,
        name_prefix="infection",
        infectious=state["I"],
        group_by=age,
        mixing=mixing,
        kind="frequency",
        contact_rate=contact,
    )
    for foi in fois:
        model.add_flow(TransitionFlow(foi.name, state["S"], state["I"], foi))
    return tuple(foi.name for foi in fois)


def _finish(
    name: str,
    model: FlowModel,
    state: Property,
    contact: Any,
    mixing: Any,
    infection_flows: tuple[str, ...],
    sum_over: tuple[Property, str] | None,
    loaded: Mapping[str, Any],
) -> BuiltModel:
    compiled = model.compile()
    n_compartments = compiled.pmap.size
    expected = EXPECTED_COMPARTMENTS[name]
    if n_compartments != expected:
        raise ValueError(f"{name} has {n_compartments} compartments, expected {expected}")
    params = parameter_values(name, dict(loaded))
    y0 = compiled.initial_state(compiled.prepare(params))
    return BuiltModel(
        name=name,
        compiled=compiled,
        parameters=params,
        y0=y0,
        state=state,
        plan=save_plan(infection_flows=infection_flows, sum_over=sum_over),
        contact=contact,
        mixing=mixing,
        n_compartments=n_compartments,
        infection_flows=infection_flows,
    )
