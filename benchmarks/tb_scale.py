"""Synthetic Kiribati-shaped model for TB-scale benchmarks.

Shape only — no real demography or screening data. 10 disease states × 8 uneven
age bands × 2 reachability strata = 160 compartments; yearly saves 1850–2035;
an :class:`~summer4.results.outputset.OutputSet` of about 150 named outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from summer4 import (
    REMAINDER,
    Compartments,
    EntryFlow,
    Everything,
    ExitFlow,
    FlowMass,
    FlowModel,
    InitialPopulation,
    Lookup,
    Multiply,
    OutputSet,
    Param,
    Property,
    PropertyMap,
    SavePlan,
    Source,
    Split,
    Time,
    TraitChain,
    TransitionFlow,
    floor,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix

# Kiribati age lower bounds (years).
AGE_BREAKS: tuple[str, ...] = ("0", "3", "5", "10", "15", "18", "40", "65")

STATES: tuple[str, ...] = (
    "mtb_naive",
    "early_latent",
    "late_latent",
    "asympt",
    "clin_inf",
    "on_treatment",
    "recovered",
    "failed",
    "screened",
    "protected",
)

# Susceptible sources for the four reinfection paths (rel_sus on each flow).
SUSCEPTIBLE_SOURCES: tuple[str, ...] = ("mtb_naive", "late_latent", "recovered", "failed")

T0: float = 1850.0
T1: float = 2035.0
SAVE_TIMES: np.ndarray = np.arange(T0, T1 + 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TbScaleModel:
    """Compiled synthetic model plus the outputs and default run kwargs."""

    state: Property
    age: Property
    reach: Property
    pmap: PropertyMap
    compiled: Any
    outputs: OutputSet
    params: dict[str, Any]
    save: SavePlan
    flow_names: tuple[str, ...]

    @property
    def n_compartments(self) -> int:
        return int(self.pmap.size)

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    @property
    def n_flows(self) -> int:
        return len(self.flow_names)


def _death_table(age: Property) -> Any:
    """Smooth per-age death rates over the full horizon (shape only)."""
    n_age = len(age.traits)
    times = np.linspace(T0, T1, 20)
    # Older bands die faster; rates stay small so a 185-year euler run is stable.
    base = np.linspace(0.002, 0.04, n_age)
    values = np.outer(1.0 + 0.15 * np.sin((times - T0) / 40.0), base)
    return Data.table(times, values, over=age).interp("linear")


def _mixing_stack(n_age: int, n_years: int) -> np.ndarray:
    """Yearly (n_years, n_age, n_age) matrices — mild assortative diagonal."""
    eye = np.eye(n_age, dtype=np.float64)
    off = (np.ones((n_age, n_age)) - eye) / max(n_age - 1, 1)
    base = 0.7 * eye + 0.3 * off
    stack = np.stack([base * (1.0 + 0.05 * np.sin(i / 17.0)) for i in range(n_years)])
    return stack.astype(np.float64)


def _build_outputs(
    state: Property, age: Property, reach: Property, flow_names: tuple[str, ...]
) -> OutputSet:
    """About 150 named outputs: totals, per-age series, ratios, cumulatives."""
    outputs = OutputSet()
    outputs["population"] = Compartments().total()
    outputs["infected"] = Compartments(where=state["clin_inf"] | state["asympt"]).total()
    outputs["latent"] = Compartments(where=state["early_latent"] | state["late_latent"]).total()
    outputs["on_tx"] = Compartments(where=state["on_treatment"]).total()
    outputs["reachable_pop"] = Compartments(where=reach["reachable"]).total()
    outputs["prevalence"] = outputs.ref("infected") / outputs.ref("population")
    outputs["latent_frac"] = outputs.ref("latent") / outputs.ref("population")
    outputs["reach_frac_out"] = outputs.ref("reachable_pop") / outputs.ref("population")

    for name in STATES:
        outputs[f"total_{name}"] = Compartments(where=state[name]).total()
        outputs[f"frac_{name}"] = outputs.ref(f"total_{name}") / outputs.ref("population")

    for trait in age.traits:
        band = age[trait]
        outputs[f"pop_age_{trait}"] = Compartments(where=band).total()
        outputs[f"inf_age_{trait}"] = Compartments(
            where=(state["clin_inf"] | state["asympt"]) & band
        ).total()
        outputs[f"prev_age_{trait}"] = outputs.ref(f"inf_age_{trait}") / outputs.ref(
            f"pop_age_{trait}"
        )
        outputs[f"latent_age_{trait}"] = Compartments(
            where=(state["early_latent"] | state["late_latent"]) & band
        ).total()
        outputs[f"tx_age_{trait}"] = Compartments(where=state["on_treatment"] & band).total()
        outputs[f"naive_age_{trait}"] = Compartments(where=state["mtb_naive"] & band).total()
        outputs[f"rec_age_{trait}"] = Compartments(where=state["recovered"] & band).total()
        outputs[f"reach_age_{trait}"] = Compartments(where=reach["reachable"] & band).total()

    # Flow masses (midpoint), scaled, and cumulative from 2020.
    # Keep churn as midpoint-only so the named set stays near ~150.
    primary_flows = [
        name
        for name in flow_names
        if name.startswith("inf_")
        or name
        in (
            "progress_early",
            "progress_late",
            "activate",
            "detect",
            "recover_tx",
            "fail_tx",
            "death",
            "ageing",
            "screen",
            "relapse",
            "recycle",
            "births",
        )
    ]
    for fname in primary_flows:
        outputs[f"flow_{fname}"] = FlowMass(fname).total().midpoint()
        outputs[f"cum_{fname}"] = outputs.ref(f"flow_{fname}").cumulative(start=2020.0)
        outputs[f"scaled_{fname}"] = outputs.ref(f"flow_{fname}") * Param("output_scale")

    for name in flow_names:
        if name.startswith("churn_"):
            outputs[f"flow_{name}"] = FlowMass(name).total().midpoint()

    for trait in age.traits:
        outputs[f"inc_age_{trait}"] = (
            FlowMass("inf_mtb_naive", where=Source(age[trait])).total().midpoint()
        )
        outputs[f"death_age_{trait}"] = (
            FlowMass("death", where=Source(age[trait])).total().midpoint()
        )

    # Multi-flow incidence via Output algebra — FlowMass over several infection
    # flows disagrees on selected edge dims after source-specific adjusts.
    inf_refs = [outputs.ref(f"flow_inf_{src}") for src in SUSCEPTIBLE_SOURCES]
    incidence = inf_refs[0]
    for ref in inf_refs[1:]:
        incidence = incidence + ref
    outputs["incidence"] = incidence
    outputs["cum_incidence"] = outputs.ref("incidence").cumulative(start=2020.0)
    outputs["incidence_per_capita"] = outputs.ref("incidence") / outputs.ref("population")
    outputs["scaled_prevalence"] = outputs.ref("prevalence") * Param("output_scale")
    outputs["deaths"] = outputs.ref("flow_death")
    outputs["cum_deaths"] = outputs.ref("cum_death")

    return outputs


def build_tb_scale_model() -> TbScaleModel:
    """Construct, compile, and attach the ~150-output set."""
    state = Property("state", STATES)
    age = Property("age", AGE_BREAKS)
    reach = Property("reach", ("reachable", "unreachable"))
    pmap = PropertyMap.from_property(state).stratify(age).stratify(reach)

    n_age = len(age.traits)
    n_years = int(T1 - T0) + 1
    mixing_stack = _mixing_stack(n_age, n_years)
    death_rate = _death_table(age)

    foi = ForceOfInfection(
        "infection",
        infectious=state["clin_inf"] | state["asympt"],
        group_by=age,
        kind=FOIKind.GENERALISED,
        exponent=Param("infection_pop_scale"),
        contact_rate=Param("contact_rate"),
        mixing=MixingMatrix(
            age,
            Lookup(Param("mixing"), floor(Time() - T0)),
            normalize="none",
            check_reciprocal=False,
        ),
        infectiousness=[
            (state["asympt"], Param("rel_inf_asympt")),
            (age["0"], 0.0),
            (age["3"], 0.0),
            (age["5"], 0.0),
            (age["10"], 0.0),
        ],
    )

    rel_sus = {
        "mtb_naive": 1.0,
        "late_latent": Param("rel_sus_latent"),
        "recovered": Param("rel_sus_recovered"),
        "failed": Param("rel_sus_failed"),
    }

    model = FlowModel(pmap)
    flow_names: list[str] = []

    for source in SUSCEPTIBLE_SOURCES:
        name = f"inf_{source}"
        model.add_flow(
            TransitionFlow(
                name,
                state[source],
                state["early_latent"],
                foi,
                adjust=(Multiply(rel_sus[source]),),
            )
        )
        flow_names.append(name)

    model.add_flow(
        TransitionFlow(
            "progress_early", state["early_latent"], state["late_latent"], Param("prog_early")
        )
    )
    model.add_flow(
        TransitionFlow("progress_late", state["late_latent"], state["asympt"], Param("prog_late"))
    )
    model.add_flow(
        TransitionFlow("activate", state["asympt"], state["clin_inf"], Param("activation"))
    )
    model.add_flow(
        TransitionFlow("detect", state["clin_inf"], state["on_treatment"], Param("detection"))
    )
    model.add_flow(
        TransitionFlow("self_clear", state["early_latent"], state["mtb_naive"], Param("self_clear"))
    )
    model.add_flow(
        TransitionFlow("recover_nat", state["clin_inf"], state["recovered"], Param("recover_nat"))
    )
    model.add_flow(
        TransitionFlow("recover_tx", state["on_treatment"], state["recovered"], Param("recover_tx"))
    )
    model.add_flow(
        TransitionFlow("fail_tx", state["on_treatment"], state["failed"], Param("fail_tx"))
    )
    model.add_flow(
        TransitionFlow("relapse", state["recovered"], state["late_latent"], Param("relapse"))
    )
    model.add_flow(
        TransitionFlow(
            "screen", state["clin_inf"] & reach["reachable"], state["screened"], Param("screen")
        )
    )
    model.add_flow(
        TransitionFlow("protect", state["mtb_naive"], state["protected"], Param("protect"))
    )
    flow_names.extend(
        [
            "progress_early",
            "progress_late",
            "activate",
            "detect",
            "self_clear",
            "recover_nat",
            "recover_tx",
            "fail_tx",
            "relapse",
            "screen",
            "protect",
        ]
    )

    # Extra fixed-rate clinical churn so named-flow count sits near ~40.
    churn_pairs = (
        ("failed", "late_latent"),
        ("screened", "on_treatment"),
        ("protected", "mtb_naive"),
        ("late_latent", "asympt"),
        ("recovered", "early_latent"),
        ("failed", "clin_inf"),
        ("screened", "recovered"),
        ("protected", "late_latent"),
        ("asympt", "recovered"),
        ("on_treatment", "failed"),
        ("clin_inf", "asympt"),
        ("early_latent", "protected"),
        ("mtb_naive", "screened"),
        ("recovered", "protected"),
        ("failed", "screened"),
        ("screened", "failed"),
    )
    for i, (src, dest) in enumerate(churn_pairs):
        name = f"churn_{i}_{src}"
        model.add_flow(TransitionFlow(name, state[src], state[dest], 0.0005 * (i + 1)))
        flow_names.append(name)

    death = model.add_flow(ExitFlow("death", Everything(), death_rate))
    flow_names.append("death")
    # Recycle deaths into (mtb_naive, age 0), preserving reachability via sum_over(age).
    model.add_flow(EntryFlow("recycle", state["mtb_naive"] & age["0"], death.sum_over(age)))
    flow_names.append("recycle")

    # Stratum births (importation into youngest naive band).
    model.add_flow(EntryFlow("births", state["mtb_naive"] & age["0"], Param("birth_rate")))
    flow_names.append("births")

    ageing = TraitChain.from_breakpoints(age)
    model.add_flow(
        TransitionFlow(
            "ageing",
            age.present(),
            age.present(),
            1.0,
            pairing=ageing,
        )
    )
    flow_names.append("ageing")

    pop = InitialPopulation(
        {
            state["mtb_naive"]: Param("pop") - Param("seed"),
            state["clin_inf"]: Param("seed"),
        },
        splits=(
            Split(age, {t: 1.0 / n_age for t in age.traits}),
            Split(reach, {"reachable": Param("reach_frac"), "unreachable": REMAINDER}),
        ),
    )
    model.set_initial_population(pop)

    compiled = model.compile()
    names = tuple(flow_names)
    outputs = _build_outputs(state, age, reach, names)
    save = outputs.plan(SavePlan(ts=SAVE_TIMES))

    params: dict[str, Any] = {
        "pop": 80_000.0,
        "seed": 40.0,
        "reach_frac": 0.7,
        "contact_rate": 0.35,
        "infection_pop_scale": 0.85,
        "rel_inf_asympt": 0.4,
        "rel_sus_latent": 0.5,
        "rel_sus_recovered": 0.25,
        "rel_sus_failed": 0.35,
        "prog_early": 0.2,
        "prog_late": 0.05,
        "activation": 0.1,
        "detection": 0.3,
        "self_clear": 0.02,
        "recover_nat": 0.05,
        "recover_tx": 0.7,
        "fail_tx": 0.1,
        "relapse": 0.02,
        "screen": 0.05,
        "protect": 0.001,
        "birth_rate": 200.0,
        "output_scale": 1.0,
        "mixing": mixing_stack,
    }
    return TbScaleModel(
        state=state,
        age=age,
        reach=reach,
        pmap=pmap,
        compiled=compiled,
        outputs=outputs,
        params=params,
        save=save,
        flow_names=names,
    )


def measure_jaxpr_eqns(fn: Any, *args: Any) -> int:
    """Equation count of ``jax.make_jaxpr(fn)(*args)``."""
    import jax

    return len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
