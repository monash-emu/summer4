"""Whole-population force of infection: ``ForceOfInfection(group_by=None)``."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    FlowMass,
    FlowModel,
    GroupedRate,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Target,
    TargetSet,
    TransitionFlow,
)
from summer4.epi import BayesianModel, FOIKind, ForceOfInfection, MixingMatrix, Poisson, Uniform
from summer4.flows.compiled import CompiledModel

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("child", "adult"))
PMAP = PropertyMap.from_property(STATE).stratify(AGE)
# Unequal prevalence by age: 1% of children and 20% of adults are infectious.
Y = np.array([990.0, 1200.0, 10.0, 300.0, 0.0, 0.0])
I_TOTAL = 310.0
N_TOTAL = 2500.0
RUN = dict(t0=0.0, t1=60.0, dt=0.5, solver="euler")


def _sir(foi: object, pmap: PropertyMap = PMAP) -> CompiledModel:
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], foi))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], 0.1))
    return model.compile()


def _capture(cm: CompiledModel, name: str, y: Any = Y, params: Any = None) -> np.ndarray:
    return np.asarray(cm.observe(0.0, y, params or {"beta": 0.2}).captures[name].data)


def _susceptible_per_capita(cm: CompiledModel, params: Any = None) -> np.ndarray:
    dy = np.asarray(cm.vector_field(0.0, PropertyData.wrap(PMAP, Y), params or {"beta": 0.2}).data)
    s_idx = PMAP.select(STATE["S"])
    # Recovery does not touch S, so -dS/S is the per-capita force of infection.
    return -dy[s_idx] / Y[s_idx]


def test_group_by_defaults_to_none() -> None:
    foi = ForceOfInfection("foi", infectious=STATE["I"])
    assert foi.group_by is None


def test_frequency_is_total_infectious_over_total_population() -> None:
    cm = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    np.testing.assert_allclose(_susceptible_per_capita(cm), 0.2 * I_TOTAL / N_TOTAL, rtol=1e-6)
    capture = _capture(cm, "foi")
    assert capture.shape == ()
    np.testing.assert_allclose(capture, 0.2 * I_TOTAL / N_TOTAL, rtol=1e-6)


def test_differs_from_per_group_without_mixing() -> None:
    whole = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    per_age = _sir(
        ForceOfInfection("foi", infectious=STATE["I"], group_by=AGE, contact_rate=Param("beta"))
    )
    per_age_rate = _susceptible_per_capita(per_age)
    np.testing.assert_allclose(per_age_rate, [0.2 * 10 / 1000, 0.2 * 300 / 1500], rtol=1e-6)
    assert not np.allclose(_susceptible_per_capita(whole), per_age_rate)


def test_matches_dummy_population_idiom_trajectory() -> None:
    whole = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    pop = Property("pop", ("all",))
    dummy_map = PMAP.stratify(pop)
    dummy = _sir(
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            group_by=pop,
            mixing=MixingMatrix(pop, np.array([[1.0]]), check_reciprocal=False),
            contact_rate=Param("beta"),
        ),
        dummy_map,
    )
    a = whole.run({"beta": 0.3}, Y, **RUN)
    b = dummy.run({"beta": 0.3}, Y, **RUN)
    np.testing.assert_allclose(
        np.asarray(a["compartments"].values.data),
        np.asarray(b["compartments"].values.data),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(a["foi"].values).reshape(-1),
        np.asarray(b["foi"].values.data).reshape(-1),
        rtol=1e-6,
    )


def test_default_save_has_foi_series() -> None:
    cm = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    res = cm.run({"beta": 0.2}, Y, **RUN)
    out = res["foi"]
    assert out.dims == ("time",)
    comps = np.asarray(res["compartments"].values.data)
    expected = 0.2 * comps[:, PMAP.select(STATE["I"])].sum(axis=1) / comps.sum(axis=1)
    np.testing.assert_allclose(np.asarray(out.values).reshape(-1), expected, rtol=1e-5)


def test_density_and_generalised() -> None:
    density = _sir(
        ForceOfInfection(
            "foi", infectious=STATE["I"], kind=FOIKind.DENSITY, contact_rate=Param("beta")
        )
    )
    np.testing.assert_allclose(_capture(density, "foi"), 0.2 * I_TOTAL, rtol=1e-6)
    generalised = _sir(
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            kind=FOIKind.GENERALISED,
            exponent=0.5,
            contact_rate=Param("beta"),
        )
    )
    np.testing.assert_allclose(
        _capture(generalised, "foi"), 0.2 * I_TOTAL / np.sqrt(N_TOTAL), rtol=1e-6
    )


def test_denominator_selector() -> None:
    cm = _sir(
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            denominator=AGE["adult"],
            contact_rate=Param("beta"),
        )
    )
    np.testing.assert_allclose(_capture(cm, "foi"), 0.2 * I_TOTAL / 1500.0, rtol=1e-6)


def test_infectiousness_pairs_and_normalisation() -> None:
    weights = [(AGE["child"], 2.0)]
    raw = _sir(
        ForceOfInfection(
            "foi", infectious=STATE["I"], infectiousness=weights, contact_rate=Param("beta")
        )
    )
    weighted_i = 2.0 * 10.0 + 300.0
    np.testing.assert_allclose(_capture(raw, "foi"), 0.2 * weighted_i / N_TOTAL, rtol=1e-6)

    population = _sir(
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            infectiousness=weights,
            normalize_infectiousness="population",
            contact_rate=Param("beta"),
        )
    )
    mean_w_pop = (2.0 * 1000.0 + 1.0 * 1500.0) / N_TOTAL
    np.testing.assert_allclose(
        _capture(population, "foi"), 0.2 * weighted_i / mean_w_pop / N_TOTAL, rtol=1e-6
    )

    mean = _sir(
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            infectiousness=weights,
            normalize_infectiousness="mean",
            contact_rate=Param("beta"),
        )
    )
    # Three child compartments weigh 2, three adult compartments weigh 1.
    np.testing.assert_allclose(_capture(mean, "foi"), 0.2 * weighted_i / 1.5 / N_TOTAL, rtol=1e-6)


def test_custom_kind_receives_ungrouped_rates() -> None:
    seen: list[tuple[Property, ...]] = []

    def freq(infectious: GroupedRate, denominator: GroupedRate) -> GroupedRate:
        seen.append(infectious.properties)
        seen.append(denominator.properties)
        return infectious / denominator

    cm = _sir(ForceOfInfection("foi", infectious=STATE["I"], kind=freq, contact_rate=0.2))
    np.testing.assert_allclose(_capture(cm, "foi"), 0.2 * I_TOTAL / N_TOTAL, rtol=1e-6)
    assert seen and all(props == () for props in seen)


def test_mixing_without_group_by_raises() -> None:
    with pytest.raises(ValueError, match="needs group_by"):
        ForceOfInfection(
            "foi",
            infectious=STATE["I"],
            mixing=MixingMatrix(AGE, np.ones((2, 2)), check_reciprocal=False),
        )


def test_trait_mapping_infectiousness_without_group_by_raises() -> None:
    with pytest.raises(TypeError, match="group_by is None"):
        ForceOfInfection("foi", infectious=STATE["I"], infectiousness={"child": 2.0})


def test_digest_distinguishes_whole_population() -> None:
    whole = ForceOfInfection("foi", infectious=STATE["I"])
    again = ForceOfInfection("foi", infectious=STATE["I"])
    grouped = ForceOfInfection("foi", infectious=STATE["I"], group_by=AGE)
    assert whole.__rate_bytes__() == again.__rate_bytes__()
    assert whole.__rate_bytes__() != grouped.__rate_bytes__()


def test_jit_and_grad() -> None:
    cm = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    y = PropertyData.wrap(PMAP, Y)

    def infection_rate(beta: Any) -> Any:
        dy = cm.vector_field(0.0, y, {"beta": beta}).data
        return -jnp.sum(dy[PMAP.select(STATE["S"])])

    value, grad = jax.jit(jax.value_and_grad(infection_rate))(0.2)
    expected = 0.2 * I_TOTAL / N_TOTAL * (990.0 + 1200.0)
    np.testing.assert_allclose(float(value), expected, rtol=1e-5)
    np.testing.assert_allclose(float(grad), expected / 0.2, rtol=1e-5)


def test_bayesian_model_prefers_true_contact_rate() -> None:
    cm = _sir(ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")))
    times = np.arange(5.0, 41.0, 5.0)
    truth = cm.run({"beta": 0.3}, Y, **RUN)
    observed = np.asarray(truth["infection"].total().at_times(times).values).reshape(-1)
    bm = BayesianModel(
        cm,
        {},
        priors=[Uniform("beta", 0.05, 1.0)],
        targets=TargetSet(
            (
                Target(
                    key="infection",
                    times=times,
                    values=np.round(observed),
                    quantity=FlowMass("infection"),
                    reduce="sum",
                    likelihood=Poisson("sum"),
                ),
            )
        ),
        y0=Y,
        run_kwargs=RUN,
    )
    at_truth = float(bm.log_likelihood({"beta": 0.3}))
    assert at_truth > float(bm.log_likelihood({"beta": 0.2}))
    assert at_truth > float(bm.log_likelihood({"beta": 0.4}))
