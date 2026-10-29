"""Scalar ``Reduce``: the number of individuals in matching compartments (CX-scalar-reduce)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from summer4 import (
    Compartments,
    EntryFlow,
    ExitFlow,
    FlowModel,
    FlowRef,
    Param,
    Property,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
)
from summer4.epi import FOIKind, ForceOfInfection
from summer4.flows.rates import _rate_bytes

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

SPECIES = Property("species", ("prey", "predator"))


def _strata(n: int) -> Property:
    return Property("zone", tuple(f"z{i}" for i in range(n)))


def _predation_model(n_zones: int = 0, rate: Any = None) -> tuple[FlowModel, PropertyMap]:
    pmap = PropertyMap.from_property(SPECIES)
    if n_zones:
        pmap = pmap.stratify(_strata(n_zones))
    model = FlowModel(pmap)
    if rate is None:
        rate = Param("a") * Reduce(where=SPECIES["predator"])
    model.add_flow(ExitFlow("predation", SPECIES["prey"], rate))
    return model, pmap


def test_scalar_reduce_unstratified_flow_value() -> None:
    model, _ = _predation_model()
    y = np.array([40.0, 15.0])
    ctx = model.compile().observe(0.0, y, {"a": 0.02})
    np.testing.assert_allclose(np.asarray(ctx.flows["predation"]), [0.02 * 15.0 * 40.0])


def test_scalar_reduce_sums_across_strata() -> None:
    model, pmap = _predation_model(2)
    y = np.arange(1.0, pmap.size + 1.0)
    pred_idx = pmap.select(SPECIES["predator"])
    prey_idx = pmap.select(SPECIES["prey"])
    assert len(pred_idx) == 2 and len(prey_idx) == 2
    ctx = model.compile().observe(0.0, y, {"a": 0.5})
    flow = np.asarray(ctx.flows["predation"])
    np.testing.assert_allclose(np.sort(flow), np.sort(0.5 * y[pred_idx].sum() * y[prey_idx]))


def test_empty_reduce_is_total_population() -> None:
    model, pmap = _predation_model(2, rate=Reduce())
    y = np.arange(1.0, pmap.size + 1.0)
    ctx = model.compile().observe(0.0, y, {})
    prey_idx = pmap.select(SPECIES["prey"])
    np.testing.assert_allclose(
        np.sort(np.asarray(ctx.flows["predation"])), np.sort(y.sum() * y[prey_idx])
    )


def _lv(use_reduce: bool) -> FlowModel:
    pmap = PropertyMap.from_property(SPECIES)
    m = FlowModel(pmap)

    def count(name: str, counted: str, rate: Any) -> Any:
        if use_reduce:
            return rate * Reduce(where=SPECIES[counted])
        return ForceOfInfection(
            name, infectious=SPECIES[counted], kind=FOIKind.DENSITY, contact_rate=rate
        )

    m.add_flow(EntryFlow("prey_birth", SPECIES["prey"], count("b", "prey", Param("alpha"))))
    m.add_flow(ExitFlow("prey_death", SPECIES["prey"], Param("mu")))
    m.add_flow(ExitFlow("predation", SPECIES["prey"], count("p", "predator", Param("a"))))
    m.add_flow(
        EntryFlow("predator_birth", SPECIES["predator"], Param("e") * FlowRef("predation").sum_by())
    )
    m.add_flow(ExitFlow("predator_death", SPECIES["predator"], Param("d")))
    m.set_initial_population({SPECIES["prey"]: 40.0, SPECIES["predator"]: 15.0})
    return m


LV_PARAMS = {"alpha": 0.6, "mu": 0.1, "a": 0.02, "e": 0.5, "d": 0.25}


def _trajectory(model: FlowModel) -> np.ndarray:
    plan = SavePlan(requests={"c": SaveRequest(Compartments())}, ts=np.linspace(0.0, 50.0, 201))
    res = model.compile().run(
        LV_PARAMS, t0=0.0, t1=50.0, dt=0.1, save=plan, solver="dopri5", rtol=1e-10, atol=1e-10
    )
    return np.asarray(res["c"].values.data)


def test_lotka_volterra_matches_force_of_infection_helper() -> None:
    a, b = _trajectory(_lv(True)), _trajectory(_lv(False))
    print("LV max abs diff", np.abs(a - b).max())
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_scalar_reduce_jit_and_grad() -> None:
    model, _ = _predation_model(2)
    cm = model.compile()
    y = jnp.array([4.0, 3.0, 2.0, 1.0])

    def total(a: Any) -> Any:
        return jnp.sum(cm.vector_field(0.0, y, {"a": a}) ** 2)

    value = jax.jit(total)(0.3)
    assert np.isfinite(float(value))
    g = float(jax.grad(total)(0.3))
    eps = 1e-6
    fd = (float(total(0.3 + eps)) - float(total(0.3 - eps))) / (2 * eps)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def _n_eqns(n_zones: int) -> int:
    model, pmap = _predation_model(n_zones)
    cm = model.compile()
    jaxpr = jax.make_jaxpr(cm.vector_field)(0.0, np.ones(pmap.size), {"a": 0.1})
    return len(jaxpr.jaxpr.eqns)


def test_jaxpr_size_independent_of_compartment_count() -> None:
    assert _n_eqns(2) == _n_eqns(20)


def test_digest_distinguishes_scalar_grouped_and_selector() -> None:
    a, b = SPECIES["prey"], SPECIES["predator"]
    da, db = _rate_bytes(Reduce(where=a)), _rate_bytes(Reduce(where=b))
    dg = _rate_bytes(Reduce(SPECIES, where=a))
    assert len({da, db, dg}) == 3
    assert _rate_bytes(Reduce(where=a)) == da
    assert _rate_bytes(Reduce(SPECIES, where=a)) == dg
    assert _rate_bytes(Reduce()) != da
    assert _rate_bytes(Reduce()) != _rate_bytes(Reduce(SPECIES))


def test_reduce_without_arguments_constructs() -> None:
    r = Reduce()
    assert r.group_by is None and r.where is None
