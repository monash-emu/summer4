"""Force of infection and mixing matrix tests."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    GroupedRate,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
    derived_refs,
)
from summer4.epi import ForceOfInfection, MixingMatrix
from summer4.results.plan import GroupedOutput


class _ContactParams(NamedTuple):
    contact_rate: float
    mixing: object


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def test_frequency_vs_density_differ_by_denominator() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)

    def build(kind: str) -> object:
        m = FlowModel(pmap)
        foi = ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind=kind,  # type: ignore[arg-type]
            contact_rate=refs.contact_rate,
            mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
        )
        m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
        m.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
        return m.compile()

    # Growing population via unequal compartments so N changes the ratio.
    y = np.array([100.0, 400.0, 10.0, 40.0, 0.0, 0.0])
    params = {"contact_rate": 0.5, "mixing": np.eye(2)}
    freq = np.asarray(build("frequency").observe(0.0, y, params).captures["infection"].data)
    dens = np.asarray(build("density").observe(0.0, y, params).captures["infection"].data)
    n = np.asarray(PropertyData.wrap(pmap, y).sum_over(age).data)
    np.testing.assert_allclose(freq * n, dens)


def test_custom_kind_reimplements_frequency() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)

    def as_freq(infectious: GroupedRate, denominator: GroupedRate) -> GroupedRate:
        return infectious / denominator

    def build(kind: object) -> object:
        m = FlowModel(pmap)
        foi = ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind=kind,  # type: ignore[arg-type]
            contact_rate=refs.contact_rate,
            mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
        )
        m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
        return m.compile()

    y = np.array([100.0, 200.0, 10.0, 20.0, 0.0, 0.0])
    params = {"contact_rate": 1.0}
    a = np.asarray(build("frequency").observe(0.0, y, params).captures["infection"].data)
    b = np.asarray(build(as_freq).observe(0.0, y, params).captures["infection"].data)
    np.testing.assert_array_equal(a, b)


def test_homogeneous_lambda_identical_across_ages() -> None:
    """Identifiability gate: homogeneous mixing → max |λ_a - λ_b| == 0."""
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)
    m = FlowModel(pmap)
    K = np.ones((2, 2)) / 2.0
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="frequency",
        contact_rate=refs.contact_rate,
        mixing=MixingMatrix(age, K, normalize="none", check_reciprocal=False),
    )
    m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    m.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
    cm = m.compile()
    y0 = np.array([900.0, 800.0, 50.0, 40.0, 0.0, 0.0])
    plan = SavePlan(
        requests={"foi": SaveRequest(GroupedOutput("infection"))},
        ts=np.linspace(0.0, 30.0, 31),
    )
    res = cm.run(
        {"contact_rate": 0.4},
        y0,
        t0=0.0,
        t1=30.0,
        dt=0.1,
        save=plan,
        solver="euler",
    )
    vals = np.asarray(res["foi"].values.data)
    assert res["foi"].dims == ("time", "age")
    assert float(np.max(np.abs(vals[:, 0] - vals[:, 1]))) == 0.0


def test_assortative_lambda_differs_across_ages() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)
    m = FlowModel(pmap)
    K = np.array([[0.9, 0.1], [0.1, 0.9]])
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="frequency",
        contact_rate=refs.contact_rate,
        mixing=MixingMatrix(age, K, normalize="rows", check_reciprocal=False),
    )
    m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    m.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
    cm = m.compile()
    y0 = np.array([900.0, 800.0, 80.0, 10.0, 0.0, 0.0])
    plan = SavePlan(
        requests={"foi": SaveRequest(GroupedOutput("infection"))},
        ts=np.linspace(0.0, 30.0, 31),
    )
    res = cm.run(
        {"contact_rate": 0.4},
        y0,
        t0=0.0,
        t1=30.0,
        dt=0.1,
        save=plan,
        solver="euler",
    )
    vals = np.asarray(res["foi"].values.data)
    assert float(np.max(np.abs(vals[:, 0] - vals[:, 1]))) > 0.0


def test_nonsquare_matrix_raises_naming_sizes() -> None:
    age = Property("age", ("young", "old"))
    with pytest.raises(ValueError, match=r"\(2, 2\).*got \(2, 3\)"):
        MixingMatrix(age, np.ones((2, 3)))


def test_reciprocal_check() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    # Equal populations so homogeneous is reciprocal.
    y = np.array([100.0, 100.0, 10.0, 10.0, 0.0, 0.0])
    n = np.asarray(PropertyData.wrap(pmap, y).sum_over(age).data)
    # Reciprocal: K_ab N_a = K_ba N_b
    K_ok = np.array([[0.5, 0.5], [0.5, 0.5]])
    mix_ok = MixingMatrix(age, K_ok, normalize="none", check_reciprocal=True)
    mix_ok.check_reciprocity(K_ok, n)

    K_bad = np.array([[0.9, 0.1], [0.5, 0.5]])
    mix_bad = MixingMatrix(age, K_bad, normalize="none", check_reciprocal=True)
    with pytest.raises(ValueError, match="reciprocal"):
        mix_bad.check_reciprocity(K_bad, n)


def test_foi_matches_hand_derived_fn() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)
    K = np.array([[0.7, 0.3], [0.3, 0.7]])

    m1 = FlowModel(pmap)
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="frequency",
        contact_rate=refs.contact_rate,
        mixing=MixingMatrix(age, K, normalize="none", check_reciprocal=False),
    )
    m1.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    m1.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
    cm1 = m1.compile()

    class _Foi(NamedTuple):
        foi: object

    def derived_fn(params: object, *, y: object, t: object) -> _Foi:
        pd_y = PropertyData(pmap, y)
        i_by = pd_y.keep(state["I"], 0.0).sum_over(age).data
        n_by = pd_y.sum_over(age).data
        shedding = i_by / n_by
        beta = params["contact_rate"]  # type: ignore[index]
        foi_v = beta * (K @ shedding)
        return _Foi(foi=pd_y.broadcast_over(age, foi_v))

    m2 = FlowModel(pmap)
    fr = derived_refs(_Foi)
    m2.add_flow(TransitionFlow("inf", state["S"], state["I"], fr.foi))
    m2.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
    cm2 = m2.compile(derived_fn=derived_fn)

    y0 = np.array([900.0, 800.0, 50.0, 40.0, 50.0, 60.0])
    params = {"contact_rate": 0.35}
    plan = SavePlan(
        requests={"comp": SaveRequest(Compartments())},
        ts=np.linspace(0.0, 40.0, 41),
    )
    r1 = cm1.run(params, y0, t0=0.0, t1=40.0, dt=0.1, save=plan, solver="euler")
    r2 = cm2.run(params, y0, t0=0.0, t1=40.0, dt=0.1, save=plan, solver="euler")
    np.testing.assert_allclose(
        np.asarray(r1["comp"].values.data),
        np.asarray(r2["comp"].values.data),
        atol=1e-12,
    )


def test_grad_contact_rate_finite() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)
    m = FlowModel(pmap)
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="frequency",
        contact_rate=refs.contact_rate,
        mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
    )
    m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    m.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
    cm = m.compile()
    y0 = jnp.asarray([900.0, 800.0, 50.0, 40.0, 50.0, 60.0])

    def loss(beta: object) -> object:
        dy = cm.vector_field(0.0, y0, {"contact_rate": beta})
        return jnp.sum(jnp.asarray(dy) ** 2)

    g = jax.grad(loss)(0.4)
    assert jnp.isfinite(g)
    # Central differences
    eps = 1e-5
    num = (loss(0.4 + eps) - loss(0.4 - eps)) / (2 * eps)
    np.testing.assert_allclose(float(g), float(num), rtol=1e-3)


def test_infectiousness_normalize_population() -> None:
    """6.4: constant rescale of weights is a no-op under normalize='population'."""
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    refs = derived_refs(_ContactParams)

    def build(weights: dict[str, float], normalize: str | None) -> object:
        m = FlowModel(pmap)
        foi = ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind="frequency",
            contact_rate=refs.contact_rate,
            mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
            infectiousness={age[k]: v for k, v in weights.items()},
            normalize_infectiousness=normalize,  # type: ignore[arg-type]
        )
        m.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
        m.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.1))
        return m.compile()

    y0 = np.array([900.0, 800.0, 50.0, 40.0, 50.0, 60.0])
    params = {"contact_rate": 0.4}
    plan = SavePlan(
        requests={"comp": SaveRequest(Compartments())},
        ts=np.linspace(0.0, 20.0, 21),
    )
    w1 = {"young": 0.7, "old": 1.4}
    w2 = {"young": 1.4, "old": 2.8}  # 2× scale
    r1 = build(w1, "population").run(params, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    r2 = build(w2, "population").run(params, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    np.testing.assert_allclose(
        np.asarray(r1["comp"].values.data),
        np.asarray(r2["comp"].values.data),
        atol=1e-10,
    )
    r3 = build(w1, None).run(params, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    r4 = build(w2, None).run(params, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    assert not np.allclose(
        np.asarray(r3["comp"].values.data),
        np.asarray(r4["comp"].values.data),
    )


def test_infectiousness_wrong_property_raises() -> None:
    state, age, pmap = _sir_age()
    loc = Property("location", ("north", "south"))
    with pytest.raises(ValueError, match="location.*age|age.*location"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            infectiousness={loc["north"]: 1.0},
        )


def test_param_is_field_ref() -> None:
    from summer4.epi import Param
    from summer4.flows.rates import FieldRef

    p = Param("beta")
    assert isinstance(p, FieldRef)
    assert p.path == ("beta",)
