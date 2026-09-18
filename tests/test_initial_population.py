"""Declarative InitialPopulation: host validation and traced evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tests.helpers.jaxpr import loop_body_primitives  # noqa: E402

from summer4 import (  # noqa: E402
    REMAINDER,
    Compartments,
    EntryFlow,
    Everything,
    FlowModel,
    InitialPopulation,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Split,
    Time,
)


def _age_state_infect_map() -> tuple[Property, Property, Property, PropertyMap]:
    age = Property("age", ("0", "5", "15"))
    state = Property("state", ("naive", "incipient", "active"))
    infect = Property("infect", ("low", "high"))
    pmap = PropertyMap.from_property(age).stratify(state).stratify(infect, where=state["active"])
    return age, state, infect, pmap


def test_unknown_property_raises() -> None:
    age = Property("age", ("0", "5"))
    other = Property("other", ("a", "b"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="Unknown property"):
        InitialPopulation({age["0"]: 1.0}, splits=(Split(other, {"a": 0.5, "b": 0.5}),)).compile(
            pmap
        )


def test_by_equals_prop_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="must not equal"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, lambda p: jnp.ones(2), by=(age,)),),
        ).compile(pmap)


def test_where_mentions_prop_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="where must not mention"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": 0.5, "5": 0.5}, where=age["0"]),),
        ).compile(pmap)


def test_missing_mapping_keys_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="missing"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": 1.0}),),
        ).compile(pmap)


def test_two_remainders_raises() -> None:
    age = Property("age", ("0", "5", "15"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="REMAINDER"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": REMAINDER, "5": REMAINDER, "15": 0.1}),),
        ).compile(pmap)


def test_negative_weight_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="negative"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": -0.1, "5": 1.1}),),
        ).compile(pmap)


def test_weights_not_sum_one_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="sum to"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": 0.2, "5": 0.2}),),
        ).compile(pmap)


def test_remainder_over_one_raises() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="more than 1"):
        InitialPopulation(
            {Everything(): 1.0},
            splits=(Split(age, {"0": 1.5, "5": REMAINDER}),),
        ).compile(pmap)


def test_step_stage_base_raises() -> None:
    age = Property("age", ("0",))
    pmap = PropertyMap.from_property(age)
    with pytest.raises(ValueError, match="run-stage"):
        InitialPopulation({age["0"]: Time()}).compile(pmap)


def test_empty_base_selector_raises() -> None:
    age = Property("age", ("0", "5"))
    vacc = Property("vacc", ("yes", "no"))
    pmap = PropertyMap.from_property(age)  # no vacc
    with pytest.raises(ValueError, match="no compartments"):
        # Present(vacc) on a map without vacc matches nothing via... actually Present may error.
        # Use a selector that is valid but empty: age that doesn't exist isn't possible.
        # Stratify then select absent combo via age["0"] & impossible — use Nothing-like.
        from summer4 import Nothing

        InitialPopulation({Nothing(): 1.0}).compile(pmap)
    del vacc


def test_free_matrix_pinning_and_ragged() -> None:
    age, state, infect, pmap = _age_state_infect_map()
    plan_pin = InitialPopulation({age["0"] & state["naive"]: 100.0}).compile(pmap)
    age_i = plan_pin.prop_names.index("age")
    assert not bool(plan_pin.free[0, age_i])

    plan_free = InitialPopulation({state["naive"]: 100.0}).compile(pmap)
    assert bool(plan_free.free[0, age_i])

    # On ragged infect: members of state["active"] carry both infect codes → free;
    # members of age["0"] & state["naive"] do not carry infect → not free.
    plan_rag = InitialPopulation({state["active"]: 100.0}).compile(pmap)
    infect_i = plan_rag.prop_names.index("infect")
    assert bool(plan_rag.free[0, infect_i])
    plan_no = InitialPopulation({state["naive"]: 100.0}).compile(pmap)
    assert not bool(plan_no.free[0, infect_i])


def test_rows_j_later_override() -> None:
    age = Property("age", ("young", "old"))
    vacc = Property("vacc", ("one", "two", "none"))
    pmap = PropertyMap.from_property(age).stratify(vacc)
    plan = InitialPopulation(
        {Everything(): 1.0},
        splits=(
            Split(vacc, {"one": 0.5, "two": 0.3, "none": 0.2}),
            Split(vacc, {"one": 0.7, "two": 0.2, "none": 0.1}, where=age["old"]),
        ),
    ).compile(pmap)
    # First split loses old rows; second owns old rows only.
    old_rows = set(int(i) for i in pmap.select(age["old"]))
    young_rows = set(int(i) for i in pmap.select(age["young"]))
    assert set(int(i) for i in plan.splits[0].rows) == young_rows
    assert set(int(i) for i in plan.splits[1].rows) == old_rows


def test_ragged_even_default_tm6() -> None:
    age, state, infect, pmap = _age_state_infect_map()
    plan = InitialPopulation({age[a]: 1000.0 for a in age.traits}).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    for a in age.traits:
        tot = float(y[pmap.select(age[a])].sum())
        np.testing.assert_allclose(tot, 1000.0)
        naive = float(y[pmap.select(age[a] & state["naive"])][0])
        np.testing.assert_allclose(naive, 1000.0 / 3.0)
        inc = float(y[pmap.select(age[a] & state["incipient"])][0])
        np.testing.assert_allclose(inc, 1000.0 / 3.0)
        low = float(y[pmap.select(age[a] & state["active"] & infect["low"])][0])
        high = float(y[pmap.select(age[a] & state["active"] & infect["high"])][0])
        np.testing.assert_allclose(low, 1000.0 / 6.0)
        np.testing.assert_allclose(high, 1000.0 / 6.0)


def test_absolute_by_age_tm6() -> None:
    age, state, _infect, pmap = _age_state_infect_map()
    plan = InitialPopulation({state["naive"] & age[a]: 1000.0 for a in age.traits}).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    for a in age.traits:
        np.testing.assert_allclose(float(y[pmap.select(state["naive"] & age[a])][0]), 1000.0)
    np.testing.assert_allclose(float(y[pmap.select(state["incipient"])].sum()), 0.0)
    np.testing.assert_allclose(float(y[pmap.select(state["active"])].sum()), 0.0)


def _summer2_adjust_map() -> tuple[Any, ...]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    loc = Property("loc", ("urban", "rural"))
    vacc = Property("vacc", ("one_dose", "two_dose", "unvacc"))
    pmap = PropertyMap.from_property(state).stratify(age).stratify(loc).stratify(vacc)
    return state, age, loc, vacc, pmap


def test_summer2_adjust_fixture() -> None:
    state, age, loc, vacc, pmap = _summer2_adjust_map()
    plan = InitialPopulation(
        {state["S"]: 990.0, state["I"]: 10.0},
        splits=(
            Split(age, {"young": 0.6, "old": 0.4}),
            Split(loc, {"urban": 0.8, "rural": 0.2}),
            Split(
                vacc,
                {"one_dose": 0.7, "two_dose": 0.2, "unvacc": 0.1},
                where=age["old"],
            ),
        ),
    ).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    np.testing.assert_allclose(float(y.sum()), 1000.0)
    expected_old_urban = 990.0 * 0.4 * 0.8 * np.array([0.7, 0.2, 0.1])
    got = y[pmap.select(state["S"] & age["old"] & loc["urban"])]
    np.testing.assert_allclose(got, expected_old_urban)
    # Young rows: even vacc default 1/3
    for loc_t, loc_w in (("urban", 0.8), ("rural", 0.2)):
        got_y = y[pmap.select(state["S"] & age["young"] & loc[loc_t])]
        np.testing.assert_allclose(got_y, 990.0 * 0.6 * loc_w * (1.0 / 3.0))


def test_summer2_adjust_multi_filter() -> None:
    state, age, loc, vacc, pmap = _summer2_adjust_map()
    plan = InitialPopulation(
        {state["S"]: 990.0, state["I"]: 10.0},
        splits=(
            Split(age, {"young": 0.6, "old": 0.4}),
            Split(loc, {"urban": 0.8, "rural": 0.2}),
            Split(
                vacc,
                {"one_dose": 0.7, "two_dose": 0.2, "unvacc": 0.1},
                where=age["old"] & loc["urban"],
            ),
        ),
    ).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    np.testing.assert_allclose(float(y.sum()), 1000.0)
    expected = 990.0 * 0.4 * 0.8 * np.array([0.7, 0.2, 0.1])
    got = y[pmap.select(state["S"] & age["old"] & loc["urban"])]
    np.testing.assert_allclose(got, expected)
    # Rural old still even over vacc
    rural = y[pmap.select(state["S"] & age["old"] & loc["rural"])]
    np.testing.assert_allclose(rural, 990.0 * 0.4 * 0.2 / 3.0)


def test_kiribati_shape_ki3() -> None:
    state = Property("state", ("mtb_naive", "clin_inf", "other"))
    age = Property("age", tuple(str(i) for i in range(8)))
    reach = Property("reach", ("reachable", "unreachable"))
    pmap = PropertyMap.from_property(state).stratify(age).stratify(reach)
    plan = InitialPopulation(
        {
            state["mtb_naive"]: Param("pop") - Param("seed"),
            state["clin_inf"]: Param("seed"),
        },
        splits=(Split(reach, {"reachable": Param("frac"), "unreachable": REMAINDER}),),
    ).compile(pmap)
    params = {"pop": 8000.0, "seed": 16.0, "frac": 0.25}
    y = plan.evaluate(params)
    data = np.asarray(y.data)
    expected = (8000.0 - 16.0) / 8.0 * 0.25
    for a in age.traits:
        row = data[pmap.select(state["mtb_naive"] & age[a] & reach["reachable"])]
        np.testing.assert_allclose(float(row[0]), expected)

    def loss(frac: Any) -> Any:
        p = {"pop": 8000.0, "seed": 16.0, "frac": frac}
        yy = plan.evaluate(p)
        idx = pmap.select(reach["reachable"])
        return jnp.sum(jnp.asarray(yy.data)[idx])

    # reachable mass = (pop-seed)*frac + seed*frac = pop*frac
    g = float(jax.grad(loss)(0.25))
    np.testing.assert_allclose(g, 8000.0, rtol=1e-5)


def test_by_function_and_wrong_shape() -> None:
    age = Property("age", ("young", "old"))
    imm = Property("imm", ("yes", "no"))
    pmap = PropertyMap.from_property(age).stratify(imm)
    plan = InitialPopulation(
        {Everything(): 100.0},
        splits=(
            Split(
                imm,
                lambda p: jnp.array([[p["vy"], 1 - p["vy"]], [p["vo"], 1 - p["vo"]]]),
                by=(age,),
            ),
        ),
    ).compile(pmap)
    y = np.asarray(plan.evaluate({"vy": 0.8, "vo": 0.3}).data)
    # Age is free → even 1/2, then imm by-table: young-yes = 100 * 0.5 * 0.8.
    np.testing.assert_allclose(
        float(y[pmap.select(age["young"] & imm["yes"])][0]),
        40.0,
    )
    np.testing.assert_allclose(
        float(y[pmap.select(age["old"] & imm["yes"])][0]),
        15.0,
    )
    bad = InitialPopulation(
        {Everything(): 100.0},
        splits=(Split(imm, lambda p: jnp.array([0.5, 0.5]), by=(age,)),),
    ).compile(pmap)
    with pytest.raises(ValueError, match="shape|rank"):
        bad.evaluate({})


def test_selector_pinning_ignores_zero_weight() -> None:
    age = Property("age", ("0", "5"))
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(age).stratify(state)
    plan = InitialPopulation(
        {age["0"] & state["S"]: 100.0},
        splits=(Split(age, {"0": 0.0, "5": 1.0}, normalize=True),),
    ).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    np.testing.assert_allclose(float(y[pmap.select(age["0"] & state["S"])][0]), 100.0)


def test_overlap_adds() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    plan = InitialPopulation({state["S"]: 10.0, Everything(): 30.0}).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    # Everything splits evenly over free state; S gets 10 + 15.
    np.testing.assert_allclose(float(y[pmap.select(state["S"])][0]), 25.0)
    np.testing.assert_allclose(float(y[pmap.select(state["I"])][0]), 15.0)


def test_normalize_true() -> None:
    age = Property("age", ("a", "b"))
    pmap = PropertyMap.from_property(age)
    plan = InitialPopulation(
        {Everything(): 100.0},
        splits=(Split(age, {"a": 2.0, "b": 6.0}, normalize=True),),
    ).compile(pmap)
    y = np.asarray(plan.evaluate({}).data)
    np.testing.assert_allclose(float(y[pmap.select(age["a"])][0]), 25.0)
    np.testing.assert_allclose(float(y[pmap.select(age["b"])][0]), 75.0)


def test_model_wiring() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], 0.0))
    model.set_initial_population({state["Y"]: 7.0})
    cm = model.compile()
    params: dict[str, float] = {}
    y_auto = cm.run(params, t0=0.0, t1=1.0, dt=0.5, solver="euler")
    y_exp = cm.run(params, cm.initial_state(params), t0=0.0, t1=1.0, dt=0.5, solver="euler")
    np.testing.assert_array_equal(
        np.asarray(y_auto["compartments"].values.data),
        np.asarray(y_exp["compartments"].values.data),
    )
    override = PropertyData.wrap(pmap, np.array([99.0]))
    y_over = cm.run(params, override, t0=0.0, dt=0.5, steps=0, solver="euler")
    np.testing.assert_allclose(float(np.asarray(y_over["compartments"].values.data)[0, 0]), 99.0)

    bare = FlowModel(pmap)
    bare.add_flow(EntryFlow("in", state["Y"], 0.0))
    with pytest.raises(ValueError, match="No initial population"):
        bare.compile().run({}, t0=0.0, t1=1.0, dt=0.5)

    stored = FlowModel(pmap)
    stored.add_flow(EntryFlow("in", state["Y"], 0.0))
    pop = InitialPopulation({state["Y"]: 7.0})
    stored.set_initial_population(pop)
    cm_stored = stored.compile()
    cm_kw = FlowModel(pmap)
    cm_kw.add_flow(EntryFlow("in", state["Y"], 0.0))
    cm_kw = cm_kw.compile(init=pop)
    assert cm_stored == cm_kw

    both = FlowModel(pmap)
    both.add_flow(EntryFlow("in", state["Y"], 0.0))
    both.set_initial_population(pop)
    with pytest.raises(ValueError, match="twice"):
        both.compile(init=pop)


def test_under_jit() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], Param("beta")))
    model.set_initial_population({state["Y"]: Param("y0")})
    cm = model.compile()
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def loss(params: dict[str, float]) -> Any:
        res = cm.run(params, t0=0.0, t1=1.0, dt=0.25, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    val, grads = value_and_grad({"beta": 0.1, "y0": 2.0})
    assert np.isfinite(float(val))
    assert np.isfinite(float(grads["y0"]))


def test_jaxpr_size_independent_of_n() -> None:
    def make(n_traits: int) -> Any:
        age = Property("age", tuple(f"a{i}" for i in range(n_traits)))
        state = Property("state", ("S", "I"))
        pmap = PropertyMap.from_property(age).stratify(state)
        # Callable weights keep the traced graph independent of K (unlike a K-key mapping).
        return InitialPopulation(
            {state["S"]: Param("pop")},
            splits=(Split(age, lambda p, k=n_traits: jnp.ones(k) / float(k)),),
        ).compile(pmap)

    plan3 = make(3)
    plan30 = make(30)
    params = {"pop": 100.0}
    n3 = len(jax.make_jaxpr(plan3.evaluate)(params).jaxpr.eqns)
    n30 = len(jax.make_jaxpr(plan30.evaluate)(params).jaxpr.eqns)
    assert n3 == n30


def test_init_eval_outside_loop() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], 0.0))
    model.set_initial_population({state["Y"]: Param("y0")})
    cm = model.compile()
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def loss(params: dict[str, float]) -> Any:
        res = cm.run(params, t0=0.0, t1=1.0, dt=0.25, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    closed = jax.make_jaxpr(loss)({"y0": 3.0})
    # Init path uses reduce_prod over free-property shares; entry rate is Const so
    # the loop body should not need that prod from init.
    assert "reduce_prod" not in loop_body_primitives(closed)
