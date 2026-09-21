"""Defer: arbitrary callables in the rate slot."""

from __future__ import annotations

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from summer4 import (  # noqa: E402
    Defer,
    EntryFlow,
    Everything,
    ExitFlow,
    FlowModel,
    FlowRef,
    GroupedOutput,
    InitialPopulation,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
    Time,
    Transform,
    defer,
)
from summer4.flows.rates import _RATE_EVALUATORS, Capture, _rate_bytes  # noqa: E402
from summer4.flows.stages import build_hoist_table, rate_stage  # noqa: E402


def _scale(value: object) -> object:
    """Module-level callable. Rebuilds that use this object share a digest."""
    return jnp.asarray(value) * 2.0


def _entry_field(expr: object, t: float, params: dict[str, float]) -> float:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    compiled = model.compile()
    y = PropertyData.wrap(pmap, np.array([0.0]))
    return float(np.asarray(compiled.vector_field(t, y, params).data)[0])


def test_defer_matches_the_transform_workaround() -> None:
    def hazard(t: object, amp: object) -> object:
        return 0.1 * (1.0 + amp * jnp.sin(t))

    def via_adjust(prev: object, t: object, amp: object) -> object:
        del prev
        return hazard(t, amp)

    direct = _entry_field(defer(hazard)(Time(), Param("amp")), 0.5, {"amp": 0.25})
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(
        EntryFlow(
            "in",
            state["Y"],
            1.0,
            adjust=(Transform(via_adjust, Time(), Param("amp")),),
        )
    )
    y = PropertyData.wrap(pmap, np.array([0.0]))
    workaround = float(np.asarray(model.compile().vector_field(0.5, y, {"amp": 0.25}).data)[0])
    np.testing.assert_allclose(direct, workaround, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(direct, 0.1 * (1.0 + 0.25 * np.sin(0.5)), rtol=1e-5, atol=1e-5)


def test_staging_is_the_join_of_the_arguments() -> None:
    def f(*_args: object) -> float:
        return 0.0

    assert rate_stage(defer(f)(Param("x"), Param("y")), params_are_static=True) == "run"
    assert rate_stage(defer(f)(Time(), Param("amp")), params_are_static=True) == "step"
    # Parameters are dynamic: a param-only Defer is step-stage, not run.
    assert rate_stage(defer(f)(Param("x")), params_are_static=False) == "step"


def test_param_only_defer_hoists_and_a_step_defer_hoists_its_argument() -> None:
    def f(*_args: object) -> float:
        return 0.0

    param_only = defer(f)(Param("x"), Param("y"))
    table = build_hoist_table([param_only], params_are_static=True)
    assert len(table.entries) == 1
    assert table.entries[0].node == param_only

    product = Param("a") * Param("b")
    stepped = defer(f)(product, Time())
    stepped_table = build_hoist_table([stepped], params_are_static=True)
    assert len(stepped_table.entries) == 1
    assert stepped_table.entries[0].node == product


def test_closures_with_different_captures_do_not_share_a_digest() -> None:
    # Anti-regression for keying the digest on fn.__qualname__. Both closures
    # would be ``<lambda>`` / ``build.<locals>.cl`` and would collide.
    def build(scale: float) -> object:
        def cl(value: object) -> object:
            return jnp.asarray(value) * scale

        return defer(cl)(Param("x"))

    left = build(1.0)
    right = build(2.0)
    assert _rate_bytes(left) != _rate_bytes(right)  # type: ignore[arg-type]
    assert _entry_field(left, 0.0, {"x": 3.0}) == pytest.approx(3.0)
    assert _entry_field(right, 0.0, {"x": 3.0}) == pytest.approx(6.0)


def test_name_shares_a_cache_entry_across_distinct_functions() -> None:
    def first(value: object) -> object:
        return jnp.asarray(value) + 1.0

    def second(value: object) -> object:
        return jnp.asarray(value) + 1.0

    left = defer(first, name="plus-one")(Param("x"))
    right = defer(second, name="plus-one")(Param("x"))
    assert first is not second
    assert _rate_bytes(left) == _rate_bytes(right)
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def compile_with(expr: object) -> object:
        model = FlowModel(pmap)
        model.add_flow(EntryFlow("in", state["Y"], expr))
        return model.compile()

    a = compile_with(left)
    b = compile_with(right)
    assert a == b
    y = PropertyData.wrap(pmap, np.array([0.0]))
    traces = {"n": 0}

    @functools.partial(jax.jit, static_argnums=0)
    def _vf(model: object, state_y: PropertyData, params: dict[str, float]) -> object:
        traces["n"] += 1
        return model.vector_field(0.0, state_y, params)  # type: ignore[attr-defined]

    _vf(a, y, {"x": 1.0})
    _vf(b, y, {"x": 4.0})
    assert traces["n"] == 1


def test_trace_counts_for_draws_and_inline_rebuilds() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    y = PropertyData.wrap(pmap, np.array([0.0]))
    traces = {"n": 0}

    @functools.partial(jax.jit, static_argnums=0)
    def _vf(model: object, state_y: PropertyData, params: dict[str, float]) -> object:
        traces["n"] += 1
        return model.vector_field(0.0, state_y, params)  # type: ignore[attr-defined]

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], defer(_scale)(Param("x"))))
    compiled = model.compile()
    for value in range(50):
        _vf(compiled, y, {"x": float(value)})
    assert traces["n"] == 1

    traces["n"] = 0
    for value in range(50):

        def inline(raw: object, _value: int = value) -> object:
            return jnp.asarray(raw) * _value

        rebuilt = FlowModel(pmap)
        rebuilt.add_flow(EntryFlow("in", state["Y"], defer(inline)(Param("x"))))
        _vf(rebuilt.compile(), y, {"x": 1.0})
    assert traces["n"] == 50

    traces["n"] = 0

    @functools.partial(jax.jit, static_argnums=0)
    def _vf_module(model: object, state_y: PropertyData, params: dict[str, float]) -> object:
        traces["n"] += 1
        return model.vector_field(0.0, state_y, params)  # type: ignore[attr-defined]

    for _ in range(50):
        rebuilt = FlowModel(pmap)
        rebuilt.add_flow(EntryFlow("in", state["Y"], defer(_scale)(Param("x"))))
        _vf_module(rebuilt.compile(), y, {"x": 1.0})
    assert traces["n"] == 1


def test_keyword_order_does_not_change_the_node() -> None:
    def f(t: object, amp: object) -> object:
        return jnp.asarray(t) * amp

    left = defer(f)(t=Time(), amp=Param("amp"))
    right = defer(f)(amp=Param("amp"), t=Time())
    assert left == right
    assert _rate_bytes(left) == _rate_bytes(right)
    assert _entry_field(left, 2.0, {"amp": 0.5}) == pytest.approx(1.0)


def test_flow_ref_inside_defer_is_ordered_and_a_cycle_raises() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    death = model.add_flow(ExitFlow("death", Everything(), 0.1))
    model.add_flow(EntryFlow("birth", state["S"], defer(lambda mass: mass)(death.sum())))
    y = np.array([10.0, 5.0])
    dy = np.asarray(model.compile().vector_field(0.0, y, {}))
    np.testing.assert_allclose(dy, [-1.0 + 1.5, -0.5], rtol=1e-5, atol=1e-5)

    cyclic = FlowModel(pmap)
    cyclic.add_flow(ExitFlow("a", state["S"], defer(lambda mass: mass)(FlowRef("b"))))
    cyclic.add_flow(ExitFlow("b", state["I"], defer(lambda mass: mass)(FlowRef("a"))))
    with pytest.raises(ValueError, match="Cyclic"):
        cyclic.compile()


def test_capture_inside_defer_is_saveable() -> None:
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    pool = Capture("pool", Reduce(sum_over=age))
    model.add_flow(ExitFlow("out", Everything(), defer(lambda grouped: grouped)(pool)))
    y0 = PropertyData.wrap(pmap, np.array([1.0, 2.0, 3.0, 4.0]))
    plan = SavePlan(requests={"pool": SaveRequest(GroupedOutput("pool"))}, ts=np.array([0.0]))
    result = model.compile().run({}, y0, t0=0.0, t1=0.0, dt=1.0, save=plan)
    saved = np.asarray(result["pool"].values)
    # S+I in each age band: young 1+3, old 2+4.
    np.testing.assert_allclose(saved.reshape(-1)[:2], [4.0, 6.0], rtol=1e-5, atol=1e-5)


def test_nested_defer_evaluates_and_stages() -> None:
    def add_one(value: object) -> object:
        return jnp.asarray(value) + 1.0

    def twice(value: object) -> object:
        return jnp.asarray(value) * 2.0

    expr = defer(twice)(defer(add_one)(Param("x")))
    assert rate_stage(expr, params_are_static=True) == "run"
    assert _entry_field(expr, 0.0, {"x": 3.0}) == pytest.approx(8.0)


def test_initial_population_accepts_param_only_and_rejects_time() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def twice(value: object) -> object:
        return jnp.asarray(value) * 2.0

    pop = InitialPopulation({Everything(): defer(twice)(Param("n"))})
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], 0.0))
    model.set_initial_population(pop)
    y0 = model.compile().initial_state({"n": 4.0})
    np.testing.assert_allclose(np.asarray(y0.data), [8.0], rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="Time"):
        InitialPopulation({Everything(): defer(lambda t: t)(Time())}).compile(pmap)


def test_defer_registers_under_the_total_digest() -> None:
    assert Defer in _RATE_EVALUATORS
    assert callable(Defer.__rate_bytes__)
    node = Defer(lambda value: value, Param("x"))
    assert _rate_bytes(node).startswith(b"defer")
