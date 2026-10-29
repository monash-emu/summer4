"""``named(fn, name)``: every callable hook keys its digest on ``name``.

Each builder creates a fresh closure on every call, so a bare build misses the
jit cache by identity. A ``named`` build with the same name must hit it, a
different name must not, and a wrong name is a wrong answer (the contract in
:class:`summer4.flows.rates.Named`).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from summer4 import (  # noqa: E402
    Compartments,
    ComputedValue,
    Defer,
    EntryFlow,
    Everything,
    FlowModel,
    InitialPopulation,
    Named,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SaveFn,
    SavePlan,
    SaveRequest,
    Time,
    Transform,
    TransitionFlow,
    defer,
    derived_refs,
    named,
)
from summer4.epi import ForceOfInfection, MixingMatrix  # noqa: E402
from summer4.flows.rates import _callable_key, _rate_bytes  # noqa: E402

type Builder = Callable[[str | None], object]


def _maybe_named(fn: Callable[..., Any], name: str | None) -> Callable[..., Any]:
    """The bare closure when ``name`` is ``None``, else its ``named`` wrapper."""
    return fn if name is None else named(fn, name)


def _state_pmap() -> tuple[Property, PropertyMap]:
    state = Property("state", ("Y",))
    return state, PropertyMap.from_property(state)


def _build_transform(name: str | None) -> object:
    state, pmap = _state_pmap()

    def seasonal(prev: Any, t: Any) -> Any:
        return jnp.asarray(prev) * (1.0 + 0.1 * jnp.sin(t / 30.0))

    model = FlowModel(pmap)
    adjust = (Transform(_maybe_named(seasonal, name), Time()),)
    model.add_flow(EntryFlow("in", state["Y"], 1.0, adjust=adjust))
    return model.compile()


def _build_save(name: str | None) -> object:
    def harvest(ctx: Any) -> Any:
        return jnp.asarray(ctx.t) * 2.0

    fn = SaveFn(fn=_maybe_named(harvest, name))
    return SavePlan(requests={"harvest": SaveRequest(fn)}, ts=np.array([1.0]))


class _Derived(NamedTuple):
    rec: Any


def _build_derived(name: str | None) -> object:
    state, pmap = _state_pmap()
    refs = derived_refs(_Derived)

    def derived(params: Any, *, y: Any, t: Any) -> _Derived:
        del y
        return _Derived(rec=0.1 * (1.0 + params["amp"] * jnp.sin(t / 10.0) ** 2))

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], refs.rec))
    return model.compile(derived_fn=_maybe_named(derived, name))


def _build_prepare(name: str | None) -> object:
    state, pmap = _state_pmap()

    def prepare(p: dict[str, Any]) -> dict[str, Any]:
        return {**p, "beta": p["r0"] / p["period"]}

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], Param("beta")))
    return model.compile(prepare_fn=_maybe_named(prepare, name))


def _build_initial(name: str | None) -> object:
    state, pmap = _state_pmap()

    def base(n: Any) -> Any:
        return jnp.asarray(n) * 2.0

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], 0.0))
    population = InitialPopulation({Everything(): defer(_maybe_named(base, name))(Param("n"))})
    model.set_initial_population(population)
    return model.compile()


class _ContactParams(NamedTuple):
    contact_rate: float
    mixing: object


def _build_foi(name: str | None) -> object:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    refs = derived_refs(_ContactParams)

    def as_freq(infectious: Any, denominator: Any) -> Any:
        return infectious / denominator

    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind=_maybe_named(as_freq, name),
        contact_rate=refs.contact_rate,
        mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
    )
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    return model.compile()


_HOOKS = (
    pytest.param(_build_transform, id="transform"),
    pytest.param(_build_save, id="save-fn"),
    pytest.param(_build_derived, id="derived-fn"),
    pytest.param(_build_prepare, id="prepare-fn"),
    pytest.param(_build_initial, id="initial-base"),
    pytest.param(_build_foi, id="foi-kind"),
)


@pytest.mark.parametrize("build", _HOOKS)
def test_same_name_gives_an_equal_digest(build: Builder) -> None:
    left = build("same")
    right = build("same")
    assert left == right
    assert hash(left) == hash(right)


@pytest.mark.parametrize("build", _HOOKS)
def test_different_names_give_different_digests(build: Builder) -> None:
    assert build("left") != build("right")


@pytest.mark.parametrize("build", _HOOKS)
def test_bare_fresh_closures_give_different_digests(build: Builder) -> None:
    first = build(None)
    second = build(None)
    assert first != second
    assert build("named") != first


def test_named_transform_rebuilds_cost_one_trace_and_bare_rebuilds_cost_five() -> None:
    _state, pmap = _state_pmap()
    y = PropertyData.wrap(pmap, np.array([1.0]))
    traces = {"n": 0}

    @functools.partial(jax.jit, static_argnums=0)
    def run_once(model: Any, state_y: PropertyData) -> Any:
        traces["n"] += 1
        return model.vector_field(0.5, state_y, {})

    # Keep every model alive: a freed closure could hand its id to the next one.
    named_models = [_build_transform("seasonal") for _ in range(5)]
    for model in named_models:
        run_once(model, y)
    assert traces["n"] == 1

    traces["n"] = 0
    bare_models = [_build_transform(None) for _ in range(5)]
    for model in bare_models:
        run_once(model, y)
    assert traces["n"] == 5


def test_named_transform_gives_the_bare_trajectory() -> None:
    _state, pmap = _state_pmap()
    y0 = PropertyData.wrap(pmap, np.array([1.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0, 2.0]))

    def trajectory(model: object) -> np.ndarray:
        compiled = cast(Any, model)
        res = compiled.run({}, y0, t0=0.0, t1=2.0, dt=0.25, save=plan, solver="euler")
        return np.asarray(res["y"].values.data)

    named_traj = trajectory(_build_transform("seasonal"))
    bare_traj = trajectory(_build_transform(None))
    np.testing.assert_array_equal(named_traj, bare_traj)
    assert named_traj.shape[0] == 2


@pytest.mark.parametrize("name", ["d", None])
def test_derived_fn_still_validates_computed_paths(name: str | None) -> None:
    _state, pmap = _state_pmap()
    y0 = PropertyData.wrap(pmap, np.array([1.0]))
    params = {"amp": 0.5}

    def save(path: tuple[str, ...]) -> SavePlan:
        return SavePlan(requests={"x": SaveRequest(ComputedValue(path=path))}, ts=np.array([1.0]))

    compiled = cast(Any, _build_derived(name))
    with pytest.raises(ValueError, match="not a field of the derived schema"):
        compiled.run(params, y0, t0=0.0, t1=1.0, dt=0.5, save=save(("nope",)), solver="euler")

    res = compiled.run(params, y0, t0=0.0, t1=1.0, dt=0.5, save=save(("rec",)), solver="euler")
    expected = 0.1 * (1.0 + 0.5 * np.sin(1.0 / 10.0) ** 2)
    np.testing.assert_allclose(np.asarray(res["x"].values).reshape(-1)[-1], expected, rtol=1e-5)


def test_named_wraps_forwards_and_validates() -> None:
    def inc(value: Any) -> Any:
        return value + 1

    wrapped = named(inc, "inc")
    assert isinstance(wrapped, Named)
    assert wrapped(2) == 3
    assert wrapped == named(inc, "inc")
    assert _callable_key(wrapped) == "n:inc"
    assert _callable_key(inc) == str(id(inc))
    with pytest.raises(ValueError, match="non-empty"):
        named(inc, "")
    with pytest.raises(TypeError, match="callable"):
        named(cast(Any, 3), "x")
    with pytest.raises(TypeError, match="string name"):
        named(inc, cast(Any, 3))


def test_defer_of_a_named_function_keys_on_the_name() -> None:
    def first(value: Any) -> Any:
        return jnp.asarray(value) + 1.0

    def second(value: Any) -> Any:
        return jnp.asarray(value) + 1.0

    left = defer(named(first, "plus-one"))(Param("x"))
    right = defer(named(second, "plus-one"))(Param("x"))
    assert isinstance(left, Defer)
    assert _rate_bytes(left) == _rate_bytes(right)
    assert _rate_bytes(left) != _rate_bytes(defer(first)(Param("x")))
