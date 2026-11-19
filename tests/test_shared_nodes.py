"""Step 41 (with the part of step 39 it needs): shared nodes, ``field_refs``, no ``derived_fn``."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    CompiledModel,
    FieldRef,
    FlowModel,
    Item,
    Param,
    Params,
    Property,
    PropertyMap,
    StateArray,
    Time,
    TransitionFlow,
    defer,
    field_refs,
)
from summer4.flows.stages import rate_stage

STATE = Property("state", ("S", "I", "R"))
PMAP = PropertyMap.from_property(STATE)
Y0 = np.array([900.0, 100.0, 0.0])
PARAMS = {"b": 0.3, "g": 0.1, "a": 2.0, "c": 0.5}


class Derived(NamedTuple):
    infection: Any
    recovery: Any


class Nested(NamedTuple):
    inner: Derived
    scale: Any


def derive(params: Any, *, y: Any, t: Any) -> Derived:
    season = 1.0 + 0.1 * jnp.sin(t)
    prevalence = y[1] / jnp.sum(y)
    return Derived(infection=params["b"] * season * prevalence, recovery=params["g"] * season)


def _model(d: Derived, extra: Any = 1.0) -> CompiledModel:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], d.infection * extra))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], d.recovery))
    return model.compile()


def _hand_dy(t: float, y: np.ndarray, extra: float = 1.0) -> np.ndarray:
    season = 1.0 + 0.1 * np.sin(t)
    inf = 0.3 * season * y[1] / y.sum() * extra * y[0]
    rec = 0.1 * season * y[1]
    return np.array([-inf, inf - rec, rec])


def _derived() -> Derived:
    return field_refs(defer(derive)(Params(), y=StateArray(), t=Time()), Derived)


@pytest.mark.parametrize("hoist", [True, False])
def test_field_refs_model_matches_hand_values(hoist: bool) -> None:
    d = _derived()
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], d.infection))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], d.recovery))
    cm = model.compile(hoist=hoist)
    for t in (0.0, 1.3):
        dy = np.asarray(cm.vector_field(t, jnp.asarray(Y0), PARAMS))
        np.testing.assert_allclose(dy, _hand_dy(t, Y0), rtol=1e-12)
    result = cm.run(PARAMS, Y0, t0=0.0, t1=3.0, dt=1.0, solver="euler")
    y = Y0.copy()
    for t in (0.0, 1.0, 2.0):
        y = y + _hand_dy(t, y)
    np.testing.assert_allclose(np.asarray(result.final_state.data), y, rtol=1e-10)


def test_shared_node_is_traced_once() -> None:
    """One ``defer(sin)(Time())`` object read by three flows traces one ``sin``.

    Before the per-call memo each flow evaluated it again: three ``sin`` equations.
    """
    shared = defer(jnp.sin)(Time())
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("a", STATE["S"], STATE["I"], 0.1 + 0.01 * shared))
    model.add_flow(TransitionFlow("b", STATE["I"], STATE["R"], 0.2 + 0.01 * shared))
    model.add_flow(TransitionFlow("c", STATE["R"], STATE["S"], 0.3 + 0.01 * shared))
    cm = model.compile()
    jaxpr = jax.make_jaxpr(lambda t, y: cm.vector_field(t, y, PARAMS))(0.5, jnp.asarray(Y0))
    assert sum(eqn.primitive.name == "sin" for eqn in jaxpr.jaxpr.eqns) == 1


def test_derive_runs_once_per_vector_field_call() -> None:
    calls: list[int] = []

    def counted(params: Any, *, y: Any, t: Any) -> Derived:
        calls.append(1)
        return derive(params, y=y, t=t)

    d = field_refs(defer(counted)(Params(), y=StateArray(), t=Time()), Derived)
    cm = _model(d)
    cm.vector_field(0.0, jnp.asarray(Y0), PARAMS)
    assert len(calls) == 1


def test_parameter_products_beside_field_refs_are_hoisted() -> None:
    d = _derived()
    product = Param("a") * Param("c")
    cm = _model(d, extra=product)
    slotted = [entry.node for entry in cm.hoist_table.entries]
    assert any(node is product for node in slotted)
    dy = np.asarray(cm.vector_field(0.0, jnp.asarray(Y0), PARAMS))
    np.testing.assert_allclose(dy, _hand_dy(0.0, Y0, extra=1.0), rtol=1e-12)


def test_parameter_only_defer_is_evaluated_once_per_run() -> None:
    calls: list[int] = []

    def from_params(params: Any) -> Derived:
        calls.append(1)
        return Derived(infection=params["b"] * 0.01, recovery=params["g"])

    d = field_refs(defer(from_params)(Params()), Derived)
    assert rate_stage(d.infection) == "run"
    cm = _model(d)
    prepared = cm.prepare(PARAMS)
    n = len(calls)
    assert n >= 1
    cm.vector_field(0.0, jnp.asarray(Y0), prepared)
    cm.vector_field(1.0, jnp.asarray(Y0), prepared)
    assert len(calls) == n  # hoisted: not called again per step


def test_field_refs_of_params_are_field_refs() -> None:
    refs = field_refs(Params(), Nested)
    assert refs.scale == Param("scale")
    assert refs.inner.infection == FieldRef(("inner", "infection"))


def test_field_refs_of_an_expression_are_items() -> None:
    root = defer(derive)(Params(), y=StateArray(), t=Time())
    refs = field_refs(root, Nested)
    assert refs.scale == Item(root, "scale")
    assert refs.inner.recovery == Item(Item(root, "inner"), "recovery")
    assert rate_stage(refs.scale) == "step"


def test_stages_of_the_new_leaves() -> None:
    assert rate_stage(Params()) == "run"
    assert rate_stage(StateArray()) == "step"
    assert rate_stage(Param("x")) == "run"
    assert rate_stage(Item(Params(), "x")) == "run"


def test_item_reads_mappings_and_indices() -> None:
    def pair(params: Any) -> Any:
        return {"first": params["a"], "both": jnp.stack([params["a"], params["c"]])}

    node = defer(pair)(Params())
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("x", STATE["S"], STATE["I"], Item(Item(node, "both"), 1)))
    model.add_flow(TransitionFlow("y", STATE["I"], STATE["R"], Item(node, "first")))
    dy = np.asarray(model.compile().vector_field(0.0, jnp.asarray(Y0), PARAMS))
    np.testing.assert_allclose(dy, [-0.5 * 900, 0.5 * 900 - 2.0 * 100, 2.0 * 100])


def test_field_refs_rejects_non_namedtuples() -> None:
    with pytest.raises(TypeError, match="NamedTuple"):
        field_refs(Params(), dict)  # type: ignore[type-var]
    with pytest.raises(TypeError, match="rate expression"):
        field_refs("params", Derived)  # type: ignore[arg-type]


def test_compile_no_longer_takes_derived_fn() -> None:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("x", STATE["S"], STATE["I"], Param("b")))
    with pytest.raises(TypeError):
        model.compile(derived_fn=derive)  # type: ignore[call-arg]
