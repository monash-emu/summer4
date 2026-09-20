"""Vector-valued tables and parameter-table lookup."""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    EntryFlow,
    ExitFlow,
    FlowModel,
    FlowRef,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Time,
    TransitionFlow,
    eval_closed,
    floor,
    log,
    maximum,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix
from summer4.flows.compiled import _eval_interp
from summer4.flows.rates import Lookup, _field_paths, _flow_refs, _rate_bytes

Kind = Literal["linear", "sigmoidal", "step"]


def _table_rates(
    times: np.ndarray,
    values: np.ndarray,
    t: float,
    *,
    kind: Kind = "linear",
    sharpness: float = 1.0,
    expr: Any | None = None,
) -> np.ndarray:
    """Exit one person per age; ``-dy`` is the aligned table rate."""
    age = Property("age", tuple(str(i) for i in range(values.shape[1])))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state).stratify(age)
    rate = Data.table(times, values, over=age).interp(kind, sharpness=sharpness)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("die", state["Y"], rate if expr is None else expr(rate)))
    cm = model.compile()
    y = PropertyData.wrap(pmap, np.ones(pmap.size))
    dy = np.asarray(cm.vector_field(t, y, {}).data)
    return -dy


def _scalar_rate(
    times: np.ndarray,
    column: np.ndarray,
    t: float,
    *,
    kind: Kind = "linear",
    sharpness: float = 1.0,
) -> float:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    expr = Data(times=times, values=column).interp(kind, sharpness=sharpness)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y = PropertyData.wrap(pmap, np.array([1.0]))
    return float(np.asarray(cm.vector_field(t, y, {}).data)[0])


def test_table_interp_matches_scalar_interps() -> None:
    times = np.array([0.0, 2.0, 6.0, 10.0])
    values = np.array(
        [
            [1.0, 4.0, 0.5],
            [3.0, 4.0, 1.5],
            [3.0, 8.0, 1.5],
            [9.0, 0.0, 2.5],
        ]
    )
    probes = (-1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 12.0)
    for kind in ("linear", "sigmoidal", "step"):
        sharpness = 4.0 if kind == "sigmoidal" else 1.0
        for t in probes:
            got = _table_rates(times, values, t, kind=kind, sharpness=sharpness)
            expect = [
                _scalar_rate(times, values[:, k], t, kind=kind, sharpness=sharpness)
                for k in range(values.shape[1])
            ]
            np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)


def test_table_interp_clamps_outside_range() -> None:
    times = np.array([0.0, 2.0])
    values = np.array([[1.0, 10.0], [5.0, 20.0]])
    np.testing.assert_allclose(_table_rates(times, values, -5.0), [1.0, 10.0])
    np.testing.assert_allclose(_table_rates(times, values, 10.0), [5.0, 20.0])


def test_grouped_rate_aligns_onto_age_flow() -> None:
    """Deaths leave only ``S``; each age band gets its own column."""
    age = Property("age", ("0", "15"))
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state).stratify(age)
    times = np.array([0.0, 2.0])
    values = np.array([[1.0, 3.0], [5.0, 7.0]])
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("die", state["S"], Data.table(times, values, over=age).interp()))
    cm = model.compile()
    y = PropertyData.wrap(pmap, np.ones(pmap.size))
    dy = np.asarray(cm.vector_field(1.0, y, {}).data)
    # Order is S-0, S-15, I-0, I-15. Midpoint rates are 3 and 5.
    np.testing.assert_allclose(dy, [-3.0, -5.0, 0.0, 0.0], rtol=1e-5, atol=1e-6)


def test_log_and_maximum_apply_to_the_table() -> None:
    times = np.array([0.0, 2.0])
    values = np.array([[1.0, 0.01], [np.e, 0.01]])
    logged = _table_rates(times, values, 0.0, expr=log)
    np.testing.assert_allclose(logged, [0.0, np.log(0.01)], rtol=1e-5, atol=1e-6)
    floored = _table_rates(times, values, 0.0, expr=lambda rate: maximum(rate, 0.5))
    np.testing.assert_allclose(floored, [1.0, 0.5], rtol=1e-5, atol=1e-6)


def test_table_columns_must_match_traits() -> None:
    age = Property("age", ("0", "15"))
    with pytest.raises(ValueError, match="traits"):
        Data.table(np.array([0.0, 1.0]), np.ones((2, 3)), over=age)
    with pytest.raises(ValueError, match="do not match"):
        Data.table(
            np.array([0.0, 1.0]),
            np.ones((2, 2)),
            over=age,
            columns=("15", "0"),
        )


def test_table_from_frame_selects_trait_order() -> None:
    pd = pytest.importorskip("pandas")
    age = Property("age", ("0", "15"))
    frame = pd.DataFrame({"15": [1.0, 2.0], "0": [3.0, 4.0], "extra": [9.0, 9.0]})
    with pytest.raises(ValueError, match="do not match"):
        Data.table(np.array([0.0, 1.0]), frame, over=age)
    table = Data.table(np.array([0.0, 1.0]), frame, over=age, columns=("0", "15"))
    np.testing.assert_allclose(table.values[:, 0], [3.0, 4.0])
    np.testing.assert_allclose(table.values[:, 1], [1.0, 2.0])


def test_table_digest_covers_values() -> None:
    age = Property("age", ("0", "15"))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state).stratify(age)
    times = np.array([0.0, 1.0])

    def digest(values: np.ndarray) -> bytes:
        model = FlowModel(pmap)
        model.add_flow(ExitFlow("die", state["Y"], Data.table(times, values, over=age).interp()))
        return model.compile()._digest

    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert digest(base) == digest(base.copy())
    assert digest(base) != digest(base + 1.0)
    assert _rate_bytes(Lookup(Param("m"), 0, clamp=True)) != _rate_bytes(
        Lookup(Param("m"), 0, clamp=False)
    )


def test_lookup_indexes_and_clamps() -> None:
    stack = np.arange(12.0).reshape(3, 2, 2)
    params = {"stack": stack}
    np.testing.assert_allclose(eval_closed(Lookup(Param("stack"), 1), params), stack[1])
    np.testing.assert_allclose(eval_closed(Lookup(Param("stack"), -4), params), stack[0])
    np.testing.assert_allclose(eval_closed(Lookup(Param("stack"), 9), params), stack[-1])
    unclamped = eval_closed(Lookup(Param("stack"), 9, clamp=False), params)
    assert not np.all(np.isfinite(np.asarray(unclamped)))
    with pytest.raises(ValueError, match="eval_closed"):
        eval_closed(Lookup(Param("stack"), floor(Time())), params)


def test_lookup_paths_refs_and_bytes() -> None:
    expr = Lookup(Param("stack"), FlowRef("other") + Param("shift"))
    assert _field_paths(expr) == {("stack",), ("shift",)}
    assert _flow_refs(expr) == {"other"}
    assert _rate_bytes(expr) != _rate_bytes(Lookup(Param("other"), 0))


class _YearMatrix(NamedTuple):
    K: Any


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _foi_model(matrix: Any, *, derived_fn: Any | None = None) -> Any:
    state, age, pmap = _sir_age()
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind=FOIKind.FREQUENCY,
        contact_rate=0.4,
        mixing=MixingMatrix(age, matrix, normalize="none", check_reciprocal=False),
    )
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    return model.compile(derived_fn=derived_fn)


def test_lookup_matches_derived_fn_foi_and_is_differentiable() -> None:
    stack = np.stack(
        [
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[0.1, 0.9], [0.4, 0.6]]),
        ]
    )
    y = np.array([900.0, 800.0, 80.0, 10.0, 0.0, 0.0])

    def derived(params: dict[str, Any], *, y: Any, t: Any) -> _YearMatrix:
        del y
        table = jnp.asarray(params["stack"])
        idx = jnp.clip(jnp.floor(jnp.asarray(t) - 1850.0), 0, table.shape[0] - 1)
        return _YearMatrix(K=jnp.take(table, idx.astype(jnp.int32), axis=0))

    lookup_cm = _foi_model(Lookup(Param("stack"), floor(Time() - 1850.0)))
    derived_cm = _foi_model(Param("K"), derived_fn=derived)
    assert ("stack",) in lookup_cm.computed_paths
    for t in (1840.0, 1850.0, 1851.7, 1852.0, 1900.0):
        got = np.asarray(lookup_cm.observe(t, y, {"stack": stack}).captures["infection"].data)
        expect = np.asarray(derived_cm.observe(t, y, {"stack": stack}).captures["infection"].data)
        np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)

    def loss(table: Any) -> Any:
        dy = lookup_cm.vector_field(1851.3, y, {"stack": table})
        return jnp.sum(jnp.asarray(dy) ** 2)

    grad = jax.grad(loss)(jnp.asarray(stack))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_table_jaxpr_independent_of_knots_and_columns() -> None:
    def kernel_eqns(kind: str, n_times: int, n_cols: int) -> int:
        times = jnp.linspace(0.0, 1.0, n_times)
        n_rows = n_times + 1 if kind == "step" else n_times
        values = jnp.ones((n_rows, n_cols))

        def run(x: Any) -> Any:
            return _eval_interp(kind, times, values, x, 4.0)

        return len(jax.make_jaxpr(run)(jnp.float32(0.3)).jaxpr.eqns)

    for kind in ("linear", "sigmoidal", "step"):
        small = kernel_eqns(kind, 4, 2)
        assert kernel_eqns(kind, 40, 2) == small
        assert kernel_eqns(kind, 4, 8) == small

    def model_eqns(n_times: int) -> int:
        times = np.linspace(0.0, 10.0, n_times)
        values = np.column_stack([np.linspace(0.1, 0.2, n_times), np.linspace(0.3, 0.4, n_times)])
        age = Property("age", ("0", "15"))
        state = Property("state", ("Y",))
        pmap = PropertyMap.from_property(state).stratify(age)
        model = FlowModel(pmap)
        model.add_flow(ExitFlow("die", state["Y"], Data.table(times, values, over=age).interp()))
        cm = model.compile()
        y = np.ones(pmap.size)

        def field(t: Any) -> Any:
            return jnp.asarray(cm.vector_field(t, y, {}))

        return len(jax.make_jaxpr(field)(0.0).jaxpr.eqns)

    short = model_eqns(4)
    long = model_eqns(32)
    assert short == long
