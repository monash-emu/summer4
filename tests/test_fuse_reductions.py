"""Fusing weighted sums of scalar ``Reduce`` terms into one weighted reduction.

``FlowModel.compile(fuse_reductions=True)`` (the default) rewrites
``sum_k c_k * Reduce(where=sel_k)`` into one ``WeightedReduce``. These tests check
that the fused and term-by-term forms agree (vector field, solves and
gradients, rtol 1e-12) on overlapping, disjoint and empty selectors, that the
rewrite leaves what it must leave alone, and that the vector field shrinks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from summer4 import (
    ExitFlow,
    FlowModel,
    Multiply,
    Nothing,
    Param,
    Property,
    PropertyMap,
    Reduce,
    SavePlan,
    Time,
    TransitionFlow,
)
from summer4.flows.fuse_reduce import ReduceWeights, WeightedReduce, fuse_reductions
from summer4.flows.rates import ArrayConst, BinOp, rate_children

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

RTOL = 1e-12

STAGE = Property("stage", ("S", "E", "L", "active", "T", "R"))
FOCUS = Property("focus", ("f0", "f1", "f2"))
EXTENT = Property("extent", ("e0", "e1"))
ACTIVE = STAGE["active"]
PARAMS = {"a": 0.7, "b": 1.3, "c": 2.5, "k": 0.4, "beta": 3.0}


def _pmap() -> PropertyMap:
    """6 stages; ``active`` split 3 x 2, so 11 compartments with absent properties elsewhere."""
    return (
        PropertyMap.from_property(STAGE)
        .stratify(FOCUS, where=ACTIVE)
        .stratify(EXTENT, where=ACTIVE)
    )


def _cell(f: str, e: str) -> Any:
    return Reduce(where=ACTIVE & FOCUS[f] & EXTENT[e])


def _grid_terms() -> Any:
    """Disjoint cells, constant weights ``(i + 1) * (j + 1) / 6``."""
    return sum(
        ((i + 1) * (j + 1) / 6.0) * _cell(f, e)
        for i, f in enumerate(FOCUS.traits)
        for j, e in enumerate(EXTENT.traits)
    )


# Each case is a rate for the S -> E flow; all are linear in scalar Reduce terms.
CASES: dict[str, Any] = {
    "disjoint_const": lambda: Param("beta") * _grid_terms() / Reduce(),
    "overlapping": lambda: (
        (
            0.5 * Reduce(where=ACTIVE)
            + 2.0 * Reduce(where=FOCUS["f0"])
            + Reduce(where=EXTENT["e1"]) * 0.25
            + Reduce(where=ACTIVE & FOCUS["f0"] & EXTENT["e1"])
        )
        / Reduce()
    ),
    "empty_selectors": lambda: (
        (
            0.5 * Reduce(where=STAGE["S"] & FOCUS["f0"])  # absent property: matches nothing
            + 3.0 * Reduce(where=Nothing())
            + Reduce(where=ACTIVE)
        )
        / Reduce()
    ),
    "all_empty": lambda: (
        1.0 + 2.0 * Reduce(where=Nothing()) + Reduce(where=STAGE["E"] & EXTENT["e0"])
    ),
    "param_coefficients": lambda: (
        (
            Param("a") * Reduce(where=FOCUS["f0"])
            - Param("b") * Reduce(where=FOCUS["f1"]) / 2.0
            + Reduce(where=FOCUS["f2"]) / Param("c")
            + (Param("a") * Param("b")) * Reduce(where=EXTENT["e0"])
        )
        / Reduce()
    ),
    "nested_sub_neg": lambda: (
        Param("k")
        * (
            2.0 * (Reduce(where=FOCUS["f0"]) + Reduce(where=FOCUS["f1"]))
            - (Reduce(where=EXTENT["e0"]) * 3.0 - Reduce(where=ACTIVE))
            + -Reduce(where=FOCUS["f2"] & EXTENT["e1"])
        )
        / Reduce()
    ),
    "with_rest": lambda: (
        (
            Reduce(where=FOCUS["f0"])
            + Reduce(where=FOCUS["f1"])
            + 0.01 * Time() * Reduce(where=ACTIVE)  # step-stage coefficient: kept as an addend
            + Param("a")
        )
        / Reduce()
    ),
    "whole_population_terms": lambda: (Reduce() - Reduce(where=ACTIVE)) / Reduce(),
}


def _model(rate: Any, *, adjust_rate: Any = None) -> FlowModel:
    model = FlowModel(_pmap())
    flow_kw: dict[str, Any] = {}
    if adjust_rate is not None:
        flow_kw["adjust"] = (Multiply(adjust_rate),)
    model.add_flow(TransitionFlow("infection", STAGE["S"], STAGE["E"], rate, **flow_kw))
    model.add_flow(TransitionFlow("stabilise", STAGE["E"], STAGE["L"], 0.5))
    model.add_flow(TransitionFlow("activate", STAGE["L"], ACTIVE, 0.2))
    model.add_flow(TransitionFlow("detect", ACTIVE, STAGE["T"], 0.6))
    model.add_flow(TransitionFlow("treat", STAGE["T"], STAGE["R"], 2.0))
    model.add_flow(ExitFlow("death", STAGE["R"], 0.01))
    return model


def _y0(pmap: PropertyMap) -> np.ndarray:
    rng = np.random.default_rng(7)
    y0 = rng.uniform(1.0, 50.0, pmap.size)
    y0[pmap.select(STAGE["S"])] = 1000.0
    return y0


def _pair(model: FlowModel) -> tuple[Any, Any]:
    return model.compile(), model.compile(fuse_reductions=False)


def _has_node(expr: Any, cls: type) -> bool:
    if isinstance(expr, cls):
        return True
    return any(_has_node(child, cls) for child in rate_children(expr))


def _vf(cm: Any, y: Any, params: dict[str, float], t: float = 1.5) -> np.ndarray:
    return np.asarray(cm.vector_field(t, jnp.asarray(y), cm.prepare(params)))


def _final_state(cm: Any, y0: Any, theta: Any, solver: str) -> Any:
    p = dict(PARAMS)
    p.update({"a": theta[0], "beta": theta[1]})
    kw: dict[str, Any] = {"t0": 0.0, "t1": 20.0, "dt": 0.05, "save": SavePlan(ts=[20.0])}
    if solver == "euler":
        kw["solver"] = "euler"
    return jnp.asarray(cm.run(p, jnp.asarray(y0), **kw).final_state.data)


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_fused_rate_is_rewritten(case: str) -> None:
    fused, plain = _pair(_model(CASES[case]()))
    assert _has_node(fused.flows["infection"].rate, WeightedReduce)
    assert not _has_node(plain.flows["infection"].rate, WeightedReduce)


@pytest.mark.parametrize("case", sorted(CASES))
def test_vector_field_matches(case: str) -> None:
    fused, plain = _pair(_model(CASES[case]()))
    y0 = _y0(fused.pmap)
    np.testing.assert_allclose(_vf(fused, y0, PARAMS), _vf(plain, y0, PARAMS), rtol=RTOL)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_solve_and_grad_match(case: str, solver: str) -> None:
    fused, plain = _pair(_model(CASES[case]()))
    y0 = _y0(fused.pmap)
    theta = jnp.asarray([PARAMS["a"], PARAMS["beta"]])

    def loss(cm: Any) -> Any:
        return lambda th: jnp.sum(_final_state(cm, y0, th, solver) ** 2)

    np.testing.assert_allclose(
        _final_state(fused, y0, theta, solver), _final_state(plain, y0, theta, solver), rtol=RTOL
    )
    np.testing.assert_allclose(
        jax.grad(loss(fused))(theta), jax.grad(loss(plain))(theta), rtol=RTOL
    )


def test_adjustment_value_is_fused_and_matches() -> None:
    adjust = 0.5 + Param("a") * Reduce(where=FOCUS["f0"]) - Reduce(where=EXTENT["e1"])
    model = _model(Param("beta") * Reduce(where=ACTIVE) / Reduce(), adjust_rate=adjust / 1000.0)
    fused, plain = _pair(model)
    assert _has_node(fused.flows["infection"].adjust[0].value, WeightedReduce)
    y0 = _y0(fused.pmap)
    np.testing.assert_allclose(_vf(fused, y0, PARAMS), _vf(plain, y0, PARAMS), rtol=RTOL)


def test_shared_subtree_fuses_once() -> None:
    pool = 0.3 * Reduce(where=FOCUS["f0"]) + 0.7 * Reduce(where=FOCUS["f1"])
    model = FlowModel(_pmap())
    model.add_flow(TransitionFlow("infection", STAGE["S"], STAGE["E"], pool / Reduce()))
    model.add_flow(ExitFlow("loss", STAGE["S"], pool / Reduce()))
    cm = model.compile()
    left = cm.flows["infection"].rate
    right = cm.flows["loss"].rate
    assert isinstance(left, BinOp) and isinstance(right, BinOp)
    assert left.left is right.left  # one shared WeightedReduce, evaluated once per call


# ---------------------------------------------------------------------------
# What is left alone
# ---------------------------------------------------------------------------


def _fuse(expr: Any) -> Any:
    return fuse_reductions(expr, _pmap())


def test_single_term_is_not_fused() -> None:
    expr = Param("beta") * Reduce(where=ACTIVE) / Reduce()
    assert _fuse(expr) is expr


def test_grouped_reduce_is_not_fused() -> None:
    expr = Reduce(FOCUS, where=ACTIVE) + Reduce(FOCUS)
    assert _fuse(expr) is expr


def test_product_of_reduces_is_not_fused() -> None:
    expr = Reduce(where=FOCUS["f0"]) * Reduce(where=FOCUS["f1"])
    assert _fuse(expr) is expr


def test_state_dependent_coefficient_is_not_folded() -> None:
    expr = Time() * Reduce(where=FOCUS["f0"]) + Time() * Reduce(where=FOCUS["f1"])
    assert _fuse(expr) is expr


def test_fusion_inside_a_larger_tree() -> None:
    expr = Param("beta") * (Reduce(where=FOCUS["f0"]) + Reduce(where=FOCUS["f1"])) / Reduce()
    out = _fuse(expr)
    assert out is not expr and _has_node(out, WeightedReduce)
    # The denominator Reduce() is a single term and stays a plain Reduce.
    assert isinstance(out, BinOp) and out.op == "div" and isinstance(out.right, Reduce)


def test_constant_weights_fold_to_an_array() -> None:
    out = _fuse(_grid_terms())
    assert isinstance(out, WeightedReduce) and isinstance(out.weights, ArrayConst)
    np.testing.assert_allclose(
        out.weights.value, [(i + 1) * (j + 1) / 6.0 for i in range(3) for j in range(2)]
    )


def test_parameter_weights_are_hoisted() -> None:
    cm = _model(CASES["param_coefficients"]()).compile()
    assert cm.hoist_table is not None
    assert any(isinstance(entry.node, ReduceWeights) for entry in cm.hoist_table.entries)


def test_unselected_non_finite_compartment_does_not_leak() -> None:
    # No whole-population denominator: only the active cells are read.
    fused, plain = _pair(_model(Param("beta") * _grid_terms() / 1000.0))
    y0 = _y0(fused.pmap)
    y0[fused.pmap.select(STAGE["T"])] = np.nan
    flows_f = fused.observe(0.0, y0, PARAMS).flows["infection"]
    flows_p = plain.observe(0.0, y0, PARAMS).flows["infection"]
    assert np.all(np.isfinite(np.asarray(flows_f)))
    np.testing.assert_allclose(np.asarray(flows_f), np.asarray(flows_p), rtol=RTOL)


def test_unknown_property_still_raises_on_evaluation() -> None:
    other = Property("strain", ("ds", "mdr"))
    rate = Reduce(where=other["ds"]) + Reduce(where=FOCUS["f0"])
    model = _model(rate)
    with_fusion = model.compile()  # compiling does not resolve the bad selector
    without = model.compile(fuse_reductions=False)
    y0 = _y0(with_fusion.pmap)
    with pytest.raises(Exception) as fused_err:
        with_fusion.observe(0.0, y0, PARAMS)
    with pytest.raises(Exception) as plain_err:
        without.observe(0.0, y0, PARAMS)
    assert type(fused_err.value) is type(plain_err.value)


# ---------------------------------------------------------------------------
# Equation-count regression
# ---------------------------------------------------------------------------


def _bench_module() -> Any:
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "weighted_reduce.py"
    spec = importlib.util.spec_from_file_location("weighted_reduce_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vector_field_equation_count() -> None:
    """36 weighted cells cost about what one plain ``Reduce`` does once fused."""
    wr = _bench_module()

    def eqns(form: str, **compile_kw: Any) -> tuple[int, int]:
        model, pmap = wr.build(form)
        cm = model.compile(**compile_kw)
        return wr.vf_eqns(cm, jnp.asarray(wr.initial_state(pmap)))

    plain = eqns("plain")
    fused = eqns("terms")
    unfused = eqns("terms", fuse_reductions=False)
    assert unfused[1] > 3 * plain[1]  # the term-by-term form grows with the 36 terms
    assert fused[1] <= plain[1] + 8, (fused, plain)
    assert fused[0] <= plain[0] + 8, (fused, plain)
