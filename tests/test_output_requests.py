"""OutputSet requests: summer2-style one-call outputs, intermediates, fan-out, run(outputs=).

Each request is tested against the expression its docstring says it stands for, and the
values against NumPy on the raw save.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from summer4 import (
    Compartments,
    CompiledModel,
    Dest,
    FlowMass,
    FlowModel,
    OutputSet,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Source,
    Trait,
    TraitChain,
    TransitionFlow,
    defer,
)

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("young", "old"))
REACH = Property("reach", ("near", "far"))
PMAP = PropertyMap.from_property(STATE).stratify(AGE).stratify(REACH)
PARAMS = {"beta": 0.3, "gamma": 0.1, "ageing": 0.05}
RUN: dict[str, Any] = {"t0": 0.0, "t1": 10.0, "dt": 1.0, "solver": "euler"}


def _model() -> tuple[CompiledModel, PropertyData]:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], Param("beta")))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], Param("gamma")))
    model.add_flow(
        TransitionFlow(
            "ageing",
            AGE.present(),
            AGE.present(),
            Param("ageing"),
            pairing=TraitChain(AGE, (("young", "old"),)),
        )
    )
    y0 = PropertyData.wrap(PMAP, np.arange(1.0, PMAP.size + 1.0))
    return model.compile(), y0


def _solve(outputs: OutputSet) -> Any:
    cm, y0 = _model()
    return cm.run(PARAMS, y0, outputs=outputs, **RUN)


def _same(a: Any, b: Any) -> None:
    np.testing.assert_allclose(np.asarray(a.array()), np.asarray(b.array()), rtol=1e-12)


def _by_hand(build: Any) -> Any:
    """Evaluate a hand-built OutputSet the long way: plan, run, evaluate."""
    cm, y0 = _model()
    outputs = OutputSet()
    build(outputs)
    result = cm.run(PARAMS, y0, save=outputs.plan(), **RUN)
    return outputs.evaluate(result, PARAMS)


# --- requests are the expressions they document ---------------------------------------------


def test_compartments_request_is_select_then_sum() -> None:
    outputs = OutputSet()
    expr = outputs.compartments("infectious", STATE["I"] & AGE["old"])
    assert expr == Compartments().select(STATE["I"] & AGE["old"]).sum_by()
    named = _solve(outputs)

    def build(o: OutputSet) -> None:
        o["infectious"] = Compartments().select(STATE["I"] & AGE["old"]).sum_by()

    _same(named["infectious"], _by_hand(build)["infectious"])


def test_compartments_request_matches_numpy_on_the_raw_save() -> None:
    cm, y0 = _model()
    outputs = OutputSet()
    outputs.compartments("total")
    outputs.compartments("infectious", STATE["I"])
    named = cm.run(PARAMS, y0, outputs=outputs, **RUN)
    full = cm.run(PARAMS, y0, save=SavePlan(requests={"c": SaveRequest(Compartments())}), **RUN)
    comps = np.asarray(full["c"].array())
    mask = PMAP.kleene(STATE["I"]) == 1
    np.testing.assert_allclose(np.asarray(named["total"].array()), comps.sum(axis=-1))
    np.testing.assert_allclose(np.asarray(named["infectious"].array()), comps[:, mask].sum(axis=-1))


def test_flow_request_source_dest_and_midpoint() -> None:
    outputs = OutputSet()
    outputs.flow("inc_old", "infection", source=AGE["old"])
    outputs.flow("aged_far", "ageing", source=REACH["far"], dest=REACH["far"])
    outputs.flow("inc_mid", "infection", values="midpoint")
    named = _solve(outputs)

    def build(o: OutputSet) -> None:
        o["inc_old"] = FlowMass("infection").select(Source(AGE["old"])).sum_by()
        o["aged_far"] = (
            FlowMass("ageing").select(Source(REACH["far"]) & Dest(REACH["far"])).sum_by()
        )
        o["inc_mid"] = FlowMass("infection").sum_by().midpoint()

    hand = _by_hand(build)
    for name in ("inc_old", "aged_far", "inc_mid"):
        _same(named[name], hand[name])


def test_flow_values_default_comes_from_the_set() -> None:
    outputs = OutputSet(flow_values="midpoint")
    outputs.flow("mid", "infection")
    outputs.flow("rate", "infection", values="rate")
    named = _solve(outputs)

    def build(o: OutputSet) -> None:
        o["mid"] = FlowMass("infection").sum_by().midpoint()
        o["rate"] = FlowMass("infection").sum_by()

    hand = _by_hand(build)
    _same(named["mid"], hand["mid"])
    _same(named["rate"], hand["rate"])
    assert not np.allclose(np.asarray(named["mid"].array()), np.asarray(named["rate"].array()))


def test_requests_on_one_flow_share_one_save() -> None:
    outputs = OutputSet()
    outputs.flow("a", "infection", source=AGE["old"])
    outputs.flow("b", "infection", source=AGE["young"])
    outputs.compartments("c", STATE["S"])
    outputs.compartments("d", STATE["I"])
    assert len(outputs.plan().requests) == 2


def test_aggregate_cumulative_and_function() -> None:
    outputs = OutputSet()
    outputs.flow("inc_young", "infection", source=AGE["young"])
    outputs.flow("inc_old", "infection", source=AGE["old"])
    outputs.aggregate("inc", ["inc_young", "inc_old"])
    outputs.flow("all_inc", "infection")
    outputs.cumulative("cum", "inc", start=2.0)
    outputs.cumulative("cum_sum", "inc", method="sum")
    outputs.compartments("pop")

    def per_capita(x: Any, pop: Any, scale: Any) -> Any:
        return scale * x / pop

    outputs.function("inc_pc", per_capita, "inc", "pop", Param("gamma"))
    named = _solve(outputs)

    _same(named["inc"], named["all_inc"])

    def build(o: OutputSet) -> None:
        o["inc"] = FlowMass("infection").sum_by()
        o["pop"] = Compartments().sum_by()
        o["cum"] = o.ref("inc").cumulative_integral(start=2.0)
        o["cum_sum"] = o.ref("inc").cumsum()
        o["inc_pc"] = defer(per_capita)(o.ref("inc"), o.ref("pop"), Param("gamma"))

    hand = _by_hand(build)
    for name in ("cum", "cum_sum", "inc_pc"):
        _same(named[name], hand[name])


def test_aggregate_rejects_empty_and_bare_string() -> None:
    outputs = OutputSet()
    with pytest.raises(ValueError, match="at least one source"):
        outputs.aggregate("x", [])
    with pytest.raises(TypeError, match="sequence of output names"):
        outputs.aggregate("x", "abc")


def test_requests_return_expressions_that_chain() -> None:
    outputs = OutputSet()
    inc = outputs.flow("inc", "infection")
    outputs["weekly"] = inc.rolling(3)
    outputs["inc_x2"] = 2.0 * outputs.ref("inc")
    named = _solve(outputs)
    np.testing.assert_allclose(
        np.asarray(named["inc_x2"].array()), 2.0 * np.asarray(named["inc"].array())
    )
    assert "weekly" in named


def test_cumulative_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="'integral' or 'sum'"):
        OutputSet().cumulative("c", "x", method="mean")  # type: ignore[arg-type]


# --- intermediates -------------------------------------------------------------------------


def test_keep_false_is_computed_but_not_returned() -> None:
    outputs = OutputSet()
    outputs.compartments("pop", keep=False)
    outputs.compartments("inf", STATE["I"], keep=False)
    outputs["prev"] = outputs.ref("inf") / outputs.ref("pop")
    assert outputs.keys() == ("pop", "inf", "prev")
    assert outputs.kept() == ("prev",)
    named = _solve(outputs)
    assert tuple(named.keys()) == ("prev",)

    cm, y0 = _model()
    everything = outputs.evaluate(
        cm.run(PARAMS, y0, save=outputs.plan(), **RUN), PARAMS, names=outputs.keys()
    )
    assert tuple(everything.keys()) == ("pop", "inf", "prev")
    np.testing.assert_allclose(
        np.asarray(everything["prev"].array()),
        np.asarray(everything["inf"].array()) / np.asarray(everything["pop"].array()),
    )


def test_reassigning_resets_keep() -> None:
    outputs = OutputSet()
    outputs.compartments("pop", keep=False)
    outputs["pop"] = Compartments().sum_by()
    assert outputs.kept() == ("pop",)


def test_evaluate_names_computes_only_what_they_read() -> None:
    outputs = OutputSet()
    outputs.compartments("pop")

    def boom(x: Any) -> Any:
        raise AssertionError("an unrequested output was evaluated")

    outputs.function("never", boom, "pop")
    cm, y0 = _model()
    result = cm.run(PARAMS, y0, save=outputs.plan(), **RUN)
    named = outputs.evaluate(result, PARAMS, names=["pop"])
    assert tuple(named.keys()) == ("pop",)
    with pytest.raises(KeyError, match="no outputs"):
        outputs.evaluate(result, PARAMS, names=["nope"])


def test_subset_keeps_only_names_and_shares_the_plan() -> None:
    outputs = OutputSet(flow_values="midpoint")
    outputs.compartments("pop")
    outputs.compartments("inf", STATE["I"])
    outputs["prev"] = outputs.ref("inf") / outputs.ref("pop")
    small = outputs.subset(["prev"])
    assert small.kept() == ("prev",)
    assert small.keys() == outputs.keys()
    assert small.flow_values == "midpoint"
    assert small.plan() == outputs.plan()
    assert outputs.kept() == ("pop", "inf", "prev")  # the original is unchanged
    with pytest.raises(KeyError):
        outputs.subset(["missing"])


# --- fan-out ------------------------------------------------------------------------------


def test_fan_out_names_and_values() -> None:
    outputs = OutputSet()
    outputs.compartments("prev", STATE["I"], by=[AGE, REACH, (AGE, REACH)])
    expected = (
        ["prev"]
        + [f"prevXage_{a}" for a in AGE.traits]
        + [f"prevXreach_{r}" for r in REACH.traits]
        + [f"prevXage_{a}Xreach_{r}" for a in AGE.traits for r in REACH.traits]
    )
    assert outputs.kept() == tuple(expected)
    named = _solve(outputs)

    def build(o: OutputSet) -> None:
        for a in AGE.traits:
            o[f"a_{a}"] = Compartments().select(STATE["I"] & AGE[a]).sum_by()
            for r in REACH.traits:
                o[f"ar_{a}_{r}"] = Compartments().select(STATE["I"] & AGE[a] & REACH[r]).sum_by()
        for r in REACH.traits:
            o[f"r_{r}"] = Compartments().select(STATE["I"] & REACH[r]).sum_by()

    hand = _by_hand(build)
    for a in AGE.traits:
        _same(named[f"prevXage_{a}"], hand[f"a_{a}"])
        for r in REACH.traits:
            _same(named[f"prevXage_{a}Xreach_{r}"], hand[f"ar_{a}_{r}"])
    for r in REACH.traits:
        _same(named[f"prevXreach_{r}"], hand[f"r_{r}"])
    # The strata of one grouping add up to the total.
    total = sum(np.asarray(named[f"prevXage_{a}"].array()) for a in AGE.traits)
    np.testing.assert_allclose(total, np.asarray(named["prev"].array()))


def test_flow_fan_out_by_source_and_dest_with_midpoint() -> None:
    outputs = OutputSet(flow_values="midpoint")
    outputs.flow("aged", "ageing", by=AGE)
    outputs.flow("aged_to", "ageing", by=AGE, side="dest")
    named = _solve(outputs)

    def build(o: OutputSet) -> None:
        for a in AGE.traits:
            o[f"src_{a}"] = FlowMass("ageing").select(Source(AGE[a])).sum_by().midpoint()
            o[f"dst_{a}"] = FlowMass("ageing").select(Dest(AGE[a])).sum_by().midpoint()

    hand = _by_hand(build)
    for a in AGE.traits:
        _same(named[f"agedXage_{a}"], hand[f"src_{a}"])
        _same(named[f"aged_toXage_{a}"], hand[f"dst_{a}"])
    # Ageing leaves only the young and arrives only in the old.
    assert np.allclose(np.asarray(named["agedXage_old"].array()), 0.0)
    assert np.allclose(np.asarray(named["aged_toXage_young"].array()), 0.0)


def test_fan_out_naming_template_and_callable() -> None:
    outputs = OutputSet(naming="_{trait}")
    outputs.compartments("pop", by=AGE)
    assert outputs.kept() == ("pop", "pop_young", "pop_old")

    def naming(name: str, traits: tuple[Trait, ...]) -> str:
        return name + "[" + ",".join(f"{t.property}={t.name}" for t in traits) + "]"

    outputs = OutputSet(naming=naming)
    outputs.compartments("pop", by=[(AGE, REACH)])
    assert "pop[age=young,reach=far]" in outputs.kept()


def test_fan_out_on_a_raw_expression_and_keep_false() -> None:
    outputs = OutputSet()
    added = outputs.fan_out("s", Compartments().select(STATE["S"]), AGE, keep=False)
    assert added == ("sXage_young", "sXage_old")
    assert outputs.kept() == ()
    outputs["ratio"] = outputs.ref("sXage_old") / outputs.ref("sXage_young")
    assert outputs.kept() == ("ratio",)
    assert "ratio" in _solve(outputs)


def test_fan_out_rejects_non_properties() -> None:
    with pytest.raises(TypeError, match="by= takes a Property"):
        OutputSet().compartments("x", by=["age"])  # type: ignore[list-item]


def test_fan_out_jaxpr_grows_per_stratum_like_a_select_per_stratum() -> None:
    """One grouped sum per grouping, then a row per stratum: same slope as a loop of selects."""

    def size(n_ages: int, use_by: bool) -> int:
        ages = Property("age", tuple(f"a{i}" for i in range(n_ages)))
        pmap = PropertyMap.from_property(STATE).stratify(ages).stratify(REACH)
        model = FlowModel(pmap)
        model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], Param("gamma")))
        outputs = OutputSet()
        if use_by:
            outputs.compartments("p", by=[(ages, REACH)])
        else:
            for a in ages.traits:
                for r in REACH.traits:
                    outputs[f"pXage_{a}Xreach_{r}"] = (
                        Compartments().select(ages[a] & REACH[r]).sum_by()
                    )
        y = PropertyData.wrap(pmap, np.ones(pmap.size))
        result = model.compile().run(PARAMS, y, save=outputs.plan(), **RUN)
        names = [n for n in outputs.kept() if "X" in n]
        jaxpr = jax.make_jaxpr(lambda r: outputs.evaluate(r, PARAMS, names=names))(result)
        return len(jaxpr.eqns)

    by_slope = size(8, True) - size(2, True)
    loop_slope = size(8, False) - size(2, False)
    assert by_slope <= loop_slope
    assert size(8, True) - size(8, False) <= 20  # the one grouped sum


# --- run(outputs=) ---------------------------------------------------------------------------


def test_run_with_outputs_is_plan_run_evaluate() -> None:
    cm, y0 = _model()
    outputs = OutputSet()
    outputs.compartments("pop")
    outputs.flow("inc", "infection")
    base = SavePlan(ts=np.array([0.0, 5.0, 10.0]))
    one = cm.run(PARAMS, y0, outputs=outputs, save=base, **RUN)
    two = outputs.evaluate(cm.run(PARAMS, y0, save=outputs.plan(base), **RUN), PARAMS)
    assert tuple(one.keys()) == tuple(two.keys())
    for name in one:
        _same(one[name], two[name])
    np.testing.assert_allclose(np.asarray(one["pop"].times.values), [0.0, 5.0, 10.0])


def test_run_with_outputs_rejects_non_outputset() -> None:
    cm, y0 = _model()
    with pytest.raises(TypeError, match="OutputSet"):
        cm.run(PARAMS, y0, outputs={"x": 1}, **RUN)
