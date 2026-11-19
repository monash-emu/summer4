"""Step 52: summer4 gaps found porting the Kiribati TB model (CX51-CX56, CX58).

One section per gap; each checks the claim against the hand-written spelling the ports used.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from summer4 import (
    CompiledModel,
    Dest,
    FlowModel,
    GroupedOutput,
    Lookup,
    Multiply,
    OutputSet,
    Overwrite,
    Param,
    Params,
    Property,
    PropertyData,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
    Time,
    TraitChain,
    TransitionFlow,
    defer,
    field_refs,
    floor,
)
from summer4.data import Data
from summer4.epi.calibration import workflow as wf
from summer4.epi.calibration.posterior_runs import PosteriorRuns
from summer4.flows.rates import Capture, Const, Item

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("young", "old"))
PMAP = PropertyMap.from_property(STATE).stratify(AGE)
Y0 = np.array([600.0, 380.0, 15.0, 5.0, 0.0, 0.0])
EULER: dict[str, Any] = {"t0": 0.0, "t1": 6.0, "dt": 0.25, "solver": "euler"}


def _sir(rate: Any, *, recovery: Any = 0.1) -> CompiledModel:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], rate))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], recovery))
    return model.compile()


def _final(cm: CompiledModel, params: dict[str, Any]) -> np.ndarray:
    return np.asarray(cm.run(params, Y0, **EULER).final_state.data)


# --- gap 1 (CX51): Lookup reads a table computed from the parameters ------------------------


class _Stacks(NamedTuple):
    yearly: Any


def _build_stack(params: Any) -> dict[str, Any]:
    """A (years,) table computed once per run from two parameters."""
    years = jnp.arange(8.0)
    return {"yearly": params["base"] * jnp.exp(-params["decay"] * years)}


PARAMS_1 = {"base": 0.05, "decay": 0.3}


def test_lookup_reads_a_computed_table() -> None:
    stacks = field_refs(defer(_build_stack)(Params()), _Stacks)
    computed = _sir(Lookup(stacks.yearly, floor(Time())))
    host_stack = np.asarray(_build_stack(PARAMS_1)["yearly"])
    as_param = _sir(Lookup(Param("stack"), floor(Time())))
    np.testing.assert_allclose(
        _final(computed, PARAMS_1), _final(as_param, {"stack": host_stack}), rtol=1e-12
    )


def test_computed_lookup_table_is_hoisted() -> None:
    stacks = field_refs(defer(_build_stack)(Params()), _Stacks)
    cm = _sir(Lookup(stacks.yearly, floor(Time())))
    slotted = [entry.node for entry in cm.hoist_table.entries]
    assert any(isinstance(node, Item) and node.key == "yearly" for node in slotted)


def test_lookup_accepts_an_array_literal() -> None:
    table = np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01])
    literal = _sir(Lookup(table, floor(Time())))
    as_param = _sir(Lookup(Param("stack"), floor(Time())))
    np.testing.assert_allclose(_final(literal, {}), _final(as_param, {"stack": table}))


def test_lookup_rejects_a_scalar_table() -> None:
    with pytest.raises(TypeError, match="array-valued"):
        Lookup(2.0, floor(Time()))
    with pytest.raises(TypeError, match="array-valued"):
        Lookup(Const(2.0), floor(Time()))


# --- gap 2 (CX52): jnp on a GroupedRate argument gives a clear error ------------------------

_TABLE = Data.table(np.array([0.0, 10.0]), np.array([[0.02, 0.01], [0.04, 0.03]]), over=AGE)


def _np_survival(mu: Any, duration: Any) -> Any:
    return 1.0 - np.exp(-duration * mu)


def _jnp_survival(mu: Any, duration: Any) -> Any:
    return 1.0 - jnp.exp(-duration * mu)


def test_numpy_functions_keep_the_grouping() -> None:
    rate = defer(_np_survival)(_TABLE.interp(), Param("duration"))
    by_hand = 1.0 - np.exp(-Param("duration") * _TABLE.interp())
    params = {"duration": 2.0}
    np.testing.assert_allclose(_final(_sir(rate), params), _final(_sir(by_hand), params))


def test_jnp_function_on_a_grouped_argument_names_the_argument() -> None:
    rate = defer(_jnp_survival)(_TABLE.interp(), Param("duration"))
    with pytest.raises(TypeError, match=r"defer\(_jnp_survival\) received a GroupedRate") as err:
        _sir(rate).run({"duration": 2.0}, Y0, **EULER)
    assert "argument 0" in str(err.value)
    assert "np.exp" in str(err.value)


def test_unrelated_type_errors_are_not_rewritten() -> None:
    def broken(mu: Any) -> Any:
        raise TypeError("my own mistake")

    with pytest.raises(TypeError, match="^my own mistake$"):
        _sir(defer(broken)(_TABLE.interp())).run({}, Y0, **EULER)


# --- gap 4 (CX53): per-request naming --------------------------------------------------------

REACH = Property("reach", ("near", "far"))
PMAP_3 = PMAP.stratify(REACH)


def _solve_3(outputs: OutputSet) -> Any:
    model = FlowModel(PMAP_3)
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], Param("beta")))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], 0.1))
    y0 = PropertyData.wrap(PMAP_3, np.arange(1.0, PMAP_3.size + 1.0))
    return model.compile().run({"beta": 0.3}, y0, outputs=outputs, **EULER)


def test_per_request_whole_name_template() -> None:
    out = OutputSet()
    out.flow("recovered", "recovery", by=[(AGE, REACH)])
    out.flow("rec", "recovery", by=[(AGE, REACH)], naming="all_{name}_from_{age}_reach_{reach}")
    named = _solve_3(out)
    for age in AGE.traits:
        for reach in REACH.traits:
            got = named[f"all_rec_from_{age}_reach_{reach}"]
            want = named[f"recoveredXage_{age}Xreach_{reach}"]
            np.testing.assert_allclose(np.asarray(got.array()), np.asarray(want.array()))


def test_per_request_naming_overrides_the_set() -> None:
    out = OutputSet(naming="_{prop}={trait}")
    out.compartments("default", STATE["I"], by=AGE)
    added = out.fan_out("custom", out["default"], AGE, naming="{name}.{age}")
    assert "default_age=young" in out
    assert added == ("custom.young", "custom.old")
    out.compartments("callable", STATE["I"], by=AGE, naming=lambda n, ts: n + ts[0].name.upper())
    assert "callableYOUNG" in out


def test_suffix_template_per_request() -> None:
    out = OutputSet()
    out.compartments("prev", STATE["I"], by=[AGE, (AGE, REACH)], naming="__{trait}")
    assert "prev__young" in out
    assert "prev__old__far" in out


def test_whole_name_template_names_unknown_property() -> None:
    out = OutputSet()
    with pytest.raises(KeyError, match="state"):
        out.compartments("prev", STATE["I"], by=AGE, naming="{name}_{state}_{age}")


def test_whole_name_template_must_name_every_grouping_property() -> None:
    out = OutputSet()
    with pytest.raises(ValueError, match="two strata"):
        out.compartments("prev", STATE["I"], by=[(AGE, REACH)], naming="{name}_{age}")


# --- gaps 5 and 6 (CX54): TraitChain pairs within a state property, expression rates ---------

TB = Property("tb", ("incipient", "low", "high", "cleared"))
TB_MAP = PropertyMap.from_property(TB).stratify(AGE)
TB_Y0 = np.array([100.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SPLIT = (("incipient", "low"), ("incipient", "high"))
PARAMS_6 = {"rate": 0.4, "p": 0.3, "two": 2.0, "x": 0.05}


def _tb(flow: TransitionFlow) -> np.ndarray:
    model = FlowModel(TB_MAP)
    model.add_flow(flow)
    model.add_flow(TransitionFlow("clear", TB["low"], TB["cleared"], 0.1))
    cm = model.compile()
    return np.asarray(cm.run(PARAMS_6, TB_Y0, **EULER).final_state.data)


def test_expression_rates_match_dest_adjustments() -> None:
    p = Param("p")
    chain = TransitionFlow(
        "progression",
        TB.present(),
        TB.present(),
        Param("rate"),
        pairing=TraitChain(TB, SPLIT, rates=(1 - p, p)),
    )
    by_hand = TransitionFlow(
        "progression",
        TB.present(),
        TB.present(),
        Param("rate"),
        pairing=TraitChain(TB, SPLIT),
        adjust=[Multiply(1 - p, where=Dest(TB["low"])), Multiply(p, where=Dest(TB["high"]))],
    )
    np.testing.assert_allclose(_tb(chain), _tb(by_hand), rtol=1e-12)


def test_mixed_number_and_expression_rates() -> None:
    mixed = TraitChain(TB, SPLIT, rates=(0.5, Param("p")))
    assert mixed.static_scale(0) == 0.5
    assert mixed.static_scale(1) == 1.0
    assert len(mixed.rate_adjustments()) == 1
    numbers = TraitChain(TB, SPLIT, rates=(0.5, 0.3))
    flows = [
        TransitionFlow("progression", TB.present(), TB.present(), 0.4, pairing=pairing)
        for pairing in (mixed, numbers)
    ]
    np.testing.assert_allclose(_tb(flows[0]), _tb(flows[1]), rtol=1e-12)


def test_expression_rate_is_overwritten_like_a_scale() -> None:
    def flow(rate: Any) -> TransitionFlow:
        return TransitionFlow(
            "progression",
            TB.present(),
            TB.present(),
            Param("rate"),
            pairing=TraitChain(TB, (("incipient", "low"),), rates=(rate,)),
            adjust=[Overwrite(Param("x"), where=AGE["old"])],
        )

    np.testing.assert_allclose(_tb(flow(Param("two"))), _tb(flow(2.0)), rtol=1e-12)


def test_declared_adjustments_are_unchanged() -> None:
    p = Param("p")
    flow = TransitionFlow(
        "progression",
        TB.present(),
        TB.present(),
        1.0,
        pairing=TraitChain(TB, SPLIT, rates=(1 - p, p)),
    )
    assert flow.adjust == ()


def test_chain_rate_must_be_a_number_or_expression() -> None:
    with pytest.raises(TypeError, match="number or a rate expression"):
        TraitChain(TB, SPLIT, rates=("fast", 0.1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="number or a rate expression"):
        TraitChain(TB, SPLIT, rates=(True, 0.1))


def test_trait_chain_docstring_covers_state_pairs() -> None:
    doc = TraitChain.__doc__ or ""
    assert "clinical_progression" in doc
    assert "rates=(1 - p, p)" in doc


# --- gap 7 (CX55): a plain Capture keeps its property map ------------------------------------


def test_grouped_output_of_a_plain_capture_has_its_property_map() -> None:
    by_age = Capture("by_age", Reduce(AGE))
    cm = _sir(Param("b") * by_age / 1000.0)
    assert [p.name for p in cm.capture_meta["by_age"]] == ["age"]
    out = OutputSet()
    out["captured"] = GroupedOutput("by_age")
    out["saved"] = Reduce(AGE)
    named = cm.run({"b": 0.3}, Y0, outputs=out, **EULER)
    assert named["captured"].dims == ("time", "age")
    young = named["captured"].select(AGE["young"]).sum_by()
    np.testing.assert_allclose(
        np.asarray(young.array()), np.asarray(named["saved"].select(AGE["young"]).sum_by().array())
    )


def test_raw_save_of_a_plain_capture_is_property_data() -> None:
    cm = _sir(Param("b") * Capture("by_age", Reduce(AGE)) / 1000.0)
    plan = SavePlan(requests={"cap": SaveRequest(GroupedOutput("by_age"))})
    result = cm.run({"b": 0.3}, Y0, save=plan, **EULER)
    assert result["cap"].dims == ("time", "age")
    assert isinstance(result["cap"].values, PropertyData)


# --- W4 (CX56): differences(sign=-1) is averted burden ---------------------------------------


def _runs() -> PosteriorRuns:
    rng = np.random.default_rng(3)
    ref = rng.gamma(4.0, 25.0, size=(400, 3))
    scen = ref * rng.uniform(0.3, 1.1, size=(400, 3))
    return PosteriorRuns(
        times=np.array([0.0, 1.0, 2.0]),
        series={"baseline": {"deaths": ref}, "screening": {"deaths": scen}},
        n=400,
        scenario_names=("baseline", "screening"),
        output_names=("deaths",),
        draws={},
    )


def test_sign_minus_one_is_quantiles_of_averted_draws() -> None:
    runs = _runs()
    q = (0.025, 0.5, 0.975)
    frame = runs.differences("baseline", {"averted": "deaths"}, at=1.0, q=q, sign=-1)["screening"]
    ref = runs.samples("baseline", "deaths")[:, 1]
    scen = runs.samples("screening", "deaths")[:, 1]
    np.testing.assert_allclose(frame["averted"].to_numpy(), np.quantile(ref - scen, q))
    np.testing.assert_allclose(
        frame["averted_relative"].to_numpy(), np.quantile((ref - scen) / ref, q)
    )
    plain = runs.differences("baseline", {"averted": "deaths"}, at=1.0, q=q)["screening"]
    assert not np.allclose(frame["averted"].to_numpy(), -plain["averted"].to_numpy())
    np.testing.assert_allclose(frame["averted"].to_numpy(), -plain["averted"].to_numpy()[::-1])
    assert isinstance(frame, pd.DataFrame)


def test_sign_must_be_plus_or_minus_one() -> None:
    with pytest.raises(ValueError, match="sign"):
        _runs().differences("baseline", {"d": "deaths"}, at=1.0, sign=2)  # type: ignore[arg-type]


# --- W7, part (CX58): Optax passes value_fn, so optax.lbfgs runs -----------------------------


def test_optax_lbfgs_runs_under_optimize() -> None:
    optax = pytest.importorskip("optax")
    from tests.test_calib_workflow_optimize import _sir_bm

    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=1), batch_size=8).best(2)
    run = wf.optimize(
        bm, starts, method=wf.Optax(optax.lbfgs()), chunk_steps=10, max_steps=30, seed=0
    )
    fitted = np.asarray(run.candidates.params["infection"])
    np.testing.assert_allclose(fitted, 0.35, atol=0.02)
