"""``sum_by``: one name for summing across strata."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowMass,
    FlowModel,
    FlowRef,
    OutputSet,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Reduce,
    TransitionFlow,
)
from summer4.flows.compiled import CompiledModel

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("child", "adult"))
STRAIN = Property("strain", ("a", "b", "c"))
# Ragged: only infectious compartments carry a strain.
PMAP = PropertyMap.from_property(STATE).stratify(AGE).stratify(STRAIN, where=STATE["I"])
Y = np.arange(1.0, PMAP.size + 1.0)
RUN = dict(t0=0.0, t1=5.0, dt=1.0, solver="euler")


def _model() -> CompiledModel:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], Param("gamma")))
    return model.compile()


def _run() -> object:
    return _model().run({"gamma": 0.1}, Y, **RUN)


def _by_hand(keep: dict[str, str]) -> float:
    total = 0.0
    for label, value in zip(PMAP.labels(), Y, strict=True):
        if all(f"{prop}={trait}" in label.split("_") for prop, trait in keep.items()):
            total += value
    return total


def test_property_data_sum_by_none_one_and_two() -> None:
    pd = PropertyData.wrap(PMAP, Y)
    assert float(pd.sum_by()) == pytest.approx(Y.sum())
    by_age = pd.sum_by(AGE)
    assert by_age.pmap.labels() == ("age=child", "age=adult")
    np.testing.assert_allclose(
        np.asarray(by_age.data), [_by_hand({"age": "child"}), _by_hand({"age": "adult"})]
    )
    by_both = pd.sum_by(AGE, STRAIN)
    assert by_both.pmap.labels() == tuple(
        f"age={a}_strain={s}" for a in AGE.traits for s in STRAIN.traits
    )
    expected = [_by_hand({"age": a, "strain": s}) for a in AGE.traits for s in STRAIN.traits]
    np.testing.assert_allclose(np.asarray(by_both.data), expected)


def test_sum_by_repeated_property_raises() -> None:
    with pytest.raises(ValueError, match="repeats"):
        PropertyData.wrap(PMAP, Y).sum_by(AGE, "age")


def test_output_sum_by_keeps_time() -> None:
    comps = _run()["compartments"]
    assert comps.sum_by().dims == ("time",)
    np.testing.assert_allclose(
        np.asarray(comps.sum_by().values), np.asarray(comps.values.data).sum(axis=-1)
    )
    by_two = comps.sum_by(AGE, STRAIN)
    assert by_two.dims == ("time", "group")
    np.testing.assert_allclose(
        np.asarray(by_two.values.data)[0],
        [_by_hand({"age": a, "strain": s}) for a in AGE.traits for s in STRAIN.traits],
    )


def test_output_sum_by_on_a_time_series_raises() -> None:
    series = _run()["compartments"].sum_by()
    with pytest.raises(ValueError, match="keeps time"):
        series.sum_by()


def test_output_sum_by_side_rules() -> None:
    res = _run()
    with pytest.raises(ValueError, match="side"):
        res["recovery"].sum_by(AGE)
    with pytest.raises(ValueError, match="side"):
        res["compartments"].sum_by(AGE, side="source")
    with pytest.raises(ValueError, match="side"):
        res["compartments"].sum_by(side="source")


def test_edge_sum_by_one_and_two_properties() -> None:
    flow = _run()["recovery"]
    mass = np.asarray(flow.values.data)
    by_age = flow.sum_by(AGE, side="source")
    np.testing.assert_allclose(
        np.asarray(by_age.values.data).sum(axis=-1), mass.sum(axis=-1), rtol=1e-6
    )
    by_two = flow.sum_by(AGE, STRAIN, side="source")
    assert by_two.values.pmap.labels() == tuple(
        f"age={a}_strain={s}" for a in AGE.traits for s in STRAIN.traits
    )
    # Each recovery edge leaves one (age, strain) infectious compartment, so the columns are
    # the edges themselves, in the same order.
    np.testing.assert_allclose(np.asarray(by_two.values.data), mass)


def test_output_expression_records_sum_by() -> None:
    expr = Compartments().sum_by(AGE)
    assert expr.method == "sum_by"
    outputs = OutputSet()
    outputs["by_age"] = Compartments().sum_by(AGE)
    outputs["all"] = outputs.ref("by_age").sum_by()
    res = _model().run({"gamma": 0.1}, Y, save=outputs.plan(), **RUN)
    named = outputs.evaluate(res, {"gamma": 0.1})
    direct = res["by_age"].sum_by(AGE).sum_by()
    np.testing.assert_allclose(np.asarray(named["all"].values), np.asarray(direct.values))


def test_flowref_sum_by() -> None:
    assert FlowRef("x").sum_by() == FlowRef("x", reduce="sum")
    assert FlowRef("x").sum_by(AGE) == FlowRef("x", reduce=("sum_by", "age"))
    with pytest.raises(ValueError, match="at most one"):
        FlowRef("x").sum_by(AGE, STRAIN)


def test_group_by_fields() -> None:
    assert Compartments(group_by=AGE).group_by == AGE
    assert FlowMass("recovery", group_by=(AGE, "source")).group_by == (AGE, "source")
    assert Reduce(AGE).group_by == AGE
    assert Reduce().group_by is None
