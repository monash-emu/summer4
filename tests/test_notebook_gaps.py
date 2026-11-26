"""Code paths that only the example notebooks used to reach.

A coverage run (the suite without notebooks against each notebook, branch
coverage per test context) found these few lines covered by a notebook and by
nothing in ``tests/``. Each test names the notebook it stands in for, so the slow
notebooks can leave the per-PR test job without losing coverage. Tuning's
``on_pilot`` callback is checked in ``test_calib_workflow_tune.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from summer4 import (
    Compartments,
    ExitFlow,
    FlowModel,
    Output,
    OutputSet,
    Param,
    Params,
    Property,
    PropertyData,
    PropertyMap,
    Result,
    SavePlan,
    SaveRequest,
    TimeAxis,
    TransitionFlow,
    field_refs,
)
from summer4.data import Data
from summer4.epi.calibration import workflow as wf
from summer4.flows.rates import FieldRef


def test_select_trait_set_is_tri_state_where_the_property_is_absent() -> None:
    """01-taxonomy: ``age[("0-4", "5-9")]`` (an ``IsIn``) combined with a partial property."""
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    sev = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(age).stratify(sev, where=state["I"])

    young = set(pm.select(age[("0-4", "5-9")]).tolist())
    assert young == set(pm.select(age["0-4"]).tolist()) | set(pm.select(age["5-9"]).tolist())
    assert young.isdisjoint(pm.select(age["10+"]).tolist())

    # severity is absent on S and R, so ``sev[...]`` is unknown there and only I matches.
    either = set(pm.select(sev[("mild", "severe")]).tolist())
    assert either == set(pm.select(state["I"]).tolist())
    not_severe = set(pm.select(age[("0-4", "5-9")] & ~sev["severe"]).tolist())
    assert not_severe == young & set(pm.select(sev["mild"]).tolist())

    with pytest.raises(KeyError, match="Unknown trait"):
        age[("0-4", "70+")]


def _sir() -> tuple[Property, PropertyMap, FlowModel]:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.25))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    return state, pmap, model


def test_plan_description_prints_one_line_per_output() -> None:
    """04-running-and-results: ``print(cm.describe(plan, ...))``."""
    _state, pmap, model = _sir()
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    desc = cm.describe(plan, y0=y0, n_saves=61)
    text = str(desc)
    lines = text.splitlines()
    assert lines[0] == f"PlanDescription(n_saves=61, total={desc.total_nbytes:,} B)"
    assert len(lines) == 1 + len(desc.outputs)
    assert lines[1].startswith("  compartments: shape=(61, 3)")


def test_time_only_output_to_frame_has_time_and_value_columns() -> None:
    """04-running-and-results: ``select(...).sum_by().to_frame()``."""
    state, pmap, model = _sir()
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=10, dt=1.0, save=plan, solver="euler")
    prevalence = res["compartments"].select(state["I"]).sum_by()
    frame = prevalence.to_frame()
    assert frame.columns == ["time", "value"]
    assert frame.height == 11
    np.testing.assert_allclose(
        frame["value"].to_numpy(), np.asarray(prevalence.values).reshape(-1), rtol=1e-6
    )


def test_data_table_takes_a_frame_whose_columns_are_the_traits() -> None:
    """13-rate-math-and-tables: ``Data.table(times, DataFrame, over=age)``."""
    age = Property("age", ("0", "15", "65"))
    alive = Property("state", ("Y",))
    times = np.array([2000.0, 2010.0, 2020.0])
    deaths = pd.DataFrame(
        {"0": [0.020, 0.015, 0.010], "15": [0.004, 0.005, 0.006], "65": [0.040, 0.045, 0.050]}
    )
    table = Data.table(times, deaths, over=age)
    pmap = PropertyMap.from_property(alive).stratify(age)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("die", alive["Y"], table.interp()))
    pop = PropertyData.wrap(pmap, np.ones(pmap.size))
    mid = np.asarray(model.compile().vector_field(2005.0, pop, {}).data)
    expect = -0.5 * (deaths.iloc[0].to_numpy() + deaths.iloc[1].to_numpy())
    np.testing.assert_allclose(mid, expect, rtol=1e-5)

    with pytest.raises(ValueError, match="do not match property"):
        Data.table(times, deaths[["15", "0", "65"]], over=age)


def test_numpy_functions_without_a_rate_meaning_refuse_an_expression() -> None:
    """15-array-dispatch: ``jnp.sin(Param(...))`` refuses; so does any non-listed NumPy function."""
    with pytest.raises(TypeError):
        np.mean(Param("x"))
    with pytest.raises(TypeError):
        np.concatenate([Param("x"), Param("y")])
    with pytest.raises(TypeError):
        jnp.sin(Param("phase"))


def test_binary_numpy_ufunc_on_an_output_ref_matches_the_direct_call() -> None:
    """26-lazy-idiom-bridges: ``np.minimum(outputs.ref("incidence"), CAP)``."""
    values = np.array([1.0, 4.0, 9.0, 2.0])
    axis = TimeAxis(values=np.arange(4, dtype=np.float64), epoch=None, kind="explicit")
    result = Result(
        times=axis, outputs={"incidence": Output(times=axis, values=values, dims=("time",))}
    )
    outputs = OutputSet()
    outputs["incidence"] = Compartments()
    outputs["capped"] = np.minimum(outputs.ref("incidence"), 3.0)
    outputs["floored"] = np.maximum(2.0, outputs.ref("incidence"))
    named = outputs.evaluate(result, None)
    np.testing.assert_allclose(np.asarray(named["capped"].values), np.minimum(values, 3.0))
    np.testing.assert_allclose(np.asarray(named["floored"].values), np.maximum(2.0, values))


def test_binary_numpy_ufunc_on_an_evaluated_output_returns_an_output() -> None:
    """26-lazy-idiom-bridges: ``np.minimum(named["incidence"], CAP)`` on an evaluated output."""
    values = np.array([1.0, 4.0, 9.0, 2.0])
    axis = TimeAxis(values=np.arange(4, dtype=np.float64), epoch=None, kind="explicit")
    incidence = Output(times=axis, values=values, dims=("time",))
    capped = np.minimum(incidence, 3.0)
    assert isinstance(capped, Output) and capped.dims == ("time",)
    np.testing.assert_allclose(np.asarray(capped.values), np.minimum(values, 3.0))
    floored = np.maximum(2.0, incidence)
    assert isinstance(floored, Output)
    np.testing.assert_allclose(np.asarray(floored.values), np.maximum(2.0, values))


def test_candidates_to_frame_has_one_row_per_candidate() -> None:
    """19-calibration-design, 22-calibration-seeded-mcmc: ``design.to_frame()``."""
    cands = wf.Candidates(
        sites=("a", "b"),
        z={"a": np.zeros(3), "b": np.zeros(3)},
        params={"a": np.array([0.1, 0.2, 0.3]), "b": np.array([1.0, 2.0, 3.0])},
        log_density=np.array([-1.0, -2.0, np.nan]),
        ok=np.array([True, True, False]),
    )
    frame = cands.to_frame()
    assert list(frame.columns) == ["a", "b", "log_density", "ok"]
    assert len(frame) == 3
    np.testing.assert_allclose(frame["b"].to_numpy(), [1.0, 2.0, 3.0])
    assert frame["ok"].tolist() == [True, True, False]

    bare = wf.Candidates(sites=("a",), z={"a": np.zeros(2)}, params={"a": np.ones(2)})
    assert list(bare.to_frame().columns) == ["a"]


class _Unresolvable(NamedTuple):
    # A string annotation naming nothing in this module: ``get_type_hints`` raises
    # NameError, as it does for a schema class defined inside ``exec`` (notebooks
    # 40-save-expressions and 41-kiribati-gaps under the notebook runner).
    rate: _NotDefinedAnywhere  # type: ignore[name-defined]  # noqa: F821
    scale: float


def test_field_refs_tolerates_annotations_that_do_not_resolve() -> None:
    """40-save-expressions, 41-kiribati-gaps: ``field_refs(..., Schema)``."""
    refs = field_refs(Params(), _Unresolvable)
    assert isinstance(refs, _Unresolvable)
    assert isinstance(refs.rate, FieldRef) and refs.rate.path == ("rate",)
    assert isinstance(refs.scale, FieldRef) and refs.scale.path == ("scale",)
