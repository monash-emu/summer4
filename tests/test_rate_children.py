"""Characterisation of the rate-tree traversal passes (roadmap step 35).

Five passes walk a rate expression, all now folding over ``rate_children``:

- ``_flow_refs`` and ``_field_paths`` (``summer4.flows.rates``);
- ``_collect_capture_meta`` (``summer4.flows.compiled``);
- ``rate_stage`` and ``build_hoist_table`` (``summer4.flows.stages``).

This file records what each returns for a corpus of expressions, as literals. It was first
written against the code before ``rate_children`` replaced the duplicated child walks, so the
refactor could show it changed nothing except the deliberate fixes. The expectations now pin the
refolded passes. A line marked ``# fixed in step 35`` recorded a wrong answer before the refold
and now records the corrected one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    FlowRef,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Time,
    TransitionFlow,
    floor,
)
from summer4.data import Data
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix
from summer4.flows.compiled import _collect_capture_meta
from summer4.flows.rates import (
    ArrayConst,
    Capture,
    Const,
    Defer,
    Lookup,
    RateOps,
    Reduce,
    _field_paths,
    _flow_refs,
)
from summer4.flows.stages import build_hoist_table, rate_stage
from summer4.timevarying import gaussian_pulse, linear

_STATE = Property("state", ("S", "I", "R"))
_AGE = Property("age", ("young", "old"))
_ONE_AGE = Property("age", ("all",))


def _double(x: Any) -> Any:
    return 2.0 * x


def _add(x: Any, y: Any) -> Any:
    return x + y


def _foi(contact_rate: object) -> ForceOfInfection:
    return ForceOfInfection(
        "infection",
        infectious=_STATE["I"],
        group_by=_AGE,
        kind=FOIKind.FREQUENCY,
        contact_rate=contact_rate,
        mixing=MixingMatrix(_AGE, np.eye(2), check_reciprocal=False),
    )


def _table(arg: RateOps | None = None) -> RateOps:
    """A ``TableInterp`` over age whose argument is ``arg`` (default ``Time()``)."""
    return Data.table(np.array([0.0, 10.0]), np.array([[1.0, 2.0], [3.0, 4.0]]), over=_AGE).interp(
        arg=arg
    )


@dataclass(frozen=True)
class Case:
    """One expression and what every pass returns for it today."""

    id: str
    build: Callable[[], RateOps]
    flow_refs: set[str]
    field_paths: set[tuple[str, ...]]
    capture_meta: dict[str, tuple[str, ...]]  # capture name -> property names
    stage_static: str
    stage_dynamic: str
    hoist: list[tuple[str, str]]  # sorted (type(node).__name__, part)


CASES: tuple[Case, ...] = (
    Case(
        id="const",
        build=lambda: Const(2.0),
        flow_refs=set(),
        field_paths=set(),
        capture_meta={},
        stage_static="run",
        stage_dynamic="run",
        hoist=[],
    ),
    Case(
        id="time",
        build=lambda: Time(),
        flow_refs=set(),
        field_paths=set(),
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="param",
        build=lambda: Param("a"),
        flow_refs=set(),
        field_paths={("a",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="flow_ref",
        build=lambda: FlowRef("x"),
        flow_refs={"x"},
        field_paths=set(),
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="array_const",
        build=lambda: ArrayConst([1.0, 2.0]),
        flow_refs=set(),
        field_paths=set(),
        capture_meta={},
        stage_static="run",
        stage_dynamic="run",
        hoist=[],
    ),
    Case(
        id="binop_params",
        build=lambda: Param("a") * Param("b"),
        flow_refs=set(),
        field_paths={("a",), ("b",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[("BinOp", "value")],
    ),
    Case(
        id="unary_param",
        build=lambda: np.exp(Param("a")),
        flow_refs=set(),
        field_paths={("a",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[("UnaryOp", "value")],
    ),
    Case(
        id="interp_param_knots",
        build=lambda: linear(Time(), [0.0, 10.0], [Param("a"), Param("b")]),
        flow_refs=set(),
        field_paths={("a",), ("b",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[("Interp", "breakpoints"), ("Interp", "values")],
    ),
    Case(
        id="gaussian_pulse",
        build=lambda: gaussian_pulse(Time(), Param("c"), 2.0, Param("h")),
        flow_refs=set(),
        field_paths={("c",), ("h",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="table_interp_time",
        build=lambda: _table(),
        flow_refs=set(),
        field_paths=set(),
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="table_interp_param_arg",
        build=lambda: _table(Param("a") * Param("b")),
        flow_refs=set(),
        field_paths={("a",), ("b",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[("BinOp", "value")],
    ),
    Case(
        id="lookup",
        build=lambda: Lookup(
            Param("stack"), floor(Time()) + Param("a") * Param("b") + FlowRef("x")
        ),
        flow_refs={"x"},
        field_paths={("a",), ("b",), ("stack",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[("BinOp", "value")],
    ),
    Case(
        id="lookup_params_only",
        build=lambda: Lookup(Param("stack"), Param("a") * Param("b")),
        flow_refs=set(),
        field_paths={("a",), ("b",), ("stack",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[("Lookup", "value")],
    ),
    Case(
        id="capture_binop",
        build=lambda: Capture("c", Param("a") * FlowRef("x")),
        flow_refs={"x"},
        field_paths={("a",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="defer_flow_ref",
        build=lambda: Defer(_double, FlowRef("x")),
        flow_refs={"x"},
        field_paths=set(),
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="defer_nested",
        build=lambda: Defer(_add, Defer(_double, FlowRef("x")), Param("a")),
        flow_refs={"x"},
        field_paths={("a",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="defer_params_only",
        build=lambda: Defer(_double, Param("a") * Param("b")),
        flow_refs=set(),
        field_paths={("a",), ("b",)},
        capture_meta={},
        stage_static="run",
        stage_dynamic="step",
        hoist=[("Defer", "value")],
    ),
    Case(
        id="defer_nested_hoist",
        build=lambda: Defer(_add, Defer(_double, Param("a") * Param("b")), FlowRef("x")),
        flow_refs={"x"},
        field_paths={("a",), ("b",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[("Defer", "value")],
    ),
    Case(
        id="capture_in_defer",
        build=lambda: Defer(_double, Capture("c", Param("a") * FlowRef("x"))),
        flow_refs={"x"},
        field_paths={("a",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="capture_in_table_interp",
        build=lambda: _table(Capture("c", Param("a") * FlowRef("x"))),
        flow_refs={"x"},
        field_paths={("a",)},
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_flow_ref",
        build=lambda: _foi(Param("beta") * FlowRef("vacc")),
        flow_refs={"vacc"},  # fixed in step 35
        field_paths={("beta",)},
        capture_meta={"infection": ("age",)},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_params_only",
        build=lambda: _foi(Param("beta") * Param("scale")),
        flow_refs=set(),
        field_paths={("beta",), ("scale",)},
        capture_meta={"infection": ("age",)},
        stage_static="step",
        stage_dynamic="step",
        hoist=[("BinOp", "value")],  # fixed in step 35: the contact-rate BinOp is slotted
    ),
    Case(
        id="foi_in_binop",
        build=lambda: Param("a") * _foi(Param("beta")),
        flow_refs=set(),
        field_paths={("a",), ("beta",)},
        capture_meta={"infection": ("age",)},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_defer",
        build=lambda: Defer(_double, _foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",)},
        capture_meta={"infection": ("age",)},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_capture",
        build=lambda: Capture("c", _foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",)},
        capture_meta={"infection": ("age",)},  # fixed in step 35: Capture folds over its inner
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_table_interp",
        build=lambda: _table(_foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",)},
        capture_meta={"infection": ("age",)},  # fixed in step 35
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_lookup",
        build=lambda: Lookup(Param("stack"), _foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",), ("stack",)},
        capture_meta={"infection": ("age",)},  # fixed in step 35
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="reduce",
        build=lambda: Reduce(sum_over="age"),
        flow_refs=set(),
        field_paths=set(),
        capture_meta={},
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
)

_PARAMS = [pytest.param(case, id=case.id) for case in CASES]


@pytest.mark.parametrize("case", _PARAMS)
def test_flow_refs(case: Case) -> None:
    assert _flow_refs(case.build()) == case.flow_refs


@pytest.mark.parametrize("case", _PARAMS)
def test_field_paths(case: Case) -> None:
    assert _field_paths(case.build()) == case.field_paths


@pytest.mark.parametrize("case", _PARAMS)
def test_capture_meta(case: Case) -> None:
    meta = _collect_capture_meta(case.build())
    assert {name: tuple(p.name for p in props) for name, props in meta.items()} == (
        case.capture_meta
    )


@pytest.mark.parametrize("case", _PARAMS)
def test_rate_stage_params_static(case: Case) -> None:
    assert rate_stage(case.build(), params_are_static=True) == case.stage_static


@pytest.mark.parametrize("case", _PARAMS)
def test_rate_stage_params_dynamic(case: Case) -> None:
    assert rate_stage(case.build(), params_are_static=False) == case.stage_dynamic


@pytest.mark.parametrize("case", _PARAMS)
def test_hoist_table(case: Case) -> None:
    table = build_hoist_table([case.build()], params_are_static=True)
    got = sorted((type(entry.node).__name__, entry.part) for entry in table.entries)
    assert got == case.hoist


# Regression: a FlowRef inside a ForceOfInfection is ordered, and hoisting that FOI is exact.

_VAX_PARAMS = {"beta": 0.4, "vax_effect": 0.5, "pop_scale": 100.0, "nu": 0.1}


def _vax_foi(*, reads: str = "vaccination") -> ForceOfInfection:
    """FOI whose contact rate falls with the total flow ``reads``."""
    k = Param("vax_effect") / Param("pop_scale")  # parameter-only: hoistable
    return ForceOfInfection(
        "infection",
        infectious=_STATE["I"],
        group_by=_ONE_AGE,
        kind=FOIKind.FREQUENCY,
        contact_rate=Param("beta") * (1.0 - k * FlowRef(reads).sum()),
        mixing=MixingMatrix(_ONE_AGE, [[1.0]], check_reciprocal=False),
    )


def _vax_model(*, reads: str = "vaccination") -> FlowModel:
    """S -> I via the FOI, and S -> R at ``nu``. Infection is added first on purpose."""
    pmap = PropertyMap.from_property(_STATE).stratify(_ONE_AGE)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", _STATE["S"], _STATE["I"], _vax_foi(reads=reads)))
    model.add_flow(TransitionFlow("vaccination", _STATE["S"], _STATE["R"], Param("nu")))
    model.add_flow(TransitionFlow("recovery", _STATE["I"], _STATE["R"], 0.1))
    return model


def _vax_rhs(y: np.ndarray) -> np.ndarray:
    """Hand-written derivative, ordered by hand: vaccination flow first."""
    p = _VAX_PARAMS
    s, i, r = y
    vaccination = p["nu"] * s
    contact = p["beta"] * (1.0 - p["vax_effect"] / p["pop_scale"] * vaccination)
    infection = contact * i / (s + i + r) * s
    recovery = 0.1 * i
    return np.array([-infection - vaccination, infection - recovery, vaccination + recovery])


def test_flow_ref_in_foi_orders_vaccination_before_infection() -> None:
    cm = _vax_model().compile()
    assert cm.order.index("vaccination") < cm.order.index("infection")
    y = np.array([900.0, 100.0, 0.0])
    got = np.asarray(cm.vector_field(0.0, y, _VAX_PARAMS))
    np.testing.assert_allclose(got, _vax_rhs(y), rtol=1e-6)


def test_foi_reading_its_own_flow_is_a_cycle() -> None:
    with pytest.raises(ValueError, match="Cyclic"):
        _vax_model(reads="infection").compile()


def _euler_final(cm: Any, y0: np.ndarray, *, dt: float, t1: float) -> np.ndarray:
    pmap = cm.pmap
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([t1]))
    res = cm.run(
        _VAX_PARAMS, PropertyData.wrap(pmap, y0), t0=0.0, t1=t1, dt=dt, save=plan, solver="euler"
    )
    return np.asarray(res["y"].values.data)


def test_foi_flow_ref_trajectory_matches_hand_ordered_reference() -> None:
    y0 = np.array([900.0, 100.0, 0.0])
    got = _euler_final(_vax_model().compile(), y0, dt=0.25, t1=10.0)
    ref = y0.copy()
    for _ in range(40):
        ref = ref + 0.25 * _vax_rhs(ref)
    np.testing.assert_allclose(got.reshape(-1), ref, rtol=1e-6)


def test_foi_hoist_on_and_off_agree_and_slot_the_parameter_children() -> None:
    cm_on = _vax_model().compile(hoist=True)
    cm_off = _vax_model().compile(hoist=False)
    assert cm_on.hoist_table is not None
    assert ("BinOp", "value") in [
        (type(entry.node).__name__, entry.part) for entry in cm_on.hoist_table.entries
    ]
    y0 = np.array([900.0, 100.0, 0.0])
    np.testing.assert_allclose(
        _euler_final(cm_on, y0, dt=0.25, t1=10.0),
        _euler_final(cm_off, y0, dt=0.25, t1=10.0),
        rtol=1e-12,
    )
