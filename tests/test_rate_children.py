"""Characterisation of the rate-tree traversal passes (roadmap step 35, part 1).

Each pass walks a rate expression its own way today:

- ``_flow_refs`` and ``_field_paths`` (``summer4.flows.rates``);
- ``_collect_capture_meta`` (``summer4.flows.compiled``);
- ``rate_stage`` and ``build_hoist_table`` (``summer4.flows.stages``).

This file records what each returns for a corpus of expressions, as literals. It was written
against the code before ``rate_children`` replaced the duplicated child walks, so that the
refactor can show it changed nothing except the deliberate fixes. A line marked
``# step 35 changes this`` records today's answer where today's answer is wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from summer4 import FlowRef, Param, Property, Time, floor
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
        flow_refs=set(),  # step 35 changes this: should be {"vacc"}
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
        hoist=[],  # step 35 changes this: the contact-rate BinOp gets a value slot
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
        capture_meta={},  # step 35 changes this: Capture should fold over its inner
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_table_interp",
        build=lambda: _table(_foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",)},
        capture_meta={},  # step 35 changes this: should be {"infection": ("age",)}
        stage_static="step",
        stage_dynamic="step",
        hoist=[],
    ),
    Case(
        id="foi_in_lookup",
        build=lambda: Lookup(Param("stack"), _foi(Param("beta"))),
        flow_refs=set(),
        field_paths={("beta",), ("stack",)},
        capture_meta={},  # step 35 changes this: should be {"infection": ("age",)}
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
