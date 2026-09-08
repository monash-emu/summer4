"""Kleene algebra and selector composition tests."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import Everything, Nothing, Property, PropertyMap
from summer4.selectors import And, Not, Or, SelectorOps


def _sir_age_severity() -> tuple[Property, Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("child", "adult"))
    sev = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(age).stratify(sev, where=state["I"])
    return state, age, sev, pm


def test_kleene_not_truth_table() -> None:
    """~ maps T->F, U->U, F->T via negation of the int8 encoding."""
    values = np.array([1, 0, -1], dtype=np.int8)
    assert np.array_equal(-values, np.array([-1, 0, 1], dtype=np.int8))


def test_kleene_and_or_truth_tables() -> None:
    """min/max implement Kleene AND/OR on {-1, 0, 1}."""
    a = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int8)
    b = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1], dtype=np.int8)
    expected_and = np.array([1, 0, -1, 0, 0, -1, -1, -1, -1], dtype=np.int8)
    expected_or = np.array([1, 1, 1, 1, 0, 0, 1, 0, -1], dtype=np.int8)
    assert np.array_equal(np.minimum(a, b), expected_and)
    assert np.array_equal(np.maximum(a, b), expected_or)


def test_operators_build_and_or_not() -> None:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("child", "adult"))
    assert (state["I"] & age["child"]) == And(state["I"], age["child"])
    assert (state["I"] | age["child"]) == Or(state["I"], age["child"])
    assert ~state["I"] == Not(state["I"])


def test_selector_bool_raises() -> None:
    state = Property("state", ("S", "I"))
    with pytest.raises(TypeError, match="cannot be used as booleans"):
        bool(state["I"])
    with pytest.raises(TypeError, match="cannot be used as booleans"):
        if state["I"]:  # noqa: SIM102
            pass


def test_and_with_non_selector_is_notimplemented() -> None:
    state = Property("state", ("S", "I"))
    with pytest.raises(TypeError):
        _ = state["I"] & 1  # type: ignore[operator]


def test_everything_and_nothing() -> None:
    state, _age, _sev, pm = _sir_age_severity()
    assert pm.select(Everything()).tolist() == list(range(pm.size))
    assert pm.select(Nothing()).tolist() == []
    assert pm.select(Everything() & state["S"]).tolist() == pm.select(state["S"]).tolist()
    assert pm.select(Nothing() | state["S"]).tolist() == pm.select(state["S"]).tolist()


def test_selectors_are_hashable() -> None:
    state = Property("state", ("S", "I"))
    sel = state["S"] & ~state["I"]
    assert isinstance(sel, SelectorOps)
    cache = {sel: "ok"}
    assert cache[state["S"] & ~state["I"]] == "ok"
