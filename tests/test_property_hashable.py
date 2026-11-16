"""A Property built from a list is a tuple-backed, hashable value usable as a split= key."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import FlowModel, Param, Property, PropertyMap, TransitionFlow


def test_list_traits_are_stored_as_tuple() -> None:
    prop = Property("focus", ["low", "high"])
    assert prop.traits == ("low", "high")
    assert isinstance(prop.traits, tuple)


def test_list_and_tuple_built_properties_are_equal_and_hash_alike() -> None:
    from_list = Property("focus", ["low", "high"])
    from_tuple = Property("focus", ("low", "high"))
    assert from_list == from_tuple
    assert hash(from_list) == hash(from_tuple)
    assert {from_list: 1}[from_tuple] == 1


def test_traits_from_any_iterable() -> None:
    prop = Property("age", (f"a{i}" for i in range(3)))
    assert prop.traits == ("a0", "a1", "a2")
    hash(prop)


def test_string_traits_are_rejected() -> None:
    with pytest.raises(TypeError, match="not a string"):
        Property("stage", "abc")


def test_mangled_string_traits_are_rejected() -> None:
    with pytest.raises(TypeError, match="not a string"):
        Property._mangled("age@dest", "abc")


STAGE = Property("stage", ("incipient", "active"))


def _progression(split: object) -> TransitionFlow:
    return TransitionFlow(
        "progression", STAGE["incipient"], STAGE["active"], Param("rate"), split=split  # type: ignore[arg-type]
    )


def _field(focus: Property, split: object, params: dict[str, float]) -> np.ndarray:
    pm = PropertyMap.from_property(STAGE).stratify(focus, where=STAGE["active"])
    model = FlowModel(pm)
    model.add_flow(_progression(split))
    y = np.linspace(1.0, 2.0, pm.size)
    return np.asarray(model.compile().vector_field(0.0, y, params))


def test_list_built_property_as_split_key() -> None:
    """The tb_acf report: a list-built focus property as the split= key."""
    from_list = Property("focus", ["low", "high"])
    p = Param("p")
    params = {"rate": 0.2, "p": 0.3}
    keyed = _field(from_list, {from_list: {"low": 1 - p, "high": p}}, params)
    by_name = _field(from_list, {"focus": {"low": 1 - p, "high": p}}, params)
    np.testing.assert_allclose(keyed, by_name, rtol=1e-12)


def test_property_and_its_name_in_one_split_raise() -> None:
    focus = Property("focus", ["low", "high"])
    split = {focus: {"low": 0.5, "high": 0.5}, "focus": {"low": 0.2, "high": 0.8}}
    with pytest.raises(ValueError, match="more than once"):
        _field(focus, split, {"rate": 0.2})
