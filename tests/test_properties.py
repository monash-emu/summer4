"""Unit tests for Property and Trait."""

from __future__ import annotations

import pytest

from summer4 import IsIn, Property, Trait
from summer4.selectors import Absent, Present


def test_property_assigns_codes_in_order() -> None:
    age = Property("age", ("0-4", "5-9", "10+"))
    assert age.trait("0-4") == Trait(property="age", name="0-4", code=0)
    assert age.trait("10+").code == 2


def test_getitem_single_and_multi() -> None:
    age = Property("age", ("0-4", "5-9", "10+"))
    assert age["0-4"] == Trait(property="age", name="0-4", code=0)
    assert age["0-4", "10+"] == IsIn(property="age", names=("0-4", "10+"))
    assert age[("0-4", "5-9")] == age.isin(("0-4", "5-9"))


def test_present_and_absent() -> None:
    age = Property("age", ("0-4", "5-9"))
    assert age.present() == Present(property="age")
    assert age.absent() == Absent(property="age")


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Property("", ("a",))


def test_non_identifier_name_raises() -> None:
    with pytest.raises(ValueError, match="identifier"):
        Property("age@dest", ("young", "old"))
    with pytest.raises(ValueError, match="identifier"):
        Property("age-band", ("young", "old"))


def test_empty_traits_raises() -> None:
    with pytest.raises(ValueError, match="at least one trait"):
        Property("age", ())


def test_duplicate_traits_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Property("age", ("young", "young"))


def test_empty_trait_name_raises() -> None:
    with pytest.raises(ValueError, match="empty trait"):
        Property("age", ("young", ""))


def test_unknown_trait_raises() -> None:
    age = Property("age", ("0-4", "5-9"))
    with pytest.raises(KeyError, match="old"):
        age["old"]
    with pytest.raises(KeyError, match="old"):
        age.isin(("0-4", "old"))


def test_empty_isin_raises() -> None:
    age = Property("age", ("0-4",))
    with pytest.raises(ValueError, match="at least one"):
        age.isin(())


def test_equal_properties_are_equal() -> None:
    a = Property("age", ("0-4", "5-9"))
    b = Property("age", ("0-4", "5-9"))
    assert a == b
    assert hash(a.trait("0-4")) == hash(b.trait("0-4"))
