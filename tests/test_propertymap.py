"""Unit tests for PropertyMap and Stratification."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from summer4 import Property, PropertyMap, Stratification


def _sir() -> tuple[Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    return state, PropertyMap.from_property(state)


def test_from_property_one_row_per_trait() -> None:
    state, pm = _sir()
    assert pm.size == 3
    assert pm.n_properties == 1
    assert pm.select(state["S"]).tolist() == [0]
    assert pm.select(state["I"]).tolist() == [1]
    assert pm.select(state["R"]).tolist() == [2]
    assert pm.labels() == ("state=S", "state=I", "state=R")


def test_full_stratification_is_cartesian() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    pm = pm.stratify(age)
    assert pm.size == 6
    assert pm.select(state["S"]).tolist() == [0, 1]
    assert pm.select(age["child"]).tolist() == [0, 2, 4]
    np.testing.assert_array_equal(pm.parent_row, np.array([0, 0, 1, 1, 2, 2], dtype=np.int32))


def test_partial_stratification_is_ragged() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    sev = Property("severity", ("mild", "severe"))
    pm = pm.stratify(age).stratify(sev, where=state["I"])
    assert pm.size == 8  # S×2 + I×2×2 + R×2
    assert pm.select(sev.present()).tolist() == pm.select(state["I"]).tolist()
    assert pm.select(sev.absent()).tolist() == pm.select(state["S"] | state["R"]).tolist()


def test_stratification_expands_in_place() -> None:
    """New strata of a matched compartment are contiguous."""
    state, pm = _sir()
    sev = Property("severity", ("mild", "severe"))
    pm = pm.stratify(sev, where=state["I"])
    infected = pm.select(state["I"])
    assert infected.tolist() == [1, 2]
    assert np.array_equal(infected, np.arange(infected[0], infected[-1] + 1))
    assert pm.select(state["S"]).tolist() == [0]
    assert pm.select(state["R"]).tolist() == [3]


def test_stratification_record_apply() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    out = Stratification(age).apply(pm)
    assert out == pm.stratify(age)
    assert len(out.history) == 1
    assert out.history[0].property == age
    assert out.history[0].where is None


def test_duplicate_property_raises() -> None:
    state, pm = _sir()
    with pytest.raises(ValueError, match="already on this map"):
        pm.stratify(state)


def test_unknown_property_in_query_raises() -> None:
    _state, pm = _sir()
    age = Property("age", ("child", "adult"))
    with pytest.raises(KeyError, match="age"):
        pm.select(age["child"])


def test_unknown_trait_in_isin_raises() -> None:
    state, pm = _sir()
    from summer4.selectors import IsIn

    with pytest.raises(KeyError, match="X"):
        pm.select(IsIn(property="state", names=("X",)))


def test_select_one() -> None:
    state, pm = _sir()
    assert pm.select_one(state["I"]) == 1
    with pytest.raises(ValueError, match="exactly one"):
        pm.select_one(state["S"] | state["I"])


def test_three_valued_ragged_partition() -> None:
    """mild, ~mild, and absent(severity) partition the map."""
    state = Property("state", ("S", "I", "R"))
    sev = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(sev, where=state["I"])
    mild = set(pm.select(sev["mild"]).tolist())
    not_mild = set(pm.select(~sev["mild"]).tolist())
    absent = set(pm.select(sev.absent()).tolist())
    assert mild.isdisjoint(not_mild)
    assert mild.isdisjoint(absent)
    assert not_mild.isdisjoint(absent)
    assert mild | not_mild | absent == set(range(pm.size))
    assert mild == set(pm.select(state["I"] & sev["mild"]).tolist())
    assert not_mild == set(pm.select(sev["severe"]).tolist())
    assert absent == set(pm.select(state["S"] | state["R"]).tolist())
    assert set(pm.select(sev["mild"]).tolist()).isdisjoint(pm.select(~sev["mild"]).tolist())
    assert 0 not in mild and 0 not in not_mild


def test_present_is_two_valued() -> None:
    state = Property("state", ("S", "I"))
    sev = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(sev, where=state["I"])
    assert set(pm.select(sev.present()).tolist()) | set(pm.select(sev.absent()).tolist()) == set(
        range(pm.size)
    )
    assert pm.select(~sev.present()).tolist() == pm.select(sev.absent()).tolist()


def test_partition_covers_present() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    pm = pm.stratify(age)
    parts = pm.partition(age)
    stacked = np.concatenate(list(parts.values()))
    assert len(stacked) == len(np.unique(stacked))
    np.testing.assert_array_equal(np.sort(stacked), pm.select(age.present()))


def test_group_by_combinations() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    pm = pm.stratify(age)
    groups = pm.group_by(state, age)
    assert len(groups) == 6
    seen: set[int] = set()
    for _key, idx in groups.items():
        seen.update(int(i) for i in idx)
    assert seen == set(range(pm.size))


def test_group_by_skips_absent() -> None:
    state, pm = _sir()
    sev = Property("severity", ("mild", "severe"))
    pm = pm.stratify(sev, where=state["I"])
    groups = pm.group_by(sev)
    assert len(groups) == 2
    covered = {int(i) for _key, idx in groups.items() for i in idx}
    assert covered == set(pm.select(sev.present()).tolist())


def test_group_by_empty_raises() -> None:
    _state, pm = _sir()
    with pytest.raises(ValueError, match="at least one"):
        pm.group_by()


def test_to_dicts_omits_absent() -> None:
    state, pm = _sir()
    sev = Property("severity", ("mild", "severe"))
    pm = pm.stratify(sev, where=state["I"])
    rows = pm.to_dicts()
    assert rows[0] == {"state": "S"}
    assert rows[1] == {"state": "I", "severity": "mild"}
    assert "severity" not in rows[0]


def test_mask_matches_select() -> None:
    state, pm = _sir()
    mask = pm.mask(state["I"] | state["R"])
    assert np.flatnonzero(mask).tolist() == pm.select(state["I"] | state["R"]).tolist()


def test_query_cache_reuses_result() -> None:
    state, pm = _sir()
    sel = state["I"]
    first = pm.select(sel)
    second = pm.select(sel)
    assert first is not second
    assert np.array_equal(first, second)
    assert sel in pm._cache


def test_copy_has_empty_cache() -> None:
    state, pm = _sir()
    pm.select(state["I"])
    clone = pm.copy()
    assert clone == pm
    assert clone._cache == {}


def test_repr_contains_size_and_labels() -> None:
    _state, pm = _sir()
    text = repr(pm)
    assert "n=3" in text
    assert "state=S" in text


def test_mismatched_property_traits_raise() -> None:
    _state, pm = _sir()
    other = Property("state", ("S", "I"))
    with pytest.raises(ValueError, match="registered with traits"):
        pm.partition(other)


def test_taxonomy_modules_do_not_import_jax() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "summer4"
    for path in src.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("jax") for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("jax")


def test_len_matches_size() -> None:
    _state, pm = _sir()
    assert len(pm) == pm.size == 3


def test_from_properties_is_fully_crossed() -> None:
    state = Property("state", ("S", "I"))
    age = Property("age", ("child", "adult"))
    pm = PropertyMap.from_properties([state, age])
    assert pm == PropertyMap.from_property(state).stratify(age)
    assert pm.size == 4


def test_from_properties_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PropertyMap.from_properties([])


def test_public_column_accessors() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    pm = pm.stratify(age)
    assert pm.column_index(age) == 1
    assert pm.column_index("age") == 1
    np.testing.assert_array_equal(pm.column(age), pm.codes[:, 1])
    assert pm.label(0) == "state=S_age=child"


def test_to_frame_uses_null_for_absent() -> None:
    state, pm = _sir()
    sev = Property("severity", ("mild", "severe"))
    pm = pm.stratify(sev, where=state["I"])
    frame = pm.to_frame()
    assert frame.columns == ["state", "severity"]
    assert frame.height == pm.size
    assert frame["severity"][0] is None
    assert frame["severity"][1] == "mild"


def test_equal_maps_share_hash() -> None:
    state, pm = _sir()
    age = Property("age", ("child", "adult"))
    a = pm.stratify(age)
    b = pm.stratify(age)
    assert a == b
    assert a is not b
    assert hash(a) == hash(b)
    assert {a: "ok"}[b] == "ok"


def test_rebuilt_map_hits_jit_cache() -> None:
    from functools import partial

    import jax
    import jax.numpy as jnp

    state = Property("state", ("S", "I", "R"))
    a = PropertyMap.from_property(state)
    b = PropertyMap.from_property(state)
    c = a.stratify(Property("age", ("child", "adult")))
    assert a == b and a is not b

    @partial(jax.jit, static_argnums=0)
    def scale(pmap: PropertyMap, x: jnp.ndarray) -> jnp.ndarray:
        return x * pmap.size

    x = jnp.asarray(1.0)
    scale(a, x)
    size_after_first = scale._cache_size()
    scale(b, x)
    assert scale._cache_size() == size_after_first
    scale(c, x)
    assert scale._cache_size() == size_after_first + 1


def test_source_dest_rejected_on_compartment_map() -> None:
    from summer4 import Dest, Source

    state, pm = _sir()
    with pytest.raises(TypeError, match="flow edges"):
        pm.select(Source(state["I"]))
    with pytest.raises(TypeError, match="flow edges"):
        pm.mask(Dest(state["R"]))
    with pytest.raises(TypeError, match="flow edges"):
        pm.select(Source(state["S"]) & Dest(state["I"]))
