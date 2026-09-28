"""ContactMatrix and SettingStack: construction, validation, inspection (WP9)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.helpers import contacts as fx

from summer4 import Property
from summer4.epi import ContactMatrix, Reciprocity, SettingStack, band_label


def _long(values: np.ndarray, labels: list[Any]) -> pd.DataFrame:
    k = len(labels)
    return pd.DataFrame(
        {
            "age_from": [labels[i] for i in range(k) for _ in range(k)],
            "age_to": [labels[j] for _ in range(k) for j in range(k)],
            "contacts": values.reshape(-1),
        }
    )


# -- construction -----------------------------------------------------------


def test_from_array_keeps_orientation_and_is_read_only() -> None:
    cm = fx.survey()
    np.testing.assert_array_equal(cm.values, fx.VALUES)
    assert cm.bands == fx.BANDS
    assert cm.size == 3
    assert cm.labels == ("[0,5)", "[5,15)", "[15,inf)")
    with pytest.raises(ValueError):
        cm.values[0, 0] = 9.0
    source = np.array(fx.VALUES)
    built = ContactMatrix.from_array(source, fx.BANDS)
    source[0, 0] = 99.0
    assert built.values[0, 0] == 2.0


@pytest.mark.parametrize(
    "labels",
    [
        ["[0,5)", "[5,15)", "[15,inf)"],
        ["0-4", "5-14", "15+"],
        [0, 5, 15],
        ["0", "5", "15"],
    ],
)
def test_from_frame_parses_band_labels(labels: list[Any]) -> None:
    frame = _long(fx.VALUES, labels)
    cm = ContactMatrix.from_frame(frame)
    assert cm.bands == fx.BANDS
    np.testing.assert_array_equal(cm.values, fx.VALUES)


def test_from_frame_orders_bands_by_lower_bound_not_by_row_order() -> None:
    frame = _long(fx.VALUES, ["[0,5)", "[5,15)", "[15,inf)"]).iloc[::-1]
    cm = ContactMatrix.from_frame(frame)
    np.testing.assert_array_equal(cm.values, fx.VALUES)


def test_from_frame_accepts_polars_and_custom_column_names() -> None:
    pl = pytest.importorskip("polars")
    frame = _long(fx.VALUES, ["0-4", "5-14", "15+"]).rename(
        columns={"age_from": "participant", "age_to": "contact", "contacts": "mean"}
    )
    cm = ContactMatrix.from_frame(
        pl.from_pandas(frame), row="participant", col="contact", value="mean"
    )
    np.testing.assert_array_equal(cm.values, fx.VALUES)


def test_from_frame_with_bands_and_unparseable_labels_uses_first_appearance() -> None:
    frame = _long(fx.VALUES, ["infant", "child", "adult"])
    cm = ContactMatrix.from_frame(frame, bands=fx.BANDS)
    np.testing.assert_array_equal(cm.values, fx.VALUES)


def test_from_frame_numeric_labels_with_explicit_closed_last_band() -> None:
    cm = ContactMatrix.from_frame(_long(fx.VALUES, [0, 5, 15]), bands=(0, 5, 15, 80))
    assert cm.bands == (0.0, 5.0, 15.0, 80.0)


def test_round_trip_through_a_frame() -> None:
    cm = fx.survey()
    back = ContactMatrix.from_frame(cm.to_frame(), source_population=fx.POPULATION)
    assert back == cm
    assert hash(back) == hash(cm)


def test_to_pandas_is_participant_by_contact() -> None:
    frame = fx.survey().to_pandas()
    assert frame.index.name == "participant"
    assert frame.columns.name == "contact"
    assert frame.loc["[5,15)", "[15,inf)"] == 1.0
    assert frame.loc["[15,inf)", "[5,15)"] == 1.0
    assert frame.loc["[0,5)", "[15,inf)"] == 0.5
    assert frame.loc["[15,inf)", "[0,5)"] == 0.25


def test_band_label_formats_edges() -> None:
    assert band_label(0.0, 5.0) == "[0,5)"
    assert band_label(2.5, math.inf) == "[2.5,inf)"


# -- validation messages -----------------------------------------------------


def test_not_square_names_the_shape() -> None:
    with pytest.raises(ValueError, match=r"square; got shape \(2, 3\)"):
        ContactMatrix.from_array(np.ones((2, 3)), (0, 1, 2))


def test_band_count_names_both_counts() -> None:
    with pytest.raises(ValueError, match=r"4 band edges for a 2 x 2 matrix; expected 3"):
        ContactMatrix.from_array(np.ones((2, 2)), (0, 1, 2, 3))


def test_bands_not_increasing_names_the_first_bad_pair() -> None:
    with pytest.raises(ValueError, match=r"bands\[1\] = 5 then bands\[2\] = 5"):
        ContactMatrix.from_array(np.ones((3, 3)), (0, 5, 5, 10))


def test_only_the_last_edge_may_be_inf() -> None:
    with pytest.raises(ValueError, match=r"bands\[1\] = inf"):
        ContactMatrix.from_array(np.ones((2, 2)), (0, math.inf, math.inf))


@pytest.mark.parametrize("bad", [-0.5, math.nan, math.inf])
def test_bad_value_names_index_and_value(bad: float) -> None:
    values = np.array(fx.VALUES)
    values[1, 2] = bad
    with pytest.raises(ValueError, match=rf"values\[1, 2\] = {bad}"):
        ContactMatrix.from_array(values, fx.BANDS)


def test_prop_trait_count_names_both_counts_and_the_property() -> None:
    age = Property("age", ("young", "old"))
    with pytest.raises(ValueError, match=r"'age' has 2 traits but the matrix has 3 bands"):
        ContactMatrix.from_array(fx.VALUES, fx.BANDS, prop=age)


def test_prop_alignment_is_positional_for_non_numeric_traits() -> None:
    age = Property("age", ("infant", "child", "adult"))
    assert ContactMatrix.from_array(fx.VALUES, fx.BANDS, prop=age).prop == age


def test_numeric_traits_must_be_the_lower_edges() -> None:
    good = Property("age", ("0", "5", "15"))
    assert ContactMatrix.from_array(fx.VALUES, fx.BANDS, prop=good).prop == good
    bad = Property("age", ("0", "10", "15"))
    with pytest.raises(
        ValueError, match=r"trait '10' .* lower bound 10, but that band starts at 5"
    ):
        ContactMatrix.from_array(fx.VALUES, fx.BANDS, prop=bad)


def test_source_population_length_and_sign_name_the_problem() -> None:
    with pytest.raises(ValueError, match=r"shape \(2,\); expected \(3,\)"):
        ContactMatrix.from_array(fx.VALUES, fx.BANDS, source_population=[1.0, 2.0])
    with pytest.raises(ValueError, match=r"source_population\[1\] = 0.0"):
        ContactMatrix.from_array(fx.VALUES, fx.BANDS, source_population=[1.0, 0.0, 2.0])


def test_unparseable_label_is_named() -> None:
    with pytest.raises(ValueError, match=r"Cannot parse contact-matrix band label 'teens'"):
        ContactMatrix.from_frame(_long(fx.VALUES, ["0-4", "teens", "15+"]))


def test_non_contiguous_labels_are_named() -> None:
    with pytest.raises(ValueError, match=r"'0-4' ends at 5 but the next band '6-14' starts at 6"):
        ContactMatrix.from_frame(_long(fx.VALUES, ["0-4", "6-14", "15+"]))


def test_missing_and_duplicate_pairs_are_named() -> None:
    frame = _long(fx.VALUES, ["0-4", "5-14", "15+"])
    with pytest.raises(ValueError, match=r"no value for the pair \('0-4', '15\+'\)"):
        ContactMatrix.from_frame(frame.drop(index=2))
    doubled = pd.concat([frame, frame.iloc[[4]]])
    with pytest.raises(ValueError, match=r"pair \('5-14', '5-14'\) more than once"):
        ContactMatrix.from_frame(doubled)


def test_missing_frame_column_is_named() -> None:
    with pytest.raises(ValueError, match=r"missing columns \['n'\]"):
        ContactMatrix.from_frame(_long(fx.VALUES, [0, 5, 15]), value="n")


def test_bands_that_disagree_with_labels_are_named() -> None:
    with pytest.raises(ValueError, match=r"'5-14' starts at 5, but bands says .* starts at 10"):
        ContactMatrix.from_frame(_long(fx.VALUES, ["0-4", "5-14", "15+"]), bands=(0, 10, 15, 99))


# -- inspection --------------------------------------------------------------


def test_total_is_the_row_sum_not_the_column_sum() -> None:
    cm = fx.survey()
    np.testing.assert_allclose(cm.total(), fx.TOTAL)
    assert not np.allclose(cm.total(), fx.VALUES.sum(axis=0))


def test_mean_contacts_is_population_weighted() -> None:
    cm = fx.survey()
    assert cm.mean_contacts() == pytest.approx(fx.MEAN_CONTACTS)
    flat = cm.mean_contacts([1.0, 1.0, 1.0])
    assert flat == pytest.approx(fx.TOTAL.mean())


def test_reciprocity_error_matches_hand_computation() -> None:
    err = fx.survey().reciprocity_error()
    assert isinstance(err, Reciprocity)
    np.testing.assert_allclose(err.matrix, fx.RECIPROCITY)
    flows = fx.VALUES * fx.POPULATION[:, None]
    expected = np.linalg.norm(fx.RECIPROCITY) / np.linalg.norm(flows)
    assert err.relative == pytest.approx(expected)
    transposed = ContactMatrix.from_array(fx.VALUES.T, fx.BANDS)
    np.testing.assert_allclose(
        transposed.reciprocity_error(fx.POPULATION).matrix,
        (fx.VALUES.T * fx.POPULATION[:, None]) - (fx.VALUES.T * fx.POPULATION[:, None]).T,
    )
    assert not np.allclose(transposed.reciprocity_error(fx.POPULATION).matrix, fx.RECIPROCITY)


def test_reciprocal_matrix_has_zero_error() -> None:
    n = fx.POPULATION
    flows = np.array([[10.0, 20.0, 30.0], [20.0, 40.0, 50.0], [30.0, 50.0, 60.0]])
    cm = ContactMatrix.from_array(flows / n[:, None], fx.BANDS)
    assert cm.reciprocity_error(n).relative == pytest.approx(0.0, abs=1e-15)


def test_population_is_required_without_source_population() -> None:
    cm = ContactMatrix.from_array(fx.VALUES, fx.BANDS)
    with pytest.raises(ValueError, match="no source_population"):
        cm.mean_contacts()
    with pytest.raises(ValueError, match=r"population\[0\] = -1.0"):
        cm.reciprocity_error([-1.0, 1.0, 1.0])


# -- setting stacks ----------------------------------------------------------


def test_stack_total_is_the_elementwise_sum() -> None:
    st = fx.stack()
    assert list(st) == ["home", "school", "work"]
    assert len(st) == 3
    total = st.total()
    np.testing.assert_allclose(total.values, fx.HOME + fx.SCHOOL + fx.WORK)
    np.testing.assert_allclose(total.values, fx.VALUES)
    assert total.setting is None
    np.testing.assert_array_equal(total.source_population, fx.POPULATION)


def test_stack_items_carry_their_setting_name() -> None:
    st = fx.stack()
    assert st["school"].setting == "school"
    np.testing.assert_array_equal(st["school"].values, fx.SCHOOL)
    with pytest.raises(KeyError, match="Unknown setting 'pub'"):
        st["pub"]


def test_stack_requires_shared_bands_and_consistent_names() -> None:
    other = ContactMatrix.from_array(np.ones((2, 2)), (0, 5, 10))
    with pytest.raises(ValueError, match="must share bands; 'b'"):
        SettingStack({"a": fx.survey(), "b": other})
    named = ContactMatrix.from_array(fx.VALUES, fx.BANDS, setting="home")
    with pytest.raises(ValueError, match="key 'work' holds a matrix whose setting is 'home'"):
        SettingStack({"work": named})
    with pytest.raises(ValueError, match="at least one setting"):
        SettingStack({})


def test_stack_from_long_frame_with_setting_column() -> None:
    frames = []
    for name, values in (("home", fx.HOME), ("school", fx.SCHOOL), ("work", fx.WORK)):
        frame = _long(values, ["0-4", "5-14", "15+"])
        frame["setting"] = name
        frames.append(frame)
    st = SettingStack.from_frame(pd.concat(frames), source_population=fx.POPULATION)
    assert st == fx.stack()


def test_stack_map_applies_to_every_setting() -> None:
    doubled = fx.stack().map(
        lambda cm: ContactMatrix.from_array(
            cm.values * 2, cm.bands, setting=cm.setting, source_population=cm.source_population
        )
    )
    np.testing.assert_allclose(doubled.total().values, 2 * fx.VALUES)
    assert doubled["work"].setting == "work"
