"""ContactMatrix and SettingStack: load, validate, inspect, adapt, scale (WP9)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.helpers import contacts as fx

from summer4 import (
    Compartments,
    FlowModel,
    Param,
    Property,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Time,
    TransitionFlow,
)
from summer4.epi import (
    ContactAdaptation,
    ContactMatrix,
    ForceOfInfection,
    MixingMatrix,
    MixingNormalize,
    Rebin,
    Reciprocity,
    ScaledContacts,
    SettingStack,
    band_label,
)
from summer4.flows.rates import ArrayConst, BinOp
from summer4.timevarying import step


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


# -- rebin (step 18) ---------------------------------------------------------


def test_rebin_aggregation_matches_hand_computation() -> None:
    cm = fx.survey().rebin((0.0, 15.0, math.inf), population=fx.POPULATION)
    # rows: population-weighted mean of participant bands 0 (N=100) and 1 (N=200)
    # cols: plain sum of contact bands 0 and 1
    expected = np.array(
        [
            [(100 * (2.0 + 1.0) + 200 * (0.5 + 3.0)) / 300, (100 * 0.5 + 200 * 1.0) / 300],
            [0.25 + 1.0, 1.5],
        ]
    )
    np.testing.assert_allclose(cm.values, expected)
    assert cm.bands == (0.0, 15.0, math.inf)
    np.testing.assert_allclose(cm.source_population, [300.0, 300.0])


def test_rebin_split_matches_hand_computation() -> None:
    coarse = ContactMatrix.from_array([[4.0, 1.0], [2.0, 3.0]], (0.0, 10.0, math.inf))
    fine_pop = np.array([100.0, 300.0, 500.0])
    cm = coarse.rebin((0.0, 4.0, 10.0, math.inf), population=fine_pop)
    # rows copy the parent band; contacts with a half are split by its population share
    expected = np.array(
        [
            [4.0 * 0.25, 4.0 * 0.75, 1.0],
            [4.0 * 0.25, 4.0 * 0.75, 1.0],
            [2.0 * 0.25, 2.0 * 0.75, 3.0],
        ]
    )
    np.testing.assert_allclose(cm.values, expected)


def test_rebin_mixed_uses_the_common_refinement() -> None:
    cm = ContactMatrix.from_array([[4.0, 1.0], [2.0, 3.0]], (0.0, 10.0, 20.0))
    plan = Rebin.between(cm.bands, (0.0, 5.0, 20.0), population=[100.0, 100.0, 200.0])
    # refinement [0,5) [5,10) [10,20); new band [5,20) = refinement 1 (N=100) + 2 (N=200)
    split = np.array([[4 * 0.5, 4 * 0.5, 1.0], [4 * 0.5, 4 * 0.5, 1.0], [2 * 0.5, 2 * 0.5, 3.0]])
    merge_rows = np.array([[1.0, 0.0, 0.0], [0.0, 1 / 3, 2 / 3]])
    merge_cols = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    np.testing.assert_allclose(plan.apply(cm.values), merge_rows @ split @ merge_cols.T)


def test_rebin_conserves_total_contacts() -> None:
    cm = fx.survey()
    coarse = cm.rebin((0.0, 15.0, math.inf), population=fx.POPULATION)
    assert coarse.mean_contacts() == pytest.approx(cm.mean_contacts())


def test_rebin_onto_a_property_aligns_to_it() -> None:
    age = Property("age", ("0", "15"))
    cm = fx.survey().rebin(age, population=fx.POPULATION)
    assert cm.prop == age
    assert cm.bands == (0.0, 15.0, math.inf)
    with pytest.raises(ValueError, match="numeric lower-bound trait names"):
        fx.survey().rebin(Property("age", ("young", "old")), population=fx.POPULATION)


def test_rebin_population_bands_spread_uniformly() -> None:
    coarse = ContactMatrix.from_array([[4.0, 1.0], [2.0, 3.0]], (0.0, 10.0, math.inf))
    direct = coarse.rebin((0.0, 4.0, 10.0, math.inf), population=[100.0, 150.0, 500.0])
    spread = coarse.rebin(
        (0.0, 4.0, 10.0, math.inf),
        population=[250.0, 500.0],
        population_bands=(0.0, 10.0, math.inf),
    )
    np.testing.assert_allclose(spread.values, direct.values)
    with pytest.raises(ValueError, match=r"Cannot split the open population band \[10,inf\)"):
        coarse.rebin(
            (0.0, 10.0, 20.0, math.inf),
            population=[250.0, 500.0],
            population_bands=(0.0, 10.0, math.inf),
        )


def test_rebin_requires_population_on_the_refinement_and_matching_span() -> None:
    with pytest.raises(ValueError, match=r"common refinement \['\[0,5\)', '\[5,15\)'"):
        fx.survey().rebin((0.0, 15.0, math.inf), population=[1.0, 2.0])
    with pytest.raises(ValueError, match="must cover the same ages"):
        fx.survey().rebin((0.0, 15.0, 80.0), population=fx.POPULATION)


def test_rebin_apply_runs_under_jit() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    plan = Rebin.between(fx.BANDS, (0.0, 15.0, math.inf), population=fx.POPULATION)
    traced = jax.jit(plan.apply)(jnp.asarray(fx.VALUES))
    np.testing.assert_allclose(np.asarray(traced), plan.apply(fx.VALUES), rtol=1e-6)


# -- symmetrise and adapt ----------------------------------------------------


def test_symmetrise_zeroes_reciprocity_error_and_conserves_contacts() -> None:
    cm = fx.survey()
    sym = cm.symmetrise()
    assert sym.reciprocity_error().relative == pytest.approx(0.0, abs=1e-15)
    assert sym.mean_contacts() == pytest.approx(cm.mean_contacts())
    # hand: c'_02 = (0.5 * 100 + 0.25 * 300) / (2 * 100)
    assert sym.values[0, 2] == pytest.approx((50.0 + 75.0) / 200.0)
    assert sym.values[2, 0] == pytest.approx((50.0 + 75.0) / 600.0)


def test_adapt_density_matches_hand_computation_and_keeps_reciprocity() -> None:
    target = np.array([300.0, 200.0, 100.0])
    ratio = (target / 600.0) / (fx.POPULATION / 600.0)  # [3, 1, 1/3]
    cm = fx.survey().symmetrise()
    out = cm.adapt(target, kind="density")
    np.testing.assert_allclose(out.values, cm.values * ratio[None, :])
    np.testing.assert_array_equal(out.source_population, target)
    assert out.reciprocity_error().relative == pytest.approx(0.0, abs=1e-14)


def test_adapt_frequency_keeps_contacts_per_person() -> None:
    target = np.array([300.0, 200.0, 100.0])
    ratio = np.array([3.0, 1.0, 1.0 / 3.0])
    cm = fx.survey()
    out = cm.adapt(target)
    shifted = fx.VALUES * ratio[None, :]
    expected = shifted * (fx.TOTAL / shifted.sum(axis=1))[:, None]
    np.testing.assert_allclose(out.values, expected)
    np.testing.assert_allclose(out.total(), fx.TOTAL)
    assert not np.allclose(out.values, cm.adapt(target, kind="density").values)


def test_adapt_to_the_same_shares_is_the_identity() -> None:
    cm = fx.survey()
    for kind in ContactAdaptation:
        np.testing.assert_allclose(cm.adapt(10 * fx.POPULATION, kind=kind).values, fx.VALUES)
    with pytest.raises(ValueError, match="Unknown ContactMatrix.adapt kind 'mass'"):
        cm.adapt(fx.POPULATION, kind="mass")


def test_stack_transforms_apply_per_setting() -> None:
    st = fx.stack()
    sym = st.symmetrise()
    for name in st:
        np.testing.assert_allclose(sym[name].values, st[name].symmetrise().values)
    np.testing.assert_allclose(sym.total().values, fx.survey().symmetrise().values)
    coarse = st.rebin((0.0, 15.0, math.inf), population=fx.POPULATION)
    np.testing.assert_allclose(
        coarse.total().values,
        fx.survey().rebin((0.0, 15.0, math.inf), population=fx.POPULATION).values,
    )
    moved = st.adapt([300.0, 200.0, 100.0])
    np.testing.assert_allclose(moved.total().total(), fx.TOTAL)


# -- scale and to_mixing -----------------------------------------------------


def test_scale_by_a_number() -> None:
    np.testing.assert_allclose(fx.survey().scale(0.5).values, 0.5 * fx.VALUES)
    st = fx.stack().scale(0.0, settings=["school"])
    assert isinstance(st, SettingStack)
    np.testing.assert_allclose(st.total().values, fx.HOME + fx.WORK)
    with pytest.raises(KeyError, match=r"Unknown settings \['pub'\]"):
        fx.stack().scale(0.5, settings=["pub"])


def test_scale_by_an_expression_folds_constant_settings() -> None:
    scaled = fx.stack().scale(Param("school_scale"), settings=["school"])
    assert isinstance(scaled, ScaledContacts)
    expr = scaled.matrix()
    assert isinstance(expr, BinOp) and expr.op == "add"
    assert isinstance(expr.left, ArrayConst)
    np.testing.assert_allclose(expr.left.value, fx.HOME + fx.WORK)
    again = scaled.scale(0.5, settings=["home"])
    np.testing.assert_allclose(again.matrix().left.value, 0.5 * fx.HOME + fx.WORK)


def test_to_mixing_defaults_to_no_row_normalisation() -> None:
    age = Property("age", ("0", "5", "15"))
    mixing = fx.survey().to_mixing(age)
    assert mixing.normalize is MixingNormalize.NONE
    np.testing.assert_array_equal(mixing.matrix.value, fx.VALUES)
    with pytest.raises(ValueError, match="needs the age property"):
        fx.survey().to_mixing()
    with pytest.raises(ValueError, match="has 2 traits but the matrix has 3 bands"):
        fx.survey().to_mixing(Property("age", ("a", "b")))


def _sir(age: Property) -> tuple[Property, Any]:
    state = Property("state", ("S", "I", "R"))
    return state, PropertyMap.from_property(state).stratify(age)


def _y0(pmap: Any, state: Property, age: Property, pop: np.ndarray) -> np.ndarray:
    y0 = np.zeros(pmap.size)
    for trait, n in zip(age.traits, pop, strict=True):
        y0[pmap.select(state["S"] & age[trait])] = 0.99 * n
        y0[pmap.select(state["I"] & age[trait])] = 0.01 * n
    return y0


def test_to_mixing_drives_a_force_of_infection_like_a_hand_written_matrix() -> None:
    pytest.importorskip("jax")
    age = Property("age", ("0", "5", "15"))
    state, pmap = _sir(age)
    y0 = _y0(pmap, state, age, fx.POPULATION)

    def foi(mixing: MixingMatrix) -> np.ndarray:
        model = FlowModel(pmap)
        rate = ForceOfInfection(
            "infection", infectious=state["I"], group_by=age, mixing=mixing, contact_rate=0.1
        )
        model.add_flow(TransitionFlow("infection", state["S"], state["I"], rate))
        return np.asarray(model.compile().observe(0.0, y0, {}).captures["infection"].data)

    ours = foi(fx.survey().to_mixing(age))
    hand = foi(MixingMatrix(age, fx.VALUES, normalize="none"))
    np.testing.assert_array_equal(ours, hand)
    # 1% of every band is infectious: lambda_i = 0.1 * sum_j c_ij * 0.01
    np.testing.assert_allclose(ours, 0.1 * 0.01 * fx.TOTAL, rtol=1e-6)


def test_time_varying_school_closure_inside_the_model() -> None:
    pytest.importorskip("jax")
    age = Property("age", ("0", "5", "15"))
    state, pmap = _sir(age)
    y0 = _y0(pmap, state, age, fx.POPULATION)
    closure = step(Time(), (10.0,), (1.0, 0.2))
    mixing = fx.stack().scale(closure, settings=["school"]).to_mixing(age)
    model = FlowModel(pmap)
    rate = ForceOfInfection(
        "infection", infectious=state["I"], group_by=age, mixing=mixing, contact_rate=0.1
    )
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], rate))
    cm = model.compile()
    before = np.asarray(cm.observe(5.0, y0, {}).captures["infection"].data)
    after = np.asarray(cm.observe(15.0, y0, {}).captures["infection"].data)
    np.testing.assert_allclose(before, 0.001 * fx.TOTAL, rtol=1e-6)
    closed = fx.HOME + fx.WORK + 0.2 * fx.SCHOOL
    np.testing.assert_allclose(after, 0.001 * closed.sum(axis=1), rtol=1e-6)


def test_school_scale_is_calibratable_through_a_jitted_run() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    age = Property("age", ("0", "5", "15"))
    state, pmap = _sir(age)
    y0 = _y0(pmap, state, age, fx.POPULATION)
    mixing = fx.stack().scale(Param("school"), settings=["school"]).to_mixing(age)
    model = FlowModel(pmap)
    rate = ForceOfInfection(
        "infection", infectious=state["I"], group_by=age, mixing=mixing, contact_rate=0.05
    )
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], rate))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    cm = model.compile()
    plan = SavePlan(requests={"R": SaveRequest(Compartments(where=state["R"]))})

    @jax.jit
    def recovered(school: Any) -> Any:
        res = cm.run({"school": school}, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
        return jnp.sum(res["R"].values.data[-1])

    open_, closed = float(recovered(1.0)), float(recovered(0.0))
    assert open_ > closed
    assert float(jax.grad(recovered)(0.5)) > 0.0


def _vector_field_eqns(k: int) -> int:
    import jax
    import jax.numpy as jnp

    edges = tuple(float(5 * i) for i in range(k)) + (math.inf,)
    age = Property("age", tuple(str(5 * i) for i in range(k)))
    rng = np.random.default_rng(k)
    home = ContactMatrix.from_array(rng.uniform(0.1, 2.0, (k, k)), edges)
    school = ContactMatrix.from_array(rng.uniform(0.1, 2.0, (k, k)), edges)
    stack = SettingStack({"home": home, "school": school})
    mixing = stack.scale(step(Time(), (10.0,), (1.0, 0.2)), settings=["school"]).to_mixing(age)
    state, pmap = _sir(age)
    model = FlowModel(pmap)
    rate = ForceOfInfection(
        "infection", infectious=state["I"], group_by=age, mixing=mixing, contact_rate=0.1
    )
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], rate))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    cm = model.compile()
    y = jnp.asarray(_y0(pmap, state, age, np.full(k, 100.0)))
    return len(jax.make_jaxpr(lambda t: jnp.asarray(cm.vector_field(t, y, {})))(0.0).jaxpr.eqns)


def test_jaxpr_size_does_not_grow_with_the_number_of_bands() -> None:
    pytest.importorskip("jax")
    sizes = {k: _vector_field_eqns(k) for k in (3, 6, 15)}
    assert len(set(sizes.values())) == 1, sizes
