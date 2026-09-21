"""Generalised force of infection (kind=FOIKind.GENERALISED + exponent)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from summer4 import FlowModel, Param, Property, PropertyMap, TransitionFlow
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix


class _GenParams(NamedTuple):
    contact_rate: float
    infection_pop_scale: float


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old", "elder"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _compile_foi(
    pmap: PropertyMap,
    state: Property,
    age: Property,
    *,
    kind: object,
    exponent: object | None = None,
    mixing: np.ndarray | None = None,
    contact_rate: object = 1.0,
) -> object:
    matrix = np.eye(len(age.traits)) if mixing is None else mixing
    kwargs: dict[str, object] = {
        "infectious": state["I"],
        "group_by": age,
        "kind": kind,
        "contact_rate": contact_rate,
        "mixing": MixingMatrix(age, matrix, normalize="none", check_reciprocal=False),
    }
    if exponent is not None:
        kwargs["exponent"] = exponent
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow("inf", state["S"], state["I"], ForceOfInfection("infection", **kwargs))
    )
    return model.compile()


def test_generalised_requires_exponent() -> None:
    state, age, pmap = _sir_age()
    with pytest.raises(ValueError, match="kind=FOIKind.GENERALISED requires exponent"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind=FOIKind.GENERALISED,
        )


def test_string_kind_coerces_to_enum() -> None:
    state, age, _pmap = _sir_age()
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="frequency",
    )
    assert foi.kind is FOIKind.FREQUENCY
    with pytest.raises(ValueError, match="Unknown ForceOfInfection kind"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind="not-a-kind",
        )


def test_exponent_rejected_for_other_kinds() -> None:
    state, age, _pmap = _sir_age()
    for kind in (FOIKind.FREQUENCY, FOIKind.DENSITY):
        with pytest.raises(ValueError, match="only valid with"):
            ForceOfInfection(
                "infection",
                infectious=state["I"],
                group_by=age,
                kind=kind,
                exponent=1.0,
            )

    def as_freq(infectious: object, denominator: object) -> object:
        return infectious / denominator  # type: ignore[operator]

    with pytest.raises(ValueError, match="only valid with"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind=as_freq,
            exponent=1.0,
        )


def test_frequency_bit_identical_to_generalised_exponent_one() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    y = np.array([100.0, 200.0, 50.0, 10.0, 20.0, 5.0, 0.0, 0.0, 0.0])
    params = {"contact_rate": 0.4}
    freq = _compile_foi(
        pmap, state, age, kind=FOIKind.FREQUENCY, contact_rate=Param("contact_rate")
    )
    gen = _compile_foi(
        pmap,
        state,
        age,
        kind=FOIKind.GENERALISED,
        exponent=1.0,
        contact_rate=Param("contact_rate"),
    )
    a = np.asarray(freq.observe(0.0, y, params).captures["infection"].data)
    b = np.asarray(gen.observe(0.0, y, params).captures["infection"].data)
    np.testing.assert_array_equal(a, b)


def test_density_bit_identical_to_generalised_exponent_zero() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    y = np.array([100.0, 200.0, 50.0, 10.0, 20.0, 5.0, 0.0, 0.0, 0.0])
    params = {"contact_rate": 0.4}
    dens = _compile_foi(pmap, state, age, kind=FOIKind.DENSITY, contact_rate=Param("contact_rate"))
    gen = _compile_foi(
        pmap,
        state,
        age,
        kind=FOIKind.GENERALISED,
        exponent=0.0,
        contact_rate=Param("contact_rate"),
    )
    a = np.asarray(dens.observe(0.0, y, params).captures["infection"].data)
    b = np.asarray(gen.observe(0.0, y, params).captures["infection"].data)
    np.testing.assert_array_equal(a, b)


def test_generalised_matches_hand_formula_with_mixing() -> None:
    """Parity with summer2gen: λ = contact * M @ (I / N**exp)."""
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    # Rows: dest age; columns: source age. Non-identity so mixing matters.
    mixing = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    contact = 0.35
    exp = 0.6
    # S_young, S_old, S_elder, I_young, I_old, I_elder, R_*
    y = np.array([900.0, 800.0, 700.0, 30.0, 20.0, 10.0, 0.0, 0.0, 0.0])
    i = y[3:6]
    n = y[0:3] + y[3:6] + y[6:9]
    expect = contact * (mixing @ (i / (n**exp)))

    cm = _compile_foi(
        pmap,
        state,
        age,
        kind=FOIKind.GENERALISED,
        exponent=Param("infection_pop_scale"),
        mixing=mixing,
        contact_rate=Param("contact_rate"),
    )
    got = np.asarray(
        cm.observe(0.0, y, {"contact_rate": contact, "infection_pop_scale": exp})
        .captures["infection"]
        .data
    )
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)


def test_generalised_gradient_wrt_exponent() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    state, age, pmap = _sir_age()
    mixing = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.25, 0.25, 0.5],
        ],
        dtype=np.float64,
    )
    cm = _compile_foi(
        pmap,
        state,
        age,
        kind=FOIKind.GENERALISED,
        exponent=Param("infection_pop_scale"),
        mixing=mixing,
        contact_rate=0.4,
    )
    y = jnp.asarray([900.0, 800.0, 700.0, 30.0, 20.0, 10.0, 0.0, 0.0, 0.0])

    def loss(scale: object) -> object:
        caps = cm.observe(0.0, y, {"infection_pop_scale": scale}).captures["infection"].data
        return jnp.sum(jnp.asarray(caps) ** 2)

    value, grad = jax.value_and_grad(loss)(0.7)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def test_generalised_digest_covers_exponent() -> None:
    state, age, pmap = _sir_age()
    a = _compile_foi(pmap, state, age, kind=FOIKind.GENERALISED, exponent=0.5)
    b = _compile_foi(pmap, state, age, kind=FOIKind.GENERALISED, exponent=0.5)
    c = _compile_foi(pmap, state, age, kind=FOIKind.GENERALISED, exponent=0.8)
    assert a._digest == b._digest
    assert a._digest != c._digest


def _mass(y: np.ndarray, pmap: PropertyMap, sel: object) -> float:
    return float(np.sum(y[pmap.select(sel)]))  # type: ignore[arg-type]


def _compile_weighted(
    pmap: PropertyMap,
    *,
    infectious: object,
    group_by: Property,
    kind: object = FOIKind.GENERALISED,
    exponent: object | None = 1.0,
    mixing: np.ndarray | None = None,
    contact_rate: object = 1.0,
    infectiousness: object | None = None,
    normalize: object | None = None,
    source: object | None = None,
    dest: object | None = None,
) -> object:
    n = len(group_by.traits)
    matrix = np.eye(n) if mixing is None else mixing
    kwargs: dict[str, object] = {
        "infectious": infectious,
        "group_by": group_by,
        "kind": kind,
        "contact_rate": contact_rate,
        "mixing": MixingMatrix(group_by, matrix, normalize="none", check_reciprocal=False),
    }
    if exponent is not None:
        kwargs["exponent"] = exponent
    if infectiousness is not None:
        kwargs["infectiousness"] = infectiousness
    if normalize is not None:
        kwargs["normalize_infectiousness"] = normalize
    model = FlowModel(pmap)
    src = infectious if source is None else source
    dst = infectious if dest is None else dest
    model.add_flow(
        TransitionFlow("inf", src, dst, ForceOfInfection("infection", **kwargs))  # type: ignore[arg-type]
    )
    return model.compile()


def test_trait_map_digests_equal_selector_pairs() -> None:
    """The trait mapping is sugar for ``(group_by[trait], weight)`` pairs."""
    state, age, pmap = _sir_age()

    def digest(infectiousness: object) -> bytes:
        compiled = _compile_weighted(
            pmap,
            infectious=state["I"],
            group_by=age,
            infectiousness=infectiousness,
            source=state["S"],
            dest=state["I"],
        )
        return compiled._digest  # type: ignore[attr-defined]

    trait_map = {age["young"]: 0.5, age["old"]: 2.0}
    by_name = {"young": 0.5, "old": 2.0}
    forward = [(age["young"], 0.5), (age["old"], 2.0)]
    reversed_pairs = [(age["old"], 2.0), (age["young"], 0.5)]
    assert digest(trait_map) == digest(by_name)
    assert digest(trait_map) == digest(forward)
    assert digest(forward) == digest(reversed_pairs)
    assert digest(trait_map) != digest({age["young"]: 0.5, age["old"]: 9.0})


def test_generalised_matches_hand_formula_with_weights() -> None:
    """λ = contact * M @ (w * I / N**exp), weights constant within each age."""
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    mixing = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]],
        dtype=np.float64,
    )
    contact = 0.35
    exp = 0.6
    y = np.array([900.0, 800.0, 700.0, 30.0, 20.0, 10.0, 0.0, 0.0, 0.0])
    # from_property(state).stratify(age) is not necessarily this order; sum by selector.
    weights = {"young": 0.5, "old": 2.0, "elder": 1.0}
    infectious = np.array([_mass(y, pmap, state["I"] & age[name]) for name in age.traits])
    population = np.array([_mass(y, pmap, age[name]) for name in age.traits])
    weighted = np.array([weights[name] for name in age.traits]) * infectious
    expect = contact * (mixing @ (weighted / population**exp))
    cm = _compile_weighted(
        pmap,
        infectious=state["I"],
        group_by=age,
        exponent=exp,
        mixing=mixing,
        contact_rate=contact,
        infectiousness={age[name]: weights[name] for name in age.traits},
        source=state["S"],
        dest=state["I"],
    )
    got = np.asarray(cm.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)
    via_pairs = _compile_weighted(
        pmap,
        infectious=state["I"],
        group_by=age,
        exponent=exp,
        mixing=mixing,
        contact_rate=contact,
        infectiousness=[(age[name], weights[name]) for name in age.traits],
        source=state["S"],
        dest=state["I"],
    )
    paired = np.asarray(via_pairs.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(got, paired)


def test_mixing_over_age_while_reachability_is_present() -> None:
    """A second property is summed into the age group; it is not a mixing axis."""
    pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    age = Property("age", ("child", "adult"))
    reach = Property("reach", ("local", "remote"))
    pmap = PropertyMap.from_property(state).stratify(age).stratify(reach)
    y = np.zeros(pmap.size)
    y[pmap.select(state["S"] & age["child"] & reach["local"])] = 80.0
    y[pmap.select(state["S"] & age["child"] & reach["remote"])] = 20.0
    y[pmap.select(state["S"] & age["adult"] & reach["local"])] = 50.0
    y[pmap.select(state["S"] & age["adult"] & reach["remote"])] = 50.0
    y[pmap.select(state["I"] & age["child"] & reach["local"])] = 4.0
    y[pmap.select(state["I"] & age["child"] & reach["remote"])] = 6.0
    y[pmap.select(state["I"] & age["adult"] & reach["local"])] = 1.0
    y[pmap.select(state["I"] & age["adult"] & reach["remote"])] = 3.0
    mixing = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=np.float64)
    contact = 0.3
    infectious = np.array([10.0, 4.0])
    population = np.array([110.0, 104.0])
    expect = contact * (mixing @ (infectious / population))
    cm = _compile_weighted(
        pmap,
        infectious=state["I"],
        group_by=age,
        kind=FOIKind.FREQUENCY,
        exponent=None,
        mixing=mixing,
        contact_rate=contact,
        source=state["S"],
        dest=state["I"],
    )
    got = np.asarray(cm.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)


def _tb_weights_state() -> tuple[Property, Property, PropertyMap, np.ndarray]:
    """Subclinical and clinical infectious; age 0 and 15."""
    state = Property("state", ("S", "sub", "clin"))
    age = Property("age", ("0", "15"))
    pmap = PropertyMap.from_property(state).stratify(age)
    y = np.zeros(pmap.size)
    y[pmap.select(state["S"] & age["0"])] = 100.0
    y[pmap.select(state["sub"] & age["0"])] = 10.0
    y[pmap.select(state["clin"] & age["0"])] = 8.0
    y[pmap.select(state["S"] & age["15"])] = 200.0
    y[pmap.select(state["sub"] & age["15"])] = 20.0
    y[pmap.select(state["clin"] & age["15"])] = 40.0
    return state, age, pmap, y


def test_compartment_by_age_infectiousness_zeros_children() -> None:
    """Subclinical is 0.4×; age 0 transmits nothing. Both apply before the group sum."""
    pytest.importorskip("jax")
    state, age, pmap, y = _tb_weights_state()
    contact = 0.25
    exp = 0.5
    # Age 0: every infectious compartment is multiplied by 0. Age 15: 0.4*20 + 40.
    infectious = np.array([0.0, 48.0])
    population = np.array([118.0, 260.0])
    expect = contact * (infectious / population**exp)
    pairs = [(state["sub"], 0.4), (age["0"], 0.0)]
    cm = _compile_weighted(
        pmap,
        infectious=state["sub"] | state["clin"],
        group_by=age,
        exponent=exp,
        contact_rate=contact,
        infectiousness=pairs,
        source=state["S"],
        dest=state["sub"],
    )
    got = np.asarray(cm.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
    np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)
    unweighted = _compile_weighted(
        pmap,
        infectious=state["sub"] | state["clin"],
        group_by=age,
        exponent=exp,
        contact_rate=contact,
        source=state["S"],
        dest=state["sub"],
    )
    plain = np.asarray(unweighted.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
    assert plain[0] > 0.0
    assert got[0] == 0.0
    assert got[1] < plain[1]


def test_normalize_population_and_mean_on_compartment_weights() -> None:
    """Normalisation rescales the effective per-group weight, including within a group."""
    pytest.importorskip("jax")
    state, age, pmap, y = _tb_weights_state()
    pairs = [(state["sub"], 0.4), (age["0"], 0.0)]
    contact = 0.2
    # Weighted infectious pool before normalisation: [0, 48].
    # Population-weighted mean of compartment weights:
    # age 0 is 0; age 15 is S=1, sub=0.4, clin=1 → 200 + 8 + 40 = 248 over 378.
    mean_pop = 248.0 / 378.0
    expect_pop = contact * (np.array([0.0, 48.0]) / mean_pop) / np.array([118.0, 260.0])
    # Unweighted mean of per-group effective weights: age 0 is 0,
    # age 15 is (1 + 0.4 + 1) / 3 = 0.8, mean 0.4.
    expect_mean = contact * (np.array([0.0, 48.0]) / 0.4) / np.array([118.0, 260.0])
    for normalize, expect in (("population", expect_pop), ("mean", expect_mean)):
        cm = _compile_weighted(
            pmap,
            infectious=state["sub"] | state["clin"],
            group_by=age,
            kind=FOIKind.FREQUENCY,
            exponent=None,
            contact_rate=contact,
            infectiousness=pairs,
            normalize=normalize,
            source=state["S"],
            dest=state["sub"],
        )
        got = np.asarray(cm.observe(0.0, y, {}).captures["infection"].data)  # type: ignore[attr-defined]
        np.testing.assert_allclose(got, expect, rtol=1e-5, atol=1e-6)


def test_lookup_mixing_changes_force_over_time() -> None:
    pytest.importorskip("jax")
    from summer4 import Lookup, Time, floor

    state, age, pmap = _sir_age()
    stack = np.stack(
        [
            np.eye(3),
            np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]),
        ]
    )
    y = np.zeros(pmap.size)
    y[pmap.select(state["S"] & age["young"])] = 100.0
    y[pmap.select(state["S"] & age["old"])] = 80.0
    y[pmap.select(state["S"] & age["elder"])] = 60.0
    y[pmap.select(state["I"] & age["young"])] = 10.0
    y[pmap.select(state["I"] & age["old"])] = 20.0
    y[pmap.select(state["I"] & age["elder"])] = 30.0
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind=FOIKind.FREQUENCY,
        contact_rate=0.4,
        mixing=MixingMatrix(
            age,
            Lookup(Param("mixing"), floor(Time())),
            normalize="none",
            check_reciprocal=False,
        ),
        infectiousness=[(age["young"], 0.5)],
    )
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], foi))
    compiled = model.compile()
    assert ("mixing",) in compiled.computed_paths
    at0 = np.asarray(compiled.observe(0.0, y, {"mixing": stack}).captures["infection"].data)
    at1 = np.asarray(compiled.observe(1.0, y, {"mixing": stack}).captures["infection"].data)
    assert not np.allclose(at0, at1)


def test_selector_weight_param_is_differentiable() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    state, age, pmap, y = _tb_weights_state()
    cm = _compile_weighted(
        pmap,
        infectious=state["sub"] | state["clin"],
        group_by=age,
        kind=FOIKind.DENSITY,
        exponent=None,
        contact_rate=1.0,
        infectiousness=[(state["sub"], Param("rel_sub")), (age["0"], 0.0)],
        source=state["S"],
        dest=state["sub"],
    )
    assert ("rel_sub",) in cm.computed_paths  # type: ignore[attr-defined]
    y_j = jnp.asarray(y)

    def loss(rel_sub: object) -> object:
        data = cm.observe(0.0, y_j, {"rel_sub": rel_sub}).captures["infection"].data  # type: ignore[attr-defined]
        return jnp.sum(jnp.asarray(data) ** 2)

    value, grad = jax.value_and_grad(loss)(0.4)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def test_infectiousness_selector_matching_nothing_raises() -> None:
    pytest.importorskip("jax")
    state, age, pmap = _sir_age()
    cm = _compile_weighted(
        pmap,
        infectious=state["I"],
        group_by=age,
        kind=FOIKind.FREQUENCY,
        exponent=None,
        infectiousness=[(state["S"] & state["I"], 0.2)],
        source=state["S"],
        dest=state["I"],
    )
    y = np.ones(pmap.size)
    with pytest.raises(ValueError, match="matches no compartment"):
        cm.observe(0.0, y, {})  # type: ignore[attr-defined]


def test_infectiousness_rejects_a_bare_pair_and_a_foreign_selector_key() -> None:
    state, age, _pmap = _sir_age()
    with pytest.raises(TypeError, match="wrap a single pair"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            infectiousness=(age["young"], 0.5),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="sequence of"):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            infectiousness={state["S"] & age["young"]: 0.5},  # type: ignore[dict-item]
        )
