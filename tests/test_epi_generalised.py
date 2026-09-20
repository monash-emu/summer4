"""Generalised force of infection (kind='generalised' + exponent)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from summer4 import FlowModel, Param, Property, PropertyMap, TransitionFlow
from summer4.epi import ForceOfInfection, MixingMatrix


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
    with pytest.raises(ValueError, match='kind="generalised" requires exponent'):
        ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            kind="generalised",
        )


def test_exponent_rejected_for_other_kinds() -> None:
    state, age, _pmap = _sir_age()
    for kind in ("frequency", "density"):
        with pytest.raises(ValueError, match="only valid with"):
            ForceOfInfection(
                "infection",
                infectious=state["I"],
                group_by=age,
                kind=kind,  # type: ignore[arg-type]
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
    freq = _compile_foi(pmap, state, age, kind="frequency", contact_rate=Param("contact_rate"))
    gen = _compile_foi(
        pmap,
        state,
        age,
        kind="generalised",
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
    dens = _compile_foi(pmap, state, age, kind="density", contact_rate=Param("contact_rate"))
    gen = _compile_foi(
        pmap,
        state,
        age,
        kind="generalised",
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
        kind="generalised",
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
        kind="generalised",
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
    a = _compile_foi(pmap, state, age, kind="generalised", exponent=0.5)
    b = _compile_foi(pmap, state, age, kind="generalised", exponent=0.5)
    c = _compile_foi(pmap, state, age, kind="generalised", exponent=0.8)
    assert a._digest == b._digest
    assert a._digest != c._digest
