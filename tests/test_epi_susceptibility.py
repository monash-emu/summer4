"""``ForceOfInfection(susceptibility=...)``: weights on the recipient side of transmission."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from summer4 import FlowModel, Multiply, Param, Property, PropertyMap, TransitionFlow
from summer4.epi import ForceOfInfection, MixingMatrix, split_susceptibility

pytest.importorskip("jax")

# Asymmetric on purpose, so a transposed mixing product cannot pass.
K_ASYM = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.4],
    ]
)
CONTACT = 0.3


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("child", "adult", "elder"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _state(pmap: PropertyMap, state: Property, age: Property) -> np.ndarray:
    y = np.zeros(pmap.size)
    for i, name in enumerate(age.traits):
        y[pmap.select(state["S"] & age[name])] = 900.0 - 100.0 * i
        y[pmap.select(state["I"] & age[name])] = 10.0 * (i + 1)
        y[pmap.select(state["R"] & age[name])] = 50.0 + 5.0 * i
    return y


def _mass(y: np.ndarray, pmap: PropertyMap, sel: Any) -> float:
    return float(y[pmap.select(sel)].sum())


def _foi(state: Property, age: Property, **kwargs: Any) -> ForceOfInfection:
    options: dict[str, Any] = {
        "infectious": state["I"],
        "group_by": age,
        "contact_rate": CONTACT,
        "mixing": MixingMatrix(age, K_ASYM, normalize="none", check_reciprocal=False),
    }
    options.update(kwargs)
    return ForceOfInfection("infection", **options)


def _compile(
    pmap: PropertyMap,
    state: Property,
    foi: ForceOfInfection,
    *,
    infection_adjust: tuple[Multiply, ...] = (),
    reinfection_adjust: tuple[Multiply, ...] = (),
) -> Any:
    """SIR with reinfection: ``S -> I`` and ``R -> I`` share one force of infection."""
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow("infection", state["S"], state["I"], foi, adjust=infection_adjust)
    )
    model.add_flow(
        TransitionFlow("reinfection", state["R"], state["I"], foi, adjust=reinfection_adjust)
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    return model.compile()


def _captured(cm: Any, y: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
    return np.asarray(cm.observe(0.0, y, params or {}).captures["infection"].data)


def test_trait_keyed_susceptibility_matches_hand_formula() -> None:
    """λ_a = s_a · c · Σ_b K_ab I_b / N_b, with an asymmetric K."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    sus = {"child": 0.5, "adult": 1.0, "elder": 1.8}
    infectious = np.array([_mass(y, pmap, state["I"] & age[a]) for a in age.traits])
    population = np.array([_mass(y, pmap, age[a]) for a in age.traits])
    s = np.array([sus[a] for a in age.traits])
    expect = s * CONTACT * (K_ASYM @ (infectious / population))
    wrong_way = s * CONTACT * (K_ASYM.T @ (infectious / population))
    assert not np.allclose(expect, wrong_way)

    cm = _compile(pmap, state, _foi(state, age, susceptibility=sus))
    got = _captured(cm, y)
    np.testing.assert_allclose(got, expect, rtol=1e-6)
    # And the per-edge infection flow is λ_a · S_a.
    ctx = cm.observe(0.0, y, {})
    s_mass = np.array([_mass(y, pmap, state["S"] & age[a]) for a in age.traits])
    np.testing.assert_allclose(np.asarray(ctx.flows["infection"]), expect * s_mass, rtol=1e-6)


def test_trait_map_is_sugar_for_group_pairs() -> None:
    state, age, pmap = _sir_age()

    def digest(sus: object) -> bytes:
        return _compile(pmap, state, _foi(state, age, susceptibility=sus))._digest  # type: ignore[no-any-return]

    by_trait = {age["child"]: 0.5, age["elder"]: 2.0}
    by_name = {"child": 0.5, "elder": 2.0}
    pairs = [(age["elder"], 2.0), (age["child"], 0.5)]
    assert digest(by_trait) == digest(by_name) == digest(pairs)
    assert digest(by_trait) != digest({age["child"]: 0.5, age["elder"]: 3.0})


def test_susceptibility_and_infectiousness_digest_apart() -> None:
    """The same weights on the source side and on the recipient side are different models."""
    state, age, pmap = _sir_age()
    weights = {age["child"]: 0.5}
    as_sus = _compile(pmap, state, _foi(state, age, susceptibility=weights))
    as_inf = _compile(pmap, state, _foi(state, age, infectiousness=weights))
    assert as_sus._digest != as_inf._digest
    y = _state(pmap, state, age)
    assert not np.allclose(_captured(as_sus, y), _captured(as_inf, y))


def test_group_susceptibility_equals_flow_multiply() -> None:
    """``susceptibility={age[a]: s}`` is ``Multiply(s, where=age[a])`` on each infection flow.

    The plan asked for equal compile digests. They cannot be equal: the digest hashes
    the rate tree and the adjustment list structurally, and the two builds put the same
    number in different nodes (a ``ForceOfInfection`` field versus a flow adjustment).
    The claim is therefore numerical: identical flows and ``dy`` at a fixed state, and
    identical trajectories. The digests are asserted to differ so the reason stays
    written down.
    """
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    via_foi = _compile(pmap, state, _foi(state, age, susceptibility={age["child"]: 0.5}))
    halve = (Multiply(0.5, where=age["child"]),)
    via_adjust = _compile(
        pmap, state, _foi(state, age), infection_adjust=halve, reinfection_adjust=halve
    )
    assert via_foi._digest != via_adjust._digest
    a = via_foi.observe(0.0, y, {})
    b = via_adjust.observe(0.0, y, {})
    for name in ("infection", "reinfection"):
        np.testing.assert_allclose(np.asarray(a.flows[name]), np.asarray(b.flows[name]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(a.dy), np.asarray(b.dy), rtol=1e-6, atol=1e-6)
    # The FOI route saves the weighted λ; the adjustment route saves the unweighted one.
    unweighted = _captured(via_adjust, y)
    np.testing.assert_allclose(_captured(via_foi, y), unweighted * np.array([0.5, 1.0, 1.0]))


def test_compartment_susceptibility_equals_flow_multiply() -> None:
    """The other way round: a per-compartment pair against ``Multiply`` on one flow only."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    pairs = [(state["R"], 0.3), (state["R"] & age["elder"], 0.5)]
    via_foi = _compile(pmap, state, _foi(state, age, susceptibility=pairs))
    via_adjust = _compile(
        pmap,
        state,
        _foi(state, age),
        reinfection_adjust=(Multiply(0.3), Multiply(0.5, where=age["elder"])),
    )
    a = via_foi.observe(0.0, y, {})
    b = via_adjust.observe(0.0, y, {})
    for name in ("infection", "reinfection"):
        np.testing.assert_allclose(np.asarray(a.flows[name]), np.asarray(b.flows[name]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(a.dy), np.asarray(b.dy), rtol=1e-6, atol=1e-6)
    ra = via_foi.run({}, y, t0=0.0, t1=20.0, dt=0.1, solver="euler")
    rb = via_adjust.run({}, y, t0=0.0, t1=20.0, dt=0.1, solver="euler")
    np.testing.assert_allclose(
        np.asarray(ra.final_state.data), np.asarray(rb.final_state.data), rtol=1e-6
    )


def test_group_susceptibility_equals_row_scaled_matrix() -> None:
    """Chapter 15's claim: susceptibility of group a is row a of the mixing matrix."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    s = np.array([0.5, 1.0, 1.8])
    via_foi = _compile(
        pmap, state, _foi(state, age, susceptibility=dict(zip(age.traits, s, strict=True)))
    )
    rows = _compile(
        pmap,
        state,
        _foi(
            state,
            age,
            mixing=MixingMatrix(age, s[:, None] * K_ASYM, normalize="none", check_reciprocal=False),
        ),
    )
    np.testing.assert_allclose(_captured(via_foi, y), _captured(rows, y), rtol=1e-6)


def test_susceptibility_is_not_normalised() -> None:
    """Doubling every weight doubles λ. The same on infectiousness is normalised away."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    plain = _captured(_compile(pmap, state, _foi(state, age)), y)
    everyone = {name: 2.0 for name in age.traits}
    doubled_sus = _captured(_compile(pmap, state, _foi(state, age, susceptibility=everyone)), y)
    np.testing.assert_allclose(doubled_sus, 2.0 * plain, rtol=1e-6)
    doubled_inf = _captured(
        _compile(
            pmap,
            state,
            _foi(state, age, infectiousness=everyone, normalize_infectiousness="population"),
        ),
        y,
    )
    np.testing.assert_allclose(doubled_inf, plain, rtol=1e-6)


@pytest.mark.parametrize("normalize", ["population", "mean"])
def test_normalised_infectiousness_leaves_susceptibility_untouched(normalize: str) -> None:
    """Both surfaces at once: normalisation rescales the source side only."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    infectiousness = {age["child"]: 0.4, age["elder"]: 3.0}
    sus = {age["child"]: 0.5, age["adult"]: 1.2, age["elder"]: 2.5}
    without = _captured(
        _compile(
            pmap,
            state,
            _foi(state, age, infectiousness=infectiousness, normalize_infectiousness=normalize),
        ),
        y,
    )
    both = _captured(
        _compile(
            pmap,
            state,
            _foi(
                state,
                age,
                infectiousness=infectiousness,
                normalize_infectiousness=normalize,
                susceptibility=sus,
            ),
        ),
        y,
    )
    s = np.array([sus[age[name]] for name in age.traits])
    np.testing.assert_allclose(both, s * without, rtol=1e-6)


def test_param_weight_is_computed_path_and_differentiable() -> None:
    import jax
    import jax.numpy as jnp

    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    cm = _compile(
        pmap,
        state,
        _foi(state, age, susceptibility=[(age["child"], Param("sus_child")), (state["R"], 0.3)]),
    )
    assert ("sus_child",) in cm.computed_paths

    def child_force(s: Any) -> Any:
        return cm.observe(0.0, jnp.asarray(y), {"sus_child": s}).captures["infection"].data[0]

    grad = float(jax.grad(child_force)(jnp.asarray(0.7)))
    at_one = float(child_force(jnp.asarray(1.0)))
    # λ_child is linear in its own susceptibility, so d λ / d s = λ at s = 1.
    np.testing.assert_allclose(grad, at_one, rtol=1e-6)
    assert grad > 0.0


def test_runs_under_jit() -> None:
    import jax
    import jax.numpy as jnp

    state, age, pmap = _sir_age()
    y = jnp.asarray(_state(pmap, state, age))
    cm = _compile(
        pmap,
        state,
        _foi(state, age, susceptibility=[(age["child"], Param("s")), (state["R"], Param("r"))]),
    )

    def dy(s: Any, r: Any) -> Any:
        return cm.observe(0.0, y, {"s": s, "r": r}).dy

    eager = np.asarray(dy(jnp.asarray(0.5), jnp.asarray(0.3)))
    jitted = np.asarray(jax.jit(dy)(jnp.asarray(0.5), jnp.asarray(0.3)))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)


def test_jaxpr_does_not_grow_with_compartments() -> None:
    """Two pairs are two masked multiplies, however many compartments the map has."""
    import jax
    import jax.numpy as jnp

    def eqn_count(n_age: int, n_extra: int) -> int:
        state = Property("state", ("S", "I", "R"))
        age = Property("age", tuple(f"a{i}" for i in range(n_age)))
        extra = Property("extra", tuple(f"x{i}" for i in range(n_extra)))
        pmap = PropertyMap.from_property(state).stratify(age).stratify(extra)
        foi = ForceOfInfection(
            "infection",
            infectious=state["I"],
            group_by=age,
            contact_rate=Param("c"),
            mixing=MixingMatrix(
                age, np.full((n_age, n_age), 1.0 / n_age), normalize="none", check_reciprocal=False
            ),
            susceptibility=[(age["a0"], Param("s")), (state["R"] & extra["x0"], 0.3)],
        )
        model = FlowModel(pmap)
        model.add_flow(TransitionFlow("infection", state["S"], state["I"], foi))
        model.add_flow(TransitionFlow("reinfection", state["R"], state["I"], foi))
        cm = model.compile()
        y = jnp.ones(pmap.size)

        def field(c: Any, s: Any) -> Any:
            return cm.observe(0.0, y, {"c": c, "s": s}).dy

        return len(jax.make_jaxpr(field)(jnp.asarray(0.3), jnp.asarray(0.5)).jaxpr.eqns)

    small = eqn_count(3, 2)
    assert eqn_count(12, 2) == small
    assert eqn_count(3, 9) == small


def test_compartment_pairs_are_not_in_the_capture() -> None:
    """Group pairs scale the saved λ; per-compartment pairs scale only the rate."""
    state, age, pmap = _sir_age()
    y = _state(pmap, state, age)
    group_only = _foi(state, age, susceptibility={age["child"]: 0.5})
    mixed = _foi(state, age, susceptibility=[(age["child"], 0.5), (state["R"], 0.3)])
    assert not group_only.per_compartment
    assert mixed.per_compartment
    np.testing.assert_allclose(
        _captured(_compile(pmap, state, mixed), y),
        _captured(_compile(pmap, state, group_only), y),
        rtol=1e-6,
    )
    group_only.captured()
    with pytest.raises(ValueError, match="per-compartment susceptibility"):
        mixed.captured()


def test_compartment_by_age_susceptibility() -> None:
    """A selector-keyed weight over compartment × age multiplies only the edges it matches."""
    state = Property("state", ("S", "L", "I"))
    age = Property("age", ("0", "15"))
    pmap = PropertyMap.from_property(state).stratify(age)
    y = np.zeros(pmap.size)
    for sel, value in (
        (state["S"] & age["0"], 100.0),
        (state["L"] & age["0"], 30.0),
        (state["I"] & age["0"], 5.0),
        (state["S"] & age["15"], 200.0),
        (state["L"] & age["15"], 80.0),
        (state["I"] & age["15"], 20.0),
    ):
        y[pmap.select(sel)] = value
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        kind="density",
        contact_rate=0.01,
        mixing=MixingMatrix(age, np.eye(2), normalize="none", check_reciprocal=False),
        # Latent people are half as susceptible; latent children not at all.
        susceptibility=[(state["L"], 0.5), (state["L"] & age["0"], 0.0)],
    )
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("naive", state["S"], state["I"], foi))
    model.add_flow(TransitionFlow("latent", state["L"], state["I"], foi))
    ctx = model.compile().observe(0.0, y, {})
    force = np.array([0.01 * 5.0, 0.01 * 20.0])
    np.testing.assert_allclose(np.asarray(ctx.captures["infection"].data), force)
    np.testing.assert_allclose(np.asarray(ctx.flows["naive"]), force * np.array([100.0, 200.0]))
    np.testing.assert_allclose(np.asarray(ctx.flows["latent"]), force * np.array([0.0, 0.5 * 80.0]))


def test_split_susceptibility() -> None:
    state, age, _pmap = _sir_age()
    foi = _foi(
        state,
        age,
        susceptibility=[(age["child"], 0.5), (state["R"], 0.3), (age["adult"] | age["elder"], 2.0)],
    )
    per_group, per_compartment = split_susceptibility(foi.susceptibility, group_by=age)
    assert [sel for sel, _ in per_group] == [age["child"]]
    assert len(per_compartment) == 2
    assert split_susceptibility(None, group_by=age) == ((), ())


def test_key_on_another_property_names_both() -> None:
    state, age, _pmap = _sir_age()
    with pytest.raises(ValueError, match="Susceptibility key property 'state'.*group_by 'age'"):
        _foi(state, age, susceptibility={state["R"]: 0.3})


def test_bad_shapes_raise() -> None:
    state, age, _pmap = _sir_age()
    with pytest.raises(TypeError, match="susceptibility must be a mapping"):
        _foi(state, age, susceptibility=0.5)
    with pytest.raises(TypeError, match="susceptibility sequence"):
        _foi(state, age, susceptibility=(state["R"], 0.3))


def test_selector_matching_no_compartment_raises() -> None:
    state, age, pmap = _sir_age()
    never = state["S"] & state["R"]
    cm = _compile(pmap, state, _foi(state, age, susceptibility=[(never, 0.5)]))
    with pytest.raises(ValueError, match="susceptibility selector"):
        cm.observe(0.0, _state(pmap, state, age), {})
