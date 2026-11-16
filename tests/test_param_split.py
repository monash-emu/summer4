"""Rate-valued ``split=`` proportions: equivalence with float splits, normalisation, gradients."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    Dest,
    FlowModel,
    Multiply,
    Param,
    Property,
    PropertyMap,
    TraitChain,
    TraitMatrix,
    TransitionFlow,
)
from summer4.flows import EntryFlow
from summer4.flows.rates import Time

STAGE = Property("stage", ("incipient", "active"))
FOCUS = Property("focus", ("low", "mid", "high"))
EXTENT = Property("extent", ("low", "high"))


def _tb_pmap() -> PropertyMap:
    """Active TB split by extent everywhere, and by focus only where extent is low (ragged)."""
    pm = PropertyMap.from_property(STAGE).stratify(EXTENT)
    return pm.stratify(FOCUS, where=STAGE["active"] & EXTENT["low"])


def _field(
    pm: PropertyMap, flow: object, params: Mapping[str, object], *, hoist: bool = True
) -> np.ndarray:
    model = FlowModel(pm)
    model.add_flow(flow)  # type: ignore[arg-type]
    cm = model.compile(hoist=hoist)
    y = np.linspace(1.0, 2.0, pm.size)
    return np.asarray(cm.vector_field(0.0, y, dict(params)))


def _progression(split: object) -> TransitionFlow:
    return TransitionFlow(
        "progression", STAGE["incipient"], STAGE["active"], Param("rate"), split=split  # type: ignore[arg-type]
    )


def test_param_split_matches_float_split_on_ragged_dest() -> None:
    pm = _tb_pmap()
    p = Param("prop_infectious")
    dynamic = _field(
        pm,
        _progression({FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}}),
        {"rate": 0.2, "prop_infectious": 0.3},
    )
    static = _field(pm, _progression({FOCUS: {"low": 0.7, "mid": 0.0, "high": 0.3}}), {"rate": 0.2})
    np.testing.assert_allclose(dynamic, static, rtol=1e-12)


def test_param_split_matches_two_hand_written_flows() -> None:
    """The tb_acf idiom: two flows with complementary rates equal one flow with a split."""
    pm = PropertyMap.from_property(STAGE).stratify(
        Property("focus", ("low", "high")), where=STAGE["active"]
    )
    focus = pm.get_property("focus")
    rate, p = Param("rate"), Param("p")
    params = {"rate": 0.2, "p": 0.3}
    one = _field(
        pm,
        TransitionFlow(
            "progression",
            STAGE["incipient"],
            STAGE["active"],
            rate,
            split={focus: {"low": 1 - p, "high": p}},
        ),
        params,
    )
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow("lo", STAGE["incipient"], STAGE["active"] & focus["low"], rate * (1 - p))
    )
    model.add_flow(
        TransitionFlow("hi", STAGE["incipient"], STAGE["active"] & focus["high"], rate * p)
    )
    y = np.linspace(1.0, 2.0, pm.size)
    two = np.asarray(model.compile().vector_field(0.0, y, params))
    np.testing.assert_allclose(one, two, rtol=1e-12)


def test_singleton_group_keeps_full_flow() -> None:
    """Extent-high has no focus strata: its source sends everything to its one destination."""
    pm = _tb_pmap()
    p = Param("p")
    dy = _field(
        pm, _progression({FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}}), {"rate": 1.0, "p": 0.0}
    )
    y = np.linspace(1.0, 2.0, pm.size)
    src = int(pm.select(STAGE["incipient"] & EXTENT["high"])[0])
    dest = int(pm.select(STAGE["active"] & EXTENT["high"])[0])
    assert dy[dest] == pytest.approx(y[src])


def test_proportions_are_relative_weights() -> None:
    pm = _tb_pmap()
    split_scaled = {FOCUS: {"low": Param("a") * 2.0, "mid": Param("a") * 0.0, "high": Param("a")}}
    split_unit = {FOCUS: {"low": 2.0 / 3.0, "mid": 0.0, "high": 1.0 / 3.0}}
    dynamic = _field(pm, _progression(split_scaled), {"rate": 0.2, "a": 5.0})
    static = _field(pm, _progression(split_unit), {"rate": 0.2})
    np.testing.assert_allclose(dynamic, static, rtol=1e-12)


def test_all_zero_proportions_carry_no_flow_and_finite_gradients() -> None:
    pm = _tb_pmap()
    model = FlowModel(pm)
    a = Param("a")
    model.add_flow(_progression({FOCUS: {"low": a, "mid": a, "high": a}}))
    cm = model.compile()
    y = jnp.linspace(1.0, 2.0, pm.size)
    low = pm.select(STAGE["active"] & EXTENT["low"])
    dy = np.asarray(cm.vector_field(0.0, y, {"rate": 1.0, "a": 0.0}))
    np.testing.assert_allclose(dy[low], 0.0)
    grad = jax.grad(lambda v: cm.vector_field(0.0, y, {"rate": 1.0, "a": v})[low].sum())(0.0)
    assert np.isfinite(float(grad))


def test_gradient_through_split_proportion() -> None:
    pm = _tb_pmap()
    model = FlowModel(pm)
    p = Param("p")
    model.add_flow(_progression({FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}}))
    cm = model.compile()
    y = jnp.linspace(1.0, 2.0, pm.size)
    high = int(pm.select(STAGE["active"] & FOCUS["high"])[0])
    src = int(pm.select(STAGE["incipient"] & EXTENT["low"])[0])

    def inflow(value: float) -> jax.Array:
        return cm.vector_field(0.0, y, {"rate": 0.2, "p": value})[high]

    grad = jax.jit(jax.grad(inflow))(0.3)
    assert float(grad) == pytest.approx(0.2 * float(y[src]))


def test_hoisted_and_unhoisted_agree() -> None:
    pm = _tb_pmap()
    p = Param("p")
    flow = _progression({FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}})
    params = {"rate": 0.2, "p": 0.3}
    np.testing.assert_allclose(_field(pm, flow, params), _field(pm, flow, params, hoist=False))


def test_time_varying_split() -> None:
    pm = _tb_pmap()
    share = Time() / 10.0
    model = FlowModel(pm)
    model.add_flow(_progression({FOCUS: {"low": 1 - share, "mid": 0.0, "high": share}}))
    cm = model.compile()
    y = np.linspace(1.0, 2.0, pm.size)
    high = int(pm.select(STAGE["active"] & FOCUS["high"])[0])
    src = int(pm.select(STAGE["incipient"] & EXTENT["low"])[0])
    for t in (0.0, 2.5, 10.0):
        dy = np.asarray(cm.vector_field(t, y, {"rate": 1.0}))
        assert dy[high] == pytest.approx(y[src] * t / 10.0)


def test_mixed_static_and_param_properties() -> None:
    severity = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(STAGE).stratify(FOCUS, where=STAGE["active"])
    pm = pm.stratify(severity, where=STAGE["active"])
    p = Param("p")
    dynamic = _field(
        pm,
        _progression(
            {
                FOCUS: {"low": 1 - p, "mid": 0.0, "high": p},
                severity: {"mild": 0.75, "severe": 0.25},
            }
        ),
        {"rate": 0.2, "p": 0.4},
    )
    static = _field(
        pm,
        _progression(
            {
                FOCUS: {"low": 0.6, "mid": 0.0, "high": 0.4},
                severity: {"mild": 0.75, "severe": 0.25},
            }
        ),
        {"rate": 0.2},
    )
    np.testing.assert_allclose(dynamic, static, rtol=1e-12)


def _pairing_cases() -> list[tuple[str, Callable[[Property], object]]]:
    return [
        ("chain", lambda loc: TraitChain(loc, (("north", "south"),), rates=(0.5,))),
        ("matrix", lambda loc: TraitMatrix(loc, np.array([[0.0, 0.1], [0.2, 0.0]]))),
    ]


@pytest.mark.parametrize(("label", "make_pairing"), _pairing_cases())
def test_param_split_with_pairing(label: str, make_pairing: Callable[[Property], object]) -> None:
    """Each pairing part renormalises on its own, as the static split does."""
    loc = Property("location", ("north", "south"))
    state = Property("state", ("S", "I"))
    tier = Property("tier", ("a", "b"))
    pm = PropertyMap.from_property(state).stratify(loc).stratify(tier, where=state["I"])

    def flow(split: object) -> TransitionFlow:
        return TransitionFlow(
            "move",
            state["S"],
            state["I"],
            1.0,
            pairing=make_pairing(loc),  # type: ignore[arg-type]
            split=split,  # type: ignore[arg-type]
        )

    q = Param("q")
    dynamic = _field(pm, flow({tier: {"a": q, "b": 1 - q}}), {"q": 0.8})
    static = _field(pm, flow({tier: {"a": 0.8, "b": 0.2}}), {})
    np.testing.assert_allclose(dynamic, static, rtol=1e-12, err_msg=label)


def test_entry_flow_param_split() -> None:
    pm = PropertyMap.from_property(STAGE).stratify(FOCUS, where=STAGE["active"])
    p = Param("p")
    dynamic = _field(
        pm,
        EntryFlow(
            "arrivals", STAGE["active"], 10.0, split={FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}}
        ),
        {"p": 0.25},
    )
    static = _field(
        pm,
        EntryFlow(
            "arrivals",
            STAGE["active"],
            10.0,
            split={FOCUS: {"low": 0.75, "mid": 0.0, "high": 0.25}},
        ),
        {},
    )
    np.testing.assert_allclose(dynamic, static, rtol=1e-12)


def test_param_split_composes_with_dest_adjustment() -> None:
    pm = PropertyMap.from_property(STAGE).stratify(FOCUS, where=STAGE["active"])
    p = Param("p")
    flow = TransitionFlow(
        "progression",
        STAGE["incipient"],
        STAGE["active"],
        Param("rate"),
        split={FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}},
        adjust=[Multiply(Param("boost"), where=Dest(FOCUS["high"]))],
    )
    dy = _field(pm, flow, {"rate": 0.2, "p": 0.3, "boost": 2.0})
    y = np.linspace(1.0, 2.0, pm.size)
    src = int(pm.select(STAGE["incipient"])[0])
    high = int(pm.select(STAGE["active"] & FOCUS["high"])[0])
    low = int(pm.select(STAGE["active"] & FOCUS["low"])[0])
    assert dy[high] == pytest.approx(y[src] * 0.2 * 0.3 * 2.0)
    assert dy[low] == pytest.approx(y[src] * 0.2 * 0.7)


def test_missing_trait_still_raises() -> None:
    pm = _tb_pmap()
    model = FlowModel(pm)
    model.add_flow(_progression({FOCUS: {"low": Param("p"), "high": 1 - Param("p")}}))
    with pytest.raises(ValueError, match="must name every trait"):
        model.compile()


def test_unknown_param_is_reported() -> None:
    pm = _tb_pmap()
    model = FlowModel(pm)
    model.add_flow(_progression({FOCUS: {"low": Param("p"), "mid": 0.0, "high": 1 - Param("p")}}))
    cm = model.compile()
    y = np.ones(pm.size)
    with pytest.raises((KeyError, ValueError, TypeError)):
        cm.vector_field(0.0, y, {"rate": 0.1})


def test_digest_covers_split_expressions() -> None:
    pm = _tb_pmap()

    def compiled(name: str) -> object:
        model = FlowModel(pm)
        model.add_flow(
            _progression({FOCUS: {"low": 1 - Param(name), "mid": 0.0, "high": Param(name)}})
        )
        return model.compile()

    assert compiled("p")._digest == compiled("p")._digest  # type: ignore[attr-defined]
    assert compiled("p")._digest != compiled("q")._digest  # type: ignore[attr-defined]


def test_notebook_claim_a_one_flow_equals_three_hand_written_flows() -> None:
    """Acceptance notebook 37, section A: the tb_acf workaround on the ragged map."""
    pm = _tb_pmap()
    rate, p = Param("rate"), Param("p")
    params = {"rate": 0.1, "p": 0.3}
    one = _field(pm, _progression({FOCUS: {"low": 1 - p, "mid": 0.0, "high": p}}), params)
    model = FlowModel(pm)
    low_src = STAGE["incipient"] & EXTENT["low"]
    active_low = STAGE["active"] & EXTENT["low"]
    model.add_flow(TransitionFlow("lo", low_src, active_low & FOCUS["low"], rate * (1 - p)))
    model.add_flow(TransitionFlow("hi", low_src, active_low & FOCUS["high"], rate * p))
    model.add_flow(
        TransitionFlow(
            "ext", STAGE["incipient"] & EXTENT["high"], STAGE["active"] & EXTENT["high"], rate
        )
    )
    y = np.linspace(1.0, 2.0, pm.size)
    three = np.asarray(model.compile().vector_field(0.0, y, params))
    np.testing.assert_allclose(one, three, rtol=1e-12)
