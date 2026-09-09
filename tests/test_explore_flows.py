"""Join, rate-ref, and vector-field tests for the explore-flows spike."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from explorations.flows.prototype import (
    EntryFlow,
    ExitFlow,
    FieldRef,
    FlowModel,
    FlowRef,
    Multiply,
    Overwrite,
    TraitChain,
    TraitMatrix,
    Transform,
    TransitionFlow,
    actualize,
    as_adjust,
    as_rate,
    derived_refs,
    euler,
    identity_join,
    selector_properties,
)

from summer4 import Everything, Property, PropertyMap

FLOWS_DIR = Path(__file__).resolve().parents[1] / "explorations" / "flows"
NOTEBOOK = FLOWS_DIR / "01-flows.ipynb"
NOTEBOOK_02 = FLOWS_DIR / "02-flows.ipynb"


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _sir_age_sev() -> tuple[Property, Property, Property, PropertyMap]:
    state, age, pm = _sir_age()
    sev = Property("severity", ("mild", "severe"))
    return state, age, sev, pm.stratify(sev, where=state["I"])


def test_selector_properties_collects_bound_names() -> None:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    assert selector_properties(state["S"]) == frozenset({"state"})
    assert selector_properties(state["I"] & age["0-4"]) == frozenset({"state", "age"})
    assert selector_properties(Everything()) == frozenset()
    assert selector_properties(age.present()) == frozenset({"age"})


def test_identity_join_pairs_leftover_age() -> None:
    state, age, pm = _sir_age()
    edges = identity_join(pm, state["S"], state["I"])
    assert edges.n_edges == 3
    assert edges.src_idx.tolist() == pm.select(state["S"]).tolist()
    assert edges.dest_idx.tolist() == pm.select(state["I"]).tolist()
    np.testing.assert_allclose(edges.weight, 1.0)
    labels = pm.labels()
    for src, dest in zip(edges.src_idx, edges.dest_idx, strict=True):
        src_age = labels[int(src)].split("_")[1]
        dest_age = labels[int(dest)].split("_")[1]
        assert src_age == dest_age


def test_ageing_chain_has_no_top_band_edge() -> None:
    state, age, pm = _sir_age()
    flow = TransitionFlow(
        "ageing",
        age.present(),
        age.present(),
        0.2,
        pairing=TraitChain(age, (("0-4", "5-9"), ("5-9", "10+"))),
    )
    resolved = actualize(flow, pm)
    assert resolved.src_idx is not None
    assert resolved.dest_idx is not None
    assert resolved.src_idx.size == 6  # 3 states × 2 steps
    labels = pm.labels()
    pairs = {
        (labels[int(s)], labels[int(d)])
        for s, d in zip(resolved.src_idx, resolved.dest_idx, strict=True)
    }
    assert ("state=S_age=0-4", "state=S_age=5-9") in pairs
    assert ("state=I_age=5-9", "state=I_age=10+") in pairs
    assert not any(label.endswith("age=10+") for label, _ in pairs)


def test_ageing_chain_pair_rates_are_edge_scales() -> None:
    state, age, pm = _sir_age()
    flow = TransitionFlow(
        "ageing",
        age.present(),
        age.present(),
        1.0,
        pairing=TraitChain(age, (("0-4", "5-9"), ("5-9", "10+")), rates=(0.2, 0.1)),
    )
    resolved = actualize(flow, pm)
    labels = pm.labels()
    assert resolved.src_idx is not None
    from_young = [
        float(scale)
        for src, scale in zip(resolved.src_idx, resolved.scale, strict=True)
        if "age=0-4" in labels[int(src)]
    ]
    from_mid = [
        float(scale)
        for src, scale in zip(resolved.src_idx, resolved.scale, strict=True)
        if "age=5-9" in labels[int(src)]
    ]
    np.testing.assert_allclose(from_young, 0.2)
    np.testing.assert_allclose(from_mid, 0.1)


def test_sparse_matrix_emits_nonzero_location_edges() -> None:
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    matrix = np.array([[0.0, 0.1], [0.2, 0.0]])
    flow = TransitionFlow(
        "migration",
        loc.present(),
        loc.present(),
        1.0,
        pairing=TraitMatrix(loc, matrix),
    )
    resolved = actualize(flow, pm)
    assert resolved.src_idx is not None
    assert resolved.dest_idx is not None
    assert resolved.src_idx.size == 4  # 2 states × 2 directed loc edges
    np.testing.assert_allclose(sorted(resolved.scale.tolist()), [0.1, 0.1, 0.2, 0.2])


def test_ragged_dest_defaults_to_equal_split() -> None:
    state, age, _sev, pm = _sir_age_sev()
    edges = identity_join(pm, state["S"], state["I"])
    assert edges.n_edges == 6
    for src in pm.select(state["S"]):
        dest_w = edges.weight[edges.src_idx == src]
        np.testing.assert_allclose(dest_w, 0.5)
        assert dest_w.size == 2


def test_explicit_split_ratios() -> None:
    state, _age, sev, pm = _sir_age_sev()
    edges = identity_join(
        pm,
        state["S"],
        state["I"],
        split={sev: {"mild": 0.8, "severe": 0.2}},
    )
    labels = pm.labels()
    for src in pm.select(state["S"]):
        mask = edges.src_idx == src
        dests = edges.dest_idx[mask]
        weights = edges.weight[mask]
        by_label = {labels[int(d)]: float(w) for d, w in zip(dests, weights, strict=True)}
        mild = [w for label, w in by_label.items() if "mild" in label]
        severe = [w for label, w in by_label.items() if "severe" in label]
        np.testing.assert_allclose(mild, [0.8])
        np.testing.assert_allclose(severe, [0.2])


def test_split_product_and_renorm_with_two_dest_properties() -> None:
    state, age, pm = _sir_age()
    sev = Property("severity", ("mild", "severe"))
    strain = Property("strain", ("a", "b"))
    pm = pm.stratify(sev, where=state["I"]).stratify(strain, where=sev["severe"])
    edges = identity_join(
        pm,
        state["S"],
        state["I"],
        split={sev: {"mild": 0.8, "severe": 0.2}},
    )
    labels = pm.labels()
    src = int(pm.select(state["S"] & age["0-4"])[0])
    dests = edges.dest_idx[edges.src_idx == src]
    weights = edges.weight[edges.src_idx == src]
    by_label = {labels[int(d)]: float(w) for d, w in zip(dests, weights, strict=True)}
    assert len(by_label) == 3
    np.testing.assert_allclose(sum(by_label.values()), 1.0)
    mild = [w for label, w in by_label.items() if "mild" in label]
    severe = [w for label, w in by_label.items() if "severe" in label]
    np.testing.assert_allclose(mild, [0.8 / 1.2])
    np.testing.assert_allclose(severe, [0.2 / 1.2, 0.2 / 1.2])


def test_split_rejects_missing_trait_and_bad_sum() -> None:
    state, _age, sev, pm = _sir_age_sev()
    with pytest.raises(ValueError, match="must name every trait"):
        identity_join(pm, state["S"], state["I"], split={sev: {"mild": 1.0}})
    with pytest.raises(ValueError, match="must sum to 1"):
        identity_join(
            pm,
            state["S"],
            state["I"],
            split={sev: {"mild": 0.5, "severe": 0.6}},
        )


def test_split_rejects_matched_property_key() -> None:
    state, age, pm = _sir_age()
    with pytest.raises(ValueError, match="not a dest-only property"):
        identity_join(pm, state["S"], state["I"], split={age: {"0-4": 1.0, "5-9": 0.0, "10+": 0.0}})


def test_unmatched_free_key_raises() -> None:
    state = Property("state", ("S", "I", "R"))
    loc = Property("location", ("north", "south"))
    codes = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [2, 1],
        ],
        dtype=np.int16,
    )
    pm = PropertyMap(properties=(state, loc), codes=codes)
    with pytest.raises(ValueError, match="No destination match"):
        identity_join(pm, state["S"], state["I"])


class _DeathParams(NamedTuple):
    death_rate: float


class _ContactParams(NamedTuple):
    contact: float


def test_derived_refs_exposes_only_schema_fields() -> None:
    refs = derived_refs(_DeathParams)
    assert refs.death_rate == FieldRef(("death_rate",))
    with pytest.raises(AttributeError):
        _ = refs.foi  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="NamedTuple"):
        derived_refs(object)  # type: ignore[arg-type]


def test_add_flow_returns_flow_ref() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    ref = model.add_flow(ExitFlow("death", Everything(), 0.01))
    assert ref == FlowRef("death")


def test_field_ref_and_flow_ref_vector_field() -> None:
    state = Property("state", ("S", "I", "R"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    refs = derived_refs(_DeathParams)
    death = model.add_flow(ExitFlow("death", Everything(), refs.death_rate))
    model.add_flow(EntryFlow("birth", state["S"], death.sum()))
    vf = model.compile()
    y = np.array([900.0, 80.0, 20.0])
    dy = vf(0.0, y, {"death_rate": 0.01})
    np.testing.assert_allclose(dy, [-9.0 + 10.0, -0.8, -0.2])


def test_sum_over_location_replacement_births() -> None:
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    model = FlowModel(pm)
    refs = derived_refs(_DeathParams)
    death = model.add_flow(ExitFlow("death", Everything(), refs.death_rate))
    model.add_flow(EntryFlow("birth", state["S"], death.sum_over(loc)))
    y = np.array([100.0, 300.0, 10.0, 30.0])  # S-north, S-south, I-north, I-south
    dy = model.compile()(0.0, y, {"death_rate": 0.1})
    north_deaths = 0.1 * (100.0 + 10.0)
    south_deaths = 0.1 * (300.0 + 30.0)
    s_north, s_south = pm.select(state["S"])
    np.testing.assert_allclose(dy[s_north], -0.1 * 100.0 + north_deaths)
    np.testing.assert_allclose(dy[s_south], -0.1 * 300.0 + south_deaths)
    np.testing.assert_allclose(dy[pm.select(state["I"])], [-1.0, -3.0])
    np.testing.assert_allclose(dy.sum(), 0.0, atol=1e-12)
    assert north_deaths != south_deaths


def test_flow_ref_cycle_raises() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    model.add_flow(ExitFlow("a", state["S"], FlowRef("b").sum()))
    model.add_flow(ExitFlow("b", state["I"], FlowRef("a").sum()))
    with pytest.raises(ValueError, match="Cyclic"):
        model.compile()


def test_transition_mass_conservation() -> None:
    state, _age, sev, pm = _sir_age_sev()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            0.2,
            split={sev: {"mild": 0.7, "severe": 0.3}},
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    vf = model.compile()
    y = np.arange(pm.size, dtype=np.float64) + 1.0
    dy = vf(0.0, y, {})
    np.testing.assert_allclose(dy.sum(), 0.0, atol=1e-12)
    s_idx = pm.select(state["S"])
    leaving = 0.2 * y[s_idx]
    np.testing.assert_allclose(-dy[s_idx], leaving)


def test_as_rate_accepts_scalar_and_ref() -> None:
    refs = derived_refs(_ContactParams)
    expr = as_rate(0.25) * refs.contact
    recovered = as_rate(expr)
    assert recovered is expr


def test_jax_vector_field_matches_numpy() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from explorations.flows.propertydata import PropertyData

    state, age, pm = _sir_age()
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.3))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    y = np.array([100.0, 80.0, 50.0, 10.0, 8.0, 5.0, 0.0, 0.0, 0.0])
    dy_np = model.compile(backend="numpy")(0.0, y, {})
    vf_jax = model.compile(backend="jax")
    dy_jit = np.asarray(jax.jit(vf_jax)(0.0, jnp.asarray(y), {}))
    np.testing.assert_allclose(dy_jit, dy_np, rtol=1e-6)
    wrapped = PropertyData.wrap(pm, y)
    dy_pd = vf_jax(0.0, wrapped, {})
    np.testing.assert_allclose(np.asarray(dy_pd.data), dy_np, rtol=1e-6)
    assert dy_pd.pmap == pm


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise TypeError(f"Unexpected notebook source type: {type(source)}")


def _exec_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        stripped = source.strip()
        if not stripped:
            continue
        if stripped.startswith("%") or stripped.startswith("!"):
            raise AssertionError(f"{path.name} cell {index} uses a magic.")
        code_cells += 1
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"{path.name} cell {index} failed: {exc}") from exc
    assert code_cells > 0


def test_explore_flows_notebook_executes() -> None:
    _exec_notebook(NOTEBOOK)


class _MigBundle(NamedTuple):
    baseline: object
    seasonal: float


class _NestedDerived(NamedTuple):
    foi: float
    migration: _MigBundle


class _LeafRates(NamedTuple):
    rates: object


class _MigRates(NamedTuple):
    mig: object


class _Seasonal(NamedTuple):
    seasonal: float


class _Cap(NamedTuple):
    foi_cap: float


def test_derived_refs_recurses_into_nested_namedtuple() -> None:
    refs = derived_refs(_NestedDerived)
    assert refs.foi == FieldRef(("foi",))
    assert refs.migration.baseline == FieldRef(("migration", "baseline"))
    assert refs.migration.seasonal == FieldRef(("migration", "seasonal"))
    with pytest.raises(AttributeError):
        _ = refs.migration.unknown  # type: ignore[attr-defined]


def test_derived_refs_non_namedtuple_field_stays_leaf() -> None:
    refs = derived_refs(_LeafRates)
    assert refs.rates == FieldRef(("rates",))


def test_matrix_rate_gathers_dest_source_by_time() -> None:
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    refs = derived_refs(_MigRates)
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "migration",
            loc.present(),
            loc.present(),
            refs.mig,
            pairing=TraitMatrix(loc, mask),
        )
    )

    def derived_fn(params: object, y: object = None, t: float = 0.0) -> _MigRates:
        del params, y
        time = float(t)
        return _MigRates(mig=np.array([[0.0, time], [2.0 * time, 0.0]]))

    y = np.array([10.0, 100.0, 0.0, 0.0])
    vf = model.compile(derived_fn=derived_fn)
    s_north, s_south = pm.select(state["S"])
    dy = vf(1.0, y, {})
    np.testing.assert_allclose(dy[s_north], 80.0)
    np.testing.assert_allclose(dy[s_south], -80.0)
    np.testing.assert_allclose(dy.sum(), 0.0)
    dy2 = vf(2.0, y, {})
    np.testing.assert_allclose(dy2[s_north], 160.0)
    np.testing.assert_allclose(dy2[s_south], -160.0)


def test_nested_proxy_matrix_matches_combined_field() -> None:
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    baseline = np.array([[0.0, 0.1], [0.2, 0.0]])
    refs = derived_refs(_NestedDerived)
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "migration",
            loc.present(),
            loc.present(),
            refs.migration.baseline * refs.migration.seasonal,
            pairing=TraitMatrix(loc, mask),
        )
    )

    def derived_fn(params: object, y: object = None, t: float = 0.0) -> _NestedDerived:
        del params, y
        return _NestedDerived(
            foi=0.0,
            migration=_MigBundle(baseline=baseline, seasonal=1.0 + float(t)),
        )

    y = np.array([10.0, 40.0, 0.0, 0.0])
    dy = model.compile(derived_fn=derived_fn)(1.0, y, {})
    # seasonal=2: south->north 0.2, north->south 0.4
    s_north, s_south = pm.select(state["S"])
    np.testing.assert_allclose(dy[s_north], -0.4 * 10.0 + 0.2 * 40.0)
    np.testing.assert_allclose(dy[s_south], -0.2 * 40.0 + 0.4 * 10.0)


def test_as_adjust_defaults_to_multiply() -> None:
    adj = as_adjust(0.5)
    assert isinstance(adj, Multiply)
    assert adj.value == as_rate(0.5)


def test_adjust_multiply_default() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    refs = derived_refs(_Seasonal)
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.2, adjust=[refs.seasonal]))
    dy = model.compile()(0.0, np.array([100.0, 0.0]), _Seasonal(seasonal=0.5))
    np.testing.assert_allclose(dy, [-10.0, 10.0])


def test_adjust_overwrite_where() -> None:
    state, age, pm = _sir_age()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            0.2,
            adjust=[Overwrite(0.0, where=age["0-4"])],
        )
    )
    y = np.ones(pm.size)
    dy = model.compile()(0.0, y, {})
    np.testing.assert_allclose(dy[pm.select(state["S"] & age["0-4"])], 0.0)
    np.testing.assert_allclose(dy[pm.select(state["S"] & ~age["0-4"])], -0.2)


def test_adjust_transform_minimum() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    refs = derived_refs(_Cap)
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            0.2,
            adjust=[Transform(np.minimum, refs.foi_cap)],
        )
    )
    dy = model.compile()(0.0, np.array([100.0, 0.0]), _Cap(foi_cap=0.05))
    np.testing.assert_allclose(dy, [-5.0, 5.0])


def test_adjust_transform_depends_on_flow_ref() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    death = model.add_flow(ExitFlow("death", Everything(), 0.1))
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            0.2,
            adjust=[Transform(lambda prev, deaths: prev * deaths, death.sum())],
        )
    )
    y = np.array([100.0, 50.0])
    dy = model.compile()(0.0, y, {})
    death_sum = 0.1 * 150.0
    inf_mass = 0.2 * death_sum * 100.0
    np.testing.assert_allclose(dy[0], -0.1 * 100.0 - inf_mass)
    np.testing.assert_allclose(dy[1], -0.1 * 50.0 + inf_mass)


def test_euler_numpy_matches_manual_steps() -> None:
    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    vf = model.compile()
    y0 = np.array([100.0, 0.0])
    y_e = euler(vf, 0.0, y0, {}, dt=0.5, steps=2)
    y = y0.copy()
    t = 0.0
    for _ in range(2):
        y = y + 0.5 * vf(t, y, {})
        t += 0.5
    np.testing.assert_allclose(y_e, y)


def test_euler_jax_jit_matches_numpy_and_time_varies() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    refs = derived_refs(_MigRates)

    def derived_fn(params: dict[str, float], y: object = None, t: object = 0.0) -> _MigRates:
        del y
        seasonal = 1.0 + params["amp"] * jnp.sin(params["omega"] * t)
        baseline = jnp.asarray(np.array([[0.0, 0.1], [0.1, 0.0]]))
        return _MigRates(mig=baseline * seasonal)

    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "migration",
            loc.present(),
            loc.present(),
            refs.mig,
            pairing=TraitMatrix(loc, mask),
        )
    )
    y0 = np.array([80.0, 20.0, 0.0, 0.0])
    params = {"amp": 0.5, "omega": 1.0}
    vf_np = model.compile(derived_fn=derived_fn)
    y_np = euler(vf_np, 0.0, y0, params, dt=0.25, steps=8)
    vf_jax = model.compile(derived_fn=derived_fn, backend="jax")
    step = jax.jit(lambda y: euler(vf_jax, 0.0, y, params, dt=0.25, steps=8))
    y_j = np.asarray(step(jnp.asarray(y0)))
    np.testing.assert_allclose(y_j, y_np, rtol=1e-5)
    np.testing.assert_allclose(y_np.sum(), y0.sum())
    y_const = euler(vf_np, 0.0, y0, {"amp": 0.0, "omega": 1.0}, dt=0.25, steps=8)
    assert not np.allclose(y_np, y_const)


def test_refine_flows_notebook_executes() -> None:
    _exec_notebook(NOTEBOOK_02)
