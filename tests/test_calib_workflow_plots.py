"""Step 28: post-calibration figures (CW10 spaghetti / ribbons, CW11 scenarios)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np
import plotly.graph_objects as go
import pytest
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS

from summer4 import (
    Compartments,
    FlowModel,
    OutputSet,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TransitionFlow,
    derived_refs,
)
from summer4.epi.calibration import (
    BayesianModel,
    NormalLikelihood,
    PosteriorRuns,
    Scenario,
    Uniform,
)
from summer4.epi.calibration import workflow as wf

TIMES = np.array([0.0, 20.0, 40.0, 60.0])


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _build() -> tuple[BayesianModel, Target, Compartments]:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    qty = Compartments(where=state["I"])
    run_kwargs = {"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"}
    truth = cm.run(
        {"infection": 0.35, "recovery": 0.1},
        y0,
        save=SavePlan(requests={"I": SaveRequest(qty, ts=TIMES)}),
        **run_kwargs,
    )
    values = np.asarray(truth["I"].at_times(TIMES).values.data).reshape(-1)
    target = Target(
        key="I", times=TIMES, values=values, quantity=qty, likelihood=NormalLikelihood(sd=5.0)
    )
    bm = BayesianModel(
        cm,
        {},
        priors=(Uniform("infection", 0.05, 1.0), Uniform("recovery", 0.02, 0.5)),
        targets=TargetSet(targets=(target,)),
        y0=y0,
        run_kwargs=run_kwargs,
    )
    return bm, target, qty


@pytest.fixture(scope="module")
def setup() -> dict[str, Any]:
    bm, target, qty = _build()
    design = wf.evaluate(bm, wf.lhs(bm, 24, seed=1), batch_size=12)
    fitted = wf.optimize(
        bm,
        design.best(3),
        method=wf.Optax(learning_rate=0.05, plateau=False),
        tuning=wf.AutoTune(lr_grid=(0.05,)),
        chunk_steps=10,
        max_steps=30,
        seed=0,
    )
    sampled = bm.sample(
        NUTS(bm.numpyro_model()),
        init=fitted.candidates,
        num_warmup=20,
        num_samples=15,
        num_chains=2,
        stop=lambda row: (True, "enough") if row["samples"] >= 30 else None,
    )
    projection = OutputSet()
    projection["infectious"] = qty.total()
    scenarios = {
        "baseline": Scenario(outputs=projection),
        "faster recovery": Scenario(params={"recovery": 0.3}, outputs=projection),
    }
    runs = bm.posterior_runs(sampled, n=12, seed=0, batch_size=6, scenarios=scenarios)
    return {
        "bm": bm,
        "target": target,
        "design": design,
        "fitted": fitted,
        "sampled": sampled,
        "runs": runs,
        "scenarios": scenarios,
    }


def _segments(trace: Any) -> list[np.ndarray]:
    """Split a gapped spaghetti trace back into its lines."""
    y = np.asarray(trace.y, dtype=float)
    breaks = np.flatnonzero(np.isnan(y))
    starts = np.concatenate([[0], breaks[:-1] + 1])
    return [y[s:e] for s, e in zip(starts, breaks, strict=True)]


# --- plotly stays a lazy import ---------------------------------------------


def test_workflow_import_does_not_import_plotly() -> None:
    code = (
        "import sys; from summer4.epi.calibration import workflow as wf; "
        "print('plotly' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- CW10: spaghetti --------------------------------------------------------


def test_spaghetti_is_one_trace_with_one_line_per_draw(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    fig = wf.plot_spaghetti(runs, "infectious", n=None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "baseline: 12 draws"
    lines = _segments(fig.data[0])
    np.testing.assert_allclose(np.stack(lines), runs.samples("baseline", "infectious"))
    assert fig.layout.title.text == "infectious: 12 sampled trajectories (baseline)"
    assert fig.layout.xaxis.title.text == "time"
    assert fig.layout.yaxis.title.text == "infectious"


def test_spaghetti_subsamples_distinct_draws(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    fig = wf.plot_spaghetti(runs, "infectious", scenario="faster recovery", n=5, seed=3)
    lines = np.stack(_segments(fig.data[0]))
    assert lines.shape[0] == 5
    all_draws = runs.samples("faster recovery", "infectious")
    rows = [int(np.flatnonzero(np.all(all_draws == line, axis=1))[0]) for line in lines]
    assert len(set(rows)) == 5
    assert fig.data[0].name == "faster recovery: 5 draws"


def test_spaghetti_overlays_targets_as_markers(setup: dict[str, Any]) -> None:
    bm, target = setup["bm"], setup["target"]
    design_runs = bm.posterior_runs(setup["design"].best(4), n=None)
    fig = wf.plot_spaghetti(design_runs, "I", targets=bm.targets)
    assert [trace.name for trace in fig.data] == ["baseline: 4 draws", "target I"]
    markers = fig.data[1]
    assert markers.mode == "markers"
    np.testing.assert_allclose(markers.x, target.times)
    np.testing.assert_allclose(markers.y, target.values)


def test_targets_match_output_by_key(setup: dict[str, Any]) -> None:
    bm, target, runs = setup["bm"], setup["target"], setup["runs"]
    with pytest.raises(ValueError, match="No target is keyed 'infectious'"):
        wf.plot_spaghetti(runs, "infectious", targets=bm.targets)
    fig = wf.plot_spaghetti(runs, "infectious", targets=target)
    assert fig.data[-1].name == "target I"
    fig = wf.add_targets(go.Figure(), [target, target], name="data")
    assert [trace.name for trace in fig.data] == ["data", "data"]


@pytest.mark.parametrize("stage", ["design", "fitted", "fitted.candidates", "sampled"])
def test_spaghetti_takes_points_from_every_stage(setup: dict[str, Any], stage: str) -> None:
    bm = setup["bm"]
    points = {
        "design": setup["design"].best(3),
        "fitted": setup["fitted"],
        "fitted.candidates": setup["fitted"].candidates,
        "sampled": setup["sampled"],
    }[stage]
    runs = bm.posterior_runs(points, n=3, seed=0, scenarios=setup["scenarios"])
    fig = wf.plot_spaghetti(runs, "infectious", n=None)
    assert len(_segments(fig.data[0])) == 3


# --- CW10: ribbons ----------------------------------------------------------


def test_ribbons_have_one_band_per_quantile_pair_and_a_median(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    fig = wf.plot_ribbons(runs, "infectious")
    names = [trace.name for trace in fig.data]
    assert names == [
        "baseline 2.5–97.5%",
        "baseline 25–75%",
        "baseline median",
        "faster recovery 2.5–97.5%",
        "faster recovery 25–75%",
        "faster recovery median",
    ]
    traj = runs.samples("baseline", "infectious")
    t = len(runs.times)
    outer = np.asarray(fig.data[0].y, dtype=float)
    np.testing.assert_allclose(outer[:t], np.quantile(traj, 0.025, axis=0), rtol=1e-6)
    np.testing.assert_allclose(outer[t:], np.quantile(traj, 0.975, axis=0)[::-1], rtol=1e-6)
    np.testing.assert_allclose(fig.data[2].y, np.quantile(traj, 0.5, axis=0), rtol=1e-6)
    assert fig.data[0].fill == "toself"
    assert fig.layout.title.text == "infectious: quantiles over 12 draws (2.5–97.5% band)"


def test_ribbons_even_quantiles_and_one_scenario(setup: dict[str, Any]) -> None:
    fig = wf.plot_ribbons(setup["runs"], "infectious", q=(0.1, 0.9), scenarios=["baseline"])
    assert [trace.name for trace in fig.data] == ["baseline 10–90%"]
    with pytest.raises(ValueError, match=r"quantiles must lie in \[0, 1\]"):
        wf.plot_ribbons(setup["runs"], "infectious", q=(0.5, 1.5))


def test_layers_compose_on_one_figure(setup: dict[str, Any]) -> None:
    runs, target = setup["runs"], setup["target"]
    fig = wf.plot_ribbons(runs, "infectious", scenarios=["baseline"], title="Mine")
    same = wf.plot_spaghetti(runs, "infectious", n=4, fig=fig, name="four draws")
    wf.add_targets(fig, target)
    assert same is fig
    assert [trace.name for trace in fig.data][-2:] == ["four draws", "target I"]
    assert fig.layout.title.text == "Mine"


# --- CW11: scenarios --------------------------------------------------------


def test_difference_is_the_paired_draw_difference(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    base = runs.samples("baseline", "infectious")
    fast = runs.samples("faster recovery", "infectious")
    np.testing.assert_array_equal(
        runs.difference("faster recovery", "baseline", "infectious"), fast - base
    )
    rel = runs.difference("faster recovery", "baseline", "infectious", relative=True)
    np.testing.assert_allclose(rel, (fast - base) / base)
    with pytest.raises(KeyError, match="Unknown ref scenario"):
        runs.difference("faster recovery", "nope", "infectious")


def test_scenario_differences_have_the_expected_sign(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    fig = wf.plot_scenarios(runs, "infectious", ref="baseline")
    names = [trace.name for trace in fig.data]
    assert names == [
        "faster recovery − baseline 2.5–97.5%",
        "faster recovery − baseline 25–75%",
        "faster recovery − baseline median",
    ]
    median = np.asarray(fig.data[2].y, dtype=float)
    # Faster recovery means fewer infectious people at every time after the start.
    assert median[0] == pytest.approx(0.0)
    assert np.all(median[1:] < 0.0)
    upper = np.asarray(fig.data[0].y, dtype=float)[len(runs.times) :]
    assert np.all(upper[:-1] <= 0.0)
    assert fig.layout.shapes[0].y0 == 0.0 and fig.layout.shapes[0].line.dash == "dash"
    assert fig.layout.yaxis.title.text == "infectious: scenario − baseline"


def test_scenario_plot_relative_and_errors(setup: dict[str, Any]) -> None:
    runs: PosteriorRuns = setup["runs"]
    fig = wf.plot_scenarios(runs, "infectious", relative=True, q=(0.5,))
    median = np.asarray(fig.data[0].y, dtype=float)
    assert np.all((median[1:] < 0.0) & (median[1:] >= -1.0))
    assert fig.layout.yaxis.title.text == "(infectious − baseline) / baseline"
    with pytest.raises(KeyError, match="Unknown ref scenario"):
        wf.plot_scenarios(runs, "infectious", ref="nope")
    with pytest.raises(ValueError, match="other than the reference"):
        wf.plot_scenarios(runs, "infectious", scenarios=["baseline"])


# --- diagnostics ------------------------------------------------------------


def test_plot_design_one_panel_per_site_and_layers(setup: dict[str, Any]) -> None:
    design: wf.Candidates = setup["design"]
    fig = wf.plot_design(design)
    assert len(fig.data) == 2
    assert fig.layout.xaxis.title.text == "infection"
    assert fig.layout.xaxis2.title.text == "recovery"
    assert fig.layout.yaxis.title.text == "log density"
    ok = np.asarray(design.ok, dtype=bool)
    np.testing.assert_allclose(fig.data[1].x, np.asarray(design.params["recovery"])[ok])
    np.testing.assert_allclose(fig.data[1].y, np.asarray(design.log_density)[ok])

    wf.plot_design(setup["fitted"].candidates, fig=fig, name="optimised")
    assert len(fig.data) == 4
    assert [t.name for t in fig.data[2:]] == ["optimised", "optimised"]
    assert [t.showlegend for t in fig.data[2:]] == [True, False]
    assert fig.data[3].xaxis == "x2"
    with pytest.raises(ValueError, match="evaluated candidates"):
        wf.plot_design(wf.lhs(setup["bm"], 4, seed=0))


def test_plot_design_leaves_failed_solves_out(setup: dict[str, Any]) -> None:
    design: wf.Candidates = setup["design"]
    ok = np.ones(len(design), dtype=bool)
    ok[:5] = False
    broken = wf.Candidates(
        sites=design.sites,
        z=design.z,
        params=design.params,
        log_density=design.log_density,
        ok=ok,
    )
    fig = wf.plot_design(broken)
    assert len(fig.data[0].x) == len(design) - 5
    assert fig.data[0].name == f"{len(design) - 5} points"


def test_plot_optimisation_draws_every_start(setup: dict[str, Any]) -> None:
    fitted: wf.OptimizeRun = setup["fitted"]
    fig = wf.plot_optimisation(fitted)
    trace = fitted.loss_trace
    assert [t.name for t in fig.data] == [f"start {i}" for i in range(trace.shape[1])]
    np.testing.assert_array_equal(fig.data[0].x, 10 * np.arange(1, trace.shape[0] + 1))
    np.testing.assert_allclose(fig.data[1].y, trace[:, 1])
    assert fig.layout.xaxis.title.text == "optimiser steps"
    assert "converged" in fig.layout.title.text


def test_plot_chains_reports_rhat_and_chunks(setup: dict[str, Any]) -> None:
    sampled: wf.MCMCRun = setup["sampled"]
    fig = wf.plot_chains(sampled)
    assert len(fig.data) == 2 * 2  # two sites x two chains
    assert [t.name for t in fig.data[:2]] == ["chain 0", "chain 1"]
    titles = [a.text for a in fig.layout.annotations]
    x = sampled.samples["infection"]
    rhat = float(split_gelman_rubin(x))
    ess = float(effective_sample_size(x))
    assert titles[0] == f"infection: R-hat {rhat:.3f}, bulk ESS {ess:.0f}"
    assert sampled.chunks == 2
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(vlines) == 2  # one boundary, drawn on both panels
    assert vlines[0].x0 == pytest.approx(14.5)
    assert "after 2 chunks (stop: enough)" in fig.layout.title.text


def test_plot_chains_takes_numpyro_mcmc_and_mappings(setup: dict[str, Any]) -> None:
    sampled: wf.MCMCRun = setup["sampled"]
    assert isinstance(sampled.mcmc, MCMC)
    from_mcmc = wf.plot_chains(sampled.mcmc, sites=["recovery"])
    assert len(from_mcmc.data) == 2
    from_map = wf.plot_chains({"a": np.random.default_rng(0).normal(size=(3, 40))})
    assert len(from_map.data) == 3
    assert from_map.layout.title.text.startswith("Chains: worst R-hat")
    with pytest.raises(TypeError, match="plot_chains takes"):
        wf.plot_chains(42)
