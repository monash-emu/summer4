"""Step 47 — NUTS tuning: Laplace metric, gradient timing, pilots, ranking, tune_nuts."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from tests import gate_nuts_tuning as gate_settings

from summer4.epi.calibration import NumpyroPosterior
from summer4.epi.calibration import workflow as wf
from summer4.epi.calibration.workflow import tune

# Trace order (b, then a) differs from numpyro's sorted block order (a, then b).
_L = np.array([[1.0, 0.0, 0.0], [0.9, 0.4, 0.0], [0.5, 0.2, 0.3]])
_COV = _L @ _L.T
_PRIOR_SD = 3.0


def _gaussian_model() -> None:
    b = numpyro.sample("b", dist.Normal(0.0, _PRIOR_SD))
    with numpyro.plate("p", 2):
        a = numpyro.sample("a", dist.Normal(0.0, _PRIOR_SD))
    x = jnp.concatenate([a, jnp.atleast_1d(b)])  # sorted order: a[0], a[1], b
    numpyro.factor("g", dist.MultivariateNormal(jnp.zeros(3), _COV).log_prob(x))


def _gaussian() -> tuple[NumpyroPosterior, wf.Candidates]:
    post = NumpyroPosterior(_gaussian_model)
    init = wf.Candidates.from_z(post, {"a": np.zeros((1, 2)), "b": np.zeros(1)})
    return post, init


def _exact_covariance() -> np.ndarray:
    return np.asarray(np.linalg.inv(np.linalg.inv(_COV) + np.eye(3) / _PRIOR_SD**2))


def test_laplace_metric_recovers_a_gaussian_in_numpyro_block_order() -> None:
    post, init = _gaussian()
    assert post.prior_names() == ("b", "a")
    metric = wf.laplace_metric(post, init)
    assert metric.sites == ("a", "b")
    assert metric.element_names == ("a[0]", "a[1]", "b")
    assert metric.clipped == 0
    np.testing.assert_allclose(metric.covariance, _exact_covariance(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(metric.std, np.sqrt(np.diag(_exact_covariance())), rtol=1e-4)
    dense = metric.inverse_mass_matrix(dense=True)
    diag = metric.inverse_mass_matrix(dense=False)
    assert set(dense) == set(diag) == {("a", "b")}
    assert dense[("a", "b")].shape == (3, 3) and diag[("a", "b")].shape == (3,)


def test_laplace_metric_takes_a_site_dict_and_checks_its_sites() -> None:
    post, _ = _gaussian()
    metric = wf.laplace_metric(post, {"a": np.zeros(2), "b": np.array(0.0)})
    np.testing.assert_allclose(metric.covariance, _exact_covariance(), rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="do not match"):
        wf.laplace_metric(post, {"a": np.zeros(2)})


def test_laplace_metric_clips_a_flat_direction() -> None:
    def ridge() -> None:
        x = numpyro.sample("x", dist.Normal(0.0, 1e4))
        y = numpyro.sample("y", dist.Normal(0.0, 1e4))
        numpyro.factor("sum", -0.5 * (x + y) ** 2)  # only x + y is identified

    post = NumpyroPosterior(ridge)
    metric = wf.laplace_metric(post, {"x": np.array(0.0), "y": np.array(0.0)}, max_var=25.0)
    assert metric.clipped == 1
    assert np.max(np.linalg.eigvalsh(metric.covariance)) == pytest.approx(25.0)


@pytest.mark.parametrize("dense", [True, False])
def test_numpyro_uses_the_laplace_metric(dense: bool) -> None:
    from numpyro.infer import MCMC

    post, init = _gaussian()
    metric = wf.laplace_metric(post, init)
    setting = wf.NutsSetting(
        "fixed", {"dense_mass": dense, "adapt_mass_matrix": False}, laplace=True
    )
    mcmc = setting.make_mcmc(post.numpyro_model(), num_samples=10, num_chains=1, metric=metric)(20)
    assert isinstance(mcmc, MCMC)
    mcmc.warmup(jax.random.PRNGKey(0))
    used = mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
    expected = metric.covariance if dense else np.diag(metric.covariance)
    np.testing.assert_allclose(np.asarray(used[("a", "b")]), expected, rtol=1e-6)


def test_a_laplace_setting_without_a_metric_is_an_error() -> None:
    post, _ = _gaussian()
    setting = wf.NutsSetting("needs", {}, laplace=True)
    with pytest.raises(ValueError, match="needs a LaplaceMetric"):
        setting.make_mcmc(post.numpyro_model(), num_samples=10, num_chains=1)


def test_time_gradient_waits_for_the_computation() -> None:
    @jax.custom_jvp
    def slow(x: Any) -> Any:
        def sleep(v: Any) -> Any:
            time.sleep(0.02)
            return np.asarray(v)

        return jax.pure_callback(sleep, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    @slow.defjvp
    def _slow_jvp(primals: Any, tangents: Any) -> Any:
        return slow(primals[0]), tangents[0]

    def model() -> None:
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.factor("slow", slow(x))

    post = NumpyroPosterior(model)
    seconds = wf.time_gradient(post, {"x": np.array(0.1)}, repeats=3)
    assert 0.02 <= seconds < 1.0
    fast, init = _gaussian()
    assert 0.0 < wf.time_gradient(fast, init, repeats=5) < 0.02


def test_screens() -> None:
    labels = [s.label for s in wf.metric_screen(50)]
    assert labels == ["diag", "diag+laplace", "dense", "dense+laplace", "laplace fixed"]
    assert all(s.num_warmup == 50 for s in wf.metric_screen(50))
    fixed = wf.metric_screen()[-1]
    assert fixed.dense and fixed.laplace and fixed.kwargs["adapt_mass_matrix"] is False

    base = wf.NutsSetting("dense", {"dense_mass": True, "max_tree_depth": 6})
    variants = wf.trajectory_screen(base)
    assert [s.label for s in variants] == [
        "dense, depth 8",
        "dense, target 0.9",
        "dense, target 0.95",
        "dense, heuristic step",
    ]  # depth 6 is base's own depth, so it is left out
    assert variants[0].kwargs == {"dense_mass": True, "max_tree_depth": 8}
    assert variants[1].kwargs["max_tree_depth"] == 6 and variants[1].max_tree_depth == 6
    assert wf.NutsSetting("x", {"max_tree_depth": (12, 7)}).max_tree_depth == 7


def test_pilot_row_is_consistent_and_the_run_resumes() -> None:
    post, init = _gaussian()
    metric = wf.laplace_metric(post, init)
    rounds: list[Any] = []
    result = wf.pilot(
        post,
        init,
        wf.NutsSetting("dense+laplace", {"dense_mass": True}, laplace=True, num_warmup=60),
        metric=metric,
        grad_seconds=1e-5,
        num_chains=2,
        pilot_samples=50,
        on_round=rounds.append,
    )
    row = result.row
    assert result.run is not None and result.warmup is not None
    num_steps = result.run.extra["num_steps"]
    assert num_steps.shape == (2, 100) and row["draws"] == 200
    assert row["grad_evals"] == int(num_steps.sum())
    assert row["ess_per_grad"] * row["grad_evals"] == pytest.approx(row["ess_min"])
    assert row["num_steps_max"] == int(num_steps.max())
    assert row["treedepth_frac"] == pytest.approx(float(np.mean(num_steps >= 2**10 - 1)))
    assert row["warmup_rounds"] == len(rounds) == result.warmup.rounds
    assert row["warmup_grad_evals"] == sum(int(r["grad_evals"]) for r in rounds)
    iterations = 2 * sum(int(r["num_warmup"]) for r in rounds)
    assert row["warmup_steps_mean"] == pytest.approx(row["warmup_grad_evals"] / iterations)
    assert row["num_warmup"] == result.warmup.num_warmup
    assert row["predicted_ess_per_second"] == pytest.approx(row["ess_per_grad"] / 1e-5)
    assert row["ess_per_second"] > 0 and row["draws_per_second"] > 0
    assert 0.0 < row["accept_mean"] <= 1.0 and row["step_size"] > 0
    # A pilot's run is live: keep sampling the warmed-up chains.
    result.run.extend(lambda r: (True, "more") if r["samples"] >= 150 else None)
    assert result.run.samples["a"].shape == (2, 150, 2)


def _fake_row(label: str, warm_s: float, ess_s: float, **overrides: Any) -> dict[str, Any]:
    row = {
        "label": label,
        "warmup_passed": True,
        "warmup_failed": None,
        "warmup_seconds": warm_s,
        "warmup_grad_evals": int(warm_s * 1000),
        "divergence_frac": 0.0,
        "rhat_max": 1.0,
        "ess_per_second": ess_s,
        "ess_per_grad": ess_s / 1000,
    }
    row.update(overrides)
    return row


def _tuning_with(rows: Sequence[dict[str, Any]]) -> wf.NutsTuning:
    post, init = _gaussian()
    tuning = wf.NutsTuning(post, init, laplace=False, time_repeats=1)
    for row in rows:
        tuning.pilots.append(wf.Pilot(wf.NutsSetting(row["label"]), None, None, row))
    return tuning


def test_table_ranks_by_cost_to_the_target_ess() -> None:
    # "cheap warm-up" warms up in 1 s and samples 10 ESS/s; "fast sampler" takes 50 s and
    # samples 100 ESS/s. They cost the same at 10/9 * 50 * 10 = 555.6 effective draws.
    tuning = _tuning_with(
        [
            _fake_row("cheap warm-up", 1.0, 10.0),
            _fake_row("fast sampler", 50.0, 100.0),
            _fake_row("divergent", 0.1, 1000.0, divergence_frac=0.2),
            _fake_row("bad warm-up", 0.1, 500.0, warmup_passed=False, warmup_failed="rhat"),
        ]
    )
    small = tuning.table(target_ess=100)
    # Unusable pilots come last, however cheap.
    assert list(small.index) == ["cheap warm-up", "fast sampler", "divergent", "bad warm-up"]
    assert small.loc["cheap warm-up", "cost_seconds"] == pytest.approx(11.0)
    assert small.loc["fast sampler", "cost_grads"] == pytest.approx(50_000 + 100 / 0.1)
    assert small.loc["divergent", "why"] == "divergences"
    assert small.loc["bad warm-up", "why"] == "warm-up failed (rhat)"
    assert list(small["rank"]) == [1, 2, 3, 4]
    large = tuning.table(target_ess=10_000)
    assert list(large.index[:2]) == ["fast sampler", "cheap warm-up"]
    assert tuning.best(100).setting.label == "cheap warm-up"
    assert tuning.best(10_000).setting.label == "fast sampler"


def test_table_can_rank_by_gradient_cost() -> None:
    tuning = _tuning_with(
        [
            _fake_row("a", 1.0, 10.0, warmup_grad_evals=10**6),
            _fake_row("b", 2.0, 10.0),
        ]
    )
    assert list(tuning.table(100).index) == ["b", "a"]  # by gradients
    assert list(tuning.table(100, by="cost_seconds").index) == ["a", "b"]
    with pytest.raises(ValueError, match="by must be"):
        tuning.table(100, by="ess")


def test_skipped_laplace_settings_and_no_usable_pilot() -> None:
    post, init = _gaussian()
    tuning = wf.NutsTuning(post, init, laplace=False, time_repeats=1)
    assert tuning.metric is None
    tuning.run([wf.NutsSetting("diag+laplace", {}, laplace=True)])
    assert tuning.pilots[0].run is None
    table = tuning.table()
    assert table.loc["diag+laplace", "why"] == "skipped: no Laplace metric"
    with pytest.raises(ValueError, match="no usable pilot"):
        tuning.best()


def test_tune_nuts_equals_its_documented_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    speed = {"diag": 5.0, "dense": 20.0, "dense+laplace": 30.0, "laplace fixed": 10.0}

    def fake_pilot(post: Any, init: Any, setting: wf.NutsSetting, **kwargs: Any) -> wf.Pilot:
        ess_s = speed.get(setting.label, 1.0 + len(setting.label) / 10)
        return wf.Pilot(setting, None, None, _fake_row(setting.label, 1.0, ess_s))

    monkeypatch.setattr(tune, "pilot", fake_pilot)
    post, init = _gaussian()
    called = wf.tune_nuts(post, init, target_ess=500, num_warmup=80, seed=3)

    expanded = wf.NutsTuning(post, init, seed=3)
    expanded.run(wf.metric_screen(80))
    expanded.run(wf.trajectory_screen(expanded.best(500).setting))

    assert [p.setting for p in called.pilots] == [p.setting for p in expanded.pilots]
    assert list(called.table(500).index) == list(expanded.table(500).index)
    assert called.pilots[5].setting.label == "dense+laplace, depth 6"


def test_recommended_setting_and_factory() -> None:
    post, init = _gaussian()
    tuning = wf.NutsTuning(post, init, num_chains=2, pilot_samples=50, time_repeats=2)
    tuning.run([wf.NutsSetting("dense+laplace", {"dense_mass": True}, laplace=True, num_warmup=50)])
    best = tuning.best()
    assert best.warmup is not None
    recommended = tuning.recommended()
    assert recommended.label == "dense+laplace"
    assert recommended.num_warmup == best.warmup.num_warmup
    mcmc = tuning.make_mcmc(num_samples=30)(recommended.num_warmup)
    assert mcmc.num_chains == 2 and mcmc.num_samples == 30
    assert mcmc.num_warmup == recommended.num_warmup
    assert tuning.metric is not None
    start = recommended.initial_metric(tuning.metric)
    np.testing.assert_allclose(start[("a", "b")], tuning.metric.covariance)


# --- gate claims of notebook 29 (same seeds and sizes, tests/gate_nuts_tuning.py) ---


@pytest.fixture(scope="module")
def gate() -> gate_settings.GateRun:
    return gate_settings.run_gate()


@pytest.mark.slow
def test_gate_laplace_metric_sees_the_pooling_correlation(gate: gate_settings.GateRun) -> None:
    s = gate_settings.summary(gate)
    assert gate.joint.metric is not None and gate.joint.metric.clipped == 0
    assert s["corr_infection_z_main"] > 0.5
    # A starting metric, not the answer: every scale within a factor of three of the posterior.
    assert all(1 / 3 < r < 3 for r in s["std_ratio"].values())


@pytest.mark.slow
def test_gate_laplace_start_shortens_warmup_trajectories(gate: gate_settings.GateRun) -> None:
    table = gate.joint.table(gate_settings.TARGET_ESS)
    steps = table["warmup_steps_mean"]
    assert steps["diag+laplace"] < 0.5 * steps["diag"]
    assert steps["dense+laplace"] < 0.5 * steps["dense"]


@pytest.mark.slow
def test_gate_dense_metric_beats_diagonal_on_the_ridge(gate: gate_settings.GateRun) -> None:
    per_grad = gate.main.table(gate_settings.TARGET_ESS)["ess_per_grad"]
    assert min(per_grad["dense"], per_grad["dense+laplace"]) > 1.2 * per_grad["diag"]


@pytest.mark.slow
def test_gate_tuning_runs_two_stages_and_recommends_a_usable_setting(
    gate: gate_settings.GateRun,
) -> None:
    table = gate.joint.table(gate_settings.TARGET_ESS)
    assert len(table) == 10
    best = gate.joint.best(gate_settings.TARGET_ESS)
    assert table.loc[best.setting.label, "rank"] == 1 and table.iloc[0]["usable"]
    stage_b = [p.setting.label for p in gate.joint.pilots[5:]]
    winner_a = stage_b[0].split(", ")[0]
    assert all(label.startswith(winner_a + ", ") for label in stage_b)


@pytest.mark.slow
def test_gate_recommended_run_converges_on_every_element(gate: gate_settings.GateRun) -> None:
    assert gate.warm.converged
    assert gate.run.converged and gate.run.reason == "diagnostics"
    last = gate.run.progress.iloc[-1]
    assert last["rhat_element_max"] <= 1.01
    assert last["ess_bulk_element_min"] >= gate_settings.TARGET_ESS
    assert gate.run.extra["num_steps"].shape == gate.run.diverging.shape
