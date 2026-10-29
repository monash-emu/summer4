"""Calibration targets on transformed outputs (``BayesianModel(outputs=...)``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowMass,
    FlowModel,
    OutputSet,
    Param,
    Property,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Split,
    Target,
    TargetSet,
    TransitionFlow,
)
from summer4.epi import BayesianModel, ForceOfInfection, Poisson, Scenario, Uniform
from summer4.flows.compiled import CompiledModel

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("child", "adult"))
RUN = dict(t0=0.0, t1=70.0, dt=1.0, solver="euler")
TRUTH = {"contact_rate": 0.2, "recovery_time": 20.0}
OBS_TIMES = np.arange(7.0, 63.0, 7.0)
PRIORS = [Uniform("contact_rate", 0.001, 0.5), Uniform("recovery_time", 10.0, 50.0)]


def _model() -> CompiledModel:
    model = FlowModel(PropertyMap.from_property(STATE).stratify(AGE))
    foi = ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("contact_rate"))
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], foi))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], 1.0 / Param("recovery_time")))
    model.set_initial_population(
        {AGE["child"]: 1000.0, AGE["adult"]: 1500.0},
        [Split(STATE, {"S": 0.9, "I": 0.1, "R": 0.0})],
    )
    return model.compile()


def _weekly_outputs() -> OutputSet:
    outputs = OutputSet()
    outputs["weekly"] = FlowMass("infection").total().rolling(7, how="sum")
    return outputs


def _dense_daily_infections(cm: CompiledModel, params: Any) -> np.ndarray:
    res = cm.run(params, **RUN)
    return np.asarray(res["infection"].values.data).sum(axis=1)


def _weekly_by_hand(daily: np.ndarray) -> np.ndarray:
    # Trailing 7-point sums ending at each observation day (the grid is daily from day 0).
    return np.array([daily[int(t) - 6 : int(t) + 1].sum() for t in OBS_TIMES])


def _poisson_sum(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        sum(
            k * math.log(mu) - mu - math.lgamma(k + 1)
            for k, mu in zip(observed, predicted, strict=True)
        )
    )


def _observed() -> np.ndarray:
    return np.round(_weekly_by_hand(_dense_daily_infections(_model(), TRUTH)))


def _bm(cm: CompiledModel, outputs: OutputSet, observed: np.ndarray) -> BayesianModel:
    target = Target(key="weekly", times=OBS_TIMES, values=observed, likelihood=Poisson("sum"))
    return BayesianModel(cm, {}, PRIORS, TargetSet((target,)), outputs=outputs, run_kwargs=RUN)


def test_output_target_without_quantity_builds() -> None:
    bm = _bm(_model(), _weekly_outputs(), _observed())
    assert set(bm.save_plan.requests) == {"weekly"}
    assert bm.save_plan.requests["weekly"].ts is None  # the run's grid, not the observation times


def test_rolling_output_target_matches_hand_computation() -> None:
    cm = _model()
    observed = _observed()
    bm = _bm(cm, _weekly_outputs(), observed)
    params = {"contact_rate": 0.23, "recovery_time": 17.0}
    expected = _poisson_sum(observed, _weekly_by_hand(_dense_daily_infections(cm, params)))
    np.testing.assert_allclose(float(bm.log_likelihood(params)), expected, rtol=1e-5)


def test_pinning_the_leaf_to_observation_times_is_wrong() -> None:
    """Characterises the bug: a save pinned to the observation times breaks the window."""
    cm = _model()
    outputs = _weekly_outputs()
    pinned = SavePlan(requests={"weekly": SaveRequest(FlowMass("infection"), ts=OBS_TIMES)})
    wrong = outputs.evaluate(cm.run(TRUTH, save=pinned, **RUN), TRUTH)["weekly"]
    right = _weekly_by_hand(_dense_daily_infections(cm, TRUTH))
    wrong_values = np.asarray(wrong.at_times(OBS_TIMES).values).reshape(-1)
    assert not np.allclose(np.nan_to_num(wrong_values), right, rtol=0.05)


def test_true_parameters_score_best() -> None:
    bm = _bm(_model(), _weekly_outputs(), _observed())
    at_truth = float(bm.log_likelihood(TRUTH))
    for contact_rate in (0.18, 0.22):
        assert at_truth > float(bm.log_likelihood({**TRUTH, "contact_rate": contact_rate}))
    for recovery_time in (16.0, 24.0):
        assert at_truth > float(bm.log_likelihood({**TRUTH, "recovery_time": recovery_time}))


def test_plain_leaf_output_target_matches_quantity_target() -> None:
    cm = _model()
    infectious = Compartments(where=STATE["I"])
    res = cm.run(TRUTH, **RUN)
    comps = np.asarray(res["compartments"].values.data)
    observed = np.round(comps[OBS_TIMES.astype(int)][:, cm.pmap.select(STATE["I"])].sum(axis=1))

    outputs = OutputSet()
    outputs["I"] = infectious.total()
    by_output = BayesianModel(
        cm,
        {},
        PRIORS,
        TargetSet((Target(key="I", times=OBS_TIMES, values=observed, likelihood=Poisson("sum")),)),
        outputs=outputs,
        run_kwargs=RUN,
    )
    by_quantity = BayesianModel(
        cm,
        {},
        PRIORS,
        TargetSet(
            (
                Target(
                    key="I",
                    times=OBS_TIMES,
                    values=observed,
                    quantity=infectious,
                    reduce="sum",
                    likelihood=Poisson("sum"),
                ),
            )
        ),
        run_kwargs=RUN,
    )
    params = {"contact_rate": 0.25, "recovery_time": 15.0}
    np.testing.assert_allclose(
        float(by_output.log_likelihood(params)),
        float(by_quantity.log_likelihood(params)),
        rtol=1e-5,
    )


def test_target_key_colliding_with_an_output_leaf_raises() -> None:
    outputs = OutputSet()
    # Two leaves in one output: the second is saved under "both__2".
    outputs["both"] = FlowMass("infection").total() + FlowMass("recovery").total()
    assert "both__2" in outputs.plan().requests
    target = Target(
        key="both__2",
        times=OBS_TIMES,
        values=np.ones(OBS_TIMES.size),
        quantity=FlowMass("recovery"),
        likelihood=Poisson("sum"),
    )
    with pytest.raises(ValueError, match="both__2"):
        BayesianModel(_model(), {}, PRIORS, TargetSet((target,)), outputs=outputs, run_kwargs=RUN)


def test_sample_and_posterior_runs_with_output_target() -> None:
    bm = _bm(_model(), _weekly_outputs(), _observed())
    run = bm.sample(num_warmup=20, num_samples=20, num_chains=1, seed=0)
    assert np.asarray(run.samples["contact_rate"]).size == 20
    runs = bm.posterior_runs(run, n=5, seed=0, scenarios={"baseline": Scenario()})
    assert runs.samples("baseline", "weekly").shape[0] == 5
