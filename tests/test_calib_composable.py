"""CP2 (roadmap step 30): calibration entry points built on the workflow pieces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from numpyro.infer import MCMC, NUTS

from summer4 import (
    Compartments,
    FlowModel,
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
from summer4.epi.calibration import BayesianModel, NormalLikelihood, Uniform
from summer4.epi.calibration import workflow as wf


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_bm() -> BayesianModel:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    times = np.array([0.0, 20.0, 40.0, 60.0])
    qty = Compartments(where=state["I"])
    truth = cm.run(
        {"infection": 0.35, "recovery": 0.1},
        y0,
        t0=0.0,
        t1=80.0,
        dt=1.0,
        save=SavePlan(requests={"I": SaveRequest(qty, ts=times)}),
        solver="euler",
    )
    values = np.asarray(truth["I"].at_times(times).values.data).reshape(-1)
    targets = TargetSet(
        targets=(
            Target(
                key="I",
                times=times,
                values=values,
                quantity=qty,
                likelihood=NormalLikelihood(sd=5.0),
            ),
        )
    )
    return BayesianModel(
        cm,
        {"recovery": 0.1},
        priors=(Uniform("infection", 0.05, 1.0),),
        targets=targets,
        y0=y0,
        run_kwargs={"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"},
    )


def _starts(bm: BayesianModel, n: int = 3) -> wf.Candidates:
    return wf.evaluate(bm, wf.lhs(bm, 8, seed=1), batch_size=4).best(n)


NO_CONVERGENCE = wf.AutoTune(lr_grid=(0.05,), patience=10**6)
FIXED_ADAM = wf.Optax(learning_rate=0.05, plateau=False)


# --- CX17: a user-written optimisation method ------------------------------


class GradientDescent:
    """The smallest OptimizeMethod: fixed-step gradient descent."""

    def __init__(self, step_size: float) -> None:
        self.step_size = step_size

    def make(
        self,
        potential_fn: Any,
        *,
        learning_rate: float | None = None,
        sites: tuple[str, ...] | None = None,
        template_z: Mapping[str, Any] | None = None,
    ) -> Any:
        step_size = self.step_size

        class Backend:
            def init(self, z0: Mapping[str, Any], key: Any) -> dict[str, Any]:
                z = {k: jnp.asarray(v) for k, v in z0.items()}
                return {"z": z, "best_z": z, "best_loss": potential_fn(z)}

            def step(self, state: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
                loss, grads = jax.value_and_grad(potential_fn)(state["z"])
                z = jax.tree.map(lambda v, g: v - step_size * g, state["z"], grads)
                better = loss < state["best_loss"]
                best_z = jax.tree.map(
                    lambda cur, best: jnp.where(better, cur, best), state["z"], state["best_z"]
                )
                best_loss = jnp.where(better, loss, state["best_loss"])
                return {"z": z, "best_z": best_z, "best_loss": best_loss}, best_z, best_loss

        return Backend()


def test_user_written_method_runs_through_optimize() -> None:
    bm = _sir_bm()
    starts = _starts(bm)
    method = GradientDescent(1e-4)
    assert isinstance(method, wf.OptimizeMethod)
    assert isinstance(method.make(bm.potential_fn), wf.OptimizeBackend)
    run = wf.optimize(bm, starts, method=method, chunk_steps=10, max_steps=30)
    assert run.method_name == "gradientdescent"
    trace = run.loss_trace
    assert trace.shape[1] == len(starts)
    assert np.all(trace[-1] <= trace[0])
    assert np.isnan(run.learning_rate)


def test_optimize_rejects_an_object_without_make() -> None:
    bm = _sir_bm()
    with pytest.raises(TypeError, match="OptimizeMethod"):
        wf.optimize(bm, _starts(bm), method=optax.adam(0.05))  # type: ignore[arg-type]


# --- CX18 / CX24: OptimizeRun is resumable, history cumulative -------------


def test_extend_continues_from_stored_state() -> None:
    bm = _sir_bm()
    starts = _starts(bm)
    kw = {"method": FIXED_ADAM, "tuning": NO_CONVERGENCE, "chunk_steps": 20, "seed": 0}
    long = wf.optimize(bm, starts, max_steps=80, **kw)
    split = wf.optimize(bm, starts, max_steps=40, **kw).extend(40)
    assert split.loss_trace.shape == long.loss_trace.shape == (4, len(starts))
    np.testing.assert_allclose(split.loss_trace, long.loss_trace, rtol=1e-6)
    np.testing.assert_allclose(
        split.candidates.z["infection"], long.candidates.z["infection"], rtol=1e-6
    )


def test_optimize_equals_its_documented_expansion() -> None:
    bm = _sir_bm()
    starts = _starts(bm)
    kw = {"method": FIXED_ADAM, "tuning": NO_CONVERGENCE, "chunk_steps": 20, "seed": 3}
    short = wf.optimize(bm, starts, max_steps=40, **kw)
    long = wf.OptimizeRun(bm, starts, **kw).extend(40)
    np.testing.assert_array_equal(short.candidates.z["infection"], long.candidates.z["infection"])
    np.testing.assert_array_equal(short.loss_trace, long.loss_trace)


def test_history_is_cumulative_with_one_record_per_call() -> None:
    bm = _sir_bm()
    starts = _starts(bm)
    run = wf.optimize(bm, starts, method=FIXED_ADAM, tuning=NO_CONVERGENCE, max_steps=20)
    run.extend(20)
    stages = [record.stage for record in run.history]
    assert stages[: len(starts.history)] == [r.stage for r in starts.history]
    assert stages[len(starts.history) :] == ["optimize", "optimize"]
    assert run.history == run.candidates.history


def test_best_params_is_the_best_successful_start() -> None:
    bm = _sir_bm()
    run = wf.optimize(bm, _starts(bm), method=FIXED_ADAM, tuning=NO_CONVERGENCE, max_steps=40)
    cands = run.candidates
    i = int(np.argmax(cands.log_density))
    assert float(run.best_params["infection"]) == pytest.approx(float(cands.params["infection"][i]))


def test_chunk_program_is_public_and_size_independent() -> None:
    bm = _sir_bm()
    small = wf.OptimizeRun(bm, wf.lhs(bm, 2, seed=0), method=FIXED_ADAM, tuning=NO_CONVERGENCE)
    big = wf.OptimizeRun(bm, wf.lhs(bm, 6, seed=0), method=FIXED_ADAM, tuning=NO_CONVERGENCE)
    n_small = len(jax.make_jaxpr(small.chunk_program)(small.states).jaxpr.eqns)
    n_big = len(jax.make_jaxpr(big.chunk_program)(big.states).jaxpr.eqns)
    assert n_small == n_big


# --- CX19: find_map is a one-start optimize ---------------------------------


def test_init_point_and_from_z() -> None:
    bm = _sir_bm()
    z0 = bm.init_point(seed=4)
    assert set(z0) == {"infection"}
    np.testing.assert_array_equal(bm.init_point(seed=4)["infection"], z0["infection"])
    cands = wf.Candidates.from_z(bm, {"infection": np.asarray(z0["infection"])[None]})
    assert len(cands) == 1
    assert 0.05 < float(cands.params["infection"][0]) < 1.0


def test_find_map_equals_its_documented_expansion() -> None:
    bm = _sir_bm()
    run = bm.find_map(steps=60, seed=2)
    assert isinstance(run, wf.OptimizeRun)
    z0 = bm.init_point(2)
    start = wf.Candidates.from_z(bm, {k: jnp.asarray(v)[None] for k, v in z0.items()})
    expected = wf.optimize(
        bm, start, method=wf.Optax(optax.adam(0.05)), max_steps=60, chunk_steps=50, seed=2
    )
    np.testing.assert_array_equal(run.candidates.z["infection"], expected.candidates.z["infection"])
    assert abs(float(run.best_params["infection"]) - 0.35) < 0.05


def test_find_map_keeps_going_with_extend() -> None:
    bm = _sir_bm()
    run = bm.find_map(steps=20, seed=0)
    before = float(run.candidates.log_density[0])
    after = float(run.extend(200).candidates.log_density[0])
    assert after >= before
    assert len(run.history) == 2


# --- CX16: sample is a model-bound run_mcmc ---------------------------------


def test_sample_equals_run_mcmc_and_returns_the_run() -> None:
    bm = _sir_bm()
    starts = _starts(bm, 2)
    run = bm.sample("nuts", init=starts, num_warmup=10, num_samples=15, num_chains=2, seed=5)
    assert isinstance(run, wf.MCMCRun)
    assert run.chunks == 1 and run.reason == "num_samples"
    expected = wf.run_mcmc(
        bm,
        starts,
        num_chains=2,
        num_warmup=10,
        chunk_samples=15,
        stop=lambda row: (True, "num_samples"),
        seed=5,
    )
    np.testing.assert_array_equal(run.samples["infection"], expected.samples["infection"])
    assert np.asarray(run.idata.posterior["infection"]).shape == (2, 15)


def test_sample_takes_a_kernel_the_caller_built() -> None:
    bm = _sir_bm()
    kernel = NUTS(bm.numpyro_model(), target_accept_prob=0.95)
    run = bm.sample(kernel, num_warmup=10, num_samples=10)
    assert run.mcmc.sampler is kernel
    assert isinstance(run.mcmc, MCMC)


def test_sample_chunks_under_a_stop_rule() -> None:
    bm = _sir_bm()

    def thirty_draws(row: Mapping[str, Any]) -> tuple[bool, str] | None:
        return (True, "enough") if row["samples"] >= 30 else None

    run = bm.sample(num_warmup=10, num_samples=10, stop=thirty_draws)
    assert run.chunks == 3 and run.reason == "enough"


@pytest.mark.parametrize(
    ("kernel", "kwargs", "message"),
    [
        ("hmc2", {}, "Unknown kernel"),
        ("aies", {"num_chains": 1}, "num_chains >= 2"),
    ],
)
def test_sample_rejects_bad_kernels(kernel: str, kwargs: dict[str, Any], message: str) -> None:
    bm = _sir_bm()
    with pytest.raises(ValueError, match=message):
        bm.sample(kernel, num_warmup=5, num_samples=5, **kwargs)


def test_kernel_kwargs_are_only_for_named_kernels() -> None:
    bm = _sir_bm()
    with pytest.raises(ValueError, match="sugar for a named kernel"):
        bm.sample(NUTS(bm.numpyro_model()), num_warmup=5, num_samples=5, target_accept_prob=0.9)


def test_posterior_runs_accepts_run_objects() -> None:
    bm = _sir_bm()
    mcmc_run = bm.sample(num_warmup=10, num_samples=10, num_chains=1)
    from_run = bm.posterior_runs(mcmc_run, n=5, seed=0)
    from_idata = bm.posterior_runs(mcmc_run.idata, n=5, seed=0)
    np.testing.assert_array_equal(
        from_run.samples("baseline", "I"), from_idata.samples("baseline", "I")
    )
    opt_run = wf.optimize(bm, _starts(bm), method=FIXED_ADAM, tuning=NO_CONVERGENCE, max_steps=20)
    assert bm.posterior_runs(opt_run, n=None).samples("baseline", "I").shape[0] == 3
