"""Shared gate model for notebook 28 and ``test_gate_*`` (step 46)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from summer4 import (
    Compartments,
    FlowModel,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TransitionFlow,
)
from summer4.epi import FOIKind, ForceOfInfection, MixingMatrix
from summer4.epi.calibration import BayesianModel, NegativeBinomial, NumpyroPosterior, Uniform

# --- constants (notebook cells must match) ---
DATA_SEED = 7
K_OUTBREAKS = 6
N_DUR = 30
TRUE_INFECTION = 0.45
TRUE_MU = -1.1
TRUE_TAU = 0.1
MU_PRIOR_LOC = float(np.log(0.25))
MU_PRIOR_SCALE = 1.0
TAU_PRIOR_SCALE = 0.15
NB_DISPERSION = 20.0
POP_SIZE = 1.0e6
OPT_SEED = 0
LHS_MAIN_SEED = 0
LHS_JOINT_SEED = 0
MCMC_MAIN_SEED = 3
MCMC_JOINT_SEED = 4
MCMC_SAMPLES_PER_CHAIN = 200
MCMC_CHAINS = 2
MCMC_WARMUP_START = 40
MCMC_WARMUP_MIN = 60
OPT_MAX_STEPS = 800
OPT_CHUNK = 50
OPT_LR = 0.05
LHS_M_MAIN = 48
LHS_M_JOINT = 48
N_OPT_STARTS = 8


@dataclass(frozen=True)
class GateBundle:
    bm: BayesianModel
    joint: NumpyroPosterior
    truth: dict[str, float]
    durations: np.ndarray
    joint_model_fn: Callable[[], None]


def build_gate_models() -> GateBundle:
    """SIR ridge + partially pooled historical recovery durations."""
    rng = np.random.default_rng(DATA_SEED)
    z_hist = rng.normal(size=K_OUTBREAKS)
    z_main = float(rng.normal())
    true_recovery = float(np.exp(TRUE_MU + TRUE_TAU * z_main))
    rates_hist = np.exp(TRUE_MU + TRUE_TAU * z_hist)
    durations = rng.exponential(1.0 / rates_hist[:, None], size=(K_OUTBREAKS, N_DUR))

    state = Property("state", ("S", "I", "R"))
    pop = Property("pop", ("all",))
    pmap = PropertyMap.from_property(state).stratify(pop)
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            ForceOfInfection(
                "infection",
                infectious=state["I"],
                group_by=pop,
                mixing=MixingMatrix(pop, np.array([[1.0]])),
                kind=FOIKind.FREQUENCY,
                contact_rate=Param("infection"),
            ),
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    cm = model.compile()
    y0_values = np.zeros(pmap.size)
    y0_values[pmap.select(state["S"])] = POP_SIZE - 20.0
    y0_values[pmap.select(state["I"])] = 20.0
    y0 = PropertyData.wrap(pmap, y0_values)
    infectious = Compartments(where=state["I"])
    run_kwargs = {"t0": 0.0, "t1": 180.0, "dt": 0.5, "solver": "euler"}
    times = np.arange(0.0, 21.0, 2.0)
    clean = cm.run(
        {"infection": TRUE_INFECTION, "recovery": true_recovery},
        y0,
        save=SavePlan(requests={"I": SaveRequest(infectious, ts=times)}),
        **run_kwargs,
    )
    expected = np.asarray(clean["I"].values.data).reshape(-1)
    observed = rng.negative_binomial(20, 20 / (20 + expected)).astype(float)
    bm = BayesianModel(
        cm,
        {},
        priors=(
            Uniform("infection", 0.1, 1.5),
            Uniform("recovery", 0.02, 1.0),
        ),
        targets=TargetSet(
            targets=(
                Target(
                    key="I",
                    times=times,
                    values=observed,
                    quantity=infectious,
                    likelihood=NegativeBinomial(dispersion=NB_DISPERSION),
                ),
            )
        ),
        y0=y0,
        run_kwargs=run_kwargs,
    )
    truth = {
        "infection": TRUE_INFECTION,
        "recovery": true_recovery,
        "mu": TRUE_MU,
        "tau": TRUE_TAU,
        "z_main": z_main,
    }

    def joint_model() -> None:
        mu = numpyro.sample("mu", dist.Normal(MU_PRIOR_LOC, MU_PRIOR_SCALE))
        tau = numpyro.sample("tau", dist.HalfNormal(TAU_PRIOR_SCALE))
        infection = numpyro.sample("infection", dist.Uniform(0.1, 1.5))
        with numpyro.plate("outbreaks", K_OUTBREAKS):
            z = numpyro.sample("z", dist.Normal(0.0, 1.0))
            rate = jnp.exp(mu + tau * z)
            with numpyro.plate("dur", N_DUR):
                numpyro.sample(
                    "duration",
                    dist.Exponential(rate),
                    obs=jnp.asarray(durations.T),
                )
        z_main_s = numpyro.sample("z_main", dist.Normal(0.0, 1.0))
        recovery = jnp.exp(mu + tau * z_main_s)
        numpyro.factor(
            "main",
            bm.log_likelihood({"infection": infection, "recovery": recovery}),
        )

    joint = NumpyroPosterior(joint_model)
    return GateBundle(bm, joint, truth, durations, joint_model)


def recovery_from_joint_params(params: dict[str, Any]) -> np.ndarray:
    mu = np.asarray(params["mu"], float)
    tau = np.asarray(params["tau"], float)
    z_main = np.asarray(params["z_main"], float)
    return np.exp(mu + tau * z_main)
