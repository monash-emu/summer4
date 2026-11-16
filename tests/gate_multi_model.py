"""Shared gate models for notebook 38 and ``test_gate_*`` (step 50)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from summer4 import (
    Compartments,
    CompiledModel,
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
from summer4.epi.calibration import (
    BayesianModel,
    NegativeBinomial,
    Poisson,
    Uniform,
)

# --- constants (notebook cells must match) ---
DATA_SEED = 11
TRUE_MU = float(np.log(0.3))
TRUE_TAU = 0.25
TRUE_INFECTION = {"a": 0.55, "b": 0.75}
MU_PRIOR_LOC = float(np.log(0.25))
MU_PRIOR_SCALE = 1.0
TAU_PRIOR_SCALE = 0.3
NB_DISPERSION = 20.0
POP_SIZES = {"a": 1.0e6, "b": 4.0e5}
COHORT_START = 300.0
FIXED_RECOVERY = 0.25
MODELS = ("a", "b", "cohort")
OPT_SEED = 0
LHS_SEED = 0
MCMC_SEED = 3
MCMC_CHAINS = 2
MCMC_SAMPLES_PER_CHAIN = 200
MCMC_WARMUP_START = 60
MCMC_WARMUP_MIN = 80
OPT_MAX_STEPS = 300
OPT_CHUNK = 50
OPT_LR = 0.05
LHS_M = 48
N_OPT_STARTS = 8

CITY_TIMES = np.arange(0.0, 21.0, 2.0)
COHORT_TIMES = np.arange(0.0, 15.0, 1.0)


def build_city_model() -> tuple[CompiledModel, Property, PropertyMap]:
    """Frequency-dependent SIR with ``Param("infection")`` and ``Param("recovery")``."""
    state = Property("state", ("S", "I", "R"))
    pop = Property("pop", ("all",))
    pmap = PropertyMap.from_property(state).stratify(pop)
    flow_model = FlowModel(pmap)
    flow_model.add_flow(
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
    flow_model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    return flow_model.compile(), state, pmap


def build_cohort_model() -> tuple[CompiledModel, Property, PropertyMap]:
    """A natural-history cohort: states ``I`` and ``R`` only, one flow ``I -> R``."""
    state = Property("state", ("I", "R"))
    pmap = PropertyMap.from_property(state)
    flow_model = FlowModel(pmap)
    flow_model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    return flow_model.compile(), state, pmap


@dataclass(frozen=True)
class GateBundle:
    models: dict[str, BayesianModel]
    truth: dict[str, float]
    observed: dict[str, np.ndarray]
    times: dict[str, np.ndarray]
    joint_model: Any


def build_gate() -> GateBundle:
    """Two city models (shared compiled graph) and a cohort model, plus the joint model."""
    rng = np.random.default_rng(DATA_SEED)
    z_true = dict(zip(MODELS, rng.normal(size=len(MODELS)), strict=True))
    true_recovery = {m: float(np.exp(TRUE_MU + TRUE_TAU * z_true[m])) for m in MODELS}

    city_cm, city_state, city_pmap = build_city_model()
    run_kwargs = {"t0": 0.0, "t1": 20.0, "dt": 0.25, "solver": "euler"}
    models: dict[str, BayesianModel] = {}
    observed: dict[str, np.ndarray] = {}
    for m in ("a", "b"):
        y0_values = np.zeros(city_pmap.size)
        y0_values[city_pmap.select(city_state["S"])] = POP_SIZES[m] - 20.0
        y0_values[city_pmap.select(city_state["I"])] = 20.0
        y0 = PropertyData.wrap(city_pmap, y0_values)
        infectious = Compartments(where=city_state["I"])
        clean = city_cm.run(
            {"infection": TRUE_INFECTION[m], "recovery": true_recovery[m]},
            y0,
            save=SavePlan(requests={"I": SaveRequest(infectious, ts=CITY_TIMES)}),
            **run_kwargs,
        )
        expected = np.asarray(clean["I"].values.data).reshape(-1)
        observed[m] = rng.negative_binomial(
            NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + expected)
        ).astype(float)
        models[m] = BayesianModel(
            city_cm,
            {"recovery": FIXED_RECOVERY},
            priors=(Uniform("infection", 0.1, 1.5), Uniform("recovery", 0.02, 1.0)),
            targets=TargetSet(
                targets=(
                    Target(
                        key="I",
                        times=CITY_TIMES,
                        values=observed[m],
                        quantity=infectious,
                        likelihood=NegativeBinomial(dispersion=NB_DISPERSION),
                    ),
                )
            ),
            y0=y0,
            run_kwargs=run_kwargs,
        )

    cohort_cm, cohort_state, cohort_pmap = build_cohort_model()
    c_y0_values = np.zeros(cohort_pmap.size)
    c_y0_values[cohort_pmap.select(cohort_state["I"])] = COHORT_START
    c_y0 = PropertyData.wrap(cohort_pmap, c_y0_values)
    c_infectious = Compartments(where=cohort_state["I"])
    c_kwargs = {"t0": 0.0, "t1": 14.0, "dt": 0.25, "solver": "euler"}
    c_clean = cohort_cm.run(
        {"recovery": true_recovery["cohort"]},
        c_y0,
        save=SavePlan(requests={"I": SaveRequest(c_infectious, ts=COHORT_TIMES)}),
        **c_kwargs,
    )
    c_expected = np.asarray(c_clean["I"].values.data).reshape(-1)
    observed["cohort"] = rng.poisson(c_expected).astype(float)
    models["cohort"] = BayesianModel(
        cohort_cm,
        {},
        priors=(Uniform("recovery", 0.02, 1.0),),
        targets=TargetSet(
            targets=(
                Target(
                    key="I",
                    times=COHORT_TIMES,
                    values=observed["cohort"],
                    quantity=c_infectious,
                    likelihood=Poisson(),
                ),
            )
        ),
        y0=c_y0,
        run_kwargs=c_kwargs,
    )

    def joint_model() -> None:
        mu = numpyro.sample("mu", dist.Normal(MU_PRIOR_LOC, MU_PRIOR_SCALE))
        tau = numpyro.sample("tau", dist.HalfNormal(TAU_PRIOR_SCALE))
        z = {m: numpyro.sample(f"z_{m}", dist.Normal(0.0, 1.0)) for m in MODELS}
        for m in MODELS:
            recovery = jnp.exp(mu + tau * z[m])  # non-centred
            params = models[m].sample_params(f"{m}.", given={"recovery": recovery})
            models[m].factor(m, params)

    truth = {f"{m}.recovery": true_recovery[m] for m in MODELS}
    truth.update({f"{m}.infection": v for m, v in TRUE_INFECTION.items()})
    truth.update({"mu": TRUE_MU, "tau": TRUE_TAU})
    times = {"a": CITY_TIMES, "b": CITY_TIMES, "cohort": COHORT_TIMES}
    return GateBundle(models, truth, observed, times, joint_model)
