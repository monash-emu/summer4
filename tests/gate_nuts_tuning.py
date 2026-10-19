"""Shared gate run for notebook 29 and ``test_gate_*`` (step 47).

Notebook 28's two posteriors: the main-only SIR (a ridge between infection and
recovery) and the joint model with partially pooled outbreaks (a plate and a
non-centred latent). Seeds and sizes here must match the notebook's cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optax

from summer4.epi.calibration import workflow as wf
from tests.gate_custom_posterior import build_gate_models

LHS_M = 48
LHS_SEED = 0
N_STARTS = 4
OPT_LR = 0.05
OPT_CHUNK = 50
OPT_MAX_STEPS = 800
OPT_SEED = 0
NUM_CHAINS = 4
PILOT_SAMPLES = 150
NUM_WARMUP = 100
TARGET_ESS = 400.0
TUNE_SEED = 0
RUN_SAMPLES = 250
RUN_SEED = 1
RUN_MAX_SAMPLES = 2000


@dataclass(frozen=True)
class GateRun:
    joint: wf.NutsTuning
    main: wf.NutsTuning
    warm: wf.WarmupRun
    run: wf.MCMCRun


def optimised_starts(post: wf.Posterior) -> wf.Candidates:
    """LHS, keep the best ``N_STARTS``, Adam to convergence, then score the ends."""
    design = wf.evaluate(post, wf.lhs(post, LHS_M, seed=LHS_SEED), batch_size=16)
    opt = wf.optimize(
        post,
        design.best(N_STARTS),
        method=wf.Optax(optax.adam(OPT_LR)),
        chunk_steps=OPT_CHUNK,
        max_steps=OPT_MAX_STEPS,
        seed=OPT_SEED,
    )
    return wf.evaluate(post, opt.candidates, batch_size=N_STARTS)


def run_gate() -> GateRun:
    gate = build_gate_models()
    joint = wf.tune_nuts(
        gate.joint,
        optimised_starts(gate.joint),
        target_ess=TARGET_ESS,
        num_warmup=NUM_WARMUP,
        num_chains=NUM_CHAINS,
        pilot_samples=PILOT_SAMPLES,
        seed=TUNE_SEED,
    )
    main = wf.NutsTuning(
        gate.bm,
        optimised_starts(gate.bm),
        num_chains=NUM_CHAINS,
        pilot_samples=PILOT_SAMPLES,
        seed=TUNE_SEED,
    ).run(wf.metric_screen(NUM_WARMUP))

    recommended = joint.recommended(TARGET_ESS)
    warm = wf.warmup_until(
        joint.make_mcmc(num_samples=RUN_SAMPLES, target_ess=TARGET_ESS),
        recommended.num_warmup,
        wf.PILOT_WARMUP_RULE,  # the pilots' check, so their num_warmup carries over
        init_params=joint.init.init_params(NUM_CHAINS, seed=RUN_SEED),
        carry_metric=True,
        seed=RUN_SEED,
    )
    run = wf.sample_until(
        warm.mcmc,
        wf.StopRule(
            rhat=1.01,
            ess=TARGET_ESS,
            per_element=True,
            max_divergence_frac=0.01,
            max_samples=RUN_MAX_SAMPLES,
        ),
        seed=RUN_SEED,
        extra_fields=("num_steps", "accept_prob"),
    )
    return GateRun(joint=joint, main=main, warm=warm, run=run)


def summary(gate: GateRun) -> dict[str, Any]:
    """The numbers the notebook's *What to check* boxes and the gate tests quote."""
    import numpy as np

    joint = gate.joint.table(TARGET_ESS)
    main = gate.main.table(TARGET_ESS)
    metric = gate.joint.metric
    assert metric is not None
    # Posterior draws in unconstrained space, flattened in the metric's element order.
    draws = {k: np.asarray(v) for k, v in gate.run.samples.items()}
    n = draws[metric.sites[0]].shape[0] * draws[metric.sites[0]].shape[1]
    z = gate.joint.post.unconstrain({s: draws[s].reshape(n, *draws[s].shape[2:]) for s in draws})
    z_flat = np.concatenate([np.asarray(z[s]).reshape(n, -1) for s in metric.sites], axis=1)
    names = metric.element_names
    corr = metric.covariance / np.outer(metric.std, metric.std)
    return {
        "joint": joint,
        "main": main,
        "std_ratio": dict(zip(names, metric.std / z_flat.std(axis=0), strict=True)),
        "corr_infection_z_main": float(corr[names.index("infection"), names.index("z_main")]),
        "run_progress": gate.run.progress,
        "run_converged": gate.run.converged,
    }
