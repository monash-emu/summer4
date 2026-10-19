"""Composable calibration workflow stages (Phase J, WP19).

Re-exported as ``from summer4.epi.calibration import workflow as wf``.
"""

from __future__ import annotations

from summer4.epi.calibration.posterior import NumpyroPosterior, Posterior
from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord
from summer4.epi.calibration.workflow.design import lhs, prior_draws
from summer4.epi.calibration.workflow.evaluate import evaluate
from summer4.epi.calibration.workflow.mcmc import (
    Decision,
    MCMCRun,
    Stop,
    StopRule,
    replay,
    run_mcmc,
    sample_until,
)
from summer4.epi.calibration.workflow.optimize import (
    CMAES,
    AutoTune,
    Optax,
    OptimizeBackend,
    OptimizeMethod,
    OptimizeRun,
    optimize,
)
from summer4.epi.calibration.workflow.tune import (
    LaplaceMetric,
    NutsSetting,
    NutsTuning,
    Pilot,
    laplace_metric,
    metric_screen,
    pilot,
    time_gradient,
    trajectory_screen,
    tune_nuts,
)
from summer4.epi.calibration.workflow.warmup import (
    OnRound,
    WarmupCheck,
    WarmupRule,
    WarmupRun,
    warmup_until,
)

__all__ = [
    "AutoTune",
    "CMAES",
    "Candidates",
    "Decision",
    "LaplaceMetric",
    "MCMCRun",
    "NumpyroPosterior",
    "NutsSetting",
    "NutsTuning",
    "OnRound",
    "Optax",
    "OptimizeBackend",
    "OptimizeMethod",
    "OptimizeRun",
    "Pilot",
    "Posterior",
    "StageRecord",
    "Stop",
    "StopRule",
    "WarmupCheck",
    "WarmupRule",
    "WarmupRun",
    "evaluate",
    "laplace_metric",
    "lhs",
    "metric_screen",
    "optimize",
    "pilot",
    "prior_draws",
    "replay",
    "run_mcmc",
    "sample_until",
    "time_gradient",
    "trajectory_screen",
    "tune_nuts",
    "warmup_until",
]
