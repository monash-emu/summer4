"""Composable calibration workflow stages (Phase J, WP19).

Re-exported as ``from summer4.epi.calibration import workflow as wf``.
"""

from __future__ import annotations

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
from summer4.epi.calibration.workflow.warmup import (
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
    "MCMCRun",
    "Optax",
    "OptimizeBackend",
    "OptimizeMethod",
    "OptimizeRun",
    "StageRecord",
    "Stop",
    "StopRule",
    "WarmupCheck",
    "WarmupRule",
    "WarmupRun",
    "evaluate",
    "lhs",
    "optimize",
    "prior_draws",
    "replay",
    "run_mcmc",
    "sample_until",
    "warmup_until",
]
