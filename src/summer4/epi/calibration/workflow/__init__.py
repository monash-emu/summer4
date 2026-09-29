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
from summer4.epi.calibration.workflow.plots import (
    DEFAULT_QUANTILES,
    add_targets,
    plot_chains,
    plot_design,
    plot_optimisation,
    plot_ribbons,
    plot_scenarios,
    plot_spaghetti,
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
    "DEFAULT_QUANTILES",
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
    "add_targets",
    "evaluate",
    "lhs",
    "optimize",
    "plot_chains",
    "plot_design",
    "plot_optimisation",
    "plot_ribbons",
    "plot_scenarios",
    "plot_spaghetti",
    "prior_draws",
    "replay",
    "run_mcmc",
    "sample_until",
    "warmup_until",
]
