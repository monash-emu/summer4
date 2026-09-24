"""Composable calibration workflow stages (Phase J, WP19).

Re-exported as ``from summer4.epi.calibration import workflow as wf``.
"""

from __future__ import annotations

from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord
from summer4.epi.calibration.workflow.design import lhs, prior_draws
from summer4.epi.calibration.workflow.evaluate import evaluate
from summer4.epi.calibration.workflow.optimize import (
    CMAES,
    AutoTune,
    Optax,
    OptimizeResult,
    optimize,
)

__all__ = [
    "AutoTune",
    "CMAES",
    "Candidates",
    "Optax",
    "OptimizeResult",
    "StageRecord",
    "evaluate",
    "lhs",
    "optimize",
    "prior_draws",
]
