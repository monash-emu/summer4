"""Solver seam shared by Euler and diffrax backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from summer4.results.result import SolverInfo

# Adaptive solvers may take many microsteps per unit of ``dt``; headroom
# covers stiff contact models without a magic fixed ceiling of 4096.
_DEFAULT_MAX_STEPS_HEADROOM = 64


def default_max_steps(t0: float, t1: float, dt: float) -> int:
    """Derive a step ceiling from the integration span and ``dt``."""
    span = abs(float(t1) - float(t0))
    step = abs(float(dt))
    if step == 0.0:
        raise ValueError("dt must be non-zero when deriving max_steps.")
    nominal = max(1, math.ceil(span / step))
    return max(4096, nominal * _DEFAULT_MAX_STEPS_HEADROOM)


@dataclass(frozen=True, slots=True)
class SolveSpec:
    """Integration window and step / tolerance controls."""

    t0: float
    t1: float | None
    steps: int | None
    dt: float
    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None
    dense: bool = False
    throw: bool | None = None


@dataclass(frozen=True, slots=True)
class SolveOutput:
    """Saved arrays, optional solver stats, and optional dense interpolant."""

    saved: dict[str, Any]
    stats: SolverInfo | None
    dense: Any | None = None
