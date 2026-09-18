"""Solver seam shared by Euler and diffrax backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from summer4.results.result import SolverInfo


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


@dataclass(frozen=True, slots=True)
class SolveOutput:
    """Saved arrays, optional solver stats, and optional dense interpolant."""

    saved: dict[str, Any]
    stats: SolverInfo | None
    dense: Any | None = None
