"""Solver seam: the integration window, backend protocol, and backend output."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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
    """The integration window every backend receives.

    Build one with :meth:`window`, which validates the arguments ``run`` takes.
    Step-size control, adjoints and events belong to the backend
    (:class:`~summer4.solvers.Diffrax`), not here.
    """

    t0: float
    t1: float | None
    steps: int | None
    dt: float
    max_steps: int | None = None
    dense: bool = False
    throw: bool | None = None

    @classmethod
    def window(
        cls,
        *,
        t0: float,
        dt: float,
        t1: float | None = None,
        steps: int | None = None,
        max_steps: int | None = None,
        throw: bool | None = None,
        dense: bool = False,
    ) -> SolveSpec:
        """Validate a window: exactly one of ``t1`` / ``steps``; ``dt > 0``."""
        if (t1 is None) == (steps is None):
            raise ValueError("Provide exactly one of t1 or steps.")
        if steps is None:
            assert t1 is not None
            if dt <= 0:
                raise ValueError(f"dt must be > 0, got {dt}.")
            steps = int(round((float(t1) - float(t0)) / float(dt)))
            if steps < 0:
                raise ValueError("t1 must be >= t0.")
        return cls(
            t0=float(t0),
            t1=float(t1) if t1 is not None else None,
            steps=int(steps),
            dt=float(dt),
            max_steps=max_steps,
            dense=bool(dense),
            throw=throw,
        )

    @property
    def end(self) -> float:
        """The end of the window: ``t1``, or ``t0 + dt * steps``."""
        if self.t1 is not None:
            return float(self.t1)
        if self.steps is None:
            raise ValueError("SolveSpec needs t1 or steps.")
        return float(self.t0) + float(self.dt) * int(self.steps)

    def default_ts(self) -> NDArray[np.float64]:
        """The default save grid: ``t0`` and every step endpoint."""
        if self.steps is None:
            raise ValueError("SolveSpec needs steps for a default save grid.")
        return self.t0 + self.dt * np.arange(int(self.steps) + 1, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SolveOutput:
    """What a backend returns: saved arrays, stats, dense interpolant, end state.

    ``final_state`` is the raw state array (not yet wrapped) and
    ``final_time`` the time where integration ended.
    """

    saved: dict[str, Any]
    stats: SolverInfo | None
    dense: Any | None = None
    final_time: Any | None = None
    final_state: Any | None = None


@runtime_checkable
class SolverBackend(Protocol):
    """Anything :meth:`~summer4.flows.compiled.CompiledModel.run` can integrate with.

    :class:`~summer4.solvers.Euler` and :class:`~summer4.solvers.Diffrax` are
    the built-in backends. A backend receives the compiled model, the initial
    state, prepared params, the :class:`SolveSpec` window and the expanded
    :class:`~summer4.results.plan.SavePlan`, and returns a :class:`SolveOutput`
    keyed by the plan's request names.
    """

    @property
    def name(self) -> str:
        """Display name recorded on :attr:`SolverInfo.solver`."""
        ...

    def solve(
        self,
        model: Any,
        *,
        y0: object,
        params: object,
        spec: SolveSpec,
        plan: Any,
    ) -> SolveOutput:
        """Integrate ``model`` over ``spec`` and save what ``plan`` requests."""
        ...


__all__ = [
    "SolveOutput",
    "SolveSpec",
    "SolverBackend",
    "default_max_steps",
]
