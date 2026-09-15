"""Declarative calibration targets that merge observation times into a save plan.

This module ships gathering and residuals only. Probabilistic likelihoods and
the priors that :attr:`Target.dispersion` feeds are WP10; the field is declared
here and consumed there.

User-facing code is the same under both solvers:

- **diffrax** — :meth:`Target.contribute` merges observation times into the
  request's ``ts``, so the solver hits them exactly and
  :meth:`~summer4.results.trace.Trace.at_times` is a no-op gather;
- **Euler** — the model saves on its grid and
  :meth:`~summer4.time.TimeAxis.weights_for` supplies the precomputed
  ``(idx, weight)`` pair that ``at_times`` already applies.

**Euler cost.** Merging off-grid times into a group's ``ts`` breaks the
arithmetic-subgrid fast path, so Euler takes the general ``lax.scan`` over
every step with the whole trajectory stacked, then a lerp. Correct, but
heavier than the nested-scan path — prefer diffrax when calibrating off-grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from summer4.results.plan import Quantity, SavePlan, SaveRequest
from summer4.results.result import Result
from summer4.results.trace import Trace
from summer4.time import Epoch


def _as_float64(ts: NDArray[Any] | list[float]) -> NDArray[np.float64]:
    return np.asarray(ts, dtype=np.float64).reshape(-1)


def _merge_times(
    existing: NDArray[np.float64] | None,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sorted unique union of ``existing`` and ``times``."""
    if existing is None:
        return np.unique(_as_float64(times))
    return np.unique(np.concatenate([_as_float64(existing), _as_float64(times)]))


def _replace_request(plan: SavePlan, key: str, request: SaveRequest) -> SavePlan:
    requests = dict(plan.requests)
    requests[key] = request
    return SavePlan(
        requests=requests,
        ts=plan.ts,
        dense=plan.dense,
        solver_stats=plan.solver_stats,
    )


def _trace_array(trace: Trace) -> Any:
    values = trace.values
    return values.data if hasattr(values, "data") else values


@dataclass(frozen=True, slots=True)
class Target:
    """One observed series aligned to a named save-plan key.

    ``dispersion`` names a nuisance parameter for WP10; unused here.
    """

    key: str
    times: NDArray[np.float64]
    values: NDArray[np.float64]
    quantity: Quantity | None = None
    dispersion: str | None = None

    def __post_init__(self) -> None:
        times = _as_float64(self.times)
        values = _as_float64(self.values)
        if times.shape != values.shape:
            raise ValueError(
                f"Target {self.key!r}: times shape {times.shape} != values shape {values.shape}."
            )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_series(
        cls,
        key: str,
        s: Any,
        epoch: Epoch,
        *,
        quantity: Quantity | None = None,
        dispersion: str | None = None,
    ) -> Target:
        """Build a target from a pandas Series with a ``DatetimeIndex``.

        Pandas is imported lazily so it is never a hard dependency of
        ``summer4.results``.
        """
        import pandas as pd  # type: ignore[import-untyped]

        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(
                f"Target.from_series expects a DatetimeIndex, got {type(s.index).__name__}."
            )
        times = epoch.to_model(s.index.to_numpy())
        values = np.asarray(s.to_numpy(), dtype=np.float64)
        return cls(
            key=key,
            times=times,
            values=values,
            quantity=quantity,
            dispersion=dispersion,
        )

    def contribute(self, plan: SavePlan) -> SavePlan:
        """Merge this target's times into ``plan`` for :attr:`key`.

        When ``key`` is absent, adds ``SaveRequest(quantity, ts=self.times)``.
        When ``key`` is absent and no ``quantity`` was given, raises with the
        plan's known keys.
        """
        if self.key not in plan.requests:
            if self.quantity is None:
                known = sorted(plan.requests)
                raise KeyError(
                    f"Target key {self.key!r} is not in the save plan and no "
                    f"quantity was given. Known keys: {known}."
                )
            return _replace_request(
                plan,
                self.key,
                SaveRequest(self.quantity, ts=np.asarray(self.times, dtype=np.float64)),
            )

        existing = plan.requests[self.key]
        merged = _merge_times(existing.ts, self.times)
        return _replace_request(
            plan,
            self.key,
            SaveRequest(existing.what, ts=merged),
        )


@dataclass(frozen=True, slots=True)
class TargetSet:
    """A small static collection of :class:`Target` objects for one likelihood run."""

    targets: tuple[Target, ...]

    def plan(self, base: SavePlan) -> SavePlan:
        """Fold every target's :meth:`Target.contribute` over ``base``."""
        plan = base
        for target in self.targets:
            plan = target.contribute(plan)
        return plan

    def gather(self, result: Result) -> dict[str, Trace]:
        """Interpolate each target's key onto its observation times."""
        return {t.key: result[t.key].at_times(t.times) for t in self.targets}

    def residuals(self, result: Result) -> dict[str, Any]:
        """Return ``prediction - observation`` arrays keyed by target key.

        Predictions are flattened to match :attr:`Target.values` (so a
        single-compartment save of shape ``(T, 1)`` lines up with a length-``T``
        series).
        """
        out: dict[str, Any] = {}
        for target in self.targets:
            pred = _trace_array(result[target.key].at_times(target.times))
            # Prefer reshape to the observation shape; fall back to flat.
            try:
                aligned = pred.reshape(target.values.shape)
            except Exception:
                aligned = pred.reshape(-1)
                if aligned.shape != target.values.shape:
                    raise ValueError(
                        f"Target {target.key!r}: prediction shape {np.shape(pred)} "
                        f"incompatible with values shape {target.values.shape}."
                    ) from None
            out[target.key] = aligned - target.values
        return out

    def keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.targets)
