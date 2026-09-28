"""Warm up until good: grow numpyro's warmup until the adapted sampler passes a check.

numpyro's NUTS adapts step size and mass matrix on a schedule fixed by
``num_warmup``, and it is sensitive to that number: too short and chains end
warmup in different places with different step sizes. :func:`warmup_until`
takes the caller's ``num_warmup -> MCMC`` factory, runs ``MCMC.warmup`` in
rounds, checks each round, and on a failure doubles ``num_warmup`` and warms a
fresh ``MCMC`` up from where each chain ended (adaptation restarts with longer
windows; positions carry over). numpyro cannot re-warm one ``MCMC`` object, and
a new warmup length recompiles regardless, hence the factory. The passing
round's ``MCMC`` is left warmed up, so :func:`sample_until` continues from it
with no further warmup.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from summer4.epi.calibration.workflow.mcmc import (
    _PACKAGE_DIR,
    _check_ensemble_init,
    _require_numpyro,
    _require_pandas,
)

WarmupCheck = Callable[[Mapping[str, Any]], Sequence[str]]
"""Called with the latest :attr:`WarmupRun.progress` row; returns the names of
failed criteria (empty means the warmup is good)."""


@dataclass(frozen=True, slots=True)
class WarmupRule:
    """NUTS-oriented warmup check (``None`` disables a criterion).

    Every statistic comes from the **second half** of the round's warmup draws,
    where step size and mass matrix have mostly settled. A criterion with no
    data for the kernel (step size, acceptance, divergences and tree depth
    exist only for HMC/NUTS) is skipped, so other kernels are checked on R-hat
    alone.

    - ``min_warmup``: numpyro only adapts a mass matrix once warmup is long
      enough for its windows; shorter rounds always fail.
    - ``rhat``: split R-hat across chains; above it, chains ended warmup in
      different places. The default is the classic 1.1, looser than a sampling
      stop rule: warmup draws are not stationary (the step size is still
      adapting), so 1.05 over-extends warmup.
    - ``max_step_size_ratio``: largest over smallest adapted step size across
      chains; chains that adapted very different step sizes saw different
      geometry.
    - ``accept_tolerance``: mean acceptance may fall at most this far below the
      kernel's ``target_accept_prob``.
    - ``max_divergence_frac``: divergent transitions allowed.
    - ``max_treedepth_frac``: trajectories allowed to hit ``max_tree_depth``.
    """

    min_warmup: int = 100
    rhat: float | None = 1.1
    max_step_size_ratio: float | None = 3.0
    accept_tolerance: float | None = 0.15
    max_divergence_frac: float | None = 0.01
    max_treedepth_frac: float | None = 0.05

    def __call__(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        failed: list[str] = []
        if int(row["num_warmup"]) < int(self.min_warmup):
            failed.append("min_warmup")
        # NaN (too few draws for split R-hat) fails as well.
        if self.rhat is not None and not float(row["rhat_max"]) <= float(self.rhat):
            failed.append("rhat")
        if _exceeds(row.get("step_size_ratio"), self.max_step_size_ratio):
            failed.append("step_size_ratio")
        accept, target = row.get("accept_mean"), row.get("target_accept_prob")
        if (
            self.accept_tolerance is not None
            and accept is not None
            and target is not None
            and float(accept) < float(target) - float(self.accept_tolerance)
        ):
            failed.append("accept_prob")
        if _exceeds(row.get("divergence_frac"), self.max_divergence_frac):
            failed.append("divergences")
        if _exceeds(row.get("treedepth_frac"), self.max_treedepth_frac):
            failed.append("treedepth")
        return tuple(failed)


def _exceeds(value: Any, limit: float | None) -> bool:
    return limit is not None and value is not None and float(value) > float(limit)


class WarmupRun:
    """Record of :func:`warmup_until`: one progress row per warmup round.

    ``mcmc`` is the last round's ``MCMC`` from the caller's factory, left warmed
    up (``post_warmup_state`` set); hand it to :func:`sample_until`. ``samples``
    holds that round's warmup draws, ``(chain, draw)`` per site.
    """

    def __init__(self) -> None:
        self.mcmc: Any = None
        self.samples: dict[str, np.ndarray] = {}
        self.converged = False
        self.reason = "not started"
        self._rows: list[dict[str, Any]] = []

    @property
    def rounds(self) -> int:
        """Number of warmup rounds run."""
        return len(self._rows)

    @property
    def num_warmup(self) -> int:
        """Warmup length of the last round."""
        return int(self._rows[-1]["num_warmup"]) if self._rows else 0

    @property
    def progress(self) -> Any:
        """One row per round: its length, the checked statistics, and what failed."""
        pd = _require_pandas()
        return pd.DataFrame(self._rows).set_index("round")


def _tail_rhat(tail: Mapping[str, np.ndarray]) -> float:
    from numpyro.diagnostics import split_gelman_rubin  # type: ignore[import-untyped]

    worst = -math.inf
    for arr in tail.values():
        x = np.asarray(arr)
        if x.shape[1] < 4:
            return math.nan
        x = x.reshape(x.shape[0], x.shape[1], -1)
        worst = max(worst, float(np.max(split_gelman_rubin(x))))
    return worst


def _hmc_settings(sampler: Any) -> tuple[float | None, int | None]:
    # numpyro exposes no public accessor for these kernel settings.
    target = getattr(sampler, "_target_accept_prob", None)
    depth = getattr(sampler, "_max_tree_depth", None)
    if isinstance(depth, tuple):
        depth = depth[0]  # (warmup, sampling) depths; this is warmup
    return (
        None if target is None else float(target),
        None if depth is None else int(depth),
    )


def warmup_until(
    make_mcmc: Callable[[int], Any],
    num_warmup: int,
    check: WarmupCheck | None = None,
    *,
    init_params: Mapping[str, Any] | None = None,
    seed: int = 0,
    growth: float = 2.0,
    max_warmup: int = 4000,
    warn: bool = True,
) -> WarmupRun:
    """Warm up in rounds of growing length until ``check`` passes.

    ``make_mcmc(n)`` builds the caller's ``numpyro.infer.MCMC`` with
    ``num_warmup=n``, for example
    ``lambda n: MCMC(NUTS(bm.numpyro_model()), num_warmup=n, num_samples=200, num_chains=4)``.
    Round one uses ``num_warmup``. After each round the warmup draws are checked
    (default :class:`WarmupRule`); on a failure the length is multiplied by
    ``growth`` and a fresh ``MCMC`` warms up again, each chain starting from where
    the previous round left it. Stops on the first passing round, or when the
    next round would exceed ``max_warmup`` (warns; the last round's ``MCMC`` is
    still warmed up, so sampling can proceed).

    ``check`` is any callable returning the names of failed criteria. Afterwards
    call ``sample_until(warm.mcmc, stop)`` without ``init_params``.
    """
    from jax import random
    from numpyro.infer import HMC  # type: ignore[import-untyped]

    _require_numpyro()
    n = int(num_warmup)
    if n < 1:
        raise ValueError(f"warmup_until needs num_warmup >= 1, got {n}.")
    if float(growth) <= 1.0:
        raise ValueError(f"warmup_until growth must exceed 1, got {growth}.")
    rule = check if check is not None else WarmupRule()
    run = WarmupRun()
    key = random.PRNGKey(int(seed))
    start = time.perf_counter()

    while True:
        mcmc = make_mcmc(n)
        if int(mcmc.num_warmup) != n or mcmc.post_warmup_state is not None:
            raise ValueError(
                f"make_mcmc({n}) must return a fresh MCMC with num_warmup={n}; got "
                f"num_warmup={mcmc.num_warmup}"
                + (" already warmed up." if mcmc.post_warmup_state is not None else ".")
            )
        _check_ensemble_init(mcmc, init_params)
        hmc = isinstance(mcmc.sampler, HMC)
        fields = ("accept_prob", "num_steps", "diverging") if hmc else ()
        target, depth = _hmc_settings(mcmc.sampler) if hmc else (None, None)

        key, sub = random.split(key)
        mcmc.warmup(sub, init_params=init_params, collect_warmup=True, extra_fields=fields)
        draws = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        half = n // 2
        row: dict[str, Any] = {
            "round": run.rounds + 1,
            "num_warmup": n,
            "rhat_max": _tail_rhat({k: v[:, half:] for k, v in draws.items()}),
            "step_size_min": None,
            "step_size_max": None,
            "step_size_ratio": None,
            "accept_mean": None,
            "target_accept_prob": target,
            "divergence_frac": None,
            "treedepth_frac": None,
        }
        state: Any = mcmc.post_warmup_state
        if hmc:
            extra = {
                k: np.asarray(v)[:, half:]
                for k, v in mcmc.get_extra_fields(group_by_chain=True).items()
            }
            step = np.atleast_1d(np.asarray(state.adapt_state.step_size))
            row.update(
                step_size_min=float(step.min()),
                step_size_max=float(step.max()),
                step_size_ratio=float(step.max() / step.min()),
                accept_mean=float(np.mean(extra["accept_prob"])),
                divergence_frac=float(np.mean(extra["diverging"])),
                treedepth_frac=(
                    None if depth is None else float(np.mean(extra["num_steps"] >= 2**depth - 1))
                ),
            )
        row["seconds"] = time.perf_counter() - start
        failed = tuple(rule(dict(row)))
        row["failed"] = ", ".join(failed) if failed else None
        run._rows.append(row)
        run.mcmc, run.samples = mcmc, draws

        if not failed:
            run.converged, run.reason = True, "passed"
            break
        grown = int(math.ceil(n * float(growth)))
        if grown > int(max_warmup):
            run.converged, run.reason = False, "max_warmup"
            break
        if hmc:
            init_params = {k: np.asarray(v) for k, v in state.z.items()}
        n = grown

    if warn and not run.converged:
        warnings.warn(
            f"Warmup did not pass its check within max_warmup={max_warmup} "
            f"(last round num_warmup={run.num_warmup}, failed: {run._rows[-1]['failed']}).",
            RuntimeWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )
    return run


__all__ = [
    "WarmupCheck",
    "WarmupRule",
    "WarmupRun",
    "warmup_until",
]
