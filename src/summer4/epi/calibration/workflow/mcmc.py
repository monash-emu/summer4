"""Chunked MCMC with an automated stop (step 27).

The sampler is plain numpyro: build ``numpyro.infer.MCMC`` around any kernel,
seed it with :meth:`Candidates.init_params`, and hand it to
:func:`sample_until`. summer4 owns only the chunk loop, the stop decision and
the per-chunk record. :func:`run_mcmc` is the one-call convenience built from
those pieces.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord

# Warnings point at the caller's line, whichever public entry point was used.
_PACKAGE_DIR = str(Path(__file__).resolve().parent)

Decision = tuple[bool, str]
"""``(converged, reason)`` returned by a stop callable when sampling should end."""

Stop = Callable[[Mapping[str, Any]], Decision | None]
"""Called with the latest :attr:`MCMCRun.progress` row; ``None`` means keep going."""


def _require_numpyro() -> Any:
    try:
        import numpyro  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wf.sample_until requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return numpyro


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MCMCRun.progress requires pandas: pip install summer4[pandas]") from exc
    return pd


@dataclass(frozen=True, slots=True)
class StopRule:
    """Default stop callable for :func:`sample_until` (``None`` disables a criterion).

    Checked after every chunk, in this order: too many divergences ends the run
    unconverged (``"divergences"``); R-hat and ESS both passing ends it converged
    (``"diagnostics"``); otherwise a spent draw or time budget ends it
    unconverged (``"max_samples"`` / ``"max_seconds"``).
    """

    rhat: float | None = 1.05
    ess: float | None = 100.0
    max_divergence_frac: float | None = None
    max_samples: int = 5000
    max_seconds: float | None = None

    def __call__(self, row: Mapping[str, Any]) -> Decision | None:
        divergence = row.get("divergence_frac")
        if (
            self.max_divergence_frac is not None
            and divergence is not None
            and float(divergence) > float(self.max_divergence_frac)
        ):
            return False, "divergences"
        rhat_ok = self.rhat is None or float(row["rhat_max"]) <= float(self.rhat)
        ess_ok = self.ess is None or float(row["ess_bulk_min"]) >= float(self.ess)
        if rhat_ok and ess_ok:
            return True, "diagnostics"
        if int(row["samples"]) >= int(self.max_samples):
            return False, "max_samples"
        seconds = row.get("seconds")
        if (
            self.max_seconds is not None
            and seconds is not None
            and float(seconds) >= float(self.max_seconds)
        ):
            return False, "max_seconds"
        return None


def _site_diagnostics(samples: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    from numpyro.diagnostics import (  # type: ignore[import-untyped]
        effective_sample_size,
        split_gelman_rubin,
    )

    out: dict[str, tuple[float, float]] = {}
    for name, arr in samples.items():
        x = np.asarray(arr)
        if x.ndim > 2:
            # (chain, draw, ...) — one row per site, averaged over trailing dims
            x = x.reshape(x.shape[0], x.shape[1], -1)
            out[name] = (
                float(np.mean(split_gelman_rubin(x))),
                float(np.mean(effective_sample_size(x))),
            )
        else:
            out[name] = (float(split_gelman_rubin(x)), float(effective_sample_size(x)))
    return out


def _progress_row(
    samples: Mapping[str, np.ndarray], diverging: np.ndarray | None
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    site_rows = _site_diagnostics(samples)
    row = {
        "samples": int(np.shape(next(iter(samples.values())))[1]),
        "rhat_max": max(v[0] for v in site_rows.values()),
        "ess_bulk_min": min(v[1] for v in site_rows.values()),
        "divergence_frac": None if diverging is None else float(np.mean(diverging)),
    }
    return row, site_rows


def replay(
    stop: Stop,
    samples: Mapping[str, Any],
    chunk_samples: int,
    *,
    diverging: Any | None = None,
) -> Any:
    """Apply ``stop`` to existing draws as :func:`sample_until` would have, chunk by chunk.

    ``samples`` maps each site to a ``(chain, draw, ...)`` array, for example
    ``mcmc.get_samples(group_by_chain=True)`` or :attr:`MCMCRun.samples`
    restricted to some chains. Returns the progress table up to and including
    the chunk where ``stop`` decided (``seconds`` is ``None``: nothing is
    timed). No sampling, no warnings — use it to ask what another rule would
    have done with the same draws.
    """
    pd = _require_pandas()
    chunk = int(chunk_samples)
    arrays = {k: np.asarray(v) for k, v in samples.items()}
    div = None if diverging is None else np.asarray(diverging)
    total = int(np.shape(next(iter(arrays.values())))[1])
    rows: list[dict[str, Any]] = []
    for k, end in enumerate(range(chunk, total + 1, chunk), start=1):
        row, _ = _progress_row(
            {name: a[:, :end] for name, a in arrays.items()},
            None if div is None else div[:, :end],
        )
        row.update(chunk=k, seconds=None, decision=None, converged=None)
        decision = stop(dict(row))
        rows.append(row)
        if decision is not None:
            row["converged"], row["decision"] = bool(decision[0]), str(decision[1])
            break
    return pd.DataFrame(rows).set_index("chunk")


def _records_divergences(mcmc: Any) -> bool:
    from numpyro.infer import HMC  # type: ignore[import-untyped]

    return isinstance(mcmc.sampler, HMC)


def _check_ensemble_init(mcmc: Any, init_params: Mapping[str, Any] | None) -> None:
    from numpyro.infer.ensemble import EnsembleSampler  # type: ignore[import-untyped]

    if init_params is None or not isinstance(mcmc.sampler, EnsembleSampler):
        return
    flat = np.concatenate(
        [np.asarray(v).reshape(np.shape(v)[0], -1) for v in init_params.values()], axis=1
    )
    if len({tuple(np.round(r, decimals=8)) for r in flat}) < flat.shape[0]:
        raise ValueError(
            f"{type(mcmc.sampler).__name__} needs distinct walkers; init_params repeats a "
            "row (pass jitter > 0 to Candidates.init_params)."
        )


class MCMCRun:
    """A resumable chunked MCMC run: the numpyro ``MCMC`` plus its per-chunk record.

    Built by :func:`sample_until`. ``mcmc`` is the caller's own object, so its
    kernel, adapted step size and last state are all reachable. :meth:`extend`
    samples further chunks from where the run stopped.
    """

    def __init__(self, mcmc: Any) -> None:
        self.mcmc = mcmc
        self.sites: tuple[str, ...] = ()
        self.samples: dict[str, np.ndarray] = {}
        self.diverging: np.ndarray | None = None
        self.converged = False
        self.reason = "not started"
        self._rows: list[dict[str, Any]] = []
        self._site_rows: dict[str, tuple[float, float]] = {}
        self._seconds = 0.0
        self._divergences = _records_divergences(mcmc)

    @property
    def chunks(self) -> int:
        """Number of chunks sampled so far."""
        return len(self._rows)

    @property
    def progress(self) -> Any:
        """One row per chunk: draws per chain, worst R-hat, smallest ESS, divergences, seconds.

        Diagnostics cover every draw so far, not just that chunk. ``decision`` and
        ``converged`` are filled on each chunk that ended a call, else ``None``.
        """
        pd = _require_pandas()
        return pd.DataFrame(self._rows).set_index("chunk")

    @property
    def diagnostics(self) -> Any:
        """Split R-hat and bulk ESS per site over every draw so far."""
        pd = _require_pandas()
        frame = pd.DataFrame(
            [{"site": k, "rhat": v[0], "ess_bulk": v[1]} for k, v in self._site_rows.items()]
        )
        return frame.set_index("site")

    @property
    def idata(self) -> Any:
        """Every post-warmup draw as ``arviz.InferenceData``."""
        try:
            import arviz as az  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MCMCRun.idata requires arviz (calibration extra): pip install summer4[calibration]"
            ) from exc
        groups: dict[str, Any] = {"posterior": dict(self.samples)}
        if self.diverging is not None:
            groups["sample_stats"] = {"diverging": self.diverging}
        return az.from_dict(groups)

    def candidates(self, bm: Any) -> Candidates:
        """Each chain's latest draw as :class:`Candidates`, for re-seeding another stage."""
        last = {name: np.asarray(self.samples[name])[:, -1] for name in self.sites}
        record = StageRecord(
            stage="sample_until",
            settings={"chunks": self.chunks, "converged": self.converged, "reason": self.reason},
            seconds=self._seconds,
        )
        base = Candidates.from_params(bm, last)
        return Candidates(sites=base.sites, z=base.z, params=base.params, history=(record,))

    def extend(self, stop: Stop, *, warn: bool = True) -> MCMCRun:
        """Sample at least one more chunk from the last state, until ``stop`` decides.

        Mutates and returns ``self``. ``stop`` sees cumulative totals, so a
        ``StopRule(max_samples=...)`` budget counts draws from earlier calls.
        """
        while True:
            self.mcmc.post_warmup_state = self.mcmc.last_state
            self._sample(self.mcmc.post_warmup_state.rng_key, init_params=None)
            if self._decide(stop, warn=warn):
                return self

    def _sample(self, rng_key: Any, *, init_params: Any) -> None:
        extra = ("diverging",) if self._divergences else ()
        start = time.perf_counter()
        self.mcmc.run(rng_key, init_params=init_params, extra_fields=extra)
        chunk = self.mcmc.get_samples(group_by_chain=True)
        if not self.sites:
            self.sites = tuple(chunk)
        for name in self.sites:
            new = np.asarray(chunk[name])
            old = self.samples.get(name)
            self.samples[name] = new if old is None else np.concatenate([old, new], axis=1)
        if self._divergences:
            div = np.asarray(self.mcmc.get_extra_fields(group_by_chain=True)["diverging"])
            self.diverging = (
                div if self.diverging is None else np.concatenate([self.diverging, div], axis=1)
            )
        self._seconds += time.perf_counter() - start
        row, self._site_rows = _progress_row(
            {k: self.samples[k] for k in self.sites}, self.diverging
        )
        row.update(chunk=len(self._rows) + 1, seconds=self._seconds, decision=None, converged=None)
        self._rows.append(row)

    def _decide(self, stop: Stop, *, warn: bool) -> bool:
        row = self._rows[-1]
        decision = stop(dict(row))
        if decision is None:
            return False
        self.converged, self.reason = bool(decision[0]), str(decision[1])
        row["decision"], row["converged"] = self.reason, self.converged
        if warn and not self.converged:
            warnings.warn(
                f"MCMC stopped without converging (reason={self.reason!r}, "
                f"chunks={self.chunks}).",
                RuntimeWarning,
                skip_file_prefixes=(_PACKAGE_DIR,),
            )
        return True


def sample_until(
    mcmc: Any,
    stop: Stop,
    *,
    init_params: Mapping[str, Any] | None = None,
    seed: int = 0,
    warn: bool = True,
) -> MCMCRun:
    """Run a numpyro ``MCMC`` in chunks until ``stop`` decides.

    ``mcmc`` is built by the caller around any kernel; its ``num_warmup`` is the
    warmup and its ``num_samples`` is the chunk size. After each chunk, split
    R-hat and bulk ESS are recomputed over every draw so far (with
    ``numpyro.diagnostics``) and passed to ``stop`` as one :attr:`MCMCRun.progress`
    row. ``stop`` is a :class:`StopRule` or any callable returning
    ``(converged, reason)`` to end the run, or ``None`` to continue.
    Non-convergence warns rather than raises.

    ``init_params`` is typically ``candidates.init_params(num_chains)``; ``None``
    keeps numpyro's default initialisation. If ``mcmc`` has already been warmed
    up (``mcmc.post_warmup_state`` is set, for example by :func:`warmup_until`),
    sampling continues from that state with no further warmup, and
    ``init_params`` must be ``None``.
    """
    from jax import random

    _require_numpyro()
    if init_params is not None and mcmc.post_warmup_state is not None:
        raise ValueError(
            "sample_until got init_params for an MCMC that is already warmed up; the chains "
            "continue from mcmc.post_warmup_state. Pass init_params to the warmup instead."
        )
    _check_ensemble_init(mcmc, init_params)
    run = MCMCRun(mcmc)
    run._sample(random.PRNGKey(int(seed)), init_params=init_params)
    if run._decide(stop, warn=warn):
        return run
    return run.extend(stop, warn=warn)


def run_mcmc(
    bm: Any,
    init: Candidates | None = None,
    *,
    make_kernel: Callable[[], Any] | None = None,
    num_chains: int = 4,
    num_warmup: int = 200,
    chunk_samples: int = 200,
    stop: Stop | None = None,
    warmup: Any = None,
    jitter: float = 0.05,
    seed: int = 0,
    chain_method: str = "vectorized",
    progress_bar: bool = False,
    warn: bool = True,
) -> MCMCRun:
    """Seeded chunked MCMC in one call. Exactly equivalent to::

        def make_mcmc(n):
            return MCMC(
                make_kernel() if make_kernel else NUTS(bm.numpyro_model()),
                num_warmup=n, num_samples=chunk_samples,
                num_chains=num_chains, chain_method=chain_method, progress_bar=progress_bar,
            )

        init_params = None if init is None else init.init_params(
            num_chains, jitter=jitter, seed=seed
        )
        if warmup is None:
            mcmc = make_mcmc(num_warmup)
        else:  # e.g. warmup=WarmupRule(): grow num_warmup until good
            mcmc = warmup_until(make_mcmc, num_warmup, warmup, init_params=init_params,
                                seed=seed).mcmc
            init_params = None
        return sample_until(mcmc, stop or StopRule(), init_params=init_params, seed=seed)

    Write those lines yourself to use another kernel's options, inspect the
    ``MCMC`` object, or retry with a new kernel (for example
    ``NUTS(bm.numpyro_model(), target_accept_prob=0.95)`` after a
    ``"divergences"`` stop).
    """
    _require_numpyro()
    from numpyro.infer import MCMC, NUTS

    def make_mcmc(n: int) -> Any:
        return MCMC(
            make_kernel() if make_kernel is not None else NUTS(bm.numpyro_model()),
            num_warmup=int(n),
            num_samples=int(chunk_samples),
            num_chains=int(num_chains),
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

    init_params = (
        None if init is None else init.init_params(int(num_chains), jitter=jitter, seed=seed)
    )
    if warmup is None:
        mcmc = make_mcmc(int(num_warmup))
    else:
        from summer4.epi.calibration.workflow.warmup import warmup_until

        mcmc = warmup_until(
            make_mcmc, int(num_warmup), warmup, init_params=init_params, seed=seed, warn=warn
        ).mcmc
        init_params = None
    return sample_until(
        mcmc,
        stop if stop is not None else StopRule(),
        init_params=init_params,
        seed=seed,
        warn=warn,
    )


__all__ = [
    "Decision",
    "MCMCRun",
    "Stop",
    "StopRule",
    "replay",
    "run_mcmc",
    "sample_until",
]
