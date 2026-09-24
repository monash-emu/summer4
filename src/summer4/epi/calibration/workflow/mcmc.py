"""Seeded MCMC with automated run length (step 27)."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import random

from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord

SampleKind = Literal["nuts", "aies", "ess", "sa"]


def _require_numpyro() -> Any:
    try:
        import numpyro  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wf.run_mcmc requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return numpyro


def _require_arviz() -> Any:
    try:
        import arviz  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wf.run_mcmc requires arviz (calibration extra): pip install summer4[calibration]"
        ) from exc
    return arviz


@dataclass(frozen=True, slots=True)
class StopRule:
    """Stopping criteria for chunked MCMC (``None`` disables a criterion)."""

    rhat: float | None = 1.05
    ess: float | None = 100.0
    max_divergence_frac: float | None = None
    max_samples: int = 5000
    max_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Outcome of :func:`run_mcmc`."""

    idata: Any
    diagnostics: Any  # pandas DataFrame
    converged: bool
    reason: str
    chunks: int
    candidates: Candidates
    history: tuple[StageRecord, ...] = ()


def _kernel_cls(kind: str) -> Any:
    from numpyro.infer import AIES, ESS, NUTS, SA  # type: ignore[import-untyped]

    kernels = {"nuts": NUTS, "aies": AIES, "ess": ESS, "sa": SA}
    if kind not in kernels:
        raise ValueError(f"Unknown sample kind {kind!r}; expected one of {sorted(kernels)}.")
    return kernels[kind]


def _seed_init_params(
    init: Candidates | None,
    *,
    sites: tuple[str, ...],
    num_chains: int,
    jitter: float,
    seed: int,
    kind: str,
) -> dict[str, Any] | None:
    if init is None:
        return None
    if len(init) == 0:
        raise ValueError("run_mcmc init Candidates is empty.")
    rng = np.random.default_rng(int(seed) + 17)
    stacked: dict[str, list[Any]] = {name: [] for name in sites}
    for c in range(int(num_chains)):
        i = c % len(init)
        for name in sites:
            z = np.asarray(init.z[name][i], dtype=np.float64)
            if float(jitter) != 0.0:
                z = z + float(jitter) * rng.normal(size=np.shape(z))
            stacked[name].append(z)
    out = {name: jnp.asarray(np.stack(vals, axis=0)) for name, vals in stacked.items()}
    if kind in ("aies", "ess") and float(jitter) == 0.0:
        # Distinct unconstrained starts required for ensemble walkers.
        for name in sites:
            rows = np.asarray(out[name])
            # Compare flattened rows
            flat = rows.reshape(rows.shape[0], -1)
            if len({tuple(np.round(r, decimals=8)) for r in flat}) < num_chains:
                raise ValueError(
                    f"{kind} requires distinct init walkers; got duplicates with jitter=0."
                )
    return out


def _concat_chain_draw(acc: dict[str, Any] | None, chunk: dict[str, Any]) -> dict[str, Any]:
    if acc is None:
        return {k: np.asarray(v) for k, v in chunk.items()}
    return {k: np.concatenate([acc[k], np.asarray(chunk[k])], axis=1) for k in chunk}


def _diagnostics_frame(samples: dict[str, Any], diverging: Any | None) -> Any:
    from numpyro.diagnostics import (  # type: ignore[import-untyped]
        effective_sample_size,
        split_gelman_rubin,
    )

    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SampleResult.diagnostics requires pandas: pip install summer4[pandas]"
        ) from exc

    rows: list[dict[str, Any]] = []
    for name, arr in samples.items():
        x = np.asarray(arr)
        # (chain, draw, ...) — collapse trailing dims for scalar diagnostics
        if x.ndim > 2:
            x = x.reshape(x.shape[0], x.shape[1], -1)
            # report mean over trailing; still one row per site
            rhat = float(np.mean(split_gelman_rubin(x)))
            ess = float(np.mean(effective_sample_size(x)))
        else:
            rhat = float(split_gelman_rubin(x))
            ess = float(effective_sample_size(x))
        rows.append({"site": name, "rhat": rhat, "ess_bulk": ess, "ess_tail": ess})
    frame = pd.DataFrame(rows).set_index("site")
    if diverging is not None:
        d = np.asarray(diverging)
        frame.attrs["divergence_frac"] = float(np.mean(d)) if d.size else 0.0
    return frame


def _diagnostics_pass(stop: StopRule, frame: Any, diverging: Any | None) -> bool:
    if stop.rhat is not None and float(np.max(np.asarray(frame["rhat"]))) > float(stop.rhat):
        return False
    if stop.ess is not None and float(np.min(np.asarray(frame["ess_bulk"]))) < float(stop.ess):
        return False
    return not (
        stop.max_divergence_frac is not None
        and diverging is not None
        and float(np.mean(np.asarray(diverging))) > float(stop.max_divergence_frac)
    )


def run_mcmc(
    bm: Any,
    init: Candidates | None = None,
    *,
    kind: SampleKind | str = "nuts",
    num_chains: int = 4,
    num_warmup: int = 200,
    chunk_samples: int = 200,
    stop: StopRule | None = None,
    jitter: float = 0.05,
    seed: int = 0,
    chain_method: str = "vectorized",
    progress_bar: bool = False,
    target_accept_prob: float = 0.8,
    max_retries: int = 2,
) -> SampleResult:
    """Seeded, chunked MCMC with automated stopping.

    Builds ``numpyro.infer.MCMC`` from ``bm.numpyro_model()`` (does not call
    :meth:`BayesianModel.sample`). Diagnostics use ``numpyro.diagnostics``,
    not arviz. Non-convergence warns rather than raises.
    """
    from numpyro.infer import MCMC

    _require_numpyro()
    az = _require_arviz()
    if stop is None:
        stop = StopRule()

    kind_key = str(kind).strip().lower()
    n_chains = int(num_chains)
    if kind_key in ("aies", "ess") and n_chains < 2:
        raise ValueError(f"{kind_key} requires num_chains >= 2.")

    sites = tuple(bm.prior_names())
    init_params = _seed_init_params(
        init,
        sites=sites,
        num_chains=n_chains,
        jitter=float(jitter),
        seed=int(seed),
        kind=kind_key,
    )

    t0 = time.perf_counter()
    accept_schedule = [0.8, 0.9, 0.95]
    # Find starting index nearest to requested target_accept_prob
    start_idx = int(np.argmin(np.abs(np.asarray(accept_schedule) - float(target_accept_prob))))
    accept = float(accept_schedule[start_idx])
    warmup = int(num_warmup)
    retries = 0

    def _build_mcmc(warmup_n: int, accept_p: float) -> Any:
        kernel_kwargs: dict[str, Any] = {}
        if kind_key == "nuts":
            kernel_kwargs["target_accept_prob"] = float(accept_p)
        kernel = _kernel_cls(kind_key)(bm.numpyro_model(), **kernel_kwargs)
        return MCMC(
            kernel,
            num_warmup=int(warmup_n),
            num_samples=int(chunk_samples),
            num_chains=n_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

    mcmc = _build_mcmc(warmup, accept)
    rng = random.PRNGKey(int(seed))
    mcmc.run(rng, init_params=init_params, extra_fields=("diverging",))

    # Divergence retry on first chunk (NUTS only)
    if kind_key == "nuts" and stop.max_divergence_frac is not None:
        while retries < int(max_retries):
            extra = mcmc.get_extra_fields(group_by_chain=True)
            div = extra.get("diverging")
            if div is None:
                break
            frac = float(np.mean(np.asarray(div)))
            if frac <= float(stop.max_divergence_frac):
                break
            retries += 1
            idx = min(start_idx + retries, len(accept_schedule) - 1)
            accept = float(accept_schedule[idx])
            warmup = int(warmup * 2)
            mcmc = _build_mcmc(warmup, accept)
            mcmc.run(
                random.PRNGKey(int(seed) + retries),
                init_params=init_params,
                extra_fields=("diverging",),
            )

    acc_samples: dict[str, Any] | None = None
    acc_div: Any | None = None
    chunks = 0
    converged = False
    reason = "continuing"

    while True:
        chunks += 1
        chunk = mcmc.get_samples(group_by_chain=True)
        # Drop deterministic / non-site keys if any
        chunk = {k: v for k, v in chunk.items() if k in sites or k in bm.prior_names()}
        if not chunk:
            chunk = dict(mcmc.get_samples(group_by_chain=True))
        acc_samples = _concat_chain_draw(acc_samples, chunk)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        div = extra.get("diverging")
        if div is not None:
            acc_div = (
                np.asarray(div)
                if acc_div is None
                else np.concatenate([np.asarray(acc_div), np.asarray(div)], axis=1)
            )

        assert acc_samples is not None
        frame = _diagnostics_frame(acc_samples, acc_div)
        n_samples = int(next(iter(acc_samples.values())).shape[1])
        elapsed = time.perf_counter() - t0

        if _diagnostics_pass(stop, frame, acc_div):
            converged, reason = True, "diagnostics"
            break
        if n_samples >= int(stop.max_samples):
            converged, reason = False, "max_samples"
            break
        if stop.max_seconds is not None and elapsed >= float(stop.max_seconds):
            converged, reason = False, "max_seconds"
            break

        # Continue from last state
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("diverging",))

    if not converged:
        warnings.warn(
            f"run_mcmc did not meet StopRule (reason={reason!r}, chunks={chunks}).",
            RuntimeWarning,
            stacklevel=2,
        )

    assert acc_samples is not None
    posterior = {k: np.asarray(v) for k, v in acc_samples.items()}
    groups: dict[str, Any] = {"posterior": posterior}
    if acc_div is not None:
        groups["sample_stats"] = {"diverging": np.asarray(acc_div)}
    idata = az.from_dict(groups)

    # Last-chain unconstrained z as Candidates
    last_z = mcmc.last_state.z
    z_batch = {name: np.asarray(last_z[name]) for name in sites if name in last_z}
    # Ensure leading chain axis
    for name in list(z_batch):
        arr = z_batch[name]
        if arr.ndim == 0:
            z_batch[name] = arr.reshape(1)
    params = {k: np.asarray(v) for k, v in bm.constrain(z_batch).items()}
    cands = Candidates(sites=sites, z=z_batch, params=params)
    seconds = time.perf_counter() - t0
    record = StageRecord(
        stage="run_mcmc",
        settings={
            "kind": kind_key,
            "num_chains": n_chains,
            "num_warmup": warmup,
            "chunk_samples": int(chunk_samples),
            "chunks": chunks,
            "converged": converged,
            "reason": reason,
        },
        seconds=seconds,
    )
    return SampleResult(
        idata=idata,
        diagnostics=frame,
        converged=converged,
        reason=reason,
        chunks=chunks,
        candidates=cands,
        history=(record,),
    )


__all__ = [
    "SampleResult",
    "StopRule",
    "run_mcmc",
]
