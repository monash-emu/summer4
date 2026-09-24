"""Score a design: batched, memory-bounded ``log_density`` evaluation."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from summer4.epi.calibration.workflow.candidates import (
    Candidates,
    StageRecord,
    ok_from_log_density,
)


def evaluate(bm: Any, candidates: Candidates, *, batch_size: int = 64) -> Candidates:
    """Score each candidate with ``bm.log_density`` under a bounded batch size.

    Uses ``jax.lax.map(..., batch_size=)`` so peak memory stays proportional to
    ``batch_size``, not ``len(candidates)``. Prefer a fixed-step solver for large
    designs: vmapping an adaptive Diffrax solve runs every lane to the slowest
    lane's step count.

    Sets ``log_density`` and ``ok`` (``ok`` is false when the density hit the
    failed-solve sentinel).
    """
    if len(candidates) == 0:
        raise ValueError("evaluate() got an empty Candidates batch.")
    if candidates.sites != tuple(bm.prior_names()):
        raise ValueError(
            "evaluate site mismatch: candidates "
            f"{candidates.sites!r} vs model {bm.prior_names()!r}."
        )
    batch = max(1, int(batch_size))
    t0 = time.perf_counter()

    # Materialise z as a JAX pytree with leading axis n.
    z_tree = {name: jnp.asarray(candidates.z[name]) for name in candidates.sites}

    def _one(z_i: dict[str, Any]) -> Any:
        return bm.log_density(z_i)

    # Warm the potential once on the host before tracing the map.
    _ = bm.log_density({k: v[0] for k, v in z_tree.items()})

    mapped = jax.jit(lambda z: jax.lax.map(_one, z, batch_size=batch))
    log_density = np.asarray(mapped(z_tree))
    ok = np.asarray(ok_from_log_density(log_density))
    seconds = time.perf_counter() - t0
    record = StageRecord(
        stage="evaluate",
        settings={"batch_size": batch, "n": len(candidates)},
        seconds=seconds,
    )
    return replace(
        candidates,
        log_density=log_density,
        ok=ok,
        history=candidates.history + (record,),
    )


__all__ = ["evaluate"]
