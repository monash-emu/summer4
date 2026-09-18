"""Group SavePlan requests that share a time grid."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from summer4.results.plan import SavePlan


@dataclass(frozen=True, slots=True)
class SaveGroup:
    """One shared ``ts`` and the request keys that evaluate on it."""

    name: str  # "g0", "g1", ... stable for a given plan
    ts: NDArray[np.float64]
    keys: tuple[str, ...]


def _ts_digest(ts: NDArray[np.float64]) -> bytes:
    return hashlib.blake2b(
        np.ascontiguousarray(ts, dtype=np.float64).tobytes(), digest_size=16
    ).digest()


def group_requests(
    plan: SavePlan,
    default_ts: NDArray[np.float64],
) -> tuple[SaveGroup, ...]:
    """Bucket requests by equal ``ts`` arrays (content hash, not ``id()``).

    Requests with ``ts=None`` join the group for ``default_ts``. Groups are
    ordered by first appearance of their keys so ``name`` is deterministic.
    """
    default = np.asarray(default_ts, dtype=np.float64)
    # digest -> (ts, keys in appearance order)
    buckets: dict[bytes, tuple[NDArray[np.float64], list[str]]] = {}
    order: list[bytes] = []

    for key, req in plan.requests.items():
        ts = default if req.ts is None else np.asarray(req.ts, dtype=np.float64)
        digest = _ts_digest(ts)
        if digest not in buckets:
            buckets[digest] = (ts, [])
            order.append(digest)
        buckets[digest][1].append(key)

    groups: list[SaveGroup] = []
    for i, digest in enumerate(order):
        ts, keys = buckets[digest]
        groups.append(SaveGroup(name=f"g{i}", ts=ts, keys=tuple(keys)))
    return tuple(groups)
