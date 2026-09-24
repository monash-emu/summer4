"""Shared calibration-workflow value: a batch of scored parameter points."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

# Failed-solve sentinel from BayesianModel.numpyro_model; keep in sync.
_OK_THRESHOLD = -1.0e29


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Provenance for one workflow stage that produced or modified candidates."""

    stage: str
    settings: Mapping[str, Any]
    seconds: float


@dataclass(frozen=True, slots=True)
class Candidates:
    """Batch of parameter points in unconstrained and constrained form.

    Every workflow stage takes and/or returns :class:`Candidates` (or a result
    exposing ``.candidates``). ``log_density`` / ``ok`` stay ``None`` until
    :func:`~summer4.epi.calibration.workflow.evaluate` scores the batch.
    """

    sites: tuple[str, ...]
    z: dict[str, Any]
    params: dict[str, Any]
    log_density: Any | None = None
    ok: Any | None = None
    history: tuple[StageRecord, ...] = ()

    def __len__(self) -> int:
        if not self.sites:
            return 0
        first = np.asarray(self.z[self.sites[0]])
        return int(first.shape[0])

    def take(self, idx: Any) -> Candidates:
        """Select rows by integer index, boolean mask, or slice."""
        index = np.asarray(idx)
        z = {k: np.asarray(v)[index] for k, v in self.z.items()}
        params = {k: np.asarray(v)[index] for k, v in self.params.items()}
        ld = None if self.log_density is None else np.asarray(self.log_density)[index]
        ok = None if self.ok is None else np.asarray(self.ok)[index]
        return replace(self, z=z, params=params, log_density=ld, ok=ok)

    def best(self, n: int) -> Candidates:
        """Keep the ``n`` highest-``log_density`` rows; failed solves excluded."""
        if self.log_density is None or self.ok is None:
            raise ValueError("best() requires evaluated candidates (call evaluate first).")
        n_keep = int(n)
        if n_keep < 0:
            raise ValueError(f"best(n) expects n >= 0, got {n_keep}.")
        ld = np.asarray(self.log_density)
        ok = np.asarray(self.ok, dtype=bool)
        eligible = np.flatnonzero(ok)
        if eligible.size == 0:
            raise ValueError("best() found no successful (ok) candidates.")
        order = eligible[np.argsort(-ld[eligible])]
        chosen = order[:n_keep]
        return self.take(chosen)

    def concat(self, other: Candidates) -> Candidates:
        """Stack two batches along the candidate axis."""
        if self.sites != other.sites:
            raise ValueError(f"concat site mismatch: {self.sites!r} vs {other.sites!r}.")
        if (self.log_density is None) != (other.log_density is None):
            raise ValueError("concat requires both sides evaluated or both unevaluated.")
        z = {k: np.concatenate([np.asarray(self.z[k]), np.asarray(other.z[k])]) for k in self.sites}
        params = {
            k: np.concatenate([np.asarray(self.params[k]), np.asarray(other.params[k])])
            for k in self.sites
        }
        ld = (
            None
            if self.log_density is None
            else np.concatenate([np.asarray(self.log_density), np.asarray(other.log_density)])
        )
        ok = (
            None if self.ok is None else np.concatenate([np.asarray(self.ok), np.asarray(other.ok)])
        )
        return Candidates(
            sites=self.sites,
            z=z,
            params=params,
            log_density=ld,
            ok=ok,
            history=self.history + other.history,
        )

    def to_frame(self) -> Any:
        """One row per candidate: constrained params, ``log_density``, ``ok``."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "Candidates.to_frame requires pandas: pip install summer4[pandas]"
            ) from exc
        data: dict[str, Any] = {k: np.asarray(self.params[k]) for k in self.sites}
        if self.log_density is not None:
            data["log_density"] = np.asarray(self.log_density)
        if self.ok is not None:
            data["ok"] = np.asarray(self.ok)
        return pd.DataFrame(data)

    def save(self, path: str | Path) -> None:
        """Write ``.npz`` arrays plus a JSON sidecar for sites and history."""
        path = Path(path)
        arrays: dict[str, Any] = {}
        for name in self.sites:
            arrays[f"z/{name}"] = np.asarray(self.z[name])
            arrays[f"params/{name}"] = np.asarray(self.params[name])
        if self.log_density is not None:
            arrays["log_density"] = np.asarray(self.log_density)
        if self.ok is not None:
            arrays["ok"] = np.asarray(self.ok)
        np.savez_compressed(path, **arrays)
        meta = {
            "sites": list(self.sites),
            "history": [
                {"stage": r.stage, "settings": dict(r.settings), "seconds": r.seconds}
                for r in self.history
            ],
            "has_log_density": self.log_density is not None,
            "has_ok": self.ok is not None,
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Candidates:
        """Reload a batch written by :meth:`save`."""
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sites = tuple(meta["sites"])
        with np.load(path) as data:
            z = {name: np.asarray(data[f"z/{name}"]) for name in sites}
            params = {name: np.asarray(data[f"params/{name}"]) for name in sites}
            ld = np.asarray(data["log_density"]) if meta.get("has_log_density") else None
            ok = np.asarray(data["ok"]) if meta.get("has_ok") else None
        history = tuple(
            StageRecord(
                stage=str(item["stage"]),
                settings=dict(item["settings"]),
                seconds=float(item["seconds"]),
            )
            for item in meta.get("history", ())
        )
        return cls(sites=sites, z=z, params=params, log_density=ld, ok=ok, history=history)

    @classmethod
    def from_params(cls, bm: Any, params: Mapping[str, Any]) -> Candidates:
        """Build unevaluated candidates from constrained site arrays (leading axis ``n``)."""
        sites = tuple(bm.prior_names())
        param_dict = {name: np.asarray(params[name]) for name in sites}
        n = int(np.asarray(param_dict[sites[0]]).shape[0])
        for name in sites:
            arr = np.asarray(param_dict[name])
            if arr.shape[0] != n:
                raise ValueError(
                    f"from_params length mismatch for {name!r}: {arr.shape[0]} vs {n}."
                )
        z = {k: np.asarray(v) for k, v in bm.unconstrain(param_dict).items()}
        return cls(sites=sites, z=z, params=param_dict)

    @classmethod
    def from_idata(
        cls,
        bm: Any,
        idata: Any,
        *,
        n: int | None = None,
        seed: int = 0,
        burn_in: int = 0,
    ) -> Candidates:
        """Draw constrained posterior points from an ``arviz.InferenceData``."""
        posterior = idata.posterior
        sites = tuple(bm.prior_names())
        stacked: dict[str, np.ndarray] = {}
        for name in sites:
            if name not in posterior:
                raise KeyError(f"from_idata: site {name!r} missing from posterior.")
            arr = np.asarray(posterior[name].values)
            # (chain, draw, ...) → (chain*draw, ...)
            flat = arr.reshape(-1, *arr.shape[2:])
            if burn_in:
                # burn_in is per-chain; drop the first burn_in draws then restack
                n_chains, n_draws = arr.shape[:2]
                if burn_in >= n_draws:
                    raise ValueError(f"burn_in={burn_in} >= draws per chain ({n_draws}).")
                kept = arr[:, burn_in:, ...].reshape(-1, *arr.shape[2:])
                flat = kept
            stacked[name] = flat
        total = int(stacked[sites[0]].shape[0])
        if n is None or int(n) >= total:
            idx = np.arange(total)
        else:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(total, size=int(n), replace=False)
        params = {name: stacked[name][idx] for name in sites}
        return cls.from_params(bm, params)


def ok_from_log_density(log_density: Any) -> Any:
    """Derive the success mask from the failed-solve log-density sentinel."""
    import jax.numpy as jnp

    return jnp.asarray(log_density) > _OK_THRESHOLD


__all__ = [
    "Candidates",
    "StageRecord",
    "ok_from_log_density",
]
