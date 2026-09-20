"""Record the summer4 model matrix on each pinned JAX pixi environment.

Runs ``record_models.py`` under ``jax04`` (summer2-matched 0.4.38), ``default``
(0.6.x), and ``latest`` (bleeding-edge JAX). Diffrax is whatever each env
resolves (today still 0.7.2 across all three). Writes one JSON per JAX version
and regenerates ``benchmarks/jax-sweep.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RECORDER = ROOT / "benchmarks" / "record_models.py"
SWEEP_TABLE = ROOT / "benchmarks" / "write_jax_sweep_table.py"
ENVS = ("jax04", "default", "latest")


def _run_env(env: str) -> Path:
    """Delete any prior versioned file for this env's JAX, then record."""
    probe = subprocess.run(
        ["pixi", "run", "-e", env, "python", "-c", "import jax; print(jax.__version__)"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    jax_version = probe.stdout.strip().splitlines()[-1]
    out = ROOT / "benchmarks" / f"recorded-summer4-jax-{jax_version}.json"
    if out.exists():
        out.unlink()
    print(f"=== env={env} jax={jax_version} -> {out.name} ===", flush=True)
    completed = subprocess.run(
        [
            "pixi",
            "run",
            "-e",
            env,
            "python",
            str(RECORDER),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"recording failed for env={env} jax={jax_version}")
    _summarize(out)
    return out


def _summarize(path: Path) -> None:
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    ok = sum(1 for record in records if record["status"] == "ok")
    failed = len(records) - ok
    sample = next(
        (
            record
            for record in records
            if record["model"] == "sir"
            and record["solver"] == "diffrax-euler"
            and record["steps"] == 200
            and record["status"] == "ok"
        ),
        None,
    )
    warm = sample["warm_median_s"] if sample else None
    jax_v = sample["jax_version"] if sample else "?"
    diffrax_v = sample.get("diffrax_version") if sample else "?"
    print(
        f"summary {path.name}: ok={ok} failed={failed} "
        f"jax={jax_v} diffrax={diffrax_v} sir/euler/200 warm={warm}",
        flush=True,
    )


def main() -> None:
    paths = [_run_env(env) for env in ENVS]
    print(f"wrote {len(paths)} matrices", flush=True)
    subprocess.run([sys.executable, str(SWEEP_TABLE)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
