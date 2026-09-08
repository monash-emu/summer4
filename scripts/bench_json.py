"""Run taxonomy benchmarks and write JSON keyed by the current JAX version."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax


def main() -> None:
    jax_version = jax.__version__
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"jax-{jax_version}.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "benchmarks/",
        "--benchmark-only",
        "--benchmark-columns=min,max,median,ops",
        f"--benchmark-json={out_path}",
    ]
    subprocess.run(cmd, check=True)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["jax_version"] = jax_version
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (jax {jax_version})")


if __name__ == "__main__":
    main()
