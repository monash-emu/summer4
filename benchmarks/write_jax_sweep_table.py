"""Build ``benchmarks/jax-sweep.md`` from ``recorded-summer4-jax-*.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "jax-sweep.md"
GLOB = "recorded-summer4-jax-*.json"

# Representative cells: small + large, both solvers, short + long horizon.
FOCUS = (
    ("sir", "diffrax-euler", 200),
    ("sir", "diffrax-euler", 8000),
    ("sir", "diffrax-rk4", 200),
    ("sir", "diffrax-rk4", 8000),
    ("stress", "diffrax-euler", 200),
    ("stress", "diffrax-euler", 8000),
    ("stress", "diffrax-rk4", 200),
    ("stress", "diffrax-rk4", 8000),
)


def _fmt_time(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number < 0.001:
        return f"{number * 1e6:.1f} µs"
    if number < 1.0:
        return f"{number * 1e3:.2f} ms"
    return f"{number:.3f} s"


def _load_matrices() -> list[tuple[str, list[dict[str, Any]]]]:
    files = sorted(ROOT.glob(GLOB))
    matrices: list[tuple[str, list[dict[str, Any]]]] = []
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        jax_version = next(
            (record.get("jax_version") for record in records if record.get("jax_version")),
            path.stem.removeprefix("recorded-summer4-jax-"),
        )
        matrices.append((str(jax_version), records))
    matrices.sort(key=lambda item: tuple(int(part) for part in item[0].split(".")))
    return matrices


def _index(records: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(r["model"], r["solver"], int(r["steps"])): r for r in records}


def main() -> None:
    matrices = _load_matrices()
    if not matrices:
        raise SystemExit(f"no {GLOB} files under {ROOT}")

    machine = next(
        (
            record["machine"]
            for _, records in matrices
            for record in records
            if record.get("machine")
        ),
        "unknown",
    )
    date = next(
        (record["date"] for _, records in matrices for record in records if record.get("date")),
        "unknown",
    )
    diffrax_versions = sorted(
        {
            str(record.get("diffrax_version"))
            for _, records in matrices
            for record in records
            if record.get("diffrax_version")
        }
    )

    lines: list[str] = [
        "# Summer4 JAX version sweep",
        "",
        "Warm-JIT Diffrax timings for the same summer4 model ladder under",
        "several JAX pins. Diffrax stays at whatever each pixi env resolves",
        f"(this sweep: {', '.join(diffrax_versions) or 'unknown'}).",
        "",
        f"Machine: `{machine}`. Date: `{date}`.",
        "",
        "Environments (`pixi run -e … bench-models` / `record_jax_sweep.py`):",
        "",
        "- `jax04` — JAX **0.4.38**, same pin as `summer2bench`",
        "- `default` — JAX **0.6.x** (repo default)",
        "- `latest` — bleeding-edge JAX (`jax=*`) ",
        "",
        "Protocol matches `benchmarks/README.md`: float64, fixed `dt=0.1`,",
        "warm median of 5 calls after one discarded compile,",
        "`block_until_ready` on the reduced sum.",
        "",
        "## Focus cells (warm median)",
        "",
    ]

    header = "| model | solver | steps |"
    sep = "| --- | --- | ---: |"
    for jax_version, _ in matrices:
        header += f" JAX {jax_version} |"
        sep += " ---: |"
    lines.extend([header, sep])

    indices = [(jax_version, _index(records)) for jax_version, records in matrices]
    for model, solver, steps in FOCUS:
        row = f"| {model} | {solver} | {steps} |"
        for _, by_key in indices:
            record = by_key.get((model, solver, steps))
            if record is None or record.get("status") != "ok":
                row += " — |"
            else:
                row += f" {_fmt_time(record['warm_median_s'])} |"
        lines.append(row)

    lines.extend(
        [
            "",
            "## Full matrix",
            "",
            "| JAX | Diffrax | model | solver | steps | build | compile | warm median |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for jax_version, records in matrices:
        for record in records:
            diffrax_v = record.get("diffrax_version") or "—"
            if record.get("status") != "ok":
                lines.append(
                    f"| {jax_version} | {diffrax_v} | {record['model']} | "
                    f"{record['solver']} | {record['steps']} | — | — | "
                    f"failed: {record.get('reason') or record['status']} |"
                )
                continue
            lines.append(
                f"| {jax_version} | {diffrax_v} | {record['model']} | "
                f"{record['solver']} | {record['steps']} | "
                f"{_fmt_time(record['build_s'])} | {_fmt_time(record['compile_s'])} | "
                f"{_fmt_time(record['warm_median_s'])} |"
            )

    lines.extend(
        [
            "",
            "JSON: `benchmarks/recorded-summer4-jax-<version>.json`.",
            "Regenerate with `pixi run bench-models-jax-sweep`.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} matrices={len(matrices)}", flush=True)


if __name__ == "__main__":
    main()
