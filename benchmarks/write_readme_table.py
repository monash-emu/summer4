"""Build the comparison README table from both committed JSON matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SUMMER2 = ROOT.parent / "summer2bench" / "recorded.json"
SUMMER4 = ROOT / "recorded-summer4.json"
README = ROOT / "README.md"


def _fmt_time(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number < 0.001:
        return f"{number * 1e6:.1f} µs"
    if number < 1.0:
        return f"{number * 1e3:.2f} ms"
    return f"{number:.3f} s"


def _row(library: str, record: dict[str, Any]) -> str:
    status = record["status"]
    if status != "ok":
        reason = record.get("reason") or status
        return (
            f"| {library} | {record['model']} | {record['solver']} | {record['steps']} | "
            f"— | — | — | — | — | — | failed: {reason} |"
        )
    return (
        f"| {library} | {record['model']} | {record['solver']} | {record['steps']} | "
        f"{record['compartments']} | {_fmt_time(record['build_s'])} | "
        f"{_fmt_time(record['compile_s'])} | {_fmt_time(record['warm_median_s'])} | "
        f"{record['dtype']} | {record['jax_version']} |"
    )


def main() -> None:
    summer2 = json.loads(SUMMER2.read_text(encoding="utf-8"))
    summer4 = json.loads(SUMMER4.read_text(encoding="utf-8"))
    machine = next(
        (r["machine"] for r in summer2 + summer4 if r.get("machine")),
        "unknown",
    )
    date = next((r["date"] for r in summer2 + summer4 if r.get("date")), "unknown")
    lines = [
        "# Model timing suite",
        "",
        "Committed warm-JIT timings for the shared ladder in",
        "`summer2bench/spec.json`. Taxonomy benchmarks (`pixi run bench`) are",
        "separate and are not in this table.",
        "",
        f"Machine: `{machine}`. Date: `{date}`.",
        "",
        "Protocol: float64 (`jax_enable_x64` before compile); fixed `dt=0.1`;",
        "step counts 200, 2_000, 8_000; save full compartments plus one raw",
        "series for `infection` and `recovery`; warm median of 5 calls after",
        "one discarded compile, each ending in `jax.block_until_ready` on a",
        "reduced sum. Age-only mixing on stratified models.",
        "",
        'summer2 uses its graph runners with `solver="euler"` and',
        '`solver="rk4"` (JAX 0.4.x in `summer2bench/`). summer4 uses Diffrax',
        "`Euler` and classical RK4 with `ConstantStepSize` (JAX 0.6.x in this",
        "repo). The summer4 number includes Diffrax call overhead. Do not read",
        "a winner out of the table beyond the times shown.",
        "",
        "Diffrax `run()` reuses equinox's JIT cache across equal save plans",
        "(see `plans/diffrax-run-jit-cache.plan.md`). Re-record",
        "`benchmarks/recorded-summer4.json` after that fix if the warm column",
        "still looks like recompile-per-call (~0.3 s for `sir` / Euler / 200).",
        "",
        "JSON: `summer2bench/recorded.json`, `benchmarks/recorded-summer4.json`.",
        "",
        (
            "| library | model | solver | steps | compartments | build | "
            "compile | warm median | dtype | JAX |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in summer2:
        lines.append(_row("summer2", record))
    for record in summer4:
        lines.append(_row("summer4", record))
    lines.append("")
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {README} rows={len(summer2) + len(summer4)}", flush=True)


if __name__ == "__main__":
    main()
