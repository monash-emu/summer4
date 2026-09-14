"""Parse the coverage ledger and report completeness totals.

The ledger at ``docs/evaluation/coverage-ledger.md`` is the single source of
truth for what summer4 covers. This script reads it, computes the totals that
the rest of the documentation quotes, and (with ``--check``) verifies that the
quoted totals are still correct.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "evaluation" / "coverage-ledger.md"

STATUSES = ("full", "partial", "none")

AREA_TITLES = {
    "lifecycle": "Model lifecycle",
    "stratification": "Compartments and stratification",
    "queries": "Compartment queries",
    "flows": "Flows",
    "adjustments": "Flow adjustments",
    "mixing": "Mixing",
    "parameters": "Parameters and time-varying functions",
    "outputs": "Derived outputs",
    "solver": "Solver",
    "time": "Real-world time",
}


@dataclass(frozen=True, slots=True)
class Tally:
    """Counts of ``full`` / ``partial`` / ``none`` rows."""

    full: int
    partial: int
    none: int

    @property
    def total(self) -> int:
        return self.full + self.partial + self.none

    @property
    def complete(self) -> int:
        """Rows that need no further work."""
        return self.full

    @property
    def covered(self) -> int:
        """Rows with any working route, complete or not."""
        return self.full + self.partial

    def pct(self, value: int) -> float:
        return 100.0 * value / self.total if self.total else 0.0


def _strip(cell: str) -> str:
    return cell.strip().strip("`").strip()


def read_block(text: str, name: str) -> list[list[str]]:
    """Return the data rows of the fenced ledger table called ``name``."""
    match = re.search(
        rf"<!-- ledger:{name} -->(.*?)<!-- /ledger:{name} -->",
        text,
        re.S,
    )
    if match is None:
        raise ValueError(f"Ledger block {name!r} not found in {LEDGER}.")
    rows: list[list[str]] = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_strip(cell) for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        raise ValueError(f"Ledger block {name!r} has no data rows.")
    return rows[1:]  # drop the header


def tally(rows: list[list[str]], column: int) -> Tally:
    """Count statuses in ``column``, rejecting anything unrecognised."""
    counts = dict.fromkeys(STATUSES, 0)
    for row in rows:
        status = row[column]
        if status not in counts:
            raise ValueError(f"Unknown status {status!r} in row {row[0]!r}. Use one of {STATUSES}.")
        counts[status] += 1
    return Tally(**counts)


def by_area(rows: list[list[str]], column: int) -> dict[str, Tally]:
    """Tally the API ledger per area, preserving declaration order."""
    grouped: dict[str, list[list[str]]] = {}
    for row in rows:
        grouped.setdefault(row[2], []).append(row)
    return {area: tally(area_rows, column) for area, area_rows in grouped.items()}


def read_packages(text: str) -> list[tuple[str, str, list[str]]]:
    """Return ``(id, name, closes)`` for each declared work package."""
    packages: list[tuple[str, str, list[str]]] = []
    for row in read_block(text, "packages"):
        tokens = row[2].replace("*", "").split()
        ids = [token for token in tokens if re.fullmatch(r"[A-Z]\d+", token)]
        packages.append((row[0], row[1], ids))
    return packages


def progression(text: str) -> list[tuple[str, int, int]]:
    """Cumulative ``full`` count after each package, computed from declarations."""
    api = read_block(text, "api")
    shipped = {row[0]: row[3] for row in api}
    spike = {row[0]: row[4] for row in api}
    total = len(api)

    state = dict(shipped)
    rows = [("today", sum(v == "full" for v in state.values()), total)]
    for pkg_id, _name, closes in read_packages(text):
        if pkg_id == "WP1":
            state = dict(spike)
        for ledger_id in closes:
            if ledger_id not in state:
                raise ValueError(f"{pkg_id} closes unknown ledger ID {ledger_id!r}.")
            state[ledger_id] = "full"
        rows.append((pkg_id, sum(v == "full" for v in state.values()), total))
    return rows


def render_progression(text: str) -> str:
    """Markdown for the computed progression table."""
    lines = [
        "| After | API rows at `full` | Share |",
        "| --- | --- | --- |",
    ]
    for label, count, total in progression(text):
        lines.append(f"| {label} | {count} / {total} | {100 * count / total:.0f}% |")
    return "\n".join(lines)


def update_progression(text: str) -> str:
    """Rewrite the computed progression block in place."""
    return re.sub(
        r"(<!-- ledger:progression -->)(.*?)(<!-- /ledger:progression -->)",
        lambda m: f"{m.group(1)}\n{render_progression(text)}\n{m.group(3)}",
        text,
        flags=re.S,
    )


def render(text: str) -> str:
    """Build the human-readable report."""
    api = read_block(text, "api")
    book = read_block(text, "textbook")
    s2 = read_block(text, "summer2docs")

    shipped, spike = tally(api, 3), tally(api, 4)
    lines: list[str] = []
    lines.append("summer4 coverage")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"API ledger: {shipped.total} summer2 symbols")
    lines.append("")
    lines.append(f"{'':<38}{'complete':>12}{'covered':>12}")
    for label, t in (("shipped (summer4 today)", shipped), ("with flows spike promoted", spike)):
        lines.append(
            f"  {label:<36}"
            f"{t.complete:>4} ({t.pct(t.complete):>4.0f}%)"
            f"{t.covered:>6} ({t.pct(t.covered):>4.0f}%)"
        )
    lines.append("")
    lines.append("By area (complete / covered / total):")
    ship_areas, spike_areas = by_area(api, 3), by_area(api, 4)
    for area, title in AREA_TITLES.items():
        if area not in ship_areas:
            continue
        s, k = ship_areas[area], spike_areas[area]
        lines.append(
            f"  {title:<38}"
            f"shipped {s.complete}/{s.covered}/{s.total:<6}"
            f"spike {k.complete}/{k.covered}/{k.total}"
        )
    lines.append("")
    for name, rows, col_ship, col_spike, unit in (
        ("Textbook chapters", book, 2, 3, "chapters"),
        ("summer2 doc pages", s2, 1, 2, "pages"),
    ):
        ts, tk = tally(rows, col_ship), tally(rows, col_spike)
        lines.append(f"{name}: {ts.total} {unit}")
        lines.append(
            f"  shipped                             "
            f"{ts.full:>4} full{ts.partial:>6} partial{ts.none:>6} blocked"
        )
        lines.append(
            f"  with flows spike promoted           "
            f"{tk.full:>4} full{tk.partial:>6} partial{tk.none:>6} blocked"
        )
        lines.append("")
    lines.append("Path to 100% (API rows at `full`):")
    for label, count, total in progression(text):
        lines.append(f"  {label:<10}{count:>4} / {total}  ({100 * count / total:>3.0f}%)")
    lines.append("")
    remaining = [row for row in api if row[4] != "full"]
    lines.append(f"Distance to 100%: {len(remaining)} of {shipped.total} symbols")
    lines.append("still incomplete even with the spike promoted.")
    return "\n".join(lines)


def quoted_totals(text: str) -> dict[str, int]:
    """Totals the rest of the documentation is allowed to quote."""
    api = read_block(text, "api")
    shipped, spike = tally(api, 3), tally(api, 4)
    return {
        "api_total": shipped.total,
        "shipped_complete": shipped.complete,
        "shipped_covered": shipped.covered,
        "spike_complete": spike.complete,
        "spike_covered": spike.covered,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate the computed progression block in the ledger.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate statuses and the totals quoted in evaluation/index.md.",
    )
    args = parser.parse_args(argv)

    text = LEDGER.read_text(encoding="utf-8")
    if args.write:
        LEDGER.write_text(update_progression(text), encoding="utf-8")
        text = LEDGER.read_text(encoding="utf-8")
        print(f"Rewrote the progression block in {LEDGER}.")
    report = render(text)
    if not args.check:
        print(report)
        return 0

    if update_progression(text) != text:
        print(
            "The progression table is stale. Run: pixi run coverage-write",
            file=sys.stderr,
        )
        return 1

    totals = quoted_totals(text)
    index = (ROOT / "docs" / "evaluation" / "index.md").read_text(encoding="utf-8")
    expected = (
        f"**{totals['shipped_complete']} of {totals['api_total']}",
        f"**{totals['spike_complete']} of {totals['api_total']}",
    )
    missing = [snippet for snippet in expected if snippet not in index]
    if missing:
        print("evaluation/index.md quotes stale totals; expected to find:", file=sys.stderr)
        for snippet in missing:
            print(f"  {snippet} ...", file=sys.stderr)
        print("\nCurrent ledger says:\n" + report, file=sys.stderr)
        return 1
    print(report)
    print("\nQuoted totals are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
