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

# Status column in each ledger table (0-based, after the header is stripped).
STATUS_COLUMN = {
    "api": 3,
    "textbook": 2,
    "summer2docs": 1,
}

# Optional trailing Ported column (path under docs/, or em dash).
PORTED_COLUMN = {
    "textbook": 4,
    "summer2docs": 3,
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
    state = {row[0]: row[3] for row in api}
    total = len(api)

    rows = [("today", sum(v == "full" for v in state.values()), total)]
    for pkg_id, _name, closes in read_packages(text):
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

    status = tally(api, STATUS_COLUMN["api"])
    lines: list[str] = []
    lines.append("summer4 coverage")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"API ledger: {status.total} summer2 symbols")
    lines.append("")
    lines.append(f"{'':<38}{'complete':>12}{'covered':>12}")
    lines.append(
        f"  {'shipped (summer4 today)':<36}"
        f"{status.complete:>4} ({status.pct(status.complete):>4.0f}%)"
        f"{status.covered:>6} ({status.pct(status.covered):>4.0f}%)"
    )
    lines.append("")
    lines.append("By area (complete / covered / total):")
    areas = by_area(api, STATUS_COLUMN["api"])
    for area, title in AREA_TITLES.items():
        if area not in areas:
            continue
        t = areas[area]
        lines.append(f"  {title:<38}{t.complete}/{t.covered}/{t.total}")
    lines.append("")
    for name, rows, unit in (
        ("Textbook chapters", book, "chapters"),
        ("summer2 doc pages", s2, "pages"),
    ):
        block = "textbook" if name.startswith("Textbook") else "summer2docs"
        t = tally(rows, STATUS_COLUMN[block])
        lines.append(f"{name}: {t.total} {unit}")
        lines.append(f"  {t.full:>4} full{t.partial:>6} partial{t.none:>6} blocked")
        lines.append("")
    lines.append("Path to 100% (API rows at `full`):")
    for label, count, total in progression(text):
        lines.append(f"  {label:<10}{count:>4} / {total}  ({100 * count / total:>3.0f}%)")
    lines.append("")
    remaining = [row for row in api if row[STATUS_COLUMN["api"]] != "full"]
    lines.append(f"Distance to 100%: {len(remaining)} of {status.total} symbols still incomplete.")
    return "\n".join(lines)


def quoted_totals(text: str) -> dict[str, int]:
    """Totals the rest of the documentation is allowed to quote."""
    api = read_block(text, "api")
    status = tally(api, STATUS_COLUMN["api"])
    return {
        "api_total": status.total,
        "complete": status.complete,
        "covered": status.covered,
    }


def declared_ports(text: str) -> list[tuple[str, str, str]]:
    """Return ``(block, row_label, port_path)`` for every non-dash Ported cell."""
    found: list[tuple[str, str, str]] = []
    for block, label_col in (("textbook", 0), ("summer2docs", 0)):
        if block not in PORTED_COLUMN:
            continue
        port_col = PORTED_COLUMN[block]
        for row in read_block(text, block):
            if len(row) <= port_col:
                continue
            port = row[port_col]
            if port in {"", "—", "-", "–"}:
                continue
            found.append((block, row[label_col], port))
    return found


def publishable_unported(text: str) -> list[tuple[str, str]]:
    """Rows at ``full`` with no Ported path — backlog, not a check failure."""
    missing: list[tuple[str, str]] = []
    for block, label_col in (("textbook", 0), ("summer2docs", 0)):
        status_col = STATUS_COLUMN[block]
        port_col = PORTED_COLUMN[block]
        for row in read_block(text, block):
            if row[status_col] != "full":
                continue
            port = row[port_col] if len(row) > port_col else "—"
            if port in {"", "—", "-", "–"}:
                missing.append((block, row[label_col]))
    return missing


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

    docs_root = ROOT / "docs"
    missing_files = [
        (block, label, port)
        for block, label, port in declared_ports(text)
        if not (docs_root / port).is_file()
    ]
    if missing_files:
        print("Declared Ported paths must exist under docs/:", file=sys.stderr)
        for block, label, port in missing_files:
            print(f"  [{block}] {label}: {port}", file=sys.stderr)
        return 1

    totals = quoted_totals(text)
    index = (ROOT / "docs" / "evaluation" / "index.md").read_text(encoding="utf-8")
    expected = f"**{totals['complete']} of {totals['api_total']}"
    if expected not in index:
        print("evaluation/index.md quotes stale totals; expected to find:", file=sys.stderr)
        print(f"  {expected} ...", file=sys.stderr)
        print("\nCurrent ledger says:\n" + report, file=sys.stderr)
        return 1
    print(report)
    backlog = publishable_unported(text)
    print(
        f"\nPublishable, not yet ported: {len(backlog)} "
        f"(textbook/summer2docs rows at full with no Ported path)."
    )
    for block, label in backlog:
        print(f"  [{block}] {label}")
    print("\nQuoted totals are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
