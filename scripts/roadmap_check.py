"""Check the step runbook at ``docs/dev/roadmap.md``.

The runbook is the authoritative record of *where the work has got to*: it names
the next step, the branch it lands on and the plan it follows. A stale pointer is
worse than none, because a fresh session trusts it, so this script verifies that
the runbook still hangs together:

* the machine-readable blocks parse and use known statuses;
* exactly one step is current, and the current-position block agrees with it;
* every step cites a plan file that exists, a work package the coverage ledger
  declares, and ledger row IDs that exist;
* every step has a section with the headings a low-context worker is told to
  read.

Run it with ``pixi run roadmap``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from scripts.coverage_report import LEDGER, PORTS, read_packages
from scripts.coverage_report import read_block as read_ledger_block

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "docs" / "dev" / "roadmap.md"

STATUSES = ("done", "in-progress", "next", "planned", "blocked")
CURRENT_STATUSES = ("next", "in-progress")

# Column positions in the steps table (0-based).
STEP, PHASE, WP, BRANCH, PLAN, SECTION, STATUS, CLOSES = range(8)

# Every step section must offer these, in this order, to be followable cold.
REQUIRED_HEADINGS = ("Summary", "Read first", "Do", "Exit checks", "Handoff")

NONE_CELLS = {"", "—", "-", "–"}


def _strip(cell: str) -> str:
    """Drop whitespace and the markdown emphasis a table cell may carry."""
    return cell.strip().strip("*").strip("`").strip()


def read_roadmap_block(text: str, name: str) -> list[list[str]]:
    """Return the data rows of the fenced runbook table called ``name``."""
    match = re.search(rf"<!-- roadmap:{name} -->(.*?)<!-- /roadmap:{name} -->", text, re.S)
    if match is None:
        raise ValueError(f"Runbook block {name!r} not found in {ROADMAP}.")
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
        raise ValueError(f"Runbook block {name!r} has no data rows.")
    return rows[1:]  # drop the header


def read_current(text: str) -> dict[str, str]:
    """Return the ``Field``/``Value`` pairs of the current-position block."""
    fields: dict[str, str] = {}
    for row in read_roadmap_block(text, "current"):
        if len(row) != 2:
            raise ValueError(f"Current-position row {row!r} needs exactly two cells.")
        fields[row[0]] = row[1]
    missing = {"Step", "Status", "Branch"} - set(fields)
    if missing:
        raise ValueError(f"Current-position block is missing {sorted(missing)}.")
    return fields


def known_ids() -> set[str]:
    """Every row ID a step may claim to close."""
    api = read_ledger_block(LEDGER.read_text(encoding="utf-8"), "api")
    ports = read_ledger_block(PORTS.read_text(encoding="utf-8"), "ports", PORTS)
    return {row[0] for row in api} | {row[0] for row in ports}


def step_sections(text: str) -> dict[int, str]:
    """Map step number to the body of its ``## Step N`` section."""
    sections: dict[int, str] = {}
    matches = list(re.finditer(r"^## Step (\d+)\b(.*)$", text, re.M))
    for index, match in enumerate(matches):
        number = int(match.group(1))
        if number in sections:
            raise ValueError(f"Two sections claim to be step {number}.")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[number] = text[match.end() : end]
    return sections


def check(text: str) -> list[str]:
    """Return every problem found in the runbook, in reading order."""
    problems: list[str] = []
    rows = read_roadmap_block(text, "steps")
    ledger_text = LEDGER.read_text(encoding="utf-8")
    declared = {pkg_id for pkg_id, _name, _closes in read_packages(ledger_text)}
    ids = known_ids()
    sections = step_sections(text)

    numbers: list[int] = []
    by_number: dict[int, list[str]] = {}
    for row in rows:
        if len(row) != 8:
            problems.append(f"Step row {row[0]!r} has {len(row)} cells, expected 8.")
            continue
        try:
            number = int(row[STEP])
        except ValueError:
            problems.append(f"Step number {row[STEP]!r} is not an integer.")
            continue
        numbers.append(number)
        by_number[number] = row

        if row[STATUS] not in STATUSES:
            problems.append(f"Step {number} has unknown status {row[STATUS]!r}; use {STATUSES}.")
        if row[WP] not in NONE_CELLS and row[WP] not in declared:
            problems.append(
                f"Step {number} names work package {row[WP]!r}, "
                "which the coverage ledger does not declare."
            )
        if not (ROOT / row[PLAN]).is_file():
            problems.append(f"Step {number} cites a missing plan file: {row[PLAN]}.")
        if row[CLOSES] not in NONE_CELLS:
            for row_id in row[CLOSES].split():
                if row_id not in ids:
                    problems.append(f"Step {number} closes unknown ledger ID {row_id!r}.")

        body = sections.get(number)
        if body is None:
            problems.append(f"Step {number} has no '## Step {number}' section.")
        else:
            for heading in REQUIRED_HEADINGS:
                if not re.search(rf"^### {re.escape(heading)}\s*$", body, re.M):
                    problems.append(f"Step {number}'s section has no '### {heading}' heading.")
            if row[STATUS] == "done" and "**Landed:**" not in body:
                problems.append(
                    f"Step {number} is done but its section has no '**Landed:**' line "
                    "naming the branch and PR."
                )

    if numbers and numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"Step numbers must run 1..N without gaps; got {numbers}.")

    current_rows = [row for row in rows if len(row) == 8 and row[STATUS] in CURRENT_STATUSES]
    if len(current_rows) != 1:
        problems.append(
            f"Exactly one step must be {' or '.join(CURRENT_STATUSES)}; "
            f"found {len(current_rows)}."
        )

    try:
        current = read_current(text)
    except ValueError as error:
        problems.append(str(error))
        return problems

    try:
        current_number = int(current["Step"])
    except ValueError:
        problems.append(f"Current-position step {current['Step']!r} is not an integer.")
        return problems

    row = by_number.get(current_number)
    if row is None:
        problems.append(f"Current-position names step {current_number}, absent from the table.")
        return problems
    if current["Status"] not in CURRENT_STATUSES:
        problems.append(
            f"Current-position status {current['Status']!r} must be one of {CURRENT_STATUSES}."
        )
    if current["Status"] != row[STATUS]:
        problems.append(
            f"Current-position says step {current_number} is {current['Status']!r}, "
            f"the table says {row[STATUS]!r}."
        )
    if current["Branch"] != row[BRANCH]:
        problems.append(
            f"Current-position branch {current['Branch']!r} does not match "
            f"the table's {row[BRANCH]!r} for step {current_number}."
        )
    return problems


def summary(text: str) -> str:
    """One-line statement of where the work stands."""
    rows = read_roadmap_block(text, "steps")
    done = sum(1 for row in rows if len(row) == 8 and row[STATUS] == "done")
    current = read_current(text)
    return (
        f"{done} of {len(rows)} steps done. "
        f"Next: step {current['Step']} on {current['Branch']}."
    )


def main(argv: list[str] | None = None) -> int:
    """Report the runbook's position, or verify it with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero on any problem")
    args = parser.parse_args(argv)

    if not ROADMAP.is_file():
        print(f"The runbook is missing: {ROADMAP}", file=sys.stderr)
        return 1

    text = ROADMAP.read_text(encoding="utf-8")
    problems = check(text)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if args.check:
            return 1
    else:
        print(summary(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
