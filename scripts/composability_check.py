"""Check docs/evaluation/composability.md, the composability survey.

The survey is a Markdown table of findings (``CX*``) and work packages
(``CP*``). This script fails when a row uses a value outside the vocabulary,
when an ID repeats, when an evidence anchor no longer appears in its file
(unless the finding is ``done``), when an open finding belongs to no package,
or when the quoted totals line is stale. ``--write`` recomputes the totals.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "docs" / "evaluation" / "composability.md"

AREAS = frozenset({"run", "rates", "results", "epi", "calibration", "taxonomy"})
PRINCIPLES = frozenset({"P1", "P2", "P3", "P4"})
SEVERITIES = ("high", "medium", "low")
STATUSES = ("open", "planned", "deferred", "done", "rejected")
PACKAGED = frozenset({"open", "planned"})


@dataclass(frozen=True, slots=True)
class Finding:
    """One row of the findings table."""

    id: str
    area: str
    principles: tuple[str, ...]
    severity: str
    status: str
    evidence: tuple[str, ...]


def read_block(text: str, name: str) -> list[list[str]]:
    """Return the body rows of the table between ``<!-- composability:name -->`` markers."""
    match = re.search(
        rf"<!-- composability:{name} -->(.*?)<!-- /composability:{name} -->", text, re.S
    )
    if match is None:
        raise ValueError(f"missing <!-- composability:{name} --> block")
    rows: list[list[str]] = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        if cells and set(cells[0]) <= {"-", " "}:
            continue  # separator row
        rows.append(cells)
    return rows[1:]  # drop the header row


def parse_findings(text: str) -> list[Finding]:
    """Parse the findings table."""
    findings: list[Finding] = []
    for cells in read_block(text, "findings"):
        if len(cells) != 8:
            raise ValueError(f"findings row has {len(cells)} cells, expected 8: {cells[:1]}")
        evidence = tuple(re.findall(r"`([^`]+)`", cells[7]))
        findings.append(
            Finding(
                id=cells[0],
                area=cells[1],
                principles=tuple(p.strip() for p in cells[2].split(",")),
                severity=cells[3],
                status=cells[4],
                evidence=evidence,
            )
        )
    return findings


def parse_packages(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Parse the work-package table into ``(package id, finding ids)``."""
    packages: list[tuple[str, tuple[str, ...]]] = []
    for cells in read_block(text, "packages"):
        if len(cells) != 3:
            raise ValueError(f"packages row has {len(cells)} cells, expected 3: {cells[:1]}")
        packages.append((cells[0], tuple(cells[2].split())))
    return packages


def render_totals(findings: list[Finding]) -> str:
    """The totals line the survey must quote."""
    status = Counter(f.status for f in findings)
    open_by = Counter(f.severity for f in findings if f.status == "open")
    return (
        f"Open findings: **{status['open']}** ({open_by['high']} high, "
        f"{open_by['medium']} medium, {open_by['low']} low); "
        f"planned: {status['planned']}; deferred: {status['deferred']}; "
        f"done: {status['done']}; rejected: {status['rejected']}."
    )


def _quoted_totals(text: str) -> str:
    match = re.search(
        r"<!-- composability:totals -->(.*?)<!-- /composability:totals -->", text, re.S
    )
    if match is None:
        raise ValueError("missing <!-- composability:totals --> block")
    return match.group(1).strip()


def _check_evidence(finding: Finding, root: Path) -> list[str]:
    errors: list[str] = []
    if not finding.evidence:
        return [f"{finding.id}: no evidence"]
    cites_note = False
    for item in finding.evidence:
        path_text, _, anchor = item.partition("::")
        path = root / path_text
        if not path.is_file():
            errors.append(f"{finding.id}: evidence file {path_text} does not exist")
            continue
        if path_text.startswith("futureplans/"):
            cites_note = True
        if anchor and finding.status != "done" and anchor not in path.read_text(encoding="utf-8"):
            errors.append(
                f"{finding.id}: anchor {anchor!r} no longer appears in {path_text} "
                "(fixed? set the status to done; moved? update the anchor)"
            )
    if finding.status == "deferred" and not cites_note:
        errors.append(f"{finding.id}: deferred findings must cite a futureplans/ note")
    return errors


def check(text: str, root: Path = ROOT) -> list[str]:
    """Return every problem with the survey text; empty means it is consistent."""
    errors: list[str] = []
    findings = parse_findings(text)
    ids = [f.id for f in findings]
    for dup in sorted(i for i, n in Counter(ids).items() if n > 1):
        errors.append(f"{dup}: duplicate finding ID")
    for f in findings:
        if not re.fullmatch(r"CX\d+", f.id):
            errors.append(f"{f.id}: finding IDs look like CX12")
        if f.area not in AREAS:
            errors.append(f"{f.id}: area {f.area!r} is not one of {sorted(AREAS)}")
        bad = [p for p in f.principles if p not in PRINCIPLES]
        if bad:
            errors.append(f"{f.id}: principle(s) {bad} are not P1–P4")
        if f.severity not in SEVERITIES:
            errors.append(f"{f.id}: severity {f.severity!r} is not one of {list(SEVERITIES)}")
        if f.status not in STATUSES:
            errors.append(f"{f.id}: status {f.status!r} is not one of {list(STATUSES)}")
        errors.extend(_check_evidence(f, root))

    by_id = {f.id: f for f in findings}
    homes: Counter[str] = Counter()
    package_ids = [pid for pid, _ in parse_packages(text)]
    for dup in sorted(i for i, n in Counter(package_ids).items() if n > 1):
        errors.append(f"{dup}: duplicate package ID")
    for pid, closes in parse_packages(text):
        for fid in closes:
            if fid not in by_id:
                errors.append(f"{pid}: closes unknown finding {fid}")
            homes[fid] += 1
    for f in findings:
        if f.status in PACKAGED and homes[f.id] != 1:
            errors.append(
                f"{f.id}: {f.status} findings belong to exactly one package, found {homes[f.id]}"
            )

    expected = render_totals(findings)
    if _quoted_totals(text) != expected:
        errors.append(f"totals line is stale; run `pixi run composability-write` ({expected})")
    return errors


def write_totals(text: str) -> str:
    """Return ``text`` with the totals block recomputed."""
    expected = render_totals(parse_findings(text))
    return re.sub(
        r"(<!-- composability:totals -->)(.*?)(<!-- /composability:totals -->)",
        lambda m: f"{m.group(1)}\n{expected}\n{m.group(3)}",
        text,
        flags=re.S,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="recompute the totals line")
    args = parser.parse_args(argv)
    text = SURVEY.read_text(encoding="utf-8")
    if args.write:
        text = write_totals(text)
        SURVEY.write_text(text, encoding="utf-8")
    errors = check(text)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(render_totals(parse_findings(text)).replace("**", ""))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
