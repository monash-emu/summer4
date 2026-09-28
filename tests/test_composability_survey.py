"""The composability survey stays consistent with the code it cites."""

from __future__ import annotations

import pytest
from scripts.composability_check import (
    SURVEY,
    check,
    parse_findings,
    render_totals,
    write_totals,
)


def _row(status: str = "open", anchor: str = "src/summer4/results/result.py::class Result:") -> str:
    return f"| CX1 | run | P1 | high | {status} | A finding. | A shape. | `{anchor}` |"


def _survey(row: str, packages: str = "| CP1 | Title | CX1 |") -> str:
    text = (
        "<!-- composability:totals -->\nTOTALS\n<!-- /composability:totals -->\n"
        "<!-- composability:findings -->\n"
        "| ID | Area | Principle | Severity | Status | Finding | Proposed shape | Evidence |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"{row}\n"
        "<!-- /composability:findings -->\n"
        "<!-- composability:packages -->\n"
        "| Package | Title | Closes |\n| --- | --- | --- |\n"
        f"{packages}\n"
        "<!-- /composability:packages -->\n"
    )
    return write_totals(text)


def test_repository_survey_is_consistent() -> None:
    errors = check(SURVEY.read_text(encoding="utf-8"))
    assert errors == []


def test_repository_survey_has_findings_in_every_severity() -> None:
    findings = parse_findings(SURVEY.read_text(encoding="utf-8"))
    assert {f.severity for f in findings} == {"high", "medium", "low"}
    assert "Open findings: **" in render_totals(findings)


def test_minimal_survey_passes() -> None:
    assert check(_survey(_row())) == []


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row(status="fixed"), "status 'fixed'"),
        (_row().replace("| high |", "| severe |"), "severity 'severe'"),
        (_row().replace("| P1 |", "| P5 |"), "not P1–P4"),
        (_row().replace("| run |", "| kitchen |"), "area 'kitchen'"),
        (_row(anchor="src/summer4/results/result.py::class Gone:"), "no longer appears"),
        (_row(anchor="src/summer4/nowhere.py::x"), "does not exist"),
        (_row(status="deferred"), "must cite a futureplans/ note"),
    ],
)
def test_bad_rows_are_reported(row: str, message: str) -> None:
    errors = check(_survey(row, packages="| CP1 | Title | CX1 |"))
    assert any(message in error for error in errors), errors


def test_done_findings_may_lose_their_anchor() -> None:
    row = _row(status="done", anchor="src/summer4/results/result.py::class Gone:")
    assert check(_survey(row, packages="| CP1 | Title | |")) == []


def test_open_finding_needs_exactly_one_package() -> None:
    errors = check(_survey(_row(), packages="| CP1 | Title | |"))
    assert any("exactly one package, found 0" in e for e in errors)
    errors = check(_survey(_row(), packages="| CP1 | Title | CX1 |\n| CP2 | Other | CX1 |"))
    assert any("exactly one package, found 2" in e for e in errors)


def test_stale_totals_are_reported() -> None:
    text = _survey(_row()).replace("Open findings: **1**", "Open findings: **7**")
    assert any("totals line is stale" in e for e in check(text))
