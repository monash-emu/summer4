"""The step runbook must parse, and its position must be unambiguous."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.coverage_report import LEDGER, read_packages
from scripts.roadmap_check import (
    BRANCH,
    CLOSES,
    CURRENT_STATUSES,
    PLAN,
    REQUIRED_HEADINGS,
    ROADMAP,
    STATUS,
    STATUSES,
    STEP,
    WP,
    check,
    known_ids,
    main,
    read_current,
    read_roadmap_block,
    step_sections,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def text() -> str:
    return ROADMAP.read_text(encoding="utf-8")


def test_roadmap_exists() -> None:
    assert ROADMAP.is_file(), f"the step runbook is missing: {ROADMAP}"


def test_steps_block_parses(text: str) -> None:
    rows = read_roadmap_block(text, "steps")
    assert rows, "the runbook lists no steps"
    assert all(len(row) == 8 for row in rows), "every step row needs eight cells"


def test_step_numbers_are_consecutive(text: str) -> None:
    numbers = [int(row[STEP]) for row in read_roadmap_block(text, "steps")]
    assert numbers == list(range(1, len(numbers) + 1))


def test_statuses_are_known(text: str) -> None:
    for row in read_roadmap_block(text, "steps"):
        assert row[STATUS] in STATUSES, f"step {row[STEP]} has status {row[STATUS]!r}"


def test_exactly_one_current_step(text: str) -> None:
    rows = read_roadmap_block(text, "steps")
    current = [row for row in rows if row[STATUS] in CURRENT_STATUSES]
    assert len(current) == 1, f"{len(current)} steps are current; exactly one must be"


def test_current_block_agrees_with_the_table(text: str) -> None:
    current = read_current(text)
    rows = {int(row[STEP]): row for row in read_roadmap_block(text, "steps")}
    row = rows[int(current["Step"])]
    assert current["Status"] == row[STATUS]
    assert current["Branch"] == row[BRANCH]


def test_every_cited_plan_exists(text: str) -> None:
    for row in read_roadmap_block(text, "steps"):
        plan = ROOT / row[PLAN]
        assert plan.is_file(), f"step {row[STEP]} cites a missing plan: {row[PLAN]}"


def test_work_packages_are_declared(text: str) -> None:
    declared = {pkg for pkg, _name, _closes in read_packages(LEDGER.read_text(encoding="utf-8"))}
    for row in read_roadmap_block(text, "steps"):
        if row[WP] not in {"—", ""}:
            assert row[WP] in declared, f"step {row[STEP]} names undeclared {row[WP]}"


def test_closed_ids_exist(text: str) -> None:
    ids = known_ids()
    for row in read_roadmap_block(text, "steps"):
        if row[CLOSES] in {"—", ""}:
            continue
        for row_id in row[CLOSES].split():
            assert row_id in ids, f"step {row[STEP]} closes unknown ID {row_id!r}"


def test_every_step_has_a_followable_section(text: str) -> None:
    """A low-context worker reads these five headings; none may be missing."""
    sections = step_sections(text)
    for row in read_roadmap_block(text, "steps"):
        number = int(row[STEP])
        assert number in sections, f"step {number} has no section"
        for heading in REQUIRED_HEADINGS:
            assert f"### {heading}" in sections[number], f"step {number} lacks '### {heading}'"


def test_check_reports_nothing(text: str) -> None:
    assert check(text) == []


def test_main_succeeds() -> None:
    assert main(["--check"]) == 0
