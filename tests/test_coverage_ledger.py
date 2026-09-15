"""The coverage ledger must parse, and the totals quoted elsewhere must match."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.coverage_report import (
    LEDGER,
    STATUS_COLUMN,
    STATUSES,
    by_area,
    main,
    quoted_totals,
    read_block,
    tally,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def text() -> str:
    return LEDGER.read_text(encoding="utf-8")


def test_ledger_exists() -> None:
    assert LEDGER.is_file(), f"the coverage ledger is missing: {LEDGER}"


@pytest.mark.parametrize("block", ["api", "textbook", "summer2docs"])
def test_blocks_parse(text: str, block: str) -> None:
    rows = read_block(text, block)
    assert rows, f"ledger block {block!r} has no rows"


def test_api_ledger_shape(text: str) -> None:
    """Every API row has a unique ID and a valid status."""
    rows = read_block(text, "api")
    ids = [row[0] for row in rows]
    assert len(set(ids)) == len(ids), "duplicate ledger IDs"
    for row in rows:
        assert len(row) == 6, f"row {row[0]!r} has {len(row)} cells, expected 6"
        assert row[3] in STATUSES, f"row {row[0]!r} has bad status {row[3]!r}"


def test_textbook_ledger_shape(text: str) -> None:
    for row in read_block(text, "textbook"):
        assert len(row) == 5, f"textbook row {row[0]!r} has {len(row)} cells, expected 5"


def test_summer2docs_ledger_shape(text: str) -> None:
    for row in read_block(text, "summer2docs"):
        assert len(row) == 4, f"summer2docs row {row[0]!r} has {len(row)} cells, expected 4"


def test_declared_ports_exist(text: str) -> None:
    from scripts.coverage_report import declared_ports

    docs_root = ROOT / "docs"
    missing = [
        (block, label, port)
        for block, label, port in declared_ports(text)
        if not (docs_root / port).is_file()
    ]
    assert not missing, f"Ported paths missing under docs/: {missing}"


@pytest.mark.parametrize("block", ["api", "textbook", "summer2docs"])
def test_tallies_are_exhaustive(text: str, block: str) -> None:
    rows = read_block(text, block)
    counts = tally(rows, STATUS_COLUMN[block])
    assert counts.total == len(rows)
    assert counts.covered == counts.full + counts.partial


def test_every_area_is_named(text: str) -> None:
    from scripts.coverage_report import AREA_TITLES

    areas = by_area(read_block(text, "api"), STATUS_COLUMN["api"])
    unknown = set(areas) - set(AREA_TITLES)
    assert not unknown, f"areas missing a title in AREA_TITLES: {sorted(unknown)}"


def test_quoted_totals_are_current(text: str) -> None:
    """evaluation/index.md must quote the ledger's own arithmetic."""
    totals = quoted_totals(text)
    index = (ROOT / "docs" / "evaluation" / "index.md").read_text(encoding="utf-8")
    assert f"**{totals['complete']} of {totals['api_total']}" in index


def test_check_mode_passes() -> None:
    assert main(["--check"]) == 0
