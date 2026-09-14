"""The coverage ledger must parse, and the totals quoted elsewhere must match."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.coverage_report import (
    LEDGER,
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
    """Every API row has a unique ID and valid statuses in both columns."""
    rows = read_block(text, "api")
    ids = [row[0] for row in rows]
    assert len(set(ids)) == len(ids), "duplicate ledger IDs"
    for row in rows:
        assert len(row) == 7, f"row {row[0]!r} has {len(row)} cells, expected 7"
        assert row[3] in STATUSES, f"row {row[0]!r} has bad Shipped status {row[3]!r}"
        assert row[4] in STATUSES, f"row {row[0]!r} has bad Spike status {row[4]!r}"


def test_spike_never_worse_than_shipped(text: str) -> None:
    """Promoting the spike cannot remove a capability that already ships."""
    rank = {"none": 0, "partial": 1, "full": 2}
    for row in read_block(text, "api"):
        assert rank[row[4]] >= rank[row[3]], (
            f"row {row[0]!r} claims the spike is worse than what ships "
            f"({row[3]!r} -> {row[4]!r})"
        )


@pytest.mark.parametrize(
    ("block", "columns"),
    [("api", (3, 4)), ("textbook", (2, 3)), ("summer2docs", (1, 2))],
)
def test_tallies_are_exhaustive(text: str, block: str, columns: tuple[int, int]) -> None:
    rows = read_block(text, block)
    for column in columns:
        counts = tally(rows, column)
        assert counts.total == len(rows)
        assert counts.covered == counts.full + counts.partial


def test_every_area_is_named(text: str) -> None:
    from scripts.coverage_report import AREA_TITLES

    areas = by_area(read_block(text, "api"), 3)
    unknown = set(areas) - set(AREA_TITLES)
    assert not unknown, f"areas missing a title in AREA_TITLES: {sorted(unknown)}"


def test_quoted_totals_are_current(text: str) -> None:
    """evaluation/index.md must quote the ledger's own arithmetic."""
    totals = quoted_totals(text)
    index = (ROOT / "docs" / "evaluation" / "index.md").read_text(encoding="utf-8")
    assert f"**{totals['shipped_complete']} of {totals['api_total']}" in index
    assert f"**{totals['spike_complete']} of {totals['api_total']}" in index


def test_check_mode_passes() -> None:
    assert main(["--check"]) == 0
