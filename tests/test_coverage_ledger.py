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


@pytest.fixture(scope="module")
def ports_text() -> str:
    from scripts.coverage_report import PORTS

    return PORTS.read_text(encoding="utf-8")


def test_ports_ledger_parses(ports_text: str, text: str) -> None:
    from scripts.coverage_report import PORT_MODELS, read_ports

    rows = read_ports(ports_text, text)
    assert {row[1] for row in rows} == set(PORT_MODELS)


def test_ports_reject_bad_status(ports_text: str, text: str) -> None:
    from scripts.coverage_report import read_ports

    broken = ports_text.replace(
        "| KI1 | Kiribati | State × 8 uneven age bands × reachability map | `full` |",
        "| KI1 | Kiribati | State × 8 uneven age bands × reachability map | `mostly` |",
        1,
    )
    with pytest.raises(ValueError, match="Unknown status"):
        read_ports(broken, text)


def test_ports_reject_undeclared_package(ports_text: str, text: str) -> None:
    from scripts.coverage_report import read_ports

    # All WP10 port rows are full/`—` after step 15; demote KI22 and point it at WP99.
    broken = ports_text.replace(
        "| KI22 | Kiribati | Verified compile time, step cost and solver safety"
        " at TB scale | `full` |",
        "| KI22 | Kiribati | Verified compile time, step cost and solver safety"
        " at TB scale | `none` |",
        1,
    ).replace(
        "`MixingMatrix.validate` / vmap-safe reciprocity | — |",
        "`MixingMatrix.validate` / vmap-safe reciprocity | WP99 |",
        1,
    )
    with pytest.raises(ValueError, match="undeclared package"):
        read_ports(broken, text)


def test_ports_reject_full_row_with_package(ports_text: str, text: str) -> None:
    from scripts.coverage_report import read_ports

    broken = ports_text.replace("| `ComputedValue` over `derived_fn` | — |", "| x | WP13 |", 1)
    with pytest.raises(ValueError, match="is full but names"):
        read_ports(broken, text)


def test_port_readiness_is_current(ports_text: str, text: str) -> None:
    from scripts.coverage_report import update_port_readiness

    assert update_port_readiness(ports_text, text) == ports_text


def test_port_readiness_reaches_every_row(ports_text: str, text: str) -> None:
    """After the last ordered package, every port row is complete."""
    from scripts.coverage_report import port_readiness

    _label, final = port_readiness(ports_text, text)[-1]
    assert all(full == total for full, total in final.values())


def test_port_quotes_are_current(ports_text: str, text: str) -> None:
    from scripts.coverage_report import stale_port_quotes

    assert stale_port_quotes(ports_text, text) == []


def test_workflow_ledger_parses(text: str) -> None:
    from scripts.coverage_report import read_workflow

    rows = read_workflow(text)
    assert [row[0] for row in rows] == [f"CW{i}" for i in range(1, len(rows) + 1)]


def test_workflow_rejects_bad_status(text: str) -> None:
    from scripts.coverage_report import read_workflow

    # Force an illegal status on CW1 regardless of its current Status cell.
    broken = text.replace(
        "| CW1 | Latin hypercube design over the priors (M points) | `full` |",
        "| CW1 | Latin hypercube design over the priors (M points) | `nope` |",
        1,
    ).replace(
        "| CW1 | Latin hypercube design over the priors (M points) | `none` |",
        "| CW1 | Latin hypercube design over the priors (M points) | `nope` |",
        1,
    )
    with pytest.raises(ValueError, match="Unknown status"):
        read_workflow(broken)


def test_workflow_rejects_duplicate_id(text: str) -> None:
    from scripts.coverage_report import read_workflow

    broken = text.replace("| CW2 |", "| CW1 |", 1)
    with pytest.raises(ValueError, match="Duplicate workflow row ID"):
        read_workflow(broken)


def test_workflow_quote_is_current(text: str) -> None:
    from scripts.coverage_report import stale_workflow_quote

    assert stale_workflow_quote(text) is None


def test_workflow_stays_out_of_api_totals(text: str) -> None:
    """CW rows are tallied apart: they never change the summer2 API denominator."""
    api_ids = {row[0] for row in read_block(text, "api")}
    assert not any(row_id.startswith("CW") for row_id in api_ids)
