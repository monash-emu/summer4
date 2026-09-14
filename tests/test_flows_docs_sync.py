"""The published flows-spike pages must not drift from the tested notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# (spike notebook, published copy, number of docs-only code cells inserted)
PAIRS: list[tuple[str, str, int]] = [
    ("explorations/flows/01-flows.ipynb", "docs/dev/flows/01-flow-types.ipynb", 0),
    (
        "explorations/flows/02-flows.ipynb",
        "docs/dev/flows/02-derived-rates-and-adjustments.ipynb",
        1,
    ),
]


def _code_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


@pytest.mark.parametrize(
    ("spike", "published", "extra"),
    PAIRS,
    ids=lambda value: Path(value).name if isinstance(value, str) else str(value),
)
def test_published_code_matches_spike(spike: str, published: str, extra: int) -> None:
    """Every spike code cell appears verbatim, in order, in the published page."""
    spike_path = ROOT / spike
    published_path = ROOT / published
    assert spike_path.is_file(), f"missing spike notebook: {spike}"
    assert published_path.is_file(), f"missing published notebook: {published}"

    spike_cells = _code_sources(spike_path)
    published_cells = _code_sources(published_path)

    assert len(published_cells) == len(spike_cells) + extra, (
        f"{published} has {len(published_cells)} code cells; expected "
        f"{len(spike_cells)} from {spike} plus {extra} docs-only cell(s)."
    )
    assert published_cells[extra:] == spike_cells, (
        f"{published} has drifted from {spike}. Regenerate the published copy "
        "rather than editing it in place."
    )


@pytest.mark.parametrize(
    ("spike", "published", "extra"),
    PAIRS,
    ids=lambda value: Path(value).name if isinstance(value, str) else str(value),
)
def test_published_pages_are_marked_as_a_spike(spike: str, published: str, extra: int) -> None:
    """The published copy must warn that it is not the public API."""
    text = (ROOT / published).read_text(encoding="utf-8")
    assert "not the summer4 API" in text.lower() or "not** importable" in text
    assert "explorations/flows" in text
