"""Execute example notebooks as smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise TypeError(f"Unexpected notebook source type: {type(source)}")


def test_example_notebooks_exist() -> None:
    assert NOTEBOOK_DIR.is_dir(), "examples/notebooks/ is missing"
    assert _notebook_paths(), "examples/notebooks/ must contain at least one .ipynb"


@pytest.mark.slow
@pytest.mark.parametrize("path", _notebook_paths(), ids=lambda path: path.name)
def test_notebook_executes(path: Path) -> None:
    """Run every code cell in-process (plain Python; no IPython magics)."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        stripped = source.strip()
        if not stripped:
            continue
        if stripped.startswith("%") or stripped.startswith("!"):
            raise AssertionError(
                f"{path.name} cell {index} uses a shell/IPython magic; "
                "example notebooks must be plain Python."
            )
        code_cells += 1
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"{path.name} cell {index} failed: {exc}") from exc
    assert code_cells > 0, f"{path.name} has no executable code cells"
