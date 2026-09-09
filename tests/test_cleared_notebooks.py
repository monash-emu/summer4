"""Tests for the cleared-notebook commit check."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_cleared_notebooks import check_paths, notebook_dirt


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}),
        encoding="utf-8",
    )
    return path


def test_cleared_code_cell_is_clean(tmp_path: Path) -> None:
    path = _write_notebook(
        tmp_path / "clean.ipynb",
        [{"cell_type": "code", "execution_count": None, "outputs": [], "source": ["1\n"]}],
    )
    assert notebook_dirt(path) == []


def test_execution_count_is_dirt(tmp_path: Path) -> None:
    path = _write_notebook(
        tmp_path / "counted.ipynb",
        [{"cell_type": "code", "execution_count": 3, "outputs": [], "source": ["1\n"]}],
    )
    problems = notebook_dirt(path)
    assert len(problems) == 1
    assert "execution_count=3" in problems[0]


def test_outputs_are_dirt(tmp_path: Path) -> None:
    path = _write_notebook(
        tmp_path / "printed.ipynb",
        [
            {
                "cell_type": "code",
                "execution_count": None,
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hi\n"]}],
                "source": ["print('hi')\n"],
            }
        ],
    )
    problems = notebook_dirt(path)
    assert len(problems) == 1
    assert "1 output" in problems[0]


def test_markdown_outputs_are_ignored(tmp_path: Path) -> None:
    path = _write_notebook(
        tmp_path / "docs.ipynb",
        [{"cell_type": "markdown", "source": ["# title\n"]}],
    )
    assert notebook_dirt(path) == []


def test_check_paths_collects_all_problems(tmp_path: Path) -> None:
    clean = _write_notebook(
        tmp_path / "clean.ipynb",
        [{"cell_type": "code", "execution_count": None, "outputs": [], "source": []}],
    )
    dirty = _write_notebook(
        tmp_path / "dirty.ipynb",
        [{"cell_type": "code", "execution_count": 1, "outputs": [], "source": []}],
    )
    problems = check_paths([clean, dirty])
    assert problems
    assert all("dirty.ipynb" in item for item in problems)


def test_example_notebooks_are_cleared() -> None:
    examples = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
    paths = sorted(examples.glob("*.ipynb"))
    assert paths
    assert check_paths(paths) == []
