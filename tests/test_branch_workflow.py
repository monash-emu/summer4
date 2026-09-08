"""Tests for plan-copy and feature-branch checks."""

from __future__ import annotations

from pathlib import Path

from scripts.check_feature_branch import _under
from scripts.copy_merged_plans import copy_plans


def test_copy_plans_archives_cursor_plans(tmp_path: Path, monkeypatch: object) -> None:
    dest = tmp_path / "plans"
    incoming = tmp_path / ".cursor" / "plans"
    incoming.mkdir(parents=True)
    src = incoming / "demo.plan.md"
    src.write_text("# demo plan\n", encoding="utf-8")

    def fake_incoming() -> list[Path]:
        return [src]

    monkeypatch.setattr("scripts.copy_merged_plans._incoming_files", fake_incoming)
    copied = copy_plans(dest=dest)
    assert copied == [dest / "demo.plan.md"]
    assert (dest / "demo.plan.md").read_text(encoding="utf-8") == "# demo plan\n"

    copied_again = copy_plans(dest=dest)
    assert copied_again == []


def test_under_filters_paths() -> None:
    files = ["src/summer4/foo.py", "tests/test_foo.py", "README.md"]
    assert _under("src/summer4", files) == ["src/summer4/foo.py"]
    assert _under("tests", files) == ["tests/test_foo.py"]
    assert _under("plans", files) == []
