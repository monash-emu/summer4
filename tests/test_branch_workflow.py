"""Tests for plan-copy and feature-branch checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.check_feature_branch import _under
from scripts.copy_merged_plans import copy_plans

ROOT = Path(__file__).resolve().parents[1]


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


def _git(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _merge_a_cursor_plan(tmp_path: Path, path: str) -> tuple[Path, str]:
    """Merge a branch that adds ``.cursor/plans/x.plan.md`` in a scratch repo.

    The repo carries this checkout's real ``post-merge`` hook and copy script,
    and runs with ``PATH=path``. Returns the repo and the merge's stderr.
    """
    git = shutil.which("git")
    assert git is not None
    repo = tmp_path / "repo"
    (repo / ".githooks").mkdir(parents=True)
    (repo / "scripts").mkdir()
    shutil.copy2(ROOT / ".githooks" / "post-merge", repo / ".githooks" / "post-merge")
    shutil.copy2(ROOT / "scripts" / "copy_merged_plans.py", repo / "scripts")
    env = {"PATH": path, "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}

    subprocess.run([git, "init", "-q", "-b", "main"], cwd=repo, env=env, check=True)
    _git(repo, env, "config", "core.hooksPath", ".githooks")
    _git(repo, env, "add", ".")
    _git(repo, env, "commit", "-q", "-m", "hooks")
    _git(repo, env, "switch", "-q", "-c", "feat/x")
    plan = repo / ".cursor" / "plans" / "x.plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# x plan\n", encoding="utf-8")
    _git(repo, env, "add", ".")
    _git(repo, env, "commit", "-q", "-m", "plan")
    _git(repo, env, "switch", "-q", "main")
    merged = _git(repo, env, "merge", "--no-ff", "-m", "merge feat/x", "feat/x")
    return repo, merged.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh hook")
def test_post_merge_hook_copies_plans_without_bare_python(tmp_path: Path) -> None:
    """Regression: the hook called bare ``python``, absent outside pixi."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    trap = fake_bin / "python"
    trap.write_text("#!/bin/sh\necho 'bare python was called' >&2\nexit 127\n")
    trap.chmod(0o755)
    (fake_bin / "python3").symlink_to(sys.executable)

    repo, stderr = _merge_a_cursor_plan(tmp_path, f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    assert (repo / "plans" / "x.plan.md").read_text(encoding="utf-8") == "# x plan\n"
    assert "Copied merged plan(s) into plans/: plans/x.plan.md" in stderr
    assert "bare python was called" not in stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh hook")
def test_post_merge_hook_warns_without_any_python(tmp_path: Path) -> None:
    git = shutil.which("git")
    assert git is not None
    only_git = tmp_path / "only-git"
    only_git.mkdir()
    (only_git / "git").symlink_to(git)

    repo, stderr = _merge_a_cursor_plan(tmp_path, str(only_git))
    assert not (repo / "plans").exists()
    assert "post-merge: no Python found" in stderr
