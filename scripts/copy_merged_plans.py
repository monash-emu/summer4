"""Copy incoming ``*.plan.md`` files into ``plans/`` after a merge.

Git already brings committed ``plans/`` files across on merge. This script
is the safety net: any ``*.plan.md`` added by the merge under ``.cursor/plans/``
(or elsewhere in the tree) is copied into ``plans/`` so the workspace archive
stays complete.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS_DIR = ROOT / "plans"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _incoming_files() -> list[Path]:
    """Return plan files introduced by the merge or the latest commit."""
    merge_head = ROOT / ".git" / "MERGE_HEAD"
    ranges: list[str] = []
    if merge_head.exists():
        ranges.append("ORIG_HEAD...MERGE_HEAD")
    elif _git("rev-parse", "-q", "--verify", "ORIG_HEAD").strip():
        ranges.append("ORIG_HEAD...HEAD")
    else:
        ranges.append("HEAD")

    names: set[str] = set()
    for rev_range in ranges:
        if rev_range == "HEAD":
            out = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        else:
            out = _git("diff", "--name-only", "--diff-filter=A", rev_range)
            if not out.strip():
                out = _git("diff", "--name-only", "--diff-filter=AM", rev_range)
        names.update(line.strip() for line in out.splitlines() if line.strip())
    return [ROOT / name for name in sorted(names) if name.endswith(".plan.md")]


def copy_plans(*, dest: Path = PLANS_DIR) -> list[Path]:
    """Copy incoming plan files into ``dest``. Return paths that were copied."""
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in _incoming_files():
        if not src.is_file():
            continue
        try:
            src.relative_to(dest)
            continue
        except ValueError:
            pass
        target = dest / src.name
        if target.exists() and target.read_bytes() == src.read_bytes():
            continue
        shutil.copy2(src, target)
        copied.append(target)
    return copied


def main() -> int:
    copied = copy_plans()
    if copied:
        rel = ", ".join(str(path.relative_to(ROOT)) for path in copied)
        print(f"Copied merged plan(s) into plans/: {rel}", file=sys.stderr)
        print("Commit the copied plan files if they are not already tracked.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
