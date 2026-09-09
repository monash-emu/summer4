"""Point this repository at ``.githooks`` so commit and merge hooks run."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".githooks"


def main() -> int:
    if not HOOKS.is_dir():
        print(f"Missing hooks directory: {HOOKS}", file=sys.stderr)
        return 1
    found = False
    for hook in HOOKS.iterdir():
        if not hook.is_file() or hook.name.startswith("."):
            continue
        found = True
        mode = hook.stat().st_mode
        hook.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if not found:
        print(f"No hooks in {HOOKS}", file=sys.stderr)
        return 1
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=ROOT,
        check=True,
    )
    print("Installed git hooks from .githooks (pre-commit: Black + cleared notebooks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
