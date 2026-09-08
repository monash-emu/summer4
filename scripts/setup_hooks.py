"""Point this repository at ``.githooks`` so merges copy plans into ``plans/``."""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".githooks"


def main() -> int:
    hook = HOOKS / "post-merge"
    if not hook.is_file():
        print(f"Missing hook: {hook}", file=sys.stderr)
        return 1
    mode = hook.stat().st_mode
    hook.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=ROOT,
        check=True,
    )
    print("Installed git hooks from .githooks (post-merge copies plans into plans/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
