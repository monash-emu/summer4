"""Gate feature branches: tests, notebooks, and a plan must land with src changes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD").strip()


def _merge_base(base: str) -> str | None:
    result = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _changed_files(base: str) -> list[str]:
    merge_base = _merge_base(base)
    if merge_base is None:
        return []
    out = _git("diff", "--name-only", f"{merge_base}...HEAD")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _under(prefix: str, files: list[str]) -> list[str]:
    return [path for path in files if path == prefix or path.startswith(prefix + "/")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="main",
        help="Branch to diff against (default: main).",
    )
    args = parser.parse_args()

    branch = _current_branch()
    if branch == args.base:
        print(f"On {args.base}; feature-branch checks skipped.")
        return 0

    files = _changed_files(args.base)
    if not files:
        print(f"No changes versus {args.base}; nothing to check.")
        return 0

    src_py = [path for path in _under("src/summer4", files) if path.endswith(".py")]
    if not src_py:
        print("No src/summer4 Python changes; feature acceptance bar not required.")
        return 0

    errors: list[str] = []
    if not _under("tests", files):
        errors.append("src/summer4 changed but tests/ did not. Add tests for the new behaviour.")
    if not _under("examples/notebooks", files):
        errors.append(
            "src/summer4 changed but examples/notebooks/ did not. "
            "Add or update a runnable example notebook."
        )
    if not _under("plans", files) and not _under(".cursor/plans", files):
        errors.append(
            "src/summer4 changed but no plan was added under plans/ "
            "(or .cursor/plans/). Copy the accepted plan onto the branch."
        )

    if errors:
        print(f"Feature-branch check failed on {branch} (versus {args.base}):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"{branch}: tests, notebook, and plan are present for src/summer4 changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
