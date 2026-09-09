"""Reject notebooks that still contain outputs or execution counts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_dirt(path: Path) -> list[str]:
    """Return human-readable problems that mean ``path`` is not cleared."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: not a readable notebook ({exc})"]

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return [f"{path}: missing cells list"]

    problems: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            problems.append(f"{path}: cell {index} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue
        count = cell.get("execution_count")
        if count is not None:
            problems.append(f"{path}: cell {index} has execution_count={count}")
        outputs = cell.get("outputs", [])
        if outputs:
            problems.append(f"{path}: cell {index} has {len(outputs)} output(s)")
    return problems


def check_paths(paths: list[Path]) -> list[str]:
    """Collect dirt reports for every notebook path."""
    problems: list[str] = []
    for path in paths:
        problems.extend(notebook_dirt(path))
    return problems


def _default_notebooks() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Notebooks to check (default: every git-tracked *.ipynb).",
    )
    args = parser.parse_args(argv)
    paths = list(args.paths) if args.paths else _default_notebooks()
    problems = check_paths(paths)
    if problems:
        print("Notebooks must be cleared (no outputs, no execution_count):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"Checked {len(paths)} cleared notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
