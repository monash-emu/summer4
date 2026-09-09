"""Execute spike notebooks as plain Python (no IPython magics)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = Path(__file__).resolve().parent


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise TypeError(f"Unexpected notebook source type: {type(source)}")


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def run_notebook(path: Path) -> None:
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
                "spike notebooks must be plain Python."
            )
        code_cells += 1
        try:
            exec(compile(source, str(path), "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"{path.name} cell {index} failed: {exc}") from exc
    if code_cells == 0:
        raise AssertionError(f"{path.name} has no executable code cells")
    print(f"ok  {path.name}  ({code_cells} code cells)")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    paths = _notebook_paths()
    if not paths:
        print("No spike notebooks found.", file=sys.stderr)
        return 1
    for path in paths:
        run_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
