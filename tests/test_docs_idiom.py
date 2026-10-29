"""The published docs must not use removed or legacy spellings.

Walks every ``.md`` and ``.ipynb`` under ``docs/`` (except build output, the
developer and evaluation records, and the unbuilt ``guide/_sources`` archive)
and fails, naming file and line, on a forbidden idiom. ``docs/roadmap.md`` and
``docs/releases.md`` may mention the names, because they list legacy and
removed spellings. Textbook files are expected to fail until step C6 migrates
them, so they are marked non-strict xfail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
EXCLUDED_TOP = {"_build", "dev", "evaluation"}
ALLOWED_FILES = {DOCS / "roadmap.md", DOCS / "releases.md"}

FORBIDDEN = [
    "derived_fn",
    "derived_refs",
    "SaveFn",
    "ComputedValue",
    "sum_over",
    ".total(",
    ".cumulative(",
]
# ``kind=`` followed by a lambda, or by a bare lowercase identifier. Strings
# (quote), ``FOIKind.X`` (capitalised) and other non-identifier starts pass.
KIND_CALLABLE = re.compile(r"\bkind\s*=\s*(lambda\b|[a-z_][A-Za-z0-9_]*\b)")


def _is_excluded(path: Path) -> bool:
    rel = path.relative_to(DOCS).parts
    if rel[0] in EXCLUDED_TOP:
        return True
    return rel[:2] == ("guide", "_sources")


def _doc_files() -> list[Path]:
    files = [p for p in DOCS.rglob("*") if p.suffix in {".md", ".ipynb"}]
    return sorted(p for p in files if not _is_excluded(p))


def _lines(path: Path) -> list[tuple[str, int, str]]:
    """Return ``(location, number, text)`` for each source line in the file."""
    if path.suffix == ".md":
        return [("", i, t) for i, t in enumerate(path.read_text().splitlines(), 1)]
    out: list[tuple[str, int, str]] = []
    nb = json.loads(path.read_text())
    for c, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        for i, t in enumerate(src.splitlines(), 1):
            out.append((f"cell {c} ({cell['cell_type']}), ", i, t))
    return out


def _violations(path: Path) -> list[str]:
    found: list[str] = []
    for where, n, text in _lines(path):
        for word in FORBIDDEN:
            if word in text:
                found.append(f"{path.relative_to(DOCS)}: {where}line {n}: {word!r}")
        m = KIND_CALLABLE.search(text)
        if m:
            found.append(f"{path.relative_to(DOCS)}: {where}line {n}: kind={m.group(1)}")
    return found


def _param(path: Path) -> object:
    marks: list[pytest.MarkDecorator] = []
    if path.relative_to(DOCS).parts[0] == "textbook":
        marks.append(pytest.mark.xfail(strict=False, reason="textbook migration lands in C6"))
    return pytest.param(path, marks=marks, id=str(path.relative_to(DOCS)))


@pytest.mark.parametrize("path", [_param(p) for p in _doc_files()])
def test_no_forbidden_idioms(path: Path) -> None:
    if path in ALLOWED_FILES:
        pytest.skip("lists legacy and removed spellings")
    assert _violations(path) == []
