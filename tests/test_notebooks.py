"""Execute example notebooks as smoke tests.

Every notebook run is marked ``notebook`` and ``slow``. They run through
``pixi run test-notebooks`` (locally and in the nightly workflow), not in
``pixi run test`` or per-PR CI. What only a notebook used to reach is tested in
``tests/test_notebook_gaps.py``. Each notebook has a time budget so a hang fails
that notebook instead of holding the job.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"

# Seconds per notebook. The default covers every notebook that takes under
# about a minute on a GitHub ubuntu runner; the rest get roughly twice their
# CI time (run 45735900709). Scale them all with SUMMER4_NOTEBOOK_TIMEOUT_SCALE
# on a slower machine.
DEFAULT_TIMEOUT_S = 120.0
TIMEOUT_S = {
    "22-calibration-seeded-mcmc.ipynb": 300.0,
    "24-calibration-entry-points.ipynb": 180.0,
    "28-calibration-custom-posterior.ipynb": 180.0,
    "29-calibration-nuts-tuning.ipynb": 720.0,
    "38-calibration-multi-model.ipynb": 420.0,
    "43-laplace-fd-hessian.ipynb": 240.0,
}

# Grace after the soft timeout before a C-level watchdog dumps every thread's
# stack and exits the process (for a hang inside compiled code, where the
# soft timeout's Python exception never gets a chance to run).
HARD_TIMEOUT_GRACE_S = 60.0


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _notebook_params() -> list[object]:
    marks = [pytest.mark.notebook, pytest.mark.slow]
    return [pytest.param(path, id=path.name, marks=marks) for path in _notebook_paths()]


def notebook_timeout(name: str) -> float:
    """Return the time budget, in seconds, for the notebook called ``name``."""
    scale = float(os.environ.get("SUMMER4_NOTEBOOK_TIMEOUT_SCALE", "1"))
    return TIMEOUT_S.get(name, DEFAULT_TIMEOUT_S) * scale


class NotebookTimeout(BaseException):
    """A notebook ran past its time budget.

    A ``BaseException`` so that a notebook's own ``except Exception`` cannot
    swallow it.
    """


@contextmanager
def _time_budget(seconds: float, label: str) -> Iterator[None]:
    """Fail with ``NotebookTimeout`` after ``seconds``; hard-exit after a grace period.

    The soft limit uses ``SIGALRM`` (not on Windows). The hard limit is
    ``faulthandler``'s watchdog thread, which works on every platform and fires
    even while the main thread is stuck in compiled code.
    """
    has_alarm = hasattr(signal, "SIGALRM")
    previous: object = None
    if has_alarm:

        def _on_alarm(signum: int, frame: FrameType | None) -> None:
            raise NotebookTimeout(f"{label} ran longer than {seconds:.0f} s")

        previous = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
    faulthandler.dump_traceback_later(seconds + HARD_TIMEOUT_GRACE_S, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)  # type: ignore[arg-type]


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    raise TypeError(f"Unexpected notebook source type: {type(source)}")


def test_example_notebooks_exist() -> None:
    assert NOTEBOOK_DIR.is_dir(), "examples/notebooks/ is missing"
    assert _notebook_paths(), "examples/notebooks/ must contain at least one .ipynb"


def test_timeout_names_are_notebooks() -> None:
    names = {path.name for path in _notebook_paths()}
    assert set(TIMEOUT_S) <= names, sorted(set(TIMEOUT_S) - names)


def test_time_budget_raises_on_overrun() -> None:
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("soft timeout needs SIGALRM")
    with pytest.raises(NotebookTimeout, match="ran longer than"), _time_budget(0.05, "spin"):
        while True:
            pass


@pytest.mark.parametrize("path", _notebook_params())
def test_notebook_executes(path: Path) -> None:
    """Run every code cell in-process (plain Python; no IPython magics)."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    code_cells = 0
    with _time_budget(notebook_timeout(path.name), path.name):
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
                    "example notebooks must be plain Python."
                )
            code_cells += 1
            try:
                exec(compile(source, str(path), "exec"), namespace)
            except NotebookTimeout as exc:
                raise AssertionError(f"{path.name} cell {index}: {exc}") from exc
            except Exception as exc:
                raise AssertionError(f"{path.name} cell {index} failed: {exc}") from exc
    assert code_cells > 0, f"{path.name} has no executable code cells"
