"""Three-way timer for summer4 compiled runs: build, first call, warm calls.

Build time is model construction plus ``compile`` and excludes the first
call. Compile time is that first call. Warm time is each later call. Every
call ends in ``jax.block_until_ready`` on whatever ``call`` returns, which
must be a reduced output so XLA cannot delete the solve.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class TimedRun:
    """Host times for one built model. Warm times are not mixed into compile."""

    build_s: float
    compile_s: float
    warm_s: tuple[float, ...]


def time_run(
    build: Callable[[], Any],
    call: Callable[[Any], Any],
    n_warm: int = 1,
) -> tuple[Any, TimedRun]:
    """Time ``build``, one compile call, then ``n_warm`` warm calls.

    Returns the object ``build`` produced and the three times. ``call`` is
    invoked on that object; its return value is passed to
    ``jax.block_until_ready``.
    """
    if n_warm < 1:
        raise ValueError(f"n_warm must be at least 1, got {n_warm}")

    started = time.perf_counter()
    built = build()
    build_s = time.perf_counter() - started

    started = time.perf_counter()
    jax.block_until_ready(call(built))
    compile_s = time.perf_counter() - started

    warm: list[float] = []
    for _ in range(n_warm):
        started = time.perf_counter()
        jax.block_until_ready(call(built))
        warm.append(time.perf_counter() - started)

    return built, TimedRun(build_s=build_s, compile_s=compile_s, warm_s=tuple(warm))
