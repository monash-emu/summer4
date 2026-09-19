"""Three-way timer for summer2 runners: build, first call, warm calls.

Build time is model construction plus ``get_runner`` and excludes the first
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
    """Host times for one built runner. Warm times are not mixed into compile."""

    build_s: float
    compile_s: float
    warm_s: tuple[float, ...]


def time_runner(
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


def reduce_runner_outputs(
    runner: Any,
    parameters: dict[str, Any],
    flow_names: tuple[str, ...],
    seen: dict[str, Any] | None = None,
) -> Any:
    """Sum compartments and each named flow series.

    The return value is the reduced output later steps pass to
    ``jax.block_until_ready``. When ``seen`` is set, stash the arrays used
    for shape and dtype checks; those stashes are not a second timer.
    """
    results = runner.function(parameters=parameters)
    outputs = results["outputs"]
    derived = results["derived_outputs"]
    total = outputs.sum()
    flows: dict[str, Any] = {}
    for name in flow_names:
        series = derived[name]
        flows[name] = series
        total = total + series.sum()
    if seen is not None:
        seen["outputs"] = outputs
        seen["flows"] = flows
        seen["derived_keys"] = tuple(derived.keys())
        seen["reduced"] = total
    return total
