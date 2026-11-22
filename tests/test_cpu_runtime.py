"""XLA:CPU runtime chosen on import (``summer4._cpu_runtime``).

Each process case runs a fresh interpreter, because JAX reads ``XLA_FLAGS`` once, when its
backend starts. The child prints one JSON object on its last line. Assertions that depend on
the installed jaxlib check :func:`has_legacy_runtime`, so the suite passes on the nightly
``latest`` environment too.
"""

import json
import os
import subprocess
import sys
from importlib import metadata
from typing import Any

import pytest

from summer4._cpu_runtime import has_legacy_runtime, has_scheduler_flag

_FLAG = "--xla_cpu_use_thunk_runtime"
_SCHED = "--xla_cpu_enable_concurrency_optimized_scheduler"
JAXLIB = metadata.version("jaxlib")
LEGACY = has_legacy_runtime(JAXLIB)
SCHEDULER = has_scheduler_flag(JAXLIB)

_CHILD = """
import json, os, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import summer4
    from summer4 import runtime_info
    from jax._src import xla_bridge
    info = runtime_info()
    started = bool(xla_bridge.backends_are_initialized())
    import jax.numpy as jnp
    total = float(jnp.arange(4.0).sum())
    out = {
        "started_after_import": started,
        "runtime": info.cpu_runtime,
        "source": info.cpu_runtime_source,
        "scheduler": info.cpu_concurrency_scheduler,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "total": total,
        "warnings": [str(w.message) for w in caught],
    }
print(json.dumps(out))
"""

_CHILD_TOO_LATE = """
import json, warnings
import jax
jax.devices()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import summer4
    from summer4 import runtime_info
    info = runtime_info()
    out = {
        "runtime": info.cpu_runtime,
        "source": info.cpu_runtime_source,
        "warnings": [str(w.message) for w in caught],
    }
print(json.dumps(out))
"""


def _run(code: str, **overrides: str) -> dict[str, Any]:
    env = dict(os.environ)
    for key in ("XLA_FLAGS", "SUMMER4_CPU_RUNTIME", "JAX_PLATFORMS"):
        env.pop(key, None)
    env["JAX_PLATFORMS"] = "cpu"
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.4.31", False),
        ("0.4.32", True),
        ("0.5.3", True),
        ("0.6.2", True),
        ("0.6.2.dev20250601", True),
        ("0.7.0", False),
        ("0.7.2", False),
        ("0.11.1", False),
        (None, False),
    ],
)
def test_legacy_runtime_version_range(version: str | None, expected: bool) -> None:
    assert has_legacy_runtime(version) is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.6.2", False),
        ("0.7.0", True),
        ("0.7.1", True),
        ("0.11.1", True),
        ("0.12.1", True),
        ("0.12.0.1", True),
        ("0.13.0", False),
        (None, False),
    ],
)
def test_scheduler_flag_version_range(version: str | None, expected: bool) -> None:
    assert has_scheduler_flag(version) is expected


def test_default_selects_legacy_where_available() -> None:
    out = _run(_CHILD)
    assert out["total"] == 6.0
    assert out["warnings"] == []
    if LEGACY:
        assert out["runtime"] == "legacy"
        assert out["source"] == "default"
        assert f"{_FLAG}=false" in out["xla_flags"]
        assert out["scheduler"] is None
    else:
        assert out["source"] == "jaxlib"
        assert _FLAG not in out["xla_flags"]
    assert (f"{_SCHED}=false" in out["xla_flags"]) is SCHEDULER
    if SCHEDULER:
        assert (out["runtime"], out["scheduler"]) == ("thunk", False)


@pytest.mark.skipif(not SCHEDULER, reason="summer4 leaves the scheduler alone on this jaxlib")
def test_user_scheduler_flag_wins() -> None:
    out = _run(_CHILD, XLA_FLAGS=f"{_SCHED}=true")
    assert out["xla_flags"].count(_SCHED) == 1
    assert out["scheduler"] is True
    assert out["total"] == 6.0


def test_import_does_not_start_backend() -> None:
    assert _run(_CHILD)["started_after_import"] is False


@pytest.mark.parametrize("value", ["thunk", "0", "off"])
def test_env_opts_out(value: str) -> None:
    out = _run(_CHILD, SUMMER4_CPU_RUNTIME=value)
    assert _FLAG not in out["xla_flags"]
    assert _SCHED not in out["xla_flags"]
    assert out["warnings"] == []
    if LEGACY or SCHEDULER:
        assert out["scheduler"] is True
    if LEGACY:
        assert (out["runtime"], out["source"]) == ("thunk", "SUMMER4_CPU_RUNTIME")


def test_user_xla_flags_win() -> None:
    out = _run(_CHILD, XLA_FLAGS=f"{_FLAG}=true")
    assert out["source"] == "XLA_FLAGS"
    assert out["runtime"] == "thunk"
    assert out["xla_flags"].count(_FLAG) == 1


def test_invalid_env_value_warns_and_uses_default() -> None:
    out = _run(_CHILD, SUMMER4_CPU_RUNTIME="fast")
    assert any("SUMMER4_CPU_RUNTIME" in message for message in out["warnings"])
    assert out["total"] == 6.0


@pytest.mark.skipif(LEGACY, reason="this jaxlib has the legacy runtime")
def test_legacy_request_without_legacy_runtime_warns() -> None:
    out = _run(_CHILD, SUMMER4_CPU_RUNTIME="legacy")
    assert any("no legacy XLA:CPU runtime" in message for message in out["warnings"])
    assert _FLAG not in out["xla_flags"]


@pytest.mark.skipif(not (LEGACY or SCHEDULER), reason="summer4 sets no XLA:CPU flag here")
def test_too_late_import_warns() -> None:
    out = _run(_CHILD_TOO_LATE)
    assert (out["runtime"], out["source"]) == ("thunk", "too-late")
    expected = "legacy CPU runtime" if LEGACY else "concurrency-optimised CPU scheduler"
    assert any(expected in message for message in out["warnings"])
