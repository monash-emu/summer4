"""Host CPU devices set on import (plan R1, decision D1).

Each case runs a fresh interpreter, because JAX reads ``XLA_FLAGS`` once, when its backend starts.
The child prints one JSON object on its last line.
"""

import json
import os
import subprocess
import sys
from typing import Any

_FLAG = "--xla_force_host_platform_device_count"

_CHILD_DEVICES = """
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import summer4
    from summer4 import runtime_info
    from jax._src import xla_bridge
    info = runtime_info()
    started = bool(xla_bridge.backends_are_initialized())
    import jax
    out = {
        "started_after_import": started,
        "device_count": jax.device_count(),
        "requested": info.host_devices_requested,
        "source": info.host_devices_source,
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
        "device_count": jax.device_count(),
        "requested": info.host_devices_requested,
        "source": info.host_devices_source,
        "warnings": [str(w.message) for w in caught],
    }
print(json.dumps(out))
"""

_CHILD_SUM = """
import json
import summer4
import jax.numpy as jnp
import jax
print(json.dumps({"devices": jax.device_count(), "sum": float(jnp.ones(3).sum())}))
"""


def _run(code: str, **overrides: str) -> dict[str, Any]:
    env = dict(os.environ)
    for key in ("XLA_FLAGS", "SUMMER4_HOST_DEVICES", "JAX_PLATFORMS"):
        env.pop(key, None)
    env["JAX_PLATFORMS"] = "cpu"
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def test_default_gives_one_device_per_cpu() -> None:
    out = _run(_CHILD_DEVICES)
    assert out["device_count"] == (os.cpu_count() or 1)
    assert out["source"] == "default"
    assert out["requested"] == (os.cpu_count() or 1)
    assert out["warnings"] == []


def test_host_devices_env_sets_count() -> None:
    out = _run(_CHILD_DEVICES, SUMMER4_HOST_DEVICES="2")
    assert out["device_count"] == 2
    assert out["requested"] == 2
    assert out["source"] == "SUMMER4_HOST_DEVICES"


def test_host_devices_zero_disables() -> None:
    out = _run(_CHILD_DEVICES, SUMMER4_HOST_DEVICES="0")
    assert out["device_count"] == 1
    assert out["requested"] is None
    assert out["source"] == "disabled"


def test_xla_flags_count_is_kept() -> None:
    out = _run(_CHILD_DEVICES, XLA_FLAGS=f"{_FLAG}=3")
    assert out["device_count"] == 3
    assert out["requested"] == 3
    assert out["source"] == "XLA_FLAGS"


def test_too_late_import_warns_and_keeps_one_device() -> None:
    out = _run(_CHILD_TOO_LATE)
    assert out["device_count"] == 1
    assert out["requested"] is None
    assert out["source"] == "too-late"
    assert any("before summer4 was imported" in message for message in out["warnings"])


def test_import_does_not_start_backend() -> None:
    out = _run(_CHILD_DEVICES)
    assert out["started_after_import"] is False


def test_result_is_the_same_on_one_and_many_devices() -> None:
    one = _run(_CHILD_SUM, SUMMER4_HOST_DEVICES="1")
    many = _run(_CHILD_SUM, SUMMER4_HOST_DEVICES="4")
    assert one["devices"] == 1
    assert many["devices"] == 4
    assert one["sum"] == many["sum"] == 3.0
