"""Host CPU devices: JAX sees one CPU device per core, unless the user chose otherwise.

JAX runs a computation on one device unless it is told to split work across several. The XLA
flag ``--xla_force_host_platform_device_count`` makes JAX's CPU platform report several devices,
so that later steps can spread work across them. summer4 sets the flag when imported:

- nothing set: one device per CPU core (``os.cpu_count()``);
- ``SUMMER4_HOST_DEVICES=<n>`` with ``n >= 1``: ``n`` devices;
- ``SUMMER4_HOST_DEVICES=0`` (or ``off``, ``false``, ``no``): nothing is set;
- ``XLA_FLAGS`` already sets the flag: nothing is set, because the user chose.

JAX reads the flag when its backend starts. If JAX was already used before summer4 was imported,
summer4 sets nothing and warns. The flag adds CPU devices only: on a GPU machine JAX still uses
the GPU by default, and ``JAX_PLATFORMS`` chooses the platform as before. The outcome is recorded
in ``CHOICE`` and reported by ``summer4.runtime_info()``.
"""

import os
import re
import warnings
from dataclasses import dataclass

_FLAG = "--xla_force_host_platform_device_count"
_ENV = "SUMMER4_HOST_DEVICES"
_DISABLED_VALUES = frozenset({"0", "off", "false", "no"})


@dataclass(frozen=True)
class HostDeviceChoice:
    """What summer4 did about host devices when it was imported.

    ``requested`` is the device count that applies: the count summer4 set, or the count the user
    gave in ``XLA_FLAGS`` (``None`` if that count cannot be read). It is ``None`` when summer4
    set nothing. ``source`` is one of ``"default"``, ``"SUMMER4_HOST_DEVICES"``, ``"XLA_FLAGS"``,
    ``"disabled"`` (nothing was set, by request or because the value was invalid) or
    ``"too-late"`` (JAX's backend had already started).
    """

    requested: int | None
    source: str


def _backend_initialised() -> bool:
    """Whether JAX's backend has started. Unknown (the private API moved) counts as not started."""
    try:
        from jax._src import xla_bridge

        return bool(xla_bridge.backends_are_initialized())
    except Exception:
        return False


def _flag_count(flags: str) -> int | None:
    match = re.search(rf"{_FLAG}=(\d+)", flags)
    return int(match.group(1)) if match else None


def _apply() -> HostDeviceChoice:
    raw = os.environ.get(_ENV, "").strip()
    if raw.lower() in _DISABLED_VALUES:
        return HostDeviceChoice(None, "disabled")

    flags = os.environ.get("XLA_FLAGS", "")
    if _FLAG in flags:
        return HostDeviceChoice(_flag_count(flags), "XLA_FLAGS")

    if raw:
        try:
            count = int(raw)
        except ValueError:
            count = 0
        if count < 1:
            warnings.warn(
                f"{_ENV}={raw!r} is not a positive integer or one of 0, off, false, no; "
                "summer4 leaves the number of host devices unchanged.",
                stacklevel=3,
            )
            return HostDeviceChoice(None, "disabled")
        source = _ENV
    else:
        count = os.cpu_count() or 1
        source = "default"

    if _backend_initialised():
        warnings.warn(
            "summer4 cannot give JAX several CPU devices: JAX's backend started before summer4 "
            "was imported, so JAX has already chosen its devices. Import summer4 before using "
            f"JAX, or set XLA_FLAGS={_FLAG}=N (or {_ENV}=N) before starting Python.",
            stacklevel=3,
        )
        return HostDeviceChoice(None, "too-late")

    os.environ["XLA_FLAGS"] = f"{flags} {_FLAG}={count}".strip()
    return HostDeviceChoice(count, source)


CHOICE: HostDeviceChoice = _apply()
