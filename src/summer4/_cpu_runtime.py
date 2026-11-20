"""XLA CPU runtime: summer4 selects XLA's legacy CPU runtime where it exists, unless the user chose.

jaxlib 0.4.32 made XLA:CPU's *thunk* runtime the default. It runs each loop body as a sequence of
thunks, which is much slower for small operations inside ``lax.scan`` / ``while_loop``: exactly
how summer4's solvers step a model (jax-ml/jax#26145, #29772, #40150). The flag
``--xla_cpu_use_thunk_runtime=false`` restores the previous runtime. It is honoured from jaxlib
0.4.32 up to 0.6.2; jaxlib 0.7.0 and 0.7.1 fail at run time with it ("CpuExecutable has no
thunks"), and later jaxlibs ignore it. summer4 sets it when imported:

- jaxlib 0.4.32 <= version < 0.7.0, nothing set: the legacy runtime;
- ``SUMMER4_CPU_RUNTIME=thunk`` (or ``0``, ``off``, ``false``, ``no``): nothing is set;
- ``XLA_FLAGS`` already names the flag: nothing is set, because the user chose;
- any other jaxlib (or jaxlib not installed): the legacy flag is not set, since it has no use there.

From jaxlib 0.7 only the thunk runtime is left. Its *concurrency-optimised scheduler* orders a
loop body's operations to run them in parallel, which costs more than it gains for the small
operations of a solver step: on summer4's weighted-``Reduce`` benchmark (jaxlib 0.11.1) runs
were about twice as fast with ``--xla_cpu_enable_concurrency_optimized_scheduler=false``. summer4
sets that flag for jaxlib 0.7.0 <= version < 0.13.0, the versions checked to accept it (XLA
aborts the process on a flag it does not know, so newer versions are left alone until checked).
``SUMMER4_CPU_RUNTIME=thunk`` and an ``XLA_FLAGS`` value naming the flag leave it unset too.

The version is read from the installed package metadata, so importing summer4 still does not
start JAX's backend. JAX reads the flag when its backend starts; if JAX was already used before
summer4 was imported, summer4 sets nothing and warns. The flag affects XLA:CPU only. The outcome
is recorded in ``CHOICE`` and reported by ``summer4.runtime_info()``.
"""

import os
import re
import warnings
from dataclasses import dataclass
from importlib import metadata

from summer4._devices import _backend_initialised

_FLAG = "--xla_cpu_use_thunk_runtime"
_ENV = "SUMMER4_CPU_RUNTIME"
_THUNK_VALUES = frozenset({"thunk", "0", "off", "false", "no"})
_LEGACY_VALUES = frozenset({"legacy", "", "1", "on", "true", "yes", "auto"})
#: jaxlib versions whose XLA:CPU still has the legacy runtime and honours the flag.
LEGACY_FROM = (0, 4, 32)
LEGACY_BEFORE = (0, 7, 0)
_SCHEDULER_FLAG = "--xla_cpu_enable_concurrency_optimized_scheduler"
#: jaxlib versions checked to accept the scheduler flag (unknown flags abort XLA).
SCHEDULER_FROM = (0, 7, 0)
SCHEDULER_BEFORE = (0, 13, 0)


@dataclass(frozen=True)
class CpuRuntimeChoice:
    """What summer4 did about XLA's CPU runtime when it was imported.

    ``runtime`` is ``"legacy"``, ``"thunk"`` or ``None`` (summer4 cannot tell, e.g. the user's
    ``XLA_FLAGS`` value is unreadable). ``source`` is one of ``"default"``,
    ``"SUMMER4_CPU_RUNTIME"``, ``"XLA_FLAGS"``, ``"jaxlib"`` (this jaxlib has one runtime, so
    no runtime was chosen) or ``"too-late"`` (JAX's backend had already started).
    ``concurrency_scheduler`` is ``False`` when the thunk runtime's concurrency-optimised
    scheduler is off (summer4 or the user's ``XLA_FLAGS`` turned it off), ``True`` when XLA's
    default is in force, and ``None`` when it does not apply (legacy runtime) or is unknown.
    """

    runtime: str | None
    source: str
    jaxlib: str | None
    concurrency_scheduler: bool | None = None


def _jaxlib_version() -> str | None:
    try:
        return metadata.version("jaxlib")
    except metadata.PackageNotFoundError:
        return None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split(".")[:3]:
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def has_legacy_runtime(version: str | None) -> bool:
    """Whether jaxlib ``version`` has XLA:CPU's legacy runtime behind the flag."""
    if version is None:
        return False
    return LEGACY_FROM <= _version_tuple(version) < LEGACY_BEFORE


def has_scheduler_flag(version: str | None) -> bool:
    """Whether summer4 turns off the thunk runtime's concurrency scheduler on jaxlib ``version``."""
    if version is None:
        return False
    return SCHEDULER_FROM <= _version_tuple(version) < SCHEDULER_BEFORE


def _scheduler_from_flags(flags: str) -> bool | None:
    match = re.search(rf"{_SCHEDULER_FLAG}(?:=(\S+))?", flags)
    if match is None:
        return None
    value = (match.group(1) or "true").lower()
    return {"true": True, "1": True, "false": False, "0": False}.get(value)


def _runtime_from_flags(flags: str) -> str | None:
    match = re.search(rf"{_FLAG}(?:=(\S+))?", flags)
    if match is None:
        return None
    value = (match.group(1) or "true").lower()
    if value in ("false", "0"):
        return "legacy"
    if value in ("true", "1"):
        return "thunk"
    return None


def _apply() -> CpuRuntimeChoice:
    jaxlib = _jaxlib_version()
    flags = os.environ.get("XLA_FLAGS", "")
    if _FLAG in flags:
        return CpuRuntimeChoice(
            _runtime_from_flags(flags), "XLA_FLAGS", jaxlib, _scheduler_from_flags(flags)
        )

    raw = os.environ.get(_ENV, "").strip().lower()
    available = has_legacy_runtime(jaxlib)
    # Without the flag: jaxlib before 0.4.32 has only the legacy runtime, later ones only thunks.
    older = jaxlib is not None and _version_tuple(jaxlib) < LEGACY_FROM
    default_runtime = "legacy" if older else ("thunk" if jaxlib is not None else None)
    if raw in _THUNK_VALUES:
        # Keep XLA's own defaults: the thunk runtime and its default scheduler.
        scheduler = _scheduler_from_flags(flags)
        if scheduler is None and (available or has_scheduler_flag(jaxlib)):
            scheduler = True
        return CpuRuntimeChoice("thunk" if available else default_runtime, _ENV, jaxlib, scheduler)
    if raw not in _LEGACY_VALUES:
        warnings.warn(
            f"{_ENV}={raw!r} is not one of legacy, thunk (or 0, off, false, no); "
            "summer4 uses its default CPU runtime.",
            stacklevel=3,
        )
    if not available:
        if raw == "legacy":
            warnings.warn(
                f"{_ENV}=legacy has no effect: jaxlib {jaxlib} has no legacy XLA:CPU runtime "
                "(it exists from 0.4.32 up to 0.6.2).",
                stacklevel=3,
            )
        return _choose_scheduler(default_runtime, jaxlib, flags)

    source = _ENV if raw else "default"
    if _backend_initialised():
        warnings.warn(
            "summer4 cannot select XLA's legacy CPU runtime: JAX's backend started before "
            "summer4 was imported. Import summer4 before using JAX, or set "
            f"XLA_FLAGS={_FLAG}=false before starting Python.",
            stacklevel=3,
        )
        return CpuRuntimeChoice("thunk", "too-late", jaxlib)

    os.environ["XLA_FLAGS"] = f"{flags} {_FLAG}=false".strip()
    return CpuRuntimeChoice("legacy", source, jaxlib)


def _choose_scheduler(runtime: str | None, jaxlib: str | None, flags: str) -> CpuRuntimeChoice:
    """On a thunk-only jaxlib, turn off the concurrency-optimised scheduler where that is safe."""
    if _SCHEDULER_FLAG in flags:
        return CpuRuntimeChoice(runtime, "jaxlib", jaxlib, _scheduler_from_flags(flags))
    if runtime != "thunk" or not has_scheduler_flag(jaxlib):
        return CpuRuntimeChoice(runtime, "jaxlib", jaxlib)
    if _backend_initialised():
        warnings.warn(
            "summer4 cannot turn off XLA's concurrency-optimised CPU scheduler: JAX's backend "
            "started before summer4 was imported. Import summer4 before using JAX, or set "
            f"XLA_FLAGS={_SCHEDULER_FLAG}=false before starting Python.",
            stacklevel=4,
        )
        return CpuRuntimeChoice(runtime, "too-late", jaxlib, True)
    os.environ["XLA_FLAGS"] = f"{flags} {_SCHEDULER_FLAG}=false".strip()
    return CpuRuntimeChoice(runtime, "jaxlib", jaxlib, False)


CHOICE: CpuRuntimeChoice = _apply()
