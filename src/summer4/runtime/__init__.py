"""Runtime facts for the current process: precision, host devices and default solver."""

from dataclasses import dataclass
from typing import Any, cast

from summer4 import _devices


@dataclass(frozen=True)
class RuntimeInfo:
    """What this process will compute with, as chosen at import and by JAX's configuration.

    ``x64`` is JAX's 64-bit mode. ``host_devices_requested`` and ``host_devices_source`` say
    what summer4 did about CPU devices at import (see ``summer4._devices``). ``default_solver``
    names the solver ``run()`` uses when none is given.
    """

    x64: bool
    host_devices_requested: int | None
    host_devices_source: str
    default_solver: str

    def devices(self) -> list[Any]:
        """The devices JAX can use. Calling this initialises JAX's backend."""
        import jax

        return list(jax.devices())


def runtime_info() -> RuntimeInfo:
    """Describe the current process's runtime. Does not initialise JAX's backend."""
    import jax

    from summer4.solvers.resolve import describe_default_solver

    return RuntimeInfo(
        x64=bool(cast(Any, jax.config).jax_enable_x64),
        host_devices_requested=_devices.CHOICE.requested,
        host_devices_source=_devices.CHOICE.source,
        default_solver=describe_default_solver(),
    )
