"""Turn ``run(solver=...)`` sugar into a :class:`~summer4.solvers.SolverBackend`."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from summer4.solvers.base import SolverBackend

if TYPE_CHECKING:
    from summer4.solvers.diffrax_backend import Diffrax

KNOWN_SOLVER_NAMES: tuple[str, ...] = ("euler", "heun", "tsit5", "dopri5")

SOLVER_ENV_VAR = "SUMMER4_SOLVER"
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9

_UNSET: Any = object()
# What ``set_default_solver`` stored (None: nothing set).
_process_default: Any = None
# What SUMMER4_SOLVER named, read once on first resolution (None: not set).
_env_default: Any = _UNSET


def default_solver(*, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> Diffrax:
    """The solver ``run()`` uses when none is given: adaptive Tsit5 with a PID controller.

    It is built from the pieces you can use yourself, in one line::

        Diffrax(diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol))

    Pass the result, or your own variation of it, as ``run(solver=...)``.
    """
    from summer4.solvers.diffrax_backend import Diffrax, _import_diffrax

    diffrax = _import_diffrax()
    return Diffrax(
        diffrax.Tsit5(),
        stepsize_controller=diffrax.PIDController(rtol=float(rtol), atol=float(atol)),
    )


def set_default_solver(solver: str | SolverBackend | Any | None) -> None:
    """Set the process-wide default solver; ``None`` restores the built-in default.

    ``solver`` is anything ``run(solver=...)`` accepts. It overrides ``SUMMER4_SOLVER``, and
    is itself overridden by an explicit ``solver=`` argument to ``run``.
    """
    global _process_default
    if isinstance(solver, str) and solver.lower() not in KNOWN_SOLVER_NAMES:
        raise ValueError(_unknown_message(solver))
    _process_default = solver


def _unknown_message(name: str) -> str:
    known = ", ".join(repr(n) for n in KNOWN_SOLVER_NAMES)
    return f"Unknown solver {name!r}. Known names: {known}."


def _environment_default() -> str | None:
    """The solver name in ``SUMMER4_SOLVER``, read on first resolution (None if unset)."""
    global _env_default
    if _env_default is _UNSET:
        raw = os.environ.get(SOLVER_ENV_VAR, "").strip().lower()
        if raw and raw not in KNOWN_SOLVER_NAMES:
            raise ValueError(f"{SOLVER_ENV_VAR}={raw!r}. " + _unknown_message(raw))
        _env_default = raw or None
    return _env_default  # type: ignore[no-any-return]


def describe_default_solver() -> str:
    """Say which solver ``run()`` uses by default, and where that came from."""
    if _process_default is not None:
        label = (
            _process_default
            if isinstance(_process_default, str)
            else getattr(_process_default, "name", type(_process_default).__name__)
        )
        return f"{label} (set_default_solver)"
    env = _environment_default()
    if env is not None:
        return f"{env} ({SOLVER_ENV_VAR})"
    return f"tsit5 (rtol={DEFAULT_RTOL:g}, atol={DEFAULT_ATOL:g})"


def resolve_solver(
    solver: str | Any | None = None,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> SolverBackend:
    """Return the backend ``run`` will use for ``solver``.

    - ``None`` → the default: :func:`set_default_solver` if set, else the ``SUMMER4_SOLVER``
      name, else :func:`default_solver`. With the built-in default, ``rtol`` / ``atol``
      override its tolerances (``1e-6`` / ``1e-9``); with the others they follow the rules
      below.
    - ``"euler"`` → :class:`~summer4.solvers.Euler`.
    - ``"heun"`` / ``"tsit5"`` / ``"dopri5"`` or a bare ``diffrax.AbstractSolver``
      → :class:`~summer4.solvers.Diffrax`, with
      ``diffrax.PIDController(rtol=, atol=)`` as its stepsize controller when
      either tolerance is given (defaults ``1e-3`` / ``1e-6`` for the other).
    - A backend object (anything with ``solve``) is returned as it is.

    ``rtol`` / ``atol`` are sugar for that one controller. They are refused with
    ``"euler"`` and with a backend object: set ``stepsize_controller`` on a
    :class:`~summer4.solvers.Diffrax` backend instead.
    """
    from summer4.solvers.euler_backend import Euler

    if solver is None:
        solver = _process_default if _process_default is not None else _environment_default()
        if solver is None:
            return default_solver(
                rtol=DEFAULT_RTOL if rtol is None else float(rtol),
                atol=DEFAULT_ATOL if atol is None else float(atol),
            )
    tolerances = rtol is not None or atol is not None
    if isinstance(solver, str) and solver.lower() == "euler":
        if tolerances:
            raise ValueError("rtol/atol apply to adaptive diffrax solvers, not euler.")
        return Euler()
    if isinstance(solver, SolverBackend):
        if tolerances:
            raise ValueError(
                f"rtol/atol are sugar for a named or bare diffrax solver; with the "
                f"{type(solver).__name__} backend, set its stepsize_controller instead."
            )
        return solver

    from summer4.solvers.diffrax_backend import Diffrax, _import_diffrax

    if isinstance(solver, str) and solver.lower() not in KNOWN_SOLVER_NAMES:
        raise ValueError(_unknown_message(solver))
    controller = None
    if tolerances:
        diffrax = _import_diffrax()
        controller = diffrax.PIDController(
            rtol=1e-3 if rtol is None else float(rtol),
            atol=1e-6 if atol is None else float(atol),
        )
    return Diffrax(solver, stepsize_controller=controller)


__all__ = [
    "KNOWN_SOLVER_NAMES",
    "default_solver",
    "describe_default_solver",
    "resolve_solver",
    "set_default_solver",
]
