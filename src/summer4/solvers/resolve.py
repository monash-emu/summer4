"""Turn ``run(solver=...)`` sugar into a :class:`~summer4.solvers.SolverBackend`."""

from __future__ import annotations

from typing import Any

from summer4.solvers.base import SolverBackend

KNOWN_SOLVER_NAMES: tuple[str, ...] = ("euler", "heun", "tsit5", "dopri5")


def resolve_solver(
    solver: str | Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> SolverBackend:
    """Return the backend ``run`` will use for ``solver``.

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
        known = ", ".join(repr(n) for n in KNOWN_SOLVER_NAMES)
        raise ValueError(f"Unknown solver {solver!r}. Known names: {known}.")
    controller = None
    if tolerances:
        diffrax = _import_diffrax()
        controller = diffrax.PIDController(
            rtol=1e-3 if rtol is None else float(rtol),
            atol=1e-6 if atol is None else float(atol),
        )
    return Diffrax(solver, stepsize_controller=controller)


__all__ = ["KNOWN_SOLVER_NAMES", "resolve_solver"]
