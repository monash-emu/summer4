"""Solver backends behind :meth:`~summer4.flows.compiled.CompiledModel.run`.

``run(solver=...)`` takes a backend: :class:`Diffrax` (the caller's own diffrax
solver, stepsize controller, adjoint, event and progress meter), :class:`Euler`
(summer4's fixed-step stepper), or anything implementing :class:`SolverBackend`.
Names such as ``"dopri5"`` and ``rtol`` / ``atol`` are sugar resolved by
:func:`resolve_solver`.
"""

from summer4.solvers.base import SolveOutput, SolverBackend, SolveSpec
from summer4.solvers.diffrax_backend import Diffrax
from summer4.solvers.euler_backend import Euler
from summer4.solvers.resolve import KNOWN_SOLVER_NAMES, resolve_solver

__all__ = [
    "Diffrax",
    "Euler",
    "KNOWN_SOLVER_NAMES",
    "SolveOutput",
    "SolveSpec",
    "SolverBackend",
    "resolve_solver",
]
