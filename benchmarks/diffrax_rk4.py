"""Classical four-stage RK4 for Diffrax fixed-step runs.

Diffrax does not ship classical RK4. This tableau matches summer2's
``solvers.rk4``: weights ``1/6, 1/3, 1/3, 1/6`` and nodes ``0, 1/2, 1/2, 1``.
Dummy ``b_error`` is only valid with ``ConstantStepSize`` — do not pass
``rtol`` or ``atol`` with this solver.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from diffrax import AbstractERK, ButcherTableau, ThirdOrderHermitePolynomialInterpolation

_CLASSICAL_RK4_TABLEAU = ButcherTableau(
    a_lower=(
        np.array([0.5]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.0, 1.0]),
    ),
    b_sol=np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
    b_error=np.array([-5.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
    c=np.array([0.5, 0.5, 1.0]),
)


class ClassicalRK4(AbstractERK):
    """Classical RK4. Four rate evaluations per step. Fixed step only."""

    tableau: ClassVar[ButcherTableau] = _CLASSICAL_RK4_TABLEAU
    interpolation_cls: ClassVar[Callable[..., ThirdOrderHermitePolynomialInterpolation]] = (
        ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms: object) -> int:
        del terms
        return 4
