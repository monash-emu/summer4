"""Mixing matrices that weight stratified coupling (not people movement)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from summer4.flows.rates import ArrayConst, FieldRef, RateOps, as_rate
from summer4.properties import Property


@dataclass(frozen=True, slots=True)
class MixingMatrix:
    """Square contact matrix aligned to a grouping property's traits.

    ``matrix`` may be a :class:`FieldRef` (stays a params key — no recompile to
    swap homogeneous for assortative) or a plain array (treated as a static
    :class:`ArrayConst`).

    ``check_reciprocal`` verifies $K_{ab} N_a = K_{ba} N_b$ at **evaluation**
    time (it needs the population), not at construction. A user who expects a
    constructor error will otherwise think it passed.
    """

    prop: Property
    matrix: RateOps
    normalize: Literal["rows", "none"] = "rows"
    check_reciprocal: bool = True
    _static: NDArray[np.float64] | None = None

    def __init__(
        self,
        prop: Property,
        matrix: object,
        *,
        normalize: Literal["rows", "none"] = "rows",
        check_reciprocal: bool = True,
    ) -> None:
        n = len(prop.traits)
        static: NDArray[np.float64] | None = None
        if isinstance(matrix, (np.ndarray, list, tuple)):
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != n:
                raise ValueError(
                    f"MixingMatrix for {prop.name!r} expects shape ({n}, {n}), " f"got {arr.shape}."
                )
            if normalize == "rows":
                row_sums = arr.sum(axis=1, keepdims=True)
                if np.any(row_sums == 0):
                    raise ValueError(
                        f"MixingMatrix for {prop.name!r}: cannot normalize rows " "with a zero sum."
                    )
                arr = arr / row_sums
            static = np.ascontiguousarray(arr)
            rate: RateOps = ArrayConst(static)
        elif isinstance(matrix, RateOps):
            rate = matrix
        else:
            rate = as_rate(matrix)
            if not isinstance(rate, FieldRef):
                # as_rate of scalar is Const — not a matrix.
                raise TypeError(
                    f"MixingMatrix matrix must be an array, FieldRef, or RateOps; "
                    f"got {type(matrix).__name__}."
                )
        object.__setattr__(self, "prop", prop)
        object.__setattr__(self, "matrix", rate)
        object.__setattr__(self, "normalize", normalize)
        object.__setattr__(self, "check_reciprocal", check_reciprocal)
        object.__setattr__(self, "_static", static)

    def resolved_matrix(self, derived: object, eval_child: Any) -> Any:
        """Evaluate the matrix against ``derived``, applying row-normalize if needed."""
        import jax.numpy as jnp

        if self._static is not None:
            return jnp.asarray(self._static)
        raw = eval_child(self.matrix) if isinstance(self.matrix, RateOps) else self.matrix
        mat = jnp.asarray(raw)
        n = len(self.prop.traits)
        if mat.shape != (n, n):
            raise ValueError(
                f"MixingMatrix for {self.prop.name!r} expects shape ({n}, {n}), "
                f"got {tuple(mat.shape)}."
            )
        if self.normalize == "rows":
            row_sums = jnp.sum(mat, axis=1, keepdims=True)
            mat = mat / row_sums
        return mat

    def check_reciprocity(self, matrix: Any, population: Any) -> None:
        """Raise if $K_{ab} N_a \\neq K_{ba} N_b$ (run-time; needs population)."""
        import jax.numpy as jnp

        if not self.check_reciprocal:
            return
        k = jnp.asarray(matrix)
        n = jnp.asarray(population)
        left = k * n[:, None]
        diff = jnp.max(jnp.abs(left - left.T))

        def _raise(d: Any) -> None:
            if float(d) > 1e-6:
                raise ValueError(
                    f"MixingMatrix for {self.prop.name!r} is not reciprocal: "
                    f"max |K_ab N_a - K_ba N_b| = {float(d)}."
                )

        # Runs at concrete execution (including under jit via callback).
        import jax

        jax.debug.callback(_raise, diff)
