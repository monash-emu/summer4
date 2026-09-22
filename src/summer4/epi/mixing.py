"""Mixing matrices that weight stratified coupling (not people movement)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from summer4.enums import coerce_strenum
from summer4.flows.rates import ArrayConst, FieldRef, RateOps, as_rate
from summer4.properties import Property


class MixingNormalize(StrEnum):
    """Row treatment for a :class:`MixingMatrix`.

    Prefer these members at call sites. Bare strings such as ``"rows"`` are
    still accepted and coerced.
    """

    ROWS = "rows"
    NONE = "none"


type MixingNormalizeArg = MixingNormalize | str


@dataclass(frozen=True, slots=True)
class MixingMatrix:
    """Square contact matrix aligned to a grouping property's traits.

    ``matrix`` may be a :class:`FieldRef` (stays a params key — no recompile to
    swap homogeneous for assortative) or a plain array (treated as a static
    :class:`ArrayConst`).

    ``check_reciprocal`` verifies $K_{ab} N_a = K_{ba} N_b$ at **evaluation**
    time (it needs the population), not at construction. Defaults to ``False``
    so a matrix carried in parameters stays vmap-safe; call
    :meth:`validate` on the host when you want an eager check.
    """

    prop: Property
    matrix: RateOps
    normalize: MixingNormalize = MixingNormalize.ROWS
    check_reciprocal: bool = False
    _static: NDArray[np.float64] | None = None

    def __init__(
        self,
        prop: Property,
        matrix: object,
        *,
        normalize: MixingNormalizeArg = MixingNormalize.ROWS,
        check_reciprocal: bool = False,
    ) -> None:
        resolved = coerce_strenum(MixingNormalize, normalize, what="MixingMatrix normalize")
        n = len(prop.traits)
        static: NDArray[np.float64] | None = None
        if isinstance(matrix, (np.ndarray, list, tuple)):
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != n:
                raise ValueError(
                    f"MixingMatrix for {prop.name!r} expects shape ({n}, {n}), " f"got {arr.shape}."
                )
            if resolved is MixingNormalize.ROWS:
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
                raise TypeError(
                    f"MixingMatrix matrix must be an array, FieldRef, or RateOps; "
                    f"got {type(matrix).__name__}."
                )
        object.__setattr__(self, "prop", prop)
        object.__setattr__(self, "matrix", rate)
        object.__setattr__(self, "normalize", resolved)
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
        if self.normalize is MixingNormalize.ROWS:
            row_sums = jnp.sum(mat, axis=1, keepdims=True)
            mat = mat / row_sums
        return mat

    def check_reciprocity(self, matrix: Any, population: Any) -> None:
        """Raise if $K_{ab} N_a \\neq K_{ba} N_b$ (run-time; needs population).

        Under ``vmap`` the residual may be batched; the callback reduces with
        ``np.max`` so ``float()`` is never called on a multi-element array.
        Prefer :meth:`validate` for an eager host-side check.
        """
        import jax.numpy as jnp

        if not self.check_reciprocal:
            return
        k = jnp.asarray(matrix)
        n = jnp.asarray(population)
        left = k * n[:, None]
        diff = jnp.max(jnp.abs(left - left.T))
        name = self.prop.name

        def _raise(d: Any) -> None:
            max_diff = float(np.max(np.asarray(d)))
            if max_diff > 1e-6:
                raise ValueError(
                    f"MixingMatrix for {name!r} is not reciprocal: "
                    f"max |K_ab N_a - K_ba N_b| = {max_diff}."
                )

        import jax

        jax.debug.callback(_raise, diff)

    @staticmethod
    def validate(
        matrix: object,
        population: object,
        *,
        name: str = "mixing",
        tol: float = 1e-6,
    ) -> None:
        """Host-side reciprocity check (no JAX tracers / callbacks)."""
        k = np.asarray(matrix, dtype=np.float64)
        n = np.asarray(population, dtype=np.float64)
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError(f"MixingMatrix.validate expects a square matrix, got {k.shape}.")
        if n.shape != (k.shape[0],):
            raise ValueError(
                f"MixingMatrix.validate population shape {n.shape} does not match "
                f"matrix side {k.shape[0]}."
            )
        left = k * n[:, None]
        max_diff = float(np.max(np.abs(left - left.T)))
        if max_diff > tol:
            raise ValueError(
                f"MixingMatrix for {name!r} is not reciprocal: "
                f"max |K_ab N_a - K_ba N_b| = {max_diff}."
            )
