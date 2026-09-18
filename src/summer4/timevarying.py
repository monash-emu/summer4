"""JAX-traceable time-varying rate constructors.

These wrap structural :class:`~summer4.flows.rates.Interp` /
:class:`~summer4.flows.rates.GaussianPulse` nodes rather than closures, so
two equal expressions share a :class:`~summer4.flows.CompiledModel` jit
cache entry.
"""

from __future__ import annotations

from collections.abc import Sequence

from summer4.flows.rates import Const, GaussianPulse, Interp, RateOps, as_rate

__all__ = [
    "GaussianPulse",
    "Interp",
    "gaussian_pulse",
    "linear",
    "piecewise",
    "sigmoidal",
    "step",
]


def _as_breakpoints(breakpoints: Sequence[object]) -> tuple[RateOps, ...]:
    """Coerce breakpoints to rate nodes; enforce order when all are Const."""
    if not breakpoints:
        raise ValueError("breakpoints must be non-empty.")
    pts = tuple(as_rate(x) for x in breakpoints)
    consts = [p for p in pts if isinstance(p, Const)]
    if len(consts) == len(pts):
        vals = tuple(float(p.value) for p in consts)
        if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            raise ValueError("breakpoints must be strictly increasing.")
    return pts


def _as_values(values: Sequence[object]) -> tuple[RateOps, ...]:
    if not values:
        raise ValueError("values must be non-empty.")
    return tuple(as_rate(v) for v in values)


def linear(
    arg: object,
    breakpoints: Sequence[object],
    values: Sequence[object],
) -> Interp:
    """Linear interpolation of ``values`` at ``breakpoints``, evaluated at ``arg``.

    ``breakpoints`` and ``values`` may be floats or rate expressions
    (:class:`~summer4.flows.rates.FieldRef`, …); the count is fixed at
    construct time. Outside the knot range the result clamps to the nearest
    end value (``jax.numpy.interp`` behaviour) — it does not extrapolate.
    Evaluated breakpoints must stay strictly increasing (not sorted at
    runtime). ``len(values)`` must equal ``len(breakpoints)``.
    """
    bps = _as_breakpoints(breakpoints)
    vals = _as_values(values)
    if len(vals) != len(bps):
        raise ValueError(
            f"linear requires len(values) == len(breakpoints); got {len(vals)} vs {len(bps)}."
        )
    return Interp(kind="linear", breakpoints=bps, values=vals, arg=as_rate(arg))


def sigmoidal(
    arg: object,
    breakpoints: Sequence[object],
    values: Sequence[object],
    *,
    sharpness: float = 1.0,
) -> Interp:
    """Piecewise sigmoidal blend between neighbouring knots.

    ``sharpness`` is summer2's curvature parameter: ``1.0`` is
    linear-equivalent after end-point normalization; larger values shrink the
    transition width toward a step at each segment midpoint. Ends clamp to
    the first/last knot value. Breakpoints may be rate expressions; evaluated
    positions must stay strictly increasing. ``len(values)`` must equal
    ``len(breakpoints)``.
    """
    bps = _as_breakpoints(breakpoints)
    vals = _as_values(values)
    if len(vals) != len(bps):
        raise ValueError(
            f"sigmoidal requires len(values) == len(breakpoints); "
            f"got {len(vals)} vs {len(bps)}."
        )
    if len(bps) < 2:
        raise ValueError("sigmoidal requires at least two breakpoints.")
    return Interp(
        kind="sigmoidal",
        breakpoints=bps,
        values=vals,
        arg=as_rate(arg),
        sharpness=float(sharpness),
    )


def step(
    arg: object,
    breakpoints: Sequence[object],
    values: Sequence[object],
) -> Interp:
    """Right-continuous piecewise-constant function.

    ``len(values)`` must be ``len(breakpoints) + 1``. For breakpoints
    ``(b0, …, bn-1)`` the value is ``values[0]`` for ``arg < b0``,
    ``values[i+1]`` for ``bi <= arg < b{i+1}``, and ``values[n]`` for
    ``arg >= b{n-1}``. At a breakpoint the new (right) value is taken.
    Breakpoints may be rate expressions; evaluated positions must stay
    strictly increasing.
    """
    bps = _as_breakpoints(breakpoints)
    vals = _as_values(values)
    if len(vals) != len(bps) + 1:
        raise ValueError(
            f"step requires len(values) == len(breakpoints) + 1; "
            f"got {len(vals)} vs {len(bps)} + 1."
        )
    return Interp(kind="step", breakpoints=bps, values=vals, arg=as_rate(arg))


def piecewise(
    arg: object,
    breakpoints: Sequence[object],
    values: Sequence[object],
) -> Interp:
    """Alias of :func:`step` — summer2's ``get_piecewise_function`` name."""
    return step(arg, breakpoints, values)


def gaussian_pulse(
    arg: object,
    centre: object,
    width: object,
    height: object,
) -> GaussianPulse:
    """``height * exp(-0.5 * ((arg - centre) / width)**2)``.

    ``centre``, ``width`` and ``height`` may be rate expressions so they
    calibrate.
    """
    return GaussianPulse(
        arg=as_rate(arg),
        centre=as_rate(centre),
        width=as_rate(width),
        height=as_rate(height),
    )
