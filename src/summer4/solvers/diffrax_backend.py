"""The diffrax backend: the caller's diffrax objects, one ``SubSaveAt`` per save group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from summer4.results.groups import group_requests
from summer4.results.result import SolverInfo
from summer4.solvers.base import SolveOutput, SolveSpec, default_max_steps
from summer4.solvers.resolve import KNOWN_SOLVER_NAMES

_NAMED_SOLVERS: dict[str, str] = {
    "heun": "Heun",
    "tsit5": "Tsit5",
    "dopri5": "Dopri5",
}


# Equinox Module classes are created once so Diffrax's filter_jit keys by
# structure (CompiledModel digest + request tree), not by Python id.
_DiffraxVectorField: Any | None = None
_DiffraxSaveFn: Any | None = None


def _import_diffrax() -> Any:
    try:
        import diffrax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The diffrax solver backend requires the JAX extra. "
            "Install with: pip install summer4[jax]"
        ) from exc
    return diffrax


def resolve_diffrax_solver(solver: str | Any) -> tuple[Any, str]:
    """Return ``(diffrax solver instance, display name)`` for a name or an instance."""
    diffrax = _import_diffrax()
    if isinstance(solver, str):
        key = solver.lower()
        if key == "euler":
            raise ValueError("Use the Euler backend for solver='euler'.")
        if key not in _NAMED_SOLVERS:
            known = ", ".join(repr(n) for n in KNOWN_SOLVER_NAMES)
            raise ValueError(f"Unknown solver {solver!r}. Known names: {known}.")
        cls = getattr(diffrax, _NAMED_SOLVERS[key])
        return cls(), key
    # Expert path: a diffrax solver instance.
    if not isinstance(solver, diffrax.AbstractSolver):
        known = ", ".join(repr(n) for n in KNOWN_SOLVER_NAMES)
        raise ValueError(
            f"solver must be a name ({known}) or a diffrax solver instance, "
            f"got {type(solver).__name__}."
        )
    name = type(solver).__name__.lower()
    return solver, name


def _filtered_plan(plan: Any, keys: tuple[str, ...]) -> Any:
    from summer4.results.plan import SavePlan

    return SavePlan(
        requests={k: plan.requests[k] for k in keys},
        ts=plan.ts,
        dense=plan.dense,
        solver_stats=plan.solver_stats,
    )


def _dy_array(dy: Any) -> Any:
    from summer4.jax.propertydata import PropertyData
    from summer4.jax.state import State

    if isinstance(dy, State):
        return dy.compartments.data
    if isinstance(dy, PropertyData):
        return dy.data
    return dy


def _ensure_eqx_modules() -> tuple[Any, Any]:
    """Build the Diffrax VF / save Module classes once per process."""
    global _DiffraxVectorField, _DiffraxSaveFn
    if _DiffraxVectorField is not None and _DiffraxSaveFn is not None:
        return _DiffraxVectorField, _DiffraxSaveFn

    import equinox as eqx
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData
    from summer4.results.eval import eval_quantity

    class DiffraxVectorField(eqx.Module):
        """Structurally equal across ``run()`` when ``model`` digests match."""

        model: Any

        def __call__(self, t: Any, y: Any, args: Any) -> Any:
            return _dy_array(self.model.vector_field(t, PropertyData(self.model.pmap, y), args))

    class DiffraxSaveFn(eqx.Module):
        """Save callback; prepared params come from Diffrax ``args``, not a closure."""

        model: Any
        requests: tuple[tuple[str, Any], ...]
        keep: frozenset[str]

        def __call__(self, t: Any, y: Any, args: Any) -> dict[str, Any]:
            ctx = self.model.observe(
                t,
                PropertyData(self.model.pmap, y),
                args,
                keep=self.keep,
            )
            out: dict[str, Any] = {}
            for key, what in self.requests:
                value = eval_quantity(
                    what,
                    ctx,
                    pmap=self.model.pmap,
                    edge_maps=dict(self.model.edge_maps),
                )
                out[key] = value.data if isinstance(value, PropertyData) else jnp.asarray(value)
            return out

    _DiffraxVectorField = DiffraxVectorField
    _DiffraxSaveFn = DiffraxSaveFn
    return DiffraxVectorField, DiffraxSaveFn


def _result_code(result: Any) -> Any:
    """Extract a JIT-safe integer code from a diffrax RESULTS enum item."""
    return result._value


# Save key for the end-of-integration state; cannot collide with a group name.
_FINAL = "__summer4_final__"


@dataclass(frozen=True)
class Diffrax:
    """Integrate with diffrax, using the caller's own diffrax objects.

    ``solver`` is any ``diffrax.AbstractSolver`` (or a name: ``"heun"``,
    ``"tsit5"``, ``"dopri5"``). Every other field is passed to
    ``diffrax.diffeqsolve`` as given; ``None`` keeps summer4's default —
    ``ConstantStepSize()`` for ``stepsize_controller`` and diffrax's own
    defaults for ``adjoint``, ``event`` and ``progress_meter``::

        Diffrax(
            diffrax.Tsit5(),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            adjoint=diffrax.DirectAdjoint(),
            event=diffrax.Event(cond_fn),
        )

    ``BacksolveAdjoint`` is not supported: summer4 saves through
    ``SaveAt(subs=...)``, which diffrax refuses with that adjoint.

    ``SolveSpec.dt`` is the step for a constant controller and the first step
    ``dt0`` for an adaptive one. When an ``event`` stops the solve, saves after
    that time are ``inf`` (diffrax's convention), :attr:`SolverInfo.event` is
    true, and ``Result.final_time`` / ``final_state`` are where it stopped.
    """

    solver: Any
    stepsize_controller: Any = None
    adjoint: Any = None
    event: Any = None
    progress_meter: Any = None

    def __post_init__(self) -> None:
        instance, _name = resolve_diffrax_solver(self.solver)
        object.__setattr__(self, "solver", instance)

    @property
    def name(self) -> str:
        """Display name for :attr:`SolverInfo.solver` (the solver class, lower case)."""
        return type(self.solver).__name__.lower()

    def solve(
        self,
        model: Any,
        *,
        y0: object,
        params: object,
        spec: SolveSpec,
        plan: Any,
    ) -> SolveOutput:
        """Integrate with ``diffrax.diffeqsolve``; see :class:`SolverBackend`."""
        import jax.numpy as jnp

        from summer4.flows.stages import Prepared
        from summer4.jax.state import unpack_state

        diffrax = _import_diffrax()
        vf_cls, save_cls = _ensure_eqx_modules()

        prepared = params if isinstance(params, Prepared) else model.prepare(params)
        y_arr, _rebox = unpack_state(y0, model.pmap)
        groups = group_requests(plan, spec.default_ts())

        term = diffrax.ODETerm(vf_cls(model))
        subs: dict[str, Any] = {}
        for group in groups:
            group_plan = _filtered_plan(plan, group.keys)
            requests = tuple((key, group_plan.requests[key].what) for key in group.keys)
            keep = group_plan.flow_reads()
            subs[group.name] = diffrax.SubSaveAt(
                ts=jnp.asarray(group.ts),
                fn=save_cls(model, requests, keep),
            )
        subs[_FINAL] = diffrax.SubSaveAt(t1=True)
        saveat = diffrax.SaveAt(subs=subs, dense=bool(spec.dense))

        controller = (
            diffrax.ConstantStepSize()
            if self.stepsize_controller is None
            else self.stepsize_controller
        )

        t0 = float(spec.t0)
        t1 = spec.end
        # Include every requested save time in the integration window.
        for group in groups:
            if group.ts.size:
                t1 = max(t1, float(group.ts[-1]))

        max_steps = (
            default_max_steps(t0, t1, float(spec.dt))
            if spec.max_steps is None
            else int(spec.max_steps)
        )
        # None → False so failure surfaces as SolverInfo.ok / result_code.
        throw = False if spec.throw is None else bool(spec.throw)

        extra: dict[str, Any] = {}
        if self.adjoint is not None:
            extra["adjoint"] = self.adjoint
        if self.event is not None:
            extra["event"] = self.event
        if self.progress_meter is not None:
            extra["progress_meter"] = self.progress_meter

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=float(spec.dt),
            y0=y_arr,
            args=prepared,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=max_steps,
            throw=throw,
            **extra,
        )

        saved: dict[str, Any] = {}
        for group in groups:
            group_ys = sol.ys[group.name]
            for key in group.keys:
                saved[key] = group_ys[key]

        stats: SolverInfo | None = None
        if plan.solver_stats:
            st = sol.stats
            code = _result_code(sol.result)
            stats = SolverInfo(
                solver=self.name,
                num_steps=st.get("num_steps"),
                num_accepted_steps=st.get("num_accepted_steps"),
                num_rejected_steps=st.get("num_rejected_steps"),
                result_code=code,
                max_steps=st.get("max_steps", max_steps),
                dense=bool(spec.dense),
                event_occurred=(
                    None
                    if self.event is None
                    else code == _result_code(diffrax.RESULTS.event_occurred)
                ),
            )

        dense = sol.interpolation if spec.dense else None
        return SolveOutput(
            saved=saved,
            stats=stats,
            dense=dense,
            final_time=sol.ts[_FINAL][-1],
            final_state=sol.ys[_FINAL][-1],
        )


__all__ = [
    "Diffrax",
    "KNOWN_SOLVER_NAMES",
    "resolve_diffrax_solver",
]
