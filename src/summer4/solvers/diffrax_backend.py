"""diffrax adaptive / fixed-step backend with SubSaveAt groups."""

from __future__ import annotations

from typing import Any

from summer4.results.groups import SaveGroup
from summer4.results.result import SolverInfo
from summer4.solvers.base import SolveOutput, SolveSpec

_NAMED_SOLVERS: dict[str, str] = {
    "heun": "Heun",
    "tsit5": "Tsit5",
    "dopri5": "Dopri5",
}

KNOWN_SOLVER_NAMES: tuple[str, ...] = ("euler", "heun", "tsit5", "dopri5")

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
    """Return ``(diffrax solver instance, display name)``."""
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


def diffrax_solve(
    model: Any,
    *,
    y0: object,
    params: object,
    spec: SolveSpec,
    groups: tuple[SaveGroup, ...],
    plan: Any,
    solver: str | Any,
    solver_stats: bool,
) -> SolveOutput:
    """Integrate with diffrax; one ``SubSaveAt`` per save group."""
    import jax.numpy as jnp

    from summer4.flows.stages import Prepared
    from summer4.jax.state import unpack_state

    diffrax = _import_diffrax()
    solver_inst, solver_name = resolve_diffrax_solver(solver)
    vf_cls, save_cls = _ensure_eqx_modules()

    prepared = params if isinstance(params, Prepared) else model.prepare(params)
    y_arr, _rebox = unpack_state(y0, model.pmap)

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
    saveat = diffrax.SaveAt(subs=subs, dense=bool(spec.dense))

    if spec.rtol is not None or spec.atol is not None:
        controller: Any = diffrax.PIDController(
            rtol=1e-3 if spec.rtol is None else float(spec.rtol),
            atol=1e-6 if spec.atol is None else float(spec.atol),
        )
        dt0: float | None = float(spec.dt)
    else:
        controller = diffrax.ConstantStepSize()
        dt0 = float(spec.dt)

    t0 = float(spec.t0)
    if spec.t1 is not None:
        t1 = float(spec.t1)
    elif spec.steps is not None:
        t1 = t0 + float(spec.dt) * int(spec.steps)
    else:
        raise ValueError("diffrax backend requires t1 or steps.")

    # Include every requested save time in the integration window.
    for group in groups:
        if group.ts.size:
            t1 = max(t1, float(group.ts[-1]))

    max_steps = 4096 if spec.max_steps is None else int(spec.max_steps)
    if spec.dense and spec.max_steps is None:
        # Dense output allocates max_steps worth of coefficients.
        max_steps = max(max_steps, 4096)

    sol = diffrax.diffeqsolve(
        term,
        solver_inst,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y_arr,
        args=prepared,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
        throw=False,
    )

    saved: dict[str, Any] = {}
    for group in groups:
        group_ys = sol.ys[group.name]
        for key in group.keys:
            saved[key] = group_ys[key]

    stats: SolverInfo | None = None
    if solver_stats:
        st = sol.stats
        stats = SolverInfo(
            solver=solver_name,
            num_steps=st.get("num_steps"),
            num_accepted_steps=st.get("num_accepted_steps"),
            num_rejected_steps=st.get("num_rejected_steps"),
            result_code=_result_code(sol.result),
            max_steps=st.get("max_steps", max_steps),
            dense=bool(spec.dense),
        )

    dense = sol.interpolation if spec.dense else None
    return SolveOutput(saved=saved, stats=stats, dense=dense)
