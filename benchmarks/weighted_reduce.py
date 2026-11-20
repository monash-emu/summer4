"""Weighted sums of scalar ``Reduce`` terms: a standalone reproduction and benchmark.

A TB-shaped chain (S, E, L, active, T, R) whose ``active`` stage is split on a
``focus`` x ``extent`` grid. The infectious pool weights each grid cell by
``focus_centre[f] * extent_centre[e]``. Three ways of writing it:

- ``terms``: ``sum(w_fe * Reduce(where=active & focus[f] & extent[e]))``, one
  scalar ``Reduce`` per cell;
- ``foi``: ``ForceOfInfection(infectious=active, infectiousness=[...])``, with
  one weight pair per focus trait and per extent trait (their product is the
  cell weight);
- ``plain``: one unweighted ``Reduce`` (a different model, for scale only).

Usage (time with devq after a warm build)::

    python benchmarks/weighted_reduce.py --form terms --solver euler --what grad
    python benchmarks/weighted_reduce.py --eqns
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from summer4 import FlowModel, Param, Property, PropertyMap, Reduce, SavePlan, TransitionFlow
from summer4.epi import FOIKind, ForceOfInfection

STAGES = ("S", "E", "L", "active", "T", "R")
FORMS = ("terms", "foi", "plain")
PARAMS = {"contact_rate": 8.0, "activation_rate": 0.02}
GRAD_PARAMS = ("contact_rate", "activation_rate")


def grid(n_focus: int = 6, n_extent: int = 6) -> tuple[Property, Property, Property]:
    stage = Property("stage", STAGES)
    focus = Property("focus", tuple(f"f{i}" for i in range(n_focus)))
    extent = Property("extent", tuple(f"e{i}" for i in range(n_extent)))
    return stage, focus, extent


def centres(n: int) -> np.ndarray:
    """Cell-centre weights on ``[0, 1]``: ``(i + 0.5) / n``."""
    return (np.arange(n) + 0.5) / n


def build(
    form: str = "terms",
    n_focus: int = 6,
    n_extent: int = 6,
) -> tuple[FlowModel, PropertyMap]:
    """The model in one of the three :data:`FORMS`."""
    stage, focus, extent = grid(n_focus, n_extent)
    active = stage["active"]
    pmap = PropertyMap.from_property(stage)
    pmap = pmap.stratify(focus, where=active).stratify(extent, where=active)
    yf = centres(n_focus)
    xe = centres(n_extent)

    if form == "terms":
        infectious: Any = sum(
            float(yf[f] * xe[e]) * Reduce(where=active & focus[fn] & extent[en])
            for f, fn in enumerate(focus.traits)
            for e, en in enumerate(extent.traits)
        )
        rate: Any = Param("contact_rate") * infectious / Reduce()
    elif form == "foi":
        weights = [(focus[fn], float(yf[f])) for f, fn in enumerate(focus.traits)]
        weights += [(extent[en], float(xe[e])) for e, en in enumerate(extent.traits)]
        rate = ForceOfInfection(
            "foi",
            infectious=active,
            contact_rate=Param("contact_rate"),
            kind=FOIKind.FREQUENCY,
            infectiousness=weights,
        )
    elif form == "plain":
        rate = Param("contact_rate") * Reduce(where=active) / Reduce()
    else:
        raise ValueError(f"form must be one of {FORMS}, not {form!r}.")

    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", stage["S"], stage["E"], rate))
    model.add_flow(TransitionFlow("stabilisation", stage["E"], stage["L"], 0.5))
    model.add_flow(TransitionFlow("activation", stage["L"], active, Param("activation_rate")))
    model.add_flow(TransitionFlow("detection", active, stage["T"], 0.6))
    model.add_flow(TransitionFlow("treatment", stage["T"], stage["R"], 2.0))
    model.add_flow(TransitionFlow("relapse", stage["R"], stage["L"], 0.01))
    return model, pmap


def initial_state(pmap: PropertyMap) -> np.ndarray:
    stage = pmap.get_property("stage")
    y0 = np.zeros(pmap.size)
    y0[pmap.select(stage["S"])] = 99_000.0
    y0[pmap.select(stage["L"])] = 900.0
    active = pmap.select(stage["active"])
    y0[active] = 100.0 / active.size
    return y0


def make_fns(
    form: str,
    solver: str = "euler",
    *,
    dt: float = 0.05,
    years: float = 100.0,
    compile_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Jitted forward solve and reverse grad for ``form``, plus the pieces used to build them."""
    import jax
    import jax.numpy as jnp

    model, pmap = build(form)
    cm = model.compile(**(compile_kw or {}))
    y0 = jnp.asarray(initial_state(pmap))
    infectious = pmap.select(pmap.get_property("stage")["active"])
    kw: dict[str, Any] = {"t0": 0.0, "t1": years, "dt": dt, "save": SavePlan(ts=[years])}
    if solver == "euler":
        kw["solver"] = "euler"
    else:
        kw["max_steps"] = 100_000

    def prevalence(p: dict[str, Any]) -> Any:
        result = cm.run(p, y0, **kw)
        return jnp.sum(jnp.asarray(result.final_state.data)[infectious])

    def loss(theta: Any) -> Any:
        p = dict(PARAMS)
        p.update(zip(GRAD_PARAMS, theta, strict=True))
        return prevalence(p)

    theta = jnp.asarray([PARAMS[k] for k in GRAD_PARAMS])
    return {
        "cm": cm,
        "pmap": pmap,
        "y0": y0,
        "theta": theta,
        "solve": jax.jit(loss),
        "grad": jax.jit(jax.grad(loss)),
    }


def _nested_eqns(jaxpr: Any) -> int:
    """Equations in ``jaxpr``, counting those inside ``pjit``/``custom_jvp`` sub-jaxprs."""
    from jax.extend.core import ClosedJaxpr, Jaxpr

    total = 0
    for eqn in jaxpr.eqns:
        subs = [v for v in eqn.params.values() if isinstance(v, (ClosedJaxpr, Jaxpr))]
        if subs:
            total += sum(_nested_eqns(s.jaxpr if isinstance(s, ClosedJaxpr) else s) for s in subs)
        else:
            total += 1
    return total


def vf_eqns(cm: Any, y0: Any) -> tuple[int, int]:
    """Vector-field jaxpr equations: ``(top level, with nested sub-jaxprs inlined)``."""
    import jax

    p = cm.prepare(dict(PARAMS))
    jaxpr = jax.make_jaxpr(lambda y: cm.vector_field(0.0, y, p))(y0).jaxpr
    return len(jaxpr.eqns), _nested_eqns(jaxpr)


def best_time(fn: Callable[[Any], Any], arg: Any, repeat: int) -> float:
    import jax

    jax.block_until_ready(fn(arg))  # compile outside the timing
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn(arg))
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--form", choices=FORMS, default="terms")
    ap.add_argument("--solver", choices=("euler", "tsit5"), default="euler")
    ap.add_argument("--what", choices=("solve", "grad", "both"), default="both")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--no-fuse", action="store_true", help="compile(fuse_reductions=False)")
    ap.add_argument("--eqns", action="store_true", help="print VF jaxpr eqn counts and exit")
    args = ap.parse_args()
    compile_kw = {"fuse_reductions": False} if args.no_fuse else None

    if args.eqns:
        for form in FORMS:
            fns = make_fns(form, compile_kw=compile_kw)
            top, nested = vf_eqns(fns["cm"], fns["y0"])
            print(f"{form:6s} vf_eqns={top} vf_eqns_nested={nested}")
        return

    fns = make_fns(args.form, args.solver, compile_kw=compile_kw)
    out = [f"form={args.form}", f"solver={args.solver}", f"fuse={not args.no_fuse}"]
    top, nested = vf_eqns(fns["cm"], fns["y0"])
    out += [f"vf_eqns={top}", f"vf_eqns_nested={nested}"]
    if args.what in ("solve", "both"):
        out.append(f"solve_ms={1e3 * best_time(fns['solve'], fns['theta'], args.repeat):.2f}")
    if args.what in ("grad", "both"):
        out.append(f"grad_ms={1e3 * best_time(fns['grad'], fns['theta'], args.repeat):.2f}")
    print(" ".join(out))


if __name__ == "__main__":
    main()
