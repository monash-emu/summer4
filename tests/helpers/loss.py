"""Shared loss used by solver backend agreement tests.

Defined once so Euler and diffrax differentiate identical source.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from summer4 import (
    Compartments,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)

# Observation times shared by both backends (coincides with a fine Euler grid).
TS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
TARGET = np.asarray([200.0, 180.0, 160.0, 140.0, 120.0], dtype=np.float64)


def analytic_sir_model() -> tuple[Any, PropertyData, Property]:
    """Linear S→I→R system with an analytic closed form for S."""
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.3))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0], dtype=np.float64))
    return cm, y0, state


def infected_loss(solver: str, *, dt: float = 1e-4) -> Any:
    """Return ``loss(scale)`` comparing infected totals to ``TARGET`` at ``TS``.

    ``scale`` multiplies the infection rate via params — both backends close over
    the same compiled model and save plan, so ``jax.grad`` must agree.
    """
    cm, y0, state = analytic_sir_model()
    plan = SavePlan(
        requests={"infected": SaveRequest(Compartments(where=state["I"]), ts=TS)},
    )

    def loss(scale: Any) -> Any:
        # Rebuild rate through a trivial param multiply in derived_fn style:
        # infection rate is constant 0.3; we scale the initial infected seed.
        y = y0._with_data(y0.data * jnp.array([1.0, scale, 1.0]))
        kwargs: dict[str, Any] = {
            "t0": 0.0,
            "steps": int(round(1.0 / dt)),
            "dt": dt,
            "save": plan,
            "solver": solver,
        }
        if solver != "euler":
            kwargs["rtol"] = 1e-7
            kwargs["atol"] = 1e-9
        res = cm.run({}, y, **kwargs)
        pred = jnp.sum(jnp.asarray(res["infected"].at_times(TS).values.data), axis=-1)
        return jnp.mean((pred - TARGET) ** 2)

    return loss
