"""Install-path smoke test: a three-compartment SIR through compile, run, Result.

The CI job runs this with summer4 installed from the git tag, not from this
checkout. Infection is a constant per-capita rate, so the susceptible
compartment has a closed form: S(t) = S0 * exp(-beta * t).
"""

from __future__ import annotations

import math
import os
from importlib.metadata import version

import numpy as np

from summer4 import (
    Compartments,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    Result,
    SavePlan,
    SaveRequest,
    Trace,
    TransitionFlow,
)

# Hand-computed: dS/dt = -0.3 S with S(0) = 999, so S(1) = 999 * exp(-0.3).
BETA = 0.3
S0 = 999.0
T1 = 1.0
EXPECTED_S = S0 * math.exp(-BETA * T1)  # 740.707...
POPULATION = 1000.0


def _reject_editable_checkout() -> None:
    """Fail if this process imported the repo source instead of an install."""
    if os.environ.get("SUMMER4_SMOKE_REQUIRE_INSTALL") != "1":
        return
    import summer4

    path = summer4.__file__.replace("\\", "/")
    if "/src/summer4/" in path:
        raise SystemExit(
            f"imported the checkout at {path}, not an installed package. "
            "The smoke job must not use the editable pixi environment."
        )


def main() -> None:
    """Compile a linear SIR, run it, and check the final susceptible count."""
    _reject_editable_checkout()
    installed = version("summer4")
    if installed != "0.2.0a3":
        raise SystemExit(f"expected summer4 0.2.0a3, installed {installed}")

    state = Property("state", ("S", "I", "R"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], BETA))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    compiled = model.compile()
    y0 = PropertyData.wrap(compiled.pmap, np.array([S0, 1.0, 0.0]))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments(), ts=np.array([0.0, T1]))})
    result = compiled.run(
        {},
        y0,
        t0=0.0,
        t1=T1,
        dt=0.1,
        save=plan,
        solver="dopri5",
        rtol=1e-8,
        atol=1e-10,
    )
    if not isinstance(result, Result):
        raise SystemExit(f"run() returned {type(result).__name__}, expected Result")
    trace = result["compartments"]
    if not isinstance(trace, Trace):
        raise SystemExit(f"compartments trace has type {type(trace).__name__}")
    final = np.asarray(trace.values.data)[-1]
    if final.shape != (3,):
        raise SystemExit(f"expected 3 compartments, got shape {final.shape}")
    np.testing.assert_allclose(final[0], EXPECTED_S, rtol=1e-4, atol=1e-3)
    # Default JAX is float32, so a conserved population of 1000 drifts by ~1e-3.
    np.testing.assert_allclose(float(final.sum()), POPULATION, rtol=1e-5, atol=1e-2)
    print(
        f"downstream smoke ok: summer4 {installed}, "
        f"S({T1})={float(final[0]):.6f} (expected {EXPECTED_S:.6f})"
    )


if __name__ == "__main__":
    main()
