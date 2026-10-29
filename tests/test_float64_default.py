"""Importing summer4 turns on float64 (``jax_enable_x64``) unless the user chose otherwise."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import FlowModel, Param, Property, PropertyMap, TransitionFlow
from summer4.epi import ForceOfInfection
from summer4.flows.compiled import CompiledModel

STATE = Property("state", ("S", "I", "R"))
PMAP = PropertyMap.from_property(STATE)
# One infectious person in 100 million: float32 cannot represent S0 - 1 (its spacing at 1e8 is 8).
Y0 = np.array([1e8 - 1.0, 1.0, 0.0])
PARAMS = {"contact_rate": 0.15, "recovery_rate": 0.1}
RUN = dict(t0=0.0, t1=365.0, dt=1.0)

# Prints the flag and the dtype of a solved model, after the given setup lines.
_PROBE = """
{setup}
import jax, numpy as np
from summer4 import FlowModel, Property, PropertyMap, TransitionFlow
state = Property("state", ("S", "I"))
model = FlowModel(PropertyMap.from_property(state))
model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.1))
result = model.compile().run({{}}, np.array([1.0, 0.0]), t0=0.0, t1=1.0, dt=1.0)
print(jax.config.jax_enable_x64, result["compartments"].values.data.dtype)
"""


def _probe(setup: str = "", env_x64: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    if env_x64 is not None:
        env["JAX_ENABLE_X64"] = env_x64
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(setup=setup)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def _model() -> CompiledModel:
    model = FlowModel(PMAP)
    foi = ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("contact_rate"))
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], foi))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], Param("recovery_rate")))
    return model.compile()


def _compartments(x64: bool) -> np.ndarray:
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", x64)
    try:
        result = _model().run(PARAMS, Y0, **RUN)
        return np.asarray(result["compartments"].values.data)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_import_turns_on_float64() -> None:
    assert _probe("import summer4") == "True float64"


def test_import_after_jax_still_turns_on_float64() -> None:
    assert _probe("import jax, jax.numpy as jnp; jnp.zeros(1); import summer4") == "True float64"


@pytest.mark.parametrize(("value", "expected"), [("0", "False float32"), ("1", "True float64")])
def test_environment_variable_wins(value: str, expected: str) -> None:
    assert _probe("import summer4", env_x64=value) == expected


def test_opt_out_after_import() -> None:
    setup = "import summer4, jax; jax.config.update('jax_enable_x64', False)"
    assert _probe(setup) == "False float32"


def test_this_process_is_float64() -> None:
    if os.environ.get("JAX_ENABLE_X64") is not None:
        pytest.skip("JAX_ENABLE_X64 is set for this test run")
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64
    assert _model().run(PARAMS, Y0, **RUN)["compartments"].values.data.dtype == jnp.float64


def test_float64_keeps_small_changes_to_large_compartments() -> None:
    c64 = _compartments(True)
    c32 = _compartments(False)
    assert c64.dtype == np.float64
    assert c32.dtype == np.float32
    # Population is conserved to rounding at 1e8 in float64, but not in float32.
    assert np.abs(c64.sum(axis=1) - 1e8).max() < 1e-6
    assert np.abs(c32.astype(np.float64).sum(axis=1) - 1e8).max() > 1.0
    # Early infections show up in S in float64; in float32 S cannot move off 1e8.
    assert Y0[0] - c64[30, 0] == pytest.approx(9.97, abs=0.01)
    assert float(c32[30, 0]) == 1e8
