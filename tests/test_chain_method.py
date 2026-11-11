"""``chain_method="auto"`` is currently vectorized; ``"parallel"`` is opt-in."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import jax
import pytest

from summer4.epi.calibration import resolve_chain_method


@pytest.mark.parametrize(
    ("devices", "chains", "expected"),
    [
        (1, 4, "vectorized"),
        (4, 4, "vectorized"),
        (12, 4, "vectorized"),
        (4, 1, "vectorized"),
        (2, 4, "vectorized"),
    ],
)
def test_auto_is_vectorized_for_every_device_count(
    monkeypatch: pytest.MonkeyPatch, devices: int, chains: int, expected: str
) -> None:
    monkeypatch.setattr(jax, "local_device_count", lambda: devices)
    assert resolve_chain_method("auto", chains) == expected


@pytest.mark.parametrize("method", ["sequential", "vectorized", "parallel", "bogus"])
def test_explicit_method_passes_through(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    monkeypatch.setattr(jax, "local_device_count", lambda: 12)
    assert resolve_chain_method(method, 4) == method


_CHILD = textwrap.dedent("""
    import json
    import sys
    import numpy as np
    import jax
    from summer4 import (FlowMass, FlowModel, OutputSet, Param, Property, PropertyMap,
                         Target, TargetSet, TransitionFlow)
    from summer4.epi import BayesianModel, ForceOfInfection, Poisson, Uniform

    state = Property("state", ("S", "I", "R"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(TransitionFlow("infection", state["S"], state["I"],
        ForceOfInfection("foi", infectious=state["I"], contact_rate=Param("contact_rate"))))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    model.set_initial_population({state["S"]: 9990.0, state["I"]: 10.0})
    cm = model.compile()
    run_kwargs = dict(t0=0.0, t1=60.0, dt=1.0)
    sample_kwargs = json.loads(sys.argv[1])
    if sample_kwargs.pop("euler", False):
        run_kwargs["solver"] = "euler"
    outputs = OutputSet()
    outputs["new_infections"] = FlowMass("infection").sum_by()
    true = {"contact_rate": 0.5}
    res = cm.run(true, save=outputs.plan(), **run_kwargs)
    truth = outputs.evaluate(res, true)["new_infections"]
    obs_days = np.arange(5.0, 41.0, 5.0)
    expected = np.asarray(truth.at_times(obs_days).values).reshape(-1)
    observed = np.random.default_rng(1).poisson(expected).astype(float)
    target = Target(key="new_infections", times=obs_days, values=observed,
                    likelihood=Poisson("sum"))
    bm = BayesianModel(cm, {}, [Uniform("contact_rate", 0.1, 1.5)], TargetSet((target,)),
                       outputs=outputs, run_kwargs=run_kwargs)
    out = {"devices": jax.local_device_count()}
    run = bm.sample(num_warmup=200, num_samples=200, num_chains=4, seed=3, **sample_kwargs)
    draws = np.asarray(run.samples["contact_rate"]).reshape(-1)
    out.update(method=run.mcmc.chain_method, median=float(np.median(draws)))
    print("RESULT" + json.dumps(out))
    """)


def _run_child(**sample_kwargs: object) -> dict[str, object]:
    env = dict(os.environ, SUMMER4_HOST_DEVICES="4", JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(sample_kwargs)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def test_default_run_is_vectorized_and_recovers_parameter() -> None:
    out = _run_child()
    assert out["devices"] == 4
    assert out["method"] == "vectorized"
    assert abs(out["median"] - 0.5) < 0.05, out


def test_explicit_parallel_with_euler_recovers_parameter() -> None:
    """Parallel chains stay covered with the Euler solver (see CX45)."""
    out = _run_child(chain_method="parallel", euler=True)
    assert out["method"] == "parallel"
    assert abs(out["median"] - 0.5) < 0.05, out


def test_auto_keeps_ensemble_kernels_vectorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """AIES and ESS only support vectorized chains, so ``"auto"`` must not pick parallel."""
    from numpyro.infer import AIES

    from summer4.epi.calibration import workflow as wf

    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    captured: list[str] = []
    import numpyro.infer as ni

    def spy(*args: object, **kwargs: object) -> object:
        captured.append(str(kwargs["chain_method"]))
        raise RuntimeError("stop")

    monkeypatch.setattr(ni, "MCMC", spy)
    with pytest.raises(RuntimeError, match="stop"):
        wf.run_mcmc(
            None,  # type: ignore[arg-type]
            make_kernel=lambda: AIES(lambda: None),
            num_chains=4,
        )
    assert captured == ["vectorized"]
