"""``chain_method="auto"`` runs chains in parallel when there are enough devices."""

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
        (4, 4, "parallel"),
        (12, 4, "parallel"),
        (4, 1, "vectorized"),
        (2, 4, "vectorized"),
    ],
)
def test_auto_follows_device_count(
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
    for label, kwargs in (("auto", {}), ("vectorized", {"chain_method": "vectorized"})):
        run = bm.sample(num_warmup=200, num_samples=200, num_chains=4, seed=3, **kwargs)
        draws = np.asarray(run.samples["contact_rate"]).reshape(-1)
        out[label] = {"method": run.mcmc.chain_method, "median": float(np.median(draws))}
    print("RESULT" + json.dumps(out))
    """)


def test_auto_runs_parallel_and_recovers_parameter() -> None:
    env = dict(os.environ, SUMMER4_HOST_DEVICES="4", JAX_PLATFORMS="cpu")
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=900
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT"))
    out = json.loads(line[len("RESULT") :])
    assert out["devices"] == 4
    assert out["auto"]["method"] == "parallel"
    assert out["vectorized"]["method"] == "vectorized"
    for label in ("auto", "vectorized"):
        assert abs(out[label]["median"] - 0.5) < 0.05, out


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
