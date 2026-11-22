"""``sample_until`` compiles a bounded number of times, however many chunks it runs.

numpyro's ``MCMC.run`` recompiles on every call for ``chain_method="parallel"`` and
``"vectorized"``; ``MCMCRun`` maps the chains through a cached transform instead
(``futureplans/mcmc-chunk-recompile.md``). These tests count XLA compiles with
``jax.monitoring`` and check that the draws match numpyro's own mapping exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from summer4.epi.calibration import workflow as wf  # noqa: E402

_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_COMPILES = [0]


def _count_compiles(event: str, duration: float, **kwargs: Any) -> None:
    if event == _COMPILE_EVENT:
        _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compiles)

METHODS = ["vectorized", "parallel"]


def _model() -> None:
    x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand([3]))
    numpyro.sample("y", dist.Normal(x.sum(), 1.0), obs=0.3)


def _mcmc(method: str, **kw: Any) -> MCMC:
    if method == "parallel" and jax.local_device_count() < 2:
        pytest.skip("parallel chains need two devices")
    options: dict[str, Any] = {
        "num_warmup": 20,
        "num_samples": 10,
        "num_chains": 2,
        "chain_method": method,
        "progress_bar": False,
    }
    options.update(kw)
    return MCMC(NUTS(_model), **options)


class _Chunks:
    """Stop after ``n`` chunks, recording the compiles each chunk caused."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.compiles: list[int] = []
        self._last = _COMPILES[0]

    def __call__(self, row: Mapping[str, Any]) -> tuple[bool, str] | None:
        self.compiles.append(_COMPILES[0] - self._last)
        self._last = _COMPILES[0]
        return (True, "done") if len(self.compiles) >= self.n else None


def _plain_chunks(method: str, n: int, seed: int = 0) -> np.ndarray:
    """``n`` chunks driven with plain numpyro calls: the reference draws."""
    mcmc = _mcmc(method)
    mcmc.run(jax.random.PRNGKey(seed))
    draws = [np.asarray(mcmc.get_samples(group_by_chain=True)["x"])]
    for _ in range(n - 1):
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key)
        draws.append(np.asarray(mcmc.get_samples(group_by_chain=True)["x"]))
    return np.concatenate(draws, axis=1)


@pytest.mark.parametrize("method", METHODS)
def test_later_chunks_do_not_compile(method: str) -> None:
    stop = _Chunks(6)
    wf.sample_until(_mcmc(method), stop, seed=0, warn=False)
    # The first chunk includes warmup and the second switches numpyro to its sampling
    # phase; from then on every chunk reuses the compiled function.
    assert stop.compiles[2:] == [0, 0, 0, 0]


@pytest.mark.parametrize("method", METHODS)
def test_draws_match_numpyro_mapping(method: str) -> None:
    run = wf.sample_until(_mcmc(method), _Chunks(4), seed=3, warn=False)
    np.testing.assert_array_equal(np.asarray(run.samples["x"]), _plain_chunks(method, 4, 3))


@pytest.mark.parametrize("method", METHODS)
def test_chain_method_restored_and_mcmc_still_usable(method: str) -> None:
    mcmc = _mcmc(method)
    run = wf.sample_until(mcmc, _Chunks(2), seed=0, warn=False)
    assert mcmc.chain_method == method
    run.extend(_Chunks(1), warn=False)
    assert mcmc.chain_method == method
    # Plain numpyro continues from the run's last state with the caller's own method.
    mcmc.post_warmup_state = mcmc.last_state
    mcmc.run(mcmc.post_warmup_state.rng_key)
    assert np.asarray(mcmc.get_samples(group_by_chain=True)["x"]).shape == (2, 10, 3)


@pytest.mark.parametrize("method", METHODS)
def test_after_warmup_until_matches_numpyro(method: str) -> None:
    def make(n: int) -> MCMC:
        return _mcmc(method, num_warmup=n)

    def check(row: Mapping[str, Any]) -> tuple[str, ...]:
        return () if int(row["num_warmup"]) >= 20 else ("short",)

    warm = wf.warmup_until(make, 10, check, seed=1, warn=False)
    ref_state = warm.mcmc.post_warmup_state
    stop = _Chunks(3)
    run = wf.sample_until(warm.mcmc, stop, seed=1, warn=False)
    # Already warmed up, so every chunk is in numpyro's sampling phase: only the first compiles.
    assert stop.compiles[1:] == [0, 0]

    ref = make(warm.mcmc.num_warmup)
    ref.post_warmup_state = ref_state
    draws = []
    ref.run(jax.random.PRNGKey(1))  # sample_until starts the first chunk from its seed
    draws.append(np.asarray(ref.get_samples(group_by_chain=True)["x"]))
    for _ in range(2):
        ref.post_warmup_state = ref.last_state
        ref.run(ref.post_warmup_state.rng_key)
        draws.append(np.asarray(ref.get_samples(group_by_chain=True)["x"]))
    np.testing.assert_array_equal(np.asarray(run.samples["x"]), np.concatenate(draws, axis=1))


def test_uncached_cases_keep_numpyro_mapping() -> None:
    from summer4.epi.calibration.workflow.mcmc import _cacheable_chain_method

    assert _cacheable_chain_method(_mcmc("vectorized")) == "vectorized"
    assert _cacheable_chain_method(_mcmc("sequential")) is None
    assert _cacheable_chain_method(_mcmc("vectorized", num_chains=1)) is None
    assert _cacheable_chain_method(_mcmc("vectorized", progress_bar=True)) is None
