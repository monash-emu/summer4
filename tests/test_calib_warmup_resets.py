"""``warmup_until`` leaves numpyro's step-size-restart draws out of its HMC statistics.

At the end of each adaptation window numpyro restarts its step-size search from ten times a
fresh guess, and the next draw diverges in every chain even on a standard normal. Those
draws fall in the checked half of every round, so a short warmup failed the 1% divergence
limit whatever the posterior.
"""

from __future__ import annotations

from typing import Any

import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from summer4.epi.calibration import workflow as wf  # noqa: E402
from summer4.epi.calibration.workflow.warmup import _reset_draws  # noqa: E402


def _gauss() -> None:
    numpyro.sample("x", dist.Normal(0.0, 1.0).expand([3]))


def _factory(**nuts: Any) -> Any:
    def make(n: int) -> MCMC:
        return MCMC(
            NUTS(_gauss, **nuts),
            num_warmup=n,
            num_samples=10,
            num_chains=4,
            chain_method="vectorized",
            progress_bar=False,
        )

    return make


def _round(**kw: Any) -> dict[str, Any]:
    nuts = kw.pop("nuts", {})
    warm = wf.warmup_until(_factory(**nuts), 150, max_warmup=150, seed=0, warn=False, **kw)
    return dict(warm.progress.iloc[-1])


def test_reset_draws_follow_numpyro_windows() -> None:
    # numpyro's windows for 300: (0,74) (75,99) (100,149) (150,249) (250,299).
    assert _reset_draws(300, 1).tolist() == [1, 101, 151, 251]
    assert _reset_draws(150, 1).tolist() == [1, 101]
    assert _reset_draws(150, 2).tolist() == [1, 2, 101, 102]
    assert _reset_draws(300, 0).tolist() == []
    assert _reset_draws(10, 1).tolist() == [1]  # one window: only the start of warmup


def test_short_warmup_of_a_standard_normal_passes_divergences() -> None:
    row = _round()
    assert row["reset_divergences"] >= 3  # the post-restart draw diverges in (nearly) every chain
    assert row["divergence_frac"] == 0.0
    assert "divergences" not in (row["failed"] or "")


def test_skip_zero_checks_every_draw() -> None:
    row = _round(skip_after_reset=0)
    assert row["reset_divergences"] == 0
    assert row["divergence_frac"] > 0.01
    assert "divergences" in (row["failed"] or "")


def test_no_restarts_without_step_size_adaptation() -> None:
    row = _round(nuts={"adapt_step_size": False, "step_size": 0.5})
    assert row["reset_divergences"] == 0


def test_rejects_negative_skip() -> None:
    with pytest.raises(ValueError, match="skip_after_reset"):
        wf.warmup_until(_factory(), 150, skip_after_reset=-1)
