# Euler backend is slow per run

On the Mac mini (12 cores; `benchmarks/results/runtime_defaults.json`, step 49 R0), one `run()`
with Euler at the documented `dt` takes about 32 ms (SIR, 1000 steps), 38 ms (Lotka-Volterra,
1000 steps) and 304 ms (TB-scale, 185 steps). The default adaptive Tsit5 takes 1 ms, 1.2 ms and
16 ms, with far smaller errors, although each of its steps does more work than an Euler step. So
Euler spends roughly 30 us per step on the small models, where Tsit5 spends about 20 us per
*accepted-or-rejected* step.

That is not intrinsic to a fixed-step method, so it points at the Euler backend
(`src/summer4/solvers/euler_backend.py`): per-step overhead in its loop, how it writes saved
states, or the XLA problem already recorded in
[`tb-scale-euler-xla-simplifier.md`](tb-scale-euler-xla-simplifier.md).

**Done would be:** a profile (`jax.profiler`, or a `make_jaxpr` / HLO look at the scan body)
that says where the time goes, and either a fix or a note that it cannot be fixed. Euler is
no longer the default, so this matters for tests that pin `solver="euler"` (`SUMMER4_SOLVER=euler`
test suites) and for anyone comparing solvers, not for the default path.

## A related question: tuning pilots

`src/summer4/epi/calibration/workflow/tune.py` keeps `chain_method="vectorized"` as its default on
purpose: its pilots compare per-gradient cost, and parallel chains would measure something else.
Whether the pilots should follow `"auto"` like `BayesianModel.sample` and `wf.run_mcmc` do is
open. If they should, the cost comparisons need to say which method they measured.
