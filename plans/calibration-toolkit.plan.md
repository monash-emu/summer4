---
name: calibration-toolkit
overview: Composable calibration workflow (Phase J, WP19) — LHS design → pick best → multi-start optimisation (optax / gradient-free) with auto-tuning and restarts → MCMC seeded from optimised points with automated run length → sampled outputs, spaghetti and scenario plots.
todos:
  - id: land-plan
    content: docs/calibration-toolkit-plan branch — this plan, roadmap Phase J (steps 24–28), step-15 correction, ledger CW block + WP19
    status: pending
  - id: s24
    content: Step 24 feat/calib-candidates — Candidates, Prior.icdf, lhs, evaluate, best, save/load
    status: pending
  - id: s25
    content: Step 25 feat/calib-multistart — optimize() with optax backend, chunked vmapped scan, auto LR probe, plateau/convergence, restarts
    status: pending
  - id: s26
    content: Step 26 feat/calib-gradient-free — CMA-ES backend (evosax) on the same driver
    status: pending
  - id: s27
    content: Step 27 feat/calib-seeded-mcmc — run_mcmc seeded from Candidates, StopRule (rhat/ESS/divergence/budget), chunked continuation
    status: pending
  - id: s28
    content: Step 28 feat/calib-outputs — draws from idata/Candidates into posterior_runs, spaghetti/ribbon/scenario/diagnostic plots, end-to-end notebook
    status: pending
isProject: false
---

# Calibration toolkit (Phase J, WP19)

## Context

The user wants end-to-end calibration workflows like estival's (the summer2
ecosystem's calibration library, which the Kiribati analysis used):

1. sample a Latin hypercube of M points from the priors, keep the N best;
2. optimise from each (optax / Adam / configurable, **plus a gradient-free
   option**), with automated tuning and restarts;
3. seed a Bayesian calibration run from the optimised points, with automated
   run length;
4. sampled outputs, spaghetti plots and scenario outputs after calibration.

Not every run uses every stage. The toolkit is a **composable pipeline: plain
stage functions that pass one shared data type** (user decision). No pipeline
class for now.

What exists today (on `feat/epi-sampling`, roadmap step 14, **not yet
committed**): `BayesianModel` in `src/summer4/epi/calibration/model.py`, with
`numpyro_model()`, `log_density(z)` over unconstrained sites, `find_map()` (a
single start with a Python step loop) and `sample(kind=...)` (no `init_params`,
no convergence loop). Step 15 (`feat/epi-posterior-runs`) is planned. It adds
`bm.posterior_runs(idata, scenarios=..., batch_size=...)` with `.quantiles()`
and `.differences()`. Neither step has LHS, batched evaluation, multi-start runs,
seeding, convergence-driven stopping or plots.

**Sequencing (user decision):** land step 14 as it is and step 15 as planned.
Then new **Phase J, steps 24–28**, on the existing roadmap — **directly after
step 15 and before Track G (step 16)**. The landing order is
14 → 15 → 24 → … → 28 → 16. Steps 24–27 depend technically only on step 14,
but are not started before step 15 lands.

## Design

### Placement

Everything goes in `summer4.epi.calibration` (calibration is epi-side; see the
epi-layer rule). One new subpackage, `summer4.epi.calibration.workflow`, holds
the stage functions. It is re-exported as `from summer4.epi.calibration import
workflow as wf`. Heavy dependencies are imported lazily behind extras, as
`model.py` already does with `_require_numpyro` and similar helpers.

### The shared value: `Candidates` (`workflow/candidates.py`)

```python
@dataclass(frozen=True)
class Candidates:
    sites: tuple[str, ...]              # bm.prior_names() order
    z: dict[str, Array]                 # unconstrained, leading axis n
    params: dict[str, Array]            # constrained view, same leading axis
    log_density: Array | None           # (n,), None until evaluated
    ok: Array | None                    # (n,) bool; False = failed solve (log_density <= -1e29)
    history: tuple[StageRecord, ...]    # (stage name, settings dict, seconds): provenance
    def __len__(self) -> int
    def take(self, idx) -> Candidates
    def best(self, n: int) -> Candidates          # top-n by log_density, failed excluded; raises if unevaluated
    def concat(self, other) -> Candidates
    def to_frame(self) -> "pd.DataFrame"          # one row per candidate: params + log_density + ok
    def save(self, path) / Candidates.load(path)  # .npz + JSON sidecar; checkpoints between stages
    @classmethod
    def from_params(cls, bm, params) / from_idata(cls, bm, idata, n=None, seed=0)
```

Every stage takes `Candidates` and returns `Candidates`, or a result object
exposing `.candidates`, so stages chain in any order or can be skipped.

**New on `BayesianModel`** (step 24): `constrain(z) -> params` and
`unconstrain(params) -> z`, per site, via `numpyro.distributions.transforms.biject_to(prior.to_numpyro().support)`.
Both are jittable, and both are vmappable over a leading axis.

### Stages

| Stage | Function | Notes |
| --- | --- | --- |
| Design | `wf.lhs(bm, m, *, seed)`, `wf.prior_draws(bm, m, *, seed)` | LHS in the unit cube (numpy: permutation plus jitter per dimension, no scipy.qmc), mapped through the new `Prior.icdf(u)` for each site, including hierarchical scale sites. Host-side, producing constant arrays |
| Score | `wf.evaluate(bm, c, *, batch_size=64)` | `jax.lax.map(bm.log_density, z, batch_size=...)`, one jit with bounded memory. Sets `log_density` and `ok` |
| Select | `c.best(n)` | Plain method |
| Optimise | `wf.optimize(bm, c, *, method=wf.Optax() \| wf.CMAES(), tuning=wf.AutoTune(), reserve=None)` → `OptimizeResult` | See below |
| Sample | `wf.run_mcmc(bm, init=c \| None, *, kind="nuts", num_chains=4, stop=wf.StopRule())` → `SampleResult` | See below |
| Outputs | step-15 `bm.posterior_runs(...)` accepts `Candidates` as well as `InferenceData`; `wf.plot_*` | See below |

Canonical composition (the step-28 notebook shows this and two shorter variants):

```python
design  = wf.evaluate(bm, wf.lhs(bm, 2000, seed=0), batch_size=128)
fitted  = wf.optimize(bm, design.best(16), method=wf.Optax(), reserve=design)
post    = wf.run_mcmc(bm, init=fitted.candidates.best(4), num_chains=4)
runs    = bm.posterior_runs(post.idata, n=200, scenarios={"baseline": None, ...})
wf.plot_spaghetti(runs, "incidence", targets=bm.targets)
```

### `Prior.icdf` (step 24, `priors.py`)

`Prior.icdf(u: np.ndarray) -> np.ndarray` is host-side and uses `scipy.stats`
with each prior's own parameterisation. That covers Uniform, Normal, LogNormal,
TruncatedNormal, Beta and Gamma. scipy is already a transitive dependency of
jax. Declare it in the `calibration` extra. Test by round-tripping against the
numpyro `cdf` wherever numpyro implements one.

### Optimisation driver (steps 25–26, `workflow/optimize.py`)

- The objective is `potential_fn(z)` (= `-log_density`) from
  `bm._ensure_potential()`. Promote it to a public `bm.potential_fn` property in
  step 25.
- **One compiled program per chunk:** `jax.jit(jax.vmap(chunk))`, where `chunk`
  is a `lax.scan` over `chunk_steps` steps. Starts are the vmapped axis. A host
  loop runs chunks up to `max_steps`. The jaxpr must not grow with `steps` or
  with the number of starts. Test it with `jax.make_jaxpr` (AGENTS.md rule 2).
- **Backend protocol** (internal, `_Method`): `init(z0, key) -> state` and
  `step(state) -> (state, z_best, loss_best)`. Both backends share the chunking,
  convergence and restart logic.
  - `Optax(optimizer=None, learning_rate=0.05, plateau=True)`: by default
    `optax.inject_hyperparams(optax.adam)` chained with
    `optax.contrib.reduce_on_plateau`. Any user `GradientTransformation` is
    accepted as is, and then AutoTune's LR probe is skipped unless it was built
    with `inject_hyperparams`.
  - `CMAES(sigma0=0.1, population=None)` (step 26): evosax CMA-ES, centred on
    each start. Each step is one generation, with the population evaluated
    under `vmap`. That costs starts × population solves per generation. Say so
    in the docstring.
- **`AutoTune`** (the automated tuning and restarts):
  - `lr_grid=(1e-3, 1e-2, 1e-1)`, `probe_steps=50`, `probe_starts=4`: before
    the main run, vmap over learning rate × the top `probe_starts` starts (the
    learning rate is traced through `inject_hyperparams`), then pick the rate
    with the best median loss drop. Optax only.
  - `rtol=1e-6`, `patience=2` chunks: a start is **converged** when its
    relative loss improvement stays below `rtol` for `patience` chunks. After
    that it is frozen by a `where(active, new, old)` mask, so the program shape
    stays fixed.
  - `max_restarts=3`: a start whose loss goes non-finite, or whose `ok` is
    False, is **restarted**. It takes the next unused point from `reserve`
    (a scored `Candidates`), or a jitter around the current best when there is
    no reserve.
  - It stops when every start has converged or `max_steps` is reached.
- `OptimizeResult`: `candidates` (final points, scored), `loss_trace`
  (n_chunks × n_starts), `converged`, `restarts`, `learning_rate`, and
  `history`.

### Seeded MCMC with automated run length (step 27, `workflow/mcmc.py`)

```python
@dataclass(frozen=True)
class StopRule:
    rhat: float | None = 1.05          # user default
    ess: float | None = 100            # min bulk ESS over all sites; user default
    max_divergence_frac: float | None = None   # NUTS: retry warmup above this
    max_samples: int = 5000            # per chain
    max_seconds: float | None = None
```

Any combination can be set; `None` disables a criterion.

- The stage builds `numpyro.infer.MCMC` itself from `bm.numpyro_model()`, so
  step 14's `sample()` is untouched. It uses `chain_method="vectorized"`,
  `num_warmup`, and `num_samples=chunk_samples`.
- **Seeding:** `init_params` holds the unconstrained `z` of each chain, stacked.
  Candidates are cycled when there are fewer candidates than chains, and jitter
  (`jitter=0.05` in unconstrained space) is added. AIES/ESS walkers require
  distinct starts: raise if duplicates remain when jitter is 0. `init=None`
  keeps numpyro's default init.
- **Run length:** after each chunk, set
  `mcmc.post_warmup_state = mcmc.last_state` and call
  `mcmc.run(mcmc.post_warmup_state.rng_key)`. Accumulate
  `get_samples(group_by_chain=True)` and `diverging`. Compute diagnostics on
  the host with `numpyro.diagnostics.split_gelman_rubin` /
  `effective_sample_size`, not arviz, because the arviz pins disagree across
  envs (see Risks). Stop when all enabled criteria pass, or when a budget runs
  out.
- **Divergence retry (NUTS):** if the first chunk's divergent fraction exceeds
  `max_divergence_frac`, rebuild the kernel with `target_accept_prob` stepped
  0.8 → 0.9 → 0.95 and double `num_warmup`, with `max_retries=2`.
- Non-convergence **warns rather than raises**. `SampleResult` holds `idata`
  (built from the accumulated dict plus sample_stats), `diagnostics` (a frame
  per site with rhat, ess_bulk and ess_tail), `converged`, `reason`, `chunks`,
  and `.candidates` (the last state of each chain, for re-seeding).

### Post-calibration outputs (step 28, `workflow/plots.py`)

- `Candidates.from_idata(bm, idata, n, seed, burn_in)` draws posterior points.
  `bm.posterior_runs` accepts `Candidates`, so spaghetti plots work for LHS
  best-N and for optimised points too, not only for posteriors.
- It needs a step-15 correction, recorded in the roadmap now: step 15's runs
  object must keep **per-draw outputs** (`runs.samples(scenario, output)` →
  `(n, t)`), not only quantiles.
- Plotly figures, with plotly imported lazily and added to the `calibration`
  extra. Each figure has a title and axis labels, and targets are overlaid as
  markers:
  - `plot_spaghetti(runs, output, scenario=, n=50, targets=)`
  - `plot_ribbons(runs, output, q=(.025,.25,.5,.75,.975), scenarios=, targets=)`
  - `plot_scenarios(runs, output, ref="baseline")`, which shows differences
  - diagnostics: `plot_design(candidates)` (log density against each
    parameter), `plot_optimisation(result)` (loss traces),
    `plot_chains(sample_result)` (per-chain traces with rhat in the title)
- This does not touch `Output.plot`. That matplotlib coupling stays in
  `futureplans/trace-plot-backend-coupling.md`.

### Ledger (the pinned memory: machine-checkable)

A new block, **`Calibration workflow ledger`**, goes in
`docs/evaluation/coverage-ledger.md`, with rows `CW1`–`CW11`, all `none` at
first, and WP19 declared. It is tallied **separately** from the summer2 API
totals, because estival isn't a summer2 symbol. `scripts/coverage_report.py`
gets a new `read_block` name, and `tests/test_coverage_ledger.py` asserts that
block's statuses and quoted totals.

| ID | Capability | Step |
| --- | --- | --- |
| CW1 | LHS design over priors (M points) | 24 |
| CW2 | Batched, memory-bounded scoring of a design | 24 |
| CW3 | Pick N best | 24 |
| CW4 | Stage checkpoint to disk / reload | 24 |
| CW5 | Multi-start gradient optimisation (configurable optax) | 25 |
| CW6 | Automated optimiser tuning, convergence and restarts | 25 |
| CW7 | Multi-start gradient-free optimisation | 26 |
| CW8 | MCMC seeded from candidates | 27 |
| CW9 | Automated MCMC run length (rhat / ESS / divergence / budget) | 27 |
| CW10 | Sampled-output spaghetti and quantile ribbons | 28 |
| CW11 | Scenario outputs post calibration, plotted | 28 |

## Roadmap wrapper (for low-context workers)

The plan lands first on **`docs/calibration-toolkit-plan`**, cut from `main`.
Use a **git worktree** so the uncommitted step-14 work on `feat/epi-sampling`
is not disturbed. That branch adds:

- this file as `plans/calibration-toolkit.plan.md`;
- Phase J in `docs/dev/roadmap.md`: rows 24–28 (`planned`, WP19, plan
  `plans/calibration-toolkit.plan.md`, Section `Step N`, Closes `CW…`), plus a
  full `## Step N` section for each, with Summary, Read first, Do, Exit checks
  and Handoff;
- a Phase J line under *Steps* fixing the landing order 14 → 15 → 24–28 → 16,
  with step 15's handoff naming step 24 and step 28's naming step 16;
- a Corrections-table row for step 15's per-draw outputs;
- the ledger block, the checker support and the test.

If the roadmap checker validates `Closes` IDs against the tb-ports/ledger
IDs, extend it to accept `CW*`.

Each step section follows the existing template. For every step, *Read first*
is `AGENTS.md`, then this plan's step section, then
`src/summer4/epi/calibration/model.py`, then `docs/dev/run-stages.md`. *Exit
checks* are the standard checks plus the step's CW rows moved to `full`.
*Handoff* is the standard four-part commit.

### Step 24 — `feat/calib-candidates`
The job: `Candidates`, `StageRecord`, `Prior.icdf`, `bm.constrain` /
`bm.unconstrain`, `wf.lhs`, `wf.prior_draws`, `wf.evaluate`, `best`, and
`save`/`load`.

Tests (`tests/test_calib_workflow_candidates.py`):

- LHS stratification: each of the m strata is hit exactly once in each
  dimension.
- `icdf` round-trips for every prior.
- Bounds are respected.
- `evaluate` matches a loop of `bm.log_density` calls.
- A failed solve gives `ok` False.
- The `evaluate` jaxpr size does not depend on m.
- `best` ordering is correct.
- The save/load round trip is exact.

Notebook: `examples/notebooks/19-calibration-design.ipynb`. It scatters log
density against each parameter on the SIR fixture, and asserts the best 16
bracket the true parameter. Closes CW1–CW4.

### Step 25 — `feat/calib-multistart`
The job: `bm.potential_fn`, `wf.optimize`, `Optax`, `AutoTune`, and
`OptimizeResult`.

Tests:

- Each start converges to the same MAP as `bm.find_map` from that start.
- A start forced into NaN is restarted from `reserve`.
- Converged starts are frozen.
- The LR probe picks a finite rate.
- The chunk jaxpr size is independent of `max_steps` and of the number of
  starts.

Notebook: `20-calibration-multistart.ipynb`. It plots the loss traces for each
start, and asserts that every converged start is within tolerance of the true
parameters. Closes CW5–CW6.

### Step 26 — `feat/calib-gradient-free`
The job: the `CMAES` backend through evosax, in a new `gradient-free` extra
that keeps `calibration` light. Pin the evosax version and wrap its ask/tell
calls in the `_Method` adapter only.

Tests use the same driver contract as step 25, plus one case where gradients
are unavailable (the potential is wrapped in `stop_gradient`) and CMA-ES still
converges.

Notebook: `21-calibration-gradient-free.ipynb`. It compares Optax against
CMA-ES loss per solve. Closes CW7.

**Risk:** evosax may not resolve against the `jax06` env. If it doesn't, gate
those tests with `importorskip` in that env and write a `futureplans/` note.

### Step 27 — `feat/calib-seeded-mcmc`
The job: `wf.run_mcmc`, `StopRule`, and `SampleResult`.

Tests:

- Seeded chains start at the given `z`: the first draw is near it when warmup
  is 0.
- It stops early once rhat and ESS pass.
- It continues when a deliberately short chunk fails.
- It warns and returns `converged=False` when the budget runs out.
- Divergence retry raises `target_accept_prob`.
- AIES rejects duplicate walkers when jitter is 0.

Notebook: `22-calibration-seeded-mcmc.ipynb`. It compares chain traces seeded
from optimised points against prior-seeded chains, and asserts that the seeded
run converges in fewer chunks. Closes CW8–CW9.

### Step 28 — `feat/calib-outputs` (after step 15)
The job: `Candidates.from_idata`, `posterior_runs` accepting `Candidates`, and
the `plot_*` family.

Tests check trace counts and names, target overlay markers, scenario
difference signs, and that the functions take `Candidates` from each stage.

Notebook: `23-calibration-workflow.ipynb`. It runs the full chain end to end,
plus two shorter compositions: LHS → best → runs, and optimise → runs. It
asserts that the posterior median covers the targets. Closes CW10–CW11.

Notebook numbers are first-come. Step 15 holds `18`.

## Risks / notes for workers

- **arviz pins disagree** in step 14's uncommitted `pixi.toml`:
  `[pypi-dependencies] arviz >=1.3,<2` against feature `arviz >=0.20`. The
  arviz 1.x API was restructured. That's why convergence uses
  `numpyro.diagnostics`. Flag it to the step-14 worker. It isn't fixed here.
- vmapping an adaptive diffrax solve runs every lane to the slowest lane's step
  count. Document this in `optimize`/`evaluate`, and prefer a fixed-step solver
  for large designs.
- `-1e30` is the failed-solve sentinel from `numpyro_model`. `ok` is derived
  from it (`log_density > -1e29`). Keep the two in sync, or expose `ok`
  directly from `bm` later (futureplans note if needed).

## Verification

- Planning branch: `pixi run roadmap`, `pixi run coverage-write && pixi run
  coverage`, `pixi run test -k "roadmap or coverage_ledger"`, and `pixi run
  lint`.
- Each implementation step: the AGENTS.md required checks, plus `pixi run -e
  docs docs-strict`. The step's notebook is a blocking user gate:
  `pixi run notebook`, with a checklist in the PR body naming each notebook and
  the claim to check.
- End to end (step 28): `23-calibration-workflow.ipynb` runs every stage on the
  SIR fixture in both pixi envs (`jax06`, `jaxlatest`).
