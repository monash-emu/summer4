# CP2 — calibration entry points built on the workflow pieces

Closes composability findings `CX16`, `CX17`, `CX18`, `CX19` and `CX24`
(`docs/evaluation/composability.md`, work package `CP2`). Roadmap step 30,
branch `feat/calib-composable`.

## Problem

The model-bound entry points predate the composable workflow stages and
duplicate them: `BayesianModel.sample(kind=...)` dispatches on a string, passes
through a keyword subset, cannot be seeded and returns a sealed
`InferenceData`; `BayesianModel.find_map` runs its own Adam loop and returns
only parameters. `wf.optimize` keeps its backend protocol private (`_Method`),
types `method=` as the two built-ins, returns a sealed `OptimizeResult` with
no optimiser state (so a run cannot be extended), and records only its own
stage in `OptimizeResult.history`.

## Design

### Optimisation (`CX17`, `CX18`, `CX24`)

```python
class OptimizeBackend(Protocol):          # public; was _Method
    def init(self, z0, key) -> state      # state carries "best_z" and "best_loss"
    def step(self, state) -> (state, best_z, best_loss)

class OptimizeMethod(Protocol):           # what optimize(method=) accepts
    def make(self, potential_fn, *, sites, template_z, learning_rate=None) -> OptimizeBackend
```

`wf.Optax` and `wf.CMAES` are the built-in methods; any object with `make`
works. The learning-rate probe applies only to `wf.Optax` without a
user-supplied `optimizer`.

`wf.OptimizeRun` replaces `OptimizeResult`. It holds the backend, the batched
optimiser states, convergence and restart bookkeeping, and the loss trace:

```python
run = wf.OptimizeRun(bm, candidates, method=, tuning=, reserve=, chunk_steps=, seed=, batch_size=)
run.extend(max_steps=500)       # continue from the stored states; converged lanes stay frozen
run.candidates                  # scored best points, history = input history + one record per call
run.loss_trace, run.converged, run.restarts, run.learning_rate
run.best_params                 # constrained params of the best start, as a plain dict
run.chunk_program               # the jitted chunk over the state batch (for jaxpr checks)
```

`wf.optimize(bm, candidates, ..., max_steps=)` is exactly
`OptimizeRun(bm, candidates, ...)` followed by `.extend(max_steps=max_steps)`,
documented and tested. `run.history` is `run.candidates.history`
(cumulative).

### `BayesianModel.find_map` (`CX19`)

A one-start `wf.optimize`, returning the `OptimizeRun`:

```python
def find_map(self, init=None, *, steps=500, optimizer=None, seed=0) -> OptimizeRun:
    z0 = self.init_point(seed) if init is None else init
    start = Candidates.from_z(self, {k: v[None] for k, v in z0.items()})
    return wf.optimize(self, start, method=wf.Optax(optimizer or optax.adam(0.05)),
                       max_steps=steps, chunk_steps=min(50, steps), seed=seed)
```

`bm.find_map(...).best_params` is the old dict (now the best point seen, not
the last). New public pieces used here: `BayesianModel.init_point(seed)` and
`Candidates.from_z(bm, z)`.

### `BayesianModel.sample` (`CX16`)

A model-bound `wf.run_mcmc`, returning the `MCMCRun`:

```python
def sample(self, kernel="nuts", *, init=None, stop=None, num_warmup=500,
           num_samples=500, num_chains=1, seed=0, chain_method="vectorized",
           progress_bar=False, **kernel_kwargs) -> MCMCRun:
```

`kernel` is a numpyro kernel built by the caller (`NUTS(bm.numpyro_model(),
target_accept_prob=0.95)`) or a name (`"nuts"`, `"aies"`, `"ess"`, `"sa"`)
with `**kernel_kwargs` as sugar. `init` seeds chains from `Candidates`.
`stop=None` takes one chunk of `num_samples`; a `wf.StopRule` samples in
chunks of `num_samples` until it decides. Exactly
`wf.run_mcmc(self, init, make_kernel=..., num_chains=, num_warmup=,
chunk_samples=num_samples, stop=stop or one_chunk, seed=, chain_method=,
progress_bar=)`. `bm.posterior_runs` accepts the `MCMCRun` as `draws`.

## Callers to update

`examples/notebooks/17-priors-and-likelihoods.ipynb` (find_map),
`18-calibration.ipynb` (sample), `20-`/`21-` (optimize result attributes),
`docs/textbook/20-calibration.ipynb` (both), `tests/test_epi_calibration.py`,
`tests/test_calib_workflow_optimize.py` (private helpers), the tb-ports row
`KI19`.

## Tests

`tests/test_calib_composable.py`: a user-written `OptimizeMethod` runs through
`wf.optimize`; `extend` continues from stored state (Adam moments kept: two
extends equal one long run); `optimize` equals its expansion; history is
cumulative; `find_map` equals its expansion and `best_params` is the best
start; `sample` equals `wf.run_mcmc`, accepts a kernel instance, seeds from
`Candidates`, and chunks under a `StopRule`; `posterior_runs` accepts an
`MCMCRun`.

## Notebook

`examples/notebooks/24-calibration-entry-points.ipynb`, an acceptance page
(notebook 22 / 23 style). Step 28's notebook moves to 25.
