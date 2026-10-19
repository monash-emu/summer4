# Step 46 — `Posterior` protocol and `NumpyroPosterior` (design spike)

Implements step NH2 of `plans/tb-nh-joint-calibration.plan.md` (§4, §5 *NH2*), roadmap step 46.
This page is the opus-tier design spike. Where it differs from §4, this page wins; each difference
is listed under *Deviations from §4* with its reason. Closes `CX21`, `CX22`, `CX35` (package `CP3`).

## The protocol

`src/summer4/epi/calibration/posterior.py` (new), exported from `summer4.epi.calibration` and from
`summer4.epi.calibration.workflow` (`wf.Posterior`, `wf.NumpyroPosterior`):

```python
@runtime_checkable
class Posterior(Protocol):
    """What the workflow stages read from a model. BayesianModel satisfies it unchanged."""
    def numpyro_model(self) -> Callable[[], None]: ...        # for NUTS(...) / MCMC
    @property
    def potential_fn(self) -> Callable[[Mapping[str, Any]], Any]: ...  # -log density, unconstrained
    def log_density(self, unconstrained: Mapping[str, Any]) -> Any: ...
    def constrain(self, z: Mapping[str, Any]) -> dict[str, Any]: ...      # leading batch axis allowed
    def unconstrain(self, params: Mapping[str, Any]) -> dict[str, Any]: ...
    def prior_names(self) -> tuple[str, ...]: ...             # every latent sample site, trace order
    def site_priors(self) -> Sequence[Prior]: ...             # the sites wf.lhs can stratify
```

Exactly the members the stages read today (`grep` of `workflow/*.py`); `merge_params` is not read
and is not in the protocol. `init_point` and `postprocess_fn` are public on both implementations
but are not protocol members, because no stage reads them.

## `NumpyroPosterior`

```python
class NumpyroPosterior:
    def __init__(self, model: Callable[[], None], *, site_priors: Sequence[Prior] | None = None,
                 seed: int = 0) -> None: ...
    # Posterior members, plus:
    @property
    def postprocess_fn(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]: ...
    def init_point(self, seed: int = 0) -> dict[str, Any]: ...
    def site_shapes(self) -> dict[str, tuple[int, ...]]: ...
```

- **Construction.** One seeded trace (`handlers.trace(handlers.seed(model, seed))`) gives the latent
  sample sites: `type == "sample"`, not observed, not `infer["is_auxiliary"]`. Their order is
  `prior_names()`, their value shapes `site_shapes()`. Deterministic and observed sites are not
  sites.
- **Density.** `potential_fn`, `postprocess_fn` and the `init_point` default come from
  `numpyro.infer.util.initialize_model(PRNGKey(seed), model)`, built lazily on first access and
  cached, exactly as `BayesianModel` does. `log_density(z) = -potential_fn(dict(z))`.
- **Transforms.** `constrain` uses `numpyro.infer.util.constrain_fn(model, (), {}, z)` and
  `unconstrain` uses `unconstrain_fn`, both restricted to `prior_names()`. They run the model, so a
  support that depends on another site (`Uniform(0, mu)`) is handled. A batch axis is detected per
  call: if every leaf has `ndim == len(site_shapes()[name]) + 1`, the call is `jax.vmap`ped; if
  none has, it is unbatched; mixed is a `ValueError`. Missing site names raise `KeyError`, matching
  `BayesianModel`.
- **`site_priors()`.** Given `site_priors=`, the sequence is used as given. Each name must be in
  `prior_names()` (else `ValueError`). Default: auto-detect. A site gets
  `NumpyroPrior(name, dist)` when all of the following hold:
  - its value is scalar;
  - its distribution is the same in two traces with different seeds (`tree_flatten` leaves equal),
    so it depends on no other site;
  - `NumpyroPrior.has_icdf` is true.

  Plate members, centred children of hyper-parameters, and distributions without an icdf are left
  out, and so fall back (below).

## `NumpyroPrior` (CX21)

`priors.py`: `@dataclass(frozen=True) class NumpyroPrior: name: str; dist: Any`. It satisfies
`Prior`:

- `to_numpyro()` returns `dist`.
- `bounds()` reads `dist.support.lower_bound` / `upper_bound` when they exist; infinite bounds and
  missing attributes become `None`.
- `icdf(u)` is `np.asarray(dist.icdf(jnp.asarray(u)), float64)` after the `[0, 1]` check.
- `has_icdf` (property) probes `dist.icdf(0.5)` and catches `NotImplementedError` and
  `ImportError`. numpyro 0.16 raises `ImportError` for `Beta` and `Gamma` and
  `NotImplementedError` for `TruncatedNormal`. `icdf` raises `NotImplementedError` with the site
  name when `has_icdf` is false.

## Design stages: the fallback

`wf.lhs(post, m, seed=)` and `wf.prior_draws(post, m, seed=)`:

1. stratify (or draw i.i.d.) only the sites in `post.site_priors()`, in `prior_names()` order;
   column order and the `default_rng(seed)` stream are unchanged, so a `BayesianModel` gets
   bit-identical designs to today;
2. if any site in `prior_names()` is not covered, draw it with
   `numpyro.infer.Predictive(post.numpyro_model(), posterior_samples=<covered params>,
   batch_ndims=1, return_sites=<uncovered>)(PRNGKey(seed))`, so each fallback draw is conditional on
   that row's stratified values;
3. `post.unconstrain(params)` for all sites, as today.

Record `{"lhs_sites": [...], "predictive_sites": [...]}` in the stage's `StageRecord.settings`.
Document the fallback in both docstrings.

## Workflow signatures (CX35)

Rename the first parameter from `bm: Any` to `post: Posterior` in:

- `lhs`, `prior_draws`, `evaluate`;
- `Candidates.from_params`, `from_z`, `from_idata`;
- `OptimizeRun.__init__`, `optimize`;
- `MCMCRun.candidates`, `run_mcmc`.

Also update `warmup_until`'s docstring example. `OptimizeRun.bm` becomes `OptimizeRun.post`, with no
alias: the attribute was never documented. Update every docstring that says `bm` / `BayesianModel`
where any `Posterior` works. `posterior_runs` / `run_posterior` keep `BayesianModel`, because they
read `compiled`, `outputs` and `targets`. `BayesianModel.find_map` / `sample` are unchanged.

## `BayesianModel` changes

- **`log_likelihood(draws: Mapping[str, Any]) -> Array`** (public, traceable). It does
  `merge_params(draws)` → `_run` → `outputs.evaluate` (if any) → `targets.log_likelihood` → the
  solver-failure and NaN penalty (`-1e30`). This is exactly the body `numpyro_model` has today.
  `numpyro_model` becomes: sample each site, then
  `numpyro.factor("likelihood", self.log_likelihood(draws))`. `draws` must hold every name in
  `prior_names()` (the hierarchical likelihood scales too). A joint model samples them itself:
  `for p in bm.site_priors(): ...`. Merging an already-merged dict is idempotent.
- **`postprocess_fn`** (public property, CX22), beside `potential_fn`, from the same cached
  `initialize_model` call. `init_point` is already public. `_ensure_potential` stays private, and
  tests stop calling it (`tests/test_epi_calibration.py`).

## Tests (`tests/test_calib_posterior.py`, new)

1. **Equivalence (§4).** Build an SIR `BayesianModel` whose priors all have a numpyro icdf
   (`Uniform`, `Normal`, `LogNormal`) and `post = NumpyroPosterior(bm.numpyro_model())`. Run
   under `jax.experimental.enable_x64()`. At 32 random unconstrained points, `log_density`,
   `potential_fn`, `constrain` and `unconstrain` (batched and unbatched) agree to `atol=1e-10`.
   `prior_names()` are equal. `wf.lhs(·, 16, seed=3)` gives equal `z` and `params` to 1e-10.
   `wf.optimize(·, start, method=wf.Optax(optax.adam(0.05)), max_steps=40, seed=0)` gives equal
   `best_params` and `loss_trace`. With `site_priors=bm.site_priors()` the same holds for a model
   with `Gamma` / `TruncatedNormal` priors.
2. **Plate and non-centred latent.** Use `mu ~ Normal`, `tau ~ HalfNormal`, `z ~ Normal(0, 1)`
   under `plate("k", 5)`, `theta = mu + tau * z` (deterministic), and an observed
   `Normal(theta, 1)` with `obs=`. Check:
   - `prior_names() == ("mu", "tau", "z")` and `site_shapes()["z"] == (5,)`;
   - auto `site_priors()` names are `("mu", "tau")`;
   - `wf.lhs` stratifies `mu` and `tau` (one point per stratum) and gives `z` shape `(m, 5)`, drawn
     by Predictive (a different `seed` changes `z`);
   - `evaluate`, multi-start `optimize`, `Candidates.init_params` → `MCMC(NUTS(post.numpyro_model()))`
     and `MCMCRun.candidates(post)` all run;
   - a centred variant (`theta ~ Normal(mu, tau)` in the plate) leaves `theta` out of auto
     `site_priors()`.
3. **`NumpyroPrior`**: `bounds`, `icdf` against scipy for `Normal` / `Uniform`, `has_icdf` false for
   `Beta`, and `isinstance(NumpyroPrior(...), Prior)`.
4. **`BayesianModel.log_likelihood`**: equals the `"likelihood"` factor in a trace of
   `numpyro_model`, returns the `-1e30` penalty for a failed solve, and works inside a hand-written
   `numpyro.factor` model.
5. `isinstance(bm, Posterior)` and `isinstance(NumpyroPosterior(m), Posterior)`.

## Gate notebook `28-calibration-custom-posterior.ipynb`

An acceptance page in the style of notebooks 22 and 24: a *What to check* box per section, a titled
Plotly figure or a displayed object, and no asserts. The asserts live in
`tests/test_calib_posterior.py::test_gate_*`, on the same model at a smaller size.

- **Model.** SIR with infection rate `infection` and recovery rate `recovery`. The main data are
  noisy prevalence on the early growth phase only (days 0–20), which pins `infection - recovery`
  and leaves a ridge. Side data come from K = 6 "historical outbreaks", each a handful of observed
  infectious durations `~ Exponential(recovery_j)`, with
  `recovery_j = exp(mu + tau * z_j)` (non-centred, `plate`). The main epidemic is one more
  exchangeable outbreak: `recovery = exp(mu + tau * z_main)`. The joint model samples `infection`,
  `mu`, `tau`, `z`, `z_main`, then scores the main data with
  `numpyro.factor("main", bm.log_likelihood({"infection": ..., "recovery": ...}))`.
- **Sections.**
  1. The ridge: main-data log-likelihood heatmap over (infection, recovery), with the truth marked.
  2. The same model two ways: `bm` against `NumpyroPosterior(bm.numpyro_model())` `log_density` at
     random points, plotted on y = x, with the max difference displayed.
  3. The joint posterior object: displayed `prior_names()`, `site_shapes()`, and which sites LHS
     stratifies and which are Predictive.
  4. LHS → `evaluate` → multi-start `optimize`, for main-only (`bm`) and joint (`post`): endpoints
     in (infection, recovery). Main-only endpoints spread along the ridge; joint endpoints cluster.
  5. Staged NUTS (`warmup_until` + `sample_until`, seeded from `Candidates.init_params`) on both
     posteriors. Overlay the (infection, recovery) draws: joint is tight and covers the truth;
     main-only lies along the ridge. A second figure shows the partial pooling: per-outbreak recovery
     posteriors against the main epidemic's.

## Deviations from §4

| §4 says | This step does | Why |
|---|---|---|
| `prior_names`, `site_priors` are properties | methods | `BayesianModel` has them as methods, and §4 requires it to satisfy `Posterior` unchanged |
| `site_priors -> Mapping[str, Prior \| None]`; `NumpyroPosterior(site_priors: Mapping[str, Prior])` | `Sequence[Prior]` of the stratifiable sites; uncovered sites fall back | `BayesianModel.site_priors()` returns `tuple[Prior, ...]` and `design.py` reads it that way; a `Prior` already carries its name |
| `Prior.from_numpyro(name, dist)` (CX21) | `NumpyroPrior(name, dist)` | `Prior` is a `runtime_checkable` Protocol, so a static method on it becomes a protocol member. Every structural prior would then fail `mypy --strict` and `isinstance(x, Prior)` (`likelihoods.py` uses `isinstance`). Checked with mypy on 2026-10-19 |
| `constrain`/`unconstrain` via `transform_fn` | `constrain_fn` / `unconstrain_fn` | These handle a support that depends on another site; `transform_fn` needs the transforms precomputed |
| Notebook `NN-…` | `28-…` | 25–27 are reserved by roadmap steps 28, 31 and 44 |
