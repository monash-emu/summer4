# Multi-model calibration helpers

Roadmap step 50, branch `feat/multi-model`, package `CP9`, closes `CX46`, `CX47` and `CX48`.

This is the card and the implementation spec. The design is settled here: the implementer
invents no names, signatures, messages or test claims. A deviation is recorded at the end of this
file, under *Deviations*, with the reason.

## Why

Step 46 made the workflow stages take any `Posterior`. Since then, a joint calibration of several
models is a hand-written numpyro function that scores each compiled model with
`numpyro.factor(..., bm.log_likelihood(params))`. Examples: `28-calibration-custom-posterior`,
and tb-nh-joint-calibration's `joint.py`, which couples 60 natural-history cohorts to the VNM
transmission model.

Writing those models showed three sharp edges, recorded as composability findings:

- **`CX46`: a missing or misspelled name falls back silently.**
  `BayesianModel.log_likelihood(draws)` merges `draws` over `fixed_params`. A prior site left out
  of `draws`, or a name spelled wrong, quietly uses the fixed value (or is ignored), with no
  error. With several models whose parameters share local names (`recovery` in each), the
  mapping from joint sites to each model's names is hand-written, and this is where it goes
  wrong. tb-nh-joint-calibration wrote its own guard (`vnm_target_factor`: "every site must be
  named explicitly").
- **`CX47`: no namespacing.** Numpyro site names are global; a model's parameter names are
  local. Sampling a `BayesianModel`'s priors under a prefix, holding some at given values (a
  hierarchical rate, a shared parameter), and getting back the local-name dict is boilerplate
  every joint model rewrites (tb-nh's `sample_vnm_parameters`, `vnm_parameters_or_fixed`).
- **`CX48`: no replay after a joint fit.** `BayesianModel.posterior_runs` reads posterior sites
  by the model's own names and rejects any non-scalar site. A joint posterior names sites with
  prefixes and holds plate sites (the per-member offsets `z`), so it cannot be replayed through
  any of its member models without hand-building arrays.

Background: the funnel and centring note is `futureplans/hierarchical-centring-diagnostics.md`.
Funnel diagnostics are **not** in this step.

## Terms

- **Local name.** A parameter name a `BayesianModel` knows: a prior site
  (`bm.prior_names()`) or a key of `bm.fixed_params`.
- **Site name.** A numpyro site name in the joint model: the local name with a prefix
  (`"a.infection"`).
- **Given value.** A value the joint model supplies for a local name instead of sampling it: a
  hierarchical member value, a parameter shared with another model, or a value held fixed.

## API (all on `BayesianModel`, `src/summer4/epi/calibration/model.py`)

No new modules and no new exports. Every method below is public and documented, with Google
style docstrings and full annotations (mypy `--strict`).

### 1. `check_params` (`CX46`; the validation policy as its own piece, `P2`)

```python
def check_params(self, params: Mapping[str, Any]) -> None:
```

It reads names only (never values), so it is safe under `jax.jit` and on batched arrays. It
raises `ValueError` when either of these holds; when both hold, one error lists both, missing
first:

- **Missing:** prior sites absent from `params`, sorted.
  Message: `"missing prior sites: ['recovery']"`.
- **Unknown:** names in `params` that are neither a prior site nor a key of `fixed_params`,
  sorted. Message: `"unknown parameter names: ['recovry'] (not a prior site or a fixed_params
  name of this model)"`.

The two parts are joined with `"; "`. If any unknown name contains a `"."`, the message ends with
`". Prefixed joint-model site names? Map them with local_params(draws, prefix)."`.

Fixed names are the keys of `_params_as_dict(self.fixed_params)`: top-level names only, as
`merge_params` uses them.

### 2. `log_likelihood(..., strict=)` (`CX46`)

```python
def log_likelihood(self, draws: Mapping[str, Any], *, strict: bool = False) -> Any:
```

With `strict=True` it calls `self.check_params(draws)` first, then behaves exactly as today.
`strict=False` is unchanged: the default stays lenient, so existing callers do not break. The
docstring says when to use strict ("whenever the dict is built by hand, especially in a joint
model"), and that `factor` always uses it.

### 3. `sample_params` (`CX47`)

```python
def sample_params(
    self,
    prefix: str = "",
    *,
    given: Mapping[str, Any] | None = None,
    record: bool = True,
) -> dict[str, Any]:
```

It must be called inside a numpyro model.

1. **Validate `given`.** Every key must be a local name, or `ValueError`:
   `"given names are not parameters of this model: ['x'] (not a prior site or a fixed_params
   name)"`, with the names sorted.
2. **Sample or take each prior site**, in `site_priors()` order: the value is `given[name]` if
   present, else `numpyro.sample(prefix + name, prior.to_numpyro())`.
3. **Record given values.** If `record`, for every given name (in `given` order) call
   `numpyro.deterministic(prefix + name, value)`, so posterior samples carry it and
   `local_params` can rebuild the full dict.
4. **Return the local-name dict:** prior sites in `site_priors()` order, then any given names
   that are not prior sites (fixed-name overrides), in `given` order.

Docstring notes:

- With `prefix=""` and `record=True`, a given value whose name is already a site in the joint
  model gives numpyro's duplicate-site error. Pass `record=False`, or use a prefix.
- Plate members and centred hierarchical children are not stratified by `wf.lhs`; they are filled
  by `Predictive` (cross-reference `NumpyroPosterior`).
- It includes a three-line example of a two-model joint model with a hierarchical shared rate.

### 4. `factor` (`CX47`; a convenience built from the pieces, `P4`)

```python
def factor(self, name: str, params: Mapping[str, Any]) -> None:
```

The body is exactly `numpyro.factor(name, self.log_likelihood(params, strict=True))`, and the
docstring shows that line ("Exactly::"). The equivalence is tested (§Tests, T7).

### 5. `local_params` (`CX48`)

```python
def local_params(self, draws: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
```

This is the inverse of `sample_params`' naming, and it works on any leaf shape (batched draws
included).

- Every prior site must be present as `draws[prefix + name]`, or `ValueError`:
  `"draws lack sites for this model's priors: ['a.recovery']. Sample them with
  sample_params(prefix), or record given values (record=True)."`, with the site names sorted.
- Any fixed name present as `draws[prefix + name]` is also returned (an override recorded by
  `sample_params`).
- Order: prior sites in `site_priors()` order, then fixed names in `fixed_params` order.
- Other keys are ignored: other models' sites, hyperparameters, plate offsets.

### 6. `posterior_runs(..., params_fn=)` (`CX48`)

```python
def posterior_runs(
    self, draws: Any, *, n: int | None = 1000, burn_in: int = 0, seed: int = 0,
    scenarios: Mapping[str, Any] | None = None, batch_size: int = 64,
    params_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, Any]] | None = None,
) -> Any:
```

`run_posterior` in `posterior_runs.py` gets the same keyword.

- **`params_fn=None`:** behaviour unchanged, byte for byte (the existing tests are the check).
- **With `params_fn`:**
  1. Flatten every site to a leading draw axis: `(chain, draw, ...)` becomes `(n_all, ...)`
     after `burn_in` for `InferenceData`, and a mapping is taken as already `(n_all, ...)`.
     There is **no scalar check** on input sites, so plate sites are allowed.
  2. Subsample with the same `n` and `seed` rule as today (`_subsample_mapping`).
  3. Call `local = params_fn(flat)` once, batched: `params_fn` receives arrays with the leading
     draw axis and must return arrays with the same leading axis. `bm.local_params` already
     satisfies this.
  4. `self.check_params(local)`: strict, so a mis-mapped replay fails.
  5. Each returned value must have shape `(n,)`, one scalar per draw, or `ValueError`:
     `"params_fn must return one scalar per draw; {name!r} has shape {shape}."`
     (posterior_runs replays scalar parameters only, as today).
  6. Continue exactly as today with `local` as the draw dict. `PosteriorRuns.draws` holds the
     local dict.
- The docstring shows the typical call:
  `bm_a.posterior_runs(run.idata, params_fn=lambda d: bm_a.local_params(d, "a."))`.

Implementation hint, not binding: split `_draws_from_idata` into a flatten step and a
scalar-check step, so the `params_fn` path reuses the flatten step.

## Pitfalls handled by documentation only

All of these go in the gate notebook's prose and the docstrings above. No code.

- `wf.lhs` stratifies only parentless scalar sites; plate offsets come from `Predictive`.
- Within one compiled model a name has one value: two values of "the same" parameter need two
  names or a stratified parameter, not prefixes.
- Two `BayesianModel`s sampled with the same prefix collide in numpyro's duplicate-site error.
  Test T6 pins that, so the failure is loud.
- Write hierarchies non-centred by default (`z ~ Normal(0, 1)`, value `= exp(mu + tau z)`). The
  reason is in `futureplans/hierarchical-centring-diagnostics.md`.

## Out of scope

Recorded, not built:

- Declarative hierarchical priors and stacked (vmapped) members: one
  `futureplans/multi-model-stacked-and-declarative.md` note, written in this step's card commit.
- Funnel and centring diagnostics: `futureplans/hierarchical-centring-diagnostics.md`.
- A guide page. The gate notebook is the documentation for this step. A short guide section is a
  follow-up once the API has been used in tbcore.

## Gate notebook: `examples/notebooks/38-calibration-multi-model.ipynb`

An acceptance page in the style of notebooks 22, 24 and 28: a *What to check* box per section,
pandas-Plotly figures with titles and axis labels, and no asserts. Settings and builders live in
`tests/gate_multi_model.py`, which the notebook copies verbatim in its first code cell
("kept in sync with tests/gate_multi_model.py"), as notebook 28 does.

### Models and data (`tests/gate_multi_model.py`)

- **Settings.** `DATA_SEED = 11`; `TRUE_MU = log(0.3)`, `TRUE_TAU = 0.25`; true per-model
  recovery = `exp(TRUE_MU + TRUE_TAU * z_m)` with `z_m` from the data RNG for
  `m in ("a", "b", "cohort")`; `TRUE_INFECTION = {"a": 0.55, "b": 0.75}`.
- **City models A and B.** One `CompiledModel` from `build_city_model()`: a frequency-dependent
  SIR as in notebook 28, with `Param("infection")` and `Param("recovery")`. Two `BayesianModel`s
  share that compiled model:
  - populations 1e6 and 4e5, with 20 initial infectious each;
  - targets: infectious counts every 2 days over days 0–20, `NegativeBinomial(dispersion=20)`;
  - priors `Uniform("infection", 0.1, 1.5)` and `Uniform("recovery", 0.02, 1.0)`;
  - `fixed_params={"recovery": 0.25}`: a development default, there to make the fallback
    pitfall visible.
- **Cohort model.** A different compiled model from `build_cohort_model()`: states `I` and `R`
  only, one flow `I → R` at `Param("recovery")`, starting 300 infectious. Targets: `I` on days
  0–14, `Poisson()`. Prior `Uniform("recovery", 0.02, 1.0)`; `fixed_params={}`.
- **Joint model** `joint_model()`:
  - `mu ~ Normal(log(0.25), 1)`, `tau ~ HalfNormal(0.3)`;
  - `z_a`, `z_b`, `z_cohort ~ Normal(0, 1)` (non-centred);
  - `rec_m = exp(mu + tau z_m)`;
  - `p_m = bm_m.sample_params(f"{m}.", given={"recovery": rec_m})`, then `bm_m.factor(m, p_m)`.
- The run settings and seeds are constants in the module, as in `gate_custom_posterior.py`. The
  notebook must run in about 3 minutes or less on the mini.

### Sections

| § | Shows | What to check |
|---|---|---|
| 1 | A table of each model's `prior_names()`, and `NumpyroPosterior(joint_model).prior_names()` | All three models have a `recovery`; the joint sites are `a.infection`, `b.infection`, `mu`, `tau`, `z_*`, and no bare `recovery` |
| 2 | Pitfall: `bm_a.log_likelihood` with `"recovry"` (lenient) against `bm_a.log_likelihood({... "recovery": truth})`, plus the strict `ValueError` text, displayed as a DataFrame | The lenient misspelled call returns a finite number that differs from the correct one (it used the fixed 0.25); the strict call's message names `recovry` and `recovery` |
| 3 | `wf.lhs` → `wf.evaluate` → `wf.optimize` → `wf.warmup_until` → `wf.sample_until` on the joint posterior. The same pipeline on `bm_a` alone. One figure: box plots of recovery for A alone, and for A, B and cohort from the joint fit, with truth markers | City A alone is wide (along the ridge); joint A is narrow, with the truth inside its 10–90% interval |
| 4 | Replay: each model's `posterior_runs(run.idata, params_fn=lambda d: bm.local_params(d, prefix))`, with the median and 10–90% band against the observed data, one panel per model | Bands cover the data in all three panels; no hand-built arrays in the cell |

## Tests (`tests/test_calib_multi_model.py`, new)

Use tiny models: the cohort model alone, or an SIR with `solver="euler"` and a short horizon. The
gate tests use the shared settings module.

| ID | Asserts |
|---|---|
| T1 | `check_params` passes on exactly the prior names; passes with an extra fixed name; raises listing the missing names (sorted); raises listing the unknown names (sorted); one message holds both when both hold; the `local_params` hint appears only when an unknown name contains `"."` |
| T2 | `log_likelihood(d, strict=True)` equals `log_likelihood(d)` on a complete dict; strict raises on a misspelled key where lenient returns a finite value equal to the value at `fixed_params`' default |
| T3 | `check_params` and strict `log_likelihood` work under `jax.jit` (a jitted function that calls `log_likelihood(d, strict=True)` compiles and matches) |
| T4 | `sample_params("a.")` traced with `handlers.trace` + `seed`: sample sites are exactly `a.<prior>`; the returned keys are local names in `site_priors()` order |
| T5 | `sample_params("a.", given={"recovery": v})`: no `a.recovery` sample site; a deterministic `a.recovery` site equal to `v`; with `record=False`, no such site; a fixed-name override appears in the returned dict and as a deterministic; an unknown given name raises the spec message |
| T6 | Two models with the same prefix in one numpyro model raise numpyro's duplicate-site error (`pytest.raises(Exception, match="already")`; pin the actual substring numpyro uses) |
| T7 | Equivalence (`P4`): for a joint model written with `factor`, `NumpyroPosterior(...).log_density(z)` equals the same model written with `numpyro.factor(name, bm.log_likelihood(params, strict=True))` at 8 random unconstrained points (x64, `rtol=1e-10`) |
| T8 | `local_params` round trip: posterior-shaped draws (leading `(chain, draw)` axes) from a two-model joint `MCMC` (or `Predictive`) give the local dict with matching shapes; a missing site raises the spec message; another model's sites are ignored |
| T9 | `posterior_runs(params_fn=...)` on a joint `InferenceData` with a plate site gives the same `series` as `posterior_runs` on a hand-built local-name mapping of the same draws (same `n`, `seed`) |
| T10 | `params_fn` returning a mis-mapped dict raises `check_params`' error; one returning shape `(n, 2)` raises the "one scalar per draw" message |
| T11 | `params_fn=None` keeps today's scalar-site error for a plate site (regression) |
| T12 (gate, `test_gate_*`) | The notebook's claims on the shared settings: joint 10–90% interval of each model's recovery contains the truth; joint A's interval width is under 0.5× A-alone's; the §2 lenient value is finite and differs from the correct one |

## Composability survey

- The card commit adds:
  - `CX46` (calibration, P2, medium, `planned`), evidence
    `src/summer4/epi/calibration/model.py::params.update(dict(draws))`;
  - `CX47` (calibration, P2, P4, low, `planned`), evidence
    `src/summer4/epi/calibration/model.py::draws[prior.name] = numpyro.sample(prior.name, prior.to_numpyro())`;
  - `CX48` (calibration, P3, medium, `planned`), evidence
    `src/summer4/epi/calibration/posterior_runs.py::posterior_runs expects scalar parameter sites`;
  - package `CP9` "Multi-model calibration helpers (roadmap step 50)", closing CX46–CX48.
- The implementation commit sets all three to `done`, with a one-line "Done in step 50: …" fix
  description, then runs `pixi run composability-write`.

## Files

- `src/summer4/epi/calibration/model.py`
- `src/summer4/epi/calibration/posterior_runs.py`
- `tests/test_calib_multi_model.py` (new) and `tests/gate_multi_model.py` (new)
- `examples/notebooks/38-calibration-multi-model.ipynb` (new)
- `docs/evaluation/composability.md`
- `CHANGELOG.md` (`[Unreleased]` › Added)
- this plan (*Deviations*)

## Checks

- **Worker:** `pixi run pytest tests/test_calib_multi_model.py -q`,
  `pixi run pytest tests/test_epi_calibration.py tests/test_calib_posterior.py -q`, `pixi run
  lint`, `pixi run format-check`.
- **Orchestrator:** the roadmap's standard exit checks, with `pixi run test` and docs through
  devq on the mini.

## Deviations

- **T6** matches numpyro's actual duplicate-site message, `unique names` (`all sites must have
  unique names but got ... duplicated`), not the placeholder `already`.
- **T8, T9** build their joint draws with `Predictive` (prior draws, reshaped to
  `(chain, draw, ...)`), which the spec allowed for T8; the factor sites `Predictive` returns
  (empty arrays) are dropped, as MCMC samples do not carry them. T9 adds a hand-made plate site.
- **T12**: the joint site order is `mu, tau, z_a, z_b, z_cohort, a.infection, b.infection`
  (the `z_m` are sampled before the models), and the gate test also checks the notebook copies the
  settings module verbatim. The joint NUTS warmup in the gate run reports a
  `max_warmup` RuntimeWarning (divergences) yet the gate assertions pass.
- `params_fn` outputs are validated for names (`check_params`) before shapes, via a wrapper in
  `run_posterior`, matching spec order 4 then 5.
- **Gate solver and MCMC start (orchestrator fix).** The worker's gate ran the models with
  fixed-step Euler (`dt=0.25`) and started NUTS from the LHS design, discarding the optimiser's
  ends. Euler empties the cohort in one step at recovery 4 (failure penalty `-1e30`) and goes
  unstable beyond 8. A given hierarchical value bypasses the model's `Uniform(0.02, 1)` prior,
  so chains started near `exp(2)` stayed behind that cliff: the cohort recovery posterior had
  median 2.0 (truth 0.41), and warmup hit `max_warmup=4000` on divergences while the interval
  assertions still passed. The gate now uses the default adaptive solver, starts NUTS from
  `opt.candidates`, and asserts `warm.converged`. `sample_params`' docstring says a given value
  replaces the prior, support included. Centring was checked and was not the cause (1-8
  divergences in 800 draws under all non-centred, cohort-centred or all-centred forms).
- **Gate hierarchy centring (orchestrator fix).** With the adaptive solver the joint posterior was
  right (cohort recovery median 0.396, truth 0.407; R-hat 1.004), but 1.2-2% of warmup
  transitions still diverged, so warmup failed at `max_warmup`. The cohort's data pin its
  recovery (its log-sd is about 0.4 x mean tau, against about 1 for the cities), the inverted
  funnel case for a non-centred member. The gate centres the cohort member
  (`log_recovery_cohort ~ Normal(mu, tau)`) and keeps the cities non-centred. Warmup then passed
  after 120 steps (37 s, against failing at 3840 steps in 380 s), with 1/400 sampling
  divergences. The plan's §Models wording "non-centred" for all three is superseded.

