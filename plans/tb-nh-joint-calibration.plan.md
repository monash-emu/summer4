# Joint calibration of TB natural history and a transmission model

Calibrate the natural history of untreated pulmonary TB **jointly** with a compartmental
transmission model, so that the two share one set of natural-history parameters and one
natural-history *structure*. The historical-cohort analysis is modelled on Ragonnet et al. 2021.
The structure is tbcore's subclinical spectrum: active TB split 2 × 2 into subclinical or clinical,
and low or high infectiousness. The transmission model is tb_macro (Vietnam, "VNM"), through its
tbcore recipe.

This plan lives in `monash-emu/tb-nh-joint-calibration`, which coordinates the work. The work
itself spans three repositories:

| Repo | Local path | Role here |
|---|---|---|
| tb-nh-joint-calibration | `~/dev/EMU/tb-nh-joint-calibration` | This plan; the runbook `docs/roadmap.md` (Q2); the joint VNM analysis, NH5 and NH6 (Q1) |
| summer4 | `~/dev/EMU/s3a` | Lets workflow stages run on a hand-written joint numpyro model (step NH2) |
| tbcore | `~/dev/EMU/tb-core` (remote: the mini only, not yet on GitHub) | Cohort data, cohort model, hierarchical natural-history submodel, public VNM recipe (NH1, NH3, NH3b, NH4) |
| tb-macro-summer4 | `~/dev/EMU/tb-macro-summer4` | Not used by this plan (Q1). It may adopt the joint natural-history prior once it moves onto tbcore |

Stable IDs cited: composability `CX20`, `CX21`, `CX22`, and the new finding **`CX35`**, which NH2
adds (`docs/evaluation/composability.md`, package `CP3`). tbcore's `docs/open-questions.md`
question 4 (the value of `progression_prop_infectious`) is related but is not closed here.

---

## 1. Background

### 1.1 Ragonnet et al. 2021 (CID 73(1):e88–e96, doi:10.1093/cid/ciaa602)

- **Data.** Pulmonary TB cohorts from before chemotherapy, taken from the Tiemersma et al. 2011
  systematic review. There are 41 smear-positive (SP) and 19 usable smear-negative (SN) cohorts,
  all from Western Europe. Each cohort reports cumulative death proportions at 1–31 time points,
  over 1–31 years.
- **Code and data.** <https://github.com/romain-ragonnet/tb_natural_history>. Check the licence
  before vendoring anything (NH1).
- **Model.** A continuous-time Markov chain with states active, recovered and dead:
  - active → recovered at γ;
  - active → dead at μT + μ;
  - recovered → dead at μ.

  Background mortality μᵢ is fixed per cohort, from life tables minus population TB mortality.
- **Likelihood.** Deaths in each interval are binomial given the number at risk. The interval
  probabilities are analytic.
- **Hierarchy.** Each cohort has its own γᵢ and μTᵢ, drawn from zero-truncated normals with
  improper uniform hyperpriors. SP and SN are fitted separately, by HMC in Stan.
- **Estimates** (posterior median of the hyper-mean):
  - SP: μT 0.389/yr, γ 0.231/yr, untreated duration 1.57 yr.
  - SN: μT 0.025/yr, γ 0.130/yr, untreated duration 5.35 yr.
- **Limitations the authors state.**
  - Time zero is diagnosis, not onset.
  - Smear conversion is not modelled.
  - Selection bias is possible.
  - Loss to follow-up is not reported.
  - SN estimates are very sensitive to μ.

### 1.2 Richards et al. 2023 (Lancet Glob Health 11:e684–e692)

A three-state spectrum: minimal (not infectious), subclinical (infectious, no symptoms) and
clinical. It is fitted in a Bayesian framework to 22 untreated cohorts (5,942 people).
Code: <https://github.com/alexandrasrichards/natural_history_tb_disease>. It is the obvious second
data source for the spectrum transitions; it is optional here (NH7).

### 1.3 Why joint, and why now: the current VNM parameters are inconsistent

tbcore's `add_disease_course` (`src/tbcore/natural_history.py`) already has the spectrum:

```
           infectiousness_gain / loss
  subclin,low  <──────────────>  subclin,high      ──self_recovery──▶ recovered
      ▲ │                           ▲ │
      │ │ clinical_develop / regress│ │
      │ ▼                           │ ▼
  clin,low     <──────────────>  clin,high         ──tb_mortality[inf]──▶ death
```

Only subclinical people self-recover, and only clinical people die of TB. Yet VNM and KIR **fix**
`tb_mortality_rate_inf = 0.389` and `tb_mortality_rate_lowinf = 0.025`. Those are Ragonnet's SP
and SN hyper-means, estimated under a *three-state* model where a person dies at μT for as long as
they are active. In the spectrum, a clinical person:

- also regresses to subclinical (`clinical_regression_rate` = 1/yr);
- changes infectiousness at about 1/yr (VNM `infectiousness_loss_rate` = 1);
- recovers only by way of subclinical.

So the survival curve the spectrum model implies for a diagnosed SP cohort is not the one
Ragonnet fitted. The natural-history numbers are reused, but the structure that gave them meaning
is not. `self_recovery_rate = 0.4` is likewise fixed, not estimated.

Joint calibration fixes this by fitting the cohort survival curves *through the spectrum
structure*. The cohorts alone cannot identify every spectrum rate (§3.5), so the transmission
model's targets are what pin down the rest: the high-infectious share of prevalent cases, and
prevalence against notifications.

---

## 2. Design: one structure, two models

### 2.1 The cohort model is built by the same function as the transmission model

- **Cohort model.** A small `TBModel` with:
  - one age band (`age_breaks=(0,)`);
  - no demography, no transmission and no care;
  - `natural_history.add_disease_course(model)`, unchanged;
  - one background-death transition from every alive state to `mtb_naive`, at
    `Param("cohort_background_mortality")`.
- **Death sink.** With no infection, no births and no flows out of `mtb_naive`, `mtb_naive` is the
  death sink. Alive = active + recovered, and cumulative deaths = N₀ − alive.
- **Initial state.** A cohort starts diagnosed, so clinical:
  - SP cohorts start in `active & clin & high`;
  - SN cohorts start in `active & clin & low`.

  Smear conversion after diagnosis is then *structural* (`infectiousness_loss` and `_gain`); it
  was not in Ragonnet's model.
- **Evaluation.** Run with `CompiledModel.run`, `jax.vmap`ped over cohorts. Each cohort has its own
  background mortality, its own frailty-scaled rates (§3.2) and its own save times. Pad the save
  times to a common grid with a mask; cohorts have 1–31 observation times.
- **Consequence.** Any later change to `add_disease_course` changes the cohort likelihood
  automatically. Examples: a minimal-disease state, relapse from recovered, or subclinical
  detection.

**Fast path (NH-opt, only if profiling asks for it).** Every cohort flow is linear and does not
depend on time, so the cohort solution is `expm(Q t) · x₀`, which is exact. Extracting Q from a
`CompiledModel` is a domain-neutral summer4 capability. Write it up as a `futureplans/` note in
NH5 if the cohort part is more than about 10% of a joint gradient evaluation; otherwise do nothing.

### 2.2 Where the code goes

- **summer4:** only domain-neutral calibration plumbing (NH2). Nothing about cohorts or TB, per
  the epi-layer rule in `AGENTS.md`.
- **tbcore:**
  - the cohort data loader;
  - the cohort model builder;
  - the cohort log-likelihood;
  - the hierarchical natural-history submodel, as a numpyro function reusable by every country
    recipe;
  - the VNM recipe as public API, `tbcore.recipes.vnm` (NH3b), so that code outside tbcore's
    tests can build and score the VNM model.
- **tb-nh-joint-calibration (this repo, Q1):** the joint VNM fit and its analysis (NH5, NH6).
  It composes `hierarchical_nh` with `tbcore.recipes.vnm` and adds nothing reusable: anything
  another country recipe would need goes into tbcore instead. Environment: §2.3.

### 2.3 This repo's environment (Q1)

NH5 makes this repo a Python project. Its first commit adds:

- `pixi.toml`, matching tbcore's: `osx-arm64` and `linux-64`, Python 3.13, `jax == jaxlib ==
  0.7.2` (tbcore's pin, for parallel chains), and the same summer4 git `rev` tbcore pins
  (bumped by NH2's handoff).
- **tbcore as a path dependency**, `tbcore = { path = "../tb-core", editable = true }`. tbcore
  is not on GitHub, so a git URL would only resolve on machines with SSH access to the mini; the
  path is the same on the laptop and the mini (`~/dev/EMU/tb-core`). A path is not a pin, so the
  pin is a commit SHA in `tbcore.lock-rev` at the repo root, and a test fails if tbcore's
  `HEAD` is not that commit (or a descendant, for local development; the mini fits require an
  exact match). When tbcore moves to GitHub, replace both with a git `rev`.
- A package, `src/tb_nh_joint/`, holding the joint model (`joint.py`) and the run scripts the
  mini executes; tests in `tests/`; notebooks in `notebooks/`.
- Tasks: `test`, `format-check` (black, as tbcore), and `notebook-check` (each gate notebook
  run headless, outputs not committed).

---

## 3. Statistical model

### 3.1 Parameters

The shared natural-history parameters, φ_nh, are the seven rates `add_disease_course` reads:

| Parameter | VNM now | Role |
|---|---|---|
| `tb_mortality_rate_inf` | 0.389 fixed | clinical, high-infectiousness TB death |
| `tb_mortality_rate_lowinf` | 0.025 fixed | clinical, low-infectiousness TB death |
| `self_recovery_rate` | 0.4 fixed | subclinical → recovered |
| `clinical_regression_rate` | 1 fixed | clinical → subclinical |
| `clinical_progression_rate` | U(0.5, 5) | subclinical → clinical |
| `infectiousness_gain_rate` | U(0.1, 1) | low → high |
| `infectiousness_loss_rate` | 1 fixed | high → low |

VNM's own parameters θ_tb (transmission, detection, latency and so on) keep their current priors
(`tests/fixtures/priors/vnm_priors.yaml` in tbcore).

### 3.2 Cohort heterogeneity (non-centred)

Each historical cohort i has two log-frailties: one on TB mortality and one on resolution.

```
z_mort,i , z_rec,i ~ Normal(0, 1)
mortality rates in cohort i  = population rate × exp(σ_mort · z_mort,i)   (both inf and lowinf)
self_recovery & regression i = population rate × exp(σ_rec  · z_rec,i)
σ_mort, σ_rec ~ HalfNormal(0.5)
```

The other spectrum rates are shared by every cohort. This gives 2 × 60 latent dimensions, against
Ragonnet's 2 per cohort, and it must be non-centred: a centred version funnels under NUTS when σ
is small. Sensitivity analysis: separate σ for SP and SN cohorts.

### 3.3 The target setting is one more exchangeable cohort

The transmission model must **not** use the population rates directly. That is what reusing the
paper's hyper-mean amounts to, and it discards the between-cohort spread (SP μTᵢ ranges from 0.15
to 0.81/yr). Instead:

```
z_mort,VNM , z_rec,VNM ~ Normal(0, 1)
rates fed to the VNM model = population rates × exp(σ · z_VNM)
```

- The cohorts inform VNM through the population rates and σ.
- The VNM targets inform `z_VNM` directly, and the population rates only through it.
- σ is the explicit, criticisable statement of how far Western Europe in 1900–1950 transfers to
  Vietnam today.
- An extension worth a sensitivity run: a covariate on the population mean (era, setting).

### 3.4 Priors and likelihood

- **Population rates.** Log-normal priors. Centre them on Ragonnet's estimates where the mapping is
  direct (the TB mortality rates), and on the current VNM `BASE_PARAMS` otherwise. Use a log-sd of
  1 unless a source justifies tighter. If NH7 lands, Richards et al. replace the spectrum priors.
  Record every prior and its source in tbcore `docs/parameters.md`.
- **Cohort likelihood.** Rebuild interval deaths dᵢⱼ and the number at risk nᵢ,ⱼ₋₁ from the
  published cumulative proportions and cohort sizes, then
  `dᵢⱼ ~ Binomial(nᵢ,ⱼ₋₁, 1 − S(tⱼ)/S(tⱼ₋₁))`, where S is the alive fraction. Sensitivity:
  `BetaBinomial`, for extra variation from selection and loss to follow-up.
- **Background mortality.** μᵢ is taken from Ragonnet's data, fixed. Sensitivity: a log-normal
  around it.
- **VNM likelihood.** Unchanged: tbcore `calibration.bayesian_model`, i.e. the seven
  log-normal, logit-normal and ratio terms plus the mixing distance.

Joint log density:

```
log p = log p(φ_nh, σ) + Σ_i log p(z_i) + log p(z_VNM) + log p(θ_tb)
      + Σ_i Σ_j log Bin(d_ij | …φ_nh, z_i, μ_i)            # cohorts
      + log L_VNM(θ_tb, φ_nh ⊙ exp(σ z_VNM))                # transmission targets
```

### 3.5 Identifiability: what each data set pins down

The cohorts give two survival curves (SP and SN starts), about 60 replicates each. They identify
roughly an effective death hazard and an effective resolution hazard per starting state. That is
fewer than seven spectrum rates, so expected ridges include:

- `clinical_regression_rate` × `self_recovery_rate`, which act only through their series
  combination;
- `infectiousness_loss_rate` against the two mortality rates.

The VNM targets add the high-infectious share of prevalence (79/322 in 2017), and prevalence
against notifications and deaths. Those inform the share of time spent in each spectrum state.
NH4 must *show* the ridges from the cohorts alone (prior against posterior overlays, pair plots).
NH5 must show the joint fit removing or narrowing them. Per the gate-model rule, this ridge is
the structure that makes the joint workflow worth demonstrating.

Do not model: time before diagnosis (left truncation), or disease outside the lungs and HIV. Both
are listed as limitations in the NH5 notebook.

### 3.6 Inference

- **Sampler.** NUTS on the whole joint density, with a dense mass matrix tried against a diagonal
  one, using summer4's staged workflow (`warmup_until`, `sample_until`, `StopRule`) after NH2. With
  about 120 dimensions, ensemble and gradient-free samplers (AIES, DEMetropolisZ, ESS) are out.
  Use multi-start L-BFGS from an LHS over the non-hierarchical sites for initial points, as in
  tbcore's `fit.calibrate`.
- **If mixing is poor.** Try `numpyro.infer.HMCGibbs`: cheap conditional updates of the cohort
  latents, alternating with NUTS on the rest. Record the outcome either way.
- **Chunking.** Use large `sample_until` chunks (`futureplans/mcmc-chunk-recompile.md`). Run VNM
  fits on the Mac mini through `mini build` (laptop rules in `~/.devq/CLAUDE-devq.md`).

### 3.7 Model conflict and the cut posterior

Fit and compare three posteriors for φ_nh:

1. **NH-only:** cohorts only (NH4).
2. **Joint:** cohorts and VNM, full feedback (NH5).
3. **Cut:** the VNM targets do not feed back into φ_nh or σ. For each NH-only posterior draw of
   (φ_nh, σ), sample `z_VNM` and θ_tb conditionally (nested MCMC), or use the approximation
   below. See Plummer 2015 and Jacob et al. 2017 on modular Bayes.

Reading the comparison:

- **Joint and NH-only agree on the shared marginals.** Report joint.
- **They disagree.** Look first at the VNM side: fixed target sd's (0.1–0.2 on log or logit
  scales) can swamp 60 cohorts. Try fitted or hierarchical target sd's before calling it real
  conflict.
- **Conflict remains.** Report joint and cut side by side, plus a power-likelihood sweep
  `α ∈ {0, 0.25, 0.5, 1}` on log L_VNM.
- **Cohort model criticism.** Leave-one-cohort-out LOO (arviz, using cohort-level pointwise
  log-likelihoods) compares: shared against separate σ; binomial against beta-binomial; and the
  spectrum against Ragonnet's 3-state model.

**Two-stage approximation.** Fit NH-only, then summarise p(φ_nh, σ | cohorts) as a multivariate
normal on the log scale and use it as the prior on the VNM side. Integrate `z_VNM` there.

- This approximates the *joint* posterior, not the cut one.
- It is the cheap path when the VNM model is slow.
- Validate it against the full joint once (NH5) before relying on it.

---

## 4. summer4 change (NH2): workflow stages on any numpyro model

**New finding `CX35`** (calibration; P1, P2; medium): the workflow stages are bound to
`BayesianModel`. Today, `design.lhs`, `prior_draws`, `evaluate`, `Candidates`, `optimize` and
`MCMCRun.candidates` read these members of `bm`:

- `numpyro_model`, `potential_fn`, `log_density`;
- `constrain`, `unconstrain`;
- `prior_names`, `site_priors`, `merge_params`.

A hand-written model, such as one with hierarchical latents, two likelihoods and sites shared
between them, cannot use those stages. `MCMC(NUTS(model))` runs, but `Candidates.init_params`,
LHS, multi-start optimisation and replay do not.

Design:

```python
class Posterior(Protocol):                       # summer4.epi.calibration.workflow
    def numpyro_model(self) -> Callable[[], None]: ...
    @property
    def potential_fn(self) -> Callable[[dict], Array]: ...
    def log_density(self, params: dict) -> Array: ...
    def constrain(self, z: dict) -> dict: ...
    def unconstrain(self, params: dict) -> dict: ...
    @property
    def prior_names(self) -> tuple[str, ...]: ...
    @property
    def site_priors(self) -> Mapping[str, Prior | None]: ...   # None: no icdf, so lhs falls back

class NumpyroPosterior:                          # implements Posterior for any model fn
    def __init__(self, model: Callable[[], None], *, site_priors: Mapping[str, Prior] | None = None): ...
```

- `NumpyroPosterior` builds everything from `numpyro.infer.util` (`initialize_model`,
  `transform_fn`, `potential_energy`) and a seeded trace of `model` for the site names.
- `BayesianModel` satisfies `Posterior` unchanged.
- **Workflow signatures.** Replace `bm: BayesianModel` with `post: Posterior` wherever only those
  members are read. `posterior_runs`, which also needs `compiled`, `outputs` and `targets`, keeps
  `BayesianModel`.
- **`design.lhs`.** Covers only the sites whose prior has an `icdf`; the rest come from
  `numpyro.infer.Predictive` prior draws. Document this.
- **`BayesianModel.log_likelihood(params) -> Array`.** Public. It runs, scores and applies the
  solver-failure penalty that `numpyro_model` applies today, and `numpyro_model` calls it. This is
  the hook a joint model uses: `numpyro.factor("vnm", bm.log_likelihood(params))`.
- **Also closed:**
  - `CX21`: `Prior.from_numpyro(name, dist)`.
  - `CX22`: public `init_point` and `postprocess_fn`. `NumpyroPosterior` needs them anyway.
- **Not closed:** `CX20` (likelihood protocol). The joint model scores cohorts with numpyro
  distributions directly, which is the composable route.
- **Equivalence test.** `NumpyroPosterior(bm.numpyro_model())` and `bm` give the same
  `log_density`, `potential_fn` and `constrain`/`unconstrain` on random points (to 1e-10), and the
  same `wf.lhs` and `wf.optimize` results for the same seed.

---

## 5. Steps

Every step follows this repo's `docs/roadmap.md` § *How to run a step*: present the Summary
first; code is delegated to Cursor at the named tier; a checker one tier up runs the Sanity check.
Heavy fits and benchmarks go to the mini.

Q2 is answered: **position is recorded in `docs/roadmap.md`**, not here. From the commit that
answered Q1–Q4 on, this plan is immutable history; corrections go in the roadmap's *Corrections
to committed plans* table and override this file.

### NH1 — Reproduce Ragonnet et al. in numpyro (tbcore)

- **Summary.** Load the 60 cohorts from Ragonnet's repository and fit the paper's 3-state model in
  numpyro: analytic interval probabilities, truncated-normal hierarchy, SP and SN separately.
  Show the published hyper-means are reproduced. This is the parity gate (results match, not
  algorithms) and it gives the data loader every later step uses.
- **Repo / branch.** tbcore, `feat/nh-ragonnet-replication`, cut from and merging into `main`.
- **Worker.** sonnet implementer; opus checker.
- **Read first.** This plan §1.1, §3.4; Ragonnet's repo (README, data files, Stan model); tbcore
  `AGENTS.md`.
- **Do.**
  1. Check the licence of `romain-ragonnet/tb_natural_history`. If it is permissive, vendor the
     cohort data under `src/tbcore/data/cohorts/` with attribution. If not, write a fetch script
     and stop to ask before committing data.
  2. Add `tbcore.cohorts`:
     - `load_ragonnet_cohorts()`, returning one frame per cohort with times, n at risk, deaths,
       background μ and smear status;
     - `three_state_log_likelihood(...)`.
  3. Write a numpyro model matching the paper (improper uniform hyperpriors approximated by wide
     uniforms; note this) and fit it with NUTS, 4 chains.
  4. Add the notebook `notebooks/nh-01-ragonnet-replication.ipynb`. It overlays the published
     table against ours and shows the posterior-predictive survival curves for each cohort.
- **Files.** `src/tbcore/cohorts.py`, `src/tbcore/data/cohorts/*`, `tests/test_cohorts.py`, the
  notebook.
- **Sanity check.**
  - Cohort counts are 41 SP and 19 SN.
  - Each hyper-mean posterior median is within its published 95% CrI, and within 5% of the
    published median.
  - R-hat ≤ 1.01.
  - The test recomputes one cohort's interval probabilities by matrix exponential (scipy) and
    matches the analytic form to 1e-10.
- **User gate.** The notebook: the published-against-ours figure and its assertions.
- **Handoff.** Record any data-cleaning decisions: dropped cohorts, how deaths were rebuilt from
  proportions.

### NH2 — `Posterior` protocol and `NumpyroPosterior` (summer4)

- **Summary.** Make the workflow stages accept any `Posterior`, not only `BayesianModel`. Add
  `NumpyroPosterior` for a hand-written numpyro model, a public `BayesianModel.log_likelihood`,
  `Prior.from_numpyro`, and public `init_point` and `postprocess_fn`. Closes `CX35` (new), `CX21`
  and `CX22`.
- **Repo / branch.** summer4 (`s3a`), `feat/calib-posterior-protocol`, cut from and merging into
  `main`.
- **Worker.** opus for the protocol design and signatures (a one-page spike committed to this
  step's plan copy); sonnet implementer; opus checker.
- **Read first.** `AGENTS.md` § *Composability*; this plan §4;
  `docs/evaluation/composability.md` rows `CX20`–`CX22`;
  `src/summer4/epi/calibration/model.py`; `workflow/*.py`.
- **Do.**
  1. Add `CX35` to the composability survey, then do the design in §4.
  2. Update every workflow signature and docstring.
  3. Write the §4 equivalence test.
  4. Gate notebook `examples/notebooks/NN-calibration-custom-posterior.ipynb`. A small SIR with a
     hierarchical side data set: several "historical outbreaks" share a recovery rate with the
     main epidemic through a partially pooled latent. It must show a ridge the main data cannot
     resolve and the side data can, so that LHS, multi-start and staged NUTS visibly matter (the
     gate-model rule).
  5. Set `CX21`, `CX22` and `CX35` to `done`, then run `pixi run composability-write`.
- **Files.** `src/summer4/epi/calibration/{model.py,priors.py,workflow/*.py}`,
  `tests/test_epi_calibration*.py`, the notebook, `docs/evaluation/composability.md`.
- **Sanity check.**
  - The equivalence test passes.
  - `mypy --strict` is clean.
  - No workflow function still annotates `bm: BayesianModel` unless it reads
    `compiled`/`outputs`/`targets`.
  - `NumpyroPosterior` works on a model with a `plate` and a non-centred latent.
- **Exit checks.** The standard summer4 checks (`AGENTS.md` § *Required checks*) plus
  `pixi run -e docs docs-strict`.
- **Handoff.** Bump the summer4 version tbcore pins, or record the commit tbcore must pin.

### NH3 — Cohort model from the shared structure (tbcore)

- **Summary.** Build the cohort model with the same `add_disease_course` the country models use
  (§2.1), and score cohorts through it, vmapped over cohorts. Show that the current fixed VNM
  natural-history values imply survival curves that do not match the cohorts (§1.3).
- **Repo / branch.** tbcore, `feat/nh-cohort-model`, cut from `main` after NH1.
- **Worker.** sonnet implementer; opus checker.
- **Read first.** This plan §1.3, §2.1, §3.4; `src/tbcore/natural_history.py`, `model.py`,
  `outputs.py`.
- **Do.**
  1. Add `cohorts.build_cohort_model()` returning a `CompiledModel`, with one compile for all
     cohorts.
  2. Add `cohorts.spectrum_log_likelihood(params, cohorts, frailties)`: vmapped `run`, masked
     padded save times, binomial or beta-binomial selectable by passing the numpyro distribution
     class.
  3. Add the notebook `notebooks/nh-02-spectrum-cohort-model.ipynb`. It plots, against the cohort
     data:
     - the SP and SN cohort survival implied by VNM's current fixed values;
     - Ragonnet's 3-state fit.

     It asserts the VNM-implied 10-year SP case fatality differs from Ragonnet's by the amount the
     figure shows (whatever it is; state it).
- **Files.** `src/tbcore/cohorts.py`, `tests/test_cohorts.py`, the notebook.
- **Sanity check.**
  - Cohort survival from `run` matches `scipy.linalg.expm` of a hand-built 6-state generator, to
    1e-6, for random parameters.
  - With `clinical_regression_rate = 0`, gain = loss = 0 and SP start, survival equals the 3-state
    formula with γ = 0. This is the degenerate-case reduction.
  - A jaxpr-size check shows the vmapped likelihood does not grow with cohort count.
- **User gate.** The notebook.

### NH3b — Public VNM recipe (tbcore)

Added when Q1 placed NH5 outside tbcore. Independent of NH1–NH3; must land before NH5.

- **Summary.** Promote the VNM recipe from `tests/fixtures/vnm.py` and
  `tests/fixtures/vnm_calibration.py` to a public `tbcore.recipes.vnm`, so that this repo can
  build the VNM model and its `BayesianModel` from an installed tbcore. The fixtures re-export
  it, so tbcore's parity tests are unchanged.
- **Repo / branch.** tbcore, `feat/recipes-vnm`, cut from and merging into `main`.
- **Worker.** sonnet implementer; opus checker.
- **Read first.** tbcore `AGENTS.md`; `tests/fixtures/vnm.py`, `vnm_calibration.py`,
  `demography.py`; `tests/test_parity_vnm.py`.
- **Do.**
  1. Move the recipe to `src/tbcore/recipes/vnm.py` (and `recipes/__init__.py`). It currently
     reads files relative to `tests/`: move the VNM inputs it needs (UN population tables, WHO
     data, `vnm_base_params.yaml`, `vnm_priors.yaml`, `vnm_targets.yaml`) to
     `src/tbcore/data/recipes/vnm/` and load them with `importlib.resources`. Golden outputs stay
     under `tests/`.
  2. Expose `build_vnm_model()`, the base parameters, the priors, the targets and a
     `vnm_bayesian_model(...)` constructor, with the seven natural-history rates of §3.1
     overridable (NH5 removes them from the fixed parameters and the priors).
  3. Reduce `tests/fixtures/vnm.py` and `vnm_calibration.py` to re-exports.
- **Sanity check.**
  - `pixi run test` passes with the VNM parity tests untouched.
  - From a scratch directory outside the checkout, `python -c "from tbcore.recipes import vnm;
    vnm.build_vnm_model()"` works in tbcore's environment (proves no path reaches `tests/`).
- **Handoff.** The tbcore commit NH5 pins.

### NH4 — Hierarchical spectrum natural history, cohorts only (tbcore)

- **Summary.** Fit the §3 hierarchical model to the cohorts alone, through the spectrum structure,
  using NH2's `NumpyroPosterior` and the staged workflow. Show which spectrum rates the cohorts
  identify and which ridges remain: the case for the joint fit.
- **Repo / branch.** tbcore, `feat/nh-hierarchical`.
- **Worker.** opus implementer (model specification); opus checker.
- **Read first.** This plan §3; NH2's notebook; tbcore `calibration.py`, `fit/pipeline.py`.
- **Do.**
  1. Add `tbcore.natural_history_prior.hierarchical_nh(cohorts, priors)`. It is a numpyro
     sub-model that samples the population rates, σ and the cohort `z`, adds the cohort factor,
     samples `z_target`, and **returns** the target-setting rates for a caller to feed into a
     country model.
  2. Write the priors to `tests/fixtures/priors/nh_priors.yaml`, with sources in `docs/parameters.md`.
  3. Fit with LHS → multi-start → staged NUTS on the mini.
  4. Add the notebook `notebooks/nh-03-hierarchical-spectrum.ipynb`: prior against posterior for
     every rate, pair plots showing the §3.5 ridges, posterior-predictive survival for each cohort,
     and implied SP/SN effective duration and case fatality against Ragonnet.
- **Sanity check.**
  - No divergences after warm-up, or fewer than 0.1% with the cause recorded.
  - R-hat ≤ 1.01.
  - Simulation-based calibration on 50 synthetic data sets from the prior: rank histograms are
    uniform by a χ² test at 1%. A reduced draw count is acceptable; record it.
- **User gate.** The notebook.

### NH5 — Joint calibration with VNM (this repo)

- **Summary.** Fit cohorts and VNM jointly: `hierarchical_nh` supplies the seven rates to the VNM
  recipe, whose likelihood is `bm.log_likelihood`. Compare NH-only, VNM-only and joint marginals,
  and validate the two-stage approximation.
- **Repo / branch.** tb-nh-joint-calibration (Q1), `feat/nh-joint-vnm`, cut from and merging
  into `main`. Needs NH2, NH3b and NH4 landed.
- **Worker.** sonnet implementer for step 0; opus implementer for the rest; opus checker.
- **Do.**
  0. Set up this repo's environment (§2.3): `pixi.toml`, the tbcore path dependency and
     `tbcore.lock-rev`, `src/tb_nh_joint/`, tests and tasks. Check that `pixi install` and
     `pixi run test` work on the laptop and on the mini (`mini build -C
     dev/EMU/tb-nh-joint-calibration -- pixi run test`).
  1. Write the joint numpyro model, in `src/tb_nh_joint/joint.py`, building `bm` from
     `tbcore.recipes.vnm`:

     ```python
     def joint():
         rates = hierarchical_nh(cohorts, nh_priors)
         theta_tb = {p.name: numpyro.sample(p.name, p.to_numpyro()) for p in vnm_priors_without_nh}
         numpyro.factor("vnm", bm.log_likelihood({**theta_tb, **rates}))
     post = wf.NumpyroPosterior(joint)
     ```

  2. Remove the seven rates from VNM's fixed parameters and its uniform priors.
  3. Run the staged NUTS on the mini.
  4. Profile one gradient evaluation and split it between the cohort and VNM parts. If the cohort
     part is above 10%, write the `futureplans/` note from §2.1.
  5. Fit the §3.7 two-stage approximation and compare it with the joint.
  6. Add the notebook `notebooks/nh-04-joint-vnm.ipynb`:
     - the three posteriors for each shared rate on one figure;
     - posterior-predictive VNM targets;
     - posterior-predictive cohort curves under the joint fit;
     - the change in the headline VNM outputs (incidence, TB deaths 2025) against the
       fixed-parameter calibration.
- **Sanity check.**
  - R-hat ≤ 1.01 on every site.
  - The joint marginals of the §3.5 ridge parameters are narrower than NH-only (assert it).
  - The two-stage and joint posteriors for the shared rates agree within a stated tolerance, or
    the disagreement is reported.
- **User gate.** The notebook.

### NH6 — Conflict, cut posterior and sensitivity (this repo)

- **Summary.** Cut posterior, power-likelihood sweep, and the sensitivity runs in §3.2, §3.4 and
  §3.7: separate σ, beta-binomial, gamma population priors, uncertain μᵢ, fitted target sd's. Plus
  leave-one-cohort-out LOO. Includes the Q4 sensitivity run: a fitted fraction of SP cohorts
  starting subclinical.
- **Repo / branch.** tb-nh-joint-calibration, `feat/nh-sensitivity`, cut from `main` after NH5.
- **Worker.** sonnet implementer; opus checker.
- **User gate.** `notebooks/nh-05-sensitivity.ipynb`: one panel per sensitivity for the shared rates and the
  headline VNM outputs, with assertions of whichever conclusions the prose draws.

### NH7 (optional) — Richards et al. 2023 as a second natural-history source

Blocked on open question Q3. Map their minimal/subclinical/clinical states onto tbcore's 2 × 2
spectrum, add their cohort likelihood as a further factor in `hierarchical_nh`, and refit NH4 and
NH5.

---

## 6. Open questions (answered 2026-10-19)

Each question lists its options; the decision is in bold.

- **Q1. Where does the joint VNM analysis live?** VNM has not moved onto tbcore yet (tbcore
  `docs/model.md` §1). The proposed default was tbcore, against `tests/fixtures/vnm.py`, moving
  to tb-macro-summer4 later. **Decision: in this repo, tb-nh-joint-calibration.** NH5 and NH6
  land here; the reusable parts (cohorts, `hierarchical_nh`) stay in tbcore. Consequences:
  this repo gets an environment (§2.3), and promoting the VNM recipe to `tbcore.recipes.vnm` is
  now required, as step NH3b.
- **Q2. Which runbook records position?** **Decision (the default): `docs/roadmap.md` in
  tb-nh-joint-calibration records position for every NH step, using summer4's step contract
  (Summary, Read first, Do, Exit checks, Handoff). NH2 is also added as a step in summer4's
  `docs/dev/roadmap.md`, because summer4's checks require it there.**
- **Q3. How should Richards' minimal state map onto tbcore?** Options:
  - (a) low-infectiousness is "minimal" (needs `rel_infectiousness_lowinf` near 0 for those
    people);
  - (b) add a third infectiousness level, minimal, to `add_disease_course`. This is a structural
    change for both country models.

  **Decision (the default): defer; decide after NH5.** NH7 stays blocked.
- **Q4. Cohort initial state.** All SP cohorts start clinical and high-infectiousness. Should a
  fitted fraction start subclinical, given diagnosis by screening or radiography in some cohorts?
  **Decision (the default): no; one sensitivity run in NH6.**

## 7. References

- Ragonnet R, Flegg JA, Brilleman SL, et al. Revisiting the natural history of pulmonary
  tuberculosis: a Bayesian estimation of natural recovery and mortality rates. *Clin Infect Dis*
  2021;73(1):e88–e96.
- Tiemersma EW, van der Werf MJ, Borgdorff MW, Williams BG, Nagelkerke NJD. Natural history of
  tuberculosis: duration and fatality of untreated pulmonary tuberculosis in HIV negative
  patients. *PLoS One* 2011;6(4):e17601.
- Richards AS, Sossen B, Emery JC, et al. Quantifying progression and regression across the
  spectrum of pulmonary tuberculosis: a data synthesis study. *Lancet Glob Health*
  2023;11:e684–e692.
- Plummer M. Cuts in Bayesian graphical models. *Stat Comput* 2015;25:37–43.
- Jacob PE, Murray LM, Holmes CC, Robert CP. Better together? Statistical learning in models made
  of modules. arXiv:1708.08719, 2017.
