---
name: tb-macro-summer4-port
description: Port the TB macroeconomics model (summer3wip, numpyro NUTS) to summer4 in a new pixi repo, monash-emu/tb-macro-summer4 — first on today's summer4 API at numerical parity, then upgraded package by package until no hand-written glue remains.
---

# tb_macro → summer4 (`monash-emu/tb-macro-summer4`)

## Context

`monash-emu/tb_macroeconomics` is a work-in-progress TB natural-history model on
**summer3wip** (`summer3` from `monash-emu/summer3wip`), with one notebook that
builds the model, runs it, makes a synthetic infection target and calibrates
`contact_rate` with numpyro NUTS.

This plan lands it in a **new repository**, `monash-emu/tb-macro-summer4`,
managed with pixi, depending on summer4 from GitHub. It is also kept in summer4
as `plans/tb-macro-summer4-port.plan.md` and copied into the new repo's `plans/`
in phase M0.

**Authoritative gap record:** summer4 `docs/evaluation/tb-ports.md`, rows
`TM1`–`TM9`. **Feature work:** `plans/tb-ports-feature-completeness.plan.md`.

Unlike Kiribati, **tb_macro is implementable on summer4 today.** Every `TM` row
has a route; the non-`full` ones need hand-written glue (a `derived_fn` force of
infection, a hand-built initial state, a hand-written numpyro model). So this
port runs in two stages:

1. **M0–M2:** port on the current summer4 API with the glue, at numerical parity.
2. **M3:** as each summer4 package lands, replace one piece of glue with the
   package's API, keeping parity green. The end state has no `derived_fn` and no
   hand-built `y0`.

Decisions taken with the user: numpyro for calibration (summer4 WP10 once it
lands); numerical parity against the original summer3wip model; a fresh repo
under `monash-emu`.

### What the original model is

`src/tb_macro/tb_macro/{constants,epi,utils}.py` and `notebooks/01-TBModel.ipynb`:

- **Compartments.** `disease_state`: `mtb_naive incipient contained cleared active
  treatment recovered`; `age`: `0 5 15` on everything; `infectious`: `low high`
  and `clinical`: `subclin clin` **on `active` only** (ragged).
- **Flows** (20):
  - `infect_{mtb_naive,contained,cleared,recovered}` → `incipient`, rate
    `infect_process(...)`: per age, `ones(3,3) @ (I_age / N_age ** freq_dens_exponent)
    * contact_rate * rel_sus_{comp}`, where `I` is **unweighted** `active`;
  - `containment`, `clearance`, `breakdown`; `progression` `incipient →
    clin_strat["subclin"]` (dest leaves `infectious` unspecified → summer3 scatters
    1/2 to each);
  - `increase_infectiousness` / `decrease_infectiousness` (low ↔ high),
    `clinical_develop` / `clinical_regress`;
  - `self_recovery` `(active, subclin)` → `recovered` (collapses strata);
  - `passive_detection` `active → treatment` at `Parameter("detection")` **and**
    `detection` `(active, clin) → treatment` at `recent_detection_rate *
    tanh_based_scaleup(t, shape, inflection, past_frac, 1)` — two detection flows,
    both kept;
  - `treatment_recovery`; `treatment_relapse` `treatment → (subclin, low)`;
  - `ageing_0_to_5` at `1/5`, `ageing_5_to_15` at `1/10`;
  - `seed_peak` `mtb_naive → incipient` at `get_triang_vals(t, peak_time,
    peak_height, width)`.
- **Initial population.** 1000 per age via `strat_data_from_pandas`, all
  `mtb_naive`; `build_istate` divides the ragged strata evenly.
- **Run.** Times `1800.0 … 1999.0` yearly; summer3's runner integrates over the
  save **index** (0 … 199) with diffrax `Dopri5`, `PIDController(rtol=1e-5,
  atol=1e-5, dtmax=1)`, `max_steps = 4 * 200`. Flow outputs are **instantaneous
  rates** at save times.
- **Target.** `flows["infect_mtb_naive"].sum(to time).rolling(7).sum()[7:60:7]`,
  fuzzed with `exp(N(0, 0.01))`.
- **Calibration.** `contact_rate ~ Uniform(0.001, 0.5)`; Poisson likelihood of
  the fuzzed target against the model's rolling sum; NUTS 200/200 × 4 chains;
  arviz summary, posterior and trace plots.

---

## Repository shape

```
tb-macro-summer4/
├── pixi.toml
├── pyproject.toml                 # package tb_macro (hatchling), black line-length 100
├── AGENTS.md  LICENSE              # BSD-2-Clause, attributing the original repo
├── plans/tb-macro-summer4-port.plan.md
├── src/tb_macro/
│   ├── constants.py  functions.py  model.py  outputs.py  calibration.py
├── reference/                     # runs ONLY in the reference env
│   ├── tb_macro_original/         # vendored read-only copy of the original package
│   └── make_golden.py
├── tests/
│   ├── golden/*.parquet
│   ├── test_parity.py  test_functions.py  test_calibration.py
└── notebooks/01-TBModel.ipynb     # outputs stripped
```

`pixi.toml`:

```toml
[workspace]
name = "tb-macro-summer4"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64"]

[dependencies]
python = "3.13.*"
pandas = ">=2.2"
pyarrow = "*"
jupyterlab = "*"
pytest = "*"
black = "*"

[pypi-dependencies]
# Before WP12 tags a release, pin the stack commit instead:
#   rev = "377e3f84a803d6898d42d06d7cf10d47d9875d5a"
summer4 = { git = "https://github.com/monash-emu/summer4.git", tag = "v0.2.0a1", extras = ["jax", "calibration", "pandas", "frames"] }
arviz = "*"
plotly = ">=6.6, <7"
ipywidgets = "*"
nbformat = "*"
tb_macro = { path = ".", editable = true }

[tasks]
test = "pytest -q"
format = "black src tests reference"
notebook = "jupyter lab notebooks"

[feature.reference]
platforms = ["osx-arm64", "linux-64"]
dependencies = { python = "3.13.*", pandas = "*", pyarrow = "*" }
pypi-dependencies = { summer3 = { git = "https://github.com/monash-emu/summer3wip.git", rev = "<pin the commit used>" }, jax = ">=0.9", jaxlib = ">=0.9", diffrax = ">=0.7.2, <0.8", numpyro = ">=0.20, <0.21" }
tasks = { golden = "python reference/make_golden.py" }

[environments]
default = { solve-group = "default" }
reference = { features = ["reference"], no-default-feature = true }
```

The reference environment exists only to regenerate goldens; summer3wip and
summer4 pin incompatible JAX ranges, so they must not share a solve.

### Conventions (write into the new repo's `AGENTS.md`)

- Google Python style, line length 100, black; type-annotate every function.
- **Parity is the definition of correct.** Goldens in `tests/golden/` are only
  regenerated by `pixi run -e reference golden`, in a commit that says why.
- **Glue is temporary and labelled.** Every hand-written stand-in for a summer4
  gap carries a comment `# GLUE(TMn): replaced by WPk` naming its ports row and
  package, so M3 can find it with `grep -rn "GLUE("`. M3 is done when that grep
  is empty.
- Notebooks are a user gate: each phase PR lists the notebooks to run and the
  claim to check.
- Plans are immutable records in `plans/`.

---

## Phases

| Phase | Gate (summer4) | Ports rows | Acceptance |
|---|---|---|---|
| M0 scaffold and goldens | pin stack commit | `TM9` (pinned) | Goldens byte-stable; summer4 imports |
| M1 model on today's API | stack commit | `TM1`–`TM7` | Compartments and flows match goldens |
| M2 calibration notebook | stack commit | `TM7` `TM8` | Synthetic recovery; user sign-off |
| M3 upgrade | WP12, WP13, WP3, WP14, WP15, WP10 as they tag | `TM3`–`TM6` `TM8` `TM9` | Parity green after each swap; no `GLUE(` left |

### M0 — Scaffold and goldens

1. Create `monash-emu/tb-macro-summer4` (ask the user before creating the repo),
   add the layout, `pixi.toml`, `AGENTS.md`, `LICENSE`, CI running `pixi run test`
   on `linux-64`, and a manual-dispatch job that regenerates goldens and fails on
   any difference.
2. Vendor the original `tb_macro` package read-only into
   `reference/tb_macro_original/`, and the notebook's `infect_process`,
   population and `base_params` into `reference/make_golden.py`.
3. `make_golden.py` builds the model exactly as the notebook does, runs it through
   `CompartmentalEpiModel.run(params, solver_kwargs=...)` with tightened
   tolerances (`PIDController(rtol=1e-9, atol=1e-9, dtmax=1)`, `max_steps` raised
   to 20 000) so reference error is far below the parity tolerance, and writes:
   - `compartments.parquet` — every compartment, 200 yearly rows, columns named
     by the summer3 compartment string;
   - `flows.parquet` — every flow's per-edge instantaneous values summed per flow
     name, plus per-age sums for `infect_*` and `detection`;
   - `target.parquet` — the unfuzzed rolling-7 infection series at `[7:60:7]`;
   - `params.yaml` — `base_params`.
4. Fixtures: `base` (the notebook's `base_params`), and `reinfection` (all
   `rel_sus_* = 0.5`, `contain = clearance_rate = breakdown_rate = 0.1`,
   `freq_dens_exponent = 0.7`, `detection = 0.3`, `treatment_relapse = 0.05`),
   because `base` switches off most of the model and would hide wiring errors.

**Acceptance:** two golden runs give identical bytes; `pixi run test` imports
summer4 and reads the goldens.

### M1 — Model on today's summer4 API

`src/tb_macro/model.py`, `functions.py`:

- **Map** (`TM1`): `state`, `age`, then `stratify(infectious, where=state["active"])`,
  `stratify(clinical, where=state["active"])`.
- **Flows** (`TM2`), one `TransitionFlow` per original flow, names unchanged:
  - `progression`: `state["incipient"] → state["active"] & clinical["subclin"]`;
    `infectious` is dest-only and splits evenly by default, matching summer3's 1/2
    scatter. Assert the edge weights are 0.5 in a test.
  - `self_recovery`, `passive_detection`, `detection` collapse strata (source-only
    properties are dropped).
  - `treatment_relapse`: `state["treatment"] → state["active"] & clinical["subclin"]
    & infectious["low"]`.
  - Keep both detection flows and note the duplication in a docstring.
- **Ageing** (`TM3`): `# GLUE(TM3): replaced by WP13` —
  `TraitChain(age, (("0", "5"), ("5", "15")), rates=(1/5, 1/10))`.
- **Time functions** (`TM4`): `# GLUE(TM4): replaced by WP13` — `seed_peak` and
  `detection` rates as `Transform(fn, Time(), Param(...))` over the original
  `get_triang_vals` and `tanh_based_scaleup` (copied into `functions.py`).
- **Force of infection** (`TM5`): `# GLUE(TM5): replaced by WP14` — a
  `derived_fn(params, *, y, t)` returning the per-age FOI
  `ones(3,3) @ (I_age / N_age ** freq_dens_exponent) * contact_rate`
  broadcast over `age`, referenced by `FieldRef`; the four `infect_*` flows
  multiply it by `Param(f"rel_sus_{comp}")`. The derived struct passes params
  through.
- **Initial population** (`TM6`): `# GLUE(TM6): replaced by WP3` — `y0` built
  with `PropertyData.at[state["mtb_naive"] & age[a]].set(...)`; divide by the
  number of compartments each selector matches so the ragged even-split rule is
  explicit.
- **Run**: `compiled.run(params, y0, t0=1800.0, t1=1999.0, dt=1.0,
  solver="dopri5", rtol=1e-9, atol=1e-9, max_steps=20_000,
  save=SavePlan(ts=np.arange(1800.0, 2000.0)))`. summer3's index time `i` is
  year `1800 + i`, and every time function in the original takes that **index**
  as `t` (the notebook sets `seed_peak_time = 30.0` and
  `passive_detection_inflection = 40.0`, meaning years 1830 and 1840). Reproduce
  that: pass `Time() - 1800.0` into the time functions, and document it.
- **Outputs** (`TM7`), `outputs.py`: `FlowMass("infect_mtb_naive")`, summed over
  edges, `rolling(7, how="sum")`, sliced at the original `[7:60:7]` positions.

**Tests — `test_parity.py`:** for both fixtures, compartments at `rtol=1e-6` with
an absolute floor of `1e-9`; every flow's instantaneous rate; the target series;
the progression edge weights. `test_functions.py`: the two time functions against
the originals at 100 times.

### M2 — Calibration notebook

`notebooks/01-TBModel.ipynb` and `src/tb_macro/calibration.py`:

- Build, run, plot compartments by disease state (plotly, as the original).
- Synthetic target: the rolling-sum series at `contact_rate = 0.25`, fuzzed with a
  **seeded** `exp(N(0, 0.01))`.
- `# GLUE(TM8): replaced by WP10` — a hand-written numpyro model: `contact_rate ~
  Uniform(0.001, 0.5)`, run inside the model, `Poisson(target).log_prob(modelled)`
  as `numpyro.factor`, NUTS 200/200 × 4 chains with `chain_method="vectorized"`;
  arviz summary, posterior and trace plots.

**Tests — `test_calibration.py`** (marked slow): a 100/100 × 2-chain run recovers
`contact_rate = 0.25` inside its 95% interval and reports no divergences.

**User gate:** the user runs the notebook and signs off that the fit and posterior
look like the original's.

### M3 — Upgrade as summer4 packages land

One PR per package, each bumping the summer4 tag, replacing the matching `GLUE`
blocks, and keeping `test_parity.py` green **without changing the goldens**:

| summer4 package | Replace | With |
|---|---|---|
| WP12 | `rev =` pin | `tag = "v0.2.0a1"` (`TM9`) |
| WP13 | `GLUE(TM3)`, `GLUE(TM4)` | `TraitChain.from_breakpoints(age)`; triangular seed and tanh scale-up as rate trees (`abs`, `clip`, `tanh`) |
| WP3 | `GLUE(TM6)` | `InitialPopulation(base={state["mtb_naive"] & age[a]: 1000.0 for a in AGES})` with the even default |
| WP14 | `GLUE(TM5)` | `ForceOfInfection(kind="generalised", exponent=Param("freq_dens_exponent"), infectious=state["active"], group_by=age, contact_rate=Param("contact_rate"))` with `adjust=[Multiply(Param(f"rel_sus_{c}"))]` per flow; the `derived_fn` is deleted |
| WP15 | `rolling` slicing by hand | an `OutputSet` entry and `Result.to_frame` for plotting; `TM7`'s unrolled jaxpr is gone |
| WP10 | `GLUE(TM8)` | `BayesianModel(..., priors=[Uniform("contact_rate", 0.001, 0.5)], targets=TargetSet([Target(..., likelihood=Poisson())]))` and `sample(kind="nuts")` |

**Acceptance:** after the last swap, `grep -rn "GLUE(" src notebooks` is empty,
parity is green, `test_calibration.py` passes, and summer4's
`docs/evaluation/tb-ports.md` shows every `TM*` row `full`. If a swap cannot keep
parity, stop and open a summer4 issue naming the row; do not edit goldens to fit.

---

## Verification

- `pixi run test` (including slow tests in CI nightly) green at every phase.
- `pixi run -e reference golden` reproduces goldens byte-for-byte.
- Each phase PR body: ports rows touched, summer4 tag or commit, max relative
  parity error per fixture, notebooks for the user to run.
