# Age-stratified SEIRS case study: completeness verdict and notebook

## Context

Two questions were asked: how completely can `feat/sparse-targets` express an
age-stratified SEIRS problem **using only existing features**, and can that be
demonstrated in a notebook.

The verdict is **yes, entirely — no new core `summer4` code required**. I built
and ran the whole problem against the installed package to confirm it, including
the calibration. What the ledger records as `none` for mixing matrices (`M1`),
infectiousness adjustments (`A4`) and initial population (`L4` `L5` `S8`) means
*no declarative primitive*, not *unreachable*: each is a few lines of JAX inside
`derived_fn`, or an array handed to `run(params, y0, ...)`.

Two things are worth landing beyond the notebook. First, per the project's
convention that analyses become versioned repo artifacts, the verdict belongs in
`docs/evaluation/` with ledger IDs other branches can resolve. Second, running
the problem surfaced four genuine sharp edges — one of which silently produced a
wrong model in my own first attempt — and those belong in `futureplans/`.

Coverage status does not change: this is documentation over shipped features, so
no ledger row moves and `pixi run coverage` totals stay put.

## Completeness verdict (validated by execution)

| Requirement | How it is expressed today | Ledger |
|---|---|---|
| SEIRS × 3 age bands | `PropertyMap.from_property(state).stratify(age)` → 12 rows | `S1` `S2` `full` |
| Ageing population, skews old | `TraitChain(age, pairs, rates=(1/15, 1/50))` as one named flow; `ExitFlow(Everything())` + `EntryFlow` replacement births | `S6` `partial` (no bundled `AgeStratification`), `F3` `F5` `full` |
| Skewed-old initial state, `I == 0` | `PropertyData.wrap(pmap, zeros).at[state["S"]].set([150k, 400k, 450k])` passed as `y0` | `L4` `L5` `S8` `none` = **no helper**, not unreachable |
| Homogeneous mixing matrix | plain `numpy` `K` in `derived_fn`; `K @ eff` | `M1` `none` — hand-written |
| Age-varying infectiousness | `params["infness"] * I_a / N_a` before the matmul | `A4` `none` — hand-written |
| Per-age force of infection | `pd.broadcast_over(age, foi)` → `pmap.size` vector; `_align_rate` gathers it per row | `F7` `F8` `partial` — hand-written |
| Parameterised infection inflow | `EntryFlow("importation", state["I"], refs.imp, split={age: {...}})`, Gaussian pulse in JAX | `F6` `full`; time shape hand-written (`P6`–`P8` `none`) |
| Outputs at target parameters | `run(..., solver="tsit5")` → `Result`; `FlowMass("progression", where=Dest(age[a]))` | `D1` `D2` `V2` `full` |
| Noise, then sparse selection | host-side seeded `numpy`; `Target(..., quantity=...)`; `TargetSet.plan(SavePlan())` | WP11, this branch |
| Calibrate against them | `jax.jit(jax.value_and_grad(loss))` + `optax.adam`; loss over `TargetSet.residuals` | WP11 ships residuals; likelihood is WP10 |

**Measured recovery** (homogeneous mixing, 22 weekly observations per age band,
10% lognormal noise, 600 Adam steps at lr 0.05, ~6.5 s):

| Parameter | True | Recovered |
|---|---|---|
| `contact_rate` | 0.550 | 0.551 |
| `seed_rate` | 5.000 | 4.893 |

Fitting the infectiousness gradient as well requires assortative mixing; see
the finding below. The notebook executes in about 31 seconds.

The one substantive modelling finding: **a homogeneous mixing matrix cannot
identify age-varying infectiousness.** Capturing the force of infection with
`ComputedValue` shows why exactly — under a homogeneous matrix the largest
difference in it between bands, over all times, is 0.0, so the infectiousness
weights enter only through a single scalar sum. A fit that frees them recovers
an *inverted* age gradient (1.401 / 1.259 / 0.716 against 0.603 / 0.862 /
1.207) at a loss slightly below the loss at the generating parameters. Swapping
in an assortative matrix — a one-line change, because the matrix is a parameter
— makes the force of infection band-specific and the gradient identifiable
(0.487 / 0.982 / 1.168, monotone as generated). This becomes the spine of the
notebook, and the strongest available argument for WP6.

```{note}
Corrections made while building, recorded here rather than silently:

1. An earlier draft of this plan claimed a flat likelihood ridge cured by a
   normalisation constraint. That flatness was an artefact of
   `PropertyData.where` polarity — the force of infection was being computed
   over S + E + R. With the correct force of infection the profile has a real
   minimum, and the true finding is the mixing result above. The normalisation
   is still used, but as a redundancy fix rather than an identifiability one.
2. The loss is least squares on the **log** scale, via `TargetSet.gather`
   rather than `TargetSet.residuals`. Raw-scale residuals are dominated by the
   peak and Adam stalls on them (an early fit reached `contact_rate` 0.195
   against 0.550). Log-scale is also the estimator the multiplicative noise
   implies.
3. The initial population is not written down at all: the demography is
   integrated alone for 300 years with transmission off, and its final state
   becomes `y0`. The steady state is checkable in closed form, which makes the
   "skews old" requirement an assertion rather than a choice of numbers.
```

## Deliverables

Branch `docs/age-stratified-seirs-case-study` off `feat/sparse-targets`
(`Target`/`TargetSet` are not on `origin/main`). Documentation only — no
`src/summer4` changes, so `scripts/check_feature_branch.py` does not require an
`examples/notebooks/` twin.

### 1. `docs/case-studies/` — new section

- `docs/case-studies/index.md` — section intro plus a `{toctree}` listing
  `age-stratified-seirs`. Say what a case study is (an integrated problem, as
  opposed to the one-feature-per-chapter user guide).
- Register `case-studies/index` in `docs/index.md`'s toctree, after `user/index`.

### 2. `docs/case-studies/age-stratified-seirs.ipynb` — the notebook

~21 cells, strict markdown/code alternation, matching `docs/user/10-targets-and-fitting.ipynb`
in voice and rigour: short declarative prose, every claim ending in an `assert`
or `np.testing.assert_allclose`, `print` only for numbers a reader wants, seeded
RNG, no magics, hand-formatted to 100 columns.

1. **H1** — the problem, the model, and an honesty note naming which pieces are
   hand-written with their ledger IDs.
2. Imports; `pd.options.plotting.backend = "plotly"`; `state`/`age` properties.
3. **H2 The compartment space** → `pmap`, `labels()`, `assert pmap.size == 12`.
4. **H2 An ageing population that skews old** → `TraitChain` with per-band rates
   `1/15`, `1/50`; universal death; replacement births into `state["S"] & age["0-14"]`
   via `death.sum()`. Assert the top band has no outgoing ageing edge.
5. **H2 Force of infection, mixing, and age-varying infectiousness** → the
   algebra written out, why `broadcast_over` is the bridge from a 3-vector to a
   12-row rate (`_align_rate`, `flows/compiled.py:193`), and an explicit warning
   that `PropertyData.where(sel, x)` **replaces** matching compartments, so
   infectious prevalence is `pd.where(~state["I"], 0.0).sum_over(age)`.
6. `derived_fn` returning a `NamedTuple` of `foi` and `imp`; `derived_refs`; the
   infection/progression/recovery/waning flows.
7. **H2 Parameterised introduction of infection** → `EntryFlow` with a Gaussian
   pulse and `split=` across age bands; skewed-old `y0`;
   `assert y0 I total == 0.0`.
8. **H2 Run it** → `compile(derived_fn=...)`, `run(solver="tsit5", epoch=Epoch(...))`.
   Plotly: per-age prevalence, per-age notifications, and the population
   structure drifting under ageing (which shows the demography is live).
9. **H2 Noisy, sparse observations** → seeded weekly sampling, 10% lognormal
   noise, three `Target`s with `quantity=FlowMass("progression", where=Dest(age[a]))`,
   `TargetSet.plan(SavePlan())`. Assert per-key `ts.size`, and assert equal times
   collapse to one save group via `group_requests`. Plotly scatter of observations
   over the true curves.
10. **H2 Why the naive parameterisation is not identifiable** → the profile
    sweep above, asserting the loss spread is below a tolerance, then the
    population-weighted normalisation as the fix.
11. **H2 Calibrate** → loss closing over `TargetSet.residuals`, scaled by the
    observations; `jax.jit(jax.value_and_grad(...))`; `optax.adam`; print
    recovered vs true; assert each within tolerance; plotly loss-convergence trace.
12. **H2 Reporting run** → a *separate* dense `SavePlan` with fitted parameters
    (a likelihood plan and a reporting plan are two plans, not two modes);
    plotly overlay of fit vs observations per band; `resample("W")`; `cumulative()`.
13. **H2 What this needed that summer4 does not provide** → closing assessment
    pointing at the evaluation page and WP3/WP5/WP6/WP10.

Notebook mechanics that must be respected: `nbformat` 4 / `nbformat_minor` 5,
every cell with an 8-hex `"id"` and `"metadata": {}`, code cells with
`"execution_count": null` and `"outputs": []`, and the standard `kernelspec` /
`language_info` trailer — `pixi run check-notebooks` and the pre-commit hook
reject anything else.

Two constraints found by running it, which the notebook must work with rather
than around:

- `cm.describe(plan)` **cannot** be used here. It takes no `params`, so it calls
  `derived_fn(None, ...)` and dies with `'NoneType' object is not subscriptable`.
  Show the memory saving from the plans' own `ts` sizes instead, and say why.
- `TargetSet.residuals` has no reduction hook. Per-age `FlowMass(..., where=Dest(age[a]))`
  yields `(T, 1)`, which reshapes onto a `(T,)` target. A `sum_over=(age, "dest")`
  grouped trace yields `(T, 3)` and raises — so target one band per `Target`.

### 3. `docs/dev/plotting.md` — the convention, recorded

Adopt `pd.options.plotting.backend = "plotly"` plus `trace.to_pandas().plot()` —
the summer2 documentation idiom, and no new core code. Record that `Trace.plot()`
stays matplotlib (`results/trace.py:469`) and is the quick host-side path, that
`docs/conf.py:84` already serves the plotly CDN site-wide so figures render in
the Sphinx build, and that notebooks are executed per-kernel by `myst_nb` so the
backend option does not leak between them. Register in `docs/dev/index.md`.

### 4. `docs/evaluation/age-stratified-seirs-case-study.md`

The verdict table above as a durable, citable page: requirement → mechanism →
ledger ID, the measured recovery numbers, the identifiability finding, and the
four sharp edges with file:line pointers. Register in `docs/evaluation/index.md`'s
toctree. This is the artifact other branches and agents cite; the notebook is
the runnable demonstration of it.

### 5. Corrections to existing evaluation prose

- `docs/evaluation/gaps.md:133` — close the "A plotting convention" gap, pointing
  at `docs/dev/plotting.md`.
- `docs/evaluation/gaps.md:60` — currently claims derived outputs and a results
  object are "not started" and "the binding constraint". Three phases stale;
  correct it while in the file.

### 6. `futureplans/` notes — one concern per file

- `propertydata-where-polarity.md` — `where(sel, other)` replaces matching
  compartments (`jax/propertydata.py:112`), i.e. `Series.mask` semantics under a
  name every modeller reads as `keep`. This silently produced an FOI over
  `S+E+R` in my first run: the model still integrated, the epidemic just
  saturated instantly. Proposal: a `keep`/`only` alias, or a docstring that
  leads with the polarity.
- `describe-requires-params.md` — `CompiledModel.describe` (`flows/compiled.py:548`)
  has no `params` argument and passes `None` to `derived_fn`, so plan sizing is
  unavailable for exactly the models that most need it.
- `targetset-residual-reduction.md` — `TargetSet.residuals`
  (`results/targets.py:166`) reshapes but cannot reduce, so a stratified save
  cannot meet a 1-D series without a `SaveFn`. Relevant to WP10.
- `trace-plot-backend-coupling.md` — `Trace.plot()` hardcodes matplotlib and
  passes `**kwargs` to `frame.plot()`, so `plot(legend=False)` (used in
  `docs/user/09` and `10`) raises under the plotly backend.

### 7. `plans/age-stratified-seirs-case-study.plan.md`

This plan, copied onto the branch per `AGENTS.md`, carrying the notebook gate below.

## Notebook gate — blocking

> Do not merge until the user has run this in `pixi run notebook` and ticked:
>
> - [ ] `docs/case-studies/age-stratified-seirs.ipynb` — the five Plotly figures
>   read as a modeller's own working notebook, the demographic transient is
>   visibly doing something, and the mixing/identifiability section earns its
>   place rather than reading as an apology.
> - [ ] The fit recovers the target parameters to the stated tolerances on the
>   user's machine, and 31 s of execution is acceptable for a docs build.
> - [ ] The honesty framing is right: the notebook says plainly that the force
>   of infection, the mixing matrix and the infectiousness weighting are
>   hand-written, without either apologising for it or burying it.

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
```

```bash
pixi run test && pixi run check-branch
```

```bash
pixi run coverage
```

```bash
pixi run -e docs docs-strict
```

`docs-strict` is the load-bearing one: it executes the notebook
(`nb_execution_raise_on_error = True`, 300 s timeout) and fails on any missing
toctree entry, so it proves both that the case study runs end to end and that
the four new pages are wired in. `pixi run coverage` must report unchanged
totals — this branch adds no capability. Confirm the plotly figures actually
render in `docs/_build/html` rather than trusting the CDN line in `conf.py`.
