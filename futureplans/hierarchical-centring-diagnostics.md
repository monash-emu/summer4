# Centred vs non-centred hierarchies: the funnel, and diagnosing it automatically

Written up from a design discussion (2026-11-17) on multi-model and hierarchical calibration
through `NumpyroPosterior` (roadmap step 46, `CX35`). Nothing here is scheduled. It records:

- what the centred/non-centred choice does to NUTS;
- what that means for summer4 users writing joint models;
- what an automated check in `summer4.epi.calibration.workflow` could look like.

Related: tbcore `hierarchical_nh` (`tb-core/src/tbcore/natural_history_prior.py`) is
non-centred; notebook `examples/notebooks/28-calibration-custom-posterior.ipynb` is
non-centred.

## 1. The two parameterisations

K groups (cohorts, sites, countries, submodels), each with a value θ_k around a population
location μ with spread σ.

**Centred:** sample θ_k directly.

```
μ ~ …;  σ ~ HalfNormal(…)
θ_k ~ Normal(μ, σ)                 # or log θ_k ~ Normal(log μ, σ) for rates
```

**Non-centred:** sample a standardised offset and compute θ_k.

```
μ ~ …;  σ ~ HalfNormal(…)
z_k ~ Normal(0, 1)
θ_k = μ + σ·z_k                    # or θ_k = μ·exp(σ·z_k) for rates (tbcore's frailties)
```

The model and the posterior over (μ, σ, θ) are the same in both. What changes is the coordinate
system the sampler (and an optimiser) moves in.

## 2. The funnel (Neal's funnel)

In the centred form, how far θ_k can sit from μ depends on σ:

- Large σ: θ_k ranges widely; the posterior is broad.
- Small σ: every θ_k is squeezed to within about σ of μ; the posterior is a narrow neck with
  very high density.

Plotted as log σ against θ_k, the posterior is a funnel. With K groups the neck pinches in K
dimensions at once.

### Why NUTS fails on it

NUTS uses one step size (and one mass matrix) per chain, adapted in warmup.

- A step size that suits the mouth overshoots in the neck. The energy error blows up, NUTS
  records a **divergent transition**, and it rejects the trajectory.
- A chain that does get into the neck can take only tiny steps, so it sticks.

Consequences:

1. **Bias.** The small-σ region is under-sampled, so σ is overestimated. The fit shows too little
   pooling, and the groups look more different than they are. Small σ is exactly the regime
   that matters: the data say the groups are similar.
2. **Divergences.** The visible symptom. They mean the results are wrong, not just noisy, and
   they cluster at low σ.
3. **Poor mixing.** R-hat is high, ESS is low, and chains disagree, depending on which ones
   reached the neck.
4. **Tuning doesn't fix it.** A diagonal mass matrix and one step size can't absorb curvature
   that varies by orders of magnitude. Raising `target_accept_prob` reduces divergences at a
   large speed cost, and it seldom removes the bias.

### Why non-centring fixes it

z_k ~ Normal(0, 1) does not depend on σ, so the prior geometry is a round Gaussian with no neck.
The σ-dependence moves into a deterministic transform the sampler never has to navigate.

### Non-centring is not always better

The choice depends on how strongly each group's own data pin its θ_k.

| Situation | Better form | Why |
|---|---|---|
| Little data per group, or small σ (strong pooling) | **Non-centred** | The prior dominates the shape, and the centred prior is the funnel |
| Lots of data per group, so each θ_k is well identified | **Centred** | The likelihood pins θ_k, so z_k = (θ_k − μ)/σ is tightly coupled to μ and σ. The non-centred form now has the bad geometry (an "inverted funnel"). |

The answer can differ between groups in one model. Per-site choice, and partial centring in
between the two forms, are available through numpyro's
`numpyro.infer.reparam.LocScaleReparam(centered=...)`.

## 3. Implications for summer4 joint models

- **The typical members are weakly identified.** Historical cohorts with a few data points, a
  "target setting" that is one more exchangeable member, and countries informed only by
  aggregate targets all fit this. **Non-centred is the right default**, as in tbcore.
- **A member with rich data may want centring.** For example, a setting with dense time-series
  targets that pin its own rates.
- **Optimisation is affected too, not only NUTS.** In the centred form the density grows
  without bound as σ → 0 with every θ_k = μ, so a proper MAP may not exist. `wf.optimize`
  (multi-start Adam) slides down the funnel and reports complete pooling. Its ends then make bad
  `Candidates.init_params` for NUTS.
- **LHS doesn't see the plate.** `wf.lhs` stratifies only scalar, parentless sites with an icdf.
  Plate members and centred children are filled by `Predictive`
  (`NumpyroPosterior._auto_site_priors`). In the centred form, that puts the starting θ_k at
  prior spread, far from the funnel's neck.
- **The σ prior matters.** A HalfNormal-type prior with mass near zero but a modest tail helps
  both forms. A wide σ prior widens the funnel's mouth.

## 4. What we already record

- `MCMCRun.diverging`: per-draw `(chain, draw)` divergence flags. `MCMCRun.extra` holds
  requested extra fields such as `energy`, `num_steps` and `adapt_state.step_size`.
  `MCMCRun.idata` carries `diverging` into `sample_stats`
  (`src/summer4/epi/calibration/workflow/mcmc.py`).
- `WarmupRule.max_divergence_frac` (default 0.01) and `StopRule.max_divergence_frac` gate on the
  *rate* of divergences (`workflow/warmup.py`, `workflow/mcmc.py`).
- `NumpyroPosterior` already detects "centred hierarchical children" (a site whose distribution
  changes with the trace seed) to keep them out of LHS (`posterior.py::_auto_site_priors`).

So we detect *that* divergences happen, but not *where* they happen or whether a funnel is the
cause, and we don't advise on parameterisation.

## 5. Automated inspection: candidate checks

All of these work on draws we already keep. The one input they need is the hierarchy: which
scale site governs which member sites. That can come from the user as a mapping
(`{"frailty_sd_mort": ["frailty_z_mort"], "tau": ["z", "z_main"]}`), or be inferred from the
model trace (5.7).

### 5.1 Where are the divergences? (cheap; the most useful first check)

For each scale site σ, compare the distribution of log σ in divergent draws against
non-divergent draws: a quantile shift or a KS statistic. If divergent draws concentrate in the
lower tail of σ, flag it: "divergences sit at small σ, a funnel signature". Do the same for
each member site, to find *which* groups are involved.

### 5.2 Funnel correlation

- Centred form: correlation, across draws, between log σ and log|θ_k − μ|. A strong positive
  correlation means θ_k's spread is controlled by σ, so the funnel is present.
- Non-centred form: correlation of z_k with μ and with log σ. A strong correlation (an inverted
  funnel) means this group's data pin θ_k and it would sample better centred.

Report one row per (scale, member), with the worst members first.

### 5.3 Per-group recommendation: data strength (the core advisor)

From a pilot run in either parameterisation, compute for each member k:

```
r_k = posterior sd of θ_k  /  posterior mean of σ
```

- r_k ≈ 1: the data add little beyond the hierarchy, so **non-centre** k.
- r_k ≪ 1 (say < 0.3): the data pin θ_k, so **centre** k.

The output is a suggested per-group centring. It could feed `LocScaleReparam(centered=...)`
directly, or become a recommendation printed in the report. This follows the standard
heuristic in the literature (Papaspiliopoulos, Roberts & Sköld 2007; Betancourt & Girolami
2013).

### 5.4 Learned centring (numpyro, no new code)

`LocScaleReparam(centered=None)` makes the centring a learnable parameter, fitted by SVI
("variationally inferred parameterisation", Gorinova, Moore & Hoffman 2020). This could be a
`wf` stage before NUTS: fit the centring by SVI, then fix it and sample. It only applies to
location-scale sites written as `Normal(loc, scale)` (centred), so it needs the user's model in
that shape.

### 5.5 Sampler-health signals

- **E-BFMI** from `energy` (requires `extra_fields=("energy",)`). Values below about 0.3 indicate
  heavy-tailed or funnel geometry.
- **Tree depth saturation** (`num_steps` at 2^max_tree_depth − 1).
- **Adapted step size**, small relative to the prior scale of the hyperparameters.

All three are already in reach through `MCMCRun.extra`; `WarmupRule` could gate on E-BFMI as it
does on divergences.

### 5.6 Optimiser end check

After `wf.optimize` on a `Posterior` with known scale sites: flag ends where σ has run to its
lower support (or shrunk by orders of magnitude from its start) while the log density is still
rising. Message: "optimiser collapsed into the funnel; MAP may not exist in this
parameterisation". This catches the problem before NUTS is ever started from those ends.

### 5.7 Structural detection from the trace

Infer the hierarchy with no user mapping:

- `numpyro.infer.inspect.get_dependencies(model)` (check which numpyro versions ship it) gives
  each site's parents. A `Normal`/`LogNormal` site whose `scale` parent is a positive latent is
  a centred child of that scale.
- A `Normal(0, 1)` plate site whose value feeds a deterministic along with a scale site is a
  non-centred offset. That needs a value-dependency trace, harder than the parent graph; the
  user mapping is the fallback.

With this, `NumpyroPosterior` can warn at construction time, before any sampling: "site θ
(plate of 60) is centred on scale σ; consider non-centring if the groups are weakly
identified".

### 5.8 Plots (for the gate notebook, not the API)

`arviz.plot_pair(run.idata, var_names=[σ, θ_k…], divergences=True)` with log σ on one axis
shows the funnel and the red divergence markers at once. `arviz.plot_parallel` shows
divergent trajectories bunching in the funnel neck. A helper could pick the worst (scale,
member) pairs from 5.2 and plot only those, since K = 60 members is too many to plot all.

### 5.9 Simulation check

For a new hierarchical structure, simulate from a small-σ truth and fit:

- Does the posterior cover the true σ?
- Do divergences appear?

A one-off simulation-based-calibration-style test catches funnel bias that the fit-to-real-data
diagnostics can only suspect.

## 6. A possible API shape

```python
report = wf.hierarchy_report(
    post,                      # any Posterior
    run,                       # MCMCRun (or OptimizeRun for 5.6)
    scales={"frailty_sd_mort": ["frailty_z_mort"]},   # optional if 5.7 infers it
)
report.table        # per (scale, member): divergence shift, funnel corr, r_k, recommendation
report.worst(5)     # pairs to plot
report.centring()   # {site: True/False/float} suitable for LocScaleReparam
```

It is domain-neutral, so it belongs in summer4's workflow. tbcore and tb-nh-joint-calibration
would be the first users: the 60-cohort frailty plate plus the VNM target member is exactly
the case where some members are weak (short cohorts) and one may be strong (VNM targets).

**Done when:** a report like the above runs on notebook 28 and on the tb-nh joint model. It
must flag a deliberately centred version of notebook 28 (and pass on the non-centred one), and
the gate notebook shows the funnel pair plot.
