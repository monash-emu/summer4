# Multi-model calibration: stacked members and declarative hierarchies

Out of scope for roadmap step 50 (`plans/multi-model-helpers.plan.md`), which adds
`BayesianModel.check_params`, `sample_params`, `factor`, `local_params` and
`posterior_runs(params_fn=)`. Two further gaps came up in the same design discussion.

## Stacked members (many models of one structure)

When K members share one compiled model and differ only in parameter values and data (60 TB
cohorts, sites, countries), the efficient form is one run `jax.vmap`ped over a leading member
axis inside a `numpyro.plate`, with save times padded to a common grid and masked. tbcore does this
by hand (`tbcore.cohorts.spectrum_log_likelihood`). summer4 offers only K separate
`BayesianModel`s factored in a Python loop, which traces K runs, so compile time grows linearly in
K (`AGENTS.md` § JAX, point 3). A `TargetSet` bakes one member's data in, so `log_likelihood`
cannot be vmapped across members with different data.

Done would look like: a batched target set (data `(K, T)` plus a mask) and a
`BayesianModel.log_likelihood` that accepts a leading member axis on parameters, tested to trace
once, not K times (`jax.make_jaxpr` size independent of K).

## Declarative hierarchical priors

Every hierarchy is hand-written numpyro today (`mu`, `tau`, `z ~ Normal(0, 1)`, value
`= exp(mu + tau z)`). A declarative prior, for example
`Hierarchical("recovery", loc=LogNormal(...), scale=HalfNormal(...), members=("a", "b"),
centred=False)`, would generate those sites, could carry the centring choice per member
(`futureplans/hierarchical-centring-diagnostics.md`), and would let `wf.lhs` stratify the
hyperparameters by name.

Done would look like: a joint model written with no hand-written numpyro, with a density test
against the hand-written equivalent.
