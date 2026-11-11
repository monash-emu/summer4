# Release notes

## 0.3.0a2: runtime defaults

This release changes three defaults so that models are more accurate and use your whole computer
without any set-up. Guide page 10, {doc}`What summer4 sets up for you <guide/10-runtime>`, says
what each one is, how to check it with `summer4.runtime_info()`, and how to change it.

- **`run()` is more accurate.** With no `solver=`, it uses an adaptive Tsit5 solver, not fixed-step
  Euler. It adjusts its step to the model, so a repeating cycle stays closed, and it is also much
  faster than Euler at a step size that is accurate enough. Because the answers are closer to the
  exact ones, numbers from an existing model will change a little (or, if Euler was inaccurate,
  a lot). To get the old numbers, pass `solver="euler"` to `run()`, call
  `summer4.solvers.set_default_solver("euler")`, or start Python with `SUMMER4_SOLVER=euler`.
  `dt` is still needed: it is the first step and sets the save times.
- **Your CPU cores are visible to JAX.** Importing summer4 makes JAX report one CPU device per core.
  Switch it off with `SUMMER4_HOST_DEVICES=0`. Import summer4 before you use JAX.
- **`chain_method` accepts `"auto"`** (currently `"vectorized"`), and `resolve_chain_method` is
  public. Parallel chains are opt-in (`chain_method="parallel"`) and are not yet compatible with
  gradients through the adaptive solver on recent JAX; they work with `solver="euler"`.

New functions: `summer4.runtime_info()`, `summer4.solvers.default_solver()`,
`summer4.solvers.set_default_solver()` and `summer4.epi.calibration.resolve_chain_method()`.

One thing to watch: an older `derived_fn` that turns parameters into Python numbers (for example
`float(params["beta"])`) fails under the adaptive default. Keep the values as JAX arrays, or pass
`solver="euler"`.

## 0.3.0a1: the clean surface (step 48)

This release removes the renamed spellings that were kept only for compatibility, and
leaves one way to write each thing. The documentation has been rebuilt from scratch on the
unified idiom: a short guide, the textbook chapters migrated to the same idiom, and a roadmap
page. The documentation is published at
<https://monash-emu.github.io/summer4/>. Nothing is kept behind a deprecation warning: each
removed spelling is gone, so update your code using the table below. One addition:
`Reduce(where=sel)` with no `group_by` is now the scalar number of individuals in the matching
compartments (`Reduce()` is the whole population), so mass-action rates such as
`Param("a") * Reduce(where=species["predator"])` need no epidemiology helper.

### Migration table

| Removed | Use instead | Example |
| --- | --- | --- |
| `Output.sum_over(prop)` | `Output.sum_by(prop)` | `infectious.sum_over(age)` becomes `infectious.sum_by(age)` |
| `Output.total()` | `Output.sum_by()` | `infectious.total()` becomes `infectious.sum_by()` |
| `sum_over` / `total` on an `OutputSet` expression | `sum_by` | `outputs.ref("by_age").sum_over(age)` becomes `outputs.ref("by_age").sum_by(age)` |
| Save-spec `total()` | `sum_by()` | `FlowMass(flow).total()` becomes `FlowMass(flow).sum_by()` |
| Save-spec `owner(sum_over=...)` | `owner(group_by=...)` | `owner(sum_over=state)` becomes `owner(group_by=state)` |
| `PropertyData.sum_over(prop)` | `PropertyData.sum_by(prop)` | `data.sum_over(age)` becomes `data.sum_by(age)` |
| `FlowRef.sum()` | `FlowRef.sum_by()` | `flow.sum()` becomes `flow.sum_by()` |
| `FlowRef.sum_over(prop)` | `FlowRef.sum_by(prop)` | `flow.sum_over(age)` becomes `flow.sum_by(age)` |
| `Reduce(sum_over=...)` | `Reduce(group_by=...)` | `Reduce(sum_over=age)` becomes `Reduce(group_by=age)` |
| `Output.cumulative()` on a rate | `Output.cumulative_integral()` | `new_cases.cumulative()` becomes `new_cases.cumulative_integral()` |
| `Output.cumulative()` on amounts per interval | `Output.cumsum()` | `new_cases.integrate_intervals().cumulative()` becomes `new_cases.integrate_intervals().cumsum()` |
| A callable base value in `InitialPopulation` | A number, a `Param`, or `defer(f)(Param(...))` | `lambda p: p["a"] * 10` becomes `defer(f)(Param("a"))`, where `f` multiplies its argument by 10 |
| A callable `weights=` table in `Split` | A mapping of numbers or `Param`s, or `defer(f)(Param(...))` | `Split(imm, lambda p: table(p["v"]))` becomes `Split(imm, defer(table)(Param("v")))` |

The `cumulative()` rows matter because the two replacements count differently. Use
`cumulative_integral()` for a rate (people per unit time), and `cumsum()` for amounts already
summed over each interval.

The older ways of writing a model that remain (`compile(derived_fn=...)`, `derived_refs`,
`SaveFn`, `ComputedValue` and `ForceOfInfection(kind=...)` with a plain function) are not
removed in this release. They are listed, with their replacements, on the
{doc}`roadmap <roadmap>` page.

## Earlier releases

For earlier releases, see the [CHANGELOG on GitHub](https://github.com/monash-emu/summer4/blob/main/CHANGELOG.md).
