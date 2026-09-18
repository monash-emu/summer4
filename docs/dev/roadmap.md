# Roadmap — the next step, and how to run it

**This file is the authoritative record of where the work has got to.** The
coverage ledger says what summer4 *can do*; this file says what to *do next*.
A new session that is told only "next step" should be able to start work from
this page alone.

```{admonition} Contract for anyone finishing a step
:class: important

1. The *Current position* block names exactly one step. It is the only place
   that records position; nothing else on this site should claim to.
2. Statuses are exactly `done`, `in-progress`, `next`, `planned` or `blocked`.
   Nothing else parses.
3. Finishing a step means committing the handoff **on the step's own branch**,
   so it merges with the work. A step that lands without its handoff has left
   the next worker with nothing.
4. Run `pixi run roadmap` after editing. `tests/test_roadmap.py` fails if a
   status is misspelled, if the current step is missing or ambiguous, if a step
   cites a plan file that does not exist, or if a step section is incomplete.
```

## How to run a step

Follow this exactly. It assumes you know nothing about the project.

1. Read the **Current position** block below. It names one step number.
2. Jump to that step's `## Step N` section. **Present its *Summary* to the user
   before doing anything else** — three or four sentences covering what the step
   ships, which branch it lands on, and what it closes. Then begin; you do not
   need approval to start, but stop and ask if the summary names an open
   question.
3. Read everything under *Read first*, in the order listed. `AGENTS.md` is
   always first and is not optional.
4. Cut the branch named in the step table, from the branch named in *Cut from*.
   Copy the step's plan onto the branch under `plans/` if it is not already
   there (`pixi run check-branch` fails a feature branch with no plan).
5. Do the work under *Do*, following the plan section it cites. **Where *Do*
   contradicts the cited plan, *Do* wins** — plans under `plans/` are immutable
   history, so corrections made since they were written live here, in
   [Corrections to committed plans](#corrections-to-committed-plans).
6. Run every command under *Exit checks*. All of them must pass.
7. Open a PR into the branch named in *Merges into*. Put the user-gate checklist
   in the body: every notebook the step ships or unblocks, and the claim to
   check in each. **The notebooks are a blocking manual sign-off** — the branch
   is not merged until the user has run them (`pixi run notebook`).
8. Make the *Handoff* commit last, on this branch, before the merge. It is one
   commit, `Hand off after step N: <slug>.`, and it does four things:

   - sets step N's status to `done` in the step table, and adds a line
     `**Landed:** <branch>, PR #<n>, <date>.` to the top of step N's section
     (the checker requires it, so a done step can always be traced to its PR);
   - sets step N+1 to `next` and rewrites the *Current position* block;
   - adds a `### What the previous worker left you` section to step N+1 —
     deviations from the plan, anything deferred, every `futureplans/` note
     written, and anything that turned out to be untrue;
   - moves any ledger rows the step closed, in this same commit.

Two conventions that are easy to get wrong:

- **Ledger rows move in the same commit as the code that moves them.** Both
  `docs/evaluation/coverage-ledger.md` and `docs/evaluation/tb-ports.md`, then
  `pixi run coverage-write` and `pixi run coverage`.
- **Notebook numbering is first-come.** Where a step names a notebook number,
  that number assumes the default order below. If an earlier-numbered slot has
  been taken by a step that landed out of order, use the next free two-digit
  prefix and correct this file in the same commit.

## Current position

<!-- roadmap:current -->
| Field | Value |
| --- | --- |
| Step | 2 |
| Status | next |
| Branch | `chore/release-v0.2` |
| Cut from | `main` |
| Last landed | `chore/merge-flows-stack` |
<!-- /roadmap:current -->

## Steps

Phases A–F are sequential. Track G is independent of them and may run at any
time after step 1. Phase H is downstream work in other repositories and needs
only step 2's tag.

<!-- roadmap:steps -->
| Step | Phase | WP | Branch | Plan | Section | Status | Closes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | WP12 | `chore/merge-flows-stack` | `plans/wp12-release.plan.md` | Step 1 | done | — |
| 2 | A | WP12 | `chore/release-v0.2` | `plans/wp12-release.plan.md` | Step 2 | next | KI23 TM9 |
| 3 | A | WP12 | `chore/downstream-smoke-ci` | `plans/wp12-release.plan.md` | Step 3 | planned | — |
| 4 | B | WP13 | `feat/rate-math` | `plans/tb-ports-feature-completeness.plan.md` | 13.1 | planned | — |
| 5 | B | WP13 | `feat/table-interp` | `plans/tb-ports-feature-completeness.plan.md` | 13.2 | planned | KI6 KI10 KI11 TM4 |
| 6 | B | WP13 | `feat/ageing-sugar` | `plans/tb-ports-feature-completeness.plan.md` | 13.3 | planned | KI2 TM3 |
| 7 | C | WP14 | `feat/epi-generalised-foi` | `plans/tb-ports-feature-completeness.plan.md` | 14a | planned | — |
| 8 | C | WP14 | `feat/epi-compartment-infectiousness` | `plans/tb-ports-feature-completeness.plan.md` | 14b | planned | KI4 KI5 TM5 |
| 9 | D | WP15 | `feat/trace-algebra` | `plans/tb-ports-feature-completeness.plan.md` | 15a | planned | — |
| 10 | D | WP15 | `feat/output-sets` | `plans/tb-ports-feature-completeness.plan.md` | 15c | planned | KI13 KI14 KI15 KI16 KI17 |
| 11 | E | WP16 | `feat/tb-scale-bench` | `plans/tb-ports-feature-completeness.plan.md` | 16a | planned | — |
| 12 | E | WP16 | `feat/solver-safety` | `plans/tb-ports-feature-completeness.plan.md` | 16b | planned | KI22 |
| 13 | F | WP10 | `feat/epi-priors-likelihoods` | `plans/tb-ports-feature-completeness.plan.md` | 10.1 | planned | — |
| 14 | F | WP10 | `feat/epi-sampling` | `plans/tb-ports-feature-completeness.plan.md` | 10.2 | planned | — |
| 15 | F | WP10 | `feat/epi-posterior-runs` | `plans/tb-ports-feature-completeness.plan.md` | 10.3 | planned | KI18 KI19 KI20 KI21 TM8 |
| 16 | G | WP17 | `feat/foi-susceptibility` | `plans/wp17-foi-susceptibility.plan.md` | Whole | planned | — |
| 17 | G | WP9 | `feat/contact-survey-data` | `plans/wp9-contact-surveys.plan.md` | Step 17 | planned | — |
| 18 | G | WP9 | `feat/contact-matrix-adaptation` | `plans/wp9-contact-surveys.plan.md` | Step 18 | planned | — |
| 19 | G | WP9 | `docs/textbook-16-19` | `plans/wp9-contact-surveys.plan.md` | Step 19 | planned | — |
| 20 | H | — | *(new repo)* | `plans/tb-macro-summer4-port.plan.md` | Whole | planned | — |
| 21 | H | — | *(new repo)* | `plans/kiribati-tb-summer4-port.plan.md` | Whole | planned | — |
<!-- /roadmap:steps -->

`Closes` lists row IDs of {doc}`../evaluation/tb-ports` (and, where a step moves
one, {doc}`../evaluation/coverage-ledger`). A step with `—` closes no ledger row
but is still required by the steps after it.

## Corrections to committed plans

`plans/*.plan.md` are historical records and are never rewritten (see
`AGENTS.md`, *Plans land in the workspace on merge*). Everything that has since
changed about them is recorded here instead, and overrides them.

| Plan | Section | Correction |
| --- | --- | --- |
| `tb-ports-feature-completeness` | §12, all of it | Superseded by `plans/wp12-release.plan.md`. The merge target is `feat/model-stratify`, not `feat/epi-infection-mixing`; five stacked PRs are open that §12a did not know about |
| `tb-ports-feature-completeness` | WP3, all of it | **Applied** on `feat/initial-population`; the plan that shipped is `plans/initial-population.plan.md`. `KI3`, `TM6`, `L4`, `L5` and `S8` are already `full`. Port-order step 3 is a no-op — skip it |
| `tb-ports-feature-completeness` | §14c | `EpiModel` no longer exists (`plans/remove-epimodel.plan.md`). There are no `EpiModel.add_infection_*_flow` methods to extend. Builder sugar belongs on `ForceOfInfection` construction, used with `FlowModel`; `adjust=` already exists on `TransitionFlow` |
| `tb-ports-feature-completeness` | §13.3, §14e, §15e | Notebook numbers are stale: `examples/notebooks/` already reaches `12-model-stratification.ipynb`. Use the numbers in the step sections below |
| `tb-ports-feature-completeness` | §15a | Still accurate: `Trace` has no operators. The `D5` row's note in the coverage ledger already promises them in WP15; clear that note in step 10 |
| `wp12-release` | Step 2b | JAX has native Windows x86_64 CPU support; the Windows `jaxlib` wheel is experimental. Native Windows GPU is unsupported. WSL2 is the Windows GPU route, and that support is experimental. This repository's pixi platforms stay `osx-arm64` and `linux-64` |

---

## Step 1 — `chore/merge-flows-stack`

**Landed:** chore/merge-flows-stack, PR #7 (eceba1e) + PR #8 (769f9f1), 2026-09-18.

### Summary

This step puts the flows stack on `main`. Everything summer4 can do beyond the
compartment taxonomy — flows, results, solvers, time-varying functions, force of
infection, initial populations, stratification — currently lives on a tower of
five unmerged pull requests, so a clone of `main` cannot even
`import summer4.flows`. The step retargets PR #7 (`feat/model-stratify`, which
contains PRs #3, #4 and #6) onto `main`, merges it, then rebases and merges the
cookbook PR #8, and closes the superseded ones. It closes no ledger row by
itself — the tag in step 2 does that — but every later step and both downstream
ports are blocked until it lands.

### Read first

1. `AGENTS.md`
2. `plans/wp12-release.plan.md`, *Step 1*
3. `docs/evaluation/coverage-ledger.md`, *Delivery status*

### Do

Follow `plans/wp12-release.plan.md`, *Step 1*. In short: run the full check
suite on `feat/model-stratify` before touching anything; retarget and merge PR
#7 into `main` with a **merge commit** (not a squash — the ledger's landed-phase
table cites per-branch tips); rebase `docs/cookbook-custom-rates` onto `main`
and merge PR #8; close PRs #3, #4 and #6 with a comment naming the merge commit;
verify from a scratch clone that `main` imports `summer4.flows`.

Cut from: `feat/model-stratify`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus, from a clean clone of
`main` in a scratch directory:

```bash
python -c "import summer4; from summer4.flows import FlowModel; print(summer4.__version__)"
```

### Handoff

Set step 1 `done` with its *Landed* line; set step 2 `next` and rewrite the
*Current position* block; record under step 2 anything surprising about the
merge (conflicts resolved, tests that needed fixing, PRs that did not close
themselves).

---

## Step 2 — `chore/release-v0.2`

### Summary

This step makes summer4 installable by tag instead of by commit SHA. It fixes
the packaging first: `import summer4` already imports JAX, so the "NumPy only"
claim in the installation docs is false — JAX, jaxlib, diffrax and equinox move
into the core dependencies, and a new `frames` extra declares the polars and
pyarrow that `Trace.to_pandas` silently needs. Then it bumps the version to
`0.2.0a1`, tags it, and documents the git install honestly, including that JAX
has no native Windows support. It closes `KI23` and `TM9`, which unblocks the
tb_macro port (step 20).

### What the previous worker left you

- Stack merge was clean: PR #7 retargeted from `feat/remove-epimodel` to `main` and merge-committed; cookbook PR #8 rebased onto main with no conflicts (the intermediate merge commit dropped away; only `c679c3c`/`40550b9` cookbook commit remained).
- PR #3 auto-closed/merged with #7 (it already targeted `main`); #4 and #6 closed manually with supersession comments naming `eceba1e`.
- No stacked flows PRs remain open.
- Clean-clone import of `summer4.flows.FlowModel` works. `summer4.__version__` is missing as a module attribute; `importlib.metadata.version("summer4")` returns `0.1.0a0` — step 2 should expose version honestly when bumping to `0.2.0a1`.
- Delivery-status warning admonition was removed in this handoff (step 1), so step 2's delivery rewrite should not try to delete it again; still do the rest of 2c (tb-ports KI23/TM9, pointer polish).
- Tip was fully green (`pixi run` full suite) on `feat/model-stratify` before merge; no test fixes needed on the merge branch.

### Read first

1. `AGENTS.md`
2. `plans/wp12-release.plan.md`, *Step 2*
3. `pyproject.toml`, `docs/getting-started/installation.md`
4. `docs/evaluation/coverage-ledger.md`, *Delivery status*

### Do

Follow `plans/wp12-release.plan.md`, *Step 2*. The delivery-status rewrite is
part of this step, not a follow-up: the "The flows stack is not on `main`"
warning admonition is false once step 1 lands, and the landed-phases table stays
as history.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step). After tagging, the readiness
table in `docs/evaluation/tb-ports.md` must read 7 / 23 and 5 / 9 on its `today`
row.

### Handoff

Set step 2 `done`, step 3 `next`. Record the tag name and the packaging arm
actually chosen.

---

## Step 3 — `chore/downstream-smoke-ci`

### Summary

This step adds a CI job that installs summer4 the way a downstream repository
will: a scratch pixi project depending on the git tag, not an editable local
checkout. It then runs a three-compartment SIR through `compile` → `run` →
`Result`. The point is to catch packaging regressions — a missing dependency, a
module left out of the wheel — that a local editable install hides completely.
It closes no ledger row and ships no notebook.

### Read first

1. `AGENTS.md`
2. `plans/wp12-release.plan.md`, *Step 3*
3. `.github/workflows/` — the existing quick and full suites

### Do

Follow `plans/wp12-release.plan.md`, *Step 3* (`tb-ports-feature-completeness`
§12d is the original scope statement).

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus the new job green on the PR.

### Handoff

Set step 3 `done`, step 4 `next`. Phase A is finished; say so, and note that
step 20 (tb_macro port) is now unblocked and may run in parallel with phase B.

---

## Step 4 — `feat/rate-math`

### Summary

The rate tree can only add, subtract, multiply and divide. Both tuberculosis
models need more: `tanh` scale-ups, triangular seeds built from `abs` and
`clip`, `log`-derived interpolation knots, and `N ** exponent` for a generalised
force of infection. Today each of those forces a `Transform` or a `derived_fn`
callback, which blocks hoisting and obscures the parameter dependencies. This
step adds a `UnaryOp` node, extends `BinOp` with `pow`, `maximum` and `minimum`,
and exports the operators and dunders. It closes no row on its own — steps 5 and
6 do — but nothing else in phase B or C works without it.

### Read first

1. `AGENTS.md`, especially *JAX is the primary runtime target*
2. `plans/tb-ports-feature-completeness.plan.md` — *Before you start*, **the
   five-site rule**, and §13.1
3. `src/summer4/flows/rates.py`, `src/summer4/flows/compiled.py`
4. `futureplans/derived-fn-blocks-hoisting.md`

### Do

Follow §13.1. **The five-site rule is the thing that fails silently** — a new
node must be wired into the evaluator, `_flow_refs`, `_field_paths`,
`_rate_bytes` and the exports, and a missing `_rate_bytes` case makes two
distinct nodes collide in the jit cache.

Create `examples/notebooks/13-rate-math-and-tables.ipynb` here; steps 5 and 6
extend it.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step).

### Handoff

Set step 4 `done`, step 5 `next`. Record which operators landed and any that
were deferred.

---

## Step 5 — `feat/table-interp`

### Summary

`Interp` is scalar, so a per-age time series has to become one interpolation
tree per age band — Kiribati's eight death-rate series are about 1,200 knot
nodes written out longhand. This step adds `Data.table` and a `TableInterp` node
that evaluates a whole `(T, K)` table in one batched interpolation returning a
`GroupedRate`, plus a `Lookup` node that gathers a row out of an array carried in
the parameters, so a yearly mixing matrix can be built once per run instead of
rebuilt on every vector-field call. It closes `KI6`, `KI10`, `KI11` and `TM4`.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §13.2, and the five-site rule
3. `src/summer4/data.py`, `src/summer4/timevarying.py`
4. `futureplans/mixing-matrix-per-call-normalisation.md`

### Do

Follow §13.2. Verify with `jax.make_jaxpr` that the equation count does not grow
with the number of knots `T` or columns `K`; that is the whole point of the
node. Extend `examples/notebooks/13-rate-math-and-tables.ipynb`.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step). Move `KI6`, `KI10`, `KI11`,
`TM4` to `full` in `docs/evaluation/tb-ports.md`, set their `Closed by` to `—`,
update `Route today`, and run `pixi run coverage-write` in the same commit.

### Handoff

Set step 5 `done`, step 6 `next`. Record the jaxpr sizes you measured.

---

## Step 6 — `feat/ageing-sugar`

### Summary

Both models age people through uneven bands at a rate of one over the band
width, and both hand-write the chain of transitions to do it. This step adds
`TraitChain.from_breakpoints`, which reads the numeric lower bounds out of an
age property's traits and builds that chain. It is pure sugar: the acceptance
test is that it compiles to the same digest as the hand-written version. It
closes `KI2` and `TM3`, finishes WP13, and carries the blocking user sign-off
for the notebook that steps 4–6 built together.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §13.3
3. `src/summer4/flows/types.py` — `TraitChain`

### Do

Follow §13.3. Note the deliberate limit: `S6` (`AgeStratification`) stays
`partial` in the API ledger — a bundled stratification class is a summer2 shape
summer4 has rejected, not a gap this closes. Finish
`examples/notebooks/13-rate-math-and-tables.ipynb` and put all three subphases'
claims in the PR's user-gate checklist.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus `KI2` and `TM3` moved.
**User gate:** `13-rate-math-and-tables.ipynb`.

### Handoff

Set step 6 `done`, step 7 `next`. Re-read both ledgers and list, increment only,
which textbook chapters, summer2 pages and port phases WP13 unblocked.

---

## Step 7 — `feat/epi-generalised-foi`

### Summary

Kiribati's force of infection divides the infectious pool by population raised
to a calibrated exponent — a shape between frequency and density dependence that
summer4 cannot express, because the custom `kind` callable receives no
parameters. This step adds `kind="generalised"` with an `exponent`, built on the
`pow` node from step 4. The correctness bar is that the existing kinds become
special cases: frequency dependence must stay bit-identical to exponent 1, and
density to exponent 0.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §14a and §14d
3. `src/summer4/epi/infection.py`, `src/summer4/epi/mixing.py`
4. `futureplans/foi-multi-property-mixing.md`,
   `futureplans/foi-unstratified-dummy-pop.md`

### Do

Follow §14a. **Ignore §14c's `EpiModel` builder methods** — `EpiModel` was
removed; see [Corrections](#corrections-to-committed-plans). Raise if
`kind="generalised"` arrives without an exponent, or an exponent with any other
kind.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step).

### Handoff

Set step 7 `done`, step 8 `next`.

---

## Step 8 — `feat/epi-compartment-infectiousness`

### Summary

Infectiousness weights currently key only on the traits of the property the
force of infection groups by, so "subclinical cases are less infectious, and
nobody under 15 transmits" cannot be said — that is a weight per compartment
*and* age. This step accepts selector-keyed weights applied per compartment
before the group sum, keeping the existing trait-keyed form as sugar that
converts to it. It closes `KI4`, `KI5` and `TM5`, finishes WP14, and is the
machinery that step 16's susceptibility surface reuses.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §14b, §14d, §14e
3. `src/summer4/epi/infection.py`
4. `futureplans/foi-susceptibility-surface.md`

### Do

Follow §14b, §14d and §14e. Extend `examples/notebooks/09-epi-models.ipynb` with
the generalised-FOI section (steps 7 and 8 share one notebook section; this step
ships it).

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus `KI4`, `KI5`, `TM5` moved.
**User gate:** `09-epi-models.ipynb`.

### Handoff

Set step 8 `done`, step 9 `next`. Say explicitly whether the weight machinery
came out reusable for step 16's `susceptibility=`, since
`plans/wp17-foi-susceptibility.plan.md` assumes it did.

---

## Step 9 — `feat/trace-algebra`

### Summary

A `Trace` cannot be divided by another `Trace`, so every per-capita rate, every
percentage and every parameter-scaled output in Kiribati's ~150 derived outputs
would be hand-written array arithmetic. This step gives `Trace` arithmetic
operators with name-aligned broadcasting, a windowed `cumulative(start=)`, the
summer2 `midpoint()` flow-output convention needed for numerical parity, and a
`FlowMass` that sums several flows. It also folds in a deferred fix: the rolling
window currently builds a Python loop whose jaxpr grows with trajectory length.

### Read first

1. `AGENTS.md`, especially *JAX is the primary runtime target*
2. `plans/tb-ports-feature-completeness.plan.md` §15a, §15b, §15d
3. `src/summer4/results/trace.py`, `src/summer4/results/result.py`
4. `futureplans/trace-rolling-jaxpr.md`,
   `futureplans/state-ledgers-incidence.md`

### Do

Follow §15a, §15b and the `trace-rolling-jaxpr` half of §15d. Every operation
must work inside `jit`: index alignment at trace time, gathers and arithmetic
traced. Delete `futureplans/trace-rolling-jaxpr.md` and its bullet in
`futureplans/README.md` once it is folded in. Document `midpoint()` as a parity
convention, not exact incidence.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus a test that rolling jaxpr
size is flat in trajectory length.

### Handoff

Set step 9 `done`, step 10 `next`.

---

## Step 10 — `feat/output-sets`

### Summary

Kiribati declares about 150 named outputs, many defined in terms of others.
This step adds `OutputSet`: leaves are save specs, interior nodes are an
expression DAG over other named outputs and parameters, and the set contributes
only its leaves to the `SavePlan` so a calibration run never materialises dense
outputs it does not need. It adds `Result.to_frame` for the wide and long frames
the analyses expect, folds in the deferred target-residual reduction, and closes
`KI13` through `KI17` — the single biggest jump in Kiribati readiness, 13/23 to
18/23.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §15c, §15d, §15e
3. `src/summer4/results/`, `src/summer4/results/targets.py`
4. `futureplans/targetset-residual-reduction.md`

### Do

Follow §15c, the `targetset-residual-reduction` half of §15d, and §15e. Delete
that futureplans note and its README bullet. Clear the `D5` row's "`Trace` has
no operators yet; they arrive in WP15" note in the API ledger — it is no longer
true. Extend `examples/notebooks/06-flow-outputs.ipynb`.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus `KI13`–`KI17` moved.
**User gate:** `06-flow-outputs.ipynb`.

### Handoff

Set step 10 `done`, step 11 `next`. Harvest: list the textbook chapters and
summer2 pages this made cheaper to port (do not port them here).

---

## Step 11 — `feat/tb-scale-bench`

### Summary

Nothing measures a realistic model. The benchmarks cover the compartment
taxonomy only, so nobody knows what a 160-compartment, 185-year model costs to
compile or to run, or whether its jaxpr stays a sane size. This step builds a
synthetic model with Kiribati's *shape* and none of its data, and records
compile time, jaxpr equation counts, wall time per run and `vmap` over 64
parameter sets, for both the Euler and the adaptive solver. Those numbers become
the baseline that later regressions are measured against.

### Read first

1. `AGENTS.md`, *JAX is the primary runtime target*
2. `plans/tb-ports-feature-completeness.plan.md` §16a
3. `benchmarks/`, `docs/dev/benchmarking.md`, `docs/dev/performance.ipynb`

### Do

Follow §16a. The model must use what phases B–D landed: a generalised force of
infection over age with a `Lookup` mixing stack, per-age table death rates,
ageing from breakpoints, and an `OutputSet` of roughly 150 outputs. Record the
numbers in a `benchmarks/README` table — a benchmark whose results are not
written down is not a baseline.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step). The benchmark runs under
`pixi run test-all`, marked `slow`.

### Handoff

Set step 11 `done`, step 12 `next`. Put the headline numbers in the handoff, and
open a `futureplans/` note for anything that came out unexpectedly slow rather
than fixing it here.

---

## Step 12 — `feat/solver-safety`

### Summary

The adaptive solver runs with `throw=False` and a hard step ceiling of 4096, so
a 185-year run that exceeds it returns silently wrong trajectories — which, in a
calibration, become silently wrong posteriors. This step surfaces failure as
`SolverInfo.ok`, derives a sensible default step ceiling from the time span
instead of a magic number, and makes the mixing-matrix reciprocity check safe
under `vmap` (it currently calls `float()` on a traced value). It closes `KI22`
and is the last thing before calibration.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §16b, §16c
3. `src/summer4/solvers/diffrax_backend.py`, `src/summer4/epi/mixing.py`

### Do

Follow §16b and §16c. `SolverInfo.ok` is a traced boolean, not a Python one —
step 14 uses it to return an infinitely negative log-likelihood, which has to
work inside `jit`. Extend `examples/notebooks/05-solvers.ipynb` with a
deliberately too-small step ceiling.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus `KI22` moved.
**User gate:** `05-solvers.ipynb`.

### Handoff

Set step 12 `done`, step 13 `next`. Phase E is finished.

---

## Step 13 — `feat/epi-priors-likelihoods`

### Summary

Calibration targets already exist and already merge their observation times into
the save plan, but they carry no probability: `Target.dispersion` is declared and
never consumed. This step adds prior distributions, likelihoods (normal, Poisson,
negative binomial) attached to targets, and a traceable
`TargetSet.log_likelihood`. A likelihood's scale may itself be a prior, which is
how Kiribati's hierarchical target standard deviation works. It is the first of
three steps that together close WP10.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §10.1, and the separation rule
3. `src/summer4/results/targets.py`, `src/summer4/epi/`
4. `docs/case-studies/age-stratified-seirs.ipynb` — the hand-written loss this
   replaces

### Do

Follow §10.1. Calibration is epidemiological assembly, so it lives in
`summer4.epi`, and numpyro and optax stay in the `calibration` extra.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step).

### Handoff

Set step 13 `done`, step 14 `next`.

---

## Step 14 — `feat/epi-sampling`

### Summary

This step assembles the pieces into a `BayesianModel` that can be sampled: it
draws the priors, runs the model, evaluates the outputs and the likelihood, and
hands numpyro a log density. It offers NUTS where the model is differentiable
and numpyro's gradient-free ensemble samplers as the replacement for the pymc
sampler the Kiribati analysis used, plus a MAP fit through optax. Solver failure
from step 12 becomes an infinitely negative log density, so a diverged run can
never be mistaken for a good fit.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §10.2
3. `futureplans/wp10-preprocess-is-prepare-fn.md` — read before designing
   `preprocess=`
4. `docs/dev/run-stages.md`

### Do

Follow §10.2. Chains run vectorised by default so one compiled program serves
them all. The `preprocess` hook and the existing run-start `prepare_fn` overlap;
the futureplans note says how — resolve it explicitly rather than shipping two
ways to do the same thing.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step).

### Handoff

Set step 14 `done`, step 15 `next`. Record how the `preprocess` / `prepare_fn`
overlap was resolved.

---

## Step 15 — `feat/epi-posterior-runs`

### Summary

A calibrated model is only useful once you can run the posterior forward under
several scenarios and summarise the difference between them. This step adds
batched posterior runs with bounded memory, quantile frames, and the
averted-burden differences the tuberculosis analyses report, in the frame
schemas those analyses already use. It closes `KI18`–`KI21` and `TM8`, taking
both models to 100% of their capability rows, unblocks textbook chapter 20, and
ships the calibration notebook.

### Read first

1. `AGENTS.md`
2. `plans/tb-ports-feature-completeness.plan.md` §10.3, §10.4
3. `docs/evaluation/tb-ports.md` — the whole page; this step is what its verdict
   was waiting for

### Do

Follow §10.3 and §10.4. Ship `examples/notebooks/14-calibration.ipynb` and port
textbook chapter 20. Leave the hand-written loss in
`docs/case-studies/age-stratified-seirs.ipynb` alone unless the user asks — it
is a record of what calibration cost before WP10.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus `KI18`–`KI21` and `TM8`
moved. `docs/evaluation/tb-ports.md` must then read 23 of 23 and 9 of 9, with
the readiness table's `today` row equal to its `WP10` row. **User gate:**
`14-calibration.ipynb` and the chapter 20 port.

### Handoff

Set step 15 `done`, step 16 `next`. Phase F is finished; both port repositories
are now unblocked end to end.

---

## Step 16 — `feat/foi-susceptibility`

### Summary

The force of infection owns infectiousness weights but has no symmetric
susceptibility surface, so textbook chapter 15 — which is *about* that symmetry —
has to scale susceptibility with flow adjustments or by row-scaling a mixing
matrix, the very thing the chapter argues against. This step adds
`ForceOfInfection(susceptibility=...)` mirroring the infectiousness API, rewrites
the chapter onto it, and moves chapter 15 from `partial` to `full` — the last
textbook chapter outside the contact-survey block. It is independent of phases
B–F but is cheapest after step 8, whose selector-keyed weights it reuses.

### Read first

1. `AGENTS.md`
2. `plans/wp17-foi-susceptibility.plan.md`
3. `futureplans/foi-susceptibility-surface.md`
4. `src/summer4/epi/infection.py`,
   `docs/textbook/15-susceptibility-infectiousness-matrices.ipynb`

### Do

Follow `plans/wp17-foi-susceptibility.plan.md`. Susceptibility is not
normalised, unlike infectiousness — say so in the docstring and test it. Delete
`futureplans/foi-susceptibility-surface.md` and its README bullet.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus textbook row 15 at `full`
with its `Blocker` cleared. **User gate:** the rewritten chapter 15.

### Handoff

Set step 16 `done`, step 17 `next`.

---

## Step 17 — `feat/contact-survey-data`

### Summary

Textbook chapters 16 to 19 are the four chapters summer4 cannot publish at all:
they are built on empirical contact-survey matrices, and there is no way to load,
validate or inspect one. This step adds a `ContactMatrix` to `summer4.epi` with
constructors from arrays, frames and the per-setting stacks these datasets ship
in, validation that names the offending value when a matrix is not square or a
band is out of order, and inspection including reciprocity error. It carries one
open question that must be settled with the user before coding — where the survey
data itself comes from.

### Read first

1. `AGENTS.md`
2. `plans/wp9-contact-surveys.plan.md`, *Step 17* and *Open question*
3. `docs/evaluation/coverage-ledger.md` — the WP9 paragraph
4. `src/summer4/epi/mixing.py`, `src/summer4/data.py`
5. `futureplans/trace-plot-backend-coupling.md`

### Do

**Put the open question in the summary you present**, and get an answer before
writing the loaders: the plan proposes shipping no data — loaders plus a
synthetic fixture, with the notebooks embedding or fetching a small public
extract — but that is a proposal, not a decision. Then follow
`plans/wp9-contact-surveys.plan.md`, *Step 17*.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step).

### Handoff

Set step 17 `done`, step 18 `next`. **Record the answer to the data question**;
steps 18 and 19 depend on it.

---

## Step 18 — `feat/contact-matrix-adaptation`

### Summary

A survey matrix is never in the shape a model needs: its age bands differ, it is
not reciprocal against the modelled population, and it describes a different
country's demography. This step adds rebinning onto the model's age property,
reciprocity correction, adaptation to another population under both density and
frequency conventions, calibratable scaling for interventions, and conversion to
the `MixingMatrix` the force of infection already accepts. Everything must
survive `jit`, since intervention scaling is a calibration target.

### Read first

1. `AGENTS.md`, especially *JAX is the primary runtime target*
2. `plans/wp9-contact-surveys.plan.md`, *Step 18*
3. The handoff under this step, for the data decision from step 17
4. `futureplans/mixing-matrix-per-call-normalisation.md`

### Do

Follow `plans/wp9-contact-surveys.plan.md`, *Step 18*. Build constant index
arrays at trace time and check with `jax.make_jaxpr` that jaxpr size does not
grow with the number of age bands. Ship
`examples/notebooks/15-contact-matrices.ipynb`.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step). **User gate:**
`15-contact-matrices.ipynb`.

### Handoff

Set step 18 `done`, step 19 `next`.

---

## Step 19 — `docs/textbook-16-19`

### Summary

This step ports the four blocked textbook chapters — thinking about contact
surveys, understanding empiric contact data, implementing it, and adapting
mixing matrices — onto the API steps 17 and 18 built. It is documentation only,
but it is the proof that WP9 actually closed the gap: the chapters are the
acceptance test. It moves four textbook rows from `none` to `full`, which is the
largest single move left in the textbook ledger.

### Read first

1. `AGENTS.md`, especially *Documentation*
2. `plans/wp9-contact-surveys.plan.md`, *Step 19*
3. `plans/textbook-catchup.plan.md` — how earlier chapters were ported
4. `docs/evaluation/coverage-ledger.md`, textbook ledger

### Do

Follow `plans/wp9-contact-surveys.plan.md`, *Step 19*. Ported chapters carry the
source prose and figures with their BSD-2-Clause attribution, but every line of
code is written in current summer4 idiom — never a transliteration of summer2.
Every notebook on the site executes at build time, so `pixi run -e docs
docs-strict` is the real test here.

Cut from: `main`. Merges into: `main`.

### Exit checks

The [standard checks](#exit-checks-every-step), plus textbook rows 16–19 at
`full` and the totals quoted in `docs/evaluation/index.md` refreshed. **User
gate:** all four chapters.

### Handoff

Set step 19 `done`, step 20 `next`. Track G is finished; the only textbook row
still short of `full` should be none.

---

## Step 20 — tb_macro port (`monash-emu/tb-macro-summer4`)

### Summary

This step stands up the first downstream repository: a pixi project depending on
summer4 by tag, reimplementing the tuberculosis macroeconomics model, and
proving numerical parity against golden outputs generated by the original
library before anything else is judged. It can start as soon as step 2 has
tagged — the model is implementable with hand-written glue at that point — and
each later summer4 package becomes a swap in this repo that must keep the
goldens green.

### Read first

1. `AGENTS.md`
2. `plans/tb-macro-summer4-port.plan.md`
3. `docs/evaluation/tb-ports.md` — the `TM*` rows and *Limitations that stay in
   the ports*

### Do

Follow `plans/tb-macro-summer4-port.plan.md`. This work is in a **different
repository**; the only change in this one is the handoff below. Note the
duplicate detection flows called out in the ports page: they are kept for parity
and documented, not cleaned up.

Cut from: n/a. Merges into: n/a.

### Exit checks

The port repository's own parity suite, green against the pinned tag.

### Handoff

Commit the handoff in **this** repository: set step 20 `done`, step 21 `next`,
and record which summer4 APIs the port had to work around, so the next package
can absorb them.

---

## Step 21 — Kiribati port (`monash-emu/kiribati-tb-summer4`)

### Summary

This step stands up the second downstream repository — the 160-compartment,
185-year Kiribati TB screening model with its ~150 derived outputs, twelve
scenarios and full Bayesian calibration. Unlike tb_macro it is not portable
early: it needs everything through step 15. The bar is the same, numerical
parity against goldens from the original library, and then the original analyses
re-run on numpyro.

### Read first

1. `AGENTS.md`
2. `plans/kiribati-tb-summer4-port.plan.md`
3. `docs/evaluation/tb-ports.md` — the `KI*` rows and *Limitations that stay in
   the ports*

### Do

Follow `plans/kiribati-tb-summer4-port.plan.md`, phase gates included. Two
limitations belong to the port, not to summer4: spectral normalisation must use
the symmetric eigenvalue routine to stay differentiable, and the Python loop over
age-band pairs in the mixing builder must be vectorised into one matrix per year
before the solve.

Cut from: n/a. Merges into: n/a.

### Exit checks

The port repository's parity suite, green against the pinned tag, and the
original analyses reproduced.

### Handoff

Set step 21 `done`. There is no step 22: write the handoff as a short statement
of what the next tranche of work should be, and open a new planning session
rather than inventing steps here.

---

## Exit checks, every step

Unless a step says otherwise, run all of these and require all of them to pass:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage && pixi run roadmap
pixi run -e docs docs-strict
pixi run test-all
```

`check-branch` fails a branch that changed `src/summer4` without also changing
`tests/` and `examples/notebooks/`, or a feature branch with no plan under
`plans/`. `coverage` fails if a ledger status is misspelled or a quoted total is
stale. `roadmap` fails if this file's position blocks are inconsistent.

If you are blocked, **record the truth rather than working around it**: leave the
ledger row `partial`, name the real blocker in its `Route today`, write a
`futureplans/<slug>.md` note, and say so in the handoff.
