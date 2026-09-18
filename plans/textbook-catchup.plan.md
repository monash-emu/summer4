---
name: textbook-catchup
description: The post-WP6 verification sweep — port every textbook chapter and summer2 documentation page that WP5 and WP6 unblock, and refresh the evaluation pages that drifted since WP2 and WP7 landed.
---

# Catch-up — port everything WP5 and WP6 unblock

## Context

`docs/evaluation/coverage-ledger.md` measures two things that move
independently: whether summer4 *can* do something (the API ledger) and whether a
reader can *see* it done (the textbook and summer2docs ledgers, with their
`Ported` cells). Feature packages move the first. Only a porting sweep moves the
second, and the ledger prints the gap on every run as its *publishable, not yet
ported* line — six rows at the time this plan was written, before WP5 and WP6
add more.

This sweep is also how WP5 and WP6 get **verified against real material** rather
than against their own notebooks. A force-of-infection API that passes its tests
but cannot express chapter 14 has not actually landed.

**Closes: no API rows.** This plan moves textbook and summer2docs rows only, and
the percentage quoted in `docs/evaluation/index.md` does not change. Say so in
the PRs rather than implying progress.

**Depends on:** `plans/time-varying.plan.md` (WP5) and
`plans/epi-infection-mixing.plan.md` (WP6), both complete. Start from 44 / 52.

| # | Branch | Scope |
|---|---|---|
| C1 | `docs/textbook-backlog` | Chapters already `full` but never ported: 3, 5, 6, 9, 11 |
| C2 | `docs/textbook-unblocked` | Chapters WP4/WP5/WP6 unblock: 4, 8, 10, 12, 14, 15 |
| C3 | `docs/summer2-pages` | summer2 documentation pages into `docs/summer2/` |
| C4 | `docs/evaluation-refresh` | The evaluation pages that drifted since WP2 and WP7 |

C1–C3 are docs-only branches: they change no `src/summer4/*.py`, so
`check-branch` passes without a notebook under `examples/`. If a port turns out to
need an API change, **stop and open a separate feature branch for it** — do not
smuggle source changes into a porting branch.

---

## Before you start

Read `AGENTS.md`, `docs/evaluation/coverage-ledger.md` (especially *Delivery
status*), `docs/textbook/porting.md` and `docs/textbook/roadmap.md`. Branch from
the tip of WP6, not from `main`.

### The porting convention, restated because it is easy to get wrong

From `docs/textbook/porting.md`:

- **Licence and attribution.** The source textbook is BSD-2-Clause, Copyright (c)
  2022, monash-emu. Every ported chapter carries that notice, names
  `monash-emu/summer-textbook`, and **pins the commit SHA** the prose was adapted
  from. Figures are vendored under `docs/textbook/figures/<chapter>/` with their
  licence in `docs/textbook/figures/LICENSE`.
- **Carry the prose. Rewrite the code.** Adapt the source prose and figures, but
  **write every line of code in current summer4 idiom** — `from summer4 import
  ...`, `EpiModel` or `FlowModel`, `CompiledModel.run`, `SavePlan`, `Result`.
  Never transliterate a summer2 or summer API call.
- **Plot with Plotly** through `Trace.to_pandas()`. Do not use `Trace.plot`: it
  hardcodes matplotlib, forwards backend-specific kwargs, and raises under the
  Plotly backend (`futureplans/trace-plot-backend-coupling.md`).
- **Do not describe planned behaviour in the present tense.** Where a capability
  has no summer4 equivalent, say so and cite the ledger row.

### Landing and the honesty rule

Every branch ends with:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
```

`pixi run -e docs docs-strict` executes every notebook on the site with
`nb_execution_raise_on_error`, so a docs build is a test run. A ported chapter
that does not execute does not land.

Fill the `Ported` cell with the path **relative to `docs/`**. Declared `Ported`
paths must exist or `pixi run coverage` fails
(`tests/test_coverage_ledger.py::test_declared_ports_exist`).

**The honesty rule.** Attempt every port in scope. If a genuine blocker appears,
leave the row `partial` or `none` with the *real* blocker named and the `Ported`
cell empty. Do not force a port, and do not quietly move a status the code does
not support. Recording the truth is what the ledger is for — and the ledger
already carries a case where a wrong conclusion survived into a written plan
before it was caught.

---

## C1 — `docs/textbook-backlog`

Chapters already marked `full` with an empty `Ported` cell. These need no new
API; they are pure backlog, and `pixi run coverage` has been printing them as
*publishable, not yet ported* since WP2.

| Ch | Title | Ledger today |
|---|---|---|
| 3 | Thinking about flows | `full`, unported |
| 5 | Series compartments and latency | `full`, unported |
| 6 | Post-infection immunity | `full`, unported |
| 9 | Transmission assumptions | `full`, unported |
| 11 | Cyclical epidemic dynamics | `full`, unported |

Port to `docs/textbook/0N-<slug>.ipynb`, matching the naming of the three
chapters already there (`01-introduction.md`, `02-model-structures.ipynb`,
`07-numerical-solutions.ipynb`). Add each to the toctree in
`docs/textbook/index.md`.

Chapter 9 (*Transmission assumptions*) is the one to write **after** WP6, not
before: it is about frequency versus density dependence, and it should use
`ForceOfInfection(kind=...)` rather than hand-written coupling. If it reads
awkwardly through the new API, that is a finding about the API — record it in
`futureplans/` rather than working around it silently.

**Ledger:** fill five `Ported` cells. No `Status` changes. Run
`pixi run coverage` and confirm the *publishable, not yet ported* count drops by
five.

---

## C2 — `docs/textbook-unblocked`

Chapters whose blocker a landed package has now removed.

| Ch | Title | Was blocked on | Now |
|---|---|---|---|
| 4 | Thinking about flow rates | Port not written; API ready since WP4 | Write it |
| 8 | Derived outputs | Port not written; API ready since WP4 | Write it |
| 10 | The reproduction number | Time-varying parameters for $R_t$ | WP5 |
| 12 | Heterogeneous mixing introduction | Mixing matrices | WP6 |
| 14 | Assortative mixing | Mixing matrices, infectiousness adjustments | WP6 |
| 15 | Susceptibility and infectiousness matrices | Mixing matrices, infectiousness adjustments | WP6 |

**Chapter 13 is deliberately absent.** Its ledger blocker reads "Mixing matrices,
population split". WP6 supplies the first; **population split is `S8`, which is
WP3 and is not in scope.** Leave chapter 13 `none` with its blocker narrowed to
population split alone, and say so in the PR. Attempting it and half-landing it
would be worse than leaving it honest.

Chapter 10 should already have been attempted in WP5 §5.5. If it was left
`partial` there, this branch is the second attempt now that WP6 has landed —
$R_t$ for a mixed population is an eigenvalue of the next-generation matrix, and
whether summer4 can express that cleanly is a real question. If it cannot,
`partial` with the honest blocker is the right outcome and a `futureplans/` note
should say what would be needed.

Chapters 14 and 15 are the hardest test of WP6's infectiousness surface, because
they are about susceptibility *and* infectiousness matrices. If `A4` as landed
covers infectiousness but not susceptibility, that is a finding: record it, leave
the row `partial`, and name it.

**Ledger:** move 4, 8, 10, 12, 14, 15 to `full` where the port actually landed,
each with its `Ported` path. Narrow chapter 13's blocker.

---

## C3 — `docs/summer2-pages`

Into the `docs/summer2/` tree that WP5 §5.5 created, with its index carrying the
same attribution block as `docs/textbook/`.

**Portable now:**

| Page | Was blocked on | Now |
|---|---|---|
| `examples/11-flows-between-strata` | Nothing — `full`, unported | Backlog |
| `examples/03-derived-outputs` | Port not written; API ready | Write it |
| `examples/04-flow-types` | Port not written; API ready | Write it |
| `examples/10-derived-outputs-stratified` | Port not written; API ready | Write it |
| `examples/06-stratification-introduction` | Infectiousness adjustments | WP6 (`A4`) |
| `examples/08-strain-stratification` | Strain-aware FOI primitive | WP6 (6.5) |
| `examples/09-mixing-matrices` | Mixing matrices | WP6 (`M1`) |

**Not portable, and why — leave these alone:**

| Page | Blocker | Package |
|---|---|---|
| `examples/01-basic-model` | Initial population | WP3 |
| `examples/07-age-stratification` | Population split | WP3 |
| `detailed/InitialPopulationGraphobject` | Initial population, parameters | WP3 |

`detailed/time-varying-functions` was ported in WP5 §5.5; do not redo it.

`examples/08-strain-stratification` is the page that proves 6.5. If it can be
written without a `StrainStratification`-shaped convenience class, the ledger's
long-standing decision to reject that shape (`S7`) is vindicated in public. If it
cannot, that is a genuine finding about the decision, not a reason to quietly add
the class — write it up in `futureplans/` and raise it.

**Ledger:** seven summer2docs rows to `full` with `Ported` paths, where the port
landed. Three rows stay as they are with their WP3 blockers intact.

---

## C4 — `docs/evaluation-refresh`

`pixi run coverage` validates only the totals quoted in
`docs/evaluation/index.md` — the literal `**{n} of {total}` string. Everything
else on the evaluation pages is unchecked prose, and it has drifted. Three pages
still describe a package that cannot plot a run, which has been false since WP2
landed:

| File | Line | Says |
|---|---|---|
| `docs/evaluation/feature-completeness.md` | 30 | "`euler(...)` — final state only, no trajectory" |
| `docs/evaluation/user-satisfaction.md` | 29 | "Run the model \| **Partial** — `euler` returns the final state, not a trajectory" |
| `docs/evaluation/user-satisfaction.md` | 45 | "31 of 52 symbols have any working route" |
| `docs/evaluation/gaps.md` | 34 | "`euler` returns the final state only." |

Rewrite them against what actually ships:

- **`feature-completeness.md`** — the symbol-by-symbol comparison table, and the
  "layers of the intended stack" section. Flows, results, solvers, targets, and
  now time-varying functions and the epi layer all exist.
- **`user-satisfaction.md`** — the modeller walkthrough (nine steps, four of
  which were marked blocked and are no longer) and the summer2-migration
  walkthrough, whose honest answer was "you can compile a model; you cannot yet
  publish a run". Keep the page's framing admonition exactly as it is: there is
  still no user research behind it and it must not start pretending otherwise.
  The one new piece of evidence is the notebook sign-offs from WP5 and WP6 —
  cite them as what they are, a small number of expert walkthroughs.
- **`gaps.md`** — §1.2 (time-varying library, now WP5), §1.3 (the solver seam),
  §2.1 (force of infection) and §2.2 (mixing matrices) all need their **State**
  paragraphs rewritten. Sections whose blocker is WP3 or WP9 stay as they are.
- **`docs-coverage.md`** — re-derive it from the ledger's textbook and summer2docs
  tables after C1–C3, rather than editing it by hand.
- **`docs/index.md`** — its "What is missing, in one paragraph" section and its
  "Documentation streams" table both go stale when a package lands. Check both.
- **`docs/user/07-from-summer2.md`** — its "not yet in summer4" table lists
  infection primitives, infectiousness, mixing and interpolation helpers. All
  four now exist.

Audit the whole of `docs/evaluation/` rather than only the four lines named
above; those are the ones found by a single grep for `of 52` and `final state`,
not a guarantee that nothing else drifted.

**Also close the loop on the ledger's own planning table.** In the *Delivery
status* section, add the C1–C4 branches to the *Landed phases* table and update
the *Planning status of the remaining packages* table so WP3, WP9 and WP10 remain
correctly listed as having no detailed plan.

**Ledger:** no rows change in C4. It exists because the ledger's tables are
machine-checked and its neighbours are not.

> ### Notebook gate — blocking
>
> Every ported page is a user-verification artefact; that is the point of the
> sweep. Do not merge C1–C3 until the user has run each ported page in
> `pixi run notebook` and ticked:
>
> - [ ] each chapter and page reads as its source and runs.
> - [ ] the code is current summer4 idiom throughout, with no transliterated
>       summer2 calls.
> - [ ] the licence notice and pinned source SHA are present on every port.
> - [ ] any row left `partial` or `none` names a blocker the user agrees is real.
> - [ ] where a port was awkward to write, the awkwardness is filed in
>       `futureplans/` rather than absorbed silently.

---

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
```

The sweep is complete when:

- `pixi run coverage` prints a *publishable, not yet ported* count of **zero**,
  or every remaining entry has a named, agreed blocker.
- Every declared `Ported` path exists under `docs/` — the checker enforces this,
  but check it before you push rather than in CI.
- `pixi run -e docs docs-strict` executes every new notebook without error.
- No evaluation page contradicts the API ledger. Grep for `of 52`, `final state`,
  `hand-written`, `blocked` and `not yet` across `docs/` and read each hit.
- The API totals in `docs/evaluation/index.md` are **unchanged** at 44 / 52. This
  plan moves no API rows, and a changed total means something went wrong.

After this sweep the remaining distance to the published 47 / 52 ceiling is WP3
(initial population: `L4` `L5` `S8`), followed by WP9 (contact survey data,
textbook 16–19) and WP10 (calibration, textbook 20). None of the three has a
detailed plan yet; see the *Delivery status* section of the coverage ledger.
