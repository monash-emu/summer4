# Remove `EpiModel`

## Context

`summer4.epi.EpiModel` (`src/summer4/epi/model.py`) is a summer2-shaped builder that
wraps `FlowModel` and `ForceOfInfection`. Every method on it is sugar: it creates
objects the user could create directly. The user has decided it should be **removed
outright**, not extended. The follow-up branch (`plans/model-stratify.plan.md`)
adds model-level stratification to `FlowModel` only.

Outcome: `summer4.epi` exports only `ForceOfInfection`, `MixingMatrix` and `Param`.
Every notebook, test and document that used `EpiModel` uses
`FlowModel` + `TransitionFlow(..., ForceOfInfection(...))` instead, and all
assertions and results are unchanged.

Read `AGENTS.md` before starting (branch rules, required checks, notebook rules).

## Branch

```bash
git fetch origin
git switch feat/initial-population
git switch -c feat/remove-epimodel
cp <this file> plans/remove-epimodel.plan.md   # already present if you are reading it from plans/
```

Do **not** edit historical plans under `plans/` that mention `EpiModel`
(`epi-infection-mixing`, `initial-population`, `kiribati-tb-summer4-port`,
`tb-macro-summer4-port`, `tb-ports-feature-completeness`, `textbook-catchup`). They
are immutable records. Ignore `.claude/worktrees/` and `.cursor/`.

## Translation table (use for every call site)

The `EpiModel` source is in `src/summer4/epi/model.py`; read it once. Each
`EpiModel` call maps to plain objects as follows:

| `EpiModel` code | Replacement |
|---|---|
| `m = EpiModel(pmap, infectious=SEL)` | `m = FlowModel(pmap)`. Keep `SEL` in a local variable and pass `infectious=SEL` to each `ForceOfInfection`. |
| `m.set_mixing_matrix(prop, K)` | `mixing = MixingMatrix(prop, K, check_reciprocal=False)` |
| `m.set_mixing_matrix(prop, K, normalize=N, check_reciprocal=C)` | `mixing = MixingMatrix(prop, K, normalize=N, check_reciprocal=C)` |
| `m.add_infectiousness_adjustments(prop, {"a": w, ...})` | `infectiousness = {prop["a"]: w, ...}` and `normalize_infectiousness = "population"` |
| `m.add_infectiousness_adjustments(prop, {...}, normalize=X)` | same, with `normalize_infectiousness = X` |
| `m.add_infection_frequency_flow(name, src, dest, rate)` | `m.add_flow(TransitionFlow(name, src, dest, ForceOfInfection(name, infectious=SEL, group_by=mixing.prop, mixing=mixing, kind="frequency", contact_rate=rate, infectiousness=..., normalize_infectiousness=...)))` |
| `m.add_infection_density_flow(...)` | same, with `kind="density"` |
| `..._flow(..., infectious=S2)` | `infectious=S2` on that FOI |
| `..._flow(..., group_by=G)` | `group_by=G` |
| `..._flow(..., mixing=M2)` | `mixing=M2`, `group_by=M2.prop` unless `group_by` was also passed |
| no mixing matrix set and no `group_by` | `EpiModel` raised `ValueError`; there is nothing to translate |
| `m.add_transition_flow(name, src, dest, rate, **kw)` | `m.add_flow(TransitionFlow(name, src, dest, rate, **kw))` |
| `m.add_flow(flow)` | unchanged |
| `m.flow_model` | `m` |
| `m.foi(name)` | the `ForceOfInfection` object you built; keep it in a variable |
| `m.set_initial_population(...)`, `m.compile(...)`, `m.pmap` | unchanged |

**Defaults that differ.** A literal translation that ignores these changes results.
1. `EpiModel.set_mixing_matrix` defaults to `check_reciprocal=False`.
   `MixingMatrix` defaults to `check_reciprocal=True`. Always pass
   `check_reciprocal=False` unless the original passed `True`.
2. `EpiModel.add_infectiousness_adjustments` defaults to `normalize="population"`.
   `ForceOfInfection` defaults to `normalize_infectiousness=None`. If the original
   called `add_infectiousness_adjustments` without `normalize=`, pass
   `normalize_infectiousness="population"`.
3. If the original never called `add_infectiousness_adjustments`, pass neither
   `infectiousness` nor `normalize_infectiousness`.
4. Infectiousness and mixing set on an `EpiModel` applied to **every infection flow
   added after them**. Pass them to each such FOI.

Proof that the table is exact: `tests/test_epi_model.py::test_epimodel_digest_equals_declarative`
asserts that the `EpiModel` build and the plain build compile to equal models.

## Steps

### 1. Source
- Delete `src/summer4/epi/model.py`.
- `src/summer4/epi/__init__.py`: remove the `EpiModel` import and its `__all__` entry.
  Keep `ForceOfInfection`, `MixingMatrix` and `Param`.
- `grep -rn EpiModel src/` must return nothing. Also check docstrings.

### 2. Tests
- `tests/test_epi_model.py`:
  - **Move** `test_param_is_field_ref` into `tests/test_epi_foi.py`. It does not test
    `EpiModel`.
  - Delete the rest of the file. It only tests `EpiModel`, and the declarative half of
    `test_epimodel_digest_equals_declarative` is already covered by `test_epi_foi.py`.
- `tests/test_hoisting.py::test_epi_unaffected_by_hoist` (~line 188): rewrite `build()`
  with the translation table. Remove the `EpiModel` import (line 29).
- `tests/test_initial_population.py` (~line 400): delete the `epi = EpiModel(...)`
  comparison block (the part that asserts `epi.compile() == flow.compile()`). Keep
  the `FlowModel` assertions above it. Remove the import (line 31).

### 3. Example notebook
`examples/notebooks/09-epi-models.ipynb`:
- Translate every cell with the table.
- Remove item 5 from the intro list ("Use the short `EpiModel` frontend") and delete
  the final "frontend"/escape-hatch section (cells mentioning `m.flow_model`,
  `m.foi(name)` and `frontend.foi(...)`).
- Keep every other assertion, unchanged.
- In the first section, replace "Here we use the summer2-shaped frontend" with a
  sentence saying the infection flow is a `TransitionFlow` whose rate is a
  `ForceOfInfection`.
- Notebooks must be committed with outputs cleared (`pixi run check-notebooks`).

### 4. Documentation notebooks (executed by the docs build)
Translate each of these with the table. Do not change any assertion or numeric input.

- `docs/summer2/`:
  - `01-basic-model`, `03-derived-outputs`, `04-flow-types`,
    `06-stratification-introduction`, `07-age-stratification`,
    `08-strain-stratification`, `09-mixing-matrices`, `12-concurrent-diseases`,
    `initial-population-graphobject`.
  - `01` and `07` use `epi.flow_model.add_flow(...)`; replace that with
    `model.add_flow(...)`.
- `docs/textbook/`: `02`, `03`, `05`, `06`, `08`, `09`, `10`, `11`, `12`, `13`, `14`,
  `15` (all `*.ipynb` that match `grep -l EpiModel`).

Where prose in these notebooks says "`EpiModel`", "frontend" or "summer2-shaped
builder", rewrite it to describe `FlowModel` + `ForceOfInfection`. The prose rule
(user memory "textbook-port fidelity"): keep the ported prose and attribution; only the
code is summer4 idiom.

Run `grep -rln EpiModel docs examples --include='*.ipynb' | grep -v _build`. It must
be empty.

### 5. Markdown documentation
| File | Change |
|---|---|
| `docs/index.md:18` | drop `EpiModel` from the list |
| `docs/user/07-from-summer2.md:35` | "`summer4.epi` (`ForceOfInfection`)". Add one sentence under the infection-flow translation: summer2's `add_infection_*_flow` becomes a `TransitionFlow` whose rate is a `ForceOfInfection`. |
| `docs/evaluation/coverage-ledger.md` | F7/F8 "summer4" column: remove `/ EpiModel.add_infection_*_flow`. Line ~204 (WP6 row) and line ~294 (WP6 prose): leave the historical "applied" wording, and append "(`EpiModel` later removed: `plans/remove-epimodel.plan.md`)". **No status changes.** |
| `docs/evaluation/feature-completeness.md:70,140` | remove the `EpiModel` mentions |
| `docs/evaluation/gaps.md:68,82` | replace with `ForceOfInfection(kind=...)` / `ForceOfInfection(infectiousness=...)` |
| `docs/evaluation/user-satisfaction.md:28` | "`ForceOfInfection` + `TransitionFlow`" |
| `docs/textbook/roadmap.md:46` | remove "an `EpiModel` frontend" |
| `futureplans/epimodel-unstratified-dummy-pop.md` | The dummy `pop` property issue still exists for `ForceOfInfection.group_by`. Rename the file to `futureplans/foi-unstratified-dummy-pop.md` (`git mv`), change the title and body to talk about `ForceOfInfection` only, and update its link in `futureplans/README.md`. |
| `futureplans/foi-susceptibility-surface.md:5,28` | drop the `EpiModel` API proposal; keep the `ForceOfInfection(susceptibility=...)` direction |
| `futureplans/next-generation-matrix-r0.md:21` | drop "(or `EpiModel` method)" |

Final check: `grep -rn EpiModel --exclude-dir=_build --exclude-dir=.claude --exclude-dir=.cursor --exclude-dir=plans --exclude-dir=.git .`
Its only matches should be the three ledger/roadmap sentences that point at this plan.

### 6. Ledger checks
```bash
pixi run coverage-write
pixi run coverage
```
Totals must be unchanged, because no status changed.

## Acceptance criteria
- `from summer4.epi import EpiModel` raises `ImportError`.
- No test assertion was weakened. The only deleted tests are ones that only exercised
  `EpiModel`, and `test_param_is_field_ref` has moved, not been deleted.
- Every translated notebook keeps its original assertions and passes.
- All required checks pass:

```bash
pixi run lint
pixi run format-check
pixi run check-notebooks
pixi run test
pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
```

- **Manual gate:** the user signs off on `examples/notebooks/09-epi-models.ipynb`
  before merge.

## Commit / PR
One commit per step group (source + tests; example notebook; docs notebooks;
markdown + ledger). The PR description quotes ledger IDs F7, F8 and M1 and says
"no status changes".
