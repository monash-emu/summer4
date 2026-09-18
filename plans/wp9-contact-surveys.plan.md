---
name: wp9-contact-surveys
description: Load, validate, rebin, correct and adapt empirical contact-survey matrices, then port textbook chapters 16 to 19 — steps 17 to 19 of docs/dev/roadmap.md.
---

# WP9 — contact survey data (steps 17–19)

## Context

Four textbook chapters are `none` in the coverage ledger — 16 *Thinking about
contact surveys*, 17 *Understanding empiric contact data*, 18 *Implementing
empiric survey data*, 19 *Adapting mixing matrices* — and they are the only
chapters outside chapter 20 that summer4 cannot publish at all. They are the
largest remaining block in the textbook ledger.

The blocker is not modelling machinery: `MixingMatrix` and `ForceOfInfection`
landed in WP6, and a user who already has a square matrix aligned to their age
property can use it today. What is missing is everything *around* the matrix.
Empirical survey data arrives with its own age bands, as a stack of per-setting
matrices, from a different country's demography, and not reciprocal against any
population a model will use. Turning that into a `MixingMatrix` is four or five
careful transformations that every user would otherwise write by hand, get
subtly wrong, and be unable to calibrate through.

Until now WP9 has been **a paragraph in the coverage ledger and nothing else** —
the ledger says so explicitly in its *Planning status* table. This is that plan.

**Closes (API ledger):** no API rows — WP9 declares none. It moves textbook rows
16, 17, 18 and 19 from `none` to `full`. **Closes (ports ledger):** nothing;
neither tuberculosis model uses survey data directly, though Kiribati's mixing
builder is the same shape of problem. **Depends on:** WP6 (applied). Independent
of WP13–WP16 and WP10, so it may run at any time after step 1.

Read `AGENTS.md` first, then `docs/evaluation/coverage-ledger.md` (the textbook
ledger and the WP9 paragraph), then `src/summer4/epi/mixing.py`.

## The separation rule

Binding, unchanged from WP6: generic machinery goes in core (`summer4.flows`,
`summer4.results`, `summer4.data`); epidemiological assembly goes in
`summer4.epi`. A contact matrix is epidemiological assembly, so **it lives in a
new `src/summer4/epi/contacts.py`**. If a genuinely generic table loader is
needed, it goes in `summer4.data` and `contacts.py` calls it — do not put
"read a CSV of numbers" in `summer4.epi`.

## Open question — settle this before writing any code

**Where does the survey data come from?** Chapters 17 and 18 are built on real
datasets. Three options, and the user decides:

1. **Ship no data** (this plan's proposal). summer4 ships loaders plus a small
   synthetic fixture under `tests/helpers/`; the notebooks embed a small public
   extract inline, or fetch one at build time. Keeps the package small and
   sidesteps redistribution licensing, at the cost of a docs build that either
   carries a hard-coded extract or depends on the network.
2. Ship a small curated dataset inside the package, with its licence and
   attribution, as `summer4.epi.contacts.datasets`.
3. Depend on an external data package and document the install.

Every docs notebook executes at build time (`nb_execution_raise_on_error`), so
option 1's "fetch at build time" arm makes the docs build network-dependent —
say that out loud when presenting the choice. **Present this question in step
17's opening summary and get an answer before writing the loaders.** Record the
answer in the step 17 handoff; steps 18 and 19 both depend on it.

---

## Step 17 — `feat/contact-survey-data`

### 17a. The object — `src/summer4/epi/contacts.py` (new)

```python
@dataclass(frozen=True, slots=True)
class ContactMatrix:
    """Contacts per person per unit time, from group i with group j."""

    values: NDArray[np.float64]        # (K, K), row i = contacts reported BY group i
    bands: tuple[float, ...]           # K + 1 strictly increasing band edges
    prop: Property | None = None       # the age property, when already aligned
    setting: str | None = None         # "home", "work", "school", "other", None = all
    source_population: NDArray[np.float64] | None = None   # (K,), the survey's own
```

Orientation is the thing every implementation gets wrong, so **fix it once and
state it in the docstring**: `values[i, j]` is the average number of contacts a
member of group `i` has with members of group `j`, per unit time. Every method
below states which convention it assumes, and every test asserts against a
hand-computed asymmetric example, never a symmetric one (a symmetric fixture
cannot catch a transposition).

### 17b. Constructors

```python
ContactMatrix.from_array(values, bands, *, setting=None, source_population=None)
ContactMatrix.from_frame(df, *, row="age_from", col="age_to", value="contacts", bands=None)
ContactMatrix.from_settings({"home": cm_home, "work": cm_work, ...}) -> SettingStack
```

- `from_frame` accepts a polars or pandas long frame (the `frames` extra from
  WP12 declares polars). Infer `bands` from the sorted unique row labels when
  they parse as `"[0,5)"` / `"0-4"` / a numeric lower bound; raise naming the
  unparseable label otherwise.
- `SettingStack` is a frozen mapping of setting name to `ContactMatrix`, all
  sharing bands, with `.total()` summing them and `.__getitem__` returning one.
  Per-setting stacks are how these datasets ship, and chapter 19 scales settings
  independently (schools closed, workplaces at 50%).

### 17c. Validation

One `_validate` used by every constructor. Each failure raises `ValueError`
naming the offending value, not just the rule:

| Check | Message must name |
|---|---|
| square | the actual shape |
| `len(bands) == K + 1` | both counts |
| bands strictly increasing | the first non-increasing pair |
| finite and non-negative | the index and value of the first offender |
| `prop` given: `len(prop.traits) == K` | both counts and the property name |
| `source_population` given: same length, positive | the index |

`prop` alignment is by **position**, and the docstring says so: trait order is
band order. A trait-name check (traits parsing as the band lower bounds, as in
`TraitChain.from_breakpoints`) is a warning-free assertion only when the traits
happen to be numeric — do not require numeric trait names.

### 17d. Inspection

- `.total()` — contacts per person per group, `values.sum(axis=1)`.
- `.reciprocity_error(population)` — the matrix of
  $c_{ij} N_i - c_{ji} N_j$, plus a scalar relative norm. Reciprocity is the
  property chapter 17 spends its length on; make it measurable before making it
  fixable.
- `.mean_contacts(population)` — population-weighted mean, the number a survey
  usually reports and the first thing a reader checks.
- `.plot(ax=None)` — a heatmap. **Go through whatever plotting seam exists; do
  not add a second hard-coded matplotlib import.** Read
  `futureplans/trace-plot-backend-coupling.md` first: `Trace.plot` already made
  this mistake, and the note asks for the coupling to be reduced, not repeated.
  If no seam exists yet, keep `.plot` to a handful of lines and add a
  `futureplans/` note rather than inventing an abstraction here.

All of 17a–17d is host-side NumPy. Nothing in this step needs to be traceable;
`jit` matters in step 18.

### 17e. Tests — `tests/test_contact_matrices.py` (new)

Construction from array and frame; every validation message; an asymmetric
3 × 3 fixture with hand-computed `total`, `mean_contacts` and
`reciprocity_error`; a setting stack whose `total()` equals the element-wise sum;
round trip `from_frame(to_frame(cm)) == cm`. The synthetic fixture lives in
`tests/helpers/` so step 18 and the notebook can reuse it.

### 17f. Notebook

None in this step — step 18 ships
`examples/notebooks/15-contact-matrices.ipynb` covering both. If
`pixi run check-branch` demands a notebook because `src/summer4` changed, add the
loading section of that notebook here and extend it in step 18.

---

## Step 18 — `feat/contact-matrix-adaptation`

Everything a survey matrix needs before a model can use it. Each transformation
is a method returning a new `ContactMatrix` (frozen dataclass, so never mutate).

### 18a. `rebin(bands | prop, *, population)`

Map the survey's bands onto the model's. Aggregating rows is a
population-weighted mean over the source groups; aggregating columns is a plain
sum. Splitting a band assumes contacts are uniform within it — **say so in the
docstring and raise if `population` is absent**, because the weights are what
make the result meaningful.

Build the aggregation as two constant `(K_new, K_old)` matrices at construction
and apply them as two matrix products, not a Python double loop: this is the
operation most likely to be called inside a calibration.

### 18b. `symmetrise(population)`

Enforce $c_{ij} N_i = c_{ji} N_j$ by the standard average,
$c'_{ij} = (c_{ij} N_i + c_{ji} N_j) / (2 N_i)$. Test that
`reciprocity_error` afterwards is zero to floating-point tolerance, and that
total contacts are conserved.

### 18c. `adapt(population, *, kind="frequency" | "density")`

Chapter 19's subject: use a matrix collected in one demography with another.

- `"frequency"` — contact *numbers* per person are preserved and only the
  partner distribution shifts with the target population.
- `"density"` — contact rates scale with the target group's size.

Both conventions are in the literature and they disagree; implement both,
document the difference in one short paragraph with the formula, default to
`"frequency"`, and test each against a hand-written expected matrix.

### 18d. `scale(by, *, settings=None)`

Multiply a matrix, or named settings of a `SettingStack`, by a factor. **`by`
accepts a `RateOps`**, not only a float, so "schools at 20% from March 2020" is
a `step(...)` or `sigmoidal(...)` expression and is therefore calibratable. When
`by` is a plain number the result is a new `ContactMatrix`; when it is a
`RateOps` the result is a rate expression suitable for `to_mixing`.

### 18e. `to_mixing(prop, *, normalize="none")`

Return a `MixingMatrix` (`src/summer4/epi/mixing.py`). Two notes that matter:

- **Default `normalize="none"`, not `"rows"`.** An empirical matrix's row sums
  *are* the contact numbers; row-normalising throws away the thing the survey
  measured. `MixingMatrix`'s own default is `"rows"`, so pass it explicitly and
  explain the difference in the docstring.
- When any scaling is a `RateOps`, the matrix handed to `MixingMatrix` is an
  expression, which `MixingMatrix` already accepts (it takes a `FieldRef` or a
  `RateOps`). Check `check_reciprocal`'s behaviour under that path against
  `futureplans/mixing-matrix-per-call-normalisation.md` — the per-call
  normalisation concern recorded there applies directly.

### 18f. JAX discipline

Per `AGENTS.md`: the time-varying path must survive `jit`. Build every index and
aggregation array at **trace time** as a constant, then do one batched product.
Add a test using `jax.make_jaxpr` that the equation count of a scaled,
time-varying matrix expression does not grow with the number of age bands.

### 18g. Tests and notebook

Extend `tests/test_contact_matrices.py`: rebin against a hand-computed
aggregation and a hand-computed split; symmetrise to zero error; both adapt
conventions; `scale` with a float and with a `step` expression; `to_mixing`
inside a small age-stratified `ForceOfInfection` reproducing a hand-written
matrix; the jaxpr-size test.

Ship `examples/notebooks/15-contact-matrices.ipynb`: load a survey stack,
inspect it, rebin onto a model's age property, symmetrise, adapt to another
population, close schools from a date, and run an SEIR with the result. It
asserts its claims, per the feature acceptance bar. **User gate.**

---

## Step 19 — `docs/textbook-16-19`

Port the four chapters. Documentation only; no `src/summer4` change, so
`check-branch` needs no notebook.

Follow `plans/textbook-catchup.plan.md` for how earlier chapters were ported.
The rules that matter:

- Carry the source prose and figures with their BSD-2-Clause attribution, but
  **write every line of code in current summer4 idiom** — a transliteration of
  summer2 is not a port.
- Do not describe planned behaviour in the present tense. Where a chapter's
  original approach has no summer4 equivalent, say so and cite the ledger.
- Chapter 19 is the one at risk of over-claiming: it adapts matrices for
  interventions and for other populations, which is exactly step 18's surface.
  If any part of the chapter cannot be written, leave that textbook row
  `partial` with the real blocker named, and write a `futureplans/` note.

Then, in the same commit: move textbook rows 16–19 to `full`, clear their
`Blocker` cells, fill their `Ported` paths, refresh the textbook totals quoted in
`docs/evaluation/index.md`, and run `pixi run coverage-write && pixi run
coverage`. **User gate:** all four chapters.

---

## Verification (whole package)

- `pixi run -e docs docs-strict` builds with all four chapters executing.
- `pixi run coverage` green, textbook ledger showing 16–19 at `full`.
- A survey matrix loaded from a frame, rebinned, symmetrised and adapted,
  drives a `ForceOfInfection` inside `jit` with a jaxpr whose size is
  independent of band count.
- The data question from step 17 is answered in the runbook's handoff, and the
  answer is consistent with how the notebooks actually get their data.

## Deliberate non-goals

- **No contact-survey *analysis*.** summer4 consumes matrices; it does not fit
  them from participant-level diary data.
- **No bundled `AgeStratification`-style convenience class.** Alignment is by
  property and position, consistent with `S6` staying a rejected shape.
- **No socialmixr dependency.** Loaders accept what that ecosystem exports;
  summer4 does not depend on it.
