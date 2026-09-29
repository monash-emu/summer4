---
name: contact-matrix-adaptation
overview: Rebin, symmetrise, adapt and scale contact matrices, and hand them to the force of infection — WP9 / roadmap step 18.
todos:
  - id: api
    content: Rebin, ContactMatrix.rebin/symmetrise/adapt/scale/to_mixing, ContactAdaptation, ScaledContacts, SettingStack.rebin/symmetrise/adapt/scale/to_mixing
    status: completed
  - id: tests
    content: hand-computed rebin (aggregate, split, mixed), symmetrise, both adapt conventions, scale by float / Param / step, to_mixing in a FOI, jit + grad, jaxpr size vs band count
    status: completed
  - id: notebook
    content: examples/notebooks/26-contact-matrices.ipynb parts E–J
    status: completed
isProject: false
---

# Contact matrix adaptation (WP9 / step 18)

Follows `plans/wp9-contact-surveys.plan.md` *Step 18*, on top of step 17
(`plans/contact-survey-data.plan.md`). Data decision from step 17 stands: no
data ships; notebooks embed the POLYMOD extract; nothing is fetched at build
time. Closes no ledger row on its own; textbook rows 16–19 move in step 19.

## What ships (`src/summer4/epi/contacts.py`)

- **`Rebin`** — the static half of rebinning, reified: `Rebin.between(source,
  target, population=, population_bands=None)` builds two constant
  `(K_new, K_old)` matrices on the common refinement of the two band sets;
  `apply(values)` is `rows @ values @ cols.T` and accepts traced JAX arrays.
  `ContactMatrix.rebin(bands | prop, population=)` is sugar over it. Splitting
  assumes uniform contacts within a band (rows copy, columns split by
  population); aggregating is a population-weighted row mean and a column
  sum. Total contacts are conserved.
- **`symmetrise(population=None)`** — $c'_{ij} = (c_{ij}N_i + c_{ji}N_j)/(2N_i)$.
- **`adapt(population, kind="frequency" | "density", source=None)`** with a
  `ContactAdaptation` enum. Density: $c'_{ij} = c_{ij} r_j$ with $r_j$ the
  ratio of population *shares* (the textbook's chapter 19 method; keeps
  reciprocity). Frequency (default): density then each row rescaled to its
  original total.
- **`scale(by)`** — a number gives a new matrix / stack; a `RateOps` (`Param`,
  `step(Time(), ...)`) gives a **`ScaledContacts`**: terms `(setting, matrix,
  factor)`, `matrix()` builds one `(K, K)` rate expression (fixed terms folded
  into one `ArrayConst`, one multiply and one add per varying setting), and
  `to_mixing()` wraps it in a `MixingMatrix`.
- **`to_mixing(prop=None, normalize="none", check_reciprocal=False)`** on
  `ContactMatrix`, `SettingStack` and `ScaledContacts`.
- `SettingStack.rebin/symmetrise/adapt` apply per setting via `map`.

## Deviations from `wp9-contact-surveys` *Step 18*

- `scale(RateOps)` returns a `ScaledContacts` object rather than a bare rate
  expression, so it can still say which bands and property it belongs to and
  offer `to_mixing`; `ScaledContacts.matrix()` is the bare expression, and
  `MixingMatrix(age, scaled.matrix(), normalize="none")` is the composable
  spelling `to_mixing` stands for.
- `Rebin` is public (P2/P3: the static half is inspectable and reusable), and
  `population_bands=` lets a population on its own bands be spread onto the
  refinement.
- `adapt` uses population *shares*, so the absolute size of the target
  population does not matter (as in the textbook).
- The per-call normalisation concern (`futureplans/mixing-matrix-per-call-normalisation.md`)
  does not apply with `normalize="none"`; the note records how `ScaledContacts`
  interacts with hoisting and `check_reciprocal`.
