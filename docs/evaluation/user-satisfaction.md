# User satisfaction

```{admonition} What this is, and is not
:class: important

There is no user research behind this page. No survey, no interviews, no
telemetry, no issue tracker history. What follows is a **heuristic evaluation**:
task walkthroughs for three user types, and an expert review of the API surface
that exists. Treat it as a structured opinion, not as evidence about real users.

If the project wants a defensible answer to "are users satisfied?", it needs
users first — which is itself the finding.
```

## Task walkthroughs

### The modeller

*Wants: build an age-stratified SEIR model, run it, plot incidence, fit it to
case data.*

| Step | Outcome |
|---|---|
| Declare S, E, I, R | Works |
| Stratify by age | Works |
| Add severity only to the infectious | Works, and better than summer2 |
| Set the initial population | **Blocked** — build `y0` by hand with `select` |
| Add infection, progression, recovery flows | Works (`EpiModel` / `ForceOfInfection`; progression via `TransitionFlow`) |
| Run the model | Works — `CompiledModel.run` returns a `Result` trajectory |
| Request incidence by age | Works — `FlowMass` / `Trace` edge queries |
| Plot | Works — `Trace.to_pandas()` then Plotly (pandas plotting backend) |
| Calibrate | **Partial** — sparse `Target` / `TargetSet` fits exist; Bayesian workflow (WP10) does not |

Seven of nine steps complete, one partial, one blocked. The modeller can build,
run, query and plot an epidemic model; initial population remains manual, and
full Bayesian calibration is still ahead. For this user, summer4 is usable for
many modelling tasks but not yet a complete migration of summer2's calibration
story.

### The summer2 user evaluating a migration

*Wants: know whether to port an existing model.*

They will find the compartment model genuinely better — selector queries instead
of dictionaries, ragged stratification instead of compartment-name lists,
immutable maps with replayable history — and a flows / epi / results stack that
covers 49 of 52 symbols with any working route (**44 of 52** at `full`). They
can compile, run and plot; what still does not translate is initial population /
population split and Bayesian calibration. The honest answer is in
{doc}`../user/07-from-summer2`.

The risk to manage here is expectation around those remaining gaps, not around
whether a run is publishable at all.

### The contributor

*Wants: add the next layer (initial population, then contact data / calibration).*

This user is well served. The repository has a stated contribution contract
(`AGENTS.md`), an enforced feature bar (tests + notebook + plan, checked by
`pixi run check-branch`), `mypy --strict`, Hypothesis invariants, a benchmark
suite, git hooks, and a ledger whose next capability package is WP3 (initial
population / split). Onboarding friction is low and the next step is unambiguous.

Expert walkthroughs of landed behaviour (not user research) live in
`examples/notebooks/` (WP5 time-varying, WP6 epi) and the ported textbook /
summer2 pages under `docs/textbook/` and `docs/summer2/`.

## API ergonomics review

The taxonomy API is small and coherent. These are the frictions a user will hit,
in rough order of how often.

### `PropertyMap` hashing

Maps are hashable via a blake2b-16 digest of `codes` and `parent_row` plus
`properties` and `history`. Equal rebuilt maps share a hash and work as
`jax.jit` static arguments.

### Partial container protocol

`len(pmap)` works (same as `pmap.size`). `pmap[0]` and `for row in pmap` still
raise; use `pmap.to_dicts()[0]` / `iter(pmap.to_dicts())` for row access.

### No way to remove compartments

`stratify` is the only structural operation. There is no `filter`, no `drop`, no
`where` that subsets an existing map. Real models routinely need to exclude
impossible combinations after the fact — no vaccinated infants, no
treatment-experienced new diagnoses — and today the only route is to plan the
stratification order so that `where=` can express the exclusion up front. That
is not always possible.

### Tabular export

`to_frame()` returns a polars DataFrame (lazy import) with one column per
property and nulls for absent traits. `to_dicts()` remains available.

### No serialisation

A `PropertyMap` cannot be written to disk and read back. `history` is
serialisable data in principle — properties are names and trait tuples,
selectors are frozen dataclasses — but nothing converts it to and from JSON.
Reproducing a model structure therefore means re-running the Python that built
it.

### `partition` and `group_by` return types

`partition` returns a `dict[Trait, ...]`; `group_by` returns a `Groups` mapping
keyed by `tuple[Trait, ...]`. Both are Mapping-shaped; `partition` still unwraps
single-property keys and includes empty groups.

### Bulk constructor

`PropertyMap.from_properties([state, age, vax])` builds a fully-crossed map in
one call.

### `Present` / `Absent` are non-binding in flow pairing

`selector_values` binds `Trait` / `IsIn` only. `age.present()` names `age` but
does not bind it; `strict_pairing=True` raises if that would move people
across leftover strata. This is settled, not an open question.

## What is likely to satisfy

- **Error messages.** Every failure names the bad value and lists the valid
  alternatives. `Property 'age' is already on this map.`; `Unknown trait '0-5'
  for property 'age'. Known: ['0-4', '5-9', '10+']`.
- **The `__bool__` guard on selectors.** Writing `state["I"] and age["0-4"]`
  raises with an explanation instead of silently returning the second operand.
  This catches the single most likely beginner error in the whole API.
- **Immutability.** Nothing mutates; every intermediate map stays valid. This
  removes a whole class of notebook-ordering bugs that summer2 users know well.
- **Ragged stratification.** Once a user understands the three-valued rule, it
  does the right thing without special cases — and the rule is enforced by
  property-based tests rather than by documentation alone.

## Summary judgement

| Dimension | Assessment |
|---|---|
| Can a modeller complete a real task? | **Mostly** — run and plot yes; initial population and Bayesian calibration no |
| Is the implemented layer pleasant to use? | **Yes** |
| Is the implemented layer better than summer2's equivalent? | **Yes**, materially |
| Is it discoverable? | **It is now** — documentation covers taxonomy, flows, epi and ports |
| Is the contribution path clear? | **Yes** |
| Is there any evidence about real users? | **No** |

The project's risk is no longer "can anyone exercise a full run?" — they can.
The remaining risk is the distance to a *complete* migration surface: WP3
(initial population / split), then WP9 / WP10, plus the FOI susceptibility gap
that keeps chapter 15 at `partial`. Little external-user feedback has arrived
to steer those designs.
