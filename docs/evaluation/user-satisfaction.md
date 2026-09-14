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
| Set the initial population | **Blocked** |
| Add infection, progression, recovery flows | **Blocked** |
| Run the model | **Blocked** |
| Request incidence by age | **Blocked** — the index sets exist, the request mechanism does not |
| Plot | **Blocked** |
| Calibrate | **Blocked** |

Three of nine steps complete. The modeller cannot finish any task they came
with. For this user, summer4 is not yet a tool — it is a component of one.

### The summer2 user evaluating a migration

*Wants: know whether to port an existing model.*

They will find the compartment model genuinely better — selector queries instead
of dictionaries, ragged stratification instead of compartment-name lists,
immutable maps with replayable history — and then find that 46 of the 52 API
symbols their code uses have no equivalent. The honest answer for this user is
"not yet, and not close"; {doc}`../user/07-from-summer2` gives it directly
rather than making them discover it by trying.

The risk to manage here is expectation. The package is named `summer4` and
describes itself as a modelling platform; a summer2 user arriving at `pip
install summer4` reasonably expects to be able to run a model.

### The contributor

*Wants: add the flows layer.*

This user is well served. The repository has a stated contribution contract
(`AGENTS.md`), an enforced feature bar (tests + notebook + plan, checked by
`pixi run check-branch`), `mypy --strict`, Hypothesis invariants, a benchmark
suite, git hooks, and — unusually — a completed design spike with written
findings and a numbered promotion list. Onboarding friction is low and the next
step is unambiguous.

## API ergonomics review

The taxonomy API is small and coherent. These are the frictions a user will hit,
in rough order of how often.

### `PropertyMap` is unhashable

```python
{pmap: cached_result}   # TypeError: unhashable type: 'PropertyMap'
```

`PropertyMap` sets `eq=False` on the dataclass, defines `__eq__` by value, and
does not define `__hash__`. Maps therefore cannot be dictionary keys, memoisation
keys, or JAX pytree auxiliary data. This is item 1 on the spike's own promotion
list and it blocks the JAX work, not just user convenience.

### No container protocol

`len(pmap)`, `pmap[0]` and `for row in pmap` all raise. The information is
available (`pmap.size`, `pmap.to_dicts()[0]`, `iter(pmap.to_dicts())`) but the
obvious spellings do not work, and `size` is an unusual name for something
`len()` would normally give.

### No way to remove compartments

`stratify` is the only structural operation. There is no `filter`, no `drop`, no
`where` that subsets an existing map. Real models routinely need to exclude
impossible combinations after the fact — no vaccinated infants, no
treatment-experienced new diagnoses — and today the only route is to plan the
stratification order so that `where=` can express the exclusion up front. That
is not always possible.

### No tabular export

`to_dicts()` is the only structured view, despite `polars` being a development
dependency. `pmap.to_frame()` would be the obvious thing for anyone building a
compartment table for a paper or a report.

### No serialisation

A `PropertyMap` cannot be written to disk and read back. `history` is
serialisable data in principle — properties are names and trait tuples,
selectors are frozen dataclasses — but nothing converts it to and from JSON.
Reproducing a model structure therefore means re-running the Python that built
it.

### Inconsistent return types

`partition` returns a `dict`; `group_by` returns a generator. Both are
"group compartments by properties". The asymmetry is defensible on performance
grounds and surprising in use.

### No bulk constructor

Building a map always starts with `from_property` and chains. There is no
`PropertyMap.from_properties([state, age, vax])` for the common fully-crossed
case, so the simplest possible model is four lines instead of one.

### The `Present` / `Absent` binding question is unresolved

`FINDINGS.md` flags that treating `age.present()` as *binding* `age` is right
for a pairing override and surprising for a "match leftover age" reading. This
is the one open design question that affects the **already public** API, and it
should be settled before more code depends on the current behaviour.

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
| Can a modeller complete a real task? | **No** |
| Is the implemented layer pleasant to use? | **Yes** |
| Is the implemented layer better than summer2's equivalent? | **Yes**, materially |
| Is it discoverable? | **It is now** — there was no documentation before this site |
| Is the contribution path clear? | **Yes** |
| Is there any evidence about real users? | **No** |

The project's risk is not quality; the layer that exists is good. The risk is
that the distance to a usable tool is large — ten of twenty textbook chapters and
eight of eleven summer2 notebooks sit behind one tier of missing functionality —
and that no external user can currently exercise anything, so no feedback is
arriving to steer the remaining design.
