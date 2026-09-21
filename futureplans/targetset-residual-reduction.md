# `TargetSet.residuals` reshapes but cannot reduce

## What is wrong today

{meth}`TargetSet.residuals` (`src/summer4/results/targets.py:166`) gathers each
target's key at its observation times, then does

```python
pred.reshape(target.values.shape)
```

falling back to `reshape(-1)` and raising if the sizes still disagree. There is
no hook for reducing a wider prediction onto a narrower observation.

For an unstratified model this is invisible: a `Compartments(where=...)` output
is one column and reshapes cleanly. For a stratified model it bites. On a
`state x age` map:

- `Compartments(where=state["I"])` gives `(T, 3)` and raises against a `(T,)`
  target;
- `Compartments(where=state["I"], sum_over=age)` also gives `(T, 3)` — grouped,
  not totalled — and raises as well;
- there is no quantity that says "the total across all bands".

Two workarounds exist, both used in the documentation. Target one stratum per
`Target`, with `FlowMass(flow, where=Dest(age[band]))` or
`Compartments(where=state["I"] & age[band])`, each of which is a single column.
Or pass a `SaveFn` that does the reduction during the solve.

## Why it hurts

Calibrating a stratified model against an aggregate series — a national case
count, a total death count — is an entirely ordinary thing to want, and it is
the first thing that does not work. The `SaveFn` workaround also costs the
`ComputedValue` path validation and hashes by `id(fn)`, so a hook defined in a
loop retraces.

The failure is at least loud. `residuals` raises with both shapes named:

```
ValueError: Target 'I': prediction shape (3, 2) incompatible with values shape (3,).
```

## What a fix looks like

This is WP10 territory, since the likelihood layer will need the same thing.
Options:

1. A `reduce=` on `Target` — `"sum"` / `"mean"` / a callable — applied to the
   gathered prediction before comparison. Smallest change, and it keeps the
   reduction declarative and hashable.
2. A `total=True` on `Compartments` and `FlowMass`, so the *save* is already
   scalar per time. Cheaper to solve, and composes with the existing
   `sum_over`.
3. Let `Target.quantity` accept an `Output`-to-`Output` expression. Most general,
   most design work, and it needs `Output` arithmetic, which does not exist
   either.

Option 2 plus option 1 covers the common cases without new machinery.
