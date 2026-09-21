---
name: rename-trace-to-output
overview: Rename the results class Trace to Output, and the Result mapping that holds it, without changing what a saved quantity stores.
todos:
  - id: rename
    content: Rename Trace to Output, results/trace.py to output.py, and Result.traces to outputs
    status: completed
  - id: docs
    content: Update tests, notebooks, and living docs to the new name
    status: completed
isProject: true
---

# Rename `Trace` to `Output`

`Trace` is one named quantity saved from a solve: `times`, `values`, and `dims`, plus the query methods that return another one. The name says "timeseries". A later output can be produced by a run and have no time axis (a parameterised mixing matrix). The class name should not assume a time axis.

This branch only renames. Every output still carries a `TimeAxis`. Nothing constructs a quantity with no time axis yet.

## Do

- `class Trace` becomes `class Output` in `src/summer4/results/output.py` (moved from `trace.py`).
- `Result.traces` becomes `Result.outputs`. Keys are unchanged; `result["compartments"]` still works.
- Dispatch mode `"trace"` in `summer4.flows.algebra` becomes `"output"`. JAX words ("traced", "trace time") stay.
- Update tests, example notebooks, and living docs (`docs/`, `futureplans/`, `AGENTS.md`, the roadmap) so they name `Output`.
- Do not edit historical `plans/*.plan.md` other than this file. Those records describe the name the code had when they were written.

## Not in this branch

- Optional `times`, or saving a non-time quantity such as a mixing matrix.
- Name-aligned `Output` ∘ `Output` arithmetic (roadmap step 9).
