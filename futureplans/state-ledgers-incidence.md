# Opt-in incidence accumulator in State.ledgers

## What is wrong today

`Trace.incidence()` / `integrate()` recover counts from **saved instantaneous
rates** by trapezoid or Simpson quadrature over the save grid. That is
second-order (trapezoid) at best and **not** equal to the solver's own
accumulated flow mass. For calibration against case counts the bias is real
and cannot be removed from a finished `Result` — the information is gone.

## Where

- Reserved hook: `State.ledgers` in `src/summer4/jax/state.py` (empty mapping
  today; Phase 2 reserved it for cumulative quantities).
- Call sites that would feed it: flow mass updates inside
  `CompiledModel.observe` (`src/summer4/flows/compiled.py`).

## What "done" looks like

An opt-in save-plan / model flag that accumulates selected flow masses into
`State.ledgers` during the step (exact for the solver's quadrature), exposed
on the `Result` as a ledger trace. Trapezoid `Trace.incidence()` remains the
default post-hoc path for plans that did not request accumulation.
