# Opt-in flow accumulator in State.ledgers

## What is wrong today

`Output.integrate_intervals()` / `integrate()` recover interval totals from
**saved instantaneous rates** by trapezoid or Simpson quadrature over the save
grid. That is second-order (trapezoid) at best and **not** equal to the
solver's own accumulated flow. The difference cannot be removed from a
finished `Result` — the information is gone.

## Where

- Reserved hook: `State.ledgers` in `src/summer4/jax/state.py` (empty mapping
  today; Phase 2 reserved it for cumulative quantities).
- Call sites that would feed it: flow mass updates inside
  `CompiledModel.observe` (`src/summer4/flows/compiled.py`).

## What "done" looks like

An opt-in save-plan / model flag that accumulates selected flow masses into
`State.ledgers` during the step (exact for the solver's quadrature), exposed
on the `Result` as a ledger output. Trapezoid `Output.integrate_intervals()`
remains the default post-hoc path for plans that did not request accumulation.

## Evidence from the Kiribati port (2026-11-19, `CX57`)

The clean Kiribati port (`monash-emu/tb-kiribati-s2s4`) compared yearly counts from save-grid
quadrature with an accurate integral of the flow:

- baseline: midpoint counts within 0.34 % (improved solver settings: 0.003 %);
- **screening scenario 3**: notifications from midpoint counts were up to **41 %** off (2027), the
  screening campaign was overstated 2.8× and split across 2026 and 2027, because the campaign
  ramps over 0.01 years and the save grid is yearly. Only putting the ramp times on the save grid
  and passing them as `jump_ts` got within 0.7 %.

An exact running total would make yearly counts independent of the save grid. Summer2 ports that
report yearly notifications or campaign yields are the users who need it.

Status: still deferred. It needs a state extension (one accumulator per requested flow, integrated
by the solver), which touches `State`, the vector field and the save plan, too large for the
Kiribati gaps branch.
