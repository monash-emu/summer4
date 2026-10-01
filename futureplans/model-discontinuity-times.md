# Expose a model's time discontinuities for the step-size controller

## Seen on

`perf/gradient-performance`, profiling the Kiribati TB port's calibration gradient
(`docs/gradient-performance.md` on the port's `feat/gradient-performance`).

## What is wrong

summer4 knows where its rates jump in time — a `Lookup(..., floor(Time() - c))` changes value
at every integer offset, a `step` interpolation at every knot — but nothing tells the
caller's `diffrax.PIDController(jump_ts=...)`. Each port lists them by hand (Kiribati:
`np.arange(1851.0, 2035.0)` for the yearly mixing matrix).

Without `jump_ts`, an adaptive solver steps across each jump and its error estimate absorbs it.
On the Kiribati model at the calibration tolerance (Dopri5, `rtol = atol = 1.4e-4`) that is
the dominant error: log density 2.2e-3 nats from a 1e-10 solve, gradient error 0.7% median
and 3.6% max over seven posterior points. With `jump_ts` at the integer years (and a PI
controller) the same tolerance gives 6e-5 nats and 0.03% median / 0.16% max, for about the
same number of steps (740 against 723). `docs/summer4-workarounds.md` S3 in the port records
that the goldens need `rtol = 1e-10` for the same reason.

## Done when

- `CompiledModel` (or a free function over a rate tree) returns the discontinuity times in a
  window, e.g. `compiled.jump_times(t0, t1) -> np.ndarray`, from `Lookup` nodes whose index is
  `floor(Time() + const)` (or `floor(a * Time() + b)`), `step` interpolations (their knots) and
  `Data.interp("step")` tables. Parameter-dependent knots are excluded (and documented).
- The Diffrax backend docs show `PIDController(rtol=, atol=, jump_ts=compiled.jump_times(t0, t1))`;
  summer4 does not insert them silently (the controller stays the caller's object, CX1).
- A test model with a yearly `Lookup` gets the right times, and an adaptive solve with them is
  more accurate at fixed tolerance than without.
