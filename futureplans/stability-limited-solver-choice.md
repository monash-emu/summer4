# Stability-limited models: the solver docs steer users to the wrong method

## Seen on

`perf/gradient-performance`, profiling the Kiribati TB port's calibration gradient
(`docs/gradient-performance.md` and `scripts/bench/` on the port's `feat/gradient-performance`).

## What is wrong

summer4's docs and examples default to Dopri5 for adaptive solves. TB-scale compartment
models are *stability*-limited at calibration tolerances: in the Kiribati model the fastest
Jacobian eigenvalue is about -10 per year (8–11 across posterior draws, about -20 at the
prior's upper corner; `scripts/bench/stiffness.py`), so Dopri5 (real-axis stability boundary
3.3) cannot step further than about 0.3 years whatever the tolerance: 723 steps at
`rtol = 1.4e-4`, 713 at `1e-3`. The step-size controller then oscillates at the stability
edge and rejects 22% of steps (156 of 723) with diffrax's default I-controller.

Measured on that model (MAP; gradient error against a 1e-10 reference gradient over seven
posterior points; `outputs/bench/levers_all.json`, `levers_confirm.json`, `nan_rate*.json`):

| configuration | steps (acc + rej) | grad, wall / CPU vs Dopri5 | grad error median / max | NaN grads, 256 prior points |
|---|---|---|---|---|
| Dopri5, I-controller | 567 + 156 | 1.00 / 1.00 | 0.7% / 3.6% | 43 |
| Dopri5, PI 0.4/0.3 | 582 + 50 | 1.11 / 1.13 | 0.3% / 2.2% | 43 |
| Dopri5, PI + `jump_ts` | 705 + 35 | 0.89 / 0.92 | 0.03% / 0.16% | 43 |
| Tsit5, PI + `jump_ts` | 725 + 37 | 1.05 / 0.95 | 0.05% / 0.7% | 27 |
| Bosh3, PI + `jump_ts` | 905 + 94 | 1.30 / 1.21 | 0.12% / 0.29% | 0 |
| Bosh3, PI + `jump_ts`, 1024 checkpoints | 905 + 94 | 1.44 / 1.30 | 0.12% / 0.29% | 0 |
| Kvaerno5 (implicit), `jump_ts` | 281 + 6 | 0.07 / 0.03 | 2% / 15% | – |

Bosh3 needs three new vector-field evaluations per step against Dopri5's six, and its
stability boundary (2.5) loses less than that. Its reverse-mode gradient was also finite at
every prior design point, where Dopri5's was NaN at 17% (the port's
`docs/summer4-workarounds.md` S5): the NaN appears to come from Dopri5's (and Tsit5's) stages
visiting invalid states, which Bosh3's do not. Implicit Kvaerno solvers take fewer steps but
each costs a Jacobian and linear solves: 15–30x slower here. Constant steps that divide a year
(Dopri5 at 0.25) need no `jump_ts` but fail outright at 90 of 256 prior points where the
eigenvalue exceeds their stability limit.

## Done when

- The solver-backends user guide has a "stiff but not very stiff" section: how to find the
  fastest eigenvalue (`jax.jacfwd` of the vector field along a trajectory), why tolerance
  stops mattering, a PI controller (`pcoeff=0.4, icoeff=0.3`), `jump_ts` for time
  discontinuities (see `model-discontinuity-times.md`), and Bosh3 as the default suggestion
  for such models, with a figure of steps against tolerance.
- `benchmarks/test_bench_tb_scale.py` records steps and gradient time for Dopri5 and Bosh3.
- The S5 NaN is localised (which Dopri5 stage, which op) and either guarded in summer4
  (e.g. `maximum(n_grp, 0) ** exponent` in `epi/infection.py`) or documented.
