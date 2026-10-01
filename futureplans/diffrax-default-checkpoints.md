# Reverse-mode solves recompute the forward pass with diffrax's default checkpoints

## Seen on

`perf/gradient-performance`, profiling the Kiribati TB port's calibration gradient
(`docs/gradient-performance.md` on the port's `feat/gradient-performance`).

## What is wrong

`Diffrax(adjoint=None)` (`src/summer4/solvers/diffrax_backend.py`) leaves diffrax's default
`RecursiveCheckpointAdjoint(checkpoints=None)`. Equinox then stores about `sqrt(2 * max_steps)`
checkpoints (`equinox/internal/_loop/checkpointed.py`): 90 for the port's `max_steps=4096`,
153 for summer4's `default_max_steps` over 1850–2035 (11,840). A calibration solve takes
600–900 steps, so the reverse pass recomputes stretches of the forward solve (treeverse).
Memory is not the constraint here: a checkpoint is one state (160 floats) plus solver state.

## Measured

Kiribati model, MAP point, in-process interleaved, 30 rounds on a loaded 10-core M4
(`scripts/bench/levers.py`, `outputs/bench/levers_confirm.json` in the port):

| solve | default checkpoints | `checkpoints=1024` | speedup (wall / CPU) |
|---|---|---|---|
| Dopri5, I-controller, 1.4e-4 | 1.04 s | 0.88 s | 1.19x / 1.07x |
| Bosh3, PI + `jump_ts`, 1.4e-4 | 0.81 s | 0.73 s | 1.11x / 1.07x |

Gradients identical. (Wall times are inflated by machine load; the ratio is the result.)

## Done when

- The Diffrax backend docs (and the calibration user guide) say that a calibration gradient
  should set `adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=n)` with `n` at least the
  expected step count, with this measurement.
- Optionally, `Diffrax` with `adjoint=None` and a known `max_steps` passes
  `checkpoints=min(max_steps, some cap)` instead of diffrax's square-root default; that is a
  summer4 default, so record it in `docs/evaluation/composability.md` if adopted (the caller's
  own adjoint object must still win, CX2).
