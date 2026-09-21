# Changelog

All notable changes to summer4 are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [PEP 440](https://peps.python.org/pep-0440/). summer4 is an alpha: the
API is not stable.

Install from the matching git tag (see
[Installation](docs/getting-started/installation.md)). Narrative release notes
for the Sphinx site live under [docs/releases/](docs/releases/).

## [0.2.0a3] — 2026-09-21

Tag: [`v0.2.0a3`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a3)

### Highlights

- **`prepare()` boxes float params.** Plain Python `float` leaves in the params
  pytree are promoted to floating JAX arrays so Diffrax's equinox `filter_jit`
  cache hits across draws with `{str: float}` dicts. a2 fixed Module identity;
  a3 fixes the remaining static-float miss (`fix/prepare-box-float-params`).

### Fixed

- `CompiledModel.prepare` promotes Python `float` parameter leaves to floating
  JAX arrays so Diffrax's equinox `filter_jit` cache hits across draws with
  plain `{str: float}` dicts (`fix/prepare-box-float-params`, PR #16).

## [0.2.0a2] — 2026-09-20

Tag: [`v0.2.0a2`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a2)

### Highlights

- **Diffrax `run()` JIT cache.** Repeated `CompiledModel.run()` calls with the
  same save plan and solver settings now hit equinox's `filter_jit` cache. The
  Diffrax backend keeps process-stable equinox Modules for the vector field and
  `SubSaveAt` callbacks, and reads prepared parameters from Diffrax `args`, so
  a parameter draw no longer forces a recompile. On the SIR Euler 200-step
  probe this is one cold compile, then warm solves on the order of **1000×**
  faster.

### Added

- Rate-tree math (`tanh`, `clip`, `pow` and friends) that evaluates a
  parameter-only expression and can also scale a saved `Output`
  (`feat/rate-math`).
- A batched table interpolator so a per-age or yearly series is one node, not a
  rebuilt gather on every vector-field call (`feat/table-interp`).
- Example notebooks rewritten as readable user gates: summer2-style prose, a
  titled plot of each claim, and an assertion of the same claim
  (`docs/example-notebook-style`).
- Downstream smoke CI that installs summer4 from the git tag into a scratch
  pixi project (`chore/downstream-smoke-ci`).

### Fixed

- Diffrax `run()` rebuilt `ODETerm` / `SubSaveAt` callables on every call,
  keyed by Python identity, so equinox never reused a compiled solve
  (`fix/diffrax-run-jit-cache`, PR #13).

## [0.2.0a1] — 2026-09-18

Tag: [`v0.2.0a1`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a1)

First pinnable release of the flows stack on `main`: compartment taxonomy,
`FlowModel` / `CompiledModel`, results (`SavePlan`, `Result`, `Output`),
Diffrax solvers, epidemiology (`ForceOfInfection`, `MixingMatrix`),
time-varying rates, stratification, and honest packaging (JAX as a core
dependency; `frames` extra for polars / pyarrow).
