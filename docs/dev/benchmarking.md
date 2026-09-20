# Benchmarking

```bash
pixi run bench         # taxonomy table in the current environment
pixi run bench-json    # taxonomy JSON keyed by JAX version, into benchmarks/results/
pixi run bench-models  # committed model ladder → benchmarks/recorded-summer4.json
```

`benchmarks/` uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io)
for the taxonomy suite. `benchmarks/results/` is gitignored: those numbers are
machine-specific and are compared within a run, not across commits on different
hardware.

The **model ladder** (`pixi run bench-models`) is separate. Its numbers are
committed under `benchmarks/recorded-summer4.json` and
`summer2bench/recorded.json`, with the comparison table in
`benchmarks/README.md`. Do not fold model timings into `pixi run bench`
or into {doc}`performance`.

## Taxonomy: what is measured

`test_bench_taxonomy.py` covers the two operations that the layers above will
perform constantly:

| Benchmark | Sizes | Measures |
|---|---|---|
| `test_bench_build` | 10, 1 000, 100 000 compartments | Cartesian map construction |
| `test_bench_select_cold` | 10, 1 000, 100 000 | Query resolution on a map with an empty cache |

The cold-select benchmark deliberately calls `pm.copy().select(sel)` so that the
per-map query cache is bypassed. Benchmarking a warm select measures a
dictionary lookup, which is not interesting; benchmarking a cold one measures
the `int8` passes over the table, which is.

For the shape of these curves, measured live at documentation build time, see
{doc}`performance`.

## Model ladder protocol

Shared numeric spec: `summer2bench/spec.json`. Both libraries read that file.

- **dtype:** float64. Enable `jax_enable_x64` before the first compile. There is
  no float32 arm. This does not flip summer4's import-time default.
- **Warm JIT:** build and one untimed call outside the median. The timed
  statistic is the median of five later calls. Each call ends in
  `jax.block_until_ready` on a reduced sum of compartments and of each flow
  series.
- **Solvers:** summer2 `euler` and `rk4` via `get_runner`. summer4 Diffrax
  `Euler` and classical RK4 (`benchmarks/diffrax_rk4.py`) with fixed `dt`,
  `steps`, `max_steps >= steps`, and no `rtol`/`atol` (so
  `ConstantStepSize`). Never `solver="euler"` on the summer4 side — that string
  is the hand-rolled backend. The summer4 number includes Diffrax.
- **JIT:** both libraries' warm medians are true warm solves after one
  discarded compile. Diffrax `CompiledModel.run` reuses equinox's JIT cache
  across equal save plans (`plans/diffrax-run-jit-cache.plan.md`). The
  committed `benchmarks/recorded-summer4.json` was re-recorded after that
  fix.
- **Step ladder:** 200, 2_000, 8_000 at `dt=0.1` (`t1` of 20, 200, 800).
- **Saves:** full compartment trajectory plus one raw series for `infection`
  and `recovery` (no cumulative sums, per-capita rates, or `OutputSet`).
- **Mixing:** age only on stratified models. Location and strain are real
  compartments but homogeneous on those axes.

JAX versions differ by design: summer2 stays on its 0.4.x pin in
`summer2bench/`; this repo's default env is JAX 0.6.x. Do not put them in one
process.

## Adding a taxonomy benchmark

Keep them honest about the cache:

```python
def test_bench_my_operation(benchmark, ...):
    pmap = _cartesian_map((10, 10, 10))
    sel = ...

    def _cold() -> int:
        return int(pmap.copy().select(sel).size)

    assert benchmark(_cold) > 0
```

Always assert on the result. A benchmark whose work is optimised away measures
nothing, and `pytest-benchmark` will happily report a fast time for it.

## The JAX matrix (taxonomy only)

`pixi run bench-json` writes taxonomy results keyed by JAX version so that
`default` (JAX 0.6.x) and `latest` can be compared. The taxonomy does not use
JAX, so today the two differ only by noise. The harness exists now so that the
solver work has a baseline to regress against rather than needing one
retrofitted — but it should be read as infrastructure, not as evidence that
cross-version performance has been validated for anything that matters.
