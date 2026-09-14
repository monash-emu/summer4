# Benchmarking

```bash
pixi run bench        # table in the current environment
pixi run bench-json   # JSON keyed by JAX version, into benchmarks/results/
```

`benchmarks/` uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io).
`benchmarks/results/` is gitignored: benchmark numbers are machine-specific and
are compared within a run, not across commits on different hardware.

## What is measured

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

## Adding a benchmark

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

## The JAX matrix

`pixi run bench-json` writes results keyed by JAX version so that `default`
(JAX 0.6.x) and `latest` can be compared. The taxonomy does not use JAX, so
today the two differ only by noise. The harness exists now so that the solver
work has a baseline to regress against rather than needing one retrofitted — but
it should be read as infrastructure, not as evidence that cross-version
performance has been validated for anything that matters.
