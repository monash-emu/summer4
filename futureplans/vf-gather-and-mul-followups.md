# Vector-field gather / mul follow-ups (after fused scatters)

## Seen on

`feat/fuse-incidence-scatter` while profiling tb-macro-summer4 and A/B-ing
`compile(fuse_compartment_updates=...)` on the TB-scale bench.

## What shipped

`fuse_compartment_updates=True` (default): one `scatter-add` for all
entry/exit/transition masses into `dy` after the per-flow mass loop.

## Tried and reverted: fused source gathers

Concatenating every relative exit/transition `src_idx`, one `y[all_src]`, then
per-flow `slice` cut forward gather count (TB-scale **39 → 7**) and VF eqns
(**1041 → 492**), and helped warm **solve** (~1.2×). Under `grad` /
`value_and_grad` (interleaved medians + `block_until_ready`) AD was ~15%
*slower*. Grad jaxprs showed the cause: each slice’s VJP is a **pad** (33
pads on TB-scale) that the per-flow gather path never emits; reverse also
turns one fat gather into one large duplicate-index scatter into `y`.

**Do not re-land src fusion** without a design that avoids slice→pad (e.g.
keep per-flow gathers, or a gather that returns a pytree of segments without
slicing a concatenated buffer).

## Remaining gather sources

| Source | Where |
|---|---|
| Per-flow `y[src]` | intentional after the revert above |
| Rate alignment | `_align_rate` / `_align_grouped_rate` |
| FOI / infectious pool | `summer4.epi.infection` |
| Mixing `Lookup` | yearly / stacked matrices |
| `Interp` / `TableInterp` | knot / table indexing |

## Multiply / fusions (~8%)

Mostly elementwise `mul` plus a few `dot_general` (mixing). XLA already fuses
many elementwise ops. Chase hoist of param-only mixing row-normalise
([`mixing-matrix-per-call-normalisation.md`](mixing-matrix-per-call-normalisation.md)),
run-static rate trees
([`derived-fn-blocks-hoisting.md`](derived-fn-blocks-hoisting.md)), and
expanding-array adjustments when `U ≪ E`
([`adjustment-expanding-arrays.md`](adjustment-expanding-arrays.md)). Do not
densify edge scaling into a full incidence matmul.

## Benchmark hygiene

A/B walls must use `jax.block_until_ready` and prefer interleaved medians;
best-of without a ready barrier flipped AD conclusions run-to-run.

## Related

- `plans/fuse-compartment-updates.plan.md`
- `benchmarks/test_bench_tb_scale.py::test_fuse_compartment_updates_timing_ab`
