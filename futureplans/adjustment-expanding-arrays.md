# Evaluate flow adjustments as expanding arrays over unique combinations

**Status:** open. Not scheduled. Deliberately left out of
`plans/model-stratify.plan.md`.  
**Where:** `_apply_adjustments` in `src/summer4/flows/compiled.py`;
`_bind_adjust_masks` in `src/summer4/flows/actualize.py`  
**Prior art:** `monash-emu/summer3proto`, branch `polarized`:
`code/summer3/summer3/polarized/expanding.py` (`ExpandingArray`, `apply_op`) and
`notebooks/ma/expanding_ops.ipynb`. The notebook
`notebooks/epimodel.ipynb` on branch `advanced_strats` calls a
`LazyExpandingArray`, but that class's definition is not committed on any branch.

## Problem

A flow's adjustment chain is folded over **every edge**. For each adjustment `k`,
`_apply_adjustments` evaluates `value`, computes `prev * value` (or takes `value`
for `Overwrite`, or `fn(prev, ...)` for `Transform`), then applies
`jnp.where(mask_k, new, prev)`. Each `mask_k` is an `E`-sized bool array baked
into the compiled program as a constant.

So a flow with `E` edges and `k` adjustments costs `k` elementwise passes over `E`
per step, and carries `k × E` bools of XLA constants. This is true even when the
adjustments only take a handful of distinct values. For example, an infection flow
stratified by age × location × strain × vaccination has `E` in the thousands, but
adjustments keyed on age and vaccination can produce at most
`|age| × |vaccination|` distinct multipliers.

Stratifying a model after its flows are declared (`plans/model-stratify.plan.md`)
makes long adjustment chains over large `E` more common.

## The summer3proto idea

- Each adjustment is a set of mutually exclusive **categories** over the flow's
  edge table (with `source(...)` / `dest(...)` polarity), plus one value per
  category.
- The running result is an `ExpandingArray`: a vector of **unique values** (length
  `U`) and an `opidx` that maps each edge to its unique value.
- Applying an adjustment (`apply_op`), with all index work done once on the host:
  1. Pair each edge's current `opidx` with its category index for this
     adjustment.
  2. Deduplicate the pairs to get the new unique set (`U' ≤ U × n_categories`,
     and `U' ≤ E`).
  3. Gather the previous unique values at those pairs.
  4. Scatter-multiply only the pairs whose category matched (`.at[active].mul(...)`).
  5. Re-index `opidx`.
- One final gather `data[opidx]` expands the `U` unique values to the `E` edges.

The same work also proposed **precedence-ordered** adjustments, where adjustments
of equal precedence (e.g. two multiplies) may be reordered. Precedence is now part
of `plans/model-stratify.plan.md`: default levels Overwrite → Multiply →
Transform, and `precedence=` to override. Precedence is what makes it legal for an
evaluator to move commuting categorical multiplies ahead of per-edge values.

## Performance tradeoffs (reasoned, not measured)

These are unmeasured: there are no flow benchmarks in `benchmarks/` yet (it only
holds taxonomy benchmarks).

| | Today: fold over `E` | Hoist the folded chain | Expanding arrays |
|---|---|---|---|
| Per-step ops | `k × (mul + where)` over `E`; elementwise, so XLA can fuse them | 1 multiply, for chains that are static per run | `k × (gather + scatter)` over `U`, then 1 gather to `E` |
| Baked-in constants | `k` bool masks of size `E` | same, but used only in `prepare` | `k` int index arrays of size `U`, plus 1 int32 `opidx` of size `E` |
| Wins when | `E` is small | the chain is run-static (`Const`, static `Param`) | `U ≪ E`; `k` is large; time-varying *categorical* adjustments; `vmap` over particles (prepared memory `N×U` vs `N×E`); `grad` w.r.t. adjustment params (tangents of size `U`) |
| Loses when | `k·E` is large | the chain varies per step | `U ≈ E` (adjustments keyed on the finest properties); per-edge values (`FlowRef`, `Reduce` over `y`, dense params) that cannot be categorised and break the compressed prefix; scatter/gather ops don't fuse the way elementwise ops do, and each launches a separate kernel on GPU |

Caveats:
- Each step still does `E`-sized work for population × rate and for the scatter
  into `dy`. Removing adjustment cost per step therefore saves only a bounded
  fraction of step time.
- The larger expected wins are XLA constant size, compile time, `vmap` memory and
  gradient cost. That makes this WP10 (calibration) / WP16 (scale) territory.
- **Hoisting the folded chain** captures most of the per-step win for run-static
  chains without any new representation. Try it before, or alongside, expanding
  arrays. See `futureplans/derived-fn-blocks-hoisting.md` for when hoisting is
  disabled.
- Host-side index work is `O(k · E log E)` per flow (`np.unique` over integer
  pairs), once at compile. That is negligible next to `actualize`.

## What "done" looks like

1. **Benchmark first** (`benchmarks/`). Sweep `E`, `k` and the ratio `U/E`, and
   measure per-step time, jit compile time, jaxpr size (`jax.make_jaxpr`), `vmap`
   memory over particles, and `grad` time w.r.t. adjustment params. Include the
   WP16 TB-scale model.
2. Try **hoisting the folded chain** for run-static chains, and measure.
3. Only if the benchmark shows a win where `U ≪ E`: add an expanding evaluator.
   - It consumes the canonical adjustment chain from `FlowEdges.adjust` (sorted by
     precedence, with edge-level `Source`/`Dest` masks). Convert masks to category
     indices at compile.
   - It compresses the longest prefix of categorical adjustments, then falls back
     to the per-edge fold for the rest.
   - Choose it per flow by a static `U/E` threshold set from the benchmark.
4. Tests: for random chains the expanding and fold evaluators produce identical
   per-edge rates, and `CompiledModel` digests don't depend on the evaluator.

## Related

- `plans/model-stratify.plan.md`: adjustment precedence levels and edge-level
  masks, which are the prerequisites this evaluator would consume.
- `plans/flows-derived-outputs.plan.md` §1: why summer4's edge map with Kleene
  absence replaced summer3proto's `polarized` selectors.
- `docs/dev/run-stages.md`: run-stage vs step-stage hoisting.
