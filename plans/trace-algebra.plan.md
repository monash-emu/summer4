---
name: trace-algebra
overview: Name-aligned Output arithmetic, windowed cumulative, the summer2 midpoint convention, and multi-flow FlowMass.
todos:
  - id: algebra
    content: Output ∘ Output by dim name, cumulative(start, end), midpoint
    status: completed
  - id: flows
    content: FlowMass sums several flows that share selected dims
    status: completed
  - id: rolling
    content: Vectorize Output.rolling so jaxpr size is flat in trajectory length
    status: completed
  - id: tests
    content: jit/grad, alignment errors, rolling jaxpr, multi-flow mass
    status: completed
isProject: false
---

# Output algebra

Follows `plans/tb-ports-feature-completeness.plan.md` §15a, §15b, and the
rolling half of §15d. The class is `Output` (`results/output.py`), not `Trace`.

Step 4 already shipped scalar and array arithmetic, unary ops, and
`eval_closed`. This branch is the rest of §15a: name-aligned `Output` ∘ `Output`.

## Corrections applied here

- `cumulative(start=, end=)` zeros outside the window. `start` and `end` must
  be save times. `end` is the last included save; points after it are zero,
  not a frozen total.
- A combined output keeps `PropertyData` when the operands share one map, or
  when exactly one has a map and the other has no aligned axis (per-age ÷
  total). Two different maps raise.
- `FlowMass(flow=("a",))` stores the same value as `FlowMass(flow="a")`, so
  existing save-plan digests do not move.
- `midpoint()` is summer2's `raw_results=False` convention, not the solver's
  accumulated flow.
- `OutputSet` and the `D5` ledger note stay on step 10.

## Not in this branch

- `OutputSet`, `Result.to_frame` wide/long, target residual reduction (§15c,
  the other half of §15d).
- An opt-in flow accumulator (`futureplans/state-ledgers-flow-integral.md`).
