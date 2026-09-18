---
name: rate-math
overview: Add unary and extended binary rate operators, and share their numeric kernel with Trace so a parameter transform can scale a saved output.
todos:
  - id: nodes
    content: UnaryOp, BinOp pow/maximum/minimum, clip, five-site wiring, hoist
    status: completed
  - id: algebra
    content: Shared apply_unary/apply_binary plus eval_closed for parameter-only trees
    status: completed
  - id: trace
    content: Trace scalar/array arithmetic on the same kernel; defer name-aligned Trace-to-Trace
    status: completed
  - id: tests-notebook
    content: tests/test_rate_math.py and examples/notebooks/13-rate-math-and-tables.ipynb
    status: completed
isProject: false
---

# Rate math

Follows `plans/tb-ports-feature-completeness.plan.md` §13.1. The roadmap step is authoritative where they differ: the notebook is `examples/notebooks/13-rate-math-and-tables.ipynb`.

## Operators

`UnaryOp` ops: `neg`, `exp`, `log`, `abs`, `tanh`, `sqrt`, `floor`.

`BinOp` gains `pow`, `maximum`, `minimum`. `clip(x, lo, hi)` is `minimum(maximum(x, lo), hi)`; either bound may be omitted.

Shape semantics match existing `BinOp`: a `GroupedRate` combined with a scalar keeps its grouping; two `GroupedRate`s must share a grouping.

## Shared kernel

`summer4.flows.algebra.apply_unary` / `apply_binary` are the only numeric implementations. `_eval_rate` calls them. So does value-side arithmetic on a `GroupedRate`, a `PropertyData`, or a `Trace`.

Python scalars passed to `exp` / `clip` / the dunders stay symbolic (`tanh(0.5)` is a node). Arrays, grouped rates and traces are applied immediately, so the same function works inside a jitted loss on a parameter array.

## Parameter transforms and traces

`eval_closed(expr, params)` evaluates a run-stage tree (no `Time`, `FlowRef`, `Reduce` or `Capture`). That is how `tanh(Param("se"))` can both be a flow rate and scale a saved output: `trace * eval_closed(expr, params)`. A `Trace` has no parameters, so `trace * expr` raises and names `eval_closed`.

Name-aligned `Trace` ∘ `Trace`, windowed `cumulative`, `midpoint` and multi-flow `FlowMass` stay on `feat/trace-algebra`.

## Hoisting

`UnaryOp` takes the stage of its argument. A run-stage non-leaf is one hoist slot. A step-stage unary still descends, so `exp(Param("a") * Param("b") + Time())` hoists the parameter product.
