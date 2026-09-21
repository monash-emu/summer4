---
name: tb-scale-bench
overview: Synthetic Kiribati-shaped model benchmark — compile, jaxpr, wall time, vmap×64 — as the WP16 baseline.
todos:
  - id: model
    content: 10×8×2 compartments, generalised FOI + Lookup, table deaths, ageing, ~150 OutputSet
    status: completed
  - id: numbers
    content: Record euler/dopri5 compile, jaxpr, wall, and vmap×64 in benchmarks/README
    status: completed
isProject: false
---

# TB-scale benchmark

Follows `plans/tb-ports-feature-completeness.plan.md` §16a. The roadmap step is
authoritative: the suite lives at `benchmarks/test_bench_tb_scale.py`, is marked
`slow`, and is collected by `pixi run test` / `test-all`. Numbers go in
`benchmarks/README.md` — a baseline that is not written down is not a baseline.

No ledger row moves. Solver failure surfacing (`KI22`) is step 12.
