# XLA algebraic simplifier stuck on TB-scale euler

## Seen on

`feat/tb-scale-bench` (roadmap step 11 / WP16 §16a), while measuring
`benchmarks/test_bench_tb_scale.py`.

## Symptom

Every warm `CompiledModel.run(..., solver="euler")` of the synthetic
Kiribati-shaped model (160 compartments, ~120 save keys, yearly 1850–2035)
prints:

```text
Algebraic simplifier is likely stuck in a circular simplification loop
and ran for 50 runs on computation region_…
```

Wall time stays about **1 s per euler run** even after warm-up. The same model
under `solver="dopri5"` warms to tens of milliseconds. `jax.make_jaxpr` of the
vector field is about 1000 equations; of `run` about 1600–2500.

Linear vs sigmoidal `Data.table` death rates does not remove the warning.

## Why it matters

A 1 s/run floor on euler makes calibration-style `vmap` over dozens of
parameter sets look like a taxonomy cost rather than an integrate cost. The
baseline in `benchmarks/README.md` records the number; it should not be treated
as “euler is intrinsically that slow on 185 steps”.

## Done looks like

- Identify the HLO region (FOI + `Lookup` mixing vs large `SavePlan` observe vs
  `TableInterp`) that trips the simplifier loop.
- Either restructure that op so XLA finishes, or document a supported workaround
  (e.g. prefer dopri5 / tsit5 for this shape).
- Warm euler wall time on the TB-scale bench drops in line with step count when
  the save plan is held fixed, with no stuck-simplifier spam.
