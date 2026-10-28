Implements Step 31 of `unified-lazy-idiom.plan.md`.

Deviations:

- The gate model is SEIRS rather than SEIR, with waning immunity `R → S` at `omega` = 1/120 per day, because without waning the first wave burns out in about 40 of 104 weeks and the seed-31 peak sat at `log_k` = 1.10 with a 30-seed sd of 0.45.
- `pixi run jupyter nbconvert --execute` is not available in this environment (`nbconvert` is not installed). The notebook's code cells were run top to bottom in one process instead, with figures captured and no `show` output saved.
