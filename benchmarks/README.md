# Model timing suite

Committed warm-JIT timings for the shared ladder in
`summer2bench/spec.json`. Taxonomy benchmarks (`pixi run bench`) are
separate and are not in this table.

Machine: `macOS-15.7.9-arm64-arm-64bit; arm64; Apple M4; 24 GiB`. Date: `2026-09-20`.

Protocol: float64 (`jax_enable_x64` before compile); fixed `dt=0.1`;
step counts 200, 2_000, 8_000; save full compartments plus one raw
series for `infection` and `recovery`; warm median of 5 calls after
one discarded compile, each ending in `jax.block_until_ready` on a
reduced sum. Age-only mixing on stratified models.

summer2 uses its graph runners with `solver="euler"` and
`solver="rk4"` (JAX 0.4.x in `summer2bench/`). summer4 uses Diffrax
`Euler` and classical RK4 with `ConstantStepSize` (JAX 0.6.x in this
repo). The summer4 number includes Diffrax call overhead. Do not read
a winner out of the table beyond the times shown.

Diffrax `run()` reuses equinox's JIT cache across equal save plans
(see `plans/diffrax-run-jit-cache.plan.md`). The summer4 warm column
was re-recorded after that fix landed.

JSON: `summer2bench/recorded.json`, `benchmarks/recorded-summer4.json`.

| library | model | solver | steps | compartments | build | compile | warm median | dtype | JAX |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| summer2 | sir | euler | 200 | 3 | 892.44 ms | 157.33 ms | 170.1 µs | float64 | 0.4.38 |
| summer2 | sir | euler | 2000 | 3 | 648.98 ms | 163.11 ms | 1.28 ms | float64 | 0.4.38 |
| summer2 | sir | euler | 8000 | 3 | 748.44 ms | 234.36 ms | 4.80 ms | float64 | 0.4.38 |
| summer2 | sir | rk4 | 200 | 3 | 742.46 ms | 263.18 ms | 427.0 µs | float64 | 0.4.38 |
| summer2 | sir | rk4 | 2000 | 3 | 1.152 s | 329.38 ms | 2.86 ms | float64 | 0.4.38 |
| summer2 | sir | rk4 | 8000 | 3 | 850.53 ms | 274.94 ms | 13.89 ms | float64 | 0.4.38 |
| summer2 | sir_adjust | euler | 200 | 3 | 1.119 s | 251.55 ms | 858.2 µs | float64 | 0.4.38 |
| summer2 | sir_adjust | euler | 2000 | 3 | 955.68 ms | 229.13 ms | 1.35 ms | float64 | 0.4.38 |
| summer2 | sir_adjust | euler | 8000 | 3 | 763.42 ms | 199.70 ms | 5.19 ms | float64 | 0.4.38 |
| summer2 | sir_adjust | rk4 | 200 | 3 | 813.42 ms | 315.03 ms | 278.6 µs | float64 | 0.4.38 |
| summer2 | sir_adjust | rk4 | 2000 | 3 | 1.414 s | 1.405 s | 10.66 ms | float64 | 0.4.38 |
| summer2 | sir_adjust | rk4 | 8000 | 3 | 1.637 s | 705.11 ms | 16.92 ms | float64 | 0.4.38 |
| summer2 | sir_tv | euler | 200 | 3 | 2.517 s | 274.78 ms | 325.2 µs | float64 | 0.4.38 |
| summer2 | sir_tv | euler | 2000 | 3 | 768.60 ms | 214.17 ms | 2.85 ms | float64 | 0.4.38 |
| summer2 | sir_tv | euler | 8000 | 3 | 747.38 ms | 236.66 ms | 11.31 ms | float64 | 0.4.38 |
| summer2 | sir_tv | rk4 | 200 | 3 | 767.65 ms | 289.87 ms | 631.3 µs | float64 | 0.4.38 |
| summer2 | sir_tv | rk4 | 2000 | 3 | 747.23 ms | 300.14 ms | 5.52 ms | float64 | 0.4.38 |
| summer2 | sir_tv | rk4 | 8000 | 3 | 720.95 ms | 371.89 ms | 20.90 ms | float64 | 0.4.38 |
| summer2 | age_mix | euler | 200 | 48 | 708.40 ms | 217.74 ms | 943.2 µs | float64 | 0.4.38 |
| summer2 | age_mix | euler | 2000 | 48 | 673.58 ms | 208.60 ms | 8.55 ms | float64 | 0.4.38 |
| summer2 | age_mix | euler | 8000 | 48 | 887.18 ms | 220.07 ms | 34.49 ms | float64 | 0.4.38 |
| summer2 | age_mix | rk4 | 200 | 48 | 690.52 ms | 275.50 ms | 2.68 ms | float64 | 0.4.38 |
| summer2 | age_mix | rk4 | 2000 | 48 | 683.37 ms | 281.15 ms | 26.12 ms | float64 | 0.4.38 |
| summer2 | age_mix | rk4 | 8000 | 48 | 682.90 ms | 331.58 ms | 104.24 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | euler | 200 | 48 | 909.02 ms | 319.41 ms | 1.04 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | euler | 2000 | 48 | 923.09 ms | 323.38 ms | 9.24 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | euler | 8000 | 48 | 1.015 s | 375.97 ms | 43.48 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | rk4 | 200 | 48 | 1.021 s | 468.13 ms | 4.52 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | rk4 | 2000 | 48 | 1.303 s | 485.81 ms | 32.24 ms | float64 | 0.4.38 |
| summer2 | age_mix_tv | rk4 | 8000 | 48 | 2.456 s | 1.780 s | 315.26 ms | float64 | 0.4.38 |
| summer2 | stress | euler | 200 | 3840 | 4.750 s | 725.04 ms | 76.14 ms | float64 | 0.4.38 |
| summer2 | stress | euler | 2000 | 3840 | 1.343 s | 1.262 s | 673.90 ms | float64 | 0.4.38 |
| summer2 | stress | euler | 8000 | 3840 | 1.338 s | 3.312 s | 2.800 s | float64 | 0.4.38 |
| summer2 | stress | rk4 | 200 | 3840 | 1.992 s | 1.638 s | 290.89 ms | float64 | 0.4.38 |
| summer2 | stress | rk4 | 2000 | 3840 | 2.211 s | 4.478 s | 4.800 s | float64 | 0.4.38 |
| summer2 | stress | rk4 | 8000 | 3840 | 1.701 s | 9.563 s | 8.401 s | float64 | 0.4.38 |
| summer4 | sir | diffrax-euler | 200 | 3 | 180.97 ms | 443.72 ms | 1.46 ms | float64 | 0.6.2 |
| summer4 | sir | diffrax-euler | 2000 | 3 | 179.90 ms | 463.19 ms | 3.13 ms | float64 | 0.6.2 |
| summer4 | sir | diffrax-euler | 8000 | 3 | 166.87 ms | 446.11 ms | 11.09 ms | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 200 | 3 | 163.72 ms | 503.32 ms | 1.33 ms | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 2000 | 3 | 168.82 ms | 500.16 ms | 4.64 ms | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 8000 | 3 | 170.34 ms | 554.92 ms | 15.76 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 200 | 3 | 174.72 ms | 481.52 ms | 992.4 µs | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 2000 | 3 | 199.01 ms | 492.36 ms | 3.13 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 8000 | 3 | 171.00 ms | 509.25 ms | 10.44 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 200 | 3 | 166.21 ms | 528.44 ms | 981.3 µs | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 2000 | 3 | 172.63 ms | 525.86 ms | 4.70 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 8000 | 3 | 193.84 ms | 556.99 ms | 21.25 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 200 | 3 | 183.14 ms | 535.65 ms | 2.19 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 2000 | 3 | 173.49 ms | 582.90 ms | 7.56 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 8000 | 3 | 173.85 ms | 761.72 ms | 30.45 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 200 | 3 | 173.49 ms | 694.11 ms | 1.32 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 2000 | 3 | 178.31 ms | 574.61 ms | 5.75 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 8000 | 3 | 161.47 ms | 553.55 ms | 22.67 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 200 | 48 | 389.26 ms | 440.34 ms | 753.8 µs | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 2000 | 48 | 420.35 ms | 820.44 ms | 3.89 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 8000 | 48 | 429.86 ms | 499.74 ms | 14.45 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 200 | 48 | 455.15 ms | 621.49 ms | 1.28 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 2000 | 48 | 414.23 ms | 526.88 ms | 9.50 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 8000 | 48 | 378.85 ms | 589.15 ms | 38.38 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 200 | 48 | 452.81 ms | 512.67 ms | 951.3 µs | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 2000 | 48 | 379.54 ms | 468.92 ms | 4.23 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 8000 | 48 | 379.43 ms | 478.26 ms | 14.10 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 200 | 48 | 386.08 ms | 537.57 ms | 1.40 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 2000 | 48 | 409.03 ms | 545.72 ms | 10.07 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 8000 | 48 | 378.77 ms | 686.70 ms | 41.64 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 200 | 3840 | 517.39 ms | 1.104 s | 32.15 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 2000 | 3840 | 550.54 ms | 1.309 s | 323.20 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 8000 | 3840 | 754.16 ms | 2.458 s | 1.303 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 200 | 3840 | 630.63 ms | 1.295 s | 80.96 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 2000 | 3840 | 562.52 ms | 2.038 s | 868.44 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 8000 | 3840 | 522.11 ms | 4.675 s | 3.283 s | float64 | 0.6.2 |

