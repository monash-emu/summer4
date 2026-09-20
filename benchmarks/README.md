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
| summer4 | sir | diffrax-euler | 200 | 3 | 437.11 ms | 5.178 s | 2.025 s | float64 | 0.6.2 |
| summer4 | sir | diffrax-euler | 2000 | 3 | 790.38 ms | 1.916 s | 1.922 s | float64 | 0.6.2 |
| summer4 | sir | diffrax-euler | 8000 | 3 | 601.53 ms | 1.944 s | 1.421 s | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 200 | 3 | 578.91 ms | 2.385 s | 2.288 s | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 2000 | 3 | 739.59 ms | 2.828 s | 2.936 s | float64 | 0.6.2 |
| summer4 | sir | diffrax-rk4 | 8000 | 3 | 819.79 ms | 3.675 s | 1.943 s | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 200 | 3 | 520.74 ms | 1.167 s | 767.46 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 2000 | 3 | 650.11 ms | 865.78 ms | 735.24 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-euler | 8000 | 3 | 384.63 ms | 900.94 ms | 1.058 s | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 200 | 3 | 680.72 ms | 3.576 s | 1.883 s | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 2000 | 3 | 295.25 ms | 1.001 s | 663.90 ms | float64 | 0.6.2 |
| summer4 | sir_adjust | diffrax-rk4 | 8000 | 3 | 297.97 ms | 980.89 ms | 672.77 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 200 | 3 | 228.33 ms | 754.27 ms | 683.37 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 2000 | 3 | 244.82 ms | 792.31 ms | 510.09 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-euler | 8000 | 3 | 179.74 ms | 637.19 ms | 480.03 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 200 | 3 | 178.10 ms | 615.23 ms | 525.61 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 2000 | 3 | 170.67 ms | 574.04 ms | 528.58 ms | float64 | 0.6.2 |
| summer4 | sir_tv | diffrax-rk4 | 8000 | 3 | 219.51 ms | 833.41 ms | 545.91 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 200 | 48 | 656.10 ms | 924.68 ms | 559.81 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 2000 | 48 | 457.13 ms | 581.13 ms | 462.76 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-euler | 8000 | 48 | 487.03 ms | 569.17 ms | 479.45 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 200 | 48 | 430.95 ms | 585.13 ms | 534.68 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 2000 | 48 | 487.28 ms | 593.62 ms | 528.28 ms | float64 | 0.6.2 |
| summer4 | age_mix | diffrax-rk4 | 8000 | 48 | 466.88 ms | 660.40 ms | 516.97 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 200 | 48 | 417.54 ms | 516.30 ms | 468.51 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 2000 | 48 | 428.61 ms | 545.53 ms | 426.82 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-euler | 8000 | 48 | 412.43 ms | 511.47 ms | 439.75 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 200 | 48 | 427.21 ms | 588.22 ms | 502.87 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 2000 | 48 | 433.04 ms | 589.12 ms | 516.61 ms | float64 | 0.6.2 |
| summer4 | age_mix_tv | diffrax-rk4 | 8000 | 48 | 448.03 ms | 645.50 ms | 535.09 ms | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 200 | 3840 | 564.61 ms | 1.104 s | 1.094 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 2000 | 3840 | 550.57 ms | 1.357 s | 1.307 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-euler | 8000 | 3840 | 611.15 ms | 2.878 s | 2.252 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 200 | 3840 | 548.03 ms | 1.177 s | 1.110 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 2000 | 3840 | 566.36 ms | 2.195 s | 1.835 s | float64 | 0.6.2 |
| summer4 | stress | diffrax-rk4 | 8000 | 3840 | 557.01 ms | 4.694 s | 4.362 s | float64 | 0.6.2 |

