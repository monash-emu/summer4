# Summer4 JAX version sweep

Warm-JIT Diffrax timings for the same summer4 model ladder under
several JAX pins. Diffrax stays at whatever each pixi env resolves
(this sweep: 0.7.2).

Machine: `macOS-15.7.9-arm64-arm-64bit-Mach-O; arm64; Apple M4; 24 GiB`. Date: `2026-09-20`.

Environments (`pixi run -e … bench-models` / `record_jax_sweep.py`):

- `jax04` — JAX **0.4.38**, same pin as `summer2bench`
- `default` — JAX **0.6.x** (repo default)
- `latest` — bleeding-edge JAX (`jax=*`) 

Protocol matches `benchmarks/README.md`: float64, fixed `dt=0.1`,
warm median of 5 calls after one discarded compile,
`block_until_ready` on the reduced sum.

## Focus cells (warm median)

| model | solver | steps | JAX 0.4.38 | JAX 0.6.2 | JAX 0.11.1 |
| --- | --- | ---: | ---: | ---: | ---: |
| sir | diffrax-euler | 200 | 944.6 µs | 1.21 ms | 1.92 ms |
| sir | diffrax-euler | 8000 | 12.66 ms | 12.77 ms | 27.95 ms |
| sir | diffrax-rk4 | 200 | 1.14 ms | 1.19 ms | 2.90 ms |
| sir | diffrax-rk4 | 8000 | 22.70 ms | 24.11 ms | 49.17 ms |
| stress | diffrax-euler | 200 | 455.46 ms | 58.62 ms | 30.17 ms |
| stress | diffrax-euler | 8000 | 19.437 s | 1.597 s | 1.103 s |
| stress | diffrax-rk4 | 200 | 1.236 s | 151.61 ms | 82.00 ms |
| stress | diffrax-rk4 | 8000 | 87.509 s | 5.075 s | 4.061 s |

## Versus summer2 (warm median)

summer2 is its graph runner on JAX 0.4.38 (`summer2bench/recorded.json`),
not Diffrax. Multiplier is summer4 / summer2 (below 1 means summer4 is faster).

| model | solver | steps | summer2 | JAX 0.4.38 | × vs s2 | JAX 0.6.2 | × vs s2 | JAX 0.11.1 | × vs s2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sir | euler | 200 | 170.1 µs | 944.6 µs | 5.55× | 1.21 ms | 7.13× | 1.92 ms | 11.28× |
| sir | euler | 8000 | 4.80 ms | 12.66 ms | 2.64× | 12.77 ms | 2.66× | 27.95 ms | 5.82× |
| sir | rk4 | 8000 | 13.89 ms | 22.70 ms | 1.63× | 24.11 ms | 1.74× | 49.17 ms | 3.54× |
| age_mix | rk4 | 8000 | 104.24 ms | 213.57 ms | 2.05× | 57.12 ms | 0.55× | 32.99 ms | 0.32× |
| stress | euler | 8000 | 2.800 s | 19.437 s | 6.94× | 1.597 s | 0.57× | 1.103 s | 0.39× |
| stress | rk4 | 8000 | 8.401 s | 87.509 s | 10.42× | 5.075 s | 0.60× | 4.061 s | 0.48× |

On the matching JAX 0.4.38 pin, summer4 loses everywhere (Diffrax
overhead on small models; much worse on `stress`). On JAX 0.6.2 and
0.11.1, summer4 beats summer2 on `age_mix` / `stress` while summer2
still wins the tiny `sir` cells.

## Full matrix

| JAX | Diffrax | model | solver | steps | build | compile | warm median |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.4.38 | 0.7.2 | sir | diffrax-euler | 200 | 417.63 ms | 431.46 ms | 944.6 µs |
| 0.4.38 | 0.7.2 | sir | diffrax-euler | 2000 | 242.69 ms | 394.73 ms | 3.81 ms |
| 0.4.38 | 0.7.2 | sir | diffrax-euler | 8000 | 238.84 ms | 399.85 ms | 12.66 ms |
| 0.4.38 | 0.7.2 | sir | diffrax-rk4 | 200 | 248.26 ms | 473.84 ms | 1.14 ms |
| 0.4.38 | 0.7.2 | sir | diffrax-rk4 | 2000 | 239.42 ms | 449.07 ms | 5.97 ms |
| 0.4.38 | 0.7.2 | sir | diffrax-rk4 | 8000 | 241.26 ms | 473.66 ms | 22.70 ms |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-euler | 200 | 240.88 ms | 406.94 ms | 959.1 µs |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-euler | 2000 | 241.11 ms | 421.41 ms | 3.70 ms |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-euler | 8000 | 239.21 ms | 428.13 ms | 12.72 ms |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-rk4 | 200 | 239.61 ms | 465.02 ms | 1.17 ms |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-rk4 | 2000 | 278.82 ms | 488.32 ms | 5.95 ms |
| 0.4.38 | 0.7.2 | sir_adjust | diffrax-rk4 | 8000 | 243.11 ms | 496.79 ms | 22.57 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-euler | 200 | 242.01 ms | 457.51 ms | 1.11 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-euler | 2000 | 244.33 ms | 461.99 ms | 4.70 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-euler | 8000 | 241.17 ms | 478.80 ms | 17.71 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-rk4 | 200 | 240.21 ms | 507.29 ms | 1.66 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-rk4 | 2000 | 240.65 ms | 514.66 ms | 8.46 ms |
| 0.4.38 | 0.7.2 | sir_tv | diffrax-rk4 | 8000 | 242.95 ms | 546.49 ms | 30.81 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-euler | 200 | 513.93 ms | 391.08 ms | 2.61 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-euler | 2000 | 507.58 ms | 430.31 ms | 20.07 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-euler | 8000 | 524.09 ms | 469.50 ms | 77.60 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-rk4 | 200 | 515.80 ms | 466.25 ms | 5.89 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-rk4 | 2000 | 525.88 ms | 524.77 ms | 54.41 ms |
| 0.4.38 | 0.7.2 | age_mix | diffrax-rk4 | 8000 | 521.78 ms | 667.91 ms | 213.57 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-euler | 200 | 564.08 ms | 433.22 ms | 2.51 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-euler | 2000 | 555.98 ms | 450.50 ms | 19.97 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-euler | 8000 | 519.06 ms | 543.11 ms | 78.20 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-rk4 | 200 | 513.74 ms | 479.82 ms | 5.88 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-rk4 | 2000 | 515.46 ms | 526.06 ms | 54.52 ms |
| 0.4.38 | 0.7.2 | age_mix_tv | diffrax-rk4 | 8000 | 514.58 ms | 688.47 ms | 212.23 ms |
| 0.4.38 | 0.7.2 | stress | diffrax-euler | 200 | 777.15 ms | 1.349 s | 455.46 ms |
| 0.4.38 | 0.7.2 | stress | diffrax-euler | 2000 | 707.18 ms | 5.687 s | 4.614 s |
| 0.4.38 | 0.7.2 | stress | diffrax-euler | 8000 | 751.40 ms | 21.606 s | 19.437 s |
| 0.4.38 | 0.7.2 | stress | diffrax-rk4 | 200 | 728.86 ms | 2.185 s | 1.236 s |
| 0.4.38 | 0.7.2 | stress | diffrax-rk4 | 2000 | 2.482 s | 20.624 s | 14.032 s |
| 0.4.38 | 0.7.2 | stress | diffrax-rk4 | 8000 | 743.09 ms | 68.293 s | 87.509 s |
| 0.6.2 | 0.7.2 | sir | diffrax-euler | 200 | 237.28 ms | 551.45 ms | 1.21 ms |
| 0.6.2 | 0.7.2 | sir | diffrax-euler | 2000 | 327.81 ms | 1.133 s | 6.60 ms |
| 0.6.2 | 0.7.2 | sir | diffrax-euler | 8000 | 289.75 ms | 640.85 ms | 12.77 ms |
| 0.6.2 | 0.7.2 | sir | diffrax-rk4 | 200 | 198.93 ms | 609.47 ms | 1.19 ms |
| 0.6.2 | 0.7.2 | sir | diffrax-rk4 | 2000 | 192.36 ms | 578.25 ms | 5.69 ms |
| 0.6.2 | 0.7.2 | sir | diffrax-rk4 | 8000 | 185.99 ms | 943.25 ms | 24.11 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-euler | 200 | 231.16 ms | 727.25 ms | 3.06 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-euler | 2000 | 314.34 ms | 758.18 ms | 4.22 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-euler | 8000 | 300.46 ms | 831.19 ms | 17.88 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-rk4 | 200 | 485.51 ms | 1.178 s | 2.13 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-rk4 | 2000 | 225.98 ms | 729.36 ms | 6.44 ms |
| 0.6.2 | 0.7.2 | sir_adjust | diffrax-rk4 | 8000 | 219.83 ms | 811.74 ms | 26.31 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-euler | 200 | 227.07 ms | 805.10 ms | 2.16 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-euler | 2000 | 251.06 ms | 801.48 ms | 13.58 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-euler | 8000 | 211.54 ms | 748.04 ms | 34.08 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-rk4 | 200 | 236.39 ms | 899.24 ms | 2.31 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-rk4 | 2000 | 223.39 ms | 731.44 ms | 7.55 ms |
| 0.6.2 | 0.7.2 | sir_tv | diffrax-rk4 | 8000 | 219.13 ms | 711.69 ms | 33.78 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-euler | 200 | 615.92 ms | 673.80 ms | 1.84 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-euler | 2000 | 518.39 ms | 598.16 ms | 5.99 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-euler | 8000 | 515.30 ms | 1.123 s | 27.45 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-rk4 | 200 | 864.42 ms | 846.40 ms | 2.44 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-rk4 | 2000 | 505.56 ms | 1.898 s | 19.24 ms |
| 0.6.2 | 0.7.2 | age_mix | diffrax-rk4 | 8000 | 795.03 ms | 1.095 s | 57.12 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-euler | 200 | 881.75 ms | 1.346 s | 1.95 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-euler | 2000 | 642.06 ms | 677.34 ms | 6.10 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-euler | 8000 | 575.63 ms | 683.77 ms | 20.67 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-rk4 | 200 | 581.06 ms | 830.26 ms | 2.21 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-rk4 | 2000 | 532.83 ms | 771.39 ms | 15.15 ms |
| 0.6.2 | 0.7.2 | age_mix_tv | diffrax-rk4 | 8000 | 585.70 ms | 849.95 ms | 57.38 ms |
| 0.6.2 | 0.7.2 | stress | diffrax-euler | 200 | 844.03 ms | 3.241 s | 58.62 ms |
| 0.6.2 | 0.7.2 | stress | diffrax-euler | 2000 | 1.021 s | 2.150 s | 612.29 ms |
| 0.6.2 | 0.7.2 | stress | diffrax-euler | 8000 | 882.77 ms | 3.036 s | 1.597 s |
| 0.6.2 | 0.7.2 | stress | diffrax-rk4 | 200 | 948.37 ms | 2.153 s | 151.61 ms |
| 0.6.2 | 0.7.2 | stress | diffrax-rk4 | 2000 | 799.85 ms | 3.078 s | 1.031 s |
| 0.6.2 | 0.7.2 | stress | diffrax-rk4 | 8000 | 623.40 ms | 6.239 s | 5.075 s |
| 0.11.1 | 0.7.2 | sir | diffrax-euler | 200 | 934.76 ms | 1.394 s | 1.92 ms |
| 0.11.1 | 0.7.2 | sir | diffrax-euler | 2000 | 1.056 s | 1.299 s | 8.51 ms |
| 0.11.1 | 0.7.2 | sir | diffrax-euler | 8000 | 659.83 ms | 1.234 s | 27.95 ms |
| 0.11.1 | 0.7.2 | sir | diffrax-rk4 | 200 | 1.048 s | 1.493 s | 2.90 ms |
| 0.11.1 | 0.7.2 | sir | diffrax-rk4 | 2000 | 1.167 s | 1.425 s | 11.78 ms |
| 0.11.1 | 0.7.2 | sir | diffrax-rk4 | 8000 | 1.124 s | 1.540 s | 49.17 ms |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-euler | 200 | 994.22 ms | 1.530 s | 2.49 ms |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-euler | 2000 | 1.095 s | 1.436 s | 7.33 ms |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-euler | 8000 | 438.92 ms | 373.61 ms | 8.61 ms |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-rk4 | 200 | 218.01 ms | 510.51 ms | 789.7 µs |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-rk4 | 2000 | 212.40 ms | 2.372 s | 8.43 ms |
| 0.11.1 | 0.7.2 | sir_adjust | diffrax-rk4 | 8000 | 248.52 ms | 446.63 ms | 13.01 ms |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-euler | 200 | 187.06 ms | 405.52 ms | 923.4 µs |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-euler | 2000 | 187.83 ms | 385.26 ms | 5.55 ms |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-euler | 8000 | 188.33 ms | 385.21 ms | 20.32 ms |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-rk4 | 200 | 189.61 ms | 424.81 ms | 823.0 µs |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-rk4 | 2000 | 184.52 ms | 423.03 ms | 4.81 ms |
| 0.11.1 | 0.7.2 | sir_tv | diffrax-rk4 | 8000 | 195.18 ms | 419.87 ms | 17.73 ms |
| 0.11.1 | 0.7.2 | age_mix | diffrax-euler | 200 | 429.42 ms | 358.52 ms | 659.2 µs |
| 0.11.1 | 0.7.2 | age_mix | diffrax-euler | 2000 | 496.74 ms | 350.29 ms | 3.39 ms |
| 0.11.1 | 0.7.2 | age_mix | diffrax-euler | 8000 | 448.62 ms | 378.67 ms | 13.44 ms |
| 0.11.1 | 0.7.2 | age_mix | diffrax-rk4 | 200 | 430.32 ms | 400.39 ms | 1.14 ms |
| 0.11.1 | 0.7.2 | age_mix | diffrax-rk4 | 2000 | 419.74 ms | 635.84 ms | 8.62 ms |
| 0.11.1 | 0.7.2 | age_mix | diffrax-rk4 | 8000 | 424.39 ms | 404.74 ms | 32.99 ms |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-euler | 200 | 442.27 ms | 358.56 ms | 684.0 µs |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-euler | 2000 | 439.32 ms | 380.96 ms | 3.40 ms |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-euler | 8000 | 426.01 ms | 365.16 ms | 13.25 ms |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-rk4 | 200 | 425.22 ms | 406.47 ms | 1.17 ms |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-rk4 | 2000 | 427.43 ms | 437.62 ms | 9.31 ms |
| 0.11.1 | 0.7.2 | age_mix_tv | diffrax-rk4 | 8000 | 428.53 ms | 453.70 ms | 34.93 ms |
| 0.11.1 | 0.7.2 | stress | diffrax-euler | 200 | 526.73 ms | 603.57 ms | 30.17 ms |
| 0.11.1 | 0.7.2 | stress | diffrax-euler | 2000 | 531.75 ms | 861.89 ms | 293.96 ms |
| 0.11.1 | 0.7.2 | stress | diffrax-euler | 8000 | 532.67 ms | 1.889 s | 1.103 s |
| 0.11.1 | 0.7.2 | stress | diffrax-rk4 | 200 | 537.72 ms | 700.63 ms | 82.00 ms |
| 0.11.1 | 0.7.2 | stress | diffrax-rk4 | 2000 | 534.12 ms | 1.429 s | 824.24 ms |
| 0.11.1 | 0.7.2 | stress | diffrax-rk4 | 8000 | 585.91 ms | 4.689 s | 4.061 s |

JSON: `benchmarks/recorded-summer4-jax-<version>.json`.
Regenerate with `pixi run bench-models-jax-sweep`.
