---
name: sparse-targets
description: Phase 5 of flows/derived-outputs — declarative calibration targets that merge their observation times into the save plan, so a likelihood run never materialises dense outputs.
---

# Phase 5 — Sparse outputs and calibration targets (`feat/sparse-targets`)

## Context

`plans/flows-derived-outputs.plan.md` §"Phase 5" is the accepted design; this is
its execution plan. It covers the calibration case: sparse observed data, and no
wish to materialise dense outputs at all.

Phases 3 and 4 make it cheap. Phase 3 taught the save plan about per-request
times and grouped them into `SubSaveAt`s; Phase 4 made flow outputs queryable.
What is missing is the declarative object that says "these forty observations,
at these dates, against this output" and turns that into a save plan.

**Closes: no API rows.** Sparse saving has no summer2 equivalent, so the
percentage does not move, and this plan says so rather than inflating it.
Declare it in the ledger's packages block the way WP9 and WP10 already are:

```
| WP11 | Sparse outputs and calibration targets | *(no API rows; unblocks WP10 / textbook 20)* |
```

(WP8 does not exist in the ledger's numbering; WP11 avoids implying it does.)
Its value is unblocking WP10 and making `D1`–`D7` usable at calibration scale.

Depends on Phase 3 (`feat/diffrax-solver`) and Phase 4 (`feat/flow-outputs`).

## 5a. `src/summer4/results/targets.py`

```python
@dataclass(frozen=True, slots=True)
class Target:
    key: str
    times: NDArray[np.float64]
    values: NDArray[np.float64]
    quantity: Quantity | None = None   # what to save when `key` is not in the plan
    dispersion: str | None = None      # names a nuisance parameter; WP10 consumes it

    @classmethod
    def from_series(cls, key: str, s: "pd.Series", epoch: Epoch, *,
                    quantity: Quantity | None = None) -> Target: ...
    def contribute(self, plan: SavePlan) -> SavePlan: ...


@dataclass(frozen=True, slots=True)
class TargetSet:
    targets: tuple[Target, ...]

    def plan(self, base: SavePlan) -> SavePlan: ...
    def gather(self, result: Result) -> dict[str, Any]: ...
    def residuals(self, result: Result) -> dict[str, Any]: ...
```

`from_series` converts a pandas `DatetimeIndex` through `Epoch.to_model`; pandas
stays a lazy import inside the method, never a hard dependency.

`contribute` merges the target's times into its request's `ts` — sorted union,
deduped — and, when `key` is absent from the plan, adds
`SaveRequest(quantity, ts=self.times)`. When `key` is absent and no `quantity`
was given it raises, listing the plan's keys. It rests entirely on Phase 3's
`SaveRequest.ts` support and `group_requests`.

`TargetSet.plan` folds every contribution over a base plan; equal times collapse
into one save group because Phase 3 groups on the bytes of `ts`, not on `id()`.

Adopt estival's two good structural ideas:

- compile a declarative target into an evaluator holding a **precomputed integer
  index array**;
- unroll the Python loop over targets at trace time, so each target becomes a
  static gather in the jaxpr. AGENTS.md endorses exactly this shape: a small
  static set of named targets, folded into batched ops.

Drop its two problems:

- the exact-index requirement that silently discards off-grid observations —
  Phase 3 interpolates;
- the mutable global whitelist it has to flip on and off around two `get_runner`
  calls. Here **the plan is the request**: a likelihood run and a reporting run
  are two `SavePlan`s, not two transient states of one shared model.

**Scope.** This phase ships gathering and residuals. The probabilistic
likelihood and the priors that `dispersion` feeds are WP10; the field is declared
here and consumed there. Say so in the module docstring rather than half-building
a likelihood.

## 5b. One user-facing path, two strategies underneath

`TargetSet.gather` calls `result[key].at_times(target.times)` in both cases, and
the user's source is identical either way:

- **diffrax** — `contribute` merged the observation times into the group's `ts`,
  so the solver hit them exactly and `at_times` is a no-op gather;
- **Euler** — the model saves on its grid and `TimeAxis.weights_for` supplies
  the precomputed `(idx, weight)` pair that `at_times` already applies
  (`results/trace.py:170`).

**Name the Euler cost honestly** in the docstring. Merging off-grid times into
the save grid means the group's `ts` is no longer an arithmetic subgrid, so
`_is_arithmetic_subgrid` fails and the general path runs: one `lax.scan` over
every step with the whole state trajectory stacked, then a lerp. It is correct,
and it is slower and heavier than the nested-scan fast path. A user calibrating
under Euler should know why.

## 5c. Tests — `tests/test_targets.py`

- `from_series` with an `Epoch` round-trips dates to model times.
- `contribute` merges, dedupes and is idempotent; two targets with equal times
  produce one save group.
- An off-grid observation is hit to `rtol=1e-6` under diffrax and within `1e-3`
  under Euler on a fine grid — **from the same source**.
- **The phase gate:** a fit against ~40 sparse observations recovers known
  parameters to within 2%, using `optax` (declared in `pyproject.toml:22` and
  `pixi.toml:18`, and unused).
- `describe()` reports a smaller footprint for the sparse plan than the dense
  one, and the two agree at the observation times. This depends on Phase 3
  sizing each output from its own group's `ts`.
- A target naming an absent key with no `quantity` raises informatively.
- `jax.make_jaxpr` of the loss: the gather count scales with the number of
  targets, not with trajectory length. No test currently exercises
  `make_jaxpr`, and AGENTS.md's JAX guidance item 2 asks for exactly this.

## 5d. Notebook and the user gate

**Feature notebook** — `examples/notebooks/07-targets.ipynb`: fit an SIR to ~40
scattered observations, with `model.describe(plan)` reporting the memory the
sparse plan saves against the dense one, and the recovered parameters asserted
against the truth.

**Harvest: nothing.** Run `pixi run coverage` as every phase does, and record
the honest outcome — textbook 20, *Calibration and uncertainty*, needs a
Bayesian calibration workflow (WP10) and stays `none`. No chapter or summer2
page becomes publishable in this phase. Do not invent a port to fill the slot;
"nothing newly publishable" is a legitimate result of a harvest step, and this
phase's contribution is that it unblocks WP10 rather than that it closes rows.

> ### Notebook gate — blocking
>
> Do not merge until the user has run this in `pixi run notebook` and ticked:
>
> - [ ] `examples/notebooks/07-targets.ipynb` — the fit converges to the known
>   parameters, and the memory saving is shown as a number rather than asserted
>   in prose.

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

Ledger discipline: no API row changes. Add the `WP11` row to the packages block
and run `pixi run coverage-write` — the progression table will show `WP11` at
the same count as `WP7`, which is the correct and honest rendering of a package
that closes no symbols. `docs/evaluation/index.md` totals do not move.
