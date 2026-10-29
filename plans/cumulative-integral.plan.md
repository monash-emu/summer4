<!--
Accepted 2026-10-30. Historical record once merged: do not edit. Follow-up work goes in a new plan.
-->

# `cumulative_integral` and `cumsum`: split `Output.cumulative()`

## Context

`Output.cumulative()` is a plain running sum along time. It is a cumulative
count only when each saved value is already an amount for its interval: the
output of `integrate_intervals()`, a summer2-parity `midpoint()` series, or
observed counts. Users nearly always apply it to a flow **rate** (`FlowMass`
saves people per unit time), where it is dimensionally wrong (people only when
the save spacing is 1) and off by one sample (it counts `f(t0)` as a whole
interval). The correct idiom, `integrate_intervals().cumulative()`, is two calls
and returns `T-1` rows timed at each panel's right edge.

Changing `cumulative()` to integrate would silently double-integrate the
documented `integrate_intervals().cumulative()` idiom, so the name is split, as
`sum_by` replaced `total()` / `sum_over()` (`plans/sum-by.plan.md`).

## Shape

- `cumsum(*, start=, end=)` on `Output`, `OutputExpr` and the save-spec chain:
  what `cumulative()` returns today, under the exact name, for amounts per
  interval.
- `cumulative_integral(*, start=, end=, method="trapezoid")` on the same three:
  the running trapezoid integral over time of a rate. Same length `T` as the
  input, 0 at the first included save time, the interval widths from the
  concrete save times. `start` / `end` must be save times and points outside
  the window are zero, as for `cumsum`. Equal to `integrate_intervals().cumsum()`
  with a leading 0. Vectorised (one `cumsum` over panel areas), so the jaxpr
  does not grow with `T`. Only the trapezoid rule: Simpson panels do not give a
  value at every save time.
- `cumulative()` keeps returning the `cumsum` values and emits a
  `DeprecationWarning` naming both replacements (`_deprecation.warn_cumulative`).
  On an expression the warning fires once, at construction; the node records
  `cumsum`, so evaluation does not warn again. Removal is a later release.
- `OutputExpr._call` takes its method name positionally only, so a call can
  forward a `method=` keyword (`cumulative_integral(method=...)`).

Composability: accumulation (`cumsum`) and integration
(`integrate_intervals`) stay separate public pieces; `cumulative_integral` is the
convenience built from them, and a test checks the equivalence. When `OutputExpr`
is replaced by `RateOps` post-ops (`CX30`, roadmap steps 36–38), `cumsum` and
`cumulative_integral` must carry over from `_CALLS`.

## Migration (rate or amount, per call)

- Rates → `cumulative_integral()`: `examples/notebooks/06-flow-outputs.ipynb`
  (`deaths`), `examples/notebooks/32-sum-by.ipynb` (infection rate),
  `docs/case-studies/age-stratified-seirs.ipynb` (notifications are a
  `FlowMass`), `docs/summer2/03-derived-outputs.ipynb` (first example),
  `tests/test_epi_calibration.py` (`cum_I`, prevalence → person-days).
- Amounts → `cumsum()`: every `integrate_intervals().cumulative()` in
  `docs/summer2/03`, `10`, `12`; `midpoint()` series in notebook 06's
  `OutputSet`, `benchmarks/tb_scale.py` and `tests/test_output_set.py`;
  `tests/test_output_algebra.py`, `tests/test_results.py`.
- `docs/summer2/03-derived-outputs.ipynb`'s summary row said
  "(or `.cumulative()` on rates)", which was wrong; it now names
  `.cumulative_integral()` on a rate or `.integrate_intervals().cumsum()`.
- Ledger implementation columns: `D4` (`coverage-ledger.md`), `KI15`
  (`tb-ports.md`), `feature-completeness.md`. No status changes.

## Tests

`tests/test_cumulative_integral.py`: hand trapezoid on a non-uniform grid;
`start` / `end` windows, off-grid and reversed bounds, non-trapezoid method;
equality with `integrate_intervals().cumsum()` shifted by one point and with
`integrate()`; a `PropertyData` output keeps its map; one save time; `jit` and
`grad`; jaxpr equation count equal at `T = 10` and `T = 2000`; on a closed SIR,
the integral of the infection rate follows the fall in S at save spacings 0.5,
1, 2 and 5, while the running sum scales with 1 / spacing; the deprecated
spelling on `Output`, `OutputExpr` and the save-spec chain warns naming both
replacements and equals `cumsum`; an `OutputSet` evaluates both new calls.

## Notebook

`examples/notebooks/34-cumulative-integral.ipynb` (acceptance; 33 is taken by
`feat/float64-default`): cumulative infections from S depletion against
`cumulative_integral()` and the old `cumulative()` with saves every 2 days; final
counts by save spacing; the two-step idiom and a `start=` window; jaxpr size
against `T`; the deprecation table.
