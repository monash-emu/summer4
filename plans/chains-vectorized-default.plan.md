<!--
Accepted 2026-11-11 (user: "Vectorized, then tag"). Historical record: do not edit.
-->

# Hotfix: MCMC chains stay vectorized by default

Branch: `fix/chains-vectorized-default`, cut from `main` at `6c8d4b3`, the merge of step 49.

## Problem

Step 49 (`plans/runtime-defaults.plan.md`, D3) made `chain_method="auto"` resolve to `"parallel"`
when there are at least as many devices as chains. In CI's `latest` environment (JAX 0.11.1,
numpyro 0.21), parallel chains fail when they take gradients through the new adaptive default
solver (Tsit5, diffrax). The error comes from equinox's checkpointed while-loop backward pass:
"Closure-converted function called with different dynamic arguments".

Reproduced on the mini with notebook 31:

| Setup | Result |
|---|---|
| Parallel chains with Tsit5 | fails |
| Vectorized chains with Tsit5 | passes |
| Parallel chains with Euler | passes |

The `default` environment (JAX 0.6.2) passes everything.

## Fix

- `resolve_chain_method("auto", n)` returns `"vectorized"`. Explicit methods pass through
  unchanged, and the function stays public.
- `"parallel"` remains available on request. A test covers it with `solver="euler"`.
- New tracking:
  - `futureplans/parallel-chains-diffrax-grad.md`;
  - open finding CX45 under package CP8. CX44 is narrowed to host devices.
- Guide pages 08 and 10, notebook 36, `docs/releases.md` and the CHANGELOG no longer claim that
  chains run in parallel by default.
- `v0.3.0a2` is tagged on this fix's merge once CI passes in both environments.

## Done when

Parallel chains with Tsit5 and NUTS pass in the `latest` environment. Then `"auto"` goes back to
choosing `"parallel"` when the device count allows, which closes CX45.
