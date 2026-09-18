---
name: cookbook-custom-rates
description: Docs-only cookbook section with a custom contact-process recipe for domain experts.
---

# Cookbook — custom contact processes (`docs/cookbook-custom-rates`)

## Context

Domain experts often need a custom per-capita hazard (force of infection
variants, predation, birth ∝ population) but should not start from
`register_rate_eval`. The right teaching order is: compose with `Reduce`, then
customise `ForceOfInfection(kind=...)`, then (appendix) a named `RateOps`
subclass.

## Deliverable

Docs-only branch. No `src/summer4` changes.

1. **`docs/cookbook/`** — new documentation stream with `index.md`.
2. **`docs/cookbook/01-custom-rates.ipynb`** — three-rung recipe:
   - Rung 1: `Reduce` arithmetic matches `ForceOfInfection(kind="frequency")`;
     Lotka–Volterra via `ExitFlow` / `EntryFlow` / `FlowRef`.
   - Rung 2: callable `kind=` bit-matches `"frequency"`; power-frequency example.
   - Appendix: minimal `ForceOfPredation` via `register_rate_eval`, asserted
     equal to the Reduce-based LV model.
3. Wire the stream into `docs/index.md` (toctree + streams table) and mention it
   in `docs/dev/documentation.md`.

## Acceptance

- Notebook clears `pixi run check-notebooks`.
- Claims are asserted (no print-only proofs).
- `pixi run -e docs docs-strict` builds with the new section (or at least the
  notebook executes cleanly under the docs env).

```bash
pixi run check-notebooks
pixi run -e docs docs-strict
pixi run check-branch
```
