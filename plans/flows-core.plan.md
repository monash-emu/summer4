---
name: flows-core
overview: "Phase 1 of flows/derived-outputs — promote the flows spike into summer4 with EdgeMap, Present/Absent non-binding, CompiledModel, JAX-only vector field; retire explorations/flows."
todos:
  - id: binding
    content: "Present/Absent non-binding; selector_values vs selector_properties; strict_pairing"
    status: completed
  - id: edgemap
    content: "EdgeMap + Source/Dest rewrite + moves_mask + labels"
    status: completed
  - id: join
    content: "Fix pack overflow via np.unique; vectorise expansion; split FlowEdges by kind"
    status: completed
  - id: promote
    content: "summer4.flows + summer4.jax; CompiledModel; JAX-only field; typed DerivedFn"
    status: completed
  - id: retire
    content: "Delete spike; rewrite docs/ledger; drop explore-flows task"
    status: completed
  - id: notebook-tests
    content: "03-flows.ipynb + ported tests + phase1 gates; pixi checks"
    status: completed
isProject: true
---

# Phase 1 — Promote the flows spike, with edge maps (`feat/flows-core`)

Parent design: [flows-derived-outputs.plan.md](flows-derived-outputs.plan.md).

Promote `explorations/flows/` into `src/summer4/flows/` and `src/summer4/jax/`, add
`EdgeMap`, and return a `CompiledModel` from `compile()`.

## Scope (from parent plan §§1a–1f)

1. **Present/Absent non-binding.** Binding = `Trait`/`IsIn` only (`selector_values`).
   `strict_pairing=True` raises when an unbound property would move people.
2. **`EdgeMap`.** Doubled `PropertyMap` table (`{p}@source` / `{p}@dest` + markers);
   `Source`/`Dest` rewrite; `moves_mask`; demangled `labels()`.
3. **Join fixes.** Drop `_pack_keys`; `np.unique` axis=0; vectorise expansion;
   `TransitionEdges` / `ExitEdges` / `EntryEdges`.
4. **`CompiledModel`.** Static; `edges(name)`; `vector_field`; digest covers floats.
5. **mypy --strict.** JAX-only compiled field; NumPy Euler/reference in tests;
   `isinstance` for `PropertyData`; typed `DerivedFn` Protocol.
6. **Retire spike.** Delete explorations + sync tests; move docs into user guide;
   collapse Spike column in the ledger → **22 / 52 (42%)** with Q3 → `full`.

## Acceptance

- Notebook: `examples/notebooks/03-flows.ipynb`
- Gates: `strict_pairing` raises on `age.present() & state["S"] → state["I"]`;
  Source/Dest Kleene table including exit `Dest(Everything())` UNKNOWN;
  four-free-property join matches brute force.

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch && pixi run coverage
```
