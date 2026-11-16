---
name: param-split
description: Let a flow's split= proportions be rate expressions (Param, arithmetic, time), renormalised at run time, and document split= and adjust= together.
---

# Parameterised `split=` proportions

## Context

A tb_acf modeller reported that summer4 "doesn't support parameters for flow adjustments", after
writing one `TransitionFlow` per destination focus stratum with complementary rates
(`rate * (1 - p)`, `rate * p`). Adjustments do take parameters (`Multiply(Param(...),
where=Dest(...))`), but `split=` proportions were cast to `float` in `_normalize_split`, so the
direct spelling, `split={focus: {"low": 1 - p, "high": p}}`, was impossible. The guide also had no
example of a parameter in an adjustment, a `Dest(...)` mask, or the shape of the `adjust=` list.

## Design

- A split property whose proportions include any `RateOps` is **dynamic**. Plain-number
  properties keep the static path unchanged: validated `>= 0`, sum to 1, folded into the
  host-side edge weights.
- A dynamic property still has to name every trait that appears on the destination. It is left
  out of the static weights (equal share), and actualize binds a `DynamicSplit`
  (`flows/join.py`): per property, the proportion expressions in trait order, a per-edge trait
  code, and a renormalisation group per edge.
- Groups match the static normalisation: the edges of one source compartment within one
  identity join (one pairing part for `TraitChain` / `TraitMatrix`), or one destination group for
  an `EntryFlow` (`sum_by` grouping). Edges alone in their group keep their static weight.
- At run time (`_dynamic_split_weight` in `flows/compiled.py`) each proportion is evaluated as a
  scalar per batch, gathered onto edges, multiplied into the static weight, and divided by its
  group total with a segment sum. Proportions are therefore relative weights. A group whose
  proportions are all zero carries no flow; a double `where` keeps gradients finite there.
- Proportion expressions join every pass that walks a flow's rate trees: hoisting roots
  (`stages.roots_of`), field-path validation, `FlowRef` dependencies and topological order
  (`actualize.flow_rate_refs`), capture metadata, and the model digest.

## Deliverables

1. Implementation in `src/summer4/flows/{join,actualize,compiled,stages}.py`.
2. Tests: `tests/test_param_split.py` (equivalence with float splits on ragged maps, pairing
   parts and entry flows; the two-flow idiom; relative weights; all-zero groups; gradient;
   hoisting on and off; time-varying proportions; composition with `Dest` adjustments;
   validation; digest).
3. Guide: `docs/guide/05-stratification.ipynb` gains "Splitting a flow by a parameter" and
   "What an adjustment can be" (the three kinds, bare values, `where` / `Dest` / `Source`,
   ordering by kind and `precedence=`).
4. Acceptance notebook `examples/notebooks/37-param-split.ipynb`: (A) one flow equals tb_acf's
   hand-written flows, (B) the split follows the parameter in one compiled model, (C) a
   time-varying proportion, (D) `jax.grad` against finite differences.

No coverage-ledger row changes: `split=` was already `full` for numeric proportions, and
summer2 had no parameterised destination split to compare against.
