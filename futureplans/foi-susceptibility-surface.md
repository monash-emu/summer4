# ForceOfInfection lacks a first-class susceptibility surface

## Concern

`ForceOfInfection(infectiousness=...)` owns infectiousness weights, either a
trait map or selector pairs applied per compartment, optionally
population-normalised. There is no symmetric
`susceptibility=` on the FOI. Textbook chapter 15 therefore scales
susceptibility with flow `adjust=[Multiply(..., where=group[trait])]` on the
infection `TransitionFlow`, or by row-scaling a `MixingMatrix` with
`normalize="none"`.

That works, but it splits a related pair of concepts across two APIs and makes
it easy to bake susceptibility into the mixing matrix (which the chapter
argues against).

Noticed while porting chapter 15 on `docs/textbook-catchup`.

## Pointers

- `src/summer4/epi/infection.py` — `coerce_compartment_weights` and
  `apply_compartment_weights` are the selector-keyed weight machinery.
  Infectiousness already uses them, multiplied per compartment before the
  group sum. Susceptibility should call the same two functions on the
  recipient side, after mixing, and must not reuse `scale_infectious_pool`:
  susceptibility is not normalised. A selector that is not a trait of
  `group_by` cannot be folded into the `GroupedRate`; apply it to the
  compartment-aligned rate.
- `docs/textbook/15-susceptibility-infectiousness-matrices.ipynb` — Multiply /
  row-scale workaround

## Done looks like

A FOI-owned susceptibility map parallel to infectiousness, e.g.
`ForceOfInfection(susceptibility=...)` that multiplies per-group $\lambda$
(or the infection edges sourced from those groups) without requiring the user
to attach `TransitionFlow.adjust` by hand, plus a short user-guide note that
mixing matrices stay for contact structure.
