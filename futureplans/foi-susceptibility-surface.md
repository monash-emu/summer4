# ForceOfInfection lacks a first-class susceptibility surface

## Concern

`EpiModel.add_infectiousness_adjustments` / `ForceOfInfection(infectiousness=...)`
own per-group infectiousness weights (optionally population-normalised). There
is no symmetric `add_susceptibility_adjustments` or `susceptibility=` on the
FOI. Textbook chapter 15 therefore scales susceptibility with flow
`adjust=[Multiply(..., where=group[trait])]` on the infection
`TransitionFlow`, or by row-scaling a `MixingMatrix` with `normalize="none"`.

That works, but it splits a related pair of concepts across two APIs and makes
it easy to bake susceptibility into the mixing matrix (which the chapter
argues against).

Noticed while porting chapter 15 on `docs/textbook-catchup`.

## Pointers

- `src/summer4/epi/model.py` — `add_infectiousness_adjustments` only
- `src/summer4/epi/infection.py` — `infectiousness` / `normalize_infectiousness`
- `docs/textbook/15-susceptibility-infectiousness-matrices.ipynb` — Multiply /
  row-scale workaround

## Done looks like

A FOI-owned susceptibility map parallel to infectiousness, e.g.
`EpiModel.add_susceptibility_adjustments(prop, weights, normalize=...)` that
multiplies per-group $\lambda$ (or the infection edges sourced from those
groups) without requiring the user to attach `TransitionFlow.adjust` by hand,
plus a short user-guide note that mixing matrices stay for contact structure.
