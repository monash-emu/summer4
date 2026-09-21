---
name: integrate-intervals
overview: Rename Output.incidence to integrate_intervals so epidemiological words stay in summer4.epi.
todos:
  - id: rename
    content: Rename Output.incidence to integrate_intervals with no alias
    status: completed
  - id: docs
    content: Update tests, notebooks, and living docs; leave historical plans
    status: completed
isProject: true
---

# Rename `Output.incidence` to `integrate_intervals`

Per-interval quadrature is a results operation, not an epidemiological quantity. The method does not choose which series it integrates. The epidemiological name belongs only under `summer4.epi`.

## Do

- `Output.incidence` becomes `Output.integrate_intervals`. No alias.
- Error strings and `op=` labels use the new name. `integrate` points at `integrate_intervals`.
- `FlowMass` docstring, tests, example notebooks, and living docs (`docs/`, `futureplans/`, the roadmap) use the new name.
- `futureplans/state-ledgers-incidence.md` becomes `futureplans/state-ledgers-flow-integral.md`.
- Core docstrings do not use force-of-infection or incidence as examples. Those words stay in `src/summer4/epi/`.

## Not in this branch

- Historical `plans/*.plan.md`. They record the name the code had when they were written.
- Textbook and summer2 prose that discusses clinical incidence, and user-chosen output keys such as `res["incidence"]`.
- Name-aligned `Output` algebra on `feat/trace-algebra`. That branch's `midpoint` docstring and notebook still say incidence; update them when the branches meet.
