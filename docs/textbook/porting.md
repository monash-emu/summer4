# Textbook porting convention

How a summer textbook chapter becomes a runnable page under `docs/textbook/`.

## Licence and attribution

The source textbook is BSD-2-Clause, Copyright (c) 2022, monash-emu. Every
ported chapter carries that notice, names
[monash-emu/summer-textbook](https://github.com/monash-emu/summer-textbook), and
pins the **commit SHA** the prose was adapted from so a later reader can diff
against what was carried.

## Figures

Vendored figures live in `docs/textbook/figures/<chapter>/`. The licence text
and copyright line for those files belong in `docs/textbook/figures/LICENSE`.

## Plotting

Plot with Plotly (already in the `docs` and `nb` pixi environments), driven
from `Trace.to_pandas()`. Do not introduce a second plotting stack.

## Code

Carry and adapt the prose. **Write the code in current summer4 idiom**
(`from summer4 import ...`, `CompiledModel.run`, `SavePlan`, `Result`) — never
transliterate summer2 / summer API calls.

## Ledger

When a chapter ships, set its textbook ledger row to `full` (or keep `partial`
with an honest blocker) and fill the `Ported` cell with the path relative to
`docs/`. Declared ports must exist; `pixi run coverage` fails otherwise.
