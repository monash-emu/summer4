# Textbook

A textbook of infectious-disease modelling built on summer4, adapted from the
[Monash EMU summer textbook](https://github.com/monash-emu/summer-textbook).

```{admonition} Attribution
:class: note

The source textbook is BSD-2-Clause licensed, Copyright (c) 2022, monash-emu.
The chapters here are adaptations: the conceptual material is rewritten and the
code is written against the summer4 API. See {doc}`porting` for the convention.
```

Status of every chapter lives in the
{doc}`coverage ledger <../evaluation/coverage-ledger>` (including which pages
are already ported). This tree holds the published notebooks only.

```{toctree}
:maxdepth: 2

01-introduction
02-model-structures
03-thinking-about-flows
04-thinking-about-flow-rates
05-series-compartments-latency
06-post-infection-immunity
07-numerical-solutions
08-derived-outputs
09-transmission-assumptions
10-reproduction-number
11-cyclical-epidemic-dynamics
12-heterogeneous-mixing-intro
13-mixing-and-transmission-types
14-assortative-mixing
15-susceptibility-infectiousness-matrices
porting
roadmap
```
