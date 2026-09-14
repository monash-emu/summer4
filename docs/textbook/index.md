# Textbook

A textbook of infectious-disease modelling built on summer4, adapted from the
[Monash EMU summer textbook](https://github.com/monash-emu/summer-textbook).

```{admonition} Attribution
:class: note

The source textbook is BSD-2-Clause licensed, Copyright (c) 2022, monash-emu.
The chapters here are adaptations: the conceptual material is rewritten and the
code, where any exists, is written against the summer4 API.
```

```{admonition} Two of twenty chapters
:class: warning

The source textbook has twenty chapters. Most of them build, run and plot a
model. summer4 can compile flows to a vector field but `euler` returns a
final state only, so those chapters are not published as runnable notebooks
yet. Chapter 1 is prose and ports directly. Chapter 2's *structural* half is
reproducible today and is given below as "Model structures".

Everything else is catalogued in {doc}`roadmap` with the specific API that
blocks it. Flows themselves are documented in {doc}`../user/08-flows`.
```

```{toctree}
:maxdepth: 2

01-introduction
02-model-structures
roadmap
```
