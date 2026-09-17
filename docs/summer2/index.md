# summer2 documentation ports

Runnable adaptations of pages from the
[summer2 documentation](https://summer2.readthedocs.io).

```{admonition} Attribution
:class: note

The source documentation is BSD-2-Clause licensed, Copyright (c) 2022,
monash-emu ([monash-emu/summer2](https://github.com/monash-emu/summer2)).
Pages here adapt the prose and rewrite every line of code against the summer4
API. See {doc}`../textbook/porting` for the shared convention.
```

Status of every summer2 docs page lives in the
{doc}`coverage ledger <../evaluation/coverage-ledger>`. This tree holds the
published notebooks only. Numerical examples on these pages are written so the run path works under `jax.jit`.

```{toctree}
:maxdepth: 2

03-derived-outputs
04-flow-types
06-stratification-introduction
08-strain-stratification
09-mixing-matrices
10-derived-outputs-stratified
11-flows-between-strata
12-concurrent-diseases
time-varying-functions
```
