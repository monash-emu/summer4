# User guide

Task-oriented documentation for modellers building compartment spaces and
flows with summer4.

```{admonition} What this guide covers, and what it cannot
:class: important

Chapters 1–6 document the **compartment taxonomy**. Chapter 8 documents
**flows**: named transitions over a map, `CompiledModel`, edge queries with
`Source` / `Dest`, and a fixed-step Euler that returns the **final state
only**.

There is no results object, trajectory, or derived-output request API yet, so
there are no chapters on plotting outputs or fitting a model. Chapter 7 maps
the summer2 vocabulary onto what exists and names what does not;
{doc}`../evaluation/index` quantifies the gap.
```

```{toctree}
:maxdepth: 2

01-properties-and-traits
02-building-a-property-map
03-selectors-and-queries
04-ragged-stratification
05-partitions-and-groups
06-immutability-and-provenance
07-from-summer2
08-flows
```

## Reading order

If you have never used summer, read them in order — each chapter builds the map
the next one queries, and chapter 8 compiles flows over that map.

If you are coming from summer2 or summer3, start with {doc}`07-from-summer2`,
then read {doc}`04-ragged-stratification` and {doc}`08-flows`. Ragged
stratification with three-valued logic is the one genuinely new idea in the
taxonomy layer, and it changes what `~` means compared with every earlier
version of summer.
