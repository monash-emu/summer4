# User guide

Task-oriented documentation for modellers building compartment spaces with
summer4.

```{admonition} What this guide covers, and what it cannot
:class: important

summer4's public API today is the **compartment taxonomy**: properties, traits,
selectors, property maps and stratifications. Chapters 1–6 below document that
surface completely, and every code cell in them is executed when this site is
built.

There is no model, flow, rate, solver or derived-output API yet, so there are no
chapters on running a model, fitting one, or plotting its outputs. Chapter 7
maps the summer2 vocabulary onto what exists and names what does not.
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
```

## Reading order

If you have never used summer, read them in order — each chapter builds the map
the next one queries.

If you are coming from summer2 or summer3, start with {doc}`07-from-summer2`,
then read {doc}`04-ragged-stratification`. Ragged stratification with
three-valued logic is the one genuinely new idea in this layer, and it changes
what `~` means compared with every earlier version of summer.
