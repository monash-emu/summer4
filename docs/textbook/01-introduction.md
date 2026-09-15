# 1. Infectious disease modelling

*Adapted from chapter 1 of the [summer textbook](https://github.com/monash-emu/summer-textbook)
(BSD-2-Clause, Copyright (c) 2022, monash-emu).*

Infectious disease modelling sits at the intersection of several disciplines. It
can be read as an extension of infectious disease epidemiology, as a branch of
complex systems analysis, or as applied mathematics — and modellers arrive from
all of those directions, as well as from clinical medicine, public health,
engineering and data science. Very few coursework programmes are dedicated to
the field, which is part of why a textbook built around working code is useful.

## What "modelling" means here

Almost any simplified representation of a pathogen or a population can be called
a model. This textbook is concerned with a narrower object: a **dynamic,
mechanistic, population-level** model, in which the population is divided into a
finite set of mutually exclusive compartments and the flows between those
compartments are specified as rates.

Two properties make this class of model worth the effort:

- **It is mechanistic.** The parameters are quantities an epidemiologist can
  reason about — a duration of infectiousness, a contact rate, a vaccine's
  effect on susceptibility — rather than regression coefficients.
- **It is dynamic.** Because infection is transmitted between members of the
  population, the risk to a susceptible person depends on how many infectious
  people there are, which depends in turn on past transmission. This feedback is
  what separates infectious disease epidemiology from most of the rest of
  epidemiology, and it is the reason a static calculation cannot answer the
  questions that matter.

## Why this feedback changes everything

In a non-communicable disease, one person's exposure does not change another
person's risk. In an infectious disease it does. The consequences run through
everything that follows:

- Interventions have **indirect effects**. Vaccinating one group reduces risk in
  another, and the indirect effect can exceed the direct one.
- Effects are **non-linear in time**. Halving transmission does not halve the
  epidemic; depending on where in the epidemic it happens, it may delay it,
  flatten it, or prevent it entirely.
- **Thresholds exist.** Below a critical reproduction number an epidemic cannot
  establish itself at all, so small parameter changes near that threshold have
  outsized consequences.

A model is the accounting system that keeps these interactions consistent.

## Why a framework rather than bare equations

Any of the models in this textbook could be written as a system of ordinary
differential equations and handed to a solver directly. That approach stops
scaling almost immediately. Once compartments are stratified — by age, by
vaccination status, by strain, by severity — the number of equations grows as
the product of the strata, and writing them out by hand becomes both laborious
and error-prone.

A modelling framework lets you declare the *structure* and derive the equations
from it. That is the role summer is intended to play: you describe the
compartments and the flows between them, and the framework assembles the system.

```{admonition} Where summer4 is today
:class: important

summer4 currently implements the **compartment structure** half of that
description — declaring stratification axes, applying them to some or all of the
model, and querying the resulting compartments. The flow, rate and solver half
does not exist yet.

{doc}`02-model-structures` shows how far the structural half alone gets you, and
{doc}`roadmap` lists what each remaining chapter needs.
```

## Why Python

The models in this textbook are built in Python, for reasons that are practical
rather than ideological: it is the language most of the surrounding data-science
ecosystem is written in, it is widely taught, and it lets the model, the data
handling and the analysis live in one notebook. summer4 in particular uses JAX
for its planned numerical layer, which brings automatic differentiation and
just-in-time compilation to models that would otherwise be slow in pure Python.

## Scope

The source textbook covers, in order: basic model construction; flows and flow
rates; latency and series compartments; immunity structures; numerical
solutions; derived outputs; transmission assumptions; the reproduction number;
cyclical dynamics; heterogeneous mixing and contact matrices; empirical contact
survey data; and calibration with uncertainty propagation.

That arc assumes a framework that can run a model. See {doc}`roadmap` for a
chapter-by-chapter statement of what is and is not reachable with summer4 as it
stands.
