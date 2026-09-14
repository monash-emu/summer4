# API reference

The complete public API of summer4. Every name below is exported from
`summer4.__all__`; anything not listed here is internal and may change without
notice.

```{toctree}
:maxdepth: 2

summer4
```

## At a glance

### Taxonomy

```{eval-rst}
.. currentmodule:: summer4

.. autosummary::
   :toctree: generated
   :nosignatures:

   Property
   Trait
   PropertyMap
   Stratification
```

### Selectors

```{eval-rst}
.. currentmodule:: summer4

.. autosummary::
   :toctree: generated
   :nosignatures:

   IsIn
   Present
   Absent
   Everything
   Nothing
   And
   Or
   Not
```

`Selector` is a union type alias:

```python
type Selector = Trait | IsIn | Present | Absent | Everything | Nothing | And | Or | Not
```

## What is not here

There is no model, flow, rate, parameter, solver, derived-output or results API.
See {doc}`../evaluation/feature-completeness` for the full accounting and
{doc}`../user/07-from-summer2` for the summer2 symbols that have no equivalent.
