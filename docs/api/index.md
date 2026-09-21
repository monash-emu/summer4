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
   Groups
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
   Source
   Dest
```

`Selector` is a union type alias:

```python
type Selector = (
    Trait | IsIn | Present | Absent | Everything | Nothing | And | Or | Not
    | Source | Dest
)
```

`Source` and `Dest` wrap a compartment selector and evaluate on an
{class}`~summer4.flows.edges.EdgeMap` (from {meth}`CompiledModel.edges`). On a
compartment {class}`PropertyMap` they raise.

### Flows

```{eval-rst}
.. currentmodule:: summer4

.. autosummary::
   :toctree: generated
   :nosignatures:

   FlowModel
   CompiledModel
   TransitionFlow
   ExitFlow
   EntryFlow
   TraitChain
   TraitMatrix
   EdgeMap
   EdgeRoles
   euler
   actualize
   identity_join
```

### Time and results

```{eval-rst}
.. currentmodule:: summer4

.. autosummary::
   :toctree: generated
   :nosignatures:

   Epoch
   TimeAxis
   TimeGrouping
   RollingSpec
   SavePlan
   SaveRequest
   Compartments
   FlowMass
   ComputedValue
   SaveFn
   Result
   Output
   OutputSet
   SolverInfo
   Target
   TargetSet
   PropertyData
   State
```

`EVERYTHING` is the empty :class:`~summer4.results.plan.SavePlan` sentinel
(:meth:`~summer4.flows.compiled.CompiledModel.expand` fills it).
:class:`~summer4.results.targets.Target` / :class:`~summer4.results.targets.TargetSet`
merge observation times into a save plan for sparse calibration runs.
:attr:`~summer4.results.targets.Target.reduce` sums or averages a stratified
prediction down to an aggregate series. :class:`~summer4.results.outputset.OutputSet`
names a DAG of outputs; only its leaves enter the save plan, and
:meth:`~summer4.results.result.Result.to_frame` stacks the evaluated names.

### Rates and adjustments

```{eval-rst}
.. currentmodule:: summer4

.. autosummary::
   :toctree: generated
   :nosignatures:

   Const
   FieldRef
   FlowRef
   BinOp
   UnaryOp
   Multiply
   Overwrite
   Transform
   exp
   log
   tanh
   sqrt
   floor
   maximum
   minimum
   clip
   eval_closed
   derived_refs
   as_rate
   as_adjust
```

## What is not here

There is no mixing matrix, force-of-infection primitive, interpolation helpers,
initial-population wrapper, or Bayesian calibration likelihood. Sparse
`Target` / `TargetSet` gathering ships; probabilistic priors are WP10. See
{doc}`../evaluation/feature-completeness` for the full accounting and
{doc}`../user/07-from-summer2` for the summer2 symbols that have no equivalent.
