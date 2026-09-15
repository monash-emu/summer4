# `CompiledModel.describe` cannot size a plan for a model with real params

## What is wrong today

{meth}`CompiledModel.describe` (`src/summer4/flows/compiled.py:548`) has the
signature

```python
def describe(self, plan, *, t0=0.0, y0=None, n_saves=None, dt=1.0, steps=None)
```

There is no `params` argument. Internally it calls `jax.eval_shape` on a
closure that passes `None` where params would go, which reaches the model's
`derived_fn` as `derived_fn(None, y=..., t=...)`. Any hook that indexes its
params — which is any hook doing real work — raises:

```
ValueError: SavePlan failed shape inference (SaveFn must return statically
shaped arrays): 'NoneType' object is not subscriptable
```

## Why it hurts

`describe` exists to answer "how much memory will this plan cost before I
solve it", and the models where that question matters most are exactly the ones
it cannot answer. It works on a model with no `derived_fn`, or one whose hook
ignores its params — which is to say, on toy models.

`docs/user/10-targets-and-fitting.ipynb` demonstrates the sparse-versus-dense
memory saving with `describe`, and can only do so because its model has no
`derived_fn`. `docs/case-studies/age-stratified-seirs.ipynb` wanted the same
comparison, could not use `describe`, and counts rows out of the plans' own
`ts` arrays instead. That works but it is a worse answer: it counts saves, not
bytes, and it does not exercise the shape inference that `describe` is for.

## What a fix looks like

Add an optional `params` argument, threaded to the same place `run` threads it:

```python
def describe(self, plan, *, params=None, t0=0.0, y0=None, ...)
```

`jax.eval_shape` never needs concrete values, so callers can hand it a pytree of
`jax.ShapeDtypeStruct`s, or simply the real params they are about to run with.
Keeping the default as `None` preserves every current call site.

A regression test should describe a plan for a model whose `derived_fn` indexes
`params` and assert the reported `total_nbytes` matches the shapes a real `run`
produces.
