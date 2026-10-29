# A domain-neutral "rate × head count" rate expression

Found while writing `docs/guide/09-predator-prey.ipynb` (step 48).

## What is wrong today

A rate proportional to the number of individuals in some other compartment is the core of any
mass-action model: predation (`a · predators`), births (`alpha · prey`), and bimolecular
reactions. summer4 has no neutral spelling for it.

- `Reduce(species, where=species["predator"])` is a grouped rate, aligned to the compartment the
  flow leaves. It can't be summed to a scalar:
  - `defer(jnp.sum)(Reduce(...))` raises a `TypeError`, because a `GroupedRate` is not an array.
  - `Reduce` has no `.sum_by()`.
- Used directly as the predation rate on prey, that `Reduce` **silently evaluates to 0**. The prey
  group's slot holds the prey-filtered sum, which is empty. There is no error.
- The only working route is `summer4.epi.ForceOfInfection(..., kind=FOIKind.DENSITY,
  contact_rate=rate)` with `infectious=` set to the counted compartment. Guide page 09 wraps it
  in a `count_rate` helper and explains it. The epidemic naming undercuts the page's point: that
  summer4 is not only for epidemics.

## What done looks like

- A core (non-`epi`) spelling for a scalar head count, exported from `summer4`. It could be
  `Reduce(...).sum_by()`, matching `FlowRef.sum_by()`, or a `Count(selector)` leaf. Either way it
  composes with `Param` and arithmetic.
- `Reduce` used as a rate on a compartment whose group slot is empty-by-filter either raises or is
  documented, and is tested.
- Guide page 09 drops `count_rate` and the `summer4.epi` import.
- If the fix lands as a new node, check it against Phase M (steps 36–39): it should be an ordinary
  rate node, with no special case.
