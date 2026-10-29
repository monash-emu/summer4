# One-sided flow selectors fan out silently

## Today

A `TransitionFlow`'s edges pair source and destination compartments property by property:

- a property named on **neither** selector is kept (`S → V` moves each age to its own age);
- a property named on **both** is fixed (`S & adult → V & adult`);
- a property named on **only one** side is free on the other, so the flow **fans out** to
  every matching compartment, each edge at the full rate.

So `TransitionFlow("vaccination", s["S"] & age["adult"], s["V"], rate)` compiles to two
edges, adult → `V_child` and adult → `V_adult`. That doubles the outflow and moves adults
into the child stratum. `TransitionFlow("f", s["S"], s["V"] & age["adult"], ...)` moves
every age into adult. Population is still conserved, so no usual check notices. Inspect
with `compiled.edges(name)` (`src_idx`, `dest_idx`, `roles`).

Found 2026-10-30, when a Haiku-class trial agent made exactly this mistake building a
vaccination model from the summer4template docs (monash-emu/summer4template#1). The
template guide now documents the rule; summer4 itself gives no signal, and the
`TransitionFlow` docstring is one line.

## Shape

Decide between:

1. **warn** at `compile()` when a property is named on exactly one side and no `pairing=`
   or `split=` accounts for it, with an explicit opt-in to keep the fan-out (a keyword,
   or `split=`);
2. **raise** unless opted in;
3. **document only** (rule in the `TransitionFlow` docstring and user guide).

Some fan-out is intentional (for example all ages into one stratum), so first grep
tests, docs, examples and benchmarks for flows that rely on it. They must not break, and
must not spam warnings. Detection can use the edge roles compiled flows already record:
`EdgeRoles` `src_only` / `dest_only` / `free` in `src/summer4/flows/`.

Whatever the choice, the `TransitionFlow` docstring should state the rule.

## Done when

The chosen check or doc is implemented and tested, intentional fan-outs elsewhere are
unaffected or explicitly opted in, and the guide states the rule. Estimated at about two
hours of agent time, plus a design round-trip if many existing flows rely on fan-out.
