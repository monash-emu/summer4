# Deprecated lazy-idiom doors: removal inventory

Roadmap step 45 (`chore/remove-deprecated-doors`, `plans/unified-lazy-idiom.plan.md`)
removes these once a release has shipped them deprecated. Phase M (steps 36–44)
adds the replacements and the `DeprecationWarning`s; each step that deprecates a
door appends its row here.

| Door | Deprecated in | Replacement | Where |
| --- | --- | --- | --- |
| `FlowModel.compile(derived_fn=...)`, `derived_refs`, R3, `params_are_static=` | step 41 | `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)` | `src/summer4/flows/compiled.py`, `stages.py`, `rates.py` |
| `SaveFn(fn)` | step 40 | `outputs[name] = <step-stage expression>` | `src/summer4/results/plan.py` |
| `ComputedValue(path)` | step 40 | `outputs[name] = field_refs(...).path` | `src/summer4/results/plan.py` |
| InitialPopulation `callable(params)` base values and weights | step 42 | `defer(f)(Params())` | `src/summer4/flows/initial.py` |
| Bare-callable `ForceOfInfection(kind=fn)` | step 42 | `custom_kind(fn, *rate_args)` | `src/summer4/epi/infection.py` |

**Done when:** every row is gone from `src/`, the per-file
`filterwarnings` marks for summer4 deprecations are gone from `tests/`, the
old-idiom sections of `examples/notebooks/27-unified-lazy-idiom.ipynb` are
text-only "before" cells, the coverage-ledger mentions are migrated, and the
release notes list each removal with its replacement. Then delete this note
and its README bullet.

`Transform` is **not** on this list: it stays as the adjustment-level `defer`
(see the plan, step 42, for why replacing it would change precedence, overlap
checks and argument alignment).
