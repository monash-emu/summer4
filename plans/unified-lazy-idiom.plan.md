<!--
Accepted 2026-10-14. Historical record: do not edit. Corrections go in the
roadmap's "Corrections to committed plans" table; position lives in
docs/dev/roadmap.md (steps 31-45).
-->

# CP5 + CP6: one lazy idiom for rates, outputs and parameter-dependent values

Lands as `plans/unified-lazy-idiom.plan.md`. The roadmap steps are:

- 31–35: Phase L (`CP5`), the bridges.
- 36–44: Phase M (`CP6`), one node set.
- 45: removal of the deprecated doors, blocked until a release is cut.

## Context

summer4's rate tree (`RateOps`, `src/summer4/flows/rates.py`) is better than summer2's
computegraph for rates. It has staging and hoisting, value-keyed jit digests, `GroupedRate`
alignment and `FlowRef` ordering (`docs/dev/rate-expressions.md`). The one thing computegraph did
better was **uniformity**: one `Function`/`Variable` vocabulary covered parameters, derived outputs
and model variables.

summer4 has nine doors for "compute something from a parameter":

- the rate tree;
- `Transform`;
- `derived_fn`, which disables hoisting model-wide (rule R3);
- `prepare_fn`;
- InitialPopulation callables;
- `SaveFn`;
- a **second expression language**, `OutputExpr` (`src/summer4/results/outputset.py`);
- likelihood scales, which accept only a bare `Param`;
- FOI `kind=` callables, which receive no parameters (CX11).

Some features exist in one place and not another:

- `np.exp(outputs.ref("x"))` raises.
- There is no `defer` over outputs.
- The tree's child walk is duplicated six times.
- `ForceOfInfection.__flow_refs__` returns `set()` (`src/summer4/epi/infection.py:314`), so a
  `FlowRef` under a FOI is never topologically ordered.

**Phase L** is additive and closes those gaps cheaply. **Phase M** makes `RateOps` the only
expression type:

- a post-solve stage;
- output leaves as nodes;
- `defer` and `np.*` everywhere;
- `derived_fn` replaced by a typed multi-output `defer`;
- any expression saveable;
- callable doors routed through rate-expression arguments;
- finally, the stage as a **type parameter**.

The old doors are deprecated (`DeprecationWarning` naming the replacement) and removed in step 45.

**User decisions (2026-10-14):**

- Phase L runs now, before step 28.
- Phase M runs after step 28 and before Track G (16–19).
- The old doors are deprecated, then removed.
- Phase M has full scope.
- Each step is run by an orchestrator session with tiered workers.

## Type expressibility (answering the user's question)

**Typed today:**

- `RateOps` and `OutputExpr` are distinct classes, so mypy separates "rate slot" from "output".
- `derived_refs(Schema)` is cast to the schema, so derived access is typed.

**Not typed today:** stage. `Time()` in an initial population, or a step-stage likelihood scale,
fails only at compile or run time. Grouping is not typed either.

**A naive unification would lose** the rate/output class distinction, and typed derived access if
`expr["key"]` replaced `derived_refs`.

**This plan keeps both and adds more:**

- **Step 39** ships `field_refs(expr, Schema)`, cast to the schema exactly as `derived_refs` is.
- **Step 43** makes `RateOps` generic in a covariant stage marker, with
  `Run <: Step <: Post` defined in `summer4.flows.stages`.
- Operators are typed `(RateOps[S1], RateOps[S2]) -> RateOps[S1 | S2]`. A union of markers is a
  subtype of the larger marker, so mypy computes the join.
- The slots then declare their stage:

  | Slot | Type |
  |---|---|
  | `rate=` / `Multiply` / `Overwrite` | `RateOps[Step] \| float` |
  | InitialPopulation, likelihood `sd=` / `dispersion=` | `RateOps[Run] \| float` |
  | `OutputSet[...] =` | `RateOps[Post]` |

Net: more is typed after unification than before.

**Limits:**

- `np.sin(expr)` is typed `Any`. That is a NumPy-stub limit, documented as a wart.
- The named helpers (`exp`, `log`, `tanh`, `sqrt`, `floor`, `maximum`, `minimum`, `clip`) return
  `Any` today. Step 43 types them.
- Grouping markers stay out of scope (`rate-expressions.md` item 4).

Step 43 opens with a go/no-go spike. The fallback is two top-level types, `Expr` (≤ step) and
`PostExpr`, which at least restores today's distinction.

## Execution model: tiered workers

Step 0 adds this as a section *Running a step with tiered workers* in `docs/dev/roadmap.md`,
under *How to run a step*.

The session told "next step" is the **orchestrator**. It may run on any model, and it does not
implement.

1. **Present the step's *Summary*** to the user before any tool use.
2. **Cut the branch** from *Cut from*, and record the base SHA.
3. **Spawn the implementer** with `Agent(model=<step's ### Worker tier>)` and this prompt:

   > You are implementing roadmap step N of summer4. Read `AGENTS.md`, then `docs/dev/roadmap.md`
   > § Step N (*What the previous worker left you*, *Read first*, *Do*, *Files*), then
   > `plans/unified-lazy-idiom.plan.md` § Step N. Work only on branch `<branch>`; commit as you go.
   > Run only the targeted pytest files the step names (`pixi run pytest tests/<file>.py -q`) plus
   > `pixi run lint` and `pixi run format-check`. Do not edit the roadmap step table, *Current
   > position* or any *Handoff*. Do not push, open a PR, or delete, skip or loosen an existing test
   > unless the step names that test. Finish by reporting:
   > - the files changed;
   > - the commands you ran and their results;
   > - every deviation from the plan, with the reason.

4. **Run the checks.**
   - Light checks, locally: `lint`, `format-check`, `check-notebooks`, `check-branch`, `coverage`,
     `roadmap`, `composability`.
   - Heavy checks (`pixi run test`, `test-all`, `-e docs docs-strict`): these belong on the
     shared Mac mini (`~/.devq/CLAUDE-devq.md`) once a summer4 checkout exists there. When it
     does, push the branch and run them with
     `mini build -C <mini checkout> -- bash -lc 'git fetch && git checkout <branch> && git pull --ff-only && pixi run test && pixi run test-all && pixi run -e docs docs-strict'`.
     On 2026-10-14 there was no checkout on the mini, so they run locally until one exists.
   - A red check counts as a FAIL, with the log as the finding.
5. **Spawn the checker** with a fresh `Agent(model=<checker tier>)`. The checker is one tier above
   the implementer, minimum sonnet: haiku→sonnet, sonnet→opus, opus→opus (fresh context). Pass it:
   - the step's roadmap and plan sections;
   - `git diff <base>...HEAD`;
   - the step's `### Sanity check` list;
   - this instruction:

   > Verify each item by reading code or running the named targeted tests. Also verify:
   > - (a) the changed files are within the step's *Files* line plus `plans/`, `tests/`, the
   >   phase notebook and the docs it names;
   > - (b) no existing test was deleted, skipped or loosened except those the step names
   >   (`git diff <base>...HEAD -- tests/`);
   > - (c) new tests assert the step's claims, not just execution;
   > - (d) no new `# type: ignore`, `pytest.skip`, `xfail`, bare `except`, or digest keyed on
   >   `id(` outside `_callable_key`;
   > - (e) the notebook section follows the acceptance style below.
   >
   > Reply `VERDICT: PASS` or `VERDICT: FAIL`, then numbered findings, each citing a checklist item
   > and a file:line.

6. **On FAIL, escalate.**
   - Push the attempt to `wip/<slug>-<tier>`.
   - Run `git reset --hard <base>` on the step branch.
   - Respawn the implementer one tier up, with the findings and the wip branch ("reuse what is
     correct").
   - After an opus failure, stop and ask the user.
7. **On PASS, hand off.**
   - Open the PR first, with the user-gate checklist; the `**Landed:**` line needs the PR number.
   - Then make the handoff commit: the standard handoff, plus the line
     `**Worker:** <tier>; checker <tier>; attempts <n> (<tiers tried>).` under `**Landed:**`.
   - **Stop for the user gate.** Do not merge.

**Acceptance notebook style.** Follow the signed-off style of notebooks 22 and 24 (memory
acceptance-vs-example-notebooks):

- a *What to check* box per section;
- a titled Plotly plot or a displayed object;
- no asserts; asserts live in `tests/`.

Step 0 updates AGENTS.md § Feature acceptance bar and `roadmap.md:47-49`, which still demand
asserts in the notebook.

**Gate model.** It must stress the feature: a two-age-group SEIR with
`beta = Param("beta") * (1 + Param("amp") * np.sin(2*np.pi*Time()/365))`. Later steps add:

- notifications: incidence convolved with a gamma delay whose mean is `Param("delay")` (step 33);
- a NegativeBinomial target with `dispersion=np.exp(Param("log_k"))` (step 31).

**`check-branch`.** Every step touching `src/` also changes:

- `tests/`;
- the phase notebook;
- a step plan, `plans/<branch-slug>.plan.md`: a short file saying "Implements Step N of
  `unified-lazy-idiom.plan.md`", plus deviations.

The master plan is never edited after Step 0. Corrections go in the roadmap's Corrections table.

**Import rule (all steps).** `summer4.results` imports `summer4.flows` at module level. Anything
in `summer4.flows` that touches results types imports them **inside the function**. Stage markers
live in `summer4.flows.stages`.

**Test rule (all steps).** New tests assert evaluated **values**, not node classes or `kind`
strings, so they survive step 38's switch of representation.

## Step 0: land this plan (done on `docs/unified-lazy-idiom-plan`)

Branch `docs/unified-lazy-idiom-plan`, from `origin/main`.

1. **Plan file.** Add `plans/unified-lazy-idiom.plan.md`: this document, with each step below as a
   `## Step N — \`branch\`` section.

2. **Findings.** Add these rows to the `docs/evaluation/composability.md` findings block, as full
   8-cell rows (`| ID | Area | Principle | Severity | Status | Finding | Proposed shape | Evidence |`).
   All are `open`. No cell may contain ` | `; the evidence anchor must appear literally in the file.

   | ID | Area | P | Sev | Finding (short) | Evidence |
   |---|---|---|---|---|---|
   | CX26 | calibration | P1 | low | `resolve_scale` passes a rate expression through unevaluated | `` `src/summer4/epi/calibration/likelihoods.py::def resolve_scale` `` |
   | CX27 | results | P1 | medium | `OutputExpr` has no ufunc hook; the named helpers raise on it | `` `src/summer4/results/outputset.py::class OutputExpr` `` |
   | CX28 | results | P1 | medium | no `defer` over post-solve outputs | `` `src/summer4/results/outputset.py::_CALLS = frozenset(` `` |
   | CX29 | rates | P2 | medium | six duplicated child walks; FOI hides FlowRefs; capture meta skips TableInterp/Lookup | `` `src/summer4/epi/infection.py::def __flow_refs__` `` |
   | CX30 | results | P1, P4 | high | two expression languages; operators hand-written five times | `` `src/summer4/results/outputset.py::type _Kind = Literal` `` |
   | CX31 | rates | P1 | high | `derived_fn` disables hoisting model-wide; no shared-node evaluation | `` `src/summer4/flows/compiled.py::params_are_static=derived_fn is None` `` |
   | CX32 | results | P1 | medium | a step-stage expression needs `SaveFn`/`ComputedValue` to be saved | `` `src/summer4/results/plan.py::class SaveFn` `` |
   | CX33 | rates | P1 | medium | InitialPopulation callables bypass the tree | `` `src/summer4/flows/initial.py::def _base_digest` `` |
   | CX34 | rates | P2 | medium | stage is runtime-only; mypy cannot reject stage errors | `` `src/summer4/flows/stages.py::Stage = Literal` `` |

   Also rewrite CX7's Proposed shape to say "a `named(fn, name)` wrapper accepted by every
   callable hook, with the `defer(name=)` contract".

3. **Packages.**
   - Add `| CP5 | Bridges between the lazy idioms | CX7 CX26 CX27 CX28 CX29 |`.
   - Add `| CP6 | One lazy node set | CX11 CX30 CX31 CX32 CX33 CX34 |`.
   - Remove CX7 and CX11 from CP3.
   - Run `pixi run composability-write`.

4. **Roadmap rows and sections** (`docs/dev/roadmap.md`).
   - Add the tiering section.
   - Add rows 31–45, each with 8 cells: Phase `L` for 31–35, `M` for 36–45; WP `CP5` / `CP6`, and
     `—` for 45; Plan `plans/unified-lazy-idiom.plan.md`; Section `Step N`; Status `planned`
     (`blocked` for 45); Closes as below, else `—`.
   - Each `## Step N — \`branch\`` section has the five required headings, plus `### Worker`,
     `### Files` and `### Sanity check`, and the line ``Cut from: `main`. Merges into: `main`.``
     inside `### Do`.

5. **Landing order and current position.**
   - The landing-order fence becomes
     `14 → 15 → 24 → 25 → 26 → 27 → 29 → 30 → 31 → 32 → 33 → 34 → 35 → 28 → 36 → 37 → 38 → 39 → 40 → 41 → 42 → 43 → 44 → 16 → 17 → 18 → 19 → 45`.
   - *Current position*: Step 31, `next`, `fix/likelihood-scale-expr`, cut from `main`, last
     landed `fix/saveplan-ts-grid`.
   - Step 28 becomes `planned`.

6. **Handoffs.** Every step's Handoff contains its successor as literal `` step <b> `next` ``.
   - Steps 31–34 and 36–43 name the next number.
   - Step 30: append "(Re-sequenced 2026-10-14: step 31 `next`; step 28 follows step 35.)".
   - Step 35: `step 28 `next``.
   - Step 28: replace `step 16` with `step 36 `next``, and note that Track G resumes after
     step 44.
   - Step 44: `step 16 `next``.
   - Step 19: append "then, once the release after step 44 is cut, step 45 `next`". Its existing
     "step 20 `next`" stays.

7. **Previous-worker notes.**
   - Step 28: Phase L landed first, and its spellings (`defer` over outputs, `np.*` on refs,
     expression likelihood scales) are available for notebook 25.
   - Step 16: `plans/wp17-foi-susceptibility.plan.md:54,111` says to extend `__field_paths__`.
     After step 35, extend `ForceOfInfection.__rate_children__` instead.

8. **futureplans** notes, each also listed in `futureplans/README.md`:
   - `deprecated-lazy-doors-removal.md`: the step-45 inventory.
   - `rates-maximum-ndarray-inconsistency.md`: `rates.maximum(rate, ndarray)` raises while
     `np.maximum` builds a node.

9. **AGENTS.md and roadmap prose.** Update both notebook-assert passages to the acceptance style
   above.

10. **Mini checkout path.** Ask the user for it and record it in the tiering section. (Answer:
    no checkout on the mini yet; the heavy checks run locally until one exists.)

11. **Checks and PR.** Run `pixi run roadmap`, `pixi run composability`, `pixi run coverage` and
    `pixi run pytest tests/test_roadmap.py tests/test_composability_survey.py -q`, then open the
    PR.

---

## Phase L — CP5: bridges (additive, no deprecations)

Gate notebook: `examples/notebooks/26-lazy-idiom-bridges.ipynb`. Step 31 creates it, and each later
step adds one section. Number 25 stays reserved for step 28.

## Step 31 — `fix/likelihood-scale-expr` · Worker haiku · Closes CX26

**Summary.** Likelihood `sd=` and `dispersion=` accept any parameter-only expression, such as
`np.exp(Param("log_k"))`, not only a bare `Param`.

**Files.** `src/summer4/epi/calibration/likelihoods.py`, `tests/test_epi_calibration.py`.

**Do.**
- In `resolve_scale` (`likelihoods.py:46`), keep the `Prior` branch and the number branch.
- Replace the `FieldRef` branch with: `if isinstance(spec, RateOps): return eval_closed(spec, params)`.
  `eval_closed` is at `src/summer4/flows/compiled.py:418`; import it inside the function. It
  raises `ValueError` for step-stage input; let that propagate.
- Update the docstrings of `Normal.sd` and `NegativeBinomial.dispersion`.
- Add tests next to `test_epi_calibration.py:230`:
  - `resolve_scale(Param("a")*2, {"a": 1.5}) == 3.0`;
  - `np.exp(Param("log_k"))` resolves under `jax.jit` and has a finite `jax.grad`;
  - `Time()` raises `ValueError`;
  - `Normal(sd=Param("s")*2).log_prob(...)` equals the same call with `sd=2*s`.
- Notebook: create 26 with the gate model, incidence only. Section A fits `log_k` and plots the
  log-likelihood over a grid of `log_k` values.

**Sanity check.**
- No branch of `resolve_scale` returns a `RateOps`.
- The existing `Prior`/`FieldRef` tests are untouched and green.
- CX26 is `done`.

## Step 32 — `feat/output-ufuncs` · Worker haiku · Closes CX27

**Summary.** `np.exp`, `np.sin`, `np.maximum` and `np.clip` work on `outputs.ref(...)`, and so do
the named helpers. Both use the shared dispatch.

**Files.** `src/summer4/flows/algebra.py`, `src/summer4/flows/rates.py`,
`src/summer4/results/outputset.py`, and the new `tests/test_output_expr_dispatch.py`.

**Do.**
- Record in the step plan the current behaviour of `Param("s") * outputs.ref("x")`. It raises
  `TypeError` from `as_rate` (`rates.py:633`).
- `algebra.py`:
  - Add `"output_expr"` to `DispatchMode` (`:95`).
  - In `_apply_mode` (`:163`), build `OutputExpr(kind="unary", op=canonical, kids=(_coerce(x),))`,
    or `outputset._binary(canonical, a, b)`.
  - `np.clip` already routes to `rates.clip` in every mode (`:231-239`), so `rates.clip` must
    handle it.
- `OutputExpr` (`outputset.py:61`):
  - Add `__array_priority__ = 1000`.
  - Add `__array_ufunc__` calling `dispatch_ufunc(..., mode="output_expr")`.
  - Add `__array_function__` calling `dispatch_array_function(..., mode="output_expr")`.
- `rates.py` (`_unary`, `_binary`, `maximum`, `minimum`, `clip`, at `:683-759`): when any operand
  is an `OutputExpr`, build `OutputExpr` nodes. Detect it by class name check or a lazily cached
  import, not an import per call.
- `RateOps` operator dunders (`:85-119`) and `RateOps.__array_ufunc__` (`:121`) return
  `NotImplemented` when the other operand is an `OutputExpr`.
- Save specs (`Compartments`, `FlowMass`, …) get no `__array_ufunc__`. `summer4.tanh(FlowMass(...))`
  works through the `rates` helpers; `np.tanh(FlowMass(...))` is documented as unsupported until
  step 38.
- Tests, asserting values only:
  - each of `np.exp`, `np.sin`, `np.maximum(ref, 0.0)`, `np.clip(ref, 0, 1)` and `tanh(ref)`,
    once evaluated, equals the same op on the `Output`;
  - `ndarray * ref` and `Param("s") * ref` work;
  - `np.greater(ref, 0)` raises;
  - jit and grad work, using the `tests/test_output_set.py:105` pattern.
- Notebook section B: `np.log(outputs.ref("incidence"))` overlaid on `np.log` of the solved
  `Output`.

**Sanity check.**
- `DENY_OPS` still refuses comparisons.
- No new alias map.
- The ndarray-left and RateOps-left cases are tested.
- CX27 is `done`.

## Step 33 — `feat/output-defer` · Worker haiku · Closes CX28

**Summary.** `defer(f)(outputs.ref("incidence"), Param("delay"))` builds a post-solve node: the
port of summer2's `Function(f, [DerivedOutput(...), Parameter(...)])`.

**Files.** `src/summer4/flows/rates.py` (`defer`), `src/summer4/results/outputset.py`, and the new
`tests/test_output_defer.py`.

**Do.**
- In `defer()` (`rates.py:382`), `wrapped` checks every positional and keyword argument. If any is
  an `OutputExpr`, or an instance of `outputset._LEAF_TYPES` (lazy import), it returns
  `OutputExpr(kind="defer", value=fn, name=name, kids=(coerced positional..., coerced kwarg values...), kwargs=tuple((k, None) for k in kwarg names))`.
  - Kwarg values go in `kids`, so `_ref_list`/`_leaves` (`outputset.py:193-213`) see them.
  - `kwargs` stores the names only.
  - Otherwise, build a `Defer` exactly as today.
  - Widen the return annotation to `Defer | OutputExpr`.
- Add `"defer"` to `_Kind`.
- The `eval_node` branch for `defer`:
  - evaluate the kids; output kids give an `Output`, const and param kids give arrays;
  - split them back into positional and keyword using the stored names;
  - call `fn`.
- The return contract, also stated in the docstring:
  - an `Output` is returned as is;
  - an array whose shape equals the first `Output` argument's data shape is rewrapped as
    `first._with(values=PropertyData(first.values.pmap, arr) if isinstance(first.values, PropertyData) else arr)`.
    The data shape is `first.values.data.shape` when it is `PropertyData`, else
    `first.values.shape`;
  - anything else raises `TypeError` naming the function and the shapes.
- Document that `Defer(f, outputs.ref(...))` called directly still raises; use `defer(f)(...)`.
- Tests, asserting values only:
  - a gamma-kernel convolution parameterised by `Param("delay")` matches a manual NumPy
    convolution;
  - an output passed as a kwarg is saved by `plan()`;
  - a forward ref resolves;
  - `grad` with respect to `delay` is finite and non-zero;
  - a bad return shape raises;
  - an all-parameter `defer` still evaluates in a rate slot.
- Notebook section C: notifications, as delayed incidence, for three `delay` values against
  incidence. The gate model gains notifications here.

**Sanity check.**
- `tests/test_rate_defer.py` is untouched and green.
- Both return branches are tested.
- The kwarg-output case is tested.
- CX28 is `done`.

## Step 34 — `feat/named-callables` · Worker haiku · Closes CX7

**Summary.** `named(fn, "seasonal")` gives any callable hook a stable digest key, so rebuilding a
model with a fresh closure hits the jit cache. A wrong name gives a wrong answer, the same contract
as `defer(name=)`.

**Files.**
- `src/summer4/flows/rates.py`, `compiled.py`, `initial.py`, `__init__.py`;
- `src/summer4/__init__.py`;
- `src/summer4/results/plan.py`;
- `src/summer4/epi/infection.py`;
- `docs/dev/rate-expressions.md`;
- the new `tests/test_named_callables.py`.

**Do.**
- In `rates.py`, add:
  - `@dataclass(frozen=True) class Named`, with fields `fn` and `name: str`, whose `__call__`
    forwards `*args, **kwargs`;
  - `def named(fn, name) -> Named`;
  - `def _callable_key(fn) -> str`, returning `"n:" + fn.name` for a `Named` and `str(id(fn))`
    otherwise.
- Use `_callable_key` at:
  - `rates.py:1056` (`_adjust_bytes`, Transform);
  - `results/plan.py:167` (SaveFn);
  - `compiled.py:780` and `:783` (derived_fn, prepare_fn);
  - `initial.py:385` and `:407`;
  - `infection.py:340` (FOI kind).
- Leave `Defer.__rate_bytes__` as it is: `name` if given, else `str(id(fn))`. Add one branch:
  when `fn` is a `Named` and `name` is None, use `_callable_key(fn)`.
- `derived_return_schema` (`rates.py:901-924`) unwraps `Named` before reading annotations.
- Export `named` from `summer4.flows` and `summer4`.
- Tests:
  - for each of Transform, SaveFn, derived_fn, prepare_fn, InitialPopulation base and FOI kind,
    two rebuilds with fresh closures under one name give equal model digests;
  - different names give different digests;
  - bare closures give different digests;
  - a trace-count test in the style of the `rate-expressions.md` § identity-keying table;
  - `derived_fn=named(f, "d")` still validates its `ComputedValue` schema.
- Docs: add a `named` paragraph in § identity keying.
- Notebook section D: a bar plot of retraces against rebuilds, with and without `named`.

**Sanity check.**
- Every digest that keys on a callable goes through `_callable_key`.
- The `id(` uses for hoist slots and the jit cache are allowed.
- `tests/test_rate_dispatch.py` goldens are green.
- CX7 is `done`.

## Step 35 — `fix/rate-children-fold` · Worker sonnet · Closes CX29

**Summary.** One `rate_children(expr)` replaces the duplicated child walks, so a custom node needs
only `__rate_children__` and `__rate_bytes__`. It fixes FlowRefs being lost under
`ForceOfInfection`, and capture metadata being lost under `TableInterp` and `Lookup`.

**Files.**
- `src/summer4/flows/rates.py`, `stages.py`, `compiled.py`, `__init__.py`;
- `src/summer4/epi/infection.py`;
- `docs/dev/rate-expressions.md`;
- `docs/cookbook/01-custom-rates.ipynb`;
- `examples/notebooks/14-custom-rate-nodes.ipynb`;
- the new `tests/test_rate_children.py`.

**Do.**
1. **Characterisation test first, committed green on the old code, in its own commit.**
   `tests/test_rate_children.py` builds a corpus of about 15 expressions: one per built-in node,
   plus nested Defer, a FOI, Capture in Defer, and Capture in TableInterp. For each it records:
   - `_flow_refs`;
   - `_field_paths`;
   - `_collect_capture_meta`;
   - `rate_stage` for both flag values;
   - `build_hoist_table` as `[(type(node).__name__, part)]`.
2. **Add `rate_children(expr) -> tuple[RateOps, ...]`** in `rates.py`:

   | Node | Children |
   |---|---|
   | BinOp | left, right |
   | UnaryOp | arg |
   | Interp | *breakpoints, *values, arg |
   | GaussianPulse | arg, centre, width, height |
   | TableInterp | arg |
   | Lookup | table, index |
   | Capture | inner |
   | Defer | args + kwarg values |
   | leaves | () |

   Anything else returns `__rate_children__()` if it is defined, else `()`. Export
   `rate_children`, `rate_stage` and `build_hoist_table` from `summer4.flows`. This partly
   addresses CX6; leave CX6 open.
3. **Refold `_flow_refs` and `_field_paths`.** Keep the leaf arms (FlowRef, FieldRef), then fold
   over `rate_children`. A custom `__flow_refs__` or `__field_paths__` is still honoured first.
4. **Refold `_collect_capture_meta`.** Call `__capture_meta__` on each node, then fold over
   `rate_children`. `__capture_children__` is still honoured first.
5. **`rate_stage`: generic join only for the listed built-ins** (BinOp, UnaryOp, Interp,
   GaussianPulse, TableInterp, Lookup), plus Defer through its existing hook. Unknown nodes keep
   today's rules: their `__rate_stage__`, else `"step"`.
6. **`build_hoist_table`.** Keep the Interp / TableInterp / Lookup special cases and "do not
   descend into Capture". Use `rate_children` for the generic descent.
7. **Do not touch `_rate_bytes` or `_eval_rate`.**
8. **`ForceOfInfection` (`infection.py:314`).**
   - Add `__rate_stage__` returning `"step"`.
   - Add `__rate_children__` returning `contact_rate`, `exponent` (if set), every infectiousness
     value, and `mixing.matrix` (`epi/mixing.py:31`).
   - Delete its `__flow_refs__` and `__field_paths__`.
9. **Update the characterisation expectations, each with a `# fixed in step 35` comment, for these
   deliberate changes only:**
   - FOI flow refs;
   - capture meta under TableInterp and Lookup;
   - hoist slots inside a FOI's parameter-only children.
10. **Regression tests.**
    - A FlowRef inside a FOI `contact_rate` is ordered before the FOI flow, and a cycle raises.
    - The hoist-equivalence test (`test_hoisting.py:138` pattern) holds for the FOI model.
11. **Docs.**
    - `rate-expressions.md`: item 3 is done, using the existing name `__rate_children__` rather
      than `__children__`; the extension surface is down to two dunders.
    - The cookbook's `ForceOfPredation` and notebook 14's `Shedding` gain `__rate_children__`.
12. **Notebook section E.**
    - The gate model with a FOI whose contact rate reads a FlowRef (for example, scaled by
      vaccination flow), now ordered correctly.
    - A plot of FOI and incidence against a hand-ordered reference model.
    - A displayed hoist table showing the FOI's parameter children slotted.

**Sanity check.**
- The characterisation commit precedes the refactor (`git log`).
- Its later diff touches only `# fixed in step 35` lines.
- The goldens are green.
- `test_hoisting.py` and `test_rate_defer.py` are untouched.
- CX29 is `done`.

**Handoff.** Phase L is finished: step 28 `next`.

---

## Phase M — CP6: one node set (after step 28)

Gate notebook: `examples/notebooks/27-unified-lazy-idiom.ipynb`. Step 36 creates it with the gate
model written in the **old** idioms: `derived_fn` for seasonal beta, an `OutputSet`, `SaveFn` for
seasonal beta, and a callable initial population. Each later step converts one door, side by side,
with a plot of the comparison.

## Step 36 — `feat/stage-lattice` · Worker haiku

**Summary.** Stages become an ordered lattice, `run < step < post`, with one `join_stages`
helper. A `"post"` stage is rejected in rates, initial populations and `eval_closed`. No post
nodes exist yet.

**Files.** `src/summer4/flows/stages.py`, `rates.py`, `compiled.py`, `docs/dev/run-stages.md`,
the new `tests/test_stage_lattice.py`.

**Do.**
- In `stages.py:34`:
  - `Stage = Literal["run", "step", "post"]`;
  - `_STAGE_ORDER = {"run": 0, "step": 1, "post": 2}`;
  - `def join_stages(*stages: Stage) -> Stage`, which is the max and returns `"run"` for no
    arguments.
- Replace the four binary joins in `rate_stage` (`stages.py:105`, `:114`, `:119`, `:128`) and
  Defer's join (`rates.py:337`) with `join_stages`.
- Keep the default `"step"` (`:135`), the hoist conditions `== "run"` (`:183`, `:188`, `:193`),
  and the `!= "run"` checks (`compiled.py:432`, `initial.py:293/338/367`).
- Widen the validation at `stages.py:132-133` to accept `"post"`.
- In `FlowModel.compile`, raise `ValueError("flow <name> rate depends on a post-solve output; ...")`
  when any flow rate or adjustment root (use `roots_of`, `stages.py:234`) has stage `"post"`.
- Tests: a test-only custom node whose `__rate_stage__` returns `"post"` (with `__rate_bytes__`):
  - it joins upward through BinOp, Defer and Interp;
  - it is rejected by `compile`, `eval_closed` and InitialPopulation, each with a message test;
  - it is never slotted by `build_hoist_table`.
- Docs: the `run-stages.md` stage table.
- Notebook section A: the old-idiom gate model, solved, and the three stages explained, with the
  model's stage classification table displayed.

**Sanity check.**
- No `"step" if ... else "run"`-style binary join remains in `stages.py` or `rates.py`.
- The hoist and `eval_closed` conditions are unchanged.
- The `hoist=False` equivalence tests are green.

## Step 37 — `feat/post-stage-nodes` · Worker sonnet

**Summary.** Post-solve quantities become `RateOps` nodes: `Saved(quantity)`, `OutputRef(name)`
and `PostOp(method, arg, args, kwargs)`. A new `eval_post` evaluator ships alongside them.
`OutputSet` is not switched yet.

**Files.** The new `src/summer4/results/expr.py`, `src/summer4/results/outputset.py` (to import
`_CALLS` only), the new `tests/test_post_nodes.py`.

**Do.**
- In `results/expr.py`, add three frozen dataclasses subclassing `RateOps`. Each has:
  - `__rate_stage__` returning `"post"`;
  - `__rate_children__` (`()` for `Saved` and `OutputRef`, `(arg,)` for `PostOp`);
  - a total `__rate_bytes__`:
    - `Saved` uses `_quantity_bytes` from `plan.py`;
    - `OutputRef` uses `b"oref" + name`;
    - `PostOp` uses `b"post" + method + _rate_bytes(arg)`, plus each arg encoded by type:
      int, float, str, bool and None via `repr`; StrEnum via value; `Property` via name;
      `Selector` via `_selector_bytes`; ndarray via `dtype`, `shape` and `tobytes()`. Any other
      type raises `TypeError`.
- Move `_CALLS` into `expr.py`, and have `outputset` import it from there.
- `PostOp` is legal only if `method in _CALLS`.
- `eval_post(expr, *, result, params, key_for, computed)`:
  - `Saved` → `result[key_for(q)]`;
  - `OutputRef` → `computed[name]`;
  - `PostOp` → `getattr(child, method)(*args, **kwargs)`;
  - `BinOp` / `UnaryOp` with a post child → `Output._combine` / `_map_unary`;
  - `Defer` with a post child → step 33's return contract (move that helper to `expr.py`);
  - any subtree with `rate_stage == "run"` → `eval_closed(subtree, params)`;
  - else `TypeError` naming the node.
  - Walk with `rate_children`.
- Do not register the nodes with `register_rate_eval`. Step 36 already rejects them in rates.
- Tests: a parity table. Every expression in `tests/test_output_set.py`,
  `test_output_defer.py` and `test_output_expr_dispatch.py` is rebuilt with the new nodes. Each
  must satisfy `eval_post == OutputSet.evaluate` on the same solved `Result`, under `jit`.

**Sanity check.**
- The parity table covers every `OutputExpr` kind: leaf, ref, const, param, unary, binary, call,
  defer.
- `outputset.py` behaviour is unchanged.
- `expr.py` has no module-level import cycle: `python -c "import summer4.results.expr"` runs
  first in a fresh interpreter.

## Step 38 — `feat/outputset-on-rate-nodes` · Worker sonnet · Closes CX30

**Summary.** `OutputSet` stores and evaluates `RateOps`, and `OutputExpr` is deleted. That
leaves one expression language.

**Files.**
- `src/summer4/flows/rates.py` and `algebra.py`;
- `src/summer4/results/outputset.py`, `plan.py`, `expr.py`;
- `src/summer4/epi/calibration/model.py` and `posterior_runs.py`;
- notebooks 06, 18, 24, 25, 26;
- `docs/textbook/20-calibration.ipynb`;
- tests.

**Do.**
- **Post-op methods on the `RateOps` base:** `select`, `sum_over(prop, side=None)`, `total`,
  `rolling`, `cumulative`, `midpoint`, `integrate_intervals`, `integrate`, `at_times`.
  - Each returns `RateOps`: a `PostOp`.
  - Each raises `TypeError` at construction when `rate_stage(self)` is not `"post"`, so
    `Time().rolling(7)` fails.
  - `FlowRef.sum_over` (`rates.py:223`) keeps its behaviour, with a compatible signature
    `(self, prop, side=None) -> RateOps`. `side` must be None for a FlowRef.
- **Attribute paths.** These nine names are now methods, so `Param("x").total` is a bound method,
  not a path.
  - Add `FieldRef.__getitem__(name: str)` (str only; any other type raises) as the path escape:
    `Param("x")["total"]`.
  - Add `RateOps.__iter__`, which raises `TypeError`, so the `__getitem__` protocol cannot loop.
  - Re-run the collision grep over `src`, `tests`, `examples` and `docs`
    (`\.(select|sum_over|total|rolling|cumulative|midpoint|integrate_intervals|integrate|at_times)\b`
    on a FieldRef or `derived_refs` value), and record the result in the step plan.
- **Save specs.** Rename `_OutputExprChain` (`plan.py:23`) to `_PostOpChain`; its methods return
  `PostOp(Saved(self), ...)`. Save specs also get `__array_ufunc__` / `__array_function__` in
  `mode="symbolic"` via `Saved(self)`, so `np.tanh(FlowMass(...))` works.
- **`OutputSet`.**
  - `ref` returns `OutputRef`.
  - `__setitem__` accepts a `RateOps` or a save spec (wrapped as `Saved`).
  - `_leaves` and `_ref_list` walk with `rate_children`.
  - `evaluate` calls `eval_post`.
- **`as_rate`** accepts save specs (lazy import) and wraps them as `Saved`.
- **Remove the bridges.** `defer` drops step 33's special case, since a `Defer` with post
  arguments is now just a `Defer`. `algebra` drops `"output_expr"`. `RateOps` dunders drop the
  `OutputExpr` checks.
- **Delete** `OutputExpr` and `_coerce`.
- **Tests.**
  - All of `test_output_set.py`, `test_output_defer.py`, `test_output_expr_dispatch.py` and the
    calibration tests keep **their value assertions**.
  - The node-representation assertions in `test_output_set.py:31, 135-137` (`isinstance(...,
    OutputExpr)`, `.kind`, `.method`) may be rewritten to check `PostOp` / `.method`.
  - Add tests: `Param("x")["total"]` is a FieldRef; `Time().rolling(7)` raises.
- **Notebook 27 section B:** the same outputs in the old and new spelling, overlaid.

**Sanity check.**
- `grep -rnw OutputExpr src tests examples docs --exclude-dir=_build` matches only `CHANGELOG.md`
  and `docs/releases/`.
- In `tests/`, only `test_output_set.py:31, 135-137` assertions changed.
- `mypy --strict src/summer4` is green.
- CX30 is `done`.

## Step 39 — `feat/shared-nodes-and-fields` · Worker sonnet

**Summary.** A sub-expression object used by several flows is evaluated once per vector-field
call. `field_refs(expr, Schema)` reads fields of a structured `defer` result with typed access.
Together these let a new-idiom model replace `derived_fn` and keep parameter hoisting.

**Files.** `src/summer4/flows/rates.py`, `compiled.py`, `stages.py`, `__init__.py`;
`src/summer4/__init__.py`; the new `tests/test_shared_nodes.py`.

**Do.**
1. **Per-call memo.** In `_eval_rate` (`compiled.py:282`), add a per-call `memo: dict[int, Any]`
   keyed by `id(node)`, for non-leaf nodes.
   - Create it fresh in `observe` (`compiled.py:945`) and thread it through `eval_child`.
   - The hoist-slot lookup (`:300-303`) still runs first.
   - `eval_closed` uses its own fresh memo.
2. **`Item(arg, key: str | int)`.** Built by `RateOps.__getitem__`; `FieldRef` keeps its own str
   path `__getitem__` from step 38.
   - Evaluation: `getattr` for a NamedTuple with a str key, otherwise `[key]`.
   - Stage: `arg`'s stage (add it to the generic-join list).
   - `rate_children`: `(arg,)`.
   - `__rate_bytes__`: `b"item" + child + key`.
   - Registered via `register_rate_eval`.
3. **`field_refs(expr, Schema) -> Schema`.** A `cast`, modelled on `_field_ref_tree` /
   `derived_refs` (`rates.py:600-620`). It builds nested `Item` chains for nested NamedTuple
   fields.
4. **Leaves for `derived_fn`'s inputs.**
   - `StateArray()`: step stage. It evaluates to the raw `y_arr`, exactly what `derived_fn`
     receives (`compiled.py:968-969`). The name avoids the existing `summer4.State`.
   - `Params()`: run stage. It evaluates to the whole params object, and contributes **no**
     `_field_paths`, so `expand()` (`compiled.py:1090`) is unaffected.
   - Add both to `_is_leaf` (`stages.py:159`).
   - Both need `__rate_bytes__` and an `_eval_rate` arm, or a registered evaluator.
5. **jaxpr test.** A step-stage `defer(f)(Time())` with `f = lambda t: jnp.sin(t)`, shared as one
   object by three flows. `jax.make_jaxpr(cm.vector_field)` must contain exactly one `sin`
   equation. Assert the count, and also note in the test docstring that the base commit gave 3.
6. **Equivalence test.** A model using `D = field_refs(defer(f)(Params(), y=StateArray(), t=Time()), DSchema)`
   in its rates must equal the same model built with `derived_fn=f` and `derived_refs(DSchema)`,
   for `hoist` both on and off.
7. **Hoisting test.** In the new-idiom model, `Param("a") * Param("b")` used beside `D` is
   slotted in the hoist table. With `derived_fn` it is not, which is today's R3.
8. **Notebook 27 section C.** Old `derived_fn` against new `field_refs(defer(...))`:
   - trajectories overlaid;
   - a bar plot of hoist-table length and `sin` count in the jaxpr.

**Sanity check.**
- `list(Param("a"))` raises rather than hanging (`test_shared_nodes.py`).
- The memo never bypasses a hoist slot.
- The jaxpr count test is present.
- The `hoist=False` equivalence tests are green.

## Step 40 — `feat/save-expressions` · Worker sonnet · Closes CX32

**Summary.** `outputs["foi"] = <step-stage expression>` is saved during the solve. That makes
`SaveFn` and `ComputedValue` unnecessary, and both are deprecated.

**Files.**
- `src/summer4/results/plan.py`, `eval.py`, `outputset.py`, `expr.py`;
- `src/summer4/flows/compiled.py`, `actualize.py` (read only);
- tests.

**Do.**
1. **The `SavedExpr` quantity** in `plan.py`: `SavedExpr(expr: RateOps, over: Property | None = None)`.
   - Declare it with `eq=False`; `__hash__` and `__eq__` are based on `_rate_bytes(expr)` and
     `over`, because `ArrayConst` holds an ndarray.
   - `_quantity_bytes` encodes `_rate_bytes(expr)` and `over`.
   - `SavePlan.flow_reads()` (`plan.py:226-237`) includes `_flow_refs(expr)`, so `keep` does not
     drop those flows (`compiled.py:1045-1047`).
   - When `over` is None, `OutputSet` infers it at plan time from
     `_sum_over_property_name(expr)` (`flows/actualize.py:325`).
2. **`SaveContext`** (`compiled.py:796`) gains `evaluate: Callable[[RateOps], Any] | None = None`.
   - The default keeps the public constructor working.
   - `observe` sets it to a closure over the same `t`, `y_arr`, prepared params, `flow_values`,
     `captures`, hoist table and memo.
   - An expression that is not in the hoist table evaluates unhoisted, which is correct by R4.
3. **`eval_quantity`** for `SavedExpr`:
   - a scalar returns the array;
   - a `GroupedRate` returns `PropertyData(PropertyMap.from_property(over), data)`, and raises if
     `over` is None or does not match `GroupedRate.properties[0]`;
   - any other shape raises `TypeError` ("save per-compartment or per-edge values with
     Compartments / FlowMass").
4. **Other plumbing.**
   - `values_for` and `dims_for_quantity` (`eval.py:141`, `:196`) give `("time",)` or
     `("time", over.name)`, and rebuild the map from `over`.
   - Both backends' `jnp.asarray` of saves (`diffrax_backend.py:124`, `euler_backend.py:52`) then
     see only arrays. Confirm this.
5. **`OutputSet.__setitem__` lifting.**
   - A value of stage `"step"` with no post node becomes `Saved(SavedExpr(value))`.
   - A value of stage `"run"` raises ("a constant is not a time series; multiply a saved
     quantity").
   - In a mixed value, maximal step-stage subtrees are lifted into `Saved(SavedExpr(...))`. The
     rebuild passes only through BinOp, UnaryOp, PostOp and Defer (via `dataclasses.replace`, or
     `Defer(fn, *args, name=)`). A step subtree under any other node type with a post sibling
     raises with the node's name.
6. **Deprecations.** The `SaveFn(...)` and `ComputedValue(...)` constructors emit a
   `DeprecationWarning` naming the replacement. Existing tests that use them add
   `pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:summer4")`, or a
   per-test mark. `pyproject.toml` has no filterwarnings section; do not add a global one.
7. **Tests.**
   - FOI by age saved via an expression equals the `GroupedOutput(Capture)` route.
   - Seasonal beta saved as an expression equals the `SaveFn` route.
   - An expression reading a `FlowRef` saves correctly with `keep`.
   - The Euler and Diffrax backends, and `jit`, are each covered.
   - The `GroupedRate`-without-`over` error is tested.
8. **Notebook 27 section D.** FOI by age, saved by expression, overlaid on the `GroupedOutput`
   route.

**Sanity check.**
- Both backends are tested.
- `flow_reads` covers `SavedExpr`.
- The warning text names the replacement.
- CX32 is `done`.

## Step 41 — `feat/derived-fn-deprecation` · Worker sonnet · Closes CX31

**Summary.** `compile(derived_fn=...)` is deprecated in favour of `field_refs(defer(...))`. With
that, parameter hoisting is model-wide again for any model in the new idiom.

`derived_fn` models keep today's exact behaviour: R3 still applies to them and nothing is
rewritten. The `params_are_static` flag is deleted in step 45.

**Files.** `src/summer4/flows/compiled.py`, `stages.py`, `rates.py`; `docs/dev/run-stages.md`;
`docs/dev/rate-expressions.md`; `futureplans/derived-fn-blocks-hoisting.md`; tests.

**Do.**
- `FlowModel.compile(derived_fn=...)` emits a `DeprecationWarning` naming `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)`.
- Leave the `params_are_static` keyword on `rate_stage` and `build_hoist_table` as it is. Its
  36 test call sites and the custom `__rate_stage__` hooks depend on it. Document it as "only
  false for deprecated `derived_fn` models". Step 45 deletes it.
- `derived_refs` docstring: deprecated in favour of `field_refs`.
- Docs:
  - `run-stages.md`: R3 is marked "deprecated path only".
  - `rate-expressions.md` § escape hatches: the new idiom is first.
  - `futureplans/derived-fn-blocks-hoisting.md` is marked done per its README.
- Tests:
  - every existing `derived_fn` test stays green under a filterwarnings mark;
  - a test asserting that the hoist table of a `derived_fn` model is unchanged from today;
  - a test asserting that the new-idiom equivalent hoists its parameter products.
- Notebook 27 section E: a bar plot of hoist-table length and per-run wall time, old idiom against
  new.

**Sanity check.**
- No `derived_fn` model's numbers or hoist table changed. The existing tests are untouched apart
  from warning marks.
- The hoisting gain for new-idiom models is asserted by a test.
- CX31 is `done`.

## Step 42 — `feat/callable-doors` · Worker sonnet · Closes CX33 CX11

**Summary.** The last parameter-blind callable doors take rate-expression arguments:

- InitialPopulation callables are deprecated in favour of `defer(f)(Params())`, or
  `defer(f)(Param("a"), ...)`.
- `ForceOfInfection(kind=custom_kind(fn, Param("k")))` lets a custom FOI shape learn parameters.

`Transform` stays and is **not** deprecated. It already takes explicit rate-expression arguments
and is keyed by `named` (step 34). Replacing it with `Overwrite(defer(...))` would change:

- its precedence, from 2 to 0;
- the overwrite-overlap checks (`actualize.py:218-248`);
- argument alignment (`compiled.py:621-638`).

Its docs reposition it as "the adjustment-level `defer`".

**Files.** `src/summer4/flows/initial.py`, `src/summer4/epi/infection.py`,
`src/summer4/epi/__init__.py`, `src/summer4/flows/rates.py` (Transform docstring only),
`docs/cookbook/01-custom-rates.ipynb`, tests.

**Do.**
- **InitialPopulation callables** (`initial.py:33-34`, called at `:419-420` and `:490-491`) emit
  a `DeprecationWarning`. Confirm that `defer(f)(Params())` is accepted as run-stage at
  `initial.py:293/338/367`, and that it evaluates to the same base values and weight tables.
- **`custom_kind`** in `infection.py`: `@dataclass(frozen=True) class custom_kind`, with fields
  `fn` and `args: tuple[RateOps, ...]`.
  - The FOI's `kind` evaluator calls `fn(infectious, denominator, *[eval_child(a) for a in args])`.
  - The `args` join the FOI's `__rate_children__`.
  - `__rate_bytes__` folds `_callable_key(fn)` and the args' bytes.
  - A bare callable `kind` still works and emits a `DeprecationWarning`.
- **Tests.**
  - The deprecated callable initial population and its `defer` replacement give equal `y0`.
  - `custom_kind(saturating, Param("k"))` matches a hand-written equivalent.
  - `custom_kind` has a finite gradient with respect to `k`.
  - A `Param` inside `custom_kind` hoists.
- **Notebook 27 section F.** A saturating FOI with a learned exponent `k`, plotted for three
  values of `k`.

**Sanity check.**
- Both replacements have equality tests.
- Transform's behaviour is unchanged (its tests are untouched).
- CX33 and CX11 are `done`.

## Step 43 — `feat/typed-stages` · Worker opus · Checker opus (fresh) · Closes CX34

**Summary.** The stage becomes a type parameter, `RateOps[Run | Step | Post]`. mypy then rejects
three mistakes:

- a post-solve expression in a rate slot;
- `Time()` in an initial population;
- a step-stage likelihood scale.

Answers the type-expressibility question: more is typed after unification than before.

**Files.** `src/summer4/flows/stages.py`, `rates.py`, `compiled.py`, `initial.py`, `edges.py`;
`src/summer4/epi/infection.py`, `mixing.py`, `calibration/likelihoods.py`;
`src/summer4/results/expr.py`, `outputset.py`; the new `tests/test_typed_stages.py` and
`tests/typing/`; `scripts/spikes/typed_stage_spike.py`.

**Do.**
1. **Spike first.** In `scripts/spikes/typed_stage_spike.py`, define `class Post`,
   `class Step(Post)` and `class Run(Step)` in a stand-in module, with a covariant `S_co`.
   Confirm that `mypy --strict`:
   - infers `Const + Time` as `RateOps[Step]`;
   - infers `Saved * Param` as `RateOps[Post]`;
   - errors on passing a `RateOps[Post]` where `RateOps[Step]` is expected;
   - works with `@dataclass(frozen=True, slots=True)` generic subclasses.

   Record go or no-go in the step plan. On **no-go**, fall back to two classes, `Expr` (≤ step)
   and `PostExpr`, and record why.
2. **Make `RateOps` `Generic[S_co]`.** The markers live in `stages.py`.
   - `Run` leaves: `Const`, `ArrayConst`, `FieldRef`, `Params`.
   - `Step` leaves: `Time`, `FlowRef`, `Reduce`, `Capture`, `StateArray`.
   - `Post` leaves: `Saved`, `OutputRef`, `PostOp`.
   - `ForceOfInfection` is `RateOps[Step]`.
   - The interior nodes and `Item` are generic.
   - Operators return `RateOps[S1 | S2]`.
   - `defer(fn)` returns a callable typed `(*args: RateOps[S]) -> RateOps[S]`.
   - The named helpers (`exp`, `log`, `tanh`, `sqrt`, `floor`, `maximum`, `minimum`, `clip`) get
     overloads: `RateOps[S] -> RateOps[S]`, and `Output -> Output`.
3. **Annotate the slots.**
   - Flow `rate=`, `Multiply.value`, `Overwrite.value` and `Transform` args:
     `RateOps[Step] | float`.
   - InitialPopulation values, likelihood `sd=` / `dispersion=`, and `eval_closed`:
     `RateOps[Run] | float`.
   - `OutputSet.__setitem__`: `RateOps[Post] | <save spec>`.
4. **Keep every runtime check.** Typing is additive.
5. **Typing tests.** `tests/test_typed_stages.py` runs `mypy.api.run` on the files in
   `tests/typing/` and parses `# E: <message fragment>` markers. Exclude `tests/typing/` from
   pytest collection with `collect_ignore` in `conftest.py`.
   - One rejection per mistake.
   - One acceptance per slot.
6. **`mypy --strict src/summer4` is clean.**
7. **Notebook 27 section G.** The three mistakes as code strings, with `mypy.api.run` output shown
   beside the runtime error each one gave before.

**Sanity check.**
- The spike result is recorded.
- `mypy --strict` is green.
- The typing snippets cover the three rejections and the per-slot acceptances.
- No `cast` was added outside `field_refs` and `derived_refs`.
- CX34 is `done`.

## Step 44 — `docs/unified-lazy-idiom` · Worker sonnet

**Summary.** The design docs and the summer2 migration table describe the one idiom, and
notebook 27 becomes the finished user gate.

**Files.** `docs/dev/rate-expressions.md`, `run-stages.md`, `docs/user/07-from-summer2.md`,
`docs/cookbook/01-custom-rates.ipynb`, notebook 27.

**Do.**
- `rate-expressions.md`:
  - Rewrite § *Could we have a better compute graph?* as *The unified idiom*: the stage lattice,
    sources as leaves, `defer` everywhere, typed stages.
  - Keep the decision history.
- `run-stages.md`: describe the post stage.
- `07-from-summer2.md`: map `Function`, `Variable`, `DerivedOutput`, `Parameter`,
  `request_function_output` and `computegraph.defer` to the new spellings.
- Cookbook: the two-dunder extension rung.
- Notebook 27: an opening *What this page checks*, and a review of each section's *What to check*
  box.
- Run `pixi run -e docs docs-strict`.

**Sanity check.**
- No present-tense description of unlanded behaviour.
- Every deprecated door named in the docs names its replacement.

**Handoff.** Step 16 `next`.

## Step 45 — `chore/remove-deprecated-doors` · Worker haiku · `blocked` until the release after step 44

**Summary.** Remove everything in `futureplans/deprecated-lazy-doors-removal.md`:

- `derived_fn` / `derived_refs` / R3 / the `params_are_static` keyword;
- `SaveFn`;
- `ComputedValue`;
- InitialPopulation callables;
- bare-callable FOI `kind`.

Migrate the remaining callers:

- the old-idiom sections of notebook 27, which become "before" code cells shown as text;
- the 7 coverage-ledger mentions;
- `docs/textbook` uses;
- the warning marks.

**Sanity check.**
- The inventory is fully removed.
- No `filterwarnings` marks for summer4 deprecations remain.
- The release notes list each removal with its replacement.
- `pixi run coverage` is green.

---

## Critical files

- **Flows:** `src/summer4/flows/rates.py`, `compiled.py`, `stages.py`, `algebra.py`,
  `initial.py`, `actualize.py`.
- **Results:** `src/summer4/results/outputset.py`, `plan.py`, `eval.py`, `output.py`, and the new
  `expr.py`.
- **Epi:** `src/summer4/epi/infection.py`, `src/summer4/epi/calibration/likelihoods.py`,
  `model.py`, `posterior_runs.py`.
- **Solvers:** `src/summer4/solvers/diffrax_backend.py`, `euler_backend.py`.
- **Docs:** `docs/dev/roadmap.md`, `docs/evaluation/composability.md`,
  `docs/dev/rate-expressions.md`, `docs/dev/run-stages.md`, `AGENTS.md`.
- **Checkers (read only):** `scripts/roadmap_check.py`, `scripts/composability_check.py`,
  `tests/test_roadmap.py`.

## Verification

- **Every step.**
  - Targeted pytest by the implementer.
  - Light checks locally.
  - `pixi run test`, `test-all` and `docs-strict` on the mini, run by the orchestrator.
  - A `VERDICT: PASS` from the checker.
  - The user runs the phase notebook section (`pixi run notebook`) before merge.
- **Step 0.**
  - `pixi run roadmap`, `pixi run composability`, `pixi run coverage`, and the roadmap and
    composability tests are green.
  - The roadmap shows step 31 `next`.
  - Landing-order pairs resolve.
- **End of Phase L.** The five sections of notebook 26 render; CX7 and CX26–CX29 are `done`.
- **End of Phase M.**
  - Notebook 27 shows each old door beside its replacement, with identical trajectories.
  - `grep -rnw OutputExpr src` is empty.
  - `mypy --strict` is green.
  - CX11 and CX30–CX34 are `done`.
