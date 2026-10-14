Implements Step 34 of `unified-lazy-idiom.plan.md`.

## Deviations

- `Named` also rejects a non-string `name` with `TypeError`. The brief specified only the
  non-callable `TypeError` and the empty-name `ValueError`.
- The `ForceOfInfection` `kind` digest appends `_callable_key(self.kind).encode()` directly, not
  `repr(...)` of it. Unnamed bytes stay identical to the old `repr(id(...))`, because `repr` of an
  int equals its `str`.
- Section D's jitted runner evaluates `vector_field` once per rebuild, not a full `run`. This keeps
  the notebook fast; the trace count depends only on the model's digest.
- In a follow-up at the orchestrator's direction, `plans/output-defer.plan.md` gained the
  four-test bullet, and the CX7 evidence anchor was moved to
  `src/summer4/flows/rates.py::def _callable_key`.
- Notebook 26's title cell puts the step 32, 33 and 34 sentences in separate paragraphs, in step
  order, rather than adjacent.
- Post-solve `defer(named(f, ...))(outputs...)` is accepted, but post-solve nodes are not
  digested, so the name has no effect there. That path is untested; steps 36–38 revisit it.
