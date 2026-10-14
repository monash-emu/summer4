Implements Step 34 of `unified-lazy-idiom.plan.md`.

## Deviations

- `Named` also rejects a non-string `name` with `TypeError`. The brief specified only the
  non-callable `TypeError` and the empty-name `ValueError`.
- The `ForceOfInfection` `kind` digest appends `_callable_key(self.kind).encode()` directly, not
  `repr(...)` of it. Unnamed bytes stay identical to the old `repr(id(...))`, because `repr` of an
  int equals its `str`.
- Section D's jitted runner evaluates `vector_field` once per rebuild, not a full `run`. This keeps
  the notebook fast; the trace count depends only on the model's digest.
- `plans/output-defer.plan.md` was not edited. The brief's "Also fix" asks for a Deviations
  correction there, but the brief's Constraints list does not include that file, and `AGENTS.md`
  and the roadmap treat committed plans as historical records. The roadmap's *Corrections to
  committed plans* table is the designated place for it.
