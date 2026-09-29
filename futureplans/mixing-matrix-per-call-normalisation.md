# MixingMatrix re-normalises FieldRef matrices every step

`MixingMatrix.resolved_matrix` (`src/summer4/epi/mixing.py`) re-normalises a
matrix that may be a `FieldRef` on every vector-field call. When the matrix
depends only on params (no `t`/`y`), that work belongs at run start.

**Today:** frequency/density FOI pays a normalisation cost per step even for
static or prepare-time matrices.

**Done when:** the mixing matrix path declares `__rate_stage__` (or equivalent)
so a param-only matrix is hoisted, or the yearly/static stack is built in
`prepare_fn` and referenced as a constant array.

Related: yearly mixing stacks for Kiribati belong in `prepare_fn` (WP10
`BayesianModel` has no separate `preprocess=`).
mixing stack.

**After step 5:** `Lookup` gathers one row of a parameter stack inside the
vector field, so that stack is not rebuilt on every call. `resolved_matrix`
still row-normalises the gathered matrix on every call when
`normalize="rows"`. Hoisting that normalisation is still open.

**After roadmap step 18:** contact matrices reach `MixingMatrix` through
`ContactMatrix.to_mixing` / `ScaledContacts.to_mixing`, which default to
`normalize="none"`, so the per-call row normalisation does not apply to them. A
`ScaledContacts` matrix is a constant plus one `ArrayConst * factor` term per
scaled setting; with a `step(Time(), ...)` factor it is step-stage by nature,
with a `Param` factor it is param-only and would benefit from the same hoisting
this note asks for. `check_reciprocal=True` on such a matrix runs a host
callback every evaluation — leave it off inside calibration.
