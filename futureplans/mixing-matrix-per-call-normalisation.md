# MixingMatrix re-normalises FieldRef matrices every step

`MixingMatrix.resolved_matrix` (`src/summer4/epi/mixing.py`) re-normalises a
matrix that may be a `FieldRef` on every vector-field call. When the matrix
depends only on params (no `t`/`y`), that work belongs at run start.

**Today:** frequency/density FOI pays a normalisation cost per step even for
static or prepare-time matrices.

**Done when:** the mixing matrix path declares `__rate_stage__` (or equivalent)
so a param-only matrix is hoisted, or the yearly/static stack is built in
`prepare_fn` and referenced as a constant array.

Related: `futureplans/wp10-preprocess-is-prepare-fn.md` for Kiribati's yearly
mixing stack.
