# Plans

Accepted design plans live here. Feature branches copy the Cursor plan onto the
branch as `plans/<slug>.plan.md`. Merging the branch into `main` is what copies
the plan into this workspace archive.

The `post-merge` hook (`pixi run setup`) also copies any incoming `*.plan.md`
from `.cursor/plans/` into this directory if it is not already present.

Do not rewrite historical plans. Start a new file for follow-up work.
