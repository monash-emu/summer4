---
name: wp12-release
description: Land the flows stack on main, tag a pinnable v0.2.0a1 release with honest packaging, and add downstream smoke CI — steps 1 to 3 of docs/dev/roadmap.md.
---

# WP12 — pinnable release (steps 1–3)

## Context

Everything summer4 can do beyond the compartment taxonomy lives on unmerged
branches. `main` carries the taxonomy and CI; a clone of it cannot
`import summer4.flows`. Downstream work — both tuberculosis ports, and any user
who wants to try the library — has to pin a commit SHA on a stacked branch that
may be rewritten.

This plan lands the stack, tags a release, and proves the release installs. It
**supersedes §12 of `plans/tb-ports-feature-completeness.plan.md`**, which was
written when the stack tip was `feat/epi-infection-mixing` and did not know about
the five stacked pull requests that now exist. Read that section for intent; take
branch names, PR numbers and sequence from here.

**Closes (ports ledger):** `KI23` `TM9` (in step 2, when the tag exists — not in
step 1). **Closes (API ledger):** nothing; WP12 adds no capability.
**Unblocks:** step 20, the tb_macro port, which can start as soon as step 2 tags.

Ledger IDs and step numbers in this plan refer to
`docs/evaluation/coverage-ledger.md`, `docs/evaluation/tb-ports.md` and
`docs/dev/roadmap.md`.

## The stack, as it actually is

Five pull requests, each based on the one below it:

| PR | Branch | Base | Contains |
|---|---|---|---|
| #3 | `feat/trace-select-submap` | `main` | — |
| #4 | `feat/flow-outputs` | `feat/diffrax-solver` | #3 |
| #6 | `fix/describe-params` | `main` | #3, #4 |
| #7 | `feat/model-stratify` | `feat/remove-epimodel` | #3, #4, #6 |
| #8 | `docs/cookbook-custom-rates` | `feat/model-stratify` | #3, #4, #6 and an older `feat/model-stratify` |

Verified with `git merge-base --is-ancestor`. **`feat/model-stratify` contains
every other branch's work except the cookbook's own commit**, which is why one
merge of it does the job. `docs/cookbook-custom-rates` merged an older
`feat/model-stratify`, so it is *not* a superset and must be handled separately
after the main merge.

The roadmap and the plans that go with it land on `docs/roadmap-next-steps`,
merged into `feat/model-stratify` before step 1, so they travel with the stack.

---

## Step 1 — `chore/merge-flows-stack`: land the stack on `main`

The branch name exists for the roadmap's benefit and for any fix-ups the merge
needs; the merge itself happens through PR #7.

### 1a. Prove the tip is green before touching anything

On `feat/model-stratify`, with the roadmap branch already merged into it:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage && pixi run roadmap
pixi run -e docs docs-strict
pixi run test-all
```

If any of these fail, fix them on `chore/merge-flows-stack` (cut from
`feat/model-stratify`) and merge that into `feat/model-stratify` first. **Do not
merge a red tip into `main` intending to fix it there** — `main` is the branch
everything downstream pins.

### 1b. Retarget and merge PR #7

```bash
git fetch origin
gh pr edit 7 --base main
gh pr diff 7 --name-only | head -50
git diff --stat main...feat/model-stratify
```

Read the diff list. It should touch `src/summer4/`, `tests/`,
`examples/notebooks/`, `docs/`, `plans/`, `futureplans/`, `scripts/`,
`pyproject.toml`, `pixi.toml` and `pixi.lock`, and nothing outside the
repository's usual shape. Anything surprising is a reason to stop and ask.

Merge with a **merge commit**:

```bash
gh pr merge 7 --merge
```

Not `--squash`: the coverage ledger's *Landed phases* table cites each phase's
branch and tip SHA as history, and squashing makes those citations unresolvable.

### 1c. Bring the cookbook branch across

`docs/cookbook-custom-rates` is behind the merged tip:

```bash
git switch docs/cookbook-custom-rates
git fetch origin && git rebase origin/main
pixi run test && pixi run -e docs docs-strict
git push --force-with-lease
gh pr edit 8 --base main && gh pr merge 8 --merge
```

If the rebase conflicts in more than a couple of files, merge `origin/main` into
the branch instead of rebasing and say so in the handoff.

### 1d. Close the superseded pull requests

PRs #3, #4 and #6 targeted branches other than `main`, so GitHub will not close
them automatically. Close each with a comment naming the merge commit:

```bash
for pr in 3 4 6; do
  gh pr close "$pr" --comment "Superseded: landed on main via #7 (<merge SHA>)."
done
```

Leave the branches in place; the ledger's history table refers to them.

### 1e. Prove `main` is importable from a clean clone

From a scratch directory outside the working tree (use the session scratchpad,
not `/tmp` by hand):

```bash
git clone --depth 1 <repo url> summer4-check && cd summer4-check
pixi install
pixi run python -c "import summer4; from summer4.flows import FlowModel; print(summer4.__version__)"
```

This is the check that catches a module that only ever worked because of an
editable install.

### 1f. Update the delivery record and cut the next branch

On `chore/merge-flows-stack`, merged into `main` through its own small PR:

- `docs/evaluation/coverage-ledger.md`, *Delivery status*: this is where the
  warning admonition dies if it has not already. See step 2 — do it in whichever
  step actually lands first, once, and not twice.
- The roadmap handoff (below).

Then cut `chore/release-v0.2` from `main` for step 2.

### Step 1 exit criteria

- `gh pr list --state open` shows no stacked flows PR.
- A clean clone of `main` imports `summer4.flows`.
- The roadmap's current position reads step 2.

---

## Step 2 — `chore/release-v0.2`: honest packaging and a tag

### 2a. Fix the dependency story (`pyproject.toml`)

`import summer4` imports `summer4.jax`, and `results/result.py` imports jax, so
the "NumPy only" claim in `docs/getting-started/installation.md` is false today.
Two honest resolutions exist; **take the first** unless the user says otherwise:

1. **Move `jax`, `jaxlib`, `diffrax` and `equinox` from the `jax` extra into
   `[project].dependencies`.** JAX is the primary runtime target per `AGENTS.md`,
   so this matches reality. Keep an empty-but-documented `jax` extra name for one
   release if anything already installs `summer4[jax]`, or drop it and say so in
   the installation page.
2. Make the top-level import lazy, and add a test that `import summer4` succeeds
   in an environment without JAX. This is more work and buys a claim nobody has
   asked for.

Also:

- Add a `frames` extra: `polars`, `pyarrow`. `Trace.to_frame` / `to_pandas` go
  through polars and no extra currently declares it — a fresh install hits an
  `ImportError` on a documented method.
- Bump `version` to `0.2.0a1`, and the `release` / `version` strings in
  `docs/conf.py` with it.

### 2b. Document the install truthfully

`docs/getting-started/installation.md`:

```toml
[pypi-dependencies]
summer4 = { git = "https://github.com/monash-emu/summer4.git", tag = "v0.2.0a1", extras = ["calibration", "pandas", "frames"] }
```

State the platform support as it is: `osx-arm64` and `linux-64`. JAX has no
native Windows build; WSL2 is the Windows route. Remove any "NumPy only"
wording.

### 2c. Rewrite the delivery record

`docs/evaluation/coverage-ledger.md`, *Delivery status*:

- Delete the ```{admonition} The flows stack is not on `main` ``` warning. It is
  false once step 1 lands, and a false warning in the authoritative ledger is
  worse than no warning.
- Keep the *Landed phases* table as history; add a line saying every phase in it
  is now on `main` as of the tag.
- Replace the forward-looking half of *Next steps after WP3* with a pointer to
  `docs/dev/roadmap.md`, so position is recorded in exactly one place.

`docs/evaluation/tb-ports.md`:

- Move `KI23` and `TM9` to `full`, `Closed by` to `—`, `Route today` to
  "Pin `tag = "v0.2.0a1"`".
- Delete the paragraph about pinning the unmerged stack commit.
- Update the quoted `rows complete today` phrases for both models.

```bash
pixi run coverage-write && pixi run coverage
```

The readiness table's `today` row must then read 7 / 23 and 5 / 9.

### 2d. Tag

```bash
git switch main && git pull
git tag -a v0.2.0a1 -m "summer4 0.2.0a1 — flows, results, solvers, epi, stratification."
git push origin v0.2.0a1
```

Tag only after the PR is merged, so the tag names a commit on `main`.

### Step 2 exit criteria

- `pixi run coverage` green with `KI23` / `TM9` at `full`.
- The tag exists on `main` and the installation page quotes it.
- Step 20 (tb_macro port) is declared unblocked in the handoff.

---

## Step 3 — `chore/downstream-smoke-ci`

`tb-ports-feature-completeness` §12d is the scope. Add a CI job that installs
summer4 **the way a downstream repository will** — by git tag, into a scratch
pixi project, with no editable checkout — and runs a three-compartment SIR
through `compile` → `run` → `Result`.

- Job lives alongside the existing workflows in `.github/workflows/`. Put it in
  the full (pull-request) suite, not the quick push suite: it resolves an
  environment from scratch and is slow.
- It must pin the **tag**, not `main`, so the job proves what users will get.
- Assert something real about the result — a final susceptible count within
  tolerance of a hand-computed value — not merely that nothing raised.
- Run it on `osx-arm64` and `linux-64` if the runners allow; otherwise
  `linux-64` and a note saying so.

The job will fail the first time a dependency is missing from the wheel. That is
the point; fix the packaging rather than loosening the job.

### Step 3 exit criteria

- The job is green on its own PR.
- `.github/workflows/` documents in a comment why the job avoids an editable
  install.

---

## Verification (whole package)

- A clean clone of `main` at the tag, in a fresh pixi environment, runs a model.
- `pixi run coverage` and `pixi run roadmap` are green.
- `docs/evaluation/coverage-ledger.md` contains no claim that the stack is
  unmerged.
- `docs/evaluation/tb-ports.md` readiness `today` row: 7 / 23 and 5 / 9.

## What this package deliberately does not do

- **It does not publish to PyPI.** Installing from a git tag is the decided
  distribution route for now; a PyPI release is a separate decision with its own
  naming and ownership questions.
- **It does not promise API stability.** `0.2.0a1` is an alpha. Say so in the
  installation page rather than implying otherwise by tagging.
- **It does not clean up the branch names in the ledger's history table.** They
  are the record of how the work landed.
