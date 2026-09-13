# CI workflow_dispatch

## Goal

Allow running the existing CI suites manually from the GitHub Actions UI.

## Approach

- Add `workflow_dispatch` with a `suite` choice input (`quick` or `full`, default `full`).
- Extend job `if` conditions so push still runs quick, PRs still run full, and manual runs select by input.
- Default `check-branch` base to `origin/main` when `github.base_ref` is unset (manual runs).
