# GitHub Actions CI

## Goal

Run a fast feedback suite on every push and the full merge bar on pull requests.

## Approach

- Use `prefix-dev/setup-pixi` with the locked `pixi.lock` and environment caching.
- **Push (`quick`)**: `lint`, `format-check`, and `test-quick` (pytest `-m 'not slow'`) in the default JAX 0.6 env.
- **PR (`quality` + `test`)**: `lint`, `format-check`, `check-notebooks`, `check-branch`, and full `test` across `default` and `latest` environments.
- Mark notebook smoke execution as `@pytest.mark.slow` so push stays fast while PRs still run notebooks.

## Out of scope

- Benchmarks in CI
- Dependabot for action pins
