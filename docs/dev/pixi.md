# Environments with pixi

[pixi](https://pixi.sh) manages the toolchain, the Python dependencies and the
task table. `pixi.toml` is the single entry point — there is no Makefile and no
`requirements.txt`.

## Features and environments

Features are dependency groups; environments are combinations of features.

| Environment | Features | Python | JAX | Purpose |
|---|---|---|---|---|
| `default` | `jax06`, `dev` | 3.13 | 0.6.x | Primary development and the notebook kernel |
| `latest` | `jaxlatest`, `dev` | 3.13 | current | Forward-compatibility checks |
| `nb` | `jax06`, `notebooks`, `dev` | 3.13 | 0.6.x | JupyterLab |
| `docs` | `docs` | 3.13 | 0.6.x | Sphinx build (includes JAX for flow notebooks) |

The JAX matrix exists so that solver work can be benchmarked across versions
without a flag day. Taxonomy modules do not import JAX; `import summer4` does.

## Tasks

```bash
pixi run setup            # install git hooks
pixi run test             # unit, property and notebook smoke tests
pixi run test-all         # tests in default and latest
pixi run lint             # ruff + mypy --strict on src/summer4
pixi run format           # black, line length 100
pixi run format-check     # black --check
pixi run check-notebooks  # reject notebooks with outputs
pixi run check-branch     # enforce the feature bar
pixi run bench            # pytest-benchmark table
pixi run bench-json       # benchmark JSON keyed by JAX version
pixi run coverage         # feature coverage from the ledger
pixi run -e nb notebook   # JupyterLab on examples/notebooks
pixi run -e docs docs     # build this site
pixi run -e docs docs-strict   # build with warnings as errors
pixi run -e docs docs-serve    # serve at http://localhost:8765
```

## Adding a dependency

Add it to the feature that needs it, not to the workspace, unless every
environment needs it:

```toml
[feature.dev.dependencies]
pytest = ">=8.0"

[feature.docs.pypi-dependencies]
myst-nb = ">=1.1"
```

Then `pixi install -e <env>` and commit the updated `pixi.lock`. Keep the lock
file in the commit that changes `pixi.toml`; a lock that disagrees with the
manifest is the most common cause of an unreproducible CI run.

```{admonition} JAX is a core dependency
:class: note

`import summer4` imports JAX. `jax`, `jaxlib`, `diffrax` and `equinox` are in
`[project].dependencies`. The `jax` extra remains for one release as a
compatibility alias listing the same packages. Taxonomy modules still do not
import JAX.
```
