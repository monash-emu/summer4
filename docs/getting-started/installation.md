# Installation

summer4's taxonomy layer depends on **NumPy only**. JAX lives in the pixi
environment matrix so that later stages can benchmark solvers across versions
without forcing a JAX dependency on the core package.

## With pixi (recommended)

[pixi](https://pixi.sh) manages both the conda-level toolchain and the Python
dependencies, and the repository's task table is the project's real entry point.

```bash
git clone https://github.com/monash-emu/summer4
cd summer4
pixi install
pixi run setup            # install the git hooks
pixi run test
```

`pixi run setup` installs a pre-commit hook that formats staged Python with
Black (line length 100) and rejects notebooks that still carry outputs or
execution counts, plus a post-merge hook that copies merged plans into `plans/`.

### Environments

| Environment | Python | JAX | Purpose |
|-------------|--------|-----|---------|
| `default` | 3.13 | 0.6.x | Primary development, notebook kernel |
| `latest` | 3.13 | current | Forward-compatibility checks |
| `nb` | 3.13 | 0.6.x | JupyterLab |
| `docs` | 3.13 | — | Sphinx documentation build (NumPy only) |

```bash
pixi run -e latest test
pixi run -e nb notebook
pixi run -e docs docs
```

The `docs` environment deliberately omits JAX: nothing on this site needs it,
which is itself a fact about the current feature surface.

## With pip

The package builds with hatchling and is importable from a plain virtualenv:

```bash
pip install .
```

Optional extras exist for the stages that are not yet implemented, and installing
them today gets you the dependencies but no summer4 functionality that uses them:

```bash
pip install ".[jax]"           # jax, jaxlib, diffrax
pip install ".[calibration]"   # numpyro, optax
```

## Notebook kernel

The `default` environment includes `ipykernel`. In VS Code or Cursor, open the
repository, run `pixi install`, and select the interpreter at
`.pixi/envs/default/bin/python` (already configured in `.vscode/settings.json`).
For other Jupyter frontends:

```bash
pixi run register-kernel     # registers "Python (summer4)"
```

## Building this documentation

```bash
pixi run -e docs docs         # HTML into docs/_build/html
pixi run -e docs docs-serve   # http://localhost:8765
pixi run -e docs docs-strict  # warnings become errors
```

Every notebook on this site is executed during the build
(`nb_execution_raise_on_error = True`), so a documentation build failure is a
real signal that the library changed under the docs.
