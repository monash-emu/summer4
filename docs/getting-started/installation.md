# Installation

summer4's taxonomy layer depends on **NumPy only**. Compiled flows import JAX.
JAX also lives in the pixi environment matrix so solvers can be benchmarked
across versions.

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
| `docs` | 3.13 | 0.6.x | Sphinx documentation build (JAX for flow notebooks) |

```bash
pixi run -e latest test
pixi run -e nb notebook
pixi run -e docs docs
```

The `docs` environment includes JAX so the user-guide flows notebook can
compile a vector field at build time.

## With pip

The package builds with hatchling and is importable from a plain virtualenv:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[jax]"           # jax, jaxlib, diffrax, equinox
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
