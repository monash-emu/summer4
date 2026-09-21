# Installation

summer4 is an alpha (`0.2.0a3`). The API is not stable, and the package is not
published to PyPI. Install from the git tag. Release notes for each tag are
under {doc}`../releases/index`.

Importing summer4 imports JAX, so `jax`, `jaxlib`, `diffrax` and `equinox` are
core dependencies. The version string is
`importlib.metadata.version("summer4")`.

## Downstream project

A pixi project that depends on the tagged release:

```toml
[pypi-dependencies]
summer4 = { git = "https://github.com/monash-emu/summer4.git", tag = "v0.2.0a3", extras = ["calibration", "pandas", "frames"] }
```

Or with pip:

```bash
pip install "summer4[calibration,pandas,frames] @ git+https://github.com/monash-emu/summer4.git@v0.2.0a3"
```

### Extras

| Extra | Packages | When you need it |
| --- | --- | --- |
| `calibration` | numpyro, optax | Sampling and MAP fits |
| `pandas` | pandas | `Output.to_pandas` |
| `frames` | polars, pyarrow | `Output.to_frame`, and the Arrow conversion `to_pandas` uses |
| `jax` | jax, jaxlib, diffrax, equinox | Compatibility alias for this release. These packages are already core dependencies, so `pip install summer4[jax]` still works |

### Platforms

This repository's pixi environments are solved for `osx-arm64` and `linux-64`.

JAX's own support, from the
[installation table](https://docs.jax.dev/en/latest/installation.html):

- **CPU:** Linux x86_64, Linux aarch64, macOS Apple Silicon, and Windows x86_64.
  The Windows `jaxlib` wheel is experimental; you may also need the Microsoft
  Visual Studio 2019 redistributable.
- **NVIDIA GPU:** Linux yes. Native Windows no. Windows via WSL2 is
  experimental.
- **AMD GPU:** Linux yes. Native Windows no. WSL2 is experimental.

There is no Windows on ARM build.

## With pixi (contributors)

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
