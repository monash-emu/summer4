# Documentation

```bash
pixi run -e docs docs         # HTML into docs/_build/html
pixi run -e docs docs-strict  # warnings become errors
pixi run -e docs docs-serve   # http://localhost:8765
pixi run -e docs docs-clean   # remove docs/_build
```

## Stack

| Piece | Role |
|---|---|
| `sphinx` | Build system |
| `myst-nb` | Markdown (MyST) pages and executed notebooks |
| `pydata-sphinx-theme` | Theme |
| `sphinx.ext.autosummary` + `autodoc` | Generated API reference |
| `sphinx-autodoc-typehints` | Signatures from annotations |
| `sphinxcontrib-mermaid` | Diagrams |
| `sphinx-design`, `sphinx-copybutton` | Admonitions, grids, copy buttons |
| `jupyter-cache` | Notebook execution cache |

The `docs` environment installs JAX because {doc}`../user/08-flows` compiles a
vector field at build time. Taxonomy pages remain NumPy-only.

## Notebooks on this site

Notebooks are committed **cleared** and executed by Sphinx:

```python
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_cache_path = "_build/.jupyter_cache"
```

Three rules follow.

1. **Every documented claim must be asserted, not printed.** A cell that prints
   a number proves nothing on the next build; a cell that asserts it fails the
   build when the library changes.
2. **No magics, no shell escapes.** Same rule as `examples/notebooks/`.
3. **Clear outputs before committing.** `pixi run check-notebooks` enforces
   this, and the pre-commit hook installed by `pixi run setup` rejects dirty
   notebooks at commit time.

The execution cache lives under `docs/_build/`, so `docs-clean` forces a full
re-execution. Use it when a change should have invalidated a page and did not.

## Writing a page

- **User guide** pages are notebooks. They teach one idea, build the smallest
  map that shows it, and assert the result.
- **Developer guide** pages are Markdown, except {doc}`performance`, which is a
  notebook precisely because its claims are measurements.
- Cross-reference with `{doc}` so the link is checked:
  `` {doc}`../user/04-ragged-stratification` ``.
- Reference API objects with `{class}` / `{func}` so they link into the
  generated reference.

## Honesty rule

Pages must not describe planned behaviour in the present tense. Where a summer2
or textbook capability has no summer4 equivalent, say so and link to
{doc}`../evaluation/feature-completeness` rather than writing an example that
cannot run.

Every page that makes a coverage claim — {doc}`../evaluation/docs-coverage` in
particular — should be re-derived from the source repositories when the API
changes, not edited by hand from memory.

Flows are public API. Document them in the user guide against
`from summer4 import ...`. Do not keep a second, out-of-package copy.
