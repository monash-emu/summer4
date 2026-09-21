# Plotting

summer4 does not own a charting layer, and it should not. An `Output` converts to
a dataframe, and the dataframe is plotted by whatever the reader already uses.
What this page settles is what *the documentation* uses, so that notebooks do
not each make the choice again.

## The convention: Plotly through the pandas backend

Set the backend once at the top of a notebook, then plot dataframes:

```python
import pandas as pd
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pio.renderers.default = "notebook_connected"

figure = output.to_pandas().plot(title="Prevalence")
figure.update_layout(yaxis_title="people")
figure
```

This is the idiom the summer2 documentation used, so a reader arriving from
summer2 recognises it, and it needs nothing from `summer4` itself —
{meth}`~summer4.results.output.Output.to_pandas` already returns a frame indexed
by date when the result carries an {class}`~summer4.time.Epoch`.

Leave the figure as the last expression of the cell. Do not call `figure.show()`:
under Sphinx it writes a second copy of the figure into the page.

## Why `notebook_connected`

Plotly's default renderer emits `application/vnd.plotly.v1+json`. `myst-nb` has
no handler for that mime type, so the figure builds, executes, and then renders
as nothing on the site. `notebook_connected` emits `text/html` containing the
plot div plus a script tag that pulls plotly.js from the CDN, which Sphinx
embeds as-is.

Do not use the plain `notebook` renderer. It inlines the whole plotly.js bundle
— about 4 MB — into every figure.

Because each figure loads its own matching plotly.js, `conf.py` deliberately
does **not** pin a version in `html_js_files`; doing so loads a second major
version of the library into the same page.

## `Output.plot` stays matplotlib

{meth}`~summer4.results.output.Output.plot` imports matplotlib and is the quick
host-side look at a result — at a REPL, or in a test. It is not backend-aware:
it passes `**kwargs` through to `DataFrame.plot`, so calls like
`plot(legend=False)` raise under the Plotly backend, and it always sets the
x-label through `pyplot`. Prefer `to_pandas().plot(...)` in documentation, and
keep `Output.plot` for interactive work.

## Scope

Backend selection is a pandas *global*, but `myst-nb` executes each notebook in
its own kernel, so a notebook that sets the Plotly backend does not affect any
other page in the same build.
