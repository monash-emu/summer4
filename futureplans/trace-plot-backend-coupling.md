# `Trace.plot` is coupled to matplotlib

## What is wrong today

{meth}`Trace.plot` (`src/summer4/results/trace.py:469`) is:

```python
def plot(self, **kwargs: Any) -> Any:
    """Plot via matplotlib (lazy import). Host-side only."""
    import matplotlib.pyplot as plt

    frame = self.to_pandas()
    ax = frame.plot(**kwargs)
    plt.xlabel("time")
    return ax
```

Two couplings, neither visible from the call site:

1. It forwards `**kwargs` to `DataFrame.plot`, whose accepted arguments depend
   on `pandas.options.plotting.backend`. `plot(legend=False)` — used in
   `docs/user/09-running-and-results.ipynb` and
   `docs/user/10-targets-and-fitting.ipynb` — raises under the Plotly backend,
   which does not take `legend`.
2. It always calls `plt.xlabel("time")`. Under a non-matplotlib backend that
   creates and labels a stray empty matplotlib figure, then returns the other
   backend's object. Nothing fails; an extra blank figure just appears.

## Why it hurts

`docs/dev/plotting.md` makes Plotly through the pandas backend the
documentation convention, which puts the two at odds. The blast radius is
currently small, because `myst-nb` runs each notebook in its own kernel, so the
backend option set by one page does not reach another. But the coupling is a
trap for any user who sets the backend globally in their own session and then
calls `Trace.plot`.

## What a fix looks like

Keep `plot` as the quick matplotlib path, and make the coupling explicit rather
than incidental:

- Detect the active backend via `pandas.options.plotting.backend` and skip the
  `pyplot` call when it is not matplotlib.
- Or pass the axis label through the backend-agnostic route: label the frame's
  index (`to_pandas` already names it `"time"`) and let the backend render it,
  dropping the `pyplot` import from the happy path.

Either way, a test that sets the Plotly backend, calls `Trace.plot()`, and
asserts no matplotlib figure was created would pin the behaviour. Notebook call
sites that pass matplotlib-only kwargs should move to
`trace.to_pandas().plot(...)`.
