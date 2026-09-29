"""Post-calibration figures (step 28): spaghetti, ribbons, scenarios, diagnostics.

Every function returns a ``plotly.graph_objects.Figure``. Each takes ``fig=`` to
draw onto a figure you already have, so layers compose rather than being
options of one function::

    fig = wf.plot_ribbons(posterior_runs, "infectious")
    wf.plot_spaghetti(optimised_runs, "infectious", fig=fig, name="multi-start fits")
    wf.add_targets(fig, bm.targets.targets[0])

The data behind each figure is public on its input — :meth:`PosteriorRuns.samples`,
:meth:`PosteriorRuns.difference`, :attr:`OptimizeRun.loss_trace`,
:attr:`MCMCRun.samples` — so a figure you want drawn differently is a few lines
of your own Plotly. Plotly is imported lazily (``summer4[calibration]``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from summer4.epi.calibration.workflow.candidates import Candidates

DEFAULT_QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)
"""Quantiles drawn by :func:`plot_ribbons` and :func:`plot_scenarios` by default."""

_TARGET_MARKER = {"color": "black", "size": 8, "symbol": "circle"}


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "wf.plot_* requires plotly (calibration extra): pip install summer4[calibration]"
        ) from exc
    return go


def _palette() -> list[str]:
    import plotly.colors  # type: ignore[import-untyped]

    return list(plotly.colors.qualitative.Plotly)


def _rgba(color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r, g, b, alpha)``."""
    import plotly.colors

    r, g, b = plotly.colors.hex_to_rgb(color)
    return f"rgba({int(r)}, {int(g)}, {int(b)}, {alpha:g})"


def _new_or(fig: Any) -> tuple[Any, bool]:
    """Return ``(figure, created)``: ``fig`` itself, or a fresh ``go.Figure``."""
    if fig is not None:
        return fig, False
    go = _require_plotly()
    return go.Figure(), True


def _label(fig: Any, created: bool, title: str | None, default: str, **axes: str) -> None:
    """Title and axis labels on a new figure; only an explicit ``title`` on an existing one."""
    if created:
        fig.update_layout(title=title or default, **axes)
    elif title is not None:
        fig.update_layout(title=title)


def _target_series(target: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(target.times, dtype=float).reshape(-1)
    values = np.asarray(target.values, dtype=float)
    if values.size != times.size:
        raise ValueError(
            f"Target {target.key!r} holds {values.shape} values for {times.size} times; only a "
            "single series can be overlaid (reduce a stratified target first)."
        )
    return times, values.reshape(-1)


def _select_targets(targets: Any, output: str | None) -> list[Any]:
    """A ``Target`` → itself; a sequence → all; a ``TargetSet`` → keys matching ``output``."""
    inner = getattr(targets, "targets", None)
    if inner is not None and not hasattr(targets, "key"):
        chosen = [t for t in inner if output is None or t.key == output]
        if not chosen:
            keys = [t.key for t in inner]
            raise ValueError(
                f"No target is keyed {output!r} (keys: {keys}). To overlay a target on a "
                "differently named output, pass the Target itself, e.g. "
                "targets=bm.targets.targets[0]."
            )
        return chosen
    if hasattr(targets, "key"):
        return [targets]
    return list(targets)


def add_targets(
    fig: Any,
    targets: Any,
    *,
    output: str | None = None,
    name: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> Any:
    """Overlay calibration targets on ``fig`` as black markers, one trace per target.

    ``targets`` is a ``Target``, a sequence of them, or a ``TargetSet`` — whose
    targets keyed ``output`` are drawn (every target when ``output`` is
    ``None``). ``name`` defaults to ``"target <key>"``. Returns ``fig``.
    """
    for target in _select_targets(targets, output):
        times, values = _target_series(target)
        fig.add_scatter(
            x=times,
            y=values,
            mode="markers",
            marker=_TARGET_MARKER,
            name=name or f"target {target.key}",
            row=row,
            col=col,
        )
    return fig


def _gapped(times: np.ndarray, traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of ``traj`` joined into one line with ``nan`` gaps; plus the row index per point."""
    n, t = traj.shape
    x = np.tile(np.append(times, np.nan), n)
    y = np.concatenate([traj, np.full((n, 1), np.nan)], axis=1).reshape(-1)
    draw = np.repeat(np.arange(n), t + 1)
    return x, y, draw


def plot_spaghetti(
    runs: Any,
    output: str,
    *,
    scenario: str | None = None,
    n: int | None = 50,
    seed: int = 0,
    targets: Any = None,
    fig: Any = None,
    name: str | None = None,
    color: str | None = None,
    opacity: float = 0.35,
    title: str | None = None,
) -> Any:
    """One line per sampled draw of ``output`` from :class:`PosteriorRuns`.

    ``scenario`` defaults to the first scenario. ``n`` draws are picked at random
    (``seed``) when there are more; ``None`` draws every one. The lines are a
    single trace (draws separated by gaps, hover shows the draw index), so a
    layer is one legend entry. ``targets`` is passed to :func:`add_targets`
    with ``output``. Returns the figure (``fig``, when given).
    """
    scen = runs.scenario_names[0] if scenario is None else scenario
    traj = np.asarray(runs.samples(scen, output), dtype=float)
    if n is not None and int(n) < traj.shape[0]:
        rng = np.random.default_rng(int(seed))
        traj = traj[np.sort(rng.choice(traj.shape[0], size=int(n), replace=False))]
    fig, created = _new_or(fig)
    colour = color or _palette()[len(fig.data) % len(_palette())]
    x, y, draw = _gapped(np.asarray(runs.times, dtype=float), traj)
    fig.add_scatter(
        x=x,
        y=y,
        customdata=draw,
        mode="lines",
        line={"color": colour, "width": 1},
        opacity=float(opacity),
        name=name or f"{scen}: {traj.shape[0]} draws",
        hovertemplate="draw %{customdata}<br>time %{x}<br>%{y}<extra></extra>",
    )
    if targets is not None:
        add_targets(fig, targets, output=output)
    _label(
        fig,
        created,
        title,
        f"{output}: {traj.shape[0]} sampled trajectories ({scen})",
        xaxis_title="time",
        yaxis_title=output,
    )
    return fig


def _pairs(q: Sequence[float]) -> tuple[list[tuple[float, float]], float | None]:
    qs = sorted(float(v) for v in q)
    if not qs or any(not 0.0 <= v <= 1.0 for v in qs):
        raise ValueError(f"quantiles must lie in [0, 1]; got {list(q)}.")
    half = len(qs) // 2
    middle = qs[half] if len(qs) % 2 else None
    return [(qs[i], qs[-1 - i]) for i in range(half)], middle


def _add_bands(
    fig: Any,
    times: np.ndarray,
    traj: np.ndarray,
    q: Sequence[float],
    colour: str,
    label: str,
) -> None:
    """Filled band per quantile pair (outermost first, lightest) plus the middle line."""
    pairs, middle = _pairs(q)
    levels = np.quantile(traj, [v for pair in pairs for v in pair], axis=0)
    for i, (lo, hi) in enumerate(pairs):
        low, high = levels[2 * i], levels[2 * i + 1]
        fig.add_scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([low, high[::-1]]),
            fill="toself",
            fillcolor=_rgba(colour, 0.15 + 0.2 * i),
            line={"width": 0},
            hoverinfo="skip",
            legendgroup=label,
            name=f"{label} {100 * lo:g}–{100 * hi:g}%",
        )
    if middle is not None:
        fig.add_scatter(
            x=times,
            y=np.quantile(traj, middle, axis=0),
            mode="lines",
            line={"color": colour, "width": 2},
            legendgroup=label,
            name=f"{label} median" if middle == 0.5 else f"{label} {100 * middle:g}%",
        )


def plot_ribbons(
    runs: Any,
    output: str,
    *,
    q: Sequence[float] = DEFAULT_QUANTILES,
    scenarios: Sequence[str] | None = None,
    targets: Any = None,
    fig: Any = None,
    title: str | None = None,
) -> Any:
    """Quantile ribbons of ``output`` over the draws, one colour per scenario.

    Quantiles are paired from the outside in (``0.025`` with ``0.975``, ``0.25``
    with ``0.75``), each pair a filled band, darker inward; an odd middle
    quantile is a line. Traces per scenario: ``len(q) // 2`` bands plus the
    middle line. ``scenarios`` defaults to all of them. Returns the figure.
    """
    names = list(runs.scenario_names if scenarios is None else scenarios)
    fig, created = _new_or(fig)
    palette = _palette()
    times = np.asarray(runs.times, dtype=float)
    for k, scen in enumerate(names):
        traj = np.asarray(runs.samples(scen, output), dtype=float)
        _add_bands(fig, times, traj, q, palette[k % len(palette)], scen)
    if targets is not None:
        add_targets(fig, targets, output=output)
    pairs, _ = _pairs(q)
    band = f" ({100 * pairs[0][0]:g}–{100 * pairs[0][1]:g}% band)" if pairs else ""
    _label(
        fig,
        created,
        title,
        f"{output}: quantiles over {runs.n} draws{band}",
        xaxis_title="time",
        yaxis_title=output,
    )
    return fig


def plot_scenarios(
    runs: Any,
    output: str,
    *,
    ref: str = "baseline",
    scenarios: Sequence[str] | None = None,
    q: Sequence[float] = DEFAULT_QUANTILES,
    relative: bool = False,
    fig: Any = None,
    title: str | None = None,
) -> Any:
    """Each scenario's difference from ``ref`` over time, as quantile ribbons.

    Draws are paired (see :meth:`PosteriorRuns.difference`), so the band is the
    uncertainty in the effect of the scenario. Below zero means the scenario
    has less of ``output`` than ``ref``. ``relative=True`` divides by the ref
    draw. ``scenarios`` defaults to every scenario except ``ref``.
    """
    if ref not in runs.scenario_names:
        raise KeyError(f"Unknown ref scenario {ref!r}. Known: {list(runs.scenario_names)}.")
    names = [s for s in (runs.scenario_names if scenarios is None else scenarios) if s != ref]
    if not names:
        raise ValueError(f"plot_scenarios needs a scenario other than the reference {ref!r}.")
    fig, created = _new_or(fig)
    palette = _palette()
    times = np.asarray(runs.times, dtype=float)
    for k, scen in enumerate(names):
        diff = runs.difference(scen, ref, output, relative=relative)
        _add_bands(fig, times, diff, q, palette[(k + 1) % len(palette)], f"{scen} − {ref}")
    fig.add_hline(y=0.0, line_dash="dash", line_color="grey")
    ylabel = f"({output} − {ref}) / {ref}" if relative else f"{output}: scenario − {ref}"
    _label(
        fig,
        created,
        title,
        f"{output}: difference from {ref!r} over {runs.n} paired draws",
        xaxis_title="time",
        yaxis_title=ylabel,
    )
    return fig


def plot_design(
    candidates: Candidates,
    *,
    sites: Sequence[str] | None = None,
    fig: Any = None,
    name: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> Any:
    """Log density against each parameter: one panel per site, shared y axis.

    ``candidates`` must be evaluated (:func:`evaluate`, or any stage that scores,
    such as ``OptimizeRun.candidates``); failed solves (``ok`` false) are left
    out. Pass the figure back as ``fig=`` with another batch (``design.best(16)``,
    an optimisation's end points) to overlay it in the same panels. Returns the
    figure.
    """
    if candidates.log_density is None or candidates.ok is None:
        raise ValueError("plot_design needs evaluated candidates (call wf.evaluate first).")
    names = list(candidates.sites if sites is None else sites)
    ok = np.asarray(candidates.ok, dtype=bool)
    ld = np.asarray(candidates.log_density, dtype=float)[ok]
    created = fig is None
    if created:
        _require_plotly()
        from plotly.subplots import make_subplots  # type: ignore[import-untyped]

        fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True)
    colour = color or _palette()[len(fig.data) // max(1, len(names)) % len(_palette())]
    label = name or f"{int(ok.sum())} points"
    for i, site in enumerate(names):
        fig.add_scatter(
            x=np.asarray(candidates.params[site], dtype=float)[ok],
            y=ld,
            mode="markers",
            marker={"color": colour, "size": 6, "opacity": 0.7},
            name=label,
            legendgroup=label,
            showlegend=i == 0,
            row=1,
            col=i + 1,
        )
        if created:
            fig.update_xaxes(title_text=site, row=1, col=i + 1)
    if created:
        fig.update_yaxes(title_text="log density", row=1, col=1)
        fig.update_layout(title=title or f"Log density against each parameter ({label})")
    elif title is not None:
        fig.update_layout(title=title)
    return fig


def plot_optimisation(run: Any, *, fig: Any = None, title: str | None = None) -> Any:
    """Best loss per start after every chunk of an :class:`OptimizeRun`.

    One line per start against cumulative optimiser steps. Failed-solve losses
    are gaps. The y axis is logarithmic when every loss is positive. Returns
    the figure.
    """
    trace = np.asarray(run.loss_trace, dtype=float)
    trace = np.where(np.isfinite(trace) & (trace < 0.5e30), trace, np.nan)
    steps = int(run.chunk_steps) * np.arange(1, trace.shape[0] + 1)
    fig, created = _new_or(fig)
    for i in range(trace.shape[1]):
        fig.add_scatter(x=steps, y=trace[:, i], mode="lines+markers", name=f"start {i}")
    converged = np.asarray(run.converged, dtype=bool)
    finite = trace[np.isfinite(trace)]
    if created and finite.size and bool(np.all(finite > 0.0)):
        fig.update_yaxes(type="log")
    _label(
        fig,
        created,
        title,
        f"Best loss per start ({int(converged.sum())} of {converged.size} converged)",
        xaxis_title="optimiser steps",
        yaxis_title="best loss (−log density)",
    )
    return fig


def _chain_samples(source: Any) -> dict[str, np.ndarray]:
    """``(chain, draw)`` arrays from an ``MCMCRun``, a numpyro ``MCMC`` or a mapping."""
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    elif hasattr(source, "extend") and hasattr(source, "samples"):
        raw = source.samples
    elif hasattr(source, "get_samples"):
        raw = source.get_samples(group_by_chain=True)
    else:
        raise TypeError(
            "plot_chains takes an MCMCRun, a numpyro MCMC, or a mapping of (chain, draw) "
            f"arrays; got {type(source).__name__}."
        )
    out: dict[str, np.ndarray] = {}
    for site, value in raw.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim < 2:
            raise ValueError(f"Site {site!r} must have chain and draw axes; got {arr.shape}.")
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        if flat.shape[2] == 1:
            out[str(site)] = flat[:, :, 0]
        else:
            for j in range(flat.shape[2]):
                out[f"{site}[{j}]"] = flat[:, :, j]
    return out


def plot_chains(
    source: Any,
    *,
    sites: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """Per-chain traces, one panel per site, with split R-hat and bulk ESS in each title.

    ``source`` is an :class:`MCMCRun` (chunk boundaries are dotted lines, and
    the stop reason is in the title), a numpyro ``MCMC``, or a mapping of
    ``(chain, draw)`` arrays. Diagnostics use ``numpyro.diagnostics`` over every
    draw shown. Returns the figure.
    """
    _require_plotly()
    from plotly.subplots import make_subplots

    from summer4.epi.calibration.workflow.mcmc import _site_diagnostics

    samples = _chain_samples(source)
    names = list(samples if sites is None else sites)
    diag = _site_diagnostics({k: samples[k] for k in names})
    fig = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{k}: R-hat {diag[k][0]:.3f}, bulk ESS {diag[k][1]:.0f}" for k in names],
    )
    palette = _palette()
    for r, site in enumerate(names, start=1):
        arr = samples[site]
        for c in range(arr.shape[0]):
            fig.add_scatter(
                y=arr[c],
                mode="lines",
                line={"color": palette[c % len(palette)], "width": 1},
                name=f"chain {c}",
                legendgroup=f"chain {c}",
                showlegend=r == 1,
                row=r,
                col=1,
            )
        fig.update_yaxes(title_text=site, row=r, col=1)
    fig.update_xaxes(title_text="draw", row=len(names), col=1)
    worst = max(diag[k][0] for k in names)
    default = f"Chains: worst R-hat {worst:.3f}"
    if hasattr(source, "extend") and hasattr(source, "mcmc"):
        chunk = int(source.mcmc.num_samples)
        for k in range(1, int(source.chunks)):
            fig.add_vline(x=k * chunk - 0.5, line_dash="dot", line_color="grey", row="all", col=1)
        default += f" after {source.chunks} chunks (stop: {source.reason})"
    fig.update_layout(title=title or default, height=max(320, 220 * len(names)))
    return fig


__all__ = [
    "DEFAULT_QUANTILES",
    "add_targets",
    "plot_chains",
    "plot_design",
    "plot_optimisation",
    "plot_ribbons",
    "plot_scenarios",
    "plot_spaghetti",
]
