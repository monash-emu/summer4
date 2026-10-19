"""Pick NUTS hyperparameters for a posterior from short pilot runs (step 47).

A new calibration needs a mass matrix, a tree depth, a target acceptance rate
and a warm-up length before its first real run, and the best choice changes as
the model does. This module runs short, identically seeded pilots of candidate
:class:`NutsSetting` objects from the caller's :class:`Candidates` (typically
multi-start optima) and scores each one:

- **sampling efficiency**: effective draws per gradient evaluation (hardware
  free) and per wall second (timed after compilation);
- **warm-up cost**: seconds and gradient evaluations of :func:`warmup_until`'s
  doubling rounds, reported separately, because a setting that samples twice
  as fast can take ten times longer to warm up.

:meth:`NutsTuning.table` ranks pilots by the cost of reaching a stated number
of effective draws, so the trade-off between warm-up and sampling speed is
decided by how many draws are wanted. A Laplace metric
(:func:`laplace_metric`, the inverse negative Hessian at the best point) can
start the metric adaptation from the right scales. :func:`tune_nuts` is the
one-call convenience built from these pieces.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from summer4.epi.calibration.posterior import Posterior
from summer4.epi.calibration.workflow.candidates import Candidates
from summer4.epi.calibration.workflow.mcmc import (
    MCMCRun,
    _require_numpyro,
    _require_pandas,
    sample_until,
)
from summer4.epi.calibration.workflow.warmup import (
    OnRound,
    WarmupCheck,
    WarmupRule,
    WarmupRun,
    warmup_until,
)

PILOT_FIELDS = ("num_steps", "accept_prob")
"""Extra fields every pilot records (``num_steps`` counts gradient evaluations)."""

PILOT_WARMUP_RULE = WarmupRule(max_divergence_frac=None)
"""The pilots' default warm-up check: :class:`WarmupRule` without its divergence criterion.

numpyro restarts step-size adaptation at 10x the step size at the end of every
adaptation window, so a few divergent warm-up transitions are expected even on
a Gaussian; each pilot counts the divergences that matter, those after warm-up.
"""


def _point(post: Posterior, z: Candidates | Mapping[str, Any]) -> dict[str, np.ndarray]:
    """One unconstrained site dict: the best evaluated candidate, else row 0, else ``z``."""
    if isinstance(z, Candidates):
        if len(z) == 0:
            raise ValueError("need at least one candidate.")
        row = z.best(1) if z.log_density is not None and z.ok is not None else z.take([0])
        point = {name: np.asarray(row.z[name])[0] for name in row.sites}
    else:
        point = {name: np.asarray(value) for name, value in z.items()}
    names = set(post.prior_names())
    if set(point) != names:
        raise ValueError(
            f"point sites {sorted(point)} do not match the posterior's {sorted(names)}."
        )
    return point


def _flat_density(post: Posterior, point: Mapping[str, Any]) -> tuple[Any, Callable[[Any], Any]]:
    """``(vector, f)`` with ``f(vector) == post.log_density(point)``, in numpyro's site order."""
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    flat, unravel = ravel_pytree(  # type: ignore[no-untyped-call]
        {k: jnp.asarray(v) for k, v in point.items()}
    )

    def f(vec: Any) -> Any:
        return post.log_density(unravel(vec))

    return flat, f


@dataclass(frozen=True)
class LaplaceMetric:
    """Laplace approximation at a point, as a NUTS starting metric.

    ``covariance`` is the inverse negative Hessian of the log density in
    unconstrained space, over the sites in sorted-name order with each site
    raveled (``jax.flatten_util.ravel_pytree`` of the site dict). That is the
    order of numpyro's mass-matrix block ``tuple(sorted(sites))``, so plate and
    vector sites work. ``eigenvalues`` are those of the negative Hessian before
    clipping; ``clipped`` counts the directions whose variance was clipped
    (a flat or indefinite direction, or one against a prior bound).
    """

    sites: tuple[str, ...]
    shapes: Mapping[str, tuple[int, ...]]
    mode: Mapping[str, np.ndarray]
    covariance: np.ndarray
    eigenvalues: np.ndarray
    clipped: int

    @property
    def std(self) -> np.ndarray:
        """Marginal standard deviations, in :attr:`element_names` order."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def element_names(self) -> tuple[str, ...]:
        """One label per row of :attr:`covariance`: ``"mu"``, ``"z[0]"``, ..."""
        names: list[str] = []
        for site in self.sites:
            shape = tuple(self.shapes[site])
            if shape == ():
                names.append(site)
            else:
                names.extend(
                    f"{site}[{','.join(str(i) for i in idx)}]" for idx in np.ndindex(*shape)
                )
        return tuple(names)

    def inverse_mass_matrix(self, dense: bool = True) -> dict[tuple[str, ...], np.ndarray]:
        """``NUTS(inverse_mass_matrix=...)`` for a dense or a diagonal kernel."""
        matrix = self.covariance if dense else np.diag(self.covariance).copy()
        return {tuple(self.sites): matrix}


def laplace_metric(
    post: Posterior,
    z: Candidates | Mapping[str, Any],
    *,
    min_var: float = 1e-4,
    max_var: float = 25.0,
) -> LaplaceMetric:
    """Inverse negative Hessian of ``post.log_density`` at ``z``, eigenvalues clipped.

    ``z`` is one unconstrained site dict (no batch axis), or :class:`Candidates`,
    in which case the best evaluated row is used (row 0 if unevaluated). The
    Hessian is exact (``jax.hessian``, forward over reverse) for the
    discretised model; with a loose adaptive ODE tolerance it inherits the
    solver's noise, so pass a posterior with a tight solver if that matters.
    Variances ``1 / eigenvalue`` are clipped to ``[min_var, max_var]``, and a
    non-positive eigenvalue gets ``max_var``.
    """
    import jax

    point = _point(post, z)
    flat, f = _flat_density(post, point)
    hessian = np.asarray(jax.jit(jax.hessian(f))(flat), dtype=float)
    if not np.all(np.isfinite(hessian)):
        raise ValueError("the log-density Hessian at this point is not finite.")
    neg = -0.5 * (hessian + hessian.T)
    vals, vecs = np.linalg.eigh(neg)
    with np.errstate(divide="ignore"):
        raw = np.where(vals > 0, 1.0 / np.where(vals > 0, vals, 1.0), np.inf)
    var = np.clip(raw, float(min_var), float(max_var))
    clipped = int(np.sum((vals <= 0) | (raw < float(min_var)) | (raw > float(max_var))))
    sites = tuple(sorted(point))
    return LaplaceMetric(
        sites=sites,
        shapes={s: tuple(np.shape(point[s])) for s in sites},
        mode={s: point[s] for s in sites},
        covariance=(vecs * var) @ vecs.T,
        eigenvalues=vals,
        clipped=clipped,
    )


def time_gradient(
    post: Posterior, z: Candidates | Mapping[str, Any], *, repeats: int = 20
) -> float:
    """Median seconds of one log density plus gradient at ``z``, jitted and warmed.

    Each timed call ends in ``jax.block_until_ready``: JAX dispatch is
    asynchronous, so an unblocked timer would measure only the dispatch.
    """
    import jax

    if int(repeats) < 1:
        raise ValueError(f"time_gradient needs repeats >= 1, got {repeats}.")
    flat, f = _flat_density(post, _point(post, z))
    value_and_grad = jax.jit(jax.value_and_grad(f))
    jax.block_until_ready(value_and_grad(flat))  # type: ignore[no-untyped-call]
    times = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        jax.block_until_ready(value_and_grad(flat))  # type: ignore[no-untyped-call]
        times.append(time.perf_counter() - start)
    return float(np.median(times))


@dataclass(frozen=True)
class NutsSetting:
    """One candidate NUTS configuration.

    ``kwargs`` go to ``numpyro.infer.NUTS`` unchanged, so anything NUTS accepts
    is a candidate (``dense_mass``, ``max_tree_depth``, ``target_accept_prob``,
    ``step_size``, ``find_heuristic_step_size``, ``adapt_mass_matrix``, ...).
    ``laplace=True`` starts the metric from a :class:`LaplaceMetric` (dense or
    diagonal to match ``dense_mass``). ``num_warmup`` is the first warm-up
    round's length; :func:`warmup_until` doubles it until the warm-up passes.
    """

    label: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    laplace: bool = False
    num_warmup: int = 100

    @property
    def dense(self) -> bool:
        """Whether the kernel's mass matrix has a dense block."""
        return bool(self.kwargs.get("dense_mass", False))

    @property
    def max_tree_depth(self) -> int:
        """Sampling-phase maximum tree depth (numpyro's default is 10)."""
        depth = self.kwargs.get("max_tree_depth", 10)
        return int(depth[1] if isinstance(depth, tuple) else depth)

    def initial_metric(self, metric: LaplaceMetric | None) -> Any:
        """The starting ``inverse_mass_matrix`` this setting asks for (``None``: numpyro's)."""
        if not self.laplace:
            return None
        if metric is None:
            raise ValueError(f"setting {self.label!r} needs a LaplaceMetric; got None.")
        return metric.inverse_mass_matrix(dense=self.dense)

    def kernel(self, model: Callable[..., Any], inverse_mass_matrix: Any = None) -> Any:
        """``NUTS(model, **kwargs)``, with ``inverse_mass_matrix`` if given."""
        _require_numpyro()
        from numpyro.infer import NUTS  # type: ignore[import-untyped]

        kwargs = dict(self.kwargs)
        if inverse_mass_matrix is not None:
            kwargs["inverse_mass_matrix"] = inverse_mass_matrix
        return NUTS(model, **kwargs)

    def make_mcmc(
        self,
        model: Callable[..., Any],
        *,
        num_samples: int,
        num_chains: int,
        chain_method: str = "vectorized",
        metric: LaplaceMetric | None = None,
        progress_bar: bool = False,
    ) -> Callable[..., Any]:
        """A ``(num_warmup, inverse_mass_matrix=None) -> MCMC`` factory for :func:`warmup_until`.

        Without an explicit ``inverse_mass_matrix`` the kernel starts from
        :meth:`initial_metric` of ``metric``.
        """
        _require_numpyro()
        from numpyro.infer import MCMC

        default = self.initial_metric(metric)

        def make(num_warmup: int, inverse_mass_matrix: Any = None) -> Any:
            start = default if inverse_mass_matrix is None else inverse_mass_matrix
            return MCMC(
                self.kernel(model, start),
                num_warmup=int(num_warmup),
                num_samples=int(num_samples),
                num_chains=int(num_chains),
                chain_method=chain_method,
                progress_bar=progress_bar,
            )

        return make


def metric_screen(num_warmup: int = 100) -> tuple[NutsSetting, ...]:
    """Stage A: how to start and adapt the mass matrix, at numpyro's other defaults.

    ``diag``, ``diag+laplace``, ``dense``, ``dense+laplace`` and ``laplace fixed``
    (dense Laplace metric, not adapted: warm-up tunes only the step size).
    """
    n = int(num_warmup)
    dense = {"dense_mass": True}
    return (
        NutsSetting("diag", {}, num_warmup=n),
        NutsSetting("diag+laplace", {}, laplace=True, num_warmup=n),
        NutsSetting("dense", dense, num_warmup=n),
        NutsSetting("dense+laplace", dense, laplace=True, num_warmup=n),
        NutsSetting(
            "laplace fixed", {**dense, "adapt_mass_matrix": False}, laplace=True, num_warmup=n
        ),
    )


TRAJECTORY_VARIANTS: tuple[tuple[str, Mapping[str, Any]], ...] = (
    ("depth 6", {"max_tree_depth": 6}),
    ("depth 8", {"max_tree_depth": 8}),
    ("target 0.9", {"target_accept_prob": 0.9}),
    ("target 0.95", {"target_accept_prob": 0.95}),
    ("heuristic step", {"find_heuristic_step_size": True}),
)
"""Stage B variations: tree depth, target acceptance, initial step-size search."""


def trajectory_screen(
    base: NutsSetting,
    variants: Sequence[tuple[str, Mapping[str, Any]]] = TRAJECTORY_VARIANTS,
) -> tuple[NutsSetting, ...]:
    """Stage B: ``base`` with one trajectory setting changed at a time.

    Each variant's kwargs are merged over ``base.kwargs``; a variant that changes
    nothing is left out.
    """
    out: list[NutsSetting] = []
    for name, extra in variants:
        merged = {**base.kwargs, **extra}
        if merged == dict(base.kwargs):
            continue
        out.append(replace(base, label=f"{base.label}, {name}", kwargs=merged))
    return tuple(out)


@dataclass
class Pilot:
    """One pilot: its setting, warm-up record, sampled run and summary row.

    ``run`` is a live :class:`MCMCRun`: ``pilot.run.extend(stop)`` keeps
    sampling the warmed-up chains with no further warm-up.
    """

    setting: NutsSetting
    warmup: WarmupRun | None
    run: MCMCRun | None
    row: dict[str, Any]


def _pilot_row(
    setting: NutsSetting,
    warm: WarmupRun,
    run: MCMCRun,
    *,
    num_chains: int,
    pilot_samples: int,
    grad_seconds: float | None,
) -> dict[str, Any]:
    warm_rows = warm.progress
    progress = run.progress
    last = progress.iloc[-1]
    num_steps = np.asarray(run.extra["num_steps"])
    accept = np.asarray(run.extra["accept_prob"])
    draws = int(num_steps.size)
    grad_evals = int(num_steps.sum())
    # Chunk 1 includes compiling the sampler; time chunk 2 only.
    chunk_seconds = float(progress["seconds"].iloc[1] - progress["seconds"].iloc[0])
    draws_per_second = num_chains * pilot_samples / chunk_seconds
    ess = float(last["ess_bulk_element_min"])
    step = np.atleast_1d(np.asarray(run.mcmc.last_state.adapt_state.step_size, dtype=float))
    ess_per_grad = ess / grad_evals
    return {
        "label": setting.label,
        "warmup_rounds": int(warm.rounds),
        "num_warmup": int(warm.num_warmup),
        "warmup_passed": bool(warm.converged),
        "warmup_failed": warm_rows["failed"].iloc[-1],
        "warmup_seconds": float(warm_rows["seconds"].iloc[-1]),
        "warmup_grad_evals": int(warm_rows["grad_evals"].sum()),
        "step_size": float(step.mean()),
        "accept_mean": float(accept.mean()),
        "num_steps_mean": float(num_steps.mean()),
        "num_steps_max": int(num_steps.max()),
        "treedepth_frac": float(np.mean(num_steps >= 2**setting.max_tree_depth - 1)),
        "divergence_frac": float(np.mean(run.diverging)) if run.diverging is not None else 0.0,
        "rhat_max": float(last["rhat_element_max"]),
        "ess_min": ess,
        "draws": draws,
        "grad_evals": grad_evals,
        "ess_per_grad": ess_per_grad,
        "draws_per_second": draws_per_second,
        "ess_per_second": ess / draws * draws_per_second,
        "grad_seconds": grad_seconds,
        "predicted_ess_per_second": (
            None if grad_seconds is None else ess_per_grad / float(grad_seconds)
        ),
    }


def pilot(
    post: Posterior,
    init: Candidates,
    setting: NutsSetting,
    *,
    metric: LaplaceMetric | None = None,
    grad_seconds: float | None = None,
    num_chains: int = 4,
    pilot_samples: int = 200,
    chain_method: str = "vectorized",
    warmup_check: WarmupCheck | None = None,
    max_warmup: int = 2000,
    carry_metric: bool = True,
    jitter: float = 0.05,
    seed: int = 0,
    on_round: OnRound | None = None,
) -> Pilot:
    """Warm up and sample one setting briefly, and summarise it. Exactly::

        make_mcmc = setting.make_mcmc(post.numpyro_model(), num_samples=pilot_samples,
                                      num_chains=num_chains, chain_method=chain_method,
                                      metric=metric)  # starts from the Laplace metric if asked
        warm = warmup_until(make_mcmc, setting.num_warmup, warmup_check,
                            init_params=init.init_params(num_chains, jitter=jitter, seed=seed),
                            carry_metric=carry_metric, max_warmup=max_warmup, seed=seed,
                            on_round=on_round, warn=False)
        run = sample_until(warm.mcmc, <stop after 2 chunks>, seed=seed, warn=False,
                           extra_fields=("num_steps", "accept_prob"))

    The first chunk compiles the sampler, so speed is timed on the second only;
    ESS and R-hat (worst element of any site) use both. ``warmup_check``
    defaults to :data:`PILOT_WARMUP_RULE`. ``grad_seconds`` (from
    :func:`time_gradient`) adds ``predicted_ess_per_second``: the rate if every
    second were spent in the model.
    """
    samples = int(pilot_samples)
    if samples < 4:
        raise ValueError(f"pilot needs pilot_samples >= 4, got {samples}.")
    make_mcmc = setting.make_mcmc(
        post.numpyro_model(),
        num_samples=samples,
        num_chains=int(num_chains),
        chain_method=chain_method,
        metric=metric,
    )
    warm = warmup_until(
        make_mcmc,
        int(setting.num_warmup),
        PILOT_WARMUP_RULE if warmup_check is None else warmup_check,
        init_params=init.init_params(int(num_chains), jitter=jitter, seed=seed),
        carry_metric=carry_metric,
        max_warmup=max_warmup,
        seed=seed,
        on_round=on_round,
        warn=False,
    )

    def two_chunks(row: Mapping[str, Any]) -> tuple[bool, str] | None:
        return (True, "pilot") if int(row["samples"]) >= 2 * samples else None

    run = sample_until(warm.mcmc, two_chunks, seed=seed, warn=False, extra_fields=PILOT_FIELDS)
    row = _pilot_row(
        setting,
        warm,
        run,
        num_chains=int(num_chains),
        pilot_samples=samples,
        grad_seconds=grad_seconds,
    )
    return Pilot(setting=setting, warmup=warm, run=run, row=row)


class NutsTuning:
    """Pilots of NUTS settings on one posterior, ranked by cost to a target ESS.

    Built once per posterior and starting points; :meth:`run` adds pilots and
    can be called again (a later stage around the current best, a hand-written
    setting). Every pilot shares the seed, chains and starting points. On
    construction the Laplace metric (``laplace=True``, unless ``metric`` is
    given) and the gradient cost are computed at the best point of ``init``.
    """

    def __init__(
        self,
        post: Posterior,
        init: Candidates,
        *,
        laplace: bool = True,
        metric: LaplaceMetric | None = None,
        num_chains: int = 4,
        pilot_samples: int = 200,
        chain_method: str = "vectorized",
        warmup_check: WarmupCheck | None = None,
        max_warmup: int = 2000,
        jitter: float = 0.05,
        seed: int = 0,
        time_repeats: int = 20,
        max_divergence_frac: float = 0.01,
        max_rhat: float = 1.1,
        on_pilot: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.post = post
        self.init = init
        start = time.perf_counter()
        self.metric = metric if metric is not None or not laplace else laplace_metric(post, init)
        self.metric_seconds = time.perf_counter() - start
        self.grad_seconds = time_gradient(post, init, repeats=time_repeats)
        self.num_chains = int(num_chains)
        self.pilot_samples = int(pilot_samples)
        self.chain_method = chain_method
        self.warmup_check = warmup_check
        self.max_warmup = int(max_warmup)
        self.jitter = float(jitter)
        self.seed = int(seed)
        self.max_divergence_frac = float(max_divergence_frac)
        self.max_rhat = float(max_rhat)
        self.on_pilot = on_pilot
        self.pilots: list[Pilot] = []

    def run(self, settings: Sequence[NutsSetting]) -> NutsTuning:
        """Pilot each setting not already piloted (by label). Mutates and returns ``self``.

        A ``laplace`` setting with no metric is recorded as skipped, not run.
        """
        done = {p.setting.label for p in self.pilots}
        for setting in settings:
            if setting.label in done:
                continue
            done.add(setting.label)
            if setting.laplace and self.metric is None:
                result = Pilot(setting, None, None, {"label": setting.label, "skipped": True})
            else:
                result = pilot(
                    self.post,
                    self.init,
                    setting,
                    metric=self.metric,
                    grad_seconds=self.grad_seconds,
                    num_chains=self.num_chains,
                    pilot_samples=self.pilot_samples,
                    chain_method=self.chain_method,
                    warmup_check=self.warmup_check,
                    max_warmup=self.max_warmup,
                    jitter=self.jitter,
                    seed=self.seed,
                )
            self.pilots.append(result)
            if self.on_pilot is not None:
                self.on_pilot(dict(result.row))
        return self

    def _why(self, row: Mapping[str, Any]) -> str | None:
        if row.get("skipped"):
            return "skipped: no Laplace metric"
        reasons = []
        if not row["warmup_passed"]:
            reasons.append(f"warm-up failed ({row['warmup_failed']})")
        if row["divergence_frac"] > self.max_divergence_frac:
            reasons.append("divergences")
        if not row["rhat_max"] <= self.max_rhat:
            reasons.append("R-hat")
        return "; ".join(reasons) or None

    def table(self, target_ess: float = 1000.0, by: str = "cost_seconds") -> Any:
        """Pilots ranked by cost to ``target_ess`` effective draws, usable first.

        ``cost_seconds = warmup_seconds + target_ess / ess_per_second`` and
        ``cost_grads = warmup_grad_evals + target_ess / ess_per_grad`` (the
        compile- and hardware-free version). ``why`` says why a pilot is not
        usable: its warm-up failed, it diverged more than
        ``max_divergence_frac``, or its worst R-hat exceeds ``max_rhat``.
        ``by`` names the column to rank on: ``"cost_seconds"`` for this machine,
        or ``"cost_grads"`` when wall time is dominated by compilation (a cheap
        model) or the real run will be on other hardware.
        """
        if by not in ("cost_seconds", "cost_grads"):
            raise ValueError(f"by must be 'cost_seconds' or 'cost_grads', got {by!r}.")
        pd = _require_pandas()
        if not self.pilots:
            raise ValueError("no pilots yet; call run(settings) first.")
        rows = []
        for p in self.pilots:
            row = dict(p.row)
            row["why"] = self._why(row)
            if row.get("skipped"):
                row["cost_seconds"] = row["cost_grads"] = float("nan")
            else:
                row["cost_seconds"] = row["warmup_seconds"] + target_ess / row["ess_per_second"]
                row["cost_grads"] = row["warmup_grad_evals"] + target_ess / row["ess_per_grad"]
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["usable"] = frame["why"].isna()
        frame = frame.sort_values(["usable", by], ascending=[False, True])
        frame.insert(0, "rank", range(1, len(frame) + 1))
        return frame.set_index("label")

    def best(self, target_ess: float = 1000.0, by: str = "cost_seconds") -> Pilot:
        """The usable pilot that reaches ``target_ess`` effective draws most cheaply."""
        frame = self.table(target_ess, by)
        usable = frame[frame["usable"]]
        if usable.empty:
            raise ValueError("no usable pilot; see table()['why'].")
        label = usable.index[0]
        return next(p for p in self.pilots if p.setting.label == label)

    def recommended(self, target_ess: float = 1000.0, by: str = "cost_seconds") -> NutsSetting:
        """The best pilot's setting, with ``num_warmup`` at the length its warm-up passed."""
        best = self.best(target_ess, by)
        assert best.warmup is not None
        return replace(best.setting, num_warmup=int(best.warmup.num_warmup))

    def make_mcmc(
        self,
        *,
        num_samples: int,
        num_chains: int | None = None,
        target_ess: float = 1000.0,
        by: str = "cost_seconds",
        progress_bar: bool = False,
    ) -> Callable[..., Any]:
        """The recommended ``num_warmup -> MCMC`` factory, Laplace metric included if used.

        Hand it to :func:`warmup_until` with ``recommended().num_warmup``, or
        call it with that length and pass the ``MCMC`` to :func:`sample_until`.
        """
        return self.recommended(target_ess, by).make_mcmc(
            self.post.numpyro_model(),
            num_samples=num_samples,
            num_chains=self.num_chains if num_chains is None else int(num_chains),
            chain_method=self.chain_method,
            metric=self.metric,
            progress_bar=progress_bar,
        )


def tune_nuts(
    post: Posterior,
    init: Candidates,
    *,
    target_ess: float = 1000.0,
    num_warmup: int = 100,
    laplace: bool = True,
    num_chains: int = 4,
    pilot_samples: int = 200,
    chain_method: str = "vectorized",
    warmup_check: WarmupCheck | None = None,
    max_warmup: int = 2000,
    seed: int = 0,
    on_pilot: Callable[[Mapping[str, Any]], None] | None = None,
) -> NutsTuning:
    """Two-stage NUTS tuning in one call. Exactly equivalent to::

        tuning = NutsTuning(post, init, laplace=laplace, num_chains=num_chains,
                            pilot_samples=pilot_samples, chain_method=chain_method,
                            warmup_check=warmup_check, max_warmup=max_warmup, seed=seed,
                            on_pilot=on_pilot)
        tuning.run(metric_screen(num_warmup))
        tuning.run(trajectory_screen(tuning.best(target_ess).setting))
        return tuning

    Then ``tuning.table(target_ess)`` ranks the pilots and
    ``tuning.make_mcmc(num_samples=...)`` is the recommended factory.
    """
    tuning = NutsTuning(
        post,
        init,
        laplace=laplace,
        num_chains=num_chains,
        pilot_samples=pilot_samples,
        chain_method=chain_method,
        warmup_check=warmup_check,
        max_warmup=max_warmup,
        seed=seed,
        on_pilot=on_pilot,
    )
    tuning.run(metric_screen(num_warmup))
    tuning.run(trajectory_screen(tuning.best(target_ess).setting))
    return tuning


__all__ = [
    "LaplaceMetric",
    "NutsSetting",
    "NutsTuning",
    "PILOT_FIELDS",
    "PILOT_WARMUP_RULE",
    "Pilot",
    "TRAJECTORY_VARIANTS",
    "laplace_metric",
    "metric_screen",
    "pilot",
    "time_gradient",
    "trajectory_screen",
    "tune_nuts",
]
