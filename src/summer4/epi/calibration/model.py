"""Bayesian model assembly: numpyro sampling and MAP over a compiled flow model.

Run-start transforms belong on ``FlowModel.compile(prepare_fn=...)``, not a
second ``preprocess`` hook on this class (see the deleted
``futureplans/wp10-preprocess-is-prepare-fn.md`` note).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from summer4.epi.calibration.likelihoods import prior_sites
from summer4.epi.calibration.priors import Prior
from summer4.flows.compiled import CompiledModel
from summer4.flows.initial import InitialPopulation
from summer4.results.outputset import OutputSet
from summer4.results.plan import SavePlan
from summer4.results.targets import TargetSet

SampleKind = Literal["nuts", "aies", "ess", "sa"]


def _one_chunk(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Stop callable for :meth:`BayesianModel.sample`: stop after the first chunk."""
    del row
    return True, "num_samples"


def _require_numpyro() -> Any:
    try:
        import numpyro  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "BayesianModel requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return numpyro


def _require_optax() -> Any:
    try:
        import optax  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "find_map requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return optax


def _require_arviz() -> Any:
    try:
        import arviz  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "sample() requires arviz (calibration extra): pip install summer4[calibration]"
        ) from exc
    return arviz


def _params_as_dict(params: Any) -> dict[str, Any]:
    """Flatten fixed params to a writable mapping (NamedTuple → ``_asdict``)."""
    if isinstance(params, Mapping):
        return dict(params)
    asdict = getattr(params, "_asdict", None)
    if callable(asdict):
        return dict(asdict())
    raise TypeError(
        "BayesianModel fixed_params must be a mapping or NamedTuple, "
        f"got {type(params).__name__}."
    )


def _collect_priors(
    priors: Sequence[Prior],
    targets: TargetSet,
) -> tuple[Prior, ...]:
    """Priors plus hierarchical scales nested in target likelihoods, unique by name."""
    by_name: dict[str, Prior] = {}
    for prior in priors:
        if prior.name in by_name and by_name[prior.name] is not prior:
            raise ValueError(f"Duplicate prior name {prior.name!r}.")
        by_name[prior.name] = prior
    for target in targets.targets:
        if target.likelihood is None:
            continue
        for nested in prior_sites(target.likelihood):
            existing = by_name.get(nested.name)
            if existing is None:
                by_name[nested.name] = nested
            elif existing is not nested:
                raise ValueError(
                    f"Prior {nested.name!r} appears both in priors= and on "
                    f"target {target.key!r} with different objects."
                )
    return tuple(by_name.values())


class BayesianModel:
    """Numpyro-backed calibration of a :class:`~summer4.flows.compiled.CompiledModel`.

    Samples ``priors`` (and any prior-valued likelihood scales), merges them into
    ``fixed_params``, runs the model, optionally evaluates an
    :class:`~summer4.results.outputset.OutputSet`, then scores
    :meth:`TargetSet.log_likelihood`. A failed solve
    (``result.solver.ok`` is false) contributes a large finite negative
    factor (``-1e30``) — numpyro rejects non-finite ``factor`` values — so a
    diverged trajectory cannot look like a good fit.

    Run-start work (yearly mixing stacks, interpolator knots) goes on
    ``FlowModel.compile(prepare_fn=...)``. Passing ``preprocess=`` raises.
    """

    def __init__(
        self,
        compiled: CompiledModel,
        fixed_params: Any,
        priors: Sequence[Prior],
        targets: TargetSet,
        *,
        outputs: OutputSet | None = None,
        init: InitialPopulation | None = None,
        y0: Any | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        preprocess: Any = None,
    ) -> None:
        if preprocess is not None:
            raise TypeError(
                "BayesianModel does not take preprocess=; put run-start transforms on "
                "FlowModel.compile(prepare_fn=...) so they share the CompiledModel "
                "digest and hoist table."
            )
        if not isinstance(compiled, CompiledModel):
            raise TypeError(f"compiled must be a CompiledModel, got {type(compiled).__name__}.")
        self.compiled = compiled
        self.fixed_params = fixed_params
        self.priors = tuple(priors)
        self.targets = targets
        self.outputs = outputs
        self.init = init
        self.y0 = y0
        self.run_kwargs = dict(run_kwargs or {})
        self._sites = _collect_priors(self.priors, targets)
        self._save_plan = self._build_save_plan()
        self._potential_fn: Any | None = None
        self._postprocess_fn: Any | None = None
        self._init_z: Any | None = None

    def _build_save_plan(self) -> SavePlan:
        plan = SavePlan()
        plan = self.targets.plan(plan)
        if self.outputs is not None:
            plan = self.outputs.plan(plan)
        return plan

    def merge_params(self, draws: Mapping[str, Any]) -> dict[str, Any]:
        """Merge prior draws into a copy of ``fixed_params`` (as a dict)."""
        params = _params_as_dict(self.fixed_params)
        params.update(dict(draws))
        return params

    def _resolve_y0(self, params: Any) -> Any | None:
        if self.y0 is not None:
            return self.y0
        if self.init is not None:
            prepared = self.compiled.prepare(params)
            return self.init.compile(self.compiled.pmap).evaluate(prepared.params)
        return None

    def _run(self, params: Any) -> Any:
        kwargs = dict(self.run_kwargs)
        if "save" in kwargs:
            raise ValueError(
                "run_kwargs must not include save=; BayesianModel builds the save "
                "plan from targets and outputs."
            )
        y0 = self._resolve_y0(params)
        return self.compiled.run(params, y0, save=self._save_plan, **kwargs)

    def numpyro_model(self) -> Any:
        """Return the numpyro model callable (priors → run → factor)."""
        numpyro = _require_numpyro()
        import jax.numpy as jnp

        sites = self._sites
        targets = self.targets
        outputs = self.outputs

        def model() -> None:
            draws: dict[str, Any] = {}
            for prior in sites:
                draws[prior.name] = numpyro.sample(prior.name, prior.to_numpyro())
            params = self.merge_params(draws)
            result = self._run(params)
            scored = result
            if outputs is not None:
                scored = outputs.evaluate(result, params)
            ll = targets.log_likelihood(scored, params)
            ok = True if result.solver is None else result.solver.ok
            # numpyro.factor / Unit rejects non-finite log_factor; a failed solve
            # (and any NaN likelihood) becomes a large finite penalty instead.
            fail = jnp.asarray(-1.0e30, dtype=jnp.result_type(ll))
            ll = jnp.where(ok, ll, fail)
            ll = jnp.nan_to_num(ll, nan=-1.0e30, posinf=-1.0e30, neginf=-1.0e30)
            numpyro.factor("likelihood", ll)

        return model

    def _ensure_potential(self, rng_key: Any | None = None) -> tuple[Any, Any, Any]:
        """Lazily build unconstrained potential / postprocess via ``initialize_model``."""
        from jax import random
        from numpyro.infer.util import initialize_model  # type: ignore[import-untyped]

        _require_numpyro()
        if self._potential_fn is not None:
            assert self._postprocess_fn is not None and self._init_z is not None
            return self._potential_fn, self._postprocess_fn, self._init_z

        key = random.PRNGKey(0) if rng_key is None else rng_key
        params_info, potential_fn, postprocess_fn, _trace = initialize_model(
            key, self.numpyro_model()
        )
        self._potential_fn = potential_fn
        self._postprocess_fn = postprocess_fn
        self._init_z = params_info.z
        return potential_fn, postprocess_fn, params_info.z

    @property
    def potential_fn(self) -> Any:
        """Unconstrained potential ``-log_density`` (for optax / multi-start).

        Traceable under ``jax.jit``. Built lazily via numpyro
        ``initialize_model`` on first access.
        """
        potential, _post, _z = self._ensure_potential()
        return potential

    def log_density(self, unconstrained: Mapping[str, Any]) -> Any:
        """Joint log density at an **unconstrained** site dict (NUTS space).

        Traceable under ``jax.jit``. Equals ``-potential_fn(unconstrained)``.
        """
        return -self.potential_fn(dict(unconstrained))

    def _site_transforms(self) -> dict[str, Any]:
        """Per-site bijectors from unconstrained space onto prior support."""
        from numpyro.distributions.transforms import biject_to  # type: ignore[import-untyped]

        _require_numpyro()
        return {p.name: biject_to(p.to_numpyro().support) for p in self._sites}

    def constrain(self, z: Mapping[str, Any]) -> dict[str, Any]:
        """Map unconstrained site values to constrained parameters.

        Elementwise on array leaves, so a leading batch axis is fine under
        ``jax.vmap`` / ``jax.lax.map``.
        """
        import jax.numpy as jnp

        transforms = self._site_transforms()
        out: dict[str, Any] = {}
        for name, transform in transforms.items():
            if name not in z:
                raise KeyError(f"constrain missing site {name!r}.")
            out[name] = transform(jnp.asarray(z[name]))
        return out

    def unconstrain(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Map constrained parameters to unconstrained site values.

        Elementwise on array leaves (batch axis allowed).
        """
        import jax.numpy as jnp

        transforms = self._site_transforms()
        out: dict[str, Any] = {}
        for name, transform in transforms.items():
            if name not in params:
                raise KeyError(f"unconstrain missing site {name!r}.")
            out[name] = transform.inv(jnp.asarray(params[name]))
        return out

    def init_point(self, seed: int = 0) -> dict[str, Any]:
        """numpyro's initial point for this model: an **unconstrained** site dict.

        Drawn by ``numpyro.infer.util.initialize_model`` (``init_to_uniform``)
        with ``PRNGKey(seed)``; the same space as :meth:`log_density`.
        """
        from jax import random
        from numpyro.infer.util import initialize_model

        _require_numpyro()
        params_info = initialize_model(random.PRNGKey(int(seed)), self.numpyro_model())[0]
        return dict(params_info.z)

    def find_map(
        self,
        init: Mapping[str, Any] | None = None,
        *,
        steps: int = 500,
        optimizer: Any | None = None,
        seed: int = 0,
    ) -> Any:
        """Maximise the joint density from one start; return the :class:`OptimizeRun`.

        ``run.best_params`` is the MAP estimate as a dict of constrained values;
        ``run.extend(steps)`` keeps optimising. ``optimizer`` is any optax
        ``GradientTransformation`` (default ``optax.adam(0.05)``); ``init`` is an
        unconstrained site dict (default :meth:`init_point`). Exactly::

            z0 = model.init_point(seed) if init is None else init
            start = wf.Candidates.from_z(model, {k: jnp.asarray(v)[None] for k, v in z0.items()})
            return wf.optimize(model, start, method=wf.Optax(optimizer or optax.adam(0.05)),
                               max_steps=steps, chunk_steps=min(50, steps), seed=seed)
        """
        import jax.numpy as jnp

        from summer4.epi.calibration.workflow.candidates import Candidates
        from summer4.epi.calibration.workflow.optimize import Optax, optimize

        optax = _require_optax()
        z0 = self.init_point(seed) if init is None else dict(init)
        start = Candidates.from_z(self, {k: jnp.asarray(v)[None] for k, v in z0.items()})
        steps = max(1, int(steps))
        return optimize(
            self,
            start,
            method=Optax(optax.adam(0.05) if optimizer is None else optimizer),
            max_steps=steps,
            chunk_steps=min(50, steps),
            seed=seed,
        )

    def sample(
        self,
        kernel: Any = "nuts",
        *,
        init: Any = None,
        stop: Any = None,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 1,
        seed: int = 0,
        chain_method: str = "vectorized",
        progress_bar: bool = False,
        **kernel_kwargs: Any,
    ) -> Any:
        """Run MCMC on this model and return the resumable :class:`MCMCRun`.

        ``kernel`` is a numpyro kernel you built from :meth:`numpyro_model`
        (``NUTS(model.numpyro_model(), target_accept_prob=0.95)``), or a name —
        ``"nuts"``, ``"aies"``, ``"ess"``, ``"sa"`` — built with
        ``**kernel_kwargs``. ``init`` seeds the chains from :class:`Candidates`.
        With ``stop=None`` the run takes ``num_samples`` draws per chain in one
        chunk; with a :class:`StopRule` it samples in chunks of ``num_samples``
        until the rule decides. ``run.idata`` is the ``arviz.InferenceData``.
        Exactly::

            return wf.run_mcmc(model, init, make_kernel=lambda: kernel_object,
                               num_chains=num_chains, num_warmup=num_warmup,
                               chunk_samples=num_samples,
                               stop=stop or (lambda row: (True, "num_samples")),
                               seed=seed, chain_method=chain_method,
                               progress_bar=progress_bar)

        where ``kernel_object`` is ``kernel`` or the named kernel class applied to
        ``model.numpyro_model()``.
        """
        from numpyro.infer import AIES, ESS, NUTS, SA  # type: ignore[import-untyped]

        from summer4.epi.calibration.workflow.mcmc import run_mcmc

        _require_numpyro()
        if isinstance(kernel, str):
            key = kernel.strip().lower()
            kernels: dict[str, Any] = {"nuts": NUTS, "aies": AIES, "ess": ESS, "sa": SA}
            if key not in kernels:
                raise ValueError(
                    f"Unknown kernel {kernel!r}; expected a numpyro kernel or one of "
                    f"{sorted(kernels)}."
                )
            if key in ("aies", "ess") and int(num_chains) < 2:
                raise ValueError(f"{key} requires num_chains >= 2 (ensemble walkers).")
            kernel_object = kernels[key](self.numpyro_model(), **kernel_kwargs)
        else:
            if kernel_kwargs:
                raise ValueError(
                    "kernel keyword arguments are sugar for a named kernel; configure "
                    f"the {type(kernel).__name__} you passed instead."
                )
            kernel_object = kernel
        return run_mcmc(
            self,
            init,
            make_kernel=lambda: kernel_object,
            num_chains=int(num_chains),
            num_warmup=int(num_warmup),
            chunk_samples=int(num_samples),
            stop=_one_chunk if stop is None else stop,
            seed=int(seed),
            chain_method=chain_method,
            progress_bar=progress_bar,
        )

    def prior_names(self) -> tuple[str, ...]:
        """Names of every sampled site (top-level priors and hierarchical scales)."""
        return tuple(p.name for p in self._sites)

    def site_priors(self) -> tuple[Prior, ...]:
        """Prior objects for every sampled site (same order as :meth:`prior_names`)."""
        return self._sites

    def posterior_runs(
        self,
        draws: Any,
        *,
        n: int | None = 1000,
        burn_in: int = 0,
        seed: int = 0,
        scenarios: Mapping[str, Any] | None = None,
        batch_size: int = 64,
    ) -> Any:
        """Batched forward runs under scenarios; see :class:`PosteriorRuns`.

        ``draws`` is an ``arviz.InferenceData``, a mapping of constrained site
        arrays with leading axis ``n``, or (later) ``Candidates`` via ``.params``.
        ``scenarios`` maps names to ``None`` (baseline) or :class:`Scenario`.
        """
        from summer4.epi.calibration.posterior_runs import run_posterior

        return run_posterior(
            self,
            draws,
            n=n,
            burn_in=burn_in,
            seed=seed,
            scenarios=scenarios,
            batch_size=batch_size,
        )


__all__ = [
    "BayesianModel",
    "SampleKind",
]
