"""Posterior protocol and numpyro model wrapper for the calibration workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from summer4.epi.calibration.priors import NumpyroPrior, Prior


def _require_numpyro() -> Any:
    try:
        import numpyro  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Posterior support requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return numpyro


@runtime_checkable
class Posterior(Protocol):
    """What the workflow stages read from a model. :class:`BayesianModel` satisfies it."""

    def numpyro_model(self) -> Callable[[], None]: ...
    @property
    def potential_fn(self) -> Callable[[Mapping[str, Any]], Any]: ...
    def log_density(self, unconstrained: Mapping[str, Any]) -> Any: ...
    def constrain(self, z: Mapping[str, Any]) -> dict[str, Any]: ...
    def unconstrain(self, params: Mapping[str, Any]) -> dict[str, Any]: ...
    def prior_names(self) -> tuple[str, ...]: ...
    def site_priors(self) -> Sequence[Prior]: ...


def _latent_sample_sites(trace: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return ``(name, value, fn)`` for each latent sample site in trace order."""
    out: list[tuple[str, Any, Any]] = []
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        if site.get("is_observed"):
            continue
        infer = site.get("infer") or {}
        if infer.get("is_auxiliary"):
            continue
        out.append((name, site["value"], site["fn"]))
    return out


def _trace_latent_fns(model: Callable[[], None], seed: int) -> dict[str, Any]:
    from numpyro import handlers

    with handlers.trace() as tr, handlers.seed(rng_seed=seed):
        model()
    return {name: fn for name, _val, fn in _latent_sample_sites(tr)}


def _fn_independent_of_seed(fn0: Any, fn1: Any) -> bool:
    flat0, _ = jax.tree_util.tree_flatten(fn0)
    flat1, _ = jax.tree_util.tree_flatten(fn1)
    if len(flat0) != len(flat1):
        return False
    return all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(flat0, flat1, strict=True)
    )


def _auto_site_priors(
    names: tuple[str, ...],
    shapes: dict[str, tuple[int, ...]],
    model: Callable[[], None],
) -> tuple[Prior, ...]:
    fns0 = _trace_latent_fns(model, 0)
    fns1 = _trace_latent_fns(model, 1)
    out: list[Prior] = []
    for name in names:
        if shapes.get(name, ()) != ():
            continue
        fn = fns0.get(name)
        if fn is None or name not in fns1:
            continue
        if not _fn_independent_of_seed(fn, fns1[name]):
            continue
        prior = NumpyroPrior(name, fn)
        if prior.has_icdf:
            out.append(cast(Prior, prior))
    return tuple(out)


class NumpyroPosterior:
    """Wrap a hand-written numpyro model for calibration workflow stages.

    Args:
        model: A numpyro model callable (``def model(): ...``) whose latent sample
            sites define the parameterisation.
        site_priors: Optional explicit list of :class:`~summer4.epi.calibration.priors.Prior`
            objects to stratify in :func:`~summer4.epi.calibration.workflow.lhs` and
            :func:`~summer4.epi.calibration.workflow.prior_draws`. Each name must appear
            in :meth:`prior_names`. When omitted, scalar sites whose distribution does
            not depend on other sites and expose an icdf are auto-selected; plate
            members, centred hierarchical children, and icdf-less distributions are
            left out and filled via ``numpyro.infer.Predictive`` instead.
        seed: PRNG seed for the construction trace and for the lazy
            ``initialize_model`` key (density only). It does not affect
            :meth:`constrain` or :meth:`unconstrain`.

    :meth:`constrain` and :meth:`unconstrain` accept either unbatched site arrays
    (shapes matching :meth:`site_shapes`) or a leading batch axis of size ``m`` on
    every site. Mixed batched and unbatched leaves raise :class:`ValueError`.
    Each call traces the full numpyro model via ``constrain_fn`` / ``unconstrain_fn``;
    batched inputs use a single ``jax.vmap`` over the leading axis (not a Python
    loop per row). The same applies to the Predictive
    fallback in :func:`~summer4.epi.calibration.workflow.lhs` and
    :func:`~summer4.epi.calibration.workflow.prior_draws`.
    """

    def __init__(
        self,
        model: Callable[[], None],
        *,
        site_priors: Sequence[Prior] | None = None,
        seed: int = 0,
    ) -> None:
        _require_numpyro()
        self._model = model
        self._trace_seed = int(seed)
        self._names, self._shapes = self._trace_sites(self._trace_seed)
        if site_priors is None:
            self._site_priors: tuple[Prior, ...] = _auto_site_priors(
                self._names, self._shapes, model
            )
        else:
            name_set = set(self._names)
            seen: set[str] = set()
            validated: list[Prior] = []
            for prior in site_priors:
                if prior.name in seen:
                    raise ValueError(f"duplicate site_priors name {prior.name!r}.")
                seen.add(prior.name)
                if prior.name not in name_set:
                    raise ValueError(
                        f"site_priors name {prior.name!r} is not in prior_names() {self._names!r}."
                    )
                if self._shapes.get(prior.name, ()) != ():
                    raise ValueError(
                        f"site_priors name {prior.name!r} is not scalar "
                        f"(shape {self._shapes[prior.name]!r})."
                    )
                validated.append(prior)
            self._site_priors = tuple(validated)
        self._potential_fn: Any | None = None
        self._postprocess_fn: Any | None = None

    def _trace_sites(self, seed: int) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
        from numpyro import handlers

        with handlers.trace() as tr, handlers.seed(rng_seed=seed):
            self._model()
        names: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        for name, value, _fn in _latent_sample_sites(tr):
            names.append(name)
            shapes[name] = tuple(int(s) for s in np.shape(value))
        return tuple(names), shapes

    def numpyro_model(self) -> Callable[[], None]:
        """Return the wrapped numpyro model for ``MCMC(NUTS(...))``."""
        return self._model

    def _ensure_initialized(self) -> tuple[Any, Any]:
        from jax import random
        from numpyro.infer.util import initialize_model  # type: ignore[import-untyped]

        if self._potential_fn is not None:
            assert self._postprocess_fn is not None
            return self._potential_fn, self._postprocess_fn
        _params_info, potential_fn, postprocess_fn, _trace = initialize_model(
            random.PRNGKey(self._trace_seed), self._model
        )
        self._potential_fn = potential_fn
        self._postprocess_fn = postprocess_fn
        return potential_fn, postprocess_fn

    @property
    def potential_fn(self) -> Callable[[Mapping[str, Any]], Any]:
        """Unconstrained potential ``-log_density`` (for optimisation backends)."""
        fn, _post = self._ensure_initialized()
        return cast(Callable[[Mapping[str, Any]], Any], fn)

    @property
    def postprocess_fn(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """Map unconstrained dicts to constrained site values (numpyro ``postprocess_fn``)."""
        _fn, post = self._ensure_initialized()
        return cast(Callable[[Mapping[str, Any]], dict[str, Any]], post)

    def log_density(self, unconstrained: Mapping[str, Any]) -> Any:
        """Log posterior density at unconstrained site values."""
        return -self.potential_fn(dict(unconstrained))

    def _batch_axis(self, tree: Mapping[str, Any]) -> int | None:
        """``0`` if batched, ``None`` if unbatched; mixed raises ``ValueError``."""
        batched = 0
        unbatched = 0
        for name in self._names:
            if name not in tree:
                continue
            leaf = jnp.asarray(tree[name])
            expected = len(self._shapes[name])
            if leaf.ndim == expected:
                unbatched += 1
            elif leaf.ndim == expected + 1:
                batched += 1
            else:
                raise ValueError(
                    f"site {name!r}: ndim {leaf.ndim} neither batched nor unbatched "
                    f"(expected {expected} or {expected + 1})."
                )
        if batched and unbatched:
            raise ValueError("constrain/unconstrain: mixed batched and unbatched leaves.")
        return 0 if batched else None

    def constrain(self, z: Mapping[str, Any]) -> dict[str, Any]:
        """Unconstrained → constrained values for every site in :meth:`prior_names`."""
        from numpyro.infer.util import constrain_fn

        names = self._names
        for name in names:
            if name not in z:
                raise KeyError(f"constrain missing site {name!r}.")
        z_sub = {name: z[name] for name in names}
        batch = self._batch_axis(z_sub)

        def _one(row: Mapping[str, Any]) -> dict[str, Any]:
            out = constrain_fn(self._model, (), {}, dict(row))
            return {name: out[name] for name in names}

        if batch is None:
            return _one(z_sub)
        return jax.vmap(_one)(z_sub)

    def unconstrain(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Constrained → unconstrained values for every site in :meth:`prior_names`."""
        from numpyro.infer.util import unconstrain_fn

        names = self._names
        for name in names:
            if name not in params:
                raise KeyError(f"unconstrain missing site {name!r}.")
        p_sub = {name: params[name] for name in names}
        batch = self._batch_axis(p_sub)

        def _one(row: Mapping[str, Any]) -> dict[str, Any]:
            out = unconstrain_fn(self._model, (), {}, dict(row))
            return {name: out[name] for name in names}

        if batch is None:
            return _one(p_sub)
        return jax.vmap(_one)(p_sub)

    def init_point(self, seed: int = 0) -> dict[str, Any]:
        """Default unconstrained initial point from ``initialize_model``."""
        from jax import random
        from numpyro.infer.util import initialize_model

        params_info = initialize_model(random.PRNGKey(int(seed)), self._model)[0]
        return dict(params_info.z)

    def prior_names(self) -> tuple[str, ...]:
        """Latent sample site names in trace order."""
        return self._names

    def site_shapes(self) -> dict[str, tuple[int, ...]]:
        """Constrained value shape for each latent sample site."""
        return dict(self._shapes)

    def site_priors(self) -> Sequence[Prior]:
        """Priors stratified by :func:`~summer4.epi.calibration.workflow.lhs`."""
        return self._site_priors


__all__ = [
    "NumpyroPosterior",
    "Posterior",
]
