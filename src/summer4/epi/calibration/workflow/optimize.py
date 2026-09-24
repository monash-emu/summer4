"""Multi-start optimisation: chunked vmapped scan with AutoTune (step 25)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from summer4.epi.calibration.workflow.candidates import (
    Candidates,
    StageRecord,
    ok_from_log_density,
)

# Potential of a failed solve (log_density sentinel -1e30).
_FAIL_POTENTIAL = 1.0e30


def _require_optax() -> Any:
    try:
        import optax  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "wf.optimize requires the calibration extra: pip install summer4[calibration]"
        ) from exc
    return optax


def _require_evosax_cma() -> Any:
    try:
        from evosax.algorithms import CMA_ES  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "wf.CMAES requires the gradient-free extra: pip install summer4[gradient-free]"
        ) from exc
    return CMA_ES


def _batch_where(active: Any, new: Any, old: Any) -> Any:
    """Select ``new`` or ``old`` per leading-axis lane from a boolean mask."""

    def _one(a: Any, b: Any) -> Any:
        shape = (active.shape[0],) + (1,) * (a.ndim - 1)
        return jnp.where(active.reshape(shape), a, b)

    return jax.tree.map(_one, new, old)


@runtime_checkable
class _Method(Protocol):
    """Internal optimiser backend: one start's state machine (then vmapped)."""

    def init(self, z0: Mapping[str, Any], key: Any) -> Any:
        """Build optimiser state from an unconstrained start."""
        ...

    def step(self, state: Any) -> tuple[Any, Mapping[str, Any], Any]:
        """One update; return ``(state, z_best, loss_best)``."""
        ...


@dataclass(frozen=True, slots=True)
class Optax:
    """Gradient backend: any optax transform (default Adam + plateau).

    When ``optimizer`` is omitted, builds
    ``inject_hyperparams(adam)`` chained with ``reduce_on_plateau`` so
    :class:`AutoTune` can probe learning rates. A user-supplied
    ``GradientTransformation`` is used as-is; the LR probe then runs only if
    its state exposes ``hyperparams['learning_rate']``.
    """

    optimizer: Any | None = None
    learning_rate: float = 0.05
    plateau: bool = True

    def make(
        self,
        potential_fn: Any,
        *,
        learning_rate: float | None = None,
        sites: tuple[str, ...] | None = None,
        template_z: Mapping[str, Any] | None = None,
    ) -> _Method:
        del sites, template_z  # Optax works on the z pytree directly
        optax = _require_optax()
        lr = float(self.learning_rate if learning_rate is None else learning_rate)
        if self.optimizer is None:
            adam = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
            tx = optax.chain(adam, optax.contrib.reduce_on_plateau()) if self.plateau else adam
        else:
            tx = self.optimizer
        return _OptaxMethod(potential_fn, tx, default_lr=lr)


@dataclass
class _OptaxMethod:
    potential_fn: Any
    tx: Any
    default_lr: float

    def supports_lr_probe(self) -> bool:
        # Probe needs inject_hyperparams so LR is a traced hyperparam.
        return True  # refined after a sample init in optimize()

    def init(self, z0: Mapping[str, Any], key: Any) -> dict[str, Any]:
        del key  # optax adam needs no RNG
        z0 = {k: jnp.asarray(v) for k, v in z0.items()}
        opt_state = self.tx.init(z0)
        loss = self.potential_fn(z0)
        return {
            "z": z0,
            "opt_state": opt_state,
            "best_z": z0,
            "best_loss": jnp.asarray(loss),
        }

    def step(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any]:
        z = state["z"]
        loss, grads = jax.value_and_grad(self.potential_fn)(z)
        # reduce_on_plateau expects value=; plain adam ignores extra kwargs via ExtraArgs.
        updates, opt_state = self.tx.update(grads, state["opt_state"], z, value=loss)
        z_new = _require_optax().apply_updates(z, updates)
        improved = loss < state["best_loss"]
        best_z = jax.tree.map(lambda cur, best: jnp.where(improved, cur, best), z, state["best_z"])
        best_loss = jnp.where(improved, loss, state["best_loss"])
        new_state = {
            "z": z_new,
            "opt_state": opt_state,
            "best_z": best_z,
            "best_loss": best_loss,
        }
        return new_state, best_z, best_loss

    def set_learning_rate(self, state: dict[str, Any], lr: float) -> dict[str, Any]:
        opt_state = state["opt_state"]
        # chain(inject_adam, plateau): hyperparams live on the first element
        if hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams:
            opt_state = opt_state._replace(
                hyperparams={**opt_state.hyperparams, "learning_rate": lr}
            )
        elif isinstance(opt_state, tuple) and opt_state:
            inner = opt_state[0]
            if hasattr(inner, "hyperparams") and "learning_rate" in inner.hyperparams:
                inner = inner._replace(
                    hyperparams={**inner.hyperparams, "learning_rate": float(lr)}
                )
                opt_state = (inner, *opt_state[1:])
        return {**state, "opt_state": opt_state}


@dataclass(frozen=True, slots=True)
class CMAES:
    """Gradient-free CMA-ES backend via evosax (``gradient-free`` extra).

    Each ``step`` is one CMA-ES generation: the population is scored under
    ``vmap``, so cost is ``population × starts`` model solves per generation.
    Centred on each start's unconstrained ``z``.
    """

    sigma0: float = 0.1
    population: int | None = None

    def make(
        self,
        potential_fn: Any,
        *,
        learning_rate: float | None = None,
        sites: tuple[str, ...] | None = None,
        template_z: Mapping[str, Any] | None = None,
    ) -> _Method:
        del learning_rate  # unused; CMA-ES has no LR probe
        if sites is None:
            raise ValueError("CMAES.make requires sites= (parameter name order).")
        if template_z is None:
            raise ValueError("CMAES.make requires template_z= (shapes per site).")
        return _CMAESMethod(
            potential_fn=potential_fn,
            sites=sites,
            sigma0=float(self.sigma0),
            population=self.population,
            template_z=template_z,
        )


def _pack_z(z: Mapping[str, Any], sites: tuple[str, ...]) -> Any:
    parts = [jnp.ravel(jnp.asarray(z[name])) for name in sites]
    return jnp.concatenate(parts)


def _unpack_z(vec: Any, sites: tuple[str, ...], template: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    offset = 0
    for name in sites:
        shape = jnp.asarray(template[name]).shape
        size = int(np.prod(shape)) if shape else 1
        out[name] = jnp.reshape(vec[offset : offset + size], shape)
        offset += size
    return out


@dataclass
class _CMAESMethod:
    potential_fn: Any
    sites: tuple[str, ...]
    sigma0: float
    population: int | None
    template_z: Mapping[str, Any]
    _es: Any = None
    _params: Any = None
    _template: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        CMA_ES = _require_evosax_cma()
        self._template = {k: jnp.asarray(self.template_z[k]) for k in self.sites}
        dummy = _pack_z(self._template, self.sites)
        n_dims = int(dummy.size)
        pop = self.population
        if pop is None:
            pop = max(4, int(4 + 3 * np.log(max(n_dims, 1))))
        self._es = CMA_ES(population_size=int(pop), solution=dummy)
        self._params = self._es.default_params.replace(std_init=jnp.asarray(self.sigma0))

    def init(self, z0: Mapping[str, Any], key: Any) -> dict[str, Any]:
        assert self._es is not None and self._params is not None and self._template is not None
        z0 = {k: jnp.asarray(z0[k]) for k in self.sites}
        mean = _pack_z(z0, self.sites)
        es_state = self._es.init(key, mean, self._params)
        es_state = es_state.replace(mean=mean, best_solution=mean)
        loss = self.potential_fn(z0)
        es_state = es_state.replace(best_fitness=jnp.asarray(loss))
        return {
            "z": z0,
            "es_state": es_state,
            "key": key,
            "best_z": z0,
            "best_loss": jnp.asarray(loss),
        }

    def step(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any]:
        assert self._es is not None and self._params is not None and self._template is not None
        template = self._template
        key, k_ask, k_tell = random.split(state["key"], 3)
        population, es_state = self._es.ask(k_ask, state["es_state"], self._params)

        def _loss_vec(vec: Any) -> Any:
            return self.potential_fn(_unpack_z(vec, self.sites, template))

        fitness = jax.vmap(_loss_vec)(population)
        es_state, _metrics = self._es.tell(k_tell, population, fitness, es_state, self._params)
        best_z = _unpack_z(es_state.best_solution, self.sites, template)
        best_loss = jnp.asarray(es_state.best_fitness)
        z_mean = _unpack_z(es_state.mean, self.sites, template)
        new_state = {
            "z": z_mean,
            "es_state": es_state,
            "key": key,
            "best_z": best_z,
            "best_loss": best_loss,
        }
        return new_state, best_z, best_loss


@dataclass(frozen=True, slots=True)
class AutoTune:
    """Automated LR probe, chunk-wise convergence, and failed-start restarts."""

    lr_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    probe_steps: int = 50
    probe_starts: int = 4
    rtol: float = 1e-6
    patience: int = 2
    max_restarts: int = 3
    jitter: float = 0.05


@dataclass(frozen=True, slots=True)
class OptimizeResult:
    """Outcome of :func:`optimize`."""

    candidates: Candidates
    loss_trace: Any  # (n_chunks, n_starts)
    converged: Any  # (n_starts,) bool
    restarts: Any  # (n_starts,) int
    learning_rate: float
    history: tuple[StageRecord, ...] = ()


def _run_chunk(method: _Method, state: Any, chunk_steps: int) -> tuple[Any, Any, Any]:
    """Run ``chunk_steps`` updates on **one** start; callers ``vmap`` over starts."""

    def body(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any]]:
        new_state, z_best, loss_best = method.step(carry)
        return new_state, (z_best, loss_best)

    state, (zs, losses) = jax.lax.scan(body, state, None, length=int(chunk_steps))
    return state, zs, losses


def _run_chunk_vmap(method: _Method, states: Any, chunk_steps: int) -> tuple[Any, Any, Any]:
    """Batched chunk: leading axis is the start index."""
    return jax.vmap(lambda s: _run_chunk(method, s, chunk_steps))(states)


def _tree_set(tree: Any, i: int, value: Any) -> Any:
    def _set(leaf: Any, val: Any) -> Any:
        return leaf.at[i].set(val)

    return jax.tree.map(_set, tree, value)


def _probe_learning_rate(
    method_factory: Optax,
    potential_fn: Any,
    z_probe: dict[str, Any],
    tuning: AutoTune,
    seed: int,
) -> float:
    """Pick the LR with the best median loss drop over ``probe_starts``."""
    n = int(next(iter(z_probe.values())).shape[0])
    n_probe = min(int(tuning.probe_starts), n)
    z0 = {k: jnp.asarray(v[:n_probe]) for k, v in z_probe.items()}
    keys = random.split(random.PRNGKey(int(seed) + 1), n_probe)
    best_lr = float(method_factory.learning_rate)
    best_drop = float("-inf")

    for lr in tuning.lr_grid:
        backend = method_factory.make(
            potential_fn,
            learning_rate=float(lr),
            sites=tuple(z_probe.keys()),
            template_z={k: v[0] for k, v in z_probe.items()},
        )
        states = jax.vmap(backend.init)(z0, keys)
        loss0 = states["best_loss"]
        states, _zs, losses = _run_chunk_vmap(backend, states, int(tuning.probe_steps))
        del states, _zs
        loss1 = losses[:, -1]
        drop = float(jnp.median(loss0 - loss1))
        if drop > best_drop and np.isfinite(drop):
            best_drop = drop
            best_lr = float(lr)
    return best_lr


def _state_supports_lr(state: Mapping[str, Any]) -> bool:
    opt_state = state["opt_state"]
    if hasattr(opt_state, "hyperparams") and "learning_rate" in getattr(
        opt_state, "hyperparams", {}
    ):
        return True
    if isinstance(opt_state, tuple) and opt_state:
        inner = opt_state[0]
        if hasattr(inner, "hyperparams") and "learning_rate" in getattr(inner, "hyperparams", {}):
            return True
    return False


def optimize(
    bm: Any,
    candidates: Candidates,
    *,
    method: Optax | CMAES | None = None,
    tuning: AutoTune | None = None,
    reserve: Candidates | None = None,
    chunk_steps: int = 50,
    max_steps: int = 500,
    seed: int = 0,
    batch_size: int = 64,
) -> OptimizeResult:
    """Multi-start optimisation with a chunked, vmapped compiled program.

    Prefer a fixed-step solver for large start counts: vmapping an adaptive
    Diffrax solve runs every lane to the slowest lane's step count.

    The jaxpr of one chunk does not grow with ``max_steps`` (host loop) or with
    the number of starts (``vmap``). With :class:`CMAES`, each step is one
    generation and costs ``population × starts`` model solves.
    """
    if len(candidates) == 0:
        raise ValueError("optimize() got an empty Candidates batch.")
    if method is None:
        method = Optax()
    if tuning is None:
        tuning = AutoTune()
    if not isinstance(method, (Optax, CMAES)):
        raise TypeError(f"optimize method must be Optax or CMAES; got {type(method).__name__}.")

    t0 = time.perf_counter()
    potential_fn = bm.potential_fn
    sites = candidates.sites
    n = len(candidates)
    chunk_steps = max(1, int(chunk_steps))
    max_steps = max(1, int(max_steps))

    z_batch = {k: jnp.asarray(candidates.z[k]) for k in sites}
    # Warm potential
    _ = potential_fn({k: v[0] for k, v in z_batch.items()})

    lr = float("nan")
    template = {k: v[0] for k, v in z_batch.items()}
    if isinstance(method, Optax):
        probe_backend = method.make(potential_fn, sites=sites, template_z=template)
        sample_state = probe_backend.init({k: v[0] for k, v in z_batch.items()}, random.PRNGKey(0))
        use_probe = _state_supports_lr(sample_state) and method.optimizer is None
        lr = float(method.learning_rate)
        if use_probe:
            lr = _probe_learning_rate(method, potential_fn, z_batch, tuning, seed)
        backend = method.make(potential_fn, learning_rate=lr, sites=sites, template_z=template)
        method_name = "optax"
    else:
        backend = method.make(potential_fn, sites=sites, template_z=template)
        method_name = "cmaes"

    keys = random.split(random.PRNGKey(int(seed)), n)
    states = jax.vmap(backend.init)(z_batch, keys)

    chunk_fn = jax.jit(lambda s: _run_chunk_vmap(backend, s, chunk_steps))

    converged = np.zeros(n, dtype=bool)
    restarts = np.zeros(n, dtype=np.int32)
    patience_count = np.zeros(n, dtype=np.int32)
    reserve_cursor = 0
    prev_loss: np.ndarray | None = None
    loss_rows: list[np.ndarray] = []

    n_chunks = (max_steps + chunk_steps - 1) // chunk_steps
    for _ in range(n_chunks):
        new_states, _zs, losses = chunk_fn(states)
        loss_best = np.asarray(losses[:, -1])
        active = ~converged
        states = _batch_where(jnp.asarray(active), new_states, states)
        # Re-read best_loss from (possibly frozen) states
        loss_best = np.asarray(states["best_loss"])
        loss_rows.append(loss_best.copy())

        if prev_loss is not None:
            denom = np.abs(prev_loss) + 1e-12
            rel = np.abs(prev_loss - loss_best) / denom
            improved = rel >= float(tuning.rtol)
            patience_count = np.where(improved, 0, patience_count + 1)
            newly = patience_count >= int(tuning.patience)
            converged = converged | newly

        # Restart failed / non-finite lanes
        failed = ~np.isfinite(loss_best) | (loss_best >= _FAIL_POTENTIAL * 0.5)
        for i in np.flatnonzero(failed & ~converged):
            if int(restarts[i]) >= int(tuning.max_restarts):
                continue
            z_new: dict[str, Any]
            if reserve is not None and reserve_cursor < len(reserve):
                z_new = {k: np.asarray(reserve.z[k][reserve_cursor]) for k in sites}
                reserve_cursor += 1
            else:
                # Jitter around the current global best among finite losses
                finite = np.isfinite(loss_best) & (loss_best < _FAIL_POTENTIAL * 0.5)
                if np.any(finite):
                    j = int(np.argmin(np.where(finite, loss_best, np.inf)))
                    base = {k: np.asarray(states["best_z"][k][j]) for k in sites}
                else:
                    base = {k: np.asarray(z_batch[k][i]) for k in sites}
                rng = np.random.default_rng(int(seed) + 1000 + int(restarts[i]) * n + int(i))
                z_new = {
                    k: base[k] + float(tuning.jitter) * rng.normal(size=np.shape(base[k]))
                    for k in sites
                }
            key_i = random.fold_in(random.PRNGKey(int(seed)), int(i) + 17 * int(restarts[i]))
            fresh = backend.init({k: jnp.asarray(v) for k, v in z_new.items()}, key_i)
            states = _tree_set(states, int(i), fresh)
            restarts[i] = int(restarts[i]) + 1
            patience_count[i] = 0
            converged[i] = False

        prev_loss = loss_best
        if bool(np.all(converged)):
            break

    # Final scored candidates from best_z
    best_z = {k: np.asarray(states["best_z"][k]) for k in sites}
    best_params = {k: np.asarray(v) for k, v in bm.constrain(best_z).items()}
    # Score with evaluate's density for ok / log_density consistency
    scored_z = {k: jnp.asarray(best_z[k]) for k in sites}
    mapped = jax.jit(lambda z: jax.lax.map(bm.log_density, z, batch_size=max(1, int(batch_size))))
    log_density = np.asarray(mapped(scored_z))
    ok = np.asarray(ok_from_log_density(log_density))
    seconds = time.perf_counter() - t0
    record = StageRecord(
        stage="optimize",
        settings={
            "chunk_steps": chunk_steps,
            "max_steps": max_steps,
            "learning_rate": lr if method_name == "optax" else None,
            "n": n,
            "method": method_name,
        },
        seconds=seconds,
    )
    out = Candidates(
        sites=sites,
        z=best_z,
        params=best_params,
        log_density=log_density,
        ok=ok,
        history=candidates.history + (record,),
    )
    trace = np.stack(loss_rows, axis=0) if loss_rows else np.zeros((0, n))
    return OptimizeResult(
        candidates=out,
        loss_trace=trace,
        converged=converged,
        restarts=restarts,
        learning_rate=lr,
        history=(record,),
    )


# Exposed for jaxpr tests: one compiled chunk over a state batch.
def _chunk_program(backend: _Method, chunk_steps: int) -> Any:
    return jax.jit(lambda s: _run_chunk_vmap(backend, s, chunk_steps))


__all__ = [
    "AutoTune",
    "CMAES",
    "Optax",
    "OptimizeResult",
    "optimize",
]
