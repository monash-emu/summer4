"""Multi-start optimisation: chunked vmapped scan with AutoTune (steps 25, 30).

A method (:class:`Optax`, :class:`CMAES`, or any :class:`OptimizeMethod`) makes
an :class:`OptimizeBackend` — one start's ``init`` / ``step`` state machine —
which :class:`OptimizeRun` vmaps over the starts and advances in compiled
chunks. :func:`optimize` is the one-call convenience.
"""

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
class OptimizeBackend(Protocol):
    """One start's optimiser state machine; :class:`OptimizeRun` vmaps it over starts.

    ``init`` builds a state pytree from an unconstrained start ``z0`` (a site
    dict) and a PRNG key; the state must carry ``"best_z"`` and ``"best_loss"``.
    ``step`` performs one update and returns ``(state, best_z, best_loss)``.
    Both are traced under ``jax.vmap`` and ``jax.lax.scan``.
    """

    def init(self, z0: Mapping[str, Any], key: Any) -> Any:
        """Build optimiser state from an unconstrained start."""
        ...

    def step(self, state: Any) -> tuple[Any, Mapping[str, Any], Any]:
        """One update; return ``(state, z_best, loss_best)``."""
        ...


@runtime_checkable
class OptimizeMethod(Protocol):
    """What ``optimize(method=)`` accepts: a factory for an :class:`OptimizeBackend`.

    ``potential_fn`` maps an unconstrained site dict to ``-log_density``;
    ``sites`` and ``template_z`` give the site order and one start's shapes.
    ``learning_rate`` is set only when :class:`AutoTune` probed one (``Optax``).
    """

    def make(
        self,
        potential_fn: Any,
        *,
        learning_rate: float | None = None,
        sites: tuple[str, ...] | None = None,
        template_z: Mapping[str, Any] | None = None,
    ) -> OptimizeBackend:
        """Build the backend for this potential."""
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
    ) -> OptimizeBackend:
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
    ) -> OptimizeBackend:
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


def _run_chunk(method: OptimizeBackend, state: Any, chunk_steps: int) -> tuple[Any, Any, Any]:
    """Run ``chunk_steps`` updates on **one** start; callers ``vmap`` over starts."""

    def body(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any]]:
        new_state, z_best, loss_best = method.step(carry)
        return new_state, (z_best, loss_best)

    state, (zs, losses) = jax.lax.scan(body, state, None, length=int(chunk_steps))
    return state, zs, losses


def _run_chunk_vmap(method: OptimizeBackend, states: Any, chunk_steps: int) -> tuple[Any, Any, Any]:
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


class OptimizeRun:
    """A resumable multi-start optimisation: the backend, its states, and the record.

    Constructing one probes the learning rate (:class:`Optax` without a custom
    ``optimizer``, under :class:`AutoTune`), builds the backend from ``method``,
    and initialises one state per start; :meth:`extend` then advances every
    start in compiled chunks of ``chunk_steps``, freezing starts that converge
    and restarting failed ones (from ``reserve``, else jittered around the best
    start). Calling :meth:`extend` again continues from the stored states —
    Adam moments or CMA-ES covariance included.
    """

    def __init__(
        self,
        bm: Any,
        candidates: Candidates,
        *,
        method: OptimizeMethod | None = None,
        tuning: AutoTune | None = None,
        reserve: Candidates | None = None,
        chunk_steps: int = 50,
        seed: int = 0,
        batch_size: int = 64,
    ) -> None:
        if len(candidates) == 0:
            raise ValueError("optimize() got an empty Candidates batch.")
        method = Optax() if method is None else method
        if not isinstance(method, OptimizeMethod):
            raise TypeError(
                "optimize method must implement OptimizeMethod (a make(potential_fn, ...) "
                f"method); got {type(method).__name__}."
            )
        start = time.perf_counter()
        self.bm = bm
        self.method = method
        self.tuning = AutoTune() if tuning is None else tuning
        self.reserve = reserve
        self.chunk_steps = max(1, int(chunk_steps))
        self.seed = int(seed)
        self.batch_size = max(1, int(batch_size))
        self.sites = candidates.sites
        self._input_history = candidates.history
        potential_fn = bm.potential_fn
        n = len(candidates)

        self._z_start = {k: jnp.asarray(candidates.z[k]) for k in self.sites}
        _ = potential_fn({k: v[0] for k, v in self._z_start.items()})  # warm the potential
        template = {k: v[0] for k, v in self._z_start.items()}

        self.learning_rate = float("nan")
        if isinstance(method, Optax):
            probe_backend = method.make(potential_fn, sites=self.sites, template_z=template)
            sample_state = probe_backend.init(template, random.PRNGKey(0))
            self.learning_rate = float(method.learning_rate)
            if _state_supports_lr(sample_state) and method.optimizer is None:
                self.learning_rate = _probe_learning_rate(
                    method, potential_fn, self._z_start, self.tuning, self.seed
                )
            self.backend: OptimizeBackend = method.make(
                potential_fn,
                learning_rate=self.learning_rate,
                sites=self.sites,
                template_z=template,
            )
        else:
            self.backend = method.make(potential_fn, sites=self.sites, template_z=template)

        keys = random.split(random.PRNGKey(self.seed), n)
        self.states: Any = jax.vmap(self.backend.init)(self._z_start, keys)
        backend, steps = self.backend, self.chunk_steps
        self.chunk_program: Any = jax.jit(lambda s: _run_chunk_vmap(backend, s, steps))
        """The jitted chunk over the state batch; its jaxpr is independent of ``n`` and steps."""

        self._converged = np.zeros(n, dtype=bool)
        self._restarts = np.zeros(n, dtype=np.int32)
        self._patience = np.zeros(n, dtype=np.int32)
        self._reserve_cursor = 0
        self._prev_loss: np.ndarray | None = None
        self._loss_rows: list[np.ndarray] = []
        self._records: list[StageRecord] = []
        self._candidates: Candidates | None = None
        self._score: Any = None
        self._setup_seconds = time.perf_counter() - start

    @property
    def method_name(self) -> str:
        """``"optax"``, ``"cmaes"``, or the custom method's class name in lower case."""
        if isinstance(self.method, Optax):
            return "optax"
        if isinstance(self.method, CMAES):
            return "cmaes"
        return type(self.method).__name__.lower()

    @property
    def loss_trace(self) -> np.ndarray:
        """Best loss per start after every chunk so far, shape ``(chunks, starts)``."""
        n = len(self._converged)
        return np.stack(self._loss_rows, axis=0) if self._loss_rows else np.zeros((0, n))

    @property
    def converged(self) -> np.ndarray:
        """Per start: whether its loss has stopped improving (then it is frozen)."""
        return self._converged.copy()

    @property
    def restarts(self) -> np.ndarray:
        """Per start: how many times a failed start was restarted."""
        return self._restarts.copy()

    @property
    def candidates(self) -> Candidates:
        """Each start's best point, scored; ``history`` adds one record per call."""
        if self._candidates is None:
            self._candidates = self._scored()
        return self._candidates

    @property
    def history(self) -> tuple[StageRecord, ...]:
        """The cumulative stage history (the input's, then one record per call)."""
        return self.candidates.history

    @property
    def best_params(self) -> dict[str, Any]:
        """Constrained parameters of the best successful start, one value per site."""
        cands = self.candidates
        ld = np.where(np.asarray(cands.ok, dtype=bool), np.asarray(cands.log_density), -np.inf)
        i = int(np.argmax(ld))
        return {k: np.asarray(cands.params[k])[i] for k in self.sites}

    def extend(self, max_steps: int) -> OptimizeRun:
        """Advance every unconverged start by up to ``max_steps`` more steps.

        Runs ``ceil(max_steps / chunk_steps)`` chunks, stopping early once every
        start has converged. Mutates and returns ``self``.
        """
        start = time.perf_counter()
        tuning = self.tuning
        n = len(self._converged)
        n_steps = max(1, int(max_steps))
        n_chunks = (n_steps + self.chunk_steps - 1) // self.chunk_steps
        for _ in range(n_chunks):
            if bool(np.all(self._converged)):
                break
            new_states, _zs, _losses = self.chunk_program(self.states)
            active = ~self._converged
            self.states = _batch_where(jnp.asarray(active), new_states, self.states)
            loss_best = np.asarray(self.states["best_loss"])
            self._loss_rows.append(loss_best.copy())

            if self._prev_loss is not None:
                rel = np.abs(self._prev_loss - loss_best) / (np.abs(self._prev_loss) + 1e-12)
                self._patience = np.where(rel >= float(tuning.rtol), 0, self._patience + 1)
                self._converged = self._converged | (self._patience >= int(tuning.patience))

            failed = ~np.isfinite(loss_best) | (loss_best >= _FAIL_POTENTIAL * 0.5)
            for i in np.flatnonzero(failed & ~self._converged):
                self._restart(int(i), loss_best, n)
            self._prev_loss = loss_best

        seconds = time.perf_counter() - start + self._setup_seconds
        self._setup_seconds = 0.0
        self._records.append(
            StageRecord(
                stage="optimize",
                settings={
                    "chunk_steps": self.chunk_steps,
                    "max_steps": n_steps,
                    "learning_rate": self.learning_rate if self.method_name == "optax" else None,
                    "n": n,
                    "method": self.method_name,
                },
                seconds=seconds,
            )
        )
        self._candidates = None
        return self

    def _restart(self, i: int, loss_best: np.ndarray, n: int) -> None:
        tuning = self.tuning
        if int(self._restarts[i]) >= int(tuning.max_restarts):
            return
        z_new: dict[str, Any]
        if self.reserve is not None and self._reserve_cursor < len(self.reserve):
            z_new = {k: np.asarray(self.reserve.z[k][self._reserve_cursor]) for k in self.sites}
            self._reserve_cursor += 1
        else:
            # Jitter around the current global best among finite losses.
            finite = np.isfinite(loss_best) & (loss_best < _FAIL_POTENTIAL * 0.5)
            if np.any(finite):
                j = int(np.argmin(np.where(finite, loss_best, np.inf)))
                base = {k: np.asarray(self.states["best_z"][k][j]) for k in self.sites}
            else:
                base = {k: np.asarray(self._z_start[k][i]) for k in self.sites}
            rng = np.random.default_rng(self.seed + 1000 + int(self._restarts[i]) * n + i)
            z_new = {
                k: base[k] + float(tuning.jitter) * rng.normal(size=np.shape(base[k]))
                for k in self.sites
            }
        key_i = random.fold_in(random.PRNGKey(self.seed), i + 17 * int(self._restarts[i]))
        fresh = self.backend.init({k: jnp.asarray(v) for k, v in z_new.items()}, key_i)
        self.states = _tree_set(self.states, i, fresh)
        self._restarts[i] = int(self._restarts[i]) + 1
        self._patience[i] = 0
        self._converged[i] = False

    def _scored(self) -> Candidates:
        best_z = {k: np.asarray(self.states["best_z"][k]) for k in self.sites}
        best_params = {k: np.asarray(v) for k, v in self.bm.constrain(best_z).items()}
        if self._score is None:
            bm, batch = self.bm, self.batch_size
            self._score = jax.jit(lambda z: jax.lax.map(bm.log_density, z, batch_size=batch))
        log_density = np.asarray(self._score({k: jnp.asarray(v) for k, v in best_z.items()}))
        return Candidates(
            sites=self.sites,
            z=best_z,
            params=best_params,
            log_density=log_density,
            ok=np.asarray(ok_from_log_density(log_density)),
            history=self._input_history + tuple(self._records),
        )


def optimize(
    bm: Any,
    candidates: Candidates,
    *,
    method: OptimizeMethod | None = None,
    tuning: AutoTune | None = None,
    reserve: Candidates | None = None,
    chunk_steps: int = 50,
    max_steps: int = 500,
    seed: int = 0,
    batch_size: int = 64,
) -> OptimizeRun:
    """Multi-start optimisation with a chunked, vmapped compiled program.

    Exactly equivalent to::

        run = OptimizeRun(bm, candidates, method=method, tuning=tuning, reserve=reserve,
                          chunk_steps=chunk_steps, seed=seed, batch_size=batch_size)
        return run.extend(max_steps)

    ``method`` is :class:`Optax` (default), :class:`CMAES`, or any
    :class:`OptimizeMethod`. Call ``run.extend(...)`` to keep going.

    Prefer a fixed-step solver for large start counts: vmapping an adaptive
    Diffrax solve runs every lane to the slowest lane's step count. The jaxpr of
    one chunk does not grow with ``max_steps`` (host loop) or with the number of
    starts (``vmap``). With :class:`CMAES`, each step is one generation and costs
    ``population × starts`` model solves.
    """
    run = OptimizeRun(
        bm,
        candidates,
        method=method,
        tuning=tuning,
        reserve=reserve,
        chunk_steps=chunk_steps,
        seed=seed,
        batch_size=batch_size,
    )
    return run.extend(max_steps)


__all__ = [
    "AutoTune",
    "CMAES",
    "Optax",
    "OptimizeBackend",
    "OptimizeMethod",
    "OptimizeRun",
    "optimize",
]
