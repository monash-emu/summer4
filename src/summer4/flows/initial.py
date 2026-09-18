"""Declarative initial population with ragged-aware balanced splits."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from summer4.flows.join import selector_properties
from summer4.flows.rates import Const, RateOps, _rate_bytes, as_rate
from summer4.flows.stages import rate_stage
from summer4.jax.propertydata import PropertyData
from summer4.properties import Property
from summer4.propertymap import PropertyMap
from summer4.selectors import Everything, Selector

_NA: int = -1


class _Remainder:
    """Sentinel: remaining weight after other traits (at most one per mapping)."""

    def __repr__(self) -> str:
        return "REMAINDER"


REMAINDER: Final = _Remainder()

BaseValue = float | RateOps | Callable[[Any], Any]
WeightSpec = Mapping[str, float | RateOps | _Remainder] | RateOps | Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class Split:
    """Population split along one property, optionally filtered or keyed by others."""

    prop: Property
    weights: WeightSpec
    by: tuple[Property, ...] = ()
    where: Selector | None = None
    normalize: bool = False

    def __init__(
        self,
        prop: Property,
        weights: WeightSpec,
        *,
        by: Sequence[Property] | Property = (),
        where: Selector | None = None,
        normalize: bool = False,
    ) -> None:
        if isinstance(by, Property):
            by_t: tuple[Property, ...] = (by,)
        else:
            by_t = tuple(by)
        object.__setattr__(self, "prop", prop)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "by", by_t)
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "normalize", normalize)


@dataclass(frozen=True, slots=True, eq=False)
class InitialPopulation:
    """Declarative base totals plus ordered splits over a :class:`PropertyMap`."""

    base: tuple[tuple[Selector, BaseValue], ...]
    splits: tuple[Split, ...] = ()

    def __init__(
        self,
        base: Mapping[Selector, BaseValue] | Sequence[tuple[Selector, BaseValue]],
        splits: Sequence[Split] = (),
    ) -> None:
        pairs = tuple(base.items()) if isinstance(base, Mapping) else tuple(base)
        object.__setattr__(self, "base", pairs)
        object.__setattr__(self, "splits", tuple(splits))

    def compile(self, pmap: PropertyMap) -> InitPlan:
        """Validate and build host index arrays for traced evaluation."""
        return InitPlan.from_population(self, pmap)


@dataclass(frozen=True, slots=True, eq=False)
class _SplitPlan:
    prop_index: int
    rows: NDArray[np.int32]
    by_codes: NDArray[np.int32]
    by_K: tuple[int, ...]
    K: int
    weights: WeightSpec
    normalize: bool
    mapping_keys: tuple[str, ...] | None
    where_repr: str


@dataclass(frozen=True, slots=True, eq=False)
class InitPlan:
    """Compiled initial-population plan: host indices plus evaluate-time weights."""

    pmap: PropertyMap
    member: NDArray[np.bool_]
    free: NDArray[np.bool_]
    prop_names: tuple[str, ...]
    prop_codes: tuple[NDArray[np.int16], ...]
    prop_carries: tuple[NDArray[np.bool_], ...]
    prop_K: tuple[int, ...]
    splits: tuple[_SplitPlan, ...]
    base_values: tuple[BaseValue, ...]
    base_sel_repr: tuple[str, ...]

    @classmethod
    def from_population(cls, pop: InitialPopulation, pmap: PropertyMap) -> InitPlan:
        if not pop.base:
            raise ValueError("InitialPopulation.base must have at least one entry.")
        n = pmap.size
        m = pmap.n_properties
        prop_names = tuple(p.name for p in pmap.properties)
        prop_index = {p.name: i for i, p in enumerate(pmap.properties)}
        prop_codes = tuple(pmap.column(p) for p in pmap.properties)
        prop_carries = tuple(c != _NA for c in prop_codes)
        prop_K = tuple(len(p.traits) for p in pmap.properties)

        e = len(pop.base)
        member = np.zeros((e, n), dtype=np.bool_)
        free = np.zeros((e, m), dtype=np.bool_)
        base_values: list[BaseValue] = []
        base_sel_repr: list[str] = []
        for ei, (sel, value) in enumerate(pop.base):
            idx = pmap.select(sel)
            if idx.size == 0:
                raise ValueError(f"Base entry selector matches no compartments: {sel!r}.")
            member[ei, idx] = True
            base_values.append(_validate_base_value(value))
            base_sel_repr.append(repr(sel))
            for pi, carries in enumerate(prop_carries):
                codes_e = prop_codes[pi][idx]
                carried = codes_e[carries[idx]]
                free[ei, pi] = len(np.unique(carried)) >= 2 if carried.size else False

        # Validate and compile splits; rows_j account for later overrides per property.
        splits_by_prop: dict[str, list[tuple[int, Split]]] = {name: [] for name in prop_names}
        for si, split in enumerate(pop.splits):
            if split.prop.name not in prop_index:
                raise ValueError(f"Unknown property {split.prop.name!r} in Split.")
            for b in split.by:
                if b.name not in prop_index:
                    raise ValueError(f"Unknown by property {b.name!r} in Split.")
                if b.name == split.prop.name:
                    raise ValueError(f"by property {b.name!r} must not equal the split property.")
            if split.where is not None:
                where_props = selector_properties(split.where)
                if split.prop.name in where_props:
                    raise ValueError(f"where must not mention split property {split.prop.name!r}.")
            splits_by_prop[split.prop.name].append((si, split))

        split_plans: list[_SplitPlan | None] = [None] * len(pop.splits)
        for pname, ordered in splits_by_prop.items():
            pi = prop_index[pname]
            for j, (si, split) in enumerate(ordered):
                where_sel = split.where if split.where is not None else Everything()
                where_rows = set(int(x) for x in pmap.select(where_sel))
                later: set[int] = set()
                for _, later_split in ordered[j + 1 :]:
                    later_where = (
                        later_split.where if later_split.where is not None else Everything()
                    )
                    later |= set(int(x) for x in pmap.select(later_where))
                rows = np.asarray(sorted(where_rows - later), dtype=np.int32)
                # Governed rows must carry the split property and every by property.
                if rows.size:
                    carries_p = prop_carries[pi][rows]
                    if not bool(np.all(carries_p)):
                        raise ValueError(
                            f"Split on {pname!r}: governed rows must carry the property."
                        )
                    for b in split.by:
                        bi = prop_index[b.name]
                        if not bool(np.all(prop_carries[bi][rows])):
                            raise ValueError(
                                f"Split on {pname!r}: governed rows must carry by "
                                f"property {b.name!r}."
                            )
                by_codes = (
                    np.column_stack([prop_codes[prop_index[b.name]][rows] for b in split.by])
                    if split.by
                    else np.zeros((rows.size, 0), dtype=np.int32)
                ).astype(np.int32, copy=False)
                mapping_keys, weights = _validate_weights(split, pmap.properties[pi])
                split_plans[si] = _SplitPlan(
                    prop_index=pi,
                    rows=rows,
                    by_codes=by_codes,
                    by_K=tuple(len(b.traits) for b in split.by),
                    K=prop_K[pi],
                    weights=weights,
                    normalize=split.normalize,
                    mapping_keys=mapping_keys,
                    where_repr=repr(split.where),
                )
        assert all(s is not None for s in split_plans)
        return cls(
            pmap=pmap,
            member=member,
            free=free,
            prop_names=prop_names,
            prop_codes=prop_codes,
            prop_carries=prop_carries,
            prop_K=prop_K,
            splits=tuple(split_plans),  # type: ignore[arg-type]
            base_values=tuple(base_values),
            base_sel_repr=tuple(base_sel_repr),
        )

    def digest_bytes(self) -> bytes:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(b"init")
        for codes in self.prop_codes:
            hasher.update(np.ascontiguousarray(codes).tobytes())
        hasher.update(np.ascontiguousarray(self.member).tobytes())
        hasher.update(np.ascontiguousarray(self.free).tobytes())
        hasher.update(repr(self.prop_names).encode())
        for sp in self.splits:
            hasher.update(self.prop_names[sp.prop_index].encode())
            hasher.update(np.ascontiguousarray(sp.rows).tobytes())
            hasher.update(np.ascontiguousarray(sp.by_codes).tobytes())
            hasher.update(repr(sp.by_K).encode())
            hasher.update(sp.where_repr.encode())
            hasher.update(b"n1" if sp.normalize else b"n0")
            hasher.update(_weight_digest(sp.weights, sp.mapping_keys))
        for value in self.base_values:
            hasher.update(_base_digest(value))
        hasher.update(repr(self.base_sel_repr).encode())
        return hasher.digest()

    def evaluate(self, params: object) -> PropertyData:
        """Distribute base totals through shares into a :class:`PropertyData`."""
        import jax.numpy as jnp

        from summer4.flows.compiled import _eval_rate

        m = len(self.prop_names)
        # Per-property share vectors s_P[r].
        shares: list[Any] = []
        for pi in range(m):
            carries = jnp.asarray(self.prop_carries[pi])
            codes = jnp.asarray(self.prop_codes[pi])
            k = self.prop_K[pi]
            s = jnp.where(carries, 1.0 / float(k), 1.0)
            for sp in self.splits:
                if sp.prop_index != pi or sp.rows.size == 0:
                    continue
                table = _eval_weight_table(sp, params, eval_rate=_eval_rate, pmap=self.pmap)
                rows = jnp.asarray(sp.rows)
                by = jnp.asarray(sp.by_codes)
                trait_codes = codes[rows]
                if by.shape[1] == 0:
                    gathered = table[trait_codes]
                else:
                    idx = tuple(by[:, j] for j in range(by.shape[1])) + (trait_codes,)
                    gathered = table[idx]
                s = s.at[rows].set(gathered)
            shares.append(s)
        S = jnp.stack(shares, axis=0)  # [m, N]
        free = jnp.asarray(self.free)  # [E, m]
        member = jnp.asarray(self.member)  # [E, N]
        # W[e, r] = prod_P (s_P[r] if free else 1) * member
        S_b = S[None, :, :]  # [1, m, N]
        free_b = free[:, :, None]  # [E, m, 1]
        W = jnp.prod(jnp.where(free_b, S_b, 1.0), axis=1) * member
        v = jnp.stack(
            [
                _eval_base_value(val, params, eval_rate=_eval_rate, pmap=self.pmap)
                for val in self.base_values
            ]
        )
        denom = W.sum(axis=-1)
        safe = jnp.where(denom == 0, 1.0, denom)
        y = ((v[:, None] * W) / safe[:, None]).sum(axis=0)
        return PropertyData.wrap(self.pmap, y)


def _validate_base_value(value: BaseValue) -> BaseValue:
    if callable(value) and not isinstance(value, RateOps):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, RateOps):
        if rate_stage(value, params_are_static=True) != "run":
            raise ValueError(
                f"Base value rate expression must be run-stage, got step-stage {value!r}."
            )
        return value
    raise TypeError(f"Unsupported base value type {type(value).__name__}.")


def _validate_weights(split: Split, prop: Property) -> tuple[tuple[str, ...] | None, WeightSpec]:
    weights = split.weights
    if isinstance(weights, Mapping):
        if split.by:
            raise ValueError("Mapping weights are not allowed with by=.")
        keys = tuple(weights.keys())
        expected = set(prop.traits)
        got = set(keys)
        if got != expected:
            missing = expected - got
            extra = got - expected
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"extra {sorted(extra)}")
            raise ValueError(f"Split on {prop.name!r}: mapping keys {'; '.join(parts)}.")
        n_rem = sum(1 for v in weights.values() if isinstance(v, _Remainder))
        if n_rem > 1:
            raise ValueError(f"Split on {prop.name!r}: at most one REMAINDER allowed.")
        concrete: list[float] = []
        has_dynamic = False
        for trait, w in weights.items():
            if isinstance(w, _Remainder):
                continue
            if callable(w) and not isinstance(w, RateOps):
                raise TypeError(
                    f"Split on {prop.name!r}: callable weights must be the whole weights= "
                    f"argument, not a per-trait mapping value."
                )
            concrete_f = _concrete_float(w)
            if concrete_f is not None:
                if concrete_f < 0:
                    raise ValueError(f"Split on {prop.name!r}: negative weight for {trait!r}.")
                concrete.append(concrete_f)
                continue
            if isinstance(w, RateOps):
                if rate_stage(w, params_are_static=True) != "run":
                    raise ValueError(
                        f"Split on {prop.name!r}: weight for {trait!r} must be run-stage."
                    )
                has_dynamic = True
                continue
            raise TypeError(
                f"Split on {prop.name!r}: unsupported weight type {type(w).__name__} "
                f"for trait {trait!r}."
            )
        if not has_dynamic and n_rem == 0:
            total = float(sum(concrete))
            if abs(total - 1.0) > 1e-9 and not split.normalize:
                raise ValueError(
                    f"Split on {prop.name!r}: concrete weights sum to {total}, not 1 "
                    f"(pass normalize=True to allow)."
                )
        if not has_dynamic and n_rem == 1 and float(sum(concrete)) > 1.0 + 1e-9:
            raise ValueError(f"Split on {prop.name!r}: non-REMAINDER weights sum to more than 1.")
        coerced: dict[str, float | RateOps | _Remainder] = {}
        for trait, w in weights.items():
            if isinstance(w, _Remainder):
                coerced[trait] = REMAINDER
            elif isinstance(w, RateOps):
                coerced[trait] = w
            else:
                coerced[trait] = float(w)
        return keys, coerced
    if isinstance(weights, RateOps):
        if rate_stage(weights, params_are_static=True) != "run":
            raise ValueError(f"Split on {prop.name!r}: weights RateOps must be run-stage.")
        return None, weights
    if callable(weights):
        return None, weights
    raise TypeError(f"Unsupported Split.weights type {type(weights).__name__}.")


def _concrete_float(w: object) -> float | None:
    if isinstance(w, (int, float, np.integer, np.floating)):
        return float(w)
    if isinstance(w, Const):
        return float(w.value)
    return None


def _base_digest(value: BaseValue) -> bytes:
    if callable(value) and not isinstance(value, RateOps):
        return b"fn" + str(id(value)).encode()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return b"f" + np.float64(value).tobytes()
    assert isinstance(value, RateOps)
    return b"r" + _rate_bytes(value)


def _weight_digest(weights: WeightSpec, mapping_keys: tuple[str, ...] | None) -> bytes:
    if isinstance(weights, Mapping):
        assert mapping_keys is not None
        parts = [b"map", repr(mapping_keys).encode()]
        for k in mapping_keys:
            w = weights[k]
            if isinstance(w, _Remainder):
                parts.append(b"rem")
            elif isinstance(w, RateOps):
                parts.append(_rate_bytes(w))
            else:
                parts.append(b"f" + np.float64(float(w)).tobytes())
        return b"".join(parts)
    if isinstance(weights, RateOps):
        return b"r" + _rate_bytes(weights)
    return b"fn" + str(id(weights)).encode()


def _eval_base_value(
    value: BaseValue,
    params: object,
    *,
    eval_rate: Callable[..., Any],
    pmap: PropertyMap,
) -> Any:
    import jax.numpy as jnp

    if callable(value) and not isinstance(value, RateOps):
        return jnp.asarray(value(params))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return jnp.asarray(float(value))
    return jnp.asarray(
        eval_rate(
            value if isinstance(value, RateOps) else as_rate(value),
            derived=params,
            flow_values={},
            flow_meta={},
            pmap=pmap,
            t=None,
            y_arr=None,
            captures={},
        )
    )


def _eval_weight_table(
    sp: _SplitPlan,
    params: object,
    *,
    eval_rate: Callable[..., Any],
    pmap: PropertyMap,
) -> Any:
    import jax.numpy as jnp

    weights = sp.weights
    if isinstance(weights, Mapping):
        assert sp.mapping_keys is not None
        parts: list[Any] = []
        rem_i: int | None = None
        for i, key in enumerate(sp.mapping_keys):
            w = weights[key]
            if isinstance(w, _Remainder):
                rem_i = i
                parts.append(jnp.asarray(0.0))
            elif isinstance(w, RateOps):
                parts.append(
                    jnp.asarray(
                        eval_rate(
                            w,
                            derived=params,
                            flow_values={},
                            flow_meta={},
                            pmap=pmap,
                            t=None,
                            y_arr=None,
                            captures={},
                        )
                    )
                )
            else:
                parts.append(jnp.asarray(float(w)))
        stacked = jnp.stack(parts, axis=-1)
        if rem_i is not None:
            others = stacked.sum(axis=-1) - stacked[..., rem_i]
            stacked = stacked.at[..., rem_i].set(1.0 - others)
        return stacked / stacked.sum(axis=-1, keepdims=True)

    if isinstance(weights, RateOps):
        raw = eval_rate(
            weights,
            derived=params,
            flow_values={},
            flow_meta={},
            pmap=pmap,
            t=None,
            y_arr=None,
            captures={},
        )
    else:
        raw = weights(params)
    table = jnp.asarray(raw)
    expected_rank = len(sp.by_K) + 1
    if table.ndim != expected_rank:
        raise ValueError(
            f"Split weight array rank {table.ndim} does not match expected {expected_rank} "
            f"(by + trait)."
        )
    expected_shape = (*sp.by_K, sp.K)
    if tuple(int(s) for s in table.shape) != expected_shape:
        raise ValueError(
            f"Split weight shape {tuple(table.shape)} does not match expected {expected_shape}."
        )
    return table / table.sum(axis=-1, keepdims=True)
