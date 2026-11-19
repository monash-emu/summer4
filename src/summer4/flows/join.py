"""Selector binding, identity join, and pairing overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from summer4.flows.edges import EdgeMap, EdgeRoles
from summer4.flows.rates import Const, RateOps, _rate_bytes, as_rate
from summer4.properties import Property, Trait
from summer4.propertymap import PropertyMap
from summer4.selectors import (
    Absent,
    And,
    Dest,
    Everything,
    IsIn,
    Not,
    Nothing,
    Or,
    Present,
    Selector,
    Source,
)

_NA: int = -1
_SPLIT_TOL: float = 1e-9

type SplitValue = float | RateOps
type SplitSpec = Mapping[Property | str, Mapping[str, object]]
type NormalizedSplit = dict[str, dict[str, SplitValue]]


@dataclass(frozen=True, slots=True)
class DynamicSplit:
    """Run-time destination weights for split proportions given as rate expressions.

    A property whose ``split`` proportions include a rate expression (``Param``,
    ``1 - Param(...)``, a time-varying function) is left out of the static edge
    weights, which then share that property's traits equally. At run time each
    edge's static weight is multiplied by the proportion of its destination trait
    and renormalised within its group: the edges of one source compartment within
    one join (one pairing part), or, for an entry flow, one destination group.

    ``values[p]`` holds property ``names[p]``'s proportions in ``traits[p]`` order.
    ``codes[p, e]`` indexes ``traits[p]`` for edge ``e``, or is ``-1`` where the
    property is absent on that edge's destination (factor 1). Edges alone in their
    group (``singleton``) keep their static weight, as a static split would.
    """

    names: tuple[str, ...]
    traits: tuple[tuple[str, ...], ...]
    values: tuple[tuple[RateOps, ...], ...]
    codes: NDArray[np.int32]
    group: NDArray[np.int32]
    n_groups: int
    singleton: NDArray[np.bool_]

    def __post_init__(self) -> None:
        codes = np.ascontiguousarray(self.codes, dtype=np.int32)
        codes.flags.writeable = False
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "group", _freeze_int32(self.group))
        singleton = np.ascontiguousarray(self.singleton, dtype=np.bool_)
        singleton.flags.writeable = False
        object.__setattr__(self, "singleton", singleton)

    def roots(self) -> tuple[RateOps, ...]:
        """Every proportion expression, in property then trait order."""
        return tuple(value for values in self.values for value in values)

    def digest_bytes(self) -> bytes:
        """Stable bytes for model digests: names, traits, expressions and edge arrays."""
        parts: list[bytes] = [b"dynsplit"]
        for name, traits, values in zip(self.names, self.traits, self.values, strict=True):
            parts.append(name.encode())
            for trait, value in zip(traits, values, strict=True):
                parts.append(trait.encode())
                parts.append(_rate_bytes(value))
        parts.append(self.codes.tobytes())
        parts.append(self.group.tobytes())
        parts.append(self.singleton.tobytes())
        return b"\x00".join(parts)


def _is_dynamic(proportions: Mapping[str, SplitValue]) -> bool:
    return any(isinstance(value, RateOps) for value in proportions.values())


def _static_split(split: NormalizedSplit) -> dict[str, dict[str, float]]:
    """The properties whose proportions are all plain numbers."""
    return {
        name: {trait: float(cast(float, value)) for trait, value in proportions.items()}
        for name, proportions in split.items()
        if not _is_dynamic(proportions)
    }


def _build_dynamic_split(
    pmap: PropertyMap,
    split: NormalizedSplit,
    dest_idx: NDArray[np.int32],
    group: NDArray[np.int32],
) -> DynamicSplit | None:
    """Bind the rate-valued split properties to per-edge trait codes, or ``None``."""
    dynamic = {name: props for name, props in split.items() if _is_dynamic(props)}
    if not dynamic:
        return None
    names: list[str] = []
    traits: list[tuple[str, ...]] = []
    values: list[tuple[RateOps, ...]] = []
    codes = np.full((len(dynamic), int(dest_idx.size)), -1, dtype=np.int32)
    for row, (name, proportions) in enumerate(dynamic.items()):
        prop = pmap.get_property(name)
        local = tuple(proportions)
        names.append(name)
        traits.append(local)
        values.append(tuple(as_rate(proportions[trait]) for trait in local))
        lookup = np.full(len(prop.traits), -1, dtype=np.int32)
        for position, trait in enumerate(local):
            lookup[prop._index[trait]] = position
        col = np.asarray(pmap.column(prop)[dest_idx], dtype=np.int32)
        codes[row] = np.where(col != _NA, lookup[np.where(col != _NA, col, 0)], -1)
    group = np.asarray(group, dtype=np.int32)
    n_groups = int(group.max()) + 1 if group.size else 0
    counts = np.bincount(group, minlength=n_groups)
    return DynamicSplit(
        names=tuple(names),
        traits=tuple(traits),
        values=tuple(values),
        codes=codes,
        group=group,
        n_groups=n_groups,
        singleton=counts[group] == 1,
    )


def _freeze_int32(values: NDArray[np.int32]) -> NDArray[np.int32]:
    array = np.ascontiguousarray(values, dtype=np.int32)
    array.flags.writeable = False
    return array


def _freeze_float64(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def selector_values(sel: Selector) -> frozenset[str]:
    """Property names that bind pairing: ``Trait`` and ``IsIn`` leaves only."""
    match sel:
        case Trait() as trait:
            return frozenset({trait.property})
        case IsIn(property=name):
            return frozenset({name})
        case Present() | Absent() | Everything() | Nothing():
            return frozenset()
        case And(left=left, right=right) | Or(left=left, right=right):
            return selector_values(left) | selector_values(right)
        case Not(inner=inner):
            return selector_values(inner)
        case Source() | Dest():
            raise TypeError("selector_values expects a compartment selector, not Source()/Dest().")
        case _:
            raise TypeError(f"Unsupported selector {type(sel).__name__}.")


def selector_properties(sel: Selector) -> frozenset[str]:
    """Every property name mentioned in ``sel``, including ``Present``/``Absent``."""
    match sel:
        case Trait() as trait:
            return frozenset({trait.property})
        case IsIn(property=name) | Present(property=name) | Absent(property=name):
            return frozenset({name})
        case Everything() | Nothing():
            return frozenset()
        case And(left=left, right=right) | Or(left=left, right=right):
            return selector_properties(left) | selector_properties(right)
        case Not(inner=inner):
            return selector_properties(inner)
        case Source() | Dest():
            raise TypeError(
                "selector_properties expects a compartment selector, not Source()/Dest()."
            )
        case _:
            raise TypeError(f"Unsupported selector {type(sel).__name__}.")


def _property_present(pmap: PropertyMap, indices: NDArray[np.int32], col: int) -> bool:
    if indices.size == 0:
        return False
    return bool(np.any(pmap.codes[indices, col] != _NA))


def _normalize_split(split: SplitSpec | NormalizedSplit | None) -> NormalizedSplit | None:
    if split is None:
        return None
    out: NormalizedSplit = {}
    for key, proportions in split.items():
        name = key.name if isinstance(key, Property) else str(key)
        if name in out:
            raise ValueError(
                f"split= names property {name!r} more than once "
                f"(a Property and its name, or two properties with the same name)."
            )
        out[name] = {
            str(trait): value if isinstance(value, RateOps) else float(cast(float, value))
            for trait, value in proportions.items()
        }
    return out


def _validate_split(
    pmap: PropertyMap,
    dest_idx: NDArray[np.int32],
    dest_only: tuple[Property, ...],
    split: NormalizedSplit | None,
) -> NormalizedSplit:
    if not split:
        return {}
    dest_only_by_name = {prop.name: prop for prop in dest_only}
    resolved: NormalizedSplit = {}
    for name, proportions in split.items():
        if name not in dest_only_by_name:
            raise ValueError(
                f"split key {name!r} is not a dest-only property. "
                f"Dest-only: {sorted(dest_only_by_name)}."
            )
        prop = dest_only_by_name[name]
        col = pmap.codes[dest_idx, pmap.column_index(prop)]
        appearing = {prop.traits[int(code)] for code in col if code != _NA}
        named = set(proportions)
        if named != appearing:
            raise ValueError(
                f"split[{name!r}] must name every trait that appears on dest "
                f"{sorted(appearing)}; got {sorted(named)}."
            )
        if _is_dynamic(proportions):
            # Rate-valued proportions are only known at run time, where they are
            # renormalised within each group (see DynamicSplit).
            resolved[name] = dict(proportions)
            continue
        numbers = {trait: float(cast(float, value)) for trait, value in proportions.items()}
        if any(value < 0.0 for value in numbers.values()):
            raise ValueError(f"split[{name!r}] proportions must be >= 0.")
        total = sum(numbers.values())
        if abs(total - 1.0) > _SPLIT_TOL:
            raise ValueError(f"split[{name!r}] proportions must sum to 1, got {total}.")
        resolved[name] = dict(numbers)
    return resolved


def _group_weights(
    pmap: PropertyMap,
    dest_members: NDArray[np.int32],
    dest_only: tuple[Property, ...],
    split: Mapping[str, Mapping[str, float]],
) -> NDArray[np.float64]:
    """Return dest weights that sum to 1 for one free-key group (static proportions only)."""
    n = int(dest_members.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    for prop in dest_only:
        proportions = split.get(prop.name)
        if proportions is None:
            continue
        col = np.asarray(pmap.column(prop)[dest_members], dtype=np.int32)
        lookup = np.ones(len(prop.traits), dtype=np.float64)
        for trait, value in proportions.items():
            lookup[prop._index[trait]] = value
        present = col != _NA
        safe = np.where(present, col, 0)
        weights = np.where(present, weights * lookup[safe], weights)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Dest-split weights summed to 0; check split proportions.")
    return weights / total


def _dest_weights_by_group(
    pmap: PropertyMap,
    dest_idx: NDArray[np.int32],
    dest_inv: NDArray[np.int32],
    dest_counts: NDArray[np.intp],
    dest_only: tuple[Property, ...],
    split: Mapping[str, Mapping[str, float]],
) -> NDArray[np.float64]:
    """Per-destination weights, equal-split or ``split``, renormalised within each group."""
    n = int(dest_idx.size)
    n_unique = int(dest_counts.size)
    weights = np.ones(n, dtype=np.float64)
    multi = dest_counts[dest_inv] > 1
    for prop in dest_only:
        proportions = split.get(prop.name)
        if proportions is None:
            continue
        col = np.asarray(pmap.column(prop)[dest_idx], dtype=np.int32)
        lookup = np.ones(len(prop.traits), dtype=np.float64)
        for trait, value in proportions.items():
            lookup[prop._index[trait]] = value
        present = col != _NA
        safe = np.where(present, col, 0)
        factor = np.where(present, lookup[safe], 1.0)
        weights = np.where(multi, weights * factor, weights)
    totals = np.zeros(n_unique, dtype=np.float64)
    np.add.at(totals, dest_inv, np.where(multi, weights, 0.0))
    group_total = totals[dest_inv]
    if np.any(multi & (group_total <= 0.0)):
        raise ValueError("Dest-split weights summed to 0; check split proportions.")
    safe_total = np.where(group_total > 0.0, group_total, 1.0)
    return np.where(multi, weights / safe_total, 1.0)


def check_strict_pairing(edge_map: EdgeMap, bound: frozenset[str]) -> None:
    """Raise if an unbound property would move people along an edge."""
    for prop in edge_map.pmap.properties:
        if prop.name in bound:
            continue
        if np.any(edge_map.moves_mask(prop)):
            raise ValueError(
                f"strict_pairing: property {prop.name!r} is not bound by Trait/IsIn "
                f"or extra_bound but this flow moves people along it."
            )


@dataclass(frozen=True, slots=True)
class EdgeArrays:
    """Concrete source/dest index pairs plus conservation weights and join roles."""

    src_idx: NDArray[np.int32]
    dest_idx: NDArray[np.int32]
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]
    roles: EdgeRoles
    edge_map: EdgeMap
    dynamic_split: DynamicSplit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_idx", _freeze_int32(self.src_idx))
        object.__setattr__(self, "dest_idx", _freeze_int32(self.dest_idx))
        object.__setattr__(self, "weight", _freeze_float64(self.weight))
        object.__setattr__(self, "scale", _freeze_float64(self.scale))

    @property
    def n_edges(self) -> int:
        return int(self.src_idx.size)


def _expand_groups(
    pmap: PropertyMap,
    src_idx: NDArray[np.int32],
    dest_idx: NDArray[np.int32],
    free_cols: tuple[int, ...],
    dest_only: tuple[Property, ...],
    split: Mapping[str, Mapping[str, float]],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    n_src = int(src_idx.size)
    n_dest = int(dest_idx.size)
    if free_cols:
        src_codes = np.asarray(pmap.codes[src_idx][:, list(free_cols)], dtype=np.int16)
        dest_codes = np.asarray(pmap.codes[dest_idx][:, list(free_cols)], dtype=np.int16)
    else:
        src_codes = np.zeros((n_src, 0), dtype=np.int16)
        dest_codes = np.zeros((n_dest, 0), dtype=np.int16)

    stacked = np.concatenate([src_codes, dest_codes], axis=0)
    _unique, inverse = np.unique(stacked, axis=0, return_inverse=True)
    src_inv = inverse[:n_src].astype(np.int32, copy=False)
    dest_inv = inverse[n_src:].astype(np.int32, copy=False)
    n_unique = int(_unique.shape[0])
    dest_counts = np.bincount(dest_inv, minlength=n_unique)
    src_counts = np.bincount(src_inv, minlength=n_unique)

    missing = (src_counts > 0) & (dest_counts == 0)
    if np.any(missing):
        bad_group = int(np.flatnonzero(missing)[0])
        bad_src = int(src_idx[src_inv == bad_group][0])
        free_names = [pmap.properties[c].name for c in free_cols]
        raise ValueError(
            f"No destination match for source {pmap.label(bad_src)!r} on free properties "
            f"{free_names}."
        )

    dest_weights = _dest_weights_by_group(pmap, dest_idx, dest_inv, dest_counts, dest_only, split)
    src_repeats = dest_counts[src_inv].astype(np.int64, copy=False)
    n_edges = int(src_repeats.sum())
    out_src = np.repeat(src_idx, src_repeats).astype(np.int32, copy=False)

    dest_order = np.argsort(dest_inv, kind="stable")
    dest_sorted = dest_idx[dest_order]
    weight_sorted = dest_weights[dest_order]
    dest_starts = np.zeros(n_unique, dtype=np.int64)
    dest_starts[1:] = np.cumsum(dest_counts[:-1])
    starts = np.zeros(n_src, dtype=np.int64)
    if n_src:
        starts[1:] = np.cumsum(src_repeats[:-1])
    inner = np.arange(n_edges, dtype=np.int64) - np.repeat(starts, src_repeats)
    dest_pos = np.repeat(dest_starts[src_inv], src_repeats) + inner
    out_dest = dest_sorted[dest_pos].astype(np.int32, copy=False)
    out_weight = weight_sorted[dest_pos].astype(np.float64, copy=False)
    return out_src, out_dest, out_weight


def identity_join(
    pmap: PropertyMap,
    source: Selector,
    dest: Selector,
    *,
    extra_bound: frozenset[str] = frozenset(),
    split: SplitSpec | NormalizedSplit | None = None,
    scale: float = 1.0,
    strict_pairing: bool = True,
) -> EdgeArrays:
    """Pair source and dest compartments that agree on every free property.

    Binding names are :func:`selector_values` (``Trait``/``IsIn``) plus
    ``extra_bound``. ``Present``/``Absent`` filter the selected sets but do not
    bind. Properties mentioned only that way are not join keys; the resulting
    cartesian product is rejected by ``strict_pairing`` if it would move people.
    """
    src_idx = pmap.select(source)
    dest_idx = pmap.select(dest)
    if src_idx.size == 0:
        raise ValueError("Source selector matched no compartments.")
    if dest_idx.size == 0:
        raise ValueError("Destination selector matched no compartments.")

    value_bound = selector_values(source) | selector_values(dest) | extra_bound
    mentioned = selector_properties(source) | selector_properties(dest)
    free_cols: list[int] = []
    dest_only: list[Property] = []
    src_only: set[str] = set()
    paired: set[str] = set()
    for col, prop in enumerate(pmap.properties):
        src_has = _property_present(pmap, src_idx, col)
        dest_has = _property_present(pmap, dest_idx, col)
        if src_has and dest_has:
            paired.add(prop.name)
        if prop.name in value_bound:
            continue
        if src_has and dest_has:
            # Present/Absent name a filter, not a pairing value — not a join key.
            if prop.name in mentioned:
                continue
            free_cols.append(col)
        elif dest_has and not src_has:
            dest_only.append(prop)
        elif src_has and not dest_has:
            src_only.add(prop.name)

    dest_only_t = tuple(dest_only)
    split_n = _validate_split(pmap, dest_idx, dest_only_t, _normalize_split(split))
    out_src, out_dest, out_weight = _expand_groups(
        pmap, src_idx, dest_idx, tuple(free_cols), dest_only_t, _static_split(split_n)
    )
    # Each source compartment's edges form one renormalisation group.
    _src_unique, src_group = np.unique(out_src, return_inverse=True)
    dynamic_split = _build_dynamic_split(pmap, split_n, out_dest, src_group.astype(np.int32))
    n = int(out_src.size)
    roles = EdgeRoles(
        bound=value_bound,
        free=frozenset(pmap.properties[c].name for c in free_cols),
        dest_only=frozenset(prop.name for prop in dest_only_t),
        src_only=frozenset(src_only),
        paired=frozenset(paired),
    )
    edge_map = EdgeMap.from_indices(pmap, out_src, out_dest, roles)
    if strict_pairing:
        check_strict_pairing(edge_map, roles.bound)
    return EdgeArrays(
        src_idx=out_src,
        dest_idx=out_dest,
        weight=out_weight,
        scale=np.full(n, float(scale), dtype=np.float64),
        roles=roles,
        edge_map=edge_map,
        dynamic_split=dynamic_split,
    )


def _merge_roles(parts: Sequence[EdgeRoles]) -> EdgeRoles:
    def _union(getter: str) -> frozenset[str]:
        names: set[str] = set()
        for roles in parts:
            names |= getattr(roles, getter)
        return frozenset(names)

    return EdgeRoles(
        bound=_union("bound"),
        free=_union("free"),
        dest_only=_union("dest_only"),
        src_only=_union("src_only"),
        paired=_union("paired"),
    )


def _concat_edges(
    pmap: PropertyMap,
    parts: Sequence[EdgeArrays],
    *,
    strict_pairing: bool,
) -> EdgeArrays:
    if not parts:
        raise ValueError("Pairing produced no edges.")
    src = np.concatenate([part.src_idx for part in parts])
    dest = np.concatenate([part.dest_idx for part in parts])
    weight = np.concatenate([part.weight for part in parts])
    scale = np.concatenate([part.scale for part in parts])
    roles = _merge_roles([part.roles for part in parts])
    edge_map = EdgeMap.from_indices(pmap, src, dest, roles)
    if strict_pairing:
        check_strict_pairing(edge_map, roles.bound)
    return EdgeArrays(
        src_idx=src,
        dest_idx=dest,
        weight=weight,
        scale=scale,
        roles=roles,
        edge_map=edge_map,
        dynamic_split=_concat_dynamic_splits(parts),
    )


def _concat_dynamic_splits(parts: Sequence[EdgeArrays]) -> DynamicSplit | None:
    """Join per-part dynamic splits; groups stay per part (one pairing part, one source)."""
    dyns = [part.dynamic_split for part in parts]
    if all(dyn is None for dyn in dyns):
        return None
    first = next(dyn for dyn in dyns if dyn is not None)
    if any(dyn is None or dyn.names != first.names for dyn in dyns):
        raise ValueError("Pairing parts disagree on rate-valued split properties.")
    present = [cast(DynamicSplit, dyn) for dyn in dyns]
    # Parts may name different trait subsets; re-index each part's codes onto
    # the union of traits per property, keeping the shared expressions.
    traits: list[tuple[str, ...]] = []
    values: list[tuple[RateOps, ...]] = []
    remaps: list[list[NDArray[np.int32]]] = [[] for _ in present]
    for row in range(len(first.names)):
        union: dict[str, RateOps] = {}
        for dyn in present:
            for trait, value in zip(dyn.traits[row], dyn.values[row], strict=True):
                union.setdefault(trait, value)
        order = tuple(union)
        traits.append(order)
        values.append(tuple(union[trait] for trait in order))
        for i, dyn in enumerate(present):
            remaps[i].append(np.array([order.index(t) for t in dyn.traits[row]], dtype=np.int32))
    code_parts: list[NDArray[np.int32]] = []
    group_parts: list[NDArray[np.int32]] = []
    offset = 0
    for i, dyn in enumerate(present):
        codes = np.empty_like(dyn.codes)
        for row in range(len(first.names)):
            local = dyn.codes[row]
            codes[row] = np.where(local >= 0, remaps[i][row][np.maximum(local, 0)], -1)
        code_parts.append(codes)
        group_parts.append(dyn.group + offset)
        offset += dyn.n_groups
    return DynamicSplit(
        names=first.names,
        traits=tuple(traits),
        values=tuple(values),
        codes=np.concatenate(code_parts, axis=1),
        group=np.concatenate(group_parts),
        n_groups=offset,
        singleton=np.concatenate([dyn.singleton for dyn in present]),
    )


def _chain_rate(rate: object) -> float | RateOps:
    """A ``TraitChain`` rate: a plain number as ``float``, else a rate expression."""
    if isinstance(rate, (bool, np.bool_)):
        raise TypeError(f"TraitChain rate must be a number or a rate expression, got {rate!r}.")
    if isinstance(rate, (int, float, np.integer, np.floating)):
        return float(rate)
    if isinstance(rate, RateOps):
        if isinstance(rate, Const):
            return float(rate.value)
        return rate
    raise TypeError(
        f"TraitChain rate must be a number or a rate expression, got {type(rate).__name__}."
    )


@dataclass(frozen=True, slots=True)
class TraitChain:
    """Explicit ``(source trait, dest trait)`` pairs on one property.

    One named flow carries every pair: each pair moves people from the source
    trait to the dest trait of ``property`` while every other property keeps
    its trait. The pairs are any list within that property, not only a chain.
    Ageing is the common case (:meth:`from_breakpoints`), but paired moves
    between states work the same way::

        CLINICAL = (("subclin_inf", "clin_inf"), ("subclin_lowinf", "clin_lowinf"))
        TransitionFlow("clinical_progression", state.present(), state.present(),
                       Param("clinical_progression_rate"), pairing=TraitChain(state, CLINICAL))

    The reverse moves are the same pairs swapped (``(dest, src)``). Two pairs
    may share a source, which splits one source between two destinations.

    ``rates`` are optional per-pair multipliers on top of the flow ``rate``.
    Each is a number (an ageing band's ``1/5``) or a rate expression
    (``Param("p")``, ``1 - Param("p")``, a function of ``Time()``), so a split
    needs no ``Dest`` adjustments::

        p = Param("progression_prop_infectious")
        TransitionFlow("progression", state.present(), state.present(), Param("rate"),
                       pairing=TraitChain(state, (("incipient", "subclin_lowinf"),
                                                  ("incipient", "subclin_inf")),
                                          rates=(1 - p, p)))

    A number is stored on the edge ``scale``, the same place ``TraitMatrix``
    puts its entries. An expression becomes, for that pair's edges, a
    ``Multiply(rate, where=Source(property[src]) & Dest(property[dest]))``
    applied before any other adjustment (as a ``scale`` is), so an
    ``Overwrite`` still replaces it. For ageing, leave the last trait out of
    ``pairs`` so it has no outgoing edge.

    Raises:
        ValueError: if ``pairs`` is empty, or ``rates`` has a different length.
        KeyError: if a pair names a trait ``property`` does not have.
        TypeError: if a rate is neither a number nor a rate expression.
    """

    property: Property
    pairs: tuple[tuple[str, str], ...]
    rates: tuple[float | RateOps, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple((str(a), str(b)) for a, b in self.pairs))
        if not self.pairs:
            raise ValueError("TraitChain requires at least one trait pair.")
        known = set(self.property.traits)
        for src, dest in self.pairs:
            if src not in known or dest not in known:
                raise KeyError(
                    f"Unknown trait in chain {(src, dest)} for "
                    f"{self.property.name!r}. Known: {list(self.property.traits)}"
                )
        if self.rates is not None:
            if len(self.rates) != len(self.pairs):
                raise ValueError(
                    f"TraitChain.rates length {len(self.rates)} does not match "
                    f"{len(self.pairs)} pairs."
                )
            object.__setattr__(self, "rates", tuple(_chain_rate(r) for r in self.rates))

    def static_scale(self, index: int) -> float:
        """The edge ``scale`` of pair ``index``: its number, or 1 for a rate expression."""
        if self.rates is None:
            return 1.0
        rate = self.rates[index]
        return rate if isinstance(rate, float) else 1.0

    def rate_adjustments(self) -> tuple[tuple[RateOps, Selector], ...]:
        """``(rate, where)`` for each pair whose rate is an expression, in pair order."""
        if self.rates is None:
            return ()
        prop = self.property
        return tuple(
            (rate, Source(prop[src]) & Dest(prop[dest]))
            for rate, (src, dest) in zip(self.rates, self.pairs, strict=True)
            if not isinstance(rate, float)
        )

    @classmethod
    def from_breakpoints(cls, age: Property, *, unit: float = 1.0) -> TraitChain:
        """Build an ageing chain from numeric lower-bound trait names.

        Each trait must parse as a float lower bound (``\"0\"``, ``\"5\"``,
        ``\"15\"``). Consecutive traits form pairs; rates are
        ``1 / (width * unit)``. The last band has no outgoing edge. Raise if a
        trait does not parse, bounds are not strictly increasing, there are
        fewer than two traits, or ``unit`` is not positive.
        """
        if unit <= 0.0:
            raise ValueError(f"unit must be positive, got {unit}.")
        if len(age.traits) < 2:
            raise ValueError(
                f"Property {age.name!r} needs at least two traits for "
                f"from_breakpoints, got {list(age.traits)}."
            )
        bounds: list[float] = []
        for trait in age.traits:
            try:
                bound = float(trait)
            except ValueError as exc:
                raise ValueError(
                    f"Trait {trait!r} of property {age.name!r} is not a "
                    f"numeric lower bound (expected names like '0', '5', '15')."
                ) from exc
            bounds.append(bound)
        for index in range(1, len(bounds)):
            if bounds[index] <= bounds[index - 1]:
                raise ValueError(
                    f"Age breakpoints for {age.name!r} must be strictly "
                    f"increasing; got {age.traits[index - 1]!r}={bounds[index - 1]} "
                    f"then {age.traits[index]!r}={bounds[index]}."
                )
        pairs = tuple(
            (age.traits[index], age.traits[index + 1]) for index in range(len(bounds) - 1)
        )
        rates = tuple(
            1.0 / ((bounds[index + 1] - bounds[index]) * unit) for index in range(len(bounds) - 1)
        )
        return cls(age, pairs, rates=rates)


@dataclass(frozen=True, slots=True)
class TraitMatrix:
    """Dense or sparse transition matrix over one property (dest × source)."""

    property: Property
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.property.traits)
        matrix = np.ascontiguousarray(self.matrix, dtype=np.float64)
        if matrix.shape != (n, n):
            raise ValueError(
                f"TraitMatrix for {self.property.name!r} must have shape "
                f"({n}, {n}) dest×source, got {matrix.shape}."
            )
        matrix.flags.writeable = False
        object.__setattr__(self, "matrix", matrix)


def join_with_pairing(
    pmap: PropertyMap,
    source: Selector,
    dest: Selector,
    pairing: TraitChain | TraitMatrix | None,
    split: SplitSpec | NormalizedSplit | None,
    *,
    strict_pairing: bool = True,
) -> EdgeArrays:
    """Identity-join, or a chain/matrix of identity-joins on one property."""
    if pairing is None:
        return identity_join(pmap, source, dest, split=split, strict_pairing=strict_pairing)
    extra = frozenset({pairing.property.name})
    if isinstance(pairing, TraitChain):
        parts = [
            identity_join(
                pmap,
                source & pairing.property[src],
                dest & pairing.property[dest_name],
                extra_bound=extra,
                split=split,
                scale=pairing.static_scale(index),
                strict_pairing=strict_pairing,
            )
            for index, (src, dest_name) in enumerate(pairing.pairs)
        ]
        return _concat_edges(pmap, parts, strict_pairing=strict_pairing)
    parts = []
    traits = pairing.property.traits
    for dest_code, dest_name in enumerate(traits):
        for src_code, src_name in enumerate(traits):
            value = float(pairing.matrix[dest_code, src_code])
            if value == 0.0:
                continue
            parts.append(
                identity_join(
                    pmap,
                    source & pairing.property[src_name],
                    dest & pairing.property[dest_name],
                    extra_bound=extra,
                    split=split,
                    scale=value,
                    strict_pairing=strict_pairing,
                )
            )
    return _concat_edges(pmap, parts, strict_pairing=strict_pairing)
