"""Force of infection as a GroupedRate-producing rate node."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from summer4.enums import coerce_strenum
from summer4.epi.mixing import MixingMatrix
from summer4.flows.rates import (
    Capture,
    RateOps,
    as_rate,
    register_rate_eval,
)
from summer4.properties import Property, Trait
from summer4.propertymap import PropertyMap
from summer4.selectors import Selector, SelectorOps


class FOIKind(StrEnum):
    """Built-in force-of-infection shapes.

    Prefer these members at call sites. Bare strings such as ``"frequency"`` are
    still accepted and coerced. Custom shapes use a callable
    ``kind(infectious, denominator) -> GroupedRate`` instead.
    """

    FREQUENCY = "frequency"
    DENSITY = "density"
    GENERALISED = "generalised"


class InfectiousnessNormalize(StrEnum):
    """How per-group infectiousness weights are re-scaled."""

    POPULATION = "population"
    MEAN = "mean"


type FOIKindArg = FOIKind | str | Callable[..., Any]
type InfectiousnessMap = Mapping[Trait | str, object]
type InfectiousnessArg = InfectiousnessMap | Sequence[tuple[Selector, object]] | None
type SusceptibilityMap = Mapping[Trait | str, object]
type SusceptibilityArg = SusceptibilityMap | Sequence[tuple[Selector, object]] | None
type InfectiousnessNormalizeArg = InfectiousnessNormalize | str | None
type CompartmentWeights = tuple[tuple[Selector, RateOps], ...]


def coerce_compartment_weights(
    weights: InfectiousnessArg | SusceptibilityArg,
    *,
    group_by: Property,
    what: str = "infectiousness",
) -> CompartmentWeights | None:
    """Normalise trait-keyed or selector-keyed compartment weights.

    A mapping whose keys are traits of ``group_by`` (or those trait names) is
    sugar for ``(group_by[trait], weight)`` pairs. A sequence of
    ``(selector, weight)`` pairs is kept as given and may name any selector,
    including one that crosses properties (compartment and age). Both forms
    store the same pair tuple, sorted by selector digest, so a trait map and
    the pairs it expands to compare and digest equally.

    The pair list is unrolled when the weights are applied. Keep it small.

    :class:`ForceOfInfection` coerces both ``infectiousness=`` and
    ``susceptibility=`` with this function, passing ``what`` so error messages
    name the argument at fault.
    """
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        raw = _pairs_from_trait_map(weights, group_by=group_by, what=what)
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        raw = _pairs_from_sequence(weights, what=what)
    else:
        raise TypeError(
            f"{what} must be a mapping of {group_by.name!r} traits, or a sequence "
            f"of (selector, weight) pairs, not {type(weights).__name__}."
        )
    if not raw:
        return None
    from summer4.flows.rates import _rate_bytes, _selector_bytes

    return tuple(sorted(raw, key=lambda pair: _selector_bytes(pair[0]) + _rate_bytes(pair[1])))


def _pairs_from_trait_map(
    weights: Mapping[Trait | str, object],
    *,
    group_by: Property,
    what: str,
) -> list[tuple[Selector, RateOps]]:
    pairs: list[tuple[Selector, RateOps]] = []
    for key, value in weights.items():
        if isinstance(key, str):
            selector: Selector = group_by[key]
        elif isinstance(key, Trait):
            if key.property != group_by.name:
                raise ValueError(
                    f"{what.capitalize()} key property {key.property!r} does not "
                    f"match ForceOfInfection group_by {group_by.name!r}."
                )
            selector = key
        else:
            raise TypeError(
                f"{what} mapping keys must be traits of group_by {group_by.name!r} "
                f"or their names, not {type(key).__name__}. Pass a sequence of "
                "(selector, weight) pairs for any other selector."
            )
        pairs.append((selector, as_rate(value)))
    return pairs


def _pairs_from_sequence(
    weights: Sequence[tuple[Selector, object]],
    *,
    what: str,
) -> list[tuple[Selector, RateOps]]:
    if (
        isinstance(weights, tuple)
        and len(weights) == 2
        and isinstance(weights[0], SelectorOps)
        and not isinstance(weights[1], tuple)
    ):
        raise TypeError(
            f"{what} sequence must be (selector, weight) pairs; "
            "wrap a single pair in a list or a one-element tuple."
        )
    pairs: list[tuple[Selector, RateOps]] = []
    for item in weights:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"{what} sequence items must be (selector, weight) pairs, "
                f"not {type(item).__name__}."
            )
        selector, value = item
        if not isinstance(selector, SelectorOps):
            raise TypeError(
                f"{what} pair selector must be a Selector, not {type(selector).__name__}."
            )
        pairs.append((selector, as_rate(value)))
    return pairs


def apply_compartment_weights(
    data: Any,
    pmap: PropertyMap,
    pairs: CompartmentWeights | None,
    eval_child: Callable[[RateOps], Any],
    *,
    what: str,
) -> Any:
    """Multiply compartments by the product of the weights whose selectors match.

    ``data`` is aligned with ``pmap`` on its last axis. ``None`` or an empty
    pair list returns ``data`` unchanged. A selector that matches no compartment
    raises: a weight that can never apply is an error, not a silent no-op.

    ``eval_child`` evaluates one weight (a rate expression) to a scalar or an
    array that broadcasts against ``data``. Inside a rate evaluator it is the
    ``eval_child`` the evaluator was handed. :class:`ForceOfInfection` uses this
    function for infectiousness on the source side, before the group sum, and
    for susceptibility on the recipient side, after mixing. The loop is over
    pairs, not compartments: each pair is one masked multiply over the whole
    array.
    """
    if not pairs:
        return data
    import jax.numpy as jnp

    weighted = data
    for selector, rate in pairs:
        if pmap.select(selector).size == 0:
            raise ValueError(f"{what} selector {selector!r} matches no compartment on this map.")
        value = jnp.asarray(eval_child(rate))
        mask = jnp.asarray(pmap.mask(selector))
        weighted = jnp.where(mask, weighted * value, weighted)
    return weighted


def split_susceptibility(
    pairs: CompartmentWeights | None,
    *,
    group_by: Property,
) -> tuple[CompartmentWeights, CompartmentWeights]:
    """Split susceptibility pairs into per-group and per-compartment weights.

    A pair whose selector is a single trait of ``group_by`` (everything a trait
    map produces) scales the force of infection of that whole group, so it can
    be folded into the per-group rate. Every other pair — a trait of another
    property, or a combination such as ``state["L"] & age["0"]`` — varies
    within a group and is applied to the compartment-aligned rate instead.
    Returns ``(per_group, per_compartment)``; either may be empty.
    """
    if not pairs:
        return (), ()
    per_group: list[tuple[Selector, RateOps]] = []
    per_compartment: list[tuple[Selector, RateOps]] = []
    for selector, value in pairs:
        if isinstance(selector, Trait) and selector.property == group_by.name:
            per_group.append((selector, value))
        else:
            per_compartment.append((selector, value))
    return tuple(per_group), tuple(per_compartment)


def scale_infectious_pool(
    infectious: Any,
    *,
    weights: Any,
    population: Any,
    mode: InfectiousnessNormalize | None,
    pmap: PropertyMap,
    group_by: Property,
) -> Any:
    """Rescale an already-weighted infectious pool by ``normalize_infectiousness``.

    ``weights`` and ``population`` are per compartment. ``infectious`` is the
    group sum of ``weights * y`` over the infectious selector.

    ``population`` divides by the population-weighted mean of the compartment
    weights, using this same population. ``mean`` divides by the unweighted mean
    of the per-group effective weights, where a group's effective weight is the
    unweighted mean of its compartment weights (an empty group counts as 1).
    When every compartment in a group shares one weight, both match the
    historical per-trait formulas. Not for susceptibility.
    """
    from summer4.flows.compiled import GroupedRate

    if mode is None:
        return infectious
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData

    data = infectious.data
    if mode is InfectiousnessNormalize.POPULATION:
        total = jnp.sum(population)
        mean_w = jnp.sum(weights * population) / jnp.maximum(total, 1e-12)
        return GroupedRate(data / mean_w, infectious.properties)
    if mode is InfectiousnessNormalize.MEAN:
        count = PropertyData(pmap, jnp.ones(pmap.size, dtype=weights.dtype)).sum_over(group_by).data
        sum_w = PropertyData(pmap, weights).sum_over(group_by).data
        effective = jnp.where(count > 0, sum_w / jnp.maximum(count, 1.0), 1.0)
        return GroupedRate(data / jnp.mean(effective), infectious.properties)
    raise ValueError(f"Unknown normalize_infectiousness {mode!r}.")


@dataclass(frozen=True, slots=True)
class ForceOfInfection(RateOps):
    """Per-group force of infection, usable directly as a flow rate.

    Evaluates to a :class:`~summer4.flows.compiled.GroupedRate` over
    ``group_by``, so no ``broadcast_over`` round trip is required.

    :attr:`FOIKind.FREQUENCY` divides by the per-group denominator;
    :attr:`FOIKind.DENSITY` does not. :attr:`FOIKind.GENERALISED` evaluates
    ``infectious / denominator ** exponent`` (requires ``exponent``). Frequency
    is bit-identical to generalised with ``exponent=1``; density to
    ``exponent=0``. A callable ``kind(infectious, denominator) -> GroupedRate``
    is the custom path — it receives no parameters; parameterised shapes use
    :attr:`FOIKind.GENERALISED` or rate-tree arithmetic on ``Reduce``.

    ``where`` polarity on the infectious selector is KEEP (via :class:`Reduce`).

    ``infectiousness`` weights the source of transmission. A mapping of
    ``group_by`` traits is sugar for ``(group_by[trait], weight)`` pairs.
    A sequence of ``(selector, weight)`` pairs is applied multiplicatively
    per compartment before the group sum, so a weight can name a compartment
    and an age together ("subclinical cases are less infectious, and nobody
    under 15 transmits"). Compartments no pair matches keep weight 1.
    ``normalize_infectiousness`` then rescales that weighted pool; the default
    is no normalisation. See :func:`coerce_compartment_weights`.

    ``susceptibility`` weights the recipient of transmission and takes the same
    two shapes. It multiplies the force of infection *after* the mixing
    product, so with infectiousness weights :math:`w`, susceptibility weights
    :math:`s`, mixing matrix :math:`K`, contact rate :math:`c` and kind
    :math:`f`,

    .. math:: \\lambda_a = s_a \\, c \\sum_b K_{ab} \\, f(w_b I_b, N_b).

    That is what scaling row :math:`a` of :math:`K` does, and what
    ``Multiply(s_a, where=group_by[a])`` on the infection flow does; this keyword
    keeps it on the force of infection, beside infectiousness, and out of the
    contact structure. **Susceptibility is not normalised**, and there is no
    ``normalize_susceptibility=``: infectiousness is normalised so a
    population-weighted mean of 1 keeps the contact rate interpretable, and no
    such invariant applies to the recipient side. Doubling every susceptibility
    weight doubles the force of infection.

    A pair keyed on a single trait of ``group_by`` (every entry of a trait map)
    scales that group's :math:`\\lambda_a`; the rate stays a per-group
    :class:`~summer4.flows.compiled.GroupedRate`, and the value saved under
    ``name`` includes the weight. A pair whose selector names anything else — a
    compartment, or a compartment and a group together ("recovered people are
    0.3 as susceptible") — varies within a group, so the rate becomes
    compartment-aligned: each compartment's :math:`\\lambda_a` times the product
    of the per-compartment weights that match it (unmatched compartments keep
    weight 1). The value saved under ``name`` stays per group and does **not**
    include those per-compartment weights: it is the force of infection on a
    group member whose per-compartment susceptibility is 1. :meth:`captured`
    is refused in that case, because the rate is no longer per group. Like
    infectiousness, susceptibility groups on the one ``group_by`` property;
    multi-property mixing is not available.
    """

    name: str
    infectious: Selector
    group_by: Property
    mixing: MixingMatrix | None = None
    kind: FOIKind | Callable[..., Any] = FOIKind.FREQUENCY
    contact_rate: RateOps = None  # type: ignore[assignment]
    denominator: Selector | None = None
    infectiousness: CompartmentWeights | None = None
    normalize_infectiousness: InfectiousnessNormalize | None = None
    exponent: RateOps | None = None
    susceptibility: CompartmentWeights | None = None

    def __init__(
        self,
        name: str,
        *,
        infectious: Selector,
        group_by: Property,
        mixing: MixingMatrix | None = None,
        kind: FOIKindArg = FOIKind.FREQUENCY,
        contact_rate: object = 1.0,
        denominator: Selector | None = None,
        infectiousness: InfectiousnessArg = None,
        normalize_infectiousness: InfectiousnessNormalizeArg = None,
        exponent: object | None = None,
        susceptibility: SusceptibilityArg = None,
    ) -> None:
        if callable(kind):
            resolved_kind: FOIKind | Callable[..., Any] = kind
        else:
            resolved_kind = coerce_strenum(FOIKind, kind, what="ForceOfInfection kind")
        if resolved_kind is FOIKind.GENERALISED:
            if exponent is None:
                raise ValueError(
                    "ForceOfInfection kind=FOIKind.GENERALISED requires exponent= "
                    "(a float, Param, or other rate expression)."
                )
        elif exponent is not None:
            raise ValueError(
                "ForceOfInfection exponent= is only valid with "
                f"kind=FOIKind.GENERALISED, not kind={resolved_kind!r}."
            )
        resolved_norm: InfectiousnessNormalize | None
        if normalize_infectiousness is None:
            resolved_norm = None
        else:
            resolved_norm = coerce_strenum(
                InfectiousnessNormalize,
                normalize_infectiousness,
                what="normalize_infectiousness",
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "infectious", infectious)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "contact_rate", as_rate(contact_rate))
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(
            self,
            "infectiousness",
            coerce_compartment_weights(infectiousness, group_by=group_by, what="infectiousness"),
        )
        object.__setattr__(self, "normalize_infectiousness", resolved_norm)
        object.__setattr__(self, "exponent", None if exponent is None else as_rate(exponent))
        object.__setattr__(
            self,
            "susceptibility",
            coerce_compartment_weights(susceptibility, group_by=group_by, what="susceptibility"),
        )
        if mixing is not None and mixing.prop.name != group_by.name:
            raise ValueError(
                f"MixingMatrix property {mixing.prop.name!r} does not match "
                f"ForceOfInfection group_by {group_by.name!r}."
            )

    def __flow_refs__(self) -> set[str]:
        return set()

    def __field_paths__(self) -> set[tuple[str, ...]]:
        paths = set()
        from summer4.flows.rates import _field_paths

        paths |= _field_paths(self.contact_rate)
        if self.exponent is not None:
            paths |= _field_paths(self.exponent)
        if self.mixing is not None:
            paths |= _field_paths(self.mixing.matrix)
        if self.infectiousness is not None:
            for _selector, value in self.infectiousness:
                paths |= _field_paths(value)
        if self.susceptibility is not None:
            for _selector, value in self.susceptibility:
                paths |= _field_paths(value)
        return paths

    def __rate_bytes__(self) -> bytes:
        from summer4.flows.rates import _rate_bytes, _selector_bytes

        body = b"foi" + self.name.encode() + self.group_by.name.encode()
        body += _selector_bytes(self.infectious)
        body += _rate_bytes(self.contact_rate)
        if isinstance(self.kind, FOIKind):
            body += repr(self.kind.value).encode()
        else:
            body += repr(id(self.kind)).encode()
        body += repr(
            None if self.normalize_infectiousness is None else self.normalize_infectiousness.value
        ).encode()
        if self.exponent is not None:
            body += b"exp" + _rate_bytes(self.exponent)
        if self.denominator is not None:
            body += _selector_bytes(self.denominator)
        if self.mixing is not None:
            body += _rate_bytes(self.mixing.matrix)
            body += self.mixing.normalize.value.encode()
            body += b"1" if self.mixing.check_reciprocal else b"0"
        if self.infectiousness is not None:
            for selector, value in self.infectiousness:
                body += _selector_bytes(selector) + _rate_bytes(value)
        if self.susceptibility is not None:
            body += b"sus"
            for selector, value in self.susceptibility:
                body += _selector_bytes(selector) + _rate_bytes(value)
        return body

    def __capture_meta__(self) -> tuple[str, tuple[Property, ...]]:
        return self.name, (self.group_by,)

    @classmethod
    def per_trait(
        cls,
        strain: Property,
        *,
        name_prefix: str = "infection",
        infectious: Selector,
        group_by: Property,
        **kwargs: Any,
    ) -> tuple[ForceOfInfection, ...]:
        """One named FOI per strain trait, each with ``infectious & strain[s]``."""
        out: list[ForceOfInfection] = []
        for trait_name in strain.traits:
            out.append(
                cls(
                    f"{name_prefix}_{trait_name}",
                    infectious=infectious & strain[trait_name],
                    group_by=group_by,
                    **kwargs,
                )
            )
        return tuple(out)

    @property
    def per_compartment(self) -> bool:
        """True when a susceptibility pair varies within a ``group_by`` group.

        The rate is then compartment-aligned rather than a per-group
        :class:`~summer4.flows.compiled.GroupedRate`. See
        :func:`split_susceptibility`.
        """
        _per_group, per_compartment = split_susceptibility(
            self.susceptibility, group_by=self.group_by
        )
        return bool(per_compartment)

    def captured(self) -> Capture:
        """Wrap this FOI in a :class:`Capture` so it is saveable by name.

        Refused when :attr:`per_compartment` is true: the rate is then aligned
        to compartments, not groups. The force of infection is saved under
        ``name`` regardless; read it with ``GroupedOutput(name)``.
        """
        if self.per_compartment:
            raise ValueError(
                f"ForceOfInfection {self.name!r} has per-compartment susceptibility, so its "
                "rate is not per group and cannot be wrapped in Capture. The per-group force "
                f"of infection is already saved as GroupedOutput({self.name!r})."
            )
        return Capture(self.name, self)


@register_rate_eval(ForceOfInfection)
def _eval_force_of_infection(
    expr: ForceOfInfection,
    *,
    eval_child: Callable[[RateOps], Any],
    derived: Any,
    pmap: Any,
    t: Any,
    y_arr: Any,
    captures: dict[str, Any],
) -> Any:
    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData

    prop = expr.group_by
    pd = PropertyData(pmap, y_arr)
    if expr.denominator is not None:
        n_comp = pd.keep(expr.denominator, 0.0).data
        n_pd = PropertyData(pmap, n_comp).sum_over(prop)
    else:
        n_comp = y_arr
        n_pd = pd.sum_over(prop)
    if expr.infectiousness is not None:
        import jax.numpy as jnp

        compartment_weights = apply_compartment_weights(
            jnp.ones(pmap.size, dtype=jnp.asarray(y_arr).dtype),
            pmap,
            expr.infectiousness,
            eval_child,
            what="infectiousness",
        )
        weighted_y = y_arr * compartment_weights
    else:
        compartment_weights = None
        weighted_y = y_arr
    i_pd = PropertyData(pmap, weighted_y).keep(expr.infectious, 0.0).sum_over(prop)
    i_grp = GroupedRate(i_pd.data, (prop,))
    n_grp = GroupedRate(n_pd.data, (prop,))
    if compartment_weights is not None and expr.normalize_infectiousness is not None:
        i_grp = scale_infectious_pool(
            i_grp,
            weights=compartment_weights,
            population=n_comp,
            mode=expr.normalize_infectiousness,
            pmap=pmap,
            group_by=prop,
        )

    kind = expr.kind
    if callable(kind):
        shedding = kind(i_grp, n_grp)
        if not isinstance(shedding, GroupedRate):
            raise TypeError(
                f"ForceOfInfection kind callable must return GroupedRate, "
                f"got {type(shedding).__name__}."
            )
    elif kind is FOIKind.FREQUENCY:
        shedding = i_grp / n_grp
    elif kind is FOIKind.DENSITY:
        shedding = i_grp
    elif kind is FOIKind.GENERALISED:
        if expr.exponent is None:  # pragma: no cover - validated in __init__
            raise ValueError("kind=FOIKind.GENERALISED requires exponent=.")
        shedding = i_grp / (n_grp ** eval_child(expr.exponent))
    else:
        raise ValueError(f"Unknown ForceOfInfection kind {kind!r}.")

    if expr.mixing is not None:
        mat = expr.mixing.resolved_matrix(derived, eval_child)
        expr.mixing.check_reciprocity(mat, n_grp.data)
        shedding = mat @ shedding

    contact = eval_child(expr.contact_rate)
    result = shedding * contact
    if not isinstance(result, GroupedRate):
        import jax.numpy as jnp

        result = GroupedRate(jnp.asarray(contact) * jnp.asarray(shedding.data), (prop,))
    if expr.susceptibility is None:
        captures[expr.name] = result
        return result
    return _apply_susceptibility(
        result,
        expr.susceptibility,
        name=expr.name,
        group_by=prop,
        pmap=pmap,
        eval_child=eval_child,
        captures=captures,
    )


def _apply_susceptibility(
    force: Any,
    pairs: CompartmentWeights,
    *,
    name: str,
    group_by: Property,
    pmap: PropertyMap,
    eval_child: Callable[[RateOps], Any],
    captures: dict[str, Any],
) -> Any:
    """Weight the recipient side of an evaluated per-group force of infection.

    Per-group pairs scale ``force`` (a ``GroupedRate`` over ``group_by``) and
    are included in the capture. Per-compartment pairs are applied after the
    capture, to the force gathered onto compartments, which is then returned.
    No normalisation, unlike infectiousness.
    """
    import jax.numpy as jnp

    from summer4.flows.compiled import GroupedRate

    for selector, _value in pairs:
        if pmap.select(selector).size == 0:
            raise ValueError(f"susceptibility selector {selector!r} matches no compartment.")
    per_group, per_compartment = split_susceptibility(pairs, group_by=group_by)
    if per_group:
        data = jnp.asarray(force.data)
        group_weights = apply_compartment_weights(
            jnp.ones(len(group_by.traits), dtype=data.dtype),
            PropertyMap.from_property(group_by),
            per_group,
            eval_child,
            what="susceptibility",
        )
        force = GroupedRate(data * group_weights, (group_by,))
    captures[name] = force
    if not per_compartment:
        return force
    # One constant gather from groups onto compartments, then one masked
    # multiply per pair. Compartments without ``group_by`` get NaN: no
    # infection flow may leave them (the per-group rate refuses that too).
    codes = np.asarray(pmap.codes[:, pmap.column_index(group_by)], dtype=np.int32)
    valid = codes >= 0
    per_comp = jnp.take(jnp.asarray(force.data), np.where(valid, codes, 0), axis=-1)
    if not bool(valid.all()):
        per_comp = jnp.where(valid, per_comp, jnp.nan)
    return apply_compartment_weights(
        per_comp, pmap, per_compartment, eval_child, what="susceptibility"
    )
