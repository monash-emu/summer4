"""Force of infection as a GroupedRate-producing rate node."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from summer4.epi.mixing import MixingMatrix
from summer4.flows.rates import (
    Capture,
    RateOps,
    as_rate,
    register_rate_eval,
)
from summer4.properties import Property, Trait
from summer4.selectors import Selector


class FoiKind(StrEnum):
    """Built-in force-of-infection shapes.

    Custom shapes use a callable ``kind(infectious, denominator) -> GroupedRate``
    instead of a member of this enum.
    """

    FREQUENCY = "frequency"
    DENSITY = "density"
    GENERALISED = "generalised"


type FoiKindArg = FoiKind | Callable[..., Any]
type InfectiousnessMap = Mapping[Trait | str, object]
type NormalizeWeights = Literal["population", "mean"] | None


@dataclass(frozen=True, slots=True)
class ForceOfInfection(RateOps):
    """Per-group force of infection, usable directly as a flow rate.

    Evaluates to a :class:`~summer4.flows.compiled.GroupedRate` over
    ``group_by``, so no ``broadcast_over`` round trip is required.

    :attr:`FoiKind.FREQUENCY` divides by the per-group denominator;
    :attr:`FoiKind.DENSITY` does not. :attr:`FoiKind.GENERALISED` evaluates
    ``infectious / denominator ** exponent`` (requires ``exponent``). Frequency
    is bit-identical to generalised with ``exponent=1``; density to
    ``exponent=0``. A callable ``kind(infectious, denominator) -> GroupedRate``
    is the custom path — it receives no parameters; parameterised shapes use
    :attr:`FoiKind.GENERALISED` or rate-tree arithmetic on ``Reduce``.

    ``where`` polarity on the infectious selector is KEEP (via :class:`Reduce`).
    """

    name: str
    infectious: Selector
    group_by: Property
    mixing: MixingMatrix | None = None
    kind: FoiKindArg = FoiKind.FREQUENCY
    contact_rate: RateOps = None  # type: ignore[assignment]
    denominator: Selector | None = None
    infectiousness: InfectiousnessMap | None = None
    normalize_infectiousness: NormalizeWeights = None
    exponent: RateOps | None = None

    def __init__(
        self,
        name: str,
        *,
        infectious: Selector,
        group_by: Property,
        mixing: MixingMatrix | None = None,
        kind: FoiKindArg = FoiKind.FREQUENCY,
        contact_rate: object = 1.0,
        denominator: Selector | None = None,
        infectiousness: InfectiousnessMap | None = None,
        normalize_infectiousness: NormalizeWeights = None,
        exponent: object | None = None,
    ) -> None:
        if not callable(kind) and not isinstance(kind, FoiKind):
            raise TypeError(f"ForceOfInfection kind must be a FoiKind or callable, got {kind!r}.")
        if kind is FoiKind.GENERALISED:
            if exponent is None:
                raise ValueError(
                    "ForceOfInfection kind=FoiKind.GENERALISED requires exponent= "
                    "(a float, Param, or other rate expression)."
                )
        elif exponent is not None:
            raise ValueError(
                "ForceOfInfection exponent= is only valid with "
                f"kind=FoiKind.GENERALISED, not kind={kind!r}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "infectious", infectious)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "contact_rate", as_rate(contact_rate))
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "infectiousness", infectiousness)
        object.__setattr__(self, "normalize_infectiousness", normalize_infectiousness)
        object.__setattr__(self, "exponent", None if exponent is None else as_rate(exponent))
        if mixing is not None and mixing.prop.name != group_by.name:
            raise ValueError(
                f"MixingMatrix property {mixing.prop.name!r} does not match "
                f"ForceOfInfection group_by {group_by.name!r}."
            )
        if infectiousness is not None:
            for key in infectiousness:
                trait_prop = key.property if isinstance(key, Trait) else group_by.name
                if trait_prop != group_by.name:
                    raise ValueError(
                        f"Infectiousness key property {trait_prop!r} does not "
                        f"match ForceOfInfection group_by {group_by.name!r}."
                    )

    def __flow_refs__(self) -> set[str]:
        return set()

    def __field_paths__(self) -> set[tuple[str, ...]]:
        paths = set()
        # contact_rate and infectiousness FieldRefs
        from summer4.flows.rates import _field_paths

        paths |= _field_paths(self.contact_rate)
        if self.exponent is not None:
            paths |= _field_paths(self.exponent)
        if self.mixing is not None:
            paths |= _field_paths(self.mixing.matrix)
        if self.infectiousness is not None:
            for value in self.infectiousness.values():
                paths |= _field_paths(as_rate(value))
        return paths

    def __rate_bytes__(self) -> bytes:
        from summer4.flows.rates import _rate_bytes, _selector_bytes

        body = b"foi" + self.name.encode() + self.group_by.name.encode()
        body += _selector_bytes(self.infectious)
        body += _rate_bytes(self.contact_rate)
        if isinstance(self.kind, FoiKind):
            body += repr(self.kind.value).encode()
        else:
            body += repr(id(self.kind)).encode()
        body += repr(self.normalize_infectiousness).encode()
        if self.exponent is not None:
            body += b"exp" + _rate_bytes(self.exponent)
        if self.denominator is not None:
            body += _selector_bytes(self.denominator)
        if self.mixing is not None:
            body += _rate_bytes(self.mixing.matrix)
            body += self.mixing.normalize.encode()
            body += b"1" if self.mixing.check_reciprocal else b"0"
        if self.infectiousness is not None:
            for key, value in sorted(
                self.infectiousness.items(),
                key=lambda kv: kv[0].name if isinstance(kv[0], Trait) else str(kv[0]),
            ):
                k = key.name if isinstance(key, Trait) else str(key)
                body += k.encode() + _rate_bytes(as_rate(value))
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

    def captured(self) -> Capture:
        """Wrap this FOI in a :class:`Capture` so it is saveable by name."""
        return Capture(self.name, self)


def _weights_vector(
    foi: ForceOfInfection,
    *,
    eval_child: Callable[[RateOps], Any],
    n_traits: int,
) -> Any | None:
    import jax.numpy as jnp

    if not foi.infectiousness:
        return None
    weights = jnp.ones(n_traits)
    for key, value in foi.infectiousness.items():
        idx = key.code if isinstance(key, Trait) else foi.group_by.traits.index(key)
        weights = weights.at[idx].set(jnp.asarray(eval_child(as_rate(value))))
    return weights


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
    i_pd = pd.keep(expr.infectious, 0.0).sum_over(prop)
    if expr.denominator is not None:
        n_pd = pd.keep(expr.denominator, 0.0).sum_over(prop)
    else:
        n_pd = pd.sum_over(prop)
    i_grp = GroupedRate(i_pd.data, (prop,))
    n_grp = GroupedRate(n_pd.data, (prop,))

    weights = _weights_vector(expr, eval_child=eval_child, n_traits=len(prop.traits))
    if weights is not None:
        import jax.numpy as jnp

        w = weights
        if expr.normalize_infectiousness == "population":
            # Population-weighted mean of weights is 1.
            import jax.numpy as jnp

            n_data = jnp.asarray(n_grp.data)
            total_n = jnp.sum(n_data)
            mean_w = jnp.sum(w * n_data) / jnp.maximum(total_n, 1e-12)
            w = w / mean_w
        elif expr.normalize_infectiousness == "mean":
            import jax.numpy as jnp

            w = w / jnp.mean(w)
        i_grp = GroupedRate(i_grp.data * w, (prop,))

    kind = expr.kind
    if callable(kind):
        shedding = kind(i_grp, n_grp)
        if not isinstance(shedding, GroupedRate):
            raise TypeError(
                f"ForceOfInfection kind callable must return GroupedRate, "
                f"got {type(shedding).__name__}."
            )
    elif kind is FoiKind.FREQUENCY:
        shedding = i_grp / n_grp
    elif kind is FoiKind.DENSITY:
        shedding = i_grp
    elif kind is FoiKind.GENERALISED:
        if expr.exponent is None:  # pragma: no cover - validated in __init__
            raise ValueError("kind=FoiKind.GENERALISED requires exponent=.")
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
    captures[expr.name] = result
    return result
