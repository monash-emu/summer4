"""Kleene three-valued selector algebra over compartment properties.

Every selector evaluates to an ``int8`` array over compartments:

* ``1``  — true
* ``0``  — unknown (the property is not applicable)
* ``-1`` — false

Negation, conjunction and disjunction are ``-a``, ``minimum`` and
``maximum`` respectively. ``PropertyMap.select`` keeps only ``true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from summer4.properties import Trait


class SelectorOps:
    """Mixin providing ``&``, ``|`` and ``~`` for every selector node."""

    __slots__ = ()

    def __and__(self, other: object) -> And:
        if not isinstance(other, SelectorOps):
            return NotImplemented
        return And(cast(Selector, self), cast(Selector, other))

    def __or__(self, other: object) -> Or:
        if not isinstance(other, SelectorOps):
            return NotImplemented
        return Or(cast(Selector, self), cast(Selector, other))

    def __invert__(self) -> Not:
        return Not(cast(Selector, self))

    def __bool__(self) -> bool:
        raise TypeError(
            "Selectors cannot be used as booleans; combine them with & | ~ "
            "and resolve them through PropertyMap.select or PropertyMap.mask."
        )


@dataclass(frozen=True, slots=True)
class IsIn(SelectorOps):
    """True when a property's trait is one of ``names``."""

    property: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("IsIn requires at least one trait name.")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"IsIn has duplicate trait names: {self.names!r}.")


@dataclass(frozen=True, slots=True)
class Present(SelectorOps):
    """True when a property is defined on the compartment (two-valued)."""

    property: str


@dataclass(frozen=True, slots=True)
class Absent(SelectorOps):
    """True when a property is not defined on the compartment (two-valued)."""

    property: str


@dataclass(frozen=True, slots=True)
class Everything(SelectorOps):
    """True for every compartment."""


@dataclass(frozen=True, slots=True)
class Nothing(SelectorOps):
    """False for every compartment."""


@dataclass(frozen=True, slots=True)
class And(SelectorOps):
    """Kleene conjunction of two selectors."""

    left: Selector
    right: Selector


@dataclass(frozen=True, slots=True)
class Or(SelectorOps):
    """Kleene disjunction of two selectors."""

    left: Selector
    right: Selector


@dataclass(frozen=True, slots=True)
class Not(SelectorOps):
    """Kleene negation of a selector."""

    inner: Selector


@dataclass(frozen=True, slots=True)
class Source(SelectorOps):
    """Select flow edges by a predicate on the source compartment.

    On a compartment :class:`~summer4.propertymap.PropertyMap` this raises.
    Evaluate it on an :class:`~summer4.flows.edges.EdgeMap` from
    :meth:`~summer4.flows.compiled.CompiledModel.edges`.
    """

    inner: Selector


@dataclass(frozen=True, slots=True)
class Dest(SelectorOps):
    """Select flow edges by a predicate on the destination compartment.

    On a compartment :class:`~summer4.propertymap.PropertyMap` this raises.
    Evaluate it on an :class:`~summer4.flows.edges.EdgeMap` from
    :meth:`~summer4.flows.compiled.CompiledModel.edges`.
    """

    inner: Selector


type Selector = (
    Trait | IsIn | Present | Absent | Everything | Nothing | And | Or | Not | Source | Dest
)
