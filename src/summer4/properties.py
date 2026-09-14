"""Named properties and their mutually exclusive traits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from summer4.selectors import Absent, IsIn, Present, Selector, SelectorOps


@dataclass(frozen=True, slots=True)
class Trait(SelectorOps):
    """A single mutually exclusive value of a :class:`Property`.

    ``Trait`` is also a selector leaf: it is true on compartments that
    carry this property with this value, false on compartments that carry
    a different value, and unknown where the property is absent.
    """

    property: str
    name: str
    code: int


@dataclass(frozen=True, slots=True)
class Property:
    """A named group of mutually exclusive traits.

    Properties are identified by ``name``, not object identity. Trait codes
    are the integers ``0 .. len(traits) - 1`` in declaration order.
    """

    name: str
    traits: tuple[str, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Property name must be a non-empty string.")
        if not self.name.isidentifier():
            raise ValueError(
                f"Property name {self.name!r} must be a Python identifier "
                f"(cannot contain '@' or other non-identifier characters)."
            )
        if not self.traits:
            raise ValueError(f"Property {self.name!r} must have at least one trait.")
        if any(not trait for trait in self.traits):
            raise ValueError(f"Property {self.name!r} has an empty trait name.")
        if len(set(self.traits)) != len(self.traits):
            raise ValueError(f"Property {self.name!r} has duplicate traits.")
        object.__setattr__(self, "_index", {trait: i for i, trait in enumerate(self.traits)})

    def trait(self, name: str) -> Trait:
        """Return the :class:`Trait` named ``name``."""
        try:
            code = self._index[name]
        except KeyError:
            raise KeyError(
                f"Unknown trait {name!r} for property {self.name!r}. Known: {list(self.traits)}"
            ) from None
        return Trait(property=self.name, name=name, code=code)

    def __getitem__(self, key: str | Sequence[str]) -> Selector:
        """Return a selector for one trait (``age["0-4"]``) or several (``age["0-4", "5-9"]``)."""
        if isinstance(key, str):
            return self.trait(key)
        return self.isin(key)

    def isin(self, names: Iterable[str]) -> IsIn:
        """Return a selector that is true when this property is one of ``names``."""
        resolved = tuple(names)
        if not resolved:
            raise ValueError(f"Property {self.name!r}.isin() requires at least one trait.")
        for name in resolved:
            if name not in self._index:
                raise KeyError(
                    f"Unknown trait {name!r} for property {self.name!r}. Known: {list(self.traits)}"
                )
        return IsIn(property=self.name, names=resolved)

    def present(self) -> Present:
        """Return a selector that is true where this property is defined."""
        return Present(property=self.name)

    def absent(self) -> Absent:
        """Return a selector that is true where this property is not defined."""
        return Absent(property=self.name)
