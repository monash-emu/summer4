"""Public flow declarations: transition, exit, and entry."""

from __future__ import annotations

from dataclasses import dataclass

from summer4.flows.join import (
    NormalizedSplit,
    SplitSpec,
    TraitChain,
    TraitMatrix,
    _normalize_split,
)
from summer4.flows.rates import Adjustment, AdjustSpec, RateOps, _normalize_adjust, as_rate
from summer4.selectors import Selector, SelectorOps


def _as_selector(sel: Selector) -> Selector:
    if not isinstance(sel, SelectorOps):
        raise TypeError(f"Expected a Selector, got {type(sel).__name__}.")
    return sel


@dataclass(frozen=True, slots=True)
class TransitionFlow:
    """Population moving from source compartments to destination compartments."""

    name: str
    source: Selector
    dest: Selector
    rate: RateOps
    pairing: TraitChain | TraitMatrix | None = None
    split: NormalizedSplit | None = None
    absolute: bool = False
    adjust: tuple[Adjustment, ...] = ()

    def __init__(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        rate: object,
        *,
        pairing: TraitChain | TraitMatrix | None = None,
        split: SplitSpec | None = None,
        absolute: bool = False,
        adjust: AdjustSpec = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", _as_selector(source))
        object.__setattr__(self, "dest", _as_selector(dest))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "pairing", pairing)
        object.__setattr__(self, "split", _normalize_split(split))
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "adjust", _normalize_adjust(adjust))


@dataclass(frozen=True, slots=True)
class ExitFlow:
    """Population leaving source compartments to outside the system."""

    name: str
    source: Selector
    rate: RateOps
    absolute: bool = False
    adjust: tuple[Adjustment, ...] = ()

    def __init__(
        self,
        name: str,
        source: Selector,
        rate: object,
        *,
        absolute: bool = False,
        adjust: AdjustSpec = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", _as_selector(source))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "adjust", _normalize_adjust(adjust))


@dataclass(frozen=True, slots=True)
class EntryFlow:
    """Population entering destination compartments from outside the system."""

    name: str
    dest: Selector
    rate: RateOps
    split: NormalizedSplit | None = None
    adjust: tuple[Adjustment, ...] = ()

    def __init__(
        self,
        name: str,
        dest: Selector,
        rate: object,
        *,
        split: SplitSpec | None = None,
        adjust: AdjustSpec = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dest", _as_selector(dest))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "split", _normalize_split(split))
        object.__setattr__(self, "adjust", _normalize_adjust(adjust))


type FlowLike = TransitionFlow | ExitFlow | EntryFlow
