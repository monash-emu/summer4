"""EpiModel frontend over FlowModel + force of infection."""

from __future__ import annotations

from typing import Any

from summer4.epi.infection import ForceOfInfection
from summer4.epi.mixing import MixingMatrix
from summer4.flows.compiled import CompiledModel, FlowModel
from summer4.flows.rates import FlowRef
from summer4.flows.types import FlowLike, TransitionFlow
from summer4.properties import Property
from summer4.propertymap import PropertyMap
from summer4.selectors import Selector


class EpiModel:
    """Summer2-shaped frontend over :class:`FlowModel` + force of infection.

    Every builder method is sugar over objects the user could construct
    directly. Escape hatches: :attr:`flow_model`, :meth:`foi`, :meth:`add_flow`.
    """

    def __init__(self, pmap: PropertyMap, *, infectious: Selector | None = None) -> None:
        self._flow_model = FlowModel(pmap)
        self._infectious = infectious
        self._mixing: MixingMatrix | None = None
        self._infectiousness: dict[Any, object] | None = None
        self._normalize_infectiousness: str | None = None
        self._fois: dict[str, ForceOfInfection] = {}

    @property
    def flow_model(self) -> FlowModel:
        """The underlying :class:`FlowModel` (live; mutations are visible)."""
        return self._flow_model

    @property
    def pmap(self) -> PropertyMap:
        return self._flow_model.pmap

    def foi(self, name: str) -> ForceOfInfection:
        """Return the named :class:`ForceOfInfection`."""
        try:
            return self._fois[name]
        except KeyError:
            known = ", ".join(sorted(self._fois)) or "(none)"
            raise KeyError(f"Unknown FOI {name!r}. Known: {known}.") from None

    def set_mixing_matrix(
        self,
        prop: Property,
        matrix: object,
        *,
        normalize: str = "rows",
        check_reciprocal: bool = False,
    ) -> MixingMatrix:
        """Attach a default mixing matrix for subsequent infection flows."""
        self._mixing = MixingMatrix(
            prop,
            matrix,
            normalize=normalize,  # type: ignore[arg-type]
            check_reciprocal=check_reciprocal,
        )
        return self._mixing

    def add_infectiousness_adjustments(
        self,
        prop: Property,
        weights: dict[str, object],
        *,
        normalize: str | None = "population",
    ) -> None:
        """FOI-owned infectiousness weights (not stratification-owned)."""
        resolved = {prop[name]: value for name, value in weights.items()}
        self._infectiousness = resolved
        self._normalize_infectiousness = normalize

    def add_flow(self, flow: FlowLike) -> FlowRef:
        """Accept any declarative flow (escape hatch)."""
        return self._flow_model.add_flow(flow)

    def add_transition_flow(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        rate: object,
        **kwargs: Any,
    ) -> FlowRef:
        return self.add_flow(TransitionFlow(name, source, dest, rate, **kwargs))

    def add_infection_frequency_flow(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        contact_rate: object,
        *,
        infectious: Selector | None = None,
        group_by: Property | None = None,
        mixing: MixingMatrix | None = None,
    ) -> FlowRef:
        return self._add_infection(
            name,
            source,
            dest,
            contact_rate,
            kind="frequency",
            infectious=infectious,
            group_by=group_by,
            mixing=mixing,
        )

    def add_infection_density_flow(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        contact_rate: object,
        *,
        infectious: Selector | None = None,
        group_by: Property | None = None,
        mixing: MixingMatrix | None = None,
    ) -> FlowRef:
        return self._add_infection(
            name,
            source,
            dest,
            contact_rate,
            kind="density",
            infectious=infectious,
            group_by=group_by,
            mixing=mixing,
        )

    def _add_infection(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        contact_rate: object,
        *,
        kind: str,
        infectious: Selector | None,
        group_by: Property | None,
        mixing: MixingMatrix | None,
    ) -> FlowRef:
        inf = infectious if infectious is not None else self._infectious
        if inf is None:
            raise ValueError(
                "Infectious selector required: pass infectious= or "
                "construct EpiModel(..., infectious=...)."
            )
        mix = mixing if mixing is not None else self._mixing
        if group_by is None:
            if mix is not None:
                group_by = mix.prop
            else:
                raise ValueError("group_by is required when no mixing matrix is set.")
        foi = ForceOfInfection(
            name,
            infectious=inf,
            group_by=group_by,
            mixing=mix,
            kind=kind,  # type: ignore[arg-type]
            contact_rate=contact_rate,
            infectiousness=self._infectiousness,
            normalize_infectiousness=self._normalize_infectiousness,  # type: ignore[arg-type]
        )
        self._fois[name] = foi
        return self.add_flow(TransitionFlow(name, source, dest, foi))

    def set_initial_population(
        self,
        base: object,
        splits: object = (),
    ) -> Any:
        """Attach a declarative initial population on the underlying FlowModel."""
        return self._flow_model.set_initial_population(base, splits=splits)  # type: ignore[arg-type]

    def compile(self, **kwargs: Any) -> CompiledModel:
        return self._flow_model.compile(**kwargs)
