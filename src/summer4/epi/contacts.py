"""Empirical contact matrices: load, validate, inspect and adapt contact-survey data.

A contact survey reports how many people of each age band a participant met
per unit time. :class:`ContactMatrix` holds one such matrix together with its
age-band edges, and :class:`SettingStack` holds the per-setting matrices
("home", "school", "work", ...) these datasets usually ship as.

**Orientation, fixed once.** ``values[i, j]`` is the average number of contacts
a member of band ``i`` (the participant) has with members of band ``j`` (the
contact), per person per unit time. Row ``i`` is what participants in band
``i`` reported. That is the orientation :class:`~summer4.epi.ForceOfInfection`
uses (``mixing @ shedding``: rows are the people being infected). Published
tables are often the transpose (rows labelled "age of contact"); transpose them
on the way in.

Loading, validation, inspection and the population transformations
(:meth:`ContactMatrix.rebin`, :meth:`~ContactMatrix.symmetrise`,
:meth:`~ContactMatrix.adapt`) are host-side NumPy: they run once, before a
model does. What a calibration varies — the scale of a setting over time or by
parameter — is :class:`ScaledContacts`, a rate expression evaluated inside the
model under ``jit``. :meth:`Rebin.apply` also accepts traced arrays.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, NamedTuple, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from summer4.data import _require_pandas, frame_columns
from summer4.enums import coerce_strenum
from summer4.epi.mixing import MixingMatrix, MixingNormalize, MixingNormalizeArg
from summer4.flows.rates import ArrayConst, RateOps, as_rate
from summer4.properties import Property

__all__ = [
    "ContactAdaptation",
    "ContactMatrix",
    "Rebin",
    "Reciprocity",
    "ScaledContacts",
    "SettingStack",
    "band_label",
]

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_INTERVAL = re.compile(rf"^\[\s*({_NUMBER})\s*,\s*({_NUMBER}|inf)\s*\)$")
_INCLUSIVE = re.compile(rf"^({_NUMBER})\s*-\s*({_NUMBER})$")
_OPEN = re.compile(rf"^({_NUMBER})\s*\+$")
_LOWER = re.compile(rf"^({_NUMBER})$")


def _fmt(edge: float) -> str:
    if math.isinf(edge):
        return "inf"
    return str(int(edge)) if float(edge).is_integer() else repr(float(edge))


def band_label(lower: float, upper: float) -> str:
    """The label ``[lower,upper)`` used for a band in frames and tables."""
    return f"[{_fmt(lower)},{_fmt(upper)})"


def _parse_label(label: object) -> tuple[float, float | None]:
    """Parse one band label into ``(lower, upper)``; ``upper`` is ``None`` if unstated.

    Accepted: ``"[0,5)"``, ``"0-4"`` (inclusive integer ages, so ``[0,5)``),
    ``"75+"`` (``[75,inf)``) and a numeric lower bound (``0``, ``"5"``).
    """
    if isinstance(label, (int, float, np.integer, np.floating)) and not isinstance(label, bool):
        return float(label), None
    text = str(label).strip()
    if match := _INTERVAL.match(text):
        return float(match.group(1)), float(match.group(2))
    if match := _INCLUSIVE.match(text):
        return float(match.group(1)), float(match.group(2)) + 1.0
    if match := _OPEN.match(text):
        return float(match.group(1)), math.inf
    if match := _LOWER.match(text):
        return float(match.group(1)), None
    raise ValueError(
        f"Cannot parse contact-matrix band label {label!r}; expected '[0,5)', '0-4', "
        "'75+' or a numeric lower bound."
    )


def _edges_from_labels(labels: Sequence[object]) -> tuple[list[object], tuple[float, ...]]:
    """Order labels by lower bound and derive the ``K + 1`` band edges."""
    parsed = {label: _parse_label(label) for label in labels}
    ordered = sorted(labels, key=lambda label: parsed[label][0])
    edges: list[float] = [parsed[ordered[0]][0]]
    for index, label in enumerate(ordered):
        lower, upper = parsed[label]
        if index + 1 < len(ordered):
            nxt_label = ordered[index + 1]
            nxt = parsed[nxt_label][0]
            if upper is not None and upper != nxt:
                raise ValueError(
                    f"Band labels are not contiguous: {label!r} ends at {_fmt(upper)} "
                    f"but the next band {nxt_label!r} starts at {_fmt(nxt)}."
                )
            edges.append(nxt)
        else:
            edges.append(math.inf if upper is None else upper)
    return ordered, tuple(edges)


def _numeric_traits(prop: Property) -> list[float] | None:
    out: list[float] = []
    for trait in prop.traits:
        try:
            out.append(float(trait))
        except ValueError:
            return None
    return out


def _check_prop(prop: Property, bands: tuple[float, ...], *, what: str) -> None:
    """Alignment is by position; numeric trait names must also be the lower edges."""
    k = len(bands) - 1
    if len(prop.traits) != k:
        raise ValueError(
            f"{what}: property {prop.name!r} has {len(prop.traits)} traits but the matrix "
            f"has {k} bands."
        )
    numeric = _numeric_traits(prop)
    if numeric is None:
        return
    for trait, value, lower in zip(prop.traits, numeric, bands[:-1], strict=True):
        if value != lower:
            raise ValueError(
                f"{what}: trait {trait!r} of property {prop.name!r} reads as lower bound "
                f"{_fmt(value)}, but that band starts at {_fmt(lower)}."
            )


def _validate(
    values: NDArray[np.float64],
    bands: tuple[float, ...],
    prop: Property | None,
    source_population: NDArray[np.float64] | None,
) -> None:
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"ContactMatrix values must be square; got shape {values.shape}.")
    k = values.shape[0]
    if k < 1:
        raise ValueError("ContactMatrix needs at least one band.")
    if len(bands) != k + 1:
        raise ValueError(
            f"ContactMatrix has {len(bands)} band edges for a {k} x {k} matrix; "
            f"expected {k + 1}."
        )
    for index, edge in enumerate(bands):
        if math.isnan(edge) or (math.isinf(edge) and (edge < 0 or index != len(bands) - 1)):
            raise ValueError(
                f"ContactMatrix band edge bands[{index}] = {edge} is not allowed; "
                "only the last edge may be inf."
            )
    for index in range(1, len(bands)):
        if bands[index] <= bands[index - 1]:
            raise ValueError(
                "ContactMatrix bands must be strictly increasing; got "
                f"bands[{index - 1}] = {_fmt(bands[index - 1])} then "
                f"bands[{index}] = {_fmt(bands[index])}."
            )
    bad = np.argwhere(~np.isfinite(values) | (values < 0))
    if bad.size:
        i, j = (int(x) for x in bad[0])
        raise ValueError(
            f"ContactMatrix values must be finite and non-negative; "
            f"values[{i}, {j}] = {values[i, j]}."
        )
    if prop is not None:
        _check_prop(prop, bands, what="ContactMatrix")
    if source_population is not None:
        _check_population(source_population, k, what="source_population")


def _check_population(population: NDArray[np.float64], k: int, *, what: str) -> None:
    if population.shape != (k,):
        raise ValueError(
            f"{what} has shape {population.shape}; expected ({k},), one entry per band."
        )
    bad = np.flatnonzero(~np.isfinite(population) | (population <= 0))
    if bad.size:
        index = int(bad[0])
        raise ValueError(
            f"{what} must be finite and positive; {what}[{index}] = {population[index]}."
        )


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


class ContactAdaptation(StrEnum):
    """How :meth:`ContactMatrix.adapt` carries a matrix to another population.

    Prefer these members at call sites; the strings ``"frequency"`` and
    ``"density"`` are accepted and coerced.
    """

    FREQUENCY = "frequency"
    DENSITY = "density"


type ContactAdaptationArg = ContactAdaptation | str


def _check_edges(edges: tuple[float, ...], *, what: str) -> None:
    if len(edges) < 2:
        raise ValueError(f"{what} needs at least two band edges; got {len(edges)}.")
    for index, edge in enumerate(edges):
        if math.isnan(edge) or (math.isinf(edge) and (edge < 0 or index != len(edges) - 1)):
            raise ValueError(
                f"{what} edge [{index}] = {edge} is not allowed; only the last edge may be inf."
            )
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            raise ValueError(
                f"{what} must be strictly increasing; got [{index - 1}] = "
                f"{_fmt(edges[index - 1])} then [{index}] = {_fmt(edges[index])}."
            )


def _labels(edges: Sequence[float]) -> list[str]:
    return [band_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def _onto_refinement(
    population: NDArray[np.float64],
    population_bands: tuple[float, ...],
    refinement: tuple[float, ...],
) -> NDArray[np.float64]:
    """Spread a banded population over ``refinement``, uniform within each band."""
    if population_bands[0] != refinement[0] or population_bands[-1] != refinement[-1]:
        raise ValueError(
            f"population_bands span [{_fmt(population_bands[0])}, {_fmt(population_bands[-1])}) "
            f"but the bands being rebinned span [{_fmt(refinement[0])}, {_fmt(refinement[-1])})."
        )
    out = np.zeros(len(refinement) - 1)
    for r, (a, b) in enumerate(zip(refinement[:-1], refinement[1:], strict=True)):
        for p, (c, d) in enumerate(zip(population_bands[:-1], population_bands[1:], strict=True)):
            overlap = min(b, d) - max(a, c)
            if overlap <= 0:
                continue
            if math.isinf(d):
                if a == c and math.isinf(b):
                    out[r] += population[p]
                    continue
                raise ValueError(
                    f"Cannot split the open population band {band_label(c, d)} at "
                    f"{_fmt(a if a > c else b)}; give population on bands that do not need it."
                )
            out[r] += population[p] * overlap / (d - c)
    return out


@dataclass(frozen=True, slots=True, eq=False)
class Rebin:
    """The constant aggregation that moves a contact matrix onto other bands.

    ``apply(values)`` is ``rows @ values @ cols.T``: two matrix products with
    ``(K_new, K_old)`` constants built once, on the host. It accepts NumPy or
    JAX arrays, so a rebin can sit inside a jitted loss or a rate expression.

    Both band sets are first cut at every edge either of them has (the
    *common refinement*), then:

    - **splitting** a band assumes contacts are uniform within it. A
      participant in either half reports the whole band's contacts (rows copy),
      and contacts *with* a half are the band's contacts times that half's share
      of the band's population (columns split by population);
    - **aggregating** takes a population-weighted mean over the participant
      bands merged (rows) and a plain sum over the contact bands merged
      (columns).

    Both steps need the population, which is why it is required. Total contacts
    $\\sum_{ij} N_i c_{ij}$ are conserved, so :meth:`ContactMatrix.mean_contacts`
    is unchanged by a rebin.
    """

    source: tuple[float, ...]
    target: tuple[float, ...]
    rows: NDArray[np.float64]
    cols: NDArray[np.float64]
    population: NDArray[np.float64]

    @classmethod
    def between(
        cls,
        source: Sequence[float],
        target: Sequence[float],
        *,
        population: ArrayLike,
        population_bands: Sequence[float] | None = None,
    ) -> Rebin:
        """Build the rebin from ``source`` band edges to ``target`` band edges.

        ``population`` is the population per band of ``population_bands``.
        Without ``population_bands`` it must be given on the common refinement
        of the two band sets — the ``source`` bands when ``target`` only merges
        them, the ``target`` bands when it only splits them. With
        ``population_bands``, it is spread onto the refinement assuming a
        uniform population within each of its bands.
        """
        src = tuple(float(edge) for edge in source)
        tgt = tuple(float(edge) for edge in target)
        _check_edges(src, what="Rebin source bands")
        _check_edges(tgt, what="Rebin target bands")
        if src[0] != tgt[0] or src[-1] != tgt[-1]:
            raise ValueError(
                f"Rebin target bands span [{_fmt(tgt[0])}, {_fmt(tgt[-1])}) but the source "
                f"bands span [{_fmt(src[0])}, {_fmt(src[-1])}); they must cover the same ages."
            )
        refinement = tuple(sorted(set(src) | set(tgt)))
        pop = np.asarray(population, dtype=np.float64)
        if population_bands is None:
            if pop.shape != (len(refinement) - 1,):
                raise ValueError(
                    f"population has shape {pop.shape}; without population_bands it must "
                    f"have one entry per band of the common refinement "
                    f"{_labels(refinement)} ({len(refinement) - 1} bands)."
                )
        else:
            pbands = tuple(float(edge) for edge in population_bands)
            _check_edges(pbands, what="population_bands")
            if pop.shape != (len(pbands) - 1,):
                raise ValueError(
                    f"population has shape {pop.shape} for {len(pbands) - 1} population_bands."
                )
            pop = _onto_refinement(pop, pbands, refinement)
        _check_population(pop, len(refinement) - 1, what="population on the refinement")
        lowers = np.asarray(refinement[:-1])
        old = np.searchsorted(np.asarray(src), lowers, side="right") - 1
        new = np.searchsorted(np.asarray(tgt), lowers, side="right") - 1
        k_old, k_new, r = len(src) - 1, len(tgt) - 1, len(refinement) - 1
        pop_old = np.bincount(old, weights=pop, minlength=k_old)
        pop_new = np.bincount(new, weights=pop, minlength=k_new)
        split_rows = np.zeros((r, k_old))
        split_rows[np.arange(r), old] = 1.0
        split_cols = split_rows * (pop / pop_old[old])[:, None]
        merge_cols = np.zeros((k_new, r))
        merge_cols[new, np.arange(r)] = 1.0
        merge_rows = merge_cols * (pop / pop_new[new])[None, :]
        return cls(
            source=src,
            target=tgt,
            rows=_readonly(merge_rows @ split_rows),
            cols=_readonly(merge_cols @ split_cols),
            population=_readonly(pop_new),
        )

    def apply(self, values: Any) -> Any:
        """``rows @ values @ cols.T`` for a NumPy array, or a JAX array under ``jit``."""
        if isinstance(values, (np.ndarray, list, tuple)):
            return self.rows @ np.asarray(values, dtype=np.float64) @ self.cols.T
        import jax.numpy as jnp

        return jnp.asarray(self.rows) @ values @ jnp.asarray(self.cols).T


class Reciprocity(NamedTuple):
    """How far a contact matrix is from reciprocal against a population.

    ``matrix[i, j]`` is $c_{ij} N_i - c_{ji} N_j$: total contacts band ``i``
    reports with band ``j`` minus the total band ``j`` reports with band ``i``.
    It is antisymmetric and zero for a reciprocal matrix. ``relative`` is its
    Frobenius norm divided by that of $c_{ij} N_i$, so ``0`` is reciprocal and
    the value does not depend on the population's units.
    """

    matrix: NDArray[np.float64]
    relative: float


@dataclass(frozen=True, slots=True, eq=False)
class ContactMatrix:
    """Contacts per person per unit time, from band ``i`` with band ``j``.

    ``values[i, j]`` is the average number of contacts a member of band ``i``
    has with members of band ``j``. Rows are participants, columns are the
    people they met. ``bands`` holds the ``K + 1`` strictly increasing band
    edges; band ``i`` is ``[bands[i], bands[i + 1])`` and only the last edge
    may be ``inf`` (an open "75+" band).

    ``prop``, when given, is the age property the matrix is aligned to.
    Alignment is by **position** — trait ``i`` is band ``i`` — so trait names
    need not be numeric. When every trait name does parse as a number, those
    numbers must equal the lower band edges.

    ``setting`` names the setting ("home", "school", ...) or is ``None`` for
    all settings combined. ``source_population`` is the survey population per
    band; methods that need a population default to it.

    Instances are immutable; every method returns a new object.
    """

    values: NDArray[np.float64]
    bands: tuple[float, ...]
    prop: Property | None = None
    setting: str | None = None
    source_population: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        bands = tuple(float(edge) for edge in self.bands)
        source = None if self.source_population is None else _readonly(self.source_population)
        _validate(values, bands, self.prop, source)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "source_population", source)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_array(
        cls,
        values: ArrayLike,
        bands: Sequence[float],
        *,
        prop: Property | None = None,
        setting: str | None = None,
        source_population: ArrayLike | None = None,
    ) -> ContactMatrix:
        """Build from a ``(K, K)`` array already in participant-by-contact orientation."""
        source = None if source_population is None else np.asarray(source_population, float)
        return cls(
            values=np.asarray(values, dtype=np.float64),
            bands=tuple(bands),
            prop=prop,
            setting=setting,
            source_population=source,
        )

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        row: str = "age_from",
        col: str = "age_to",
        value: str = "contacts",
        bands: Sequence[float] | None = None,
        prop: Property | None = None,
        setting: str | None = None,
        source_population: ArrayLike | None = None,
    ) -> ContactMatrix:
        """Build from a long pandas or polars frame, one row per (participant, contact) band.

        ``row`` holds the participant's band and ``col`` the contact's band, so
        the result has ``values[row, col] = value``. Band labels may be
        ``"[0,5)"``, ``"0-4"`` (inclusive integer ages), ``"75+"`` or numeric
        lower bounds; bands are ordered by lower bound. With numeric lower
        bounds only, the last band is open (``inf``) unless ``bands`` says
        otherwise. When ``bands`` is given and the labels do not parse, they
        are taken in order of first appearance. Every (row, col) pair must
        appear exactly once.
        """
        columns = frame_columns(frame, (row, col, value))
        return cls._from_columns(
            columns[row],
            columns[col],
            columns[value],
            bands=bands,
            prop=prop,
            setting=setting,
            source_population=source_population,
        )

    @classmethod
    def _from_columns(
        cls,
        rows: NDArray[Any],
        cols: NDArray[Any],
        values: NDArray[Any],
        *,
        bands: Sequence[float] | None,
        prop: Property | None,
        setting: str | None,
        source_population: ArrayLike | None,
    ) -> ContactMatrix:
        labels = list(dict.fromkeys(rows.tolist()))
        extra = [label for label in dict.fromkeys(cols.tolist()) if label not in set(labels)]
        if extra:
            raise ValueError(
                f"Contact labels {extra} never appear as participant labels; "
                f"participant labels are {labels}."
            )
        if bands is None:
            ordered, edges = _edges_from_labels(labels)
        else:
            edges = tuple(float(edge) for edge in bands)
            if len(labels) != len(edges) - 1:
                raise ValueError(
                    f"Frame has {len(labels)} band labels but bands gives {len(edges) - 1} bands."
                )
            try:
                parsed = {label: _parse_label(label)[0] for label in labels}
            except ValueError:
                ordered = labels
            else:
                ordered = sorted(labels, key=lambda label: parsed[label])
                for label, lower in zip(ordered, edges[:-1], strict=True):
                    if parsed[label] != lower:
                        raise ValueError(
                            f"Band label {label!r} starts at {_fmt(parsed[label])}, but "
                            f"bands says that band starts at {_fmt(lower)}."
                        )
        index = {label: i for i, label in enumerate(ordered)}
        k = len(ordered)
        matrix = np.full((k, k), np.nan)
        for r, c, v in zip(rows.tolist(), cols.tolist(), values.tolist(), strict=True):
            i, j = index[r], index[c]
            if not np.isnan(matrix[i, j]):
                raise ValueError(f"Frame lists the pair ({r!r}, {c!r}) more than once.")
            matrix[i, j] = float(v)
        missing = np.argwhere(np.isnan(matrix))
        if missing.size:
            i, j = (int(x) for x in missing[0])
            raise ValueError(
                f"Frame has no value for the pair ({ordered[i]!r}, {ordered[j]!r}); "
                f"{len(missing)} of {k * k} pairs are missing."
            )
        return cls.from_array(
            matrix, edges, prop=prop, setting=setting, source_population=source_population
        )

    @staticmethod
    def from_settings(settings: Mapping[str, ContactMatrix]) -> SettingStack:
        """Stack per-setting matrices that share bands; see :class:`SettingStack`."""
        return SettingStack(settings)

    # -- identity -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContactMatrix):
            return NotImplemented
        return (
            self.bands == other.bands
            and self.prop == other.prop
            and self.setting == other.setting
            and np.array_equal(self.values, other.values)
            and _optional_equal(self.source_population, other.source_population)
        )

    def __hash__(self) -> int:
        source = None if self.source_population is None else self.source_population.tobytes()
        return hash((self.values.tobytes(), self.bands, self.prop, self.setting, source))

    def __repr__(self) -> str:
        where = "" if self.setting is None else f", setting={self.setting!r}"
        aligned = "" if self.prop is None else f", prop={self.prop.name!r}"
        return (
            f"ContactMatrix({self.size} bands {self.labels[0]}..{self.labels[-1]}{where}{aligned})"
        )

    # -- shape --------------------------------------------------------------

    @property
    def size(self) -> int:
        """The number of bands ``K``."""
        return int(self.values.shape[0])

    @property
    def labels(self) -> tuple[str, ...]:
        """Band labels ``"[lower,upper)"`` in band order."""
        return tuple(
            band_label(lo, hi) for lo, hi in zip(self.bands[:-1], self.bands[1:], strict=True)
        )

    # -- inspection ---------------------------------------------------------

    def total(self) -> NDArray[np.float64]:
        """Contacts per person in each band, summed over contacts: ``values.sum(axis=1)``."""
        return np.asarray(self.values.sum(axis=1))

    def population(self, population: ArrayLike | None = None) -> NDArray[np.float64]:
        """``population`` validated against the bands, or :attr:`source_population`."""
        if population is None:
            if self.source_population is None:
                raise ValueError(
                    "This ContactMatrix has no source_population; pass population= "
                    "(one entry per band)."
                )
            return self.source_population
        array = np.asarray(population, dtype=np.float64)
        _check_population(array, self.size, what="population")
        return array

    def mean_contacts(self, population: ArrayLike | None = None) -> float:
        """Population-weighted mean contacts per person, $\\sum_i N_i c_i / \\sum_i N_i$."""
        n = self.population(population)
        return float(np.dot(n, self.total()) / n.sum())

    def reciprocity_error(self, population: ArrayLike | None = None) -> Reciprocity:
        """$c_{ij} N_i - c_{ji} N_j$ and its relative norm; see :class:`Reciprocity`."""
        n = self.population(population)
        flows = self.values * n[:, None]
        diff = flows - flows.T
        scale = float(np.linalg.norm(flows))
        relative = float(np.linalg.norm(diff) / scale) if scale > 0 else 0.0
        return Reciprocity(matrix=diff, relative=relative)

    # -- adaptation (step 18) -----------------------------------------------

    def rebin(
        self,
        target: Sequence[float] | Property,
        *,
        population: ArrayLike,
        population_bands: Sequence[float] | None = None,
    ) -> ContactMatrix:
        """Move the matrix onto other bands, or onto a model's age property.

        ``target`` is a sequence of band edges or a :class:`Property` whose
        trait names are numeric lower bounds (``"0"``, ``"5"``, ``"15"``); the
        last edge is then this matrix's last edge, and the result is aligned to
        that property. See :class:`Rebin` for the assumptions — splitting a band
        assumes contacts are uniform within it — and for what ``population`` and
        ``population_bands`` must be. The result's ``source_population`` is that
        population summed onto the new bands; pass the survey's own population
        so a later :meth:`adapt` starts from the right demography.
        """
        prop: Property | None = None
        if isinstance(target, Property):
            prop = target
            lowers = _numeric_traits(target)
            if lowers is None:
                raise ValueError(
                    f"rebin onto property {target.name!r} needs numeric lower-bound trait "
                    f"names ('0', '5', '15'); got {list(target.traits)}."
                )
            edges: tuple[float, ...] = (*lowers, self.bands[-1])
        else:
            edges = tuple(float(edge) for edge in target)
        plan = Rebin.between(
            self.bands, edges, population=population, population_bands=population_bands
        )
        return replace(
            self,
            values=plan.apply(self.values),
            bands=plan.target,
            prop=prop,
            source_population=plan.population,
        )

    def symmetrise(self, population: ArrayLike | None = None) -> ContactMatrix:
        """Make the matrix reciprocal against ``population`` (default: the source).

        $c'_{ij} = (c_{ij} N_i + c_{ji} N_j) / (2 N_i)$: the total contacts
        between two bands becomes the mean of what each side reported, so
        :meth:`reciprocity_error` is zero afterwards and total contacts
        $\\sum_{ij} N_i c_{ij}$ are conserved. The result's ``source_population``
        is ``population``.
        """
        n = self.population(population)
        flows = self.values * n[:, None]
        return replace(self, values=(flows + flows.T) / (2.0 * n[:, None]), source_population=n)

    def adapt(
        self,
        population: ArrayLike,
        *,
        kind: ContactAdaptationArg = ContactAdaptation.FREQUENCY,
        source: ArrayLike | None = None,
    ) -> ContactMatrix:
        """Carry the matrix from its survey population to another one.

        With population shares $w_j = N_j / \\sum_k N_k$ in the survey
        (``source``, default :attr:`source_population`) and $w'_j$ in the target
        ``population``, both conventions start from the ratio
        $r_j = w'_j / w_j$:

        - ``"density"`` — $c'_{ij} = c_{ij} r_j$. The rate at which two
          *particular* people meet is kept, so contacts with a band grow with
          its share of the population. This is the textbook's chapter 19
          adjustment, and it keeps a reciprocal matrix reciprocal against the
          target.
        - ``"frequency"`` (default) — the density result with each row rescaled
          back to its original sum, $c'_{ij} = c_{ij} r_j \\, c_i / \\sum_k c_{ik} r_k$.
          Each band keeps the number of contacts per person it reported; only
          *who* those contacts are with shifts with the target population. The
          result is generally not reciprocal; :meth:`symmetrise` it if needed.

        The two disagree whenever the populations differ, and both appear in
        the literature; pick deliberately. The result's ``source_population``
        is ``population``.
        """
        resolved = coerce_strenum(ContactAdaptation, kind, what="ContactMatrix.adapt kind")
        src = self.population(source)
        tgt = np.asarray(population, dtype=np.float64)
        _check_population(tgt, self.size, what="population")
        ratio = (tgt / tgt.sum()) / (src / src.sum())
        values = self.values * ratio[None, :]
        if resolved is ContactAdaptation.FREQUENCY:
            sums = values.sum(axis=1)
            factor = np.divide(self.total(), sums, out=np.zeros_like(sums), where=sums > 0)
            values = values * factor[:, None]
        return replace(self, values=values, source_population=tgt)

    @overload
    def scale(self, by: RateOps) -> ScaledContacts: ...

    @overload
    def scale(self, by: float) -> ContactMatrix: ...

    def scale(self, by: RateOps | float) -> ScaledContacts | ContactMatrix:
        """Multiply every contact by ``by``.

        A number gives a new :class:`ContactMatrix`. A rate expression — a
        ``Param`` to calibrate, or ``step(Time(), ...)`` for an intervention
        from a date — gives a :class:`ScaledContacts`, whose
        :meth:`~ScaledContacts.to_mixing` evaluates inside the model.
        """
        if isinstance(by, RateOps):
            return ScaledContacts(((self.setting, self, by),))
        return replace(self, values=self.values * float(by))

    def to_mixing(
        self,
        prop: Property | None = None,
        *,
        normalize: MixingNormalizeArg = MixingNormalize.NONE,
        check_reciprocal: bool = False,
    ) -> MixingMatrix:
        """The :class:`~summer4.epi.MixingMatrix` a force of infection takes.

        ``prop`` defaults to the property this matrix is aligned to.
        **``normalize`` defaults to ``"none"``**, unlike ``MixingMatrix``'s own
        ``"rows"``: an empirical matrix's row sums *are* the contact numbers
        the survey measured, and row-normalising would throw them away (every
        band would then make one contact a day). With ``"none"`` the force of
        infection on band ``i`` is ``contact_rate * sum_j values[i, j] * I_j / N_j``
        under frequency-dependent transmission, so ``contact_rate`` is the
        probability of transmission per contact.
        """
        resolved = _resolve_prop(prop, self.prop, self.bands, what="to_mixing")
        return MixingMatrix(
            resolved, self.values, normalize=normalize, check_reciprocal=check_reciprocal
        )

    # -- export -------------------------------------------------------------

    def to_frame(
        self,
        *,
        row: str = "age_from",
        col: str = "age_to",
        value: str = "contacts",
    ) -> Any:
        """A long pandas frame, one row per (participant, contact) band pair.

        Labels are ``"[lower,upper)"``, so ``from_frame(cm.to_frame())``
        rebuilds ``cm`` (other than ``prop``, ``setting`` and
        ``source_population``, which a frame does not carry).
        """
        pd = _require_pandas()
        labels = self.labels
        k = self.size
        return pd.DataFrame(
            {
                row: [labels[i] for i in range(k) for _ in range(k)],
                col: [labels[j] for _ in range(k) for j in range(k)],
                value: self.values.reshape(-1),
            }
        )

    def to_pandas(self) -> Any:
        """A square pandas frame: index = participant band, columns = contact band.

        This is the plotting seam. ``plotly.express.imshow(cm.to_pandas())``
        draws the heatmap; summer4 does not choose a plotting library for you.
        """
        pd = _require_pandas()
        index = pd.Index(self.labels, name="participant")
        columns = pd.Index(self.labels, name="contact")
        return pd.DataFrame(np.array(self.values), index=index, columns=columns)


def _optional_equal(a: NDArray[np.float64] | None, b: NDArray[np.float64] | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b))


@dataclass(frozen=True, slots=True, eq=False)
class SettingStack(Mapping[str, ContactMatrix]):
    """Per-setting contact matrices that share one set of bands.

    A read-only mapping of setting name to :class:`ContactMatrix`, in the order
    given. Each matrix's ``setting`` is set to its key (a matrix that already
    names a different setting is an error). :meth:`total` sums the settings;
    :meth:`map` applies one transformation to every setting.
    """

    settings: tuple[tuple[str, ContactMatrix], ...]

    def __init__(self, settings: Mapping[str, ContactMatrix]) -> None:
        items = tuple(settings.items())
        if not items:
            raise ValueError("SettingStack needs at least one setting.")
        first_name, first = items[0]
        resolved: list[tuple[str, ContactMatrix]] = []
        for name, matrix in items:
            if not isinstance(matrix, ContactMatrix):
                raise TypeError(
                    f"SettingStack setting {name!r} is a {type(matrix).__name__}, "
                    "not a ContactMatrix."
                )
            if matrix.bands != first.bands:
                raise ValueError(
                    f"SettingStack settings must share bands; {name!r} has {matrix.labels} "
                    f"but {first_name!r} has {first.labels}."
                )
            if matrix.prop != first.prop:
                raise ValueError(
                    f"SettingStack settings must share prop; {name!r} and {first_name!r} differ."
                )
            if matrix.setting is not None and matrix.setting != name:
                raise ValueError(
                    f"SettingStack key {name!r} holds a matrix whose setting is "
                    f"{matrix.setting!r}."
                )
            if matrix.setting is None:
                matrix = ContactMatrix(
                    values=matrix.values,
                    bands=matrix.bands,
                    prop=matrix.prop,
                    setting=name,
                    source_population=matrix.source_population,
                )
            resolved.append((name, matrix))
        object.__setattr__(self, "settings", tuple(resolved))

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        setting: str = "setting",
        row: str = "age_from",
        col: str = "age_to",
        value: str = "contacts",
        bands: Sequence[float] | None = None,
        prop: Property | None = None,
        source_population: ArrayLike | None = None,
    ) -> SettingStack:
        """One long frame with a ``setting`` column, as per-setting survey exports ship.

        Each setting's rows are read with :meth:`ContactMatrix.from_frame`'s rules.
        """
        columns = frame_columns(frame, (setting, row, col, value))
        names = list(dict.fromkeys(columns[setting].tolist()))
        matrices: dict[str, ContactMatrix] = {}
        for name in names:
            keep = columns[setting] == name
            matrices[str(name)] = ContactMatrix._from_columns(
                columns[row][keep],
                columns[col][keep],
                columns[value][keep],
                bands=bands,
                prop=prop,
                setting=str(name),
                source_population=source_population,
            )
        return cls(matrices)

    def __getitem__(self, name: str) -> ContactMatrix:
        for key, matrix in self.settings:
            if key == name:
                return matrix
        raise KeyError(f"Unknown setting {name!r}. Known: {list(self)}.")

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self.settings)

    def __len__(self) -> int:
        return len(self.settings)

    def __repr__(self) -> str:
        return f"SettingStack({list(self)}, {self.bands_labels[0]}..{self.bands_labels[-1]})"

    @property
    def bands(self) -> tuple[float, ...]:
        """The band edges every setting shares."""
        return self.settings[0][1].bands

    @property
    def bands_labels(self) -> tuple[str, ...]:
        """The band labels every setting shares."""
        return self.settings[0][1].labels

    @property
    def prop(self) -> Property | None:
        """The age property every setting shares, if aligned."""
        return self.settings[0][1].prop

    def total(self) -> ContactMatrix:
        """All settings summed element-wise, as one matrix with ``setting=None``.

        ``source_population`` is kept when every setting carries the same one.
        """
        matrices = [matrix for _, matrix in self.settings]
        first = matrices[0].source_population
        same = all(_optional_equal(first, m.source_population) for m in matrices)
        return ContactMatrix(
            values=np.sum([m.values for m in matrices], axis=0),
            bands=self.bands,
            prop=self.prop,
            setting=None,
            source_population=first if same else None,
        )

    def map(self, fn: Callable[[ContactMatrix], ContactMatrix]) -> SettingStack:
        """Apply ``fn`` to every setting and stack the results under the same names."""
        return SettingStack({name: fn(matrix) for name, matrix in self.settings})

    def rebin(
        self,
        target: Sequence[float] | Property,
        *,
        population: ArrayLike,
        population_bands: Sequence[float] | None = None,
    ) -> SettingStack:
        """:meth:`ContactMatrix.rebin` applied to every setting."""
        return self.map(
            lambda cm: cm.rebin(target, population=population, population_bands=population_bands)
        )

    def symmetrise(self, population: ArrayLike | None = None) -> SettingStack:
        """:meth:`ContactMatrix.symmetrise` applied to every setting (so the total is too)."""
        return self.map(lambda cm: cm.symmetrise(population))

    def adapt(
        self,
        population: ArrayLike,
        *,
        kind: ContactAdaptationArg = ContactAdaptation.FREQUENCY,
        source: ArrayLike | None = None,
    ) -> SettingStack:
        """:meth:`ContactMatrix.adapt` applied to every setting.

        Under ``"frequency"`` each setting keeps its own contacts per person,
        so the total does too.
        """
        return self.map(lambda cm: cm.adapt(population, kind=kind, source=source))

    @overload
    def scale(self, by: RateOps, *, settings: Sequence[str] | None = None) -> ScaledContacts: ...

    @overload
    def scale(self, by: float, *, settings: Sequence[str] | None = None) -> SettingStack: ...

    def scale(
        self, by: RateOps | float, *, settings: Sequence[str] | None = None
    ) -> ScaledContacts | SettingStack:
        """Multiply the named ``settings`` (default: all) by ``by``.

        A number gives a new :class:`SettingStack`. A rate expression gives a
        :class:`ScaledContacts` over every setting, with ``by`` on the named
        ones — "schools at 20% from day 60" is
        ``stack.scale(step(Time(), (60.0,), (1.0, 0.2)), settings=["school"])``.
        """
        chosen = _names(list(self), settings)
        if isinstance(by, RateOps):
            return ScaledContacts(
                tuple(
                    (name, matrix, by if name in chosen else 1.0) for name, matrix in self.settings
                )
            )
        return self.map(lambda cm: cm.scale(float(by)) if cm.setting in chosen else cm)

    def to_mixing(
        self,
        prop: Property | None = None,
        *,
        normalize: MixingNormalizeArg = MixingNormalize.NONE,
        check_reciprocal: bool = False,
    ) -> MixingMatrix:
        """``self.total().to_mixing(...)``; see :meth:`ContactMatrix.to_mixing`."""
        return self.total().to_mixing(prop, normalize=normalize, check_reciprocal=check_reciprocal)


def _resolve_prop(
    prop: Property | None, aligned: Property | None, bands: tuple[float, ...], *, what: str
) -> Property:
    resolved = prop if prop is not None else aligned
    if resolved is None:
        raise ValueError(
            f"{what} needs the age property: pass prop=, or build the matrix with prop= "
            "(or rebin it onto the property)."
        )
    _check_prop(resolved, bands, what=what)
    return resolved


def _names(known: Sequence[str | None], settings: Sequence[str] | None) -> set[str | None]:
    if settings is None:
        return set(known)
    if isinstance(settings, str):
        settings = (settings,)
    unknown = [name for name in settings if name not in known]
    if unknown:
        raise KeyError(f"Unknown settings {unknown}. Known: {[k for k in known if k is not None]}.")
    return set(settings)


@dataclass(frozen=True, slots=True, eq=False)
class ScaledContacts:
    """A sum of contact matrices, each times a factor that can vary.

    ``terms`` holds ``(setting, matrix, factor)``; the contacts are
    $\\sum_s f_s C_s$. A factor is a number or any rate expression — a ``Param``
    (calibratable) or ``step(Time(), ...)`` (an intervention from a date).
    :meth:`matrix` is that sum as one rate expression; :meth:`to_mixing` hands
    it to :class:`~summer4.epi.MixingMatrix`, so the matrix is evaluated inside
    the model, under ``jit``.

    The expression is built from constants: the matrices whose factor is a
    plain number fold into one constant array, and each varying setting adds
    one multiply and one add. Its size depends on the number of *varying
    settings*, never on the number of bands.
    """

    terms: tuple[tuple[str | None, ContactMatrix, RateOps | float], ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("ScaledContacts needs at least one term.")
        first = self.terms[0][1]
        for name, matrix, _ in self.terms:
            if matrix.bands != first.bands:
                raise ValueError(
                    f"ScaledContacts terms must share bands; {name!r} has {matrix.labels}."
                )

    @property
    def bands(self) -> tuple[float, ...]:
        """The band edges every term shares."""
        return self.terms[0][1].bands

    @property
    def prop(self) -> Property | None:
        """The age property of the first term, if aligned."""
        return self.terms[0][1].prop

    def scale(
        self, by: RateOps | float, *, settings: Sequence[str] | None = None
    ) -> ScaledContacts:
        """Multiply the named settings' factors (default: all) by ``by``."""
        chosen = _names([name for name, _, _ in self.terms], settings)
        return ScaledContacts(
            tuple(
                (name, matrix, _times(factor, by) if name in chosen else factor)
                for name, matrix, factor in self.terms
            )
        )

    def matrix(self) -> RateOps:
        """$\\sum_s f_s C_s$ as one ``(K, K)`` rate expression."""
        fixed = np.zeros_like(self.terms[0][1].values)
        varying: list[tuple[ContactMatrix, RateOps]] = []
        for _, matrix, factor in self.terms:
            if isinstance(factor, RateOps):
                varying.append((matrix, factor))
            else:
                fixed = fixed + float(factor) * matrix.values
        expr: RateOps | None = ArrayConst(fixed) if np.any(fixed) or not varying else None
        for matrix, factor in varying:
            term = ArrayConst(matrix.values) * factor
            expr = term if expr is None else expr + term
        assert expr is not None
        return expr

    def to_mixing(
        self,
        prop: Property | None = None,
        *,
        normalize: MixingNormalizeArg = MixingNormalize.NONE,
        check_reciprocal: bool = False,
    ) -> MixingMatrix:
        """A :class:`~summer4.epi.MixingMatrix` over :meth:`matrix`.

        As :meth:`ContactMatrix.to_mixing`: ``normalize`` defaults to
        ``"none"`` so the survey's contact numbers survive. With
        ``check_reciprocal=True`` the check runs on every evaluation, through
        a host callback; leave it off inside calibration.
        """
        resolved = _resolve_prop(prop, self.prop, self.bands, what="to_mixing")
        return MixingMatrix(
            resolved, self.matrix(), normalize=normalize, check_reciprocal=check_reciprocal
        )


def _times(factor: RateOps | float, by: RateOps | float) -> RateOps | float:
    if isinstance(factor, RateOps) or isinstance(by, RateOps):
        if not isinstance(factor, RateOps) and float(factor) == 1.0:
            return by
        return as_rate(factor) * by
    return float(factor) * float(by)
