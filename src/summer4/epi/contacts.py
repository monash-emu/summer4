"""Empirical contact matrices: load, validate and inspect contact-survey data.

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

Everything here is host-side NumPy. Nothing in this module is traced.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from summer4.data import _require_pandas, frame_columns
from summer4.properties import Property

__all__ = ["ContactMatrix", "Reciprocity", "SettingStack", "band_label"]

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
