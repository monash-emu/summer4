"""Named outputs as a DAG over save specs, other outputs, and parameters.

Leaves are save specs (:class:`~summer4.results.plan.Compartments`,
:class:`~summer4.results.plan.FlowMass`, and the other quantities). A chain
such as ``Compartments().sum_by()`` or ``FlowMass("infection").midpoint()`` is
still that leaf: the reduction runs in :meth:`OutputSet.evaluate`, not in the
solver. Interior nodes are arithmetic and the same post-ops on
:meth:`OutputSet.ref`. :meth:`OutputSet.plan` contributes only the distinct
leaves, so a calibration run does not materialise series it never saves.

The DAG is static. Evaluating it inside ``jit`` unrolls one step per named
output; the arithmetic itself is traced. ``cumsum(start=)`` and
``cumulative_integral()`` still need a concrete time axis, as
:meth:`~summer4.results.output.Output.cumsum` does.
"""

from __future__ import annotations

import itertools
import string
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from summer4.flows.rates import RateOps
from summer4.jax.propertydata import PropertyData
from summer4.properties import Property, Trait
from summer4.results.output import Output
from summer4.results.plan import (
    Compartments,
    FlowMass,
    GroupedOutput,
    Quantity,
    SavedExpr,
    SavePlan,
    SaveRequest,
)
from summer4.results.result import Result
from summer4.selectors import Dest, Selector, Source

_LEAF_TYPES = (Compartments, FlowMass, SavedExpr, GroupedOutput)
_CALLS = frozenset(
    {
        "select",
        "sum_by",
        "rolling",
        "cumsum",
        "cumulative_integral",
        "midpoint",
        "integrate_intervals",
        "integrate",
        "at",
        "at_times",
    }
)
type _Kind = Literal[
    "leaf",
    "ref",
    "const",
    "param",
    "unary",
    "binary",
    "call",
    "defer",
    "array",
    "scalar",
    "with_values",
]


def _is_numeric(value: object) -> bool:
    import numpy as np

    if isinstance(value, (bool, int, float, np.generic, np.ndarray)):
        return True
    module = type(value).__module__
    return module.startswith("jax") or module.startswith("jaxlib")


@dataclass(frozen=True, slots=True)
class OutputExpr:
    """One node of an :class:`OutputSet` expression."""

    kind: _Kind
    quantity: Quantity | None = None
    name: str | None = None
    value: Any = None
    param: Any = None
    op: str | None = None
    method: str | None = None
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    kids: tuple[OutputExpr, ...] = ()

    @staticmethod
    def leaf(quantity: object) -> OutputExpr:
        if not isinstance(quantity, _LEAF_TYPES):
            raise TypeError(f"An output leaf must be a save spec, got {type(quantity).__name__}.")
        return OutputExpr(kind="leaf", quantity=quantity)

    @staticmethod
    def ref(name: str) -> OutputExpr:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Output name must be a non-empty string, got {name!r}.")
        return OutputExpr(kind="ref", name=name)

    def _call(self, method: str, /, *args: Any, **kwargs: Any) -> OutputExpr:
        if method not in _CALLS:
            raise AttributeError(method)
        return OutputExpr(
            kind="call",
            method=method,
            args=args,
            kwargs=tuple(kwargs.items()),
            kids=(self,),
        )

    def select(self, sel: Any) -> OutputExpr:
        return self._call("select", sel)

    def sum_by(self, *props: Any, side: Any = None) -> OutputExpr:
        """Sum across the strata, keeping time and ``props``; see :meth:`Output.sum_by`."""
        if side is None:
            return self._call("sum_by", *props)
        return self._call("sum_by", *props, side=side)

    def rolling(
        self,
        window: int,
        *,
        how: Any = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ) -> OutputExpr:
        return self._call("rolling", window, how=how, center=center, min_periods=min_periods)

    def cumsum(self, *, start: Any = None, end: Any = None) -> OutputExpr:
        """Running sum of amounts per interval; see :meth:`Output.cumsum`."""
        return self._call("cumsum", start=start, end=end)

    def cumulative_integral(
        self, *, start: Any = None, end: Any = None, method: Literal["trapezoid"] = "trapezoid"
    ) -> OutputExpr:
        """Running integral of a rate; see :meth:`Output.cumulative_integral`."""
        return self._call("cumulative_integral", start=start, end=end, method=method)

    def midpoint(self) -> OutputExpr:
        return self._call("midpoint")

    def integrate_intervals(
        self, method: Literal["trapezoid", "simpson"] = "trapezoid"
    ) -> OutputExpr:
        return self._call("integrate_intervals", method)

    def integrate(self, method: Literal["trapezoid", "simpson"] = "trapezoid") -> OutputExpr:
        return self._call("integrate", method)

    def at(self, when: Any) -> OutputExpr:
        """Value at the save time nearest ``when``; drops ``time``. See :meth:`Output.at`."""
        return self._call("at", when)

    def at_times(self, ts: Any) -> OutputExpr:
        return self._call("at_times", ts)

    def array(self) -> OutputExpr:
        """The evaluated values as a plain array; see :meth:`Output.array`.

        The node evaluates to an array, not an ``Output``. It combines with
        numbers, other arrays, ``Param`` expressions, NumPy ufuncs and
        ``defer``; :meth:`with_values` gives it an output's axes again.
        """
        return OutputExpr(kind="array", kids=(self,))

    def scalar(self) -> OutputExpr:
        """Like :meth:`array`, for a 0-d output; checks the shape is ``()``."""
        return OutputExpr(kind="scalar", kids=(self,))

    def with_values(self, values: object) -> OutputExpr:
        """This output's times, dims and map with ``values``; see :meth:`Output.with_values`."""
        return OutputExpr(kind="with_values", kids=(self, _coerce(values)))

    def __neg__(self) -> OutputExpr:
        return OutputExpr(kind="unary", op="neg", kids=(self,))

    def __abs__(self) -> OutputExpr:
        return OutputExpr(kind="unary", op="abs", kids=(self,))

    def __add__(self, other: object) -> OutputExpr:
        return _binary("add", self, other)

    def __radd__(self, other: object) -> OutputExpr:
        return _binary("add", other, self)

    def __sub__(self, other: object) -> OutputExpr:
        return _binary("sub", self, other)

    def __rsub__(self, other: object) -> OutputExpr:
        return _binary("sub", other, self)

    def __mul__(self, other: object) -> OutputExpr:
        return _binary("mul", self, other)

    def __rmul__(self, other: object) -> OutputExpr:
        return _binary("mul", other, self)

    def __truediv__(self, other: object) -> OutputExpr:
        return _binary("div", self, other)

    def __rtruediv__(self, other: object) -> OutputExpr:
        return _binary("div", other, self)

    def __pow__(self, other: object) -> OutputExpr:
        return _binary("pow", self, other)

    def __rpow__(self, other: object) -> OutputExpr:
        return _binary("pow", other, self)

    # Ahead of ndarray's own operators, so ``ndarray * ref`` reaches
    # ``__array_ufunc__`` rather than an object-array loop.
    __array_priority__: ClassVar[int] = 1000

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from summer4.flows.algebra import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs, mode="output_expr")

    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from summer4.flows.algebra import dispatch_array_function

        return dispatch_array_function(func, types, args, kwargs, mode="output_expr")


def _coerce(value: object) -> OutputExpr:
    if isinstance(value, OutputExpr):
        return value
    if isinstance(value, _LEAF_TYPES):
        return OutputExpr.leaf(value)
    if isinstance(value, RateOps):
        from summer4.flows.stages import rate_stage

        # A parameter-only expression is applied after the solve. One that reads
        # time, state or flows has no value there, so it is saved during the solve.
        if rate_stage(value) == "step":
            return OutputExpr.leaf(SavedExpr(value))
        return OutputExpr(kind="param", param=value)
    if _is_numeric(value):
        return OutputExpr(kind="const", value=value)
    raise TypeError(
        "An output expression combines save specs, output refs, numbers, arrays, "
        f"and parameter expressions. Got {type(value).__name__}."
    )


def _binary(op: str, left: object, right: object) -> OutputExpr:
    return OutputExpr(kind="binary", op=op, kids=(_coerce(left), _coerce(right)))


def _ref_list(expr: OutputExpr) -> list[str]:
    """Names this expression reads, in walk order, without duplicates."""
    found: list[str] = []
    if expr.kind == "ref" and expr.name is not None:
        found.append(expr.name)
    for kid in expr.kids:
        for name in _ref_list(kid):
            if name not in found:
                found.append(name)
    return found


def _leaves(expr: OutputExpr) -> list[Quantity]:
    if expr.kind == "leaf":
        if expr.quantity is None:
            raise TypeError("Output leaf is missing its save spec.")
        return [expr.quantity]
    found: list[Quantity] = []
    for kid in expr.kids:
        found.extend(_leaves(kid))
    return found


def _same(left: Quantity, right: Quantity) -> bool:
    return left == right


def _apply_deferred(fn: Callable[..., Any], args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Call a ``defer`` function on evaluated arguments and apply its return contract.

    An :class:`Output` is returned as is. Any other value must be an array shaped
    like the first :class:`Output` argument; it is wrapped with that argument's
    times and dims. With no :class:`Output` argument (only arrays from
    :meth:`OutputExpr.array`, numbers and parameters) the value is returned as is.
    """
    import jax.numpy as jnp

    result = fn(*args, **kwargs)
    if isinstance(result, Output):
        return result
    label = getattr(fn, "__name__", repr(fn))
    first = next((value for value in (*args, *kwargs.values()) if isinstance(value, Output)), None)
    if first is None:
        return result
    if isinstance(first.values, PropertyData):
        want = tuple(jnp.shape(first.values.data))
    else:
        want = tuple(jnp.shape(first.values))
    got = tuple(jnp.shape(result))
    if got != want:
        raise TypeError(
            f"defer({label}) returned an array of shape {got}; it must return an Output "
            f"or an array of the first Output argument's shape {want}."
        )
    if isinstance(first.values, PropertyData):
        return first._with(values=PropertyData(first.values.pmap, result))
    return first._with(values=result)


def _as_named_array(name: str, value: Any) -> Any:
    """An array-valued named output, as a JAX array (a pytree leaf of the Result)."""
    import jax.numpy as jnp

    if not _is_numeric(value):
        raise TypeError(
            f"Output {name!r} evaluated to {type(value).__name__}, not an Output or an array."
        )
    return jnp.asarray(value)


type FlowValues = Literal["rate", "midpoint"]
type Naming = str | Callable[[str, tuple[Trait, ...]], str]
type By = Property | Sequence[Property | tuple[Property, ...]]

DEFAULT_NAMING = "X{prop}_{trait}"
"""summer2 port naming: ``prevXage_15Xreach_reachable``."""


def _groupings(by: By) -> list[tuple[Property, ...]]:
    if isinstance(by, Property):
        return [(by,)]
    groupings: list[tuple[Property, ...]] = []
    for item in by:
        grouping = (item,) if isinstance(item, Property) else tuple(item)
        if not grouping or not all(isinstance(p, Property) for p in grouping):
            raise TypeError(
                "by= takes a Property, or a sequence of Properties and tuples of Properties; "
                f"got {item!r}."
            )
        groupings.append(grouping)
    return groupings


_SUFFIX_FIELDS = frozenset({"prop", "trait"})


def _template_fields(template: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}


def _stratum_name(naming: Naming, name: str, traits: tuple[Trait, ...]) -> str:
    """One stratum's output name.

    ``naming`` is a callable ``(name, traits) -> str``; a **suffix template**
    using only ``{prop}`` and ``{trait}``, formatted once per property and
    appended to ``name`` (``"X{prop}_{trait}"``); or a **whole-name template**
    using ``{name}`` and property names, each replaced by that property's trait
    (``"{name}_from_{state}_age_{age}"``).
    """
    if callable(naming):
        return naming(name, traits)
    fields = _template_fields(naming)
    if fields <= _SUFFIX_FIELDS:
        return name + "".join(naming.format(prop=t.property, trait=t.name) for t in traits)
    values = {t.property: t.name for t in traits}
    values["name"] = name
    missing = sorted(fields - set(values))
    if missing:
        raise KeyError(
            f"naming template {naming!r} names {missing}, which this grouping does not have. "
            f"Known fields: {sorted(values)}."
        )
    return naming.format(**values)


class OutputSet:
    """Declarative named outputs. Assign leaves and expressions; evaluate after a solve.

    .. code-block:: python

        outputs = OutputSet()
        outputs["population"] = Compartments()
        outputs["incidence"] = FlowMass("infection").midpoint()
        outputs["per_capita"] = outputs.ref("incidence") / outputs.ref("population")
        outputs["scaled"] = outputs.ref("incidence") * Param("scale")
        outputs["notified"] = defer(delayed)(outputs.ref("incidence"), Param("delay"))

        result = model.run(params, y0, save=outputs.plan(SavePlan(ts=times)))
        named = outputs.evaluate(result, params)

    ``plan`` saves each distinct leaf once. The save key is the first output
    name that uses that spec; a second spec in the same expression is
    ``name__2``. ``group_by`` on a save spec is that spec's save-time field,
    not a post-op; group afterwards with ``outputs.ref(name).sum_by(...)``.
    A name whose expression is only post-ops on that leaf still
    uses the name as the save key, and the solved ``Result`` holds the *raw*
    leaf there. :meth:`evaluate` is what returns the named values, including
    those post-ops. Do not read a post-processed name off the solve result.

    Two names that share a spec share one save. A key already in ``base`` is
    kept, including its times, when the quantity matches, and rejected when it
    does not. Derived names are not save keys.
    """

    def __init__(
        self,
        *,
        flow_values: FlowValues = "rate",
        naming: Naming = DEFAULT_NAMING,
    ) -> None:
        if flow_values not in ("rate", "midpoint"):
            raise ValueError(f"flow_values must be 'rate' or 'midpoint', got {flow_values!r}.")
        self._order: list[str] = []
        self._exprs: dict[str, OutputExpr] = {}
        self._hidden: set[str] = set()
        self.flow_values: FlowValues = flow_values
        self.naming: Naming = naming

    def __setitem__(self, name: str, value: Quantity | OutputExpr | RateOps) -> None:
        self.set(name, value)

    def set(
        self, name: str, value: Quantity | OutputExpr | RateOps, *, keep: bool = True
    ) -> OutputExpr:
        """Assign ``name`` and return its expression; ``outputs[name] = value`` is ``keep=True``.

        ``keep=False`` makes ``name`` an intermediate (summer2's
        ``save_results=False``): other outputs can read it with :meth:`ref`, and
        targets can score it, but :meth:`evaluate` does not return it and
        :meth:`kept` does not list it, so ``posterior_runs`` does not record it.
        Assigning a name again replaces its expression and its ``keep``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"Output name must be a non-empty string, got {name!r}.")
        if isinstance(value, OutputExpr):
            expr = value
        elif isinstance(value, _LEAF_TYPES):
            expr = OutputExpr.leaf(value)
        elif isinstance(value, RateOps):
            expr = _coerce(value)
            if expr.kind == "param":
                raise ValueError(
                    f"Output {name!r} is a constant (it reads only parameters), not a time "
                    "series. Multiply a saved quantity by it, or save an expression that "
                    "reads time, state or flows."
                )
        else:
            raise TypeError(
                f"Output {name!r} must be a save spec, a rate expression, or an expression "
                f"built from outputs.ref(...). Got {type(value).__name__}."
            )
        if name not in self._exprs:
            self._order.append(name)
        self._exprs[name] = expr
        if keep:
            self._hidden.discard(name)
        else:
            self._hidden.add(name)
        return expr

    def __getitem__(self, name: str) -> OutputExpr:
        try:
            return self._exprs[name]
        except KeyError:
            known = list(self._order)
            raise KeyError(f"OutputSet has no output {name!r}. Known: {known}.") from None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._exprs

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def keys(self) -> tuple[str, ...]:
        """Every name, kept or not, in assignment order."""
        return tuple(self._order)

    def kept(self) -> tuple[str, ...]:
        """The names :meth:`evaluate` returns by default: those not assigned with ``keep=False``."""
        return tuple(name for name in self._order if name not in self._hidden)

    def ref(self, name: str) -> OutputExpr:
        """Refer to a named output, including one assigned later."""
        return OutputExpr.ref(name)

    def subset(self, names: Iterable[str]) -> OutputSet:
        """A copy in which only ``names`` are kept; every other name becomes an intermediate.

        The expressions and save leaves are shared, so the copy solves for
        exactly what this set does. Use it where a whole set is recorded or
        scored, such as ``posterior_runs(outputs=...)``.
        """
        wanted = list(dict.fromkeys(names))
        missing = [name for name in wanted if name not in self._exprs]
        if missing:
            raise KeyError(f"OutputSet has no outputs {missing}. Known: {list(self._order)}.")
        copy = OutputSet(flow_values=self.flow_values, naming=self.naming)
        keep = set(wanted)
        for name in self._order:
            copy.set(name, self._exprs[name], keep=name in keep)
        return copy

    # --- requests: summer2-style one-call outputs ---------------------------------------
    #
    # Each request is one ``set`` call; the docstring gives the expression it
    # assigns. Assign that expression yourself whenever a request does not fit.

    def compartments(
        self,
        name: str,
        where: Selector | None = None,
        *,
        by: By = (),
        naming: Naming | None = None,
        keep: bool = True,
    ) -> OutputExpr:
        """Total size of the compartments matching ``where``, per save time.

        Stands for (summer2 ``request_output_for_compartments``)::

            outputs.set(name, Compartments().select(where).sum_by(), keep=keep)

        Every compartment request reads the one shared ``Compartments()`` save.
        ``by`` adds one output per stratum, named by ``naming`` (default: the
        set's); see :meth:`fan_out`.
        """
        leaf = OutputExpr.leaf(Compartments())
        base = leaf if where is None else leaf.select(where)
        expr = self.set(name, base.sum_by(), keep=keep)
        self._fan_out(name, base, by, side=None, post=None, naming=naming, keep=keep)
        return expr

    def flow(
        self,
        name: str,
        flow: str | tuple[str, ...],
        *,
        source: Selector | None = None,
        dest: Selector | None = None,
        values: FlowValues | None = None,
        by: By = (),
        side: Literal["source", "dest"] = "source",
        naming: Naming | None = None,
        keep: bool = True,
    ) -> OutputExpr:
        """Total of ``flow`` over the edges whose source matches ``source`` and dest ``dest``.

        Stands for (summer2 ``request_output_for_flow``)::

            sel = Source(source) & Dest(dest)          # whichever are given
            expr = FlowMass(flow).select(sel).sum_by()
            expr = expr.midpoint()                     # only when values="midpoint"
            outputs.set(name, expr, keep=keep)

        ``values`` defaults to the set's ``flow_values``: ``"rate"`` is the
        instantaneous rate at each save time (people per unit time);
        ``"midpoint"`` is summer2's flow-output convention. Every request on one
        flow reads one shared ``FlowMass(flow)`` save. ``by`` adds one output per
        stratum of the ``side`` compartment, named by ``naming`` (default: the
        set's); see :meth:`fan_out`. One request gives summer2-port names such as
        ``all_cause_mortality_from_<state>_age_<age>``::

            outputs.flow("all_cause_mortality", "all_cause_mortality", by=[(state, age)],
                         naming="{name}_from_{state}_age_{age}")
        """
        values = self.flow_values if values is None else values
        if values not in ("rate", "midpoint"):
            raise ValueError(f"values must be 'rate' or 'midpoint', got {values!r}.")
        leaf = OutputExpr.leaf(FlowMass(flow))
        parts: list[Selector] = []
        if source is not None:
            parts.append(Source(source))
        if dest is not None:
            parts.append(Dest(dest))
        base = leaf
        if parts:
            sel = parts[0] if len(parts) == 1 else parts[0] & parts[1]
            base = leaf.select(sel)
        post: Literal["midpoint"] | None = "midpoint" if values == "midpoint" else None
        total = base.sum_by()
        expr = self.set(name, total.midpoint() if post else total, keep=keep)
        self._fan_out(name, base, by, side=side, post=post, naming=naming, keep=keep)
        return expr

    def aggregate(self, name: str, sources: Sequence[str], *, keep: bool = True) -> OutputExpr:
        """Sum of the named outputs ``sources``.

        Stands for (summer2 ``request_aggregate_output``)::

            outputs.set(name, sum(outputs.ref(s) for s in sources), keep=keep)
        """
        if isinstance(sources, str):
            raise TypeError("aggregate() takes a sequence of output names, not one string.")
        names = list(sources)
        if not names:
            raise ValueError(
                f"aggregate({name!r}) needs at least one source. summer2 gave zeros for an "
                "empty list; write 0.0 * outputs.ref(<any output>) for that."
            )
        expr = self.ref(names[0])
        for other in names[1:]:
            expr = expr + self.ref(other)
        return self.set(name, expr, keep=keep)

    def cumulative_integral(
        self, name: str, source: str, *, start: Any = None, keep: bool = True
    ) -> OutputExpr:
        """Running integral of the rate ``source`` from ``start`` (a save time).

        Stands for (summer2 ``request_cumulative_output`` on a rate)::

            outputs.set(name, outputs.ref(source).cumulative_integral(start=start), keep=keep)

        Use it on a rate, such as a ``"rate"`` flow output. For amounts per
        interval (summer2-style ``"midpoint"`` flow values), use :meth:`cumsum`.
        """
        return self.set(name, self.ref(source).cumulative_integral(start=start), keep=keep)

    def cumsum(self, name: str, source: str, *, start: Any = None, keep: bool = True) -> OutputExpr:
        """Running sum of the amounts ``source`` from ``start`` (a save time).

        Stands for (summer2 ``request_cumulative_output`` on amounts)::

            outputs.set(name, outputs.ref(source).cumsum(start=start), keep=keep)

        Use it on amounts per interval, such as summer2-style ``"midpoint"``
        flow values; on a rate, use :meth:`cumulative_integral`.
        """
        return self.set(name, self.ref(source).cumsum(start=start), keep=keep)

    def function(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        keep: bool = True,
        **kwargs: Any,
    ) -> OutputExpr:
        """``fn`` applied after the solve to outputs and parameters.

        Stands for (summer2 ``request_function_output``)::

            outputs.set(name, defer(fn)(*args, **kwargs), keep=keep)

        with every string argument replaced by ``outputs.ref(string)``. Other
        arguments are passed as ``defer`` takes them: ``Param``, numbers,
        arrays and output expressions. ``fn`` follows ``defer``'s return
        contract (an ``Output``, or an array shaped like the first one).
        """
        from summer4.flows.rates import defer

        def lift(arg: Any) -> Any:
            return self.ref(arg) if isinstance(arg, str) else arg

        deferred = defer(fn)(*(lift(a) for a in args), **{k: lift(v) for k, v in kwargs.items()})
        return self.set(name, deferred, keep=keep)

    def fan_out(
        self,
        name: str,
        expr: OutputExpr,
        by: By,
        *,
        side: Literal["source", "dest"] | None = None,
        naming: Naming | None = None,
        keep: bool = True,
    ) -> tuple[str, ...]:
        """One output per stratum of ``expr``, named with ``naming`` (default: the set's).

        ``expr`` is an output with a compartment or edge axis, such as
        ``Compartments().select(where)`` or ``FlowMass("f")`` (``side`` picks
        the edge end for a flow). ``by`` is a property or a sequence of
        groupings, each a property or a tuple of properties:
        ``by=[age, reach, (age, reach)]`` adds ``{name}Xage_0``, …,
        ``{name}Xreach_r``, …, and ``{name}Xage_0Xreach_r``, …. Each grouping
        is one hidden ``sum_by`` output, and each stratum a row of it::

            outputs.set(f"{name}[age]", expr.sum_by(age), keep=False)
            outputs.set(f"{name}Xage_0", outputs.ref(f"{name}[age]").select(age["0"]).sum_by())

        ``naming`` is one of:

        - a **suffix template** using only ``{prop}`` and ``{trait}``, formatted
          once per property and appended to ``name`` (the default
          ``"X{prop}_{trait}"``);
        - a **whole-name template** using ``{name}`` and the grouping's property
          names, each replaced by that property's trait:
          ``"{name}_from_{state}_age_{age}"`` with ``by=[(state, age)]``;
        - a callable ``(name, traits) -> str``.

        ``naming=None`` uses the set's ``naming``, so one set can mix requests
        named in different styles without a loop. Two strata given the same
        name raise ``ValueError``. Returns the names added.
        """
        return self._fan_out(name, expr, by, side=side, post=None, naming=naming, keep=keep)

    def _fan_out(
        self,
        name: str,
        expr: OutputExpr,
        by: By,
        *,
        side: Literal["source", "dest"] | None,
        post: Literal["midpoint"] | None,
        naming: Naming | None = None,
        keep: bool,
    ) -> tuple[str, ...]:
        naming = self.naming if naming is None else naming
        added: list[str] = []
        for grouping in _groupings(by):
            group_name = f"{name}[{','.join(p.name for p in grouping)}]"
            grouped = expr.sum_by(*grouping) if side is None else expr.sum_by(*grouping, side=side)
            if post == "midpoint":
                grouped = grouped.midpoint()
            self.set(group_name, grouped, keep=False)
            for combo in itertools.product(*(p.traits for p in grouping)):
                traits = tuple(p.trait(t) for p, t in zip(grouping, combo, strict=True))
                sel: Selector = traits[0]
                for trait in traits[1:]:
                    sel = sel & trait
                stratum = _stratum_name(naming, name, traits)
                if stratum in added:
                    raise ValueError(
                        f"naming gives two strata of {name!r} the name {stratum!r}; "
                        "name every grouping property in the template."
                    )
                self.set(stratum, self.ref(group_name).select(sel).sum_by(), keep=keep)
                added.append(stratum)
        return tuple(added)

    def _bindings(self) -> list[tuple[Quantity, str]]:
        bound: list[tuple[Quantity, str]] = []
        used: set[str] = set()
        for name in self._order:
            for qty in _leaves(self._exprs[name]):
                if any(_same(qty, existing) for existing, _key in bound):
                    continue
                key = name
                n = 2
                while key in used:
                    key = f"{name}__{n}"
                    n += 1
                used.add(key)
                bound.append((qty, key))
        return bound

    def _topo(self) -> list[str]:
        visiting: set[str] = set()
        visited: set[str] = set()
        order: list[str] = []
        stack: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle = stack[stack.index(name) :] + [name]
                raise ValueError("OutputSet cycle: " + " -> ".join(cycle) + ".")
            if name not in self._exprs:
                known = list(self._order)
                raise KeyError(f"OutputSet has no output {name!r}. Known: {known}.")
            visiting.add(name)
            stack.append(name)
            for dep in _ref_list(self._exprs[name]):
                visit(dep)
            stack.pop()
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self._order:
            visit(name)
        return order

    def plan(self, base: SavePlan | None = None) -> SavePlan:
        """Return ``base`` plus one request per distinct leaf.

        Derived names are omitted. An existing request is kept when its
        quantity equals the leaf, so :meth:`~summer4.results.targets.TargetSet.plan`
        times survive ``outputs.plan(targets.plan(base))`` and the reverse.
        """
        self._topo()
        base = SavePlan() if base is None else base
        requests = dict(base.requests)
        for qty, key in self._bindings():
            if key in requests and not _same(requests[key].what, qty):
                raise ValueError(
                    f"OutputSet save key {key!r} is already {requests[key].what!r}, "
                    f"not {qty!r}."
                )
            if key not in requests:
                requests[key] = SaveRequest(qty)
        return SavePlan(
            requests=requests,
            ts=base.ts,
            dense=base.dense,
            solver_stats=base.solver_stats,
        )

    def evaluate(
        self,
        result: Result,
        params: object | None = None,
        *,
        names: Iterable[str] | None = None,
    ) -> Result:
        """Compute the kept outputs (or ``names``) from ``result`` and ``params``.

        Only ``names`` and the outputs they read are computed; ``names``
        defaults to :meth:`kept`. Pass ``names=outputs.keys()`` to get the
        intermediates as well.

        ``result`` must contain the leaf keys from :meth:`plan`. ``params`` is
        the object :func:`~summer4.flows.compiled.eval_closed` reads for a
        ``Param`` or other parameter-only rate expression. The returned
        :class:`~summer4.results.result.Result` is keyed by output name, in
        assignment order, and is safe to build under ``jit``. A name whose
        expression evaluates to a plain array (built with
        :meth:`OutputExpr.array` or :meth:`OutputExpr.scalar`) is in
        ``Result.arrays``; ``named[name]`` finds either kind.
        """
        order = self._topo()
        bindings = self._bindings()
        wanted = self.kept() if names is None else tuple(dict.fromkeys(names))
        unknown = [name for name in wanted if name not in self._exprs]
        if unknown:
            raise KeyError(f"OutputSet has no outputs {unknown}. Known: {list(self._order)}.")
        needed: set[str] = set()
        pending = list(wanted)
        while pending:
            current = pending.pop()
            if current not in needed:
                needed.add(current)
                pending.extend(_ref_list(self._exprs[current]))
        order = [name for name in order if name in needed]

        def key_for(qty: Quantity) -> str:
            for existing, key in bindings:
                if _same(existing, qty):
                    return key
            raise KeyError(f"No save key for leaf {qty!r}.")

        computed: dict[str, Any] = {}

        def eval_node(expr: OutputExpr) -> Any:
            if expr.kind == "leaf":
                if expr.quantity is None:
                    raise TypeError("Output leaf is missing its save spec.")
                key = key_for(expr.quantity)
                if key not in result:
                    known = list(result.keys())
                    raise KeyError(
                        f"Solved result has no leaf {key!r}. "
                        f"Run with save=outputs.plan(...). Known keys: {known}."
                    )
                return result[key]
            if expr.kind == "ref":
                if expr.name is None:
                    raise TypeError("Output ref is missing its name.")
                try:
                    return computed[expr.name]
                except KeyError:
                    raise KeyError(f"Output {expr.name!r} is not evaluated yet.") from None
            if expr.kind == "const":
                return expr.value
            if expr.kind == "param":
                if params is None:
                    raise ValueError(
                        "This output multiplies by a parameter, but evaluate() "
                        "was given params=None."
                    )
                from summer4.flows.compiled import eval_closed

                return eval_closed(expr.param, params)
            if expr.kind == "unary":
                child = eval_node(expr.kids[0])
                if isinstance(child, Output):
                    return child._map_unary(expr.op or "")
                from summer4.flows.algebra import apply_unary

                return apply_unary(expr.op or "", child)
            if expr.kind == "binary":
                left = eval_node(expr.kids[0])
                right = eval_node(expr.kids[1])
                if isinstance(left, Output) or isinstance(right, Output):
                    return Output._combine(left, right, expr.op or "")
                from summer4.flows.algebra import apply_binary

                return apply_binary(expr.op or "", left, right)
            if expr.kind == "call":
                child = eval_node(expr.kids[0])
                if not isinstance(child, Output):
                    raise TypeError(
                        f"{expr.method}() applies to an Output, got {type(child).__name__}."
                    )
                if expr.method not in _CALLS:
                    raise AttributeError(expr.method)
                fn: Callable[..., Output] = getattr(child, expr.method)
                return fn(*expr.args, **dict(expr.kwargs))
            if expr.kind in ("array", "scalar"):
                child = eval_node(expr.kids[0])
                if not isinstance(child, Output):
                    raise TypeError(
                        f"{expr.kind}() applies to an Output, got {type(child).__name__}."
                    )
                return child.array() if expr.kind == "array" else child.scalar()
            if expr.kind == "with_values":
                template = eval_node(expr.kids[0])
                if not isinstance(template, Output):
                    raise TypeError(
                        "with_values() applies to an Output, got " f"{type(template).__name__}."
                    )
                return template.with_values(eval_node(expr.kids[1]))
            if expr.kind == "defer":
                values = [eval_node(kid) for kid in expr.kids]
                n_kwargs = len(expr.kwargs)
                n_pos = len(values) - n_kwargs
                kw_names = [key for key, _none in expr.kwargs]
                kw_values = dict(zip(kw_names, values[n_pos:], strict=True))
                return _apply_deferred(expr.value, values[:n_pos], kw_values)
            raise TypeError(f"Unknown output expression {expr.kind!r}.")

        for name in order:
            value = eval_node(self._exprs[name])
            if not isinstance(value, Output):
                value = _as_named_array(name, value)
            computed[name] = value
        returned = [name for name in self._order if name in set(wanted)]
        ordered = {name: computed[name] for name in returned if isinstance(computed[name], Output)}
        arrays = {
            name: computed[name] for name in returned if not isinstance(computed[name], Output)
        }
        return Result(
            times=result.times,
            outputs=ordered,
            arrays=arrays,
            solver=result.solver,
            _state_pmap=result._state_pmap,
            final_state=result.final_state,
            final_time=result.final_time,
        )
