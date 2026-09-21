"""Named outputs as a DAG over save specs, other outputs, and parameters.

Leaves are save specs (:class:`~summer4.results.plan.Compartments`,
:class:`~summer4.results.plan.FlowMass`, and the other quantities). A chain
such as ``Compartments().total()`` or ``FlowMass("infection").midpoint()`` is
still that leaf: the reduction runs in :meth:`OutputSet.evaluate`, not in the
solver. Interior nodes are arithmetic and the same post-ops on
:meth:`OutputSet.ref`. :meth:`OutputSet.plan` contributes only the distinct
leaves, so a calibration run does not materialise series it never saves.

The DAG is static. Evaluating it inside ``jit`` unrolls one step per named
output; the arithmetic itself is traced. ``cumulative(start=)`` still needs a
concrete time axis, as :meth:`~summer4.results.output.Output.cumulative` does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from summer4.results.output import Output
from summer4.results.plan import (
    Compartments,
    ComputedValue,
    FlowMass,
    GroupedOutput,
    Quantity,
    SaveFn,
    SavePlan,
    SaveRequest,
)
from summer4.results.result import Result

_LEAF_TYPES = (Compartments, FlowMass, ComputedValue, SaveFn, GroupedOutput)
_CALLS = frozenset(
    {
        "select",
        "sum_over",
        "total",
        "rolling",
        "cumulative",
        "midpoint",
        "integrate_intervals",
        "integrate",
        "at_times",
    }
)
type _Kind = Literal["leaf", "ref", "const", "param", "unary", "binary", "call"]


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

    def _call(self, method: str, *args: Any, **kwargs: Any) -> OutputExpr:
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

    def sum_over(self, prop: Any, side: Any = None) -> OutputExpr:
        return self._call("sum_over", prop, side)

    def total(self) -> OutputExpr:
        return self._call("total")

    def rolling(
        self,
        window: int,
        *,
        how: Any = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ) -> OutputExpr:
        return self._call("rolling", window, how=how, center=center, min_periods=min_periods)

    def cumulative(self, *, start: Any = None, end: Any = None) -> OutputExpr:
        return self._call("cumulative", start=start, end=end)

    def midpoint(self) -> OutputExpr:
        return self._call("midpoint")

    def integrate_intervals(
        self, method: Literal["trapezoid", "simpson"] = "trapezoid"
    ) -> OutputExpr:
        return self._call("integrate_intervals", method)

    def integrate(self, method: Literal["trapezoid", "simpson"] = "trapezoid") -> OutputExpr:
        return self._call("integrate", method)

    def at_times(self, ts: Any) -> OutputExpr:
        return self._call("at_times", ts)

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


def _coerce(value: object) -> OutputExpr:
    if isinstance(value, OutputExpr):
        return value
    if isinstance(value, _LEAF_TYPES):
        return OutputExpr.leaf(value)
    from summer4.flows.rates import RateOps

    if isinstance(value, RateOps):
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


class OutputSet:
    """Declarative named outputs. Assign leaves and expressions; evaluate after a solve.

    .. code-block:: python

        outputs = OutputSet()
        outputs["population"] = Compartments()
        outputs["incidence"] = FlowMass("infection").midpoint()
        outputs["per_capita"] = outputs.ref("incidence") / outputs.ref("population")
        outputs["scaled"] = outputs.ref("incidence") * Param("scale")

        result = model.run(params, y0, save=outputs.plan(SavePlan(ts=times)))
        named = outputs.evaluate(result, params)

    ``plan`` saves each distinct leaf once. The save key is the first output
    name that uses that spec; a second spec in the same expression is
    ``name__2``. ``sum_over`` on a save spec is that spec's save-time field,
    not a post-op; group afterwards with ``outputs.ref(name).sum_over(...)``.
    A name whose expression is only post-ops on that leaf still
    uses the name as the save key, and the solved ``Result`` holds the *raw*
    leaf there. :meth:`evaluate` is what returns the named values, including
    those post-ops. Do not read a post-processed name off the solve result.

    Two names that share a spec share one save. A key already in ``base`` is
    kept, including its times, when the quantity matches, and rejected when it
    does not. Derived names are not save keys.
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._exprs: dict[str, OutputExpr] = {}

    def __setitem__(self, name: str, value: Quantity | OutputExpr) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Output name must be a non-empty string, got {name!r}.")
        if isinstance(value, OutputExpr):
            expr = value
        elif isinstance(value, _LEAF_TYPES):
            expr = OutputExpr.leaf(value)
        else:
            raise TypeError(
                f"Output {name!r} must be a save spec or an expression built from "
                f"outputs.ref(...). Got {type(value).__name__}."
            )
        if name not in self._exprs:
            self._order.append(name)
        self._exprs[name] = expr

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
        return tuple(self._order)

    def ref(self, name: str) -> OutputExpr:
        """Refer to a named output, including one assigned later."""
        return OutputExpr.ref(name)

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

    def evaluate(self, result: Result, params: object | None = None) -> Result:
        """Compute every named output from ``result`` and ``params``.

        ``result`` must contain the leaf keys from :meth:`plan`. ``params`` is
        the object :func:`~summer4.flows.compiled.eval_closed` reads for a
        ``Param`` or other parameter-only rate expression. The returned
        :class:`~summer4.results.result.Result` is keyed by output name, in
        assignment order, and is safe to build under ``jit``.
        """
        order = self._topo()
        bindings = self._bindings()

        def key_for(qty: Quantity) -> str:
            for existing, key in bindings:
                if _same(existing, qty):
                    return key
            raise KeyError(f"No save key for leaf {qty!r}.")

        computed: dict[str, Output] = {}

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
            raise TypeError(f"Unknown output expression {expr.kind!r}.")

        for name in order:
            value = eval_node(self._exprs[name])
            if not isinstance(value, Output):
                raise TypeError(
                    f"Output {name!r} evaluated to {type(value).__name__}, not an Output."
                )
            computed[name] = value
        ordered = {name: computed[name] for name in self._order}
        return Result(
            times=result.times,
            outputs=ordered,
            solver=result.solver,
            _state_pmap=result._state_pmap,
        )
