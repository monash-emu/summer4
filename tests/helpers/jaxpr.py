"""Jaxpr walkers for asserting work sits outside solver loops."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any


def _sub_jaxprs(eqn: Any) -> list[Any]:
    out: list[Any] = []
    for v in eqn.params.values():
        for item in v if isinstance(v, (tuple, list)) else (v,):
            if hasattr(item, "jaxpr") and hasattr(item.jaxpr, "eqns"):
                out.append(item.jaxpr)
            elif hasattr(item, "eqns"):
                out.append(item)
    return out


def _all_eqns(jaxpr: Any) -> Iterator[Any]:
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in _sub_jaxprs(eqn):
            yield from _all_eqns(sub)


def loop_body_primitives(closed: Any) -> Counter[str]:
    """Primitive names inside any scan/while body, at any depth."""
    counts: Counter[str] = Counter()
    jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in {"scan", "while"}:
            for sub in _sub_jaxprs(eqn):
                for inner in _all_eqns(sub):
                    counts[inner.primitive.name] += 1
        else:
            for sub in _sub_jaxprs(eqn):
                counts.update(loop_body_primitives(sub))
    return counts


def outside_loop_primitives(closed: Any) -> Counter[str]:
    """Primitive names not inside any scan/while body."""
    counts: Counter[str] = Counter()
    jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed

    def walk(jp: Any, *, inside_loop: bool) -> None:
        for eqn in jp.eqns:
            is_loop = eqn.primitive.name in {"scan", "while"}
            if not inside_loop and not is_loop:
                counts[eqn.primitive.name] += 1
            for sub in _sub_jaxprs(eqn):
                walk(sub, inside_loop=inside_loop or is_loop)

    walk(jaxpr, inside_loop=False)
    return counts
