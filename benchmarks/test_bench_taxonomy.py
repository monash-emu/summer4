"""Build and query benchmarks for the compartment taxonomy."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from summer4 import Property, PropertyMap


def _cartesian_map(factors: tuple[int, ...]) -> PropertyMap:
    props = [Property(f"p{i}", tuple(f"t{j}" for j in range(n))) for i, n in enumerate(factors)]
    pm = PropertyMap.from_property(props[0])
    for prop in props[1:]:
        pm = pm.stratify(prop)
    return pm


def _sir_like(n_age: int, n_loc: int) -> PropertyMap:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", tuple(f"a{i}" for i in range(n_age)))
    loc = Property("loc", tuple(f"l{i}" for i in range(n_loc)))
    return PropertyMap.from_property(state).stratify(age).stratify(loc)


@pytest.mark.parametrize("n", [10, 1_000, 100_000], ids=["10", "1k", "100k"])
def test_bench_build(benchmark: Callable[..., PropertyMap], n: int) -> None:
    if n == 10:
        factors = (10,)
    elif n == 1_000:
        factors = (10, 10, 10)
    else:
        factors = (10, 10, 10, 10, 10)
    result = benchmark(_cartesian_map, factors)
    assert result.size == n


@pytest.mark.parametrize("n", [10, 1_000, 100_000], ids=["10", "1k", "100k"])
def test_bench_select_cold(benchmark: Callable[..., object], n: int) -> None:
    if n == 10:
        pm = _cartesian_map((10,))
        sel = pm.properties[0][pm.properties[0].traits[0]]
    elif n == 1_000:
        pm = _sir_like(10, 34)
        # 3 * 10 * 34 = 1020; accept nearby size
        sel = (
            pm.properties[0][pm.properties[0].traits[1]]
            & pm.properties[1][pm.properties[1].traits[0]]
        )
    else:
        pm = _cartesian_map((10, 10, 10, 10, 10))
        sel = (
            pm.properties[0][pm.properties[0].traits[0]]
            & pm.properties[-1][pm.properties[-1].traits[0]]
        )

    def _cold() -> int:
        return int(pm.copy().select(sel).size)

    assert benchmark(_cold) >= 0


@pytest.mark.parametrize("n", [10, 1_000, 100_000], ids=["10", "1k", "100k"])
def test_bench_select_cached(benchmark: Callable[..., object], n: int) -> None:
    if n == 10:
        pm = _cartesian_map((10,))
        sel = pm.properties[0][pm.properties[0].traits[0]]
    elif n == 1_000:
        pm = _sir_like(10, 34)
        sel = pm.properties[0][pm.properties[0].traits[1]]
    else:
        pm = _cartesian_map((10, 10, 10, 10, 10))
        sel = pm.properties[2][pm.properties[2].traits[3]]
    pm.select(sel)

    def _cached() -> int:
        return int(pm.select(sel).size)

    assert benchmark(_cached) >= 0
