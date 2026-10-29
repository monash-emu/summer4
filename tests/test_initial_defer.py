"""Initial populations built with ``defer(f)(Param(...))``: hand-computed y0 and gradients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from summer4 import (  # noqa: E402
    Everything,
    InitialPopulation,
    Param,
    Property,
    PropertyMap,
    Split,
    defer,
)


def _pmap() -> tuple[Property, Property, PropertyMap]:
    age = Property("age", ("young", "old"))
    imm = Property("imm", ("yes", "no"))
    return age, imm, PropertyMap.from_property(age).stratify(imm)


def _y(pop: InitialPopulation, pmap: PropertyMap, params: dict[str, Any]) -> Any:
    return jnp.asarray(pop.compile(pmap).evaluate(params).data)


def test_deferred_base_value() -> None:
    _, _, pmap = _pmap()

    def base_fn(n: Any) -> Any:
        return jnp.asarray(n) * 2.0

    pop = InitialPopulation({Everything(): defer(base_fn)(Param("n"))})
    y = _y(pop, pmap, {"n": 50.0})
    np.testing.assert_allclose(y, np.full(4, 25.0))
    grad = jax.grad(lambda n: _y(pop, pmap, {"n": n}).sum())(50.0)
    np.testing.assert_allclose(float(grad), 2.0)


def test_deferred_per_trait_weight() -> None:
    _, imm, pmap = _pmap()

    def share(v: Any) -> Any:
        return jnp.asarray(v) * 0.5

    pop = InitialPopulation(
        {Everything(): 100.0},
        splits=(Split(imm, {"yes": defer(share)(Param("v")), "no": 1.0}, normalize=True),),
    )
    yes = pmap.select(imm["yes"])
    y = _y(pop, pmap, {"v": 0.6})
    # Weights 0.3 : 1.0 normalised; age is free, so each age gets half of the total.
    np.testing.assert_allclose(float(y[yes].sum()), 100.0 * 0.3 / 1.3)
    grad = jax.grad(lambda v: _y(pop, pmap, {"v": v})[yes].sum())(0.6)
    np.testing.assert_allclose(float(grad), 50.0 / 1.3**2)


def test_deferred_whole_table_weights() -> None:
    age, imm, pmap = _pmap()

    def table_fn(vy: Any, vo: Any) -> Any:
        return jnp.array([[vy, 1 - vy], [vo, 1 - vo]])

    pop = InitialPopulation(
        {Everything(): 100.0},
        splits=(Split(imm, defer(table_fn)(Param("vy"), Param("vo")), by=(age,)),),
    )
    young_yes = int(pmap.select(age["young"] & imm["yes"])[0])
    old_yes = int(pmap.select(age["old"] & imm["yes"])[0])
    params = {"vy": 0.8, "vo": 0.3}
    y = _y(pop, pmap, params)
    # Age is free, so each age holds 50; imm splits it by the age's row of the table.
    np.testing.assert_allclose(float(y[young_yes]), 40.0)
    np.testing.assert_allclose(float(y[old_yes]), 15.0)
    grad = jax.grad(lambda vy: _y(pop, pmap, {**params, "vy": vy})[young_yes])(0.8)
    np.testing.assert_allclose(float(grad), 50.0)


def test_callable_base_value_is_rejected_with_defer_hint() -> None:
    _, _, pmap = _pmap()
    pop = InitialPopulation({Everything(): lambda n: n})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="defer\\(") as info:
        pop.compile(pmap)
    assert "callable" in str(info.value)


def test_callable_split_weights_are_rejected_with_defer_hint() -> None:
    _, imm, pmap = _pmap()
    pop = InitialPopulation(
        {Everything(): 100.0},
        splits=(Split(imm, lambda v: v, normalize=True),),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="defer\\(") as info:
        pop.compile(pmap)
    assert "callable" in str(info.value)
