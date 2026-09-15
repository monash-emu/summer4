"""JAX-backed arrays aligned to a :class:`~summer4.propertymap.PropertyMap`."""

from jax.tree_util import register_pytree_node

from summer4.jax.propertydata import PropertyData
from summer4.jax.state import State, unpack_state
from summer4.time import TimeAxis

register_pytree_node(
    TimeAxis,
    lambda x: x.tree_flatten(),
    lambda aux, children: TimeAxis.tree_unflatten(aux, children),
)

__all__ = ["PropertyData", "State", "unpack_state"]
