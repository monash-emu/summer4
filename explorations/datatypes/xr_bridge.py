"""PropertyData <-> xarray encodings and a vendored JAX pytree registration.

xarray 2026 can hold JAX arrays natively (Array API). The only extra work is
telling JAX how to flatten ``Variable`` / ``DataArray`` / ``Dataset``. Coords
are treated as static aux (converted to tuples so the treedef is hashable).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax.tree_util import register_pytree_node

from explorations.datatypes.prototype import VARIANT_DIGEST, PropertyData
from summer4 import PropertyMap, Selector

_REGISTERED = False
_NA = -1
ABSENT_LABEL = ""


def register_xarray_pytrees() -> None:
    """Register xarray Variable, DataArray, and Dataset as JAX pytrees.

    Safe to call more than once. Coords and attrs are static (hashable tuples);
    only the data buffer is a leaf.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    def _coord_aux(
        coords: xr.Coordinates,
    ) -> tuple[tuple[str, tuple[str, ...], tuple[Any, ...]], ...]:
        items: list[tuple[str, tuple[str, ...], tuple[Any, ...]]] = []
        for name, coord in coords.items():
            values = np.asarray(coord.values)
            items.append((str(name), tuple(coord.dims), tuple(values.tolist())))
        return tuple(items)

    def _coords_from_aux(
        aux: tuple[tuple[str, tuple[str, ...], tuple[Any, ...]], ...],
    ) -> dict[str, tuple[tuple[str, ...], list[Any]]]:
        return {name: (dims, list(values)) for name, dims, values in aux}

    def _flatten_variable(var: xr.Variable) -> tuple[tuple[Any], tuple[Any, ...]]:
        return (var.data,), (var.dims, _frozen_attrs(var.attrs))

    def _unflatten_variable(aux: tuple[Any, ...], children: tuple[Any, ...]) -> xr.Variable:
        dims, attrs = aux
        (data,) = children
        return xr.Variable(dims, data, attrs=dict(attrs))

    def _flatten_dataarray(da: xr.DataArray) -> tuple[tuple[Any], tuple[Any, ...]]:
        return (da.data,), (da.dims, da.name, _coord_aux(da.coords), _frozen_attrs(da.attrs))

    def _unflatten_dataarray(aux: tuple[Any, ...], children: tuple[Any, ...]) -> xr.DataArray:
        dims, name, coord_aux, attrs = aux
        (data,) = children
        restored = {
            key: (cdims, values) for key, (cdims, values) in _coords_from_aux(coord_aux).items()
        }
        return xr.DataArray(data, dims=dims, name=name, coords=restored, attrs=dict(attrs))

    def _flatten_dataset(ds: xr.Dataset) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        names = tuple(ds.data_vars)
        leaves = tuple(ds[name].data for name in names)
        var_meta = tuple((name, ds[name].dims, _frozen_attrs(ds[name].attrs)) for name in names)
        return leaves, (names, var_meta, _coord_aux(ds.coords), _frozen_attrs(ds.attrs))

    def _unflatten_dataset(aux: tuple[Any, ...], children: tuple[Any, ...]) -> xr.Dataset:
        _names, var_meta, coord_aux, attrs = aux
        coords = {
            key: (cdims, values) for key, (cdims, values) in _coords_from_aux(coord_aux).items()
        }
        data_vars = {
            name: (dims, child, dict(var_attrs))
            for (name, dims, var_attrs), child in zip(var_meta, children, strict=True)
        }
        return xr.Dataset(data_vars, coords=coords, attrs=dict(attrs))

    register_pytree_node(xr.Variable, _flatten_variable, _unflatten_variable)
    register_pytree_node(xr.DataArray, _flatten_dataarray, _unflatten_dataarray)
    register_pytree_node(xr.Dataset, _flatten_dataset, _unflatten_dataset)
    _REGISTERED = True


def _frozen_attrs(attrs: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(k), _freeze_attr(v)) for k, v in dict(attrs).items()))


def _freeze_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, tuple(value.tolist()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_attr(v) for v in value)
    return value


def _trait_labels(pmap: PropertyMap, col: int) -> list[str]:
    prop = pmap.properties[col]
    labels: list[str] = []
    for code in pmap.codes[:, col]:
        value = int(code)
        labels.append(ABSENT_LABEL if value == _NA else prop.traits[value])
    return labels


def to_dataarray(pd: PropertyData, *, name: str = "value") -> xr.DataArray:
    """Flat encoding: last axis is ``compartment``, properties are non-dim coords."""
    pmap = pd.pmap
    leading = tuple(f"lead{i}" for i in range(pd.data.ndim - 1))
    dims = (*leading, "compartment")
    coords: dict[str, Any] = {"compartment": np.arange(pmap.size)}
    for i, prop in enumerate(pmap.properties):
        coords[prop.name] = ("compartment", _trait_labels(pmap, i))
    return xr.DataArray(pd.data, dims=dims, coords=coords, name=name)


def from_dataarray(
    da: xr.DataArray,
    pmap: PropertyMap,
    *,
    variant: str = VARIANT_DIGEST,
) -> PropertyData:
    """Recover PropertyData from a flat DataArray. ``pmap`` is required (static)."""
    if "compartment" not in da.dims:
        raise ValueError("DataArray must have a 'compartment' dimension.")
    if da.sizes["compartment"] != pmap.size:
        raise ValueError(
            f"compartment size {da.sizes['compartment']} does not match pmap.size {pmap.size}."
        )
    return PropertyData(pmap, jnp.asarray(da.data), variant=variant)


def selector_mask(da: xr.DataArray, pmap: PropertyMap, sel: Selector) -> xr.DataArray:
    """Boolean mask along ``compartment`` from a summer4 selector (Kleene-true)."""
    mask = pmap.mask(sel)
    return xr.DataArray(
        mask,
        dims=("compartment",),
        coords={"compartment": da.coords["compartment"]},
    )


def presence_signature(row: np.ndarray, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name, code in zip(names, row, strict=True) if int(code) != _NA)


def to_blocks(pd: PropertyData) -> dict[tuple[str, ...], xr.DataArray]:
    """Presence-signature encoding: one dense N-D DataArray per present-property set."""
    pmap = pd.pmap
    names = tuple(prop.name for prop in pmap.properties)
    props = {prop.name: prop for prop in pmap.properties}
    codes = np.asarray(pmap.codes)
    data = np.asarray(pd.data)
    leading = data.shape[:-1]
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, row in enumerate(codes):
        groups.setdefault(presence_signature(row, names), []).append(i)

    blocks: dict[tuple[str, ...], xr.DataArray] = {}
    for sig, rows in groups.items():
        row_idx = np.asarray(rows)
        if not sig:
            dims = tuple(f"lead{i}" for i in range(len(leading))) + ("row",)
            blocks[sig] = xr.DataArray(data[..., row_idx], dims=dims)
            continue
        shape = leading + tuple(len(props[name].traits) for name in sig)
        cube = np.full(shape, np.nan, dtype=np.result_type(data, np.float32))
        sub = codes[row_idx]
        name_index = {name: i for i, name in enumerate(names)}
        idx = tuple(sub[:, name_index[name]].astype(np.intp) for name in sig)
        cube[tuple(slice(None) for _ in leading) + idx] = data[..., row_idx]
        coords = {name: list(props[name].traits) for name in sig}
        dims = tuple(f"lead{i}" for i in range(len(leading))) + sig
        blocks[sig] = xr.DataArray(cube, dims=dims, coords=coords)
    return blocks


def blocks_to_datatree(blocks: dict[tuple[str, ...], xr.DataArray]) -> Any:
    """Wrap presence-signature blocks in an xarray DataTree when available."""
    # Slash paths nest; overlapping signatures (state/age vs state/age/severity)
    # collide as parent variables. Flat names keep one node per block.
    mapping = {"__".join(sig) if sig else "_empty": da for sig, da in blocks.items()}
    if hasattr(xr, "DataTree"):
        try:
            return xr.DataTree.from_dict(mapping)
        except (TypeError, ValueError, AttributeError, KeyError):
            return mapping
    return mapping


def from_blocks(
    blocks: dict[tuple[str, ...], xr.DataArray],
    pmap: PropertyMap,
    *,
    variant: str = VARIANT_DIGEST,
) -> PropertyData:
    """Scatter dense blocks back onto the flat compartment axis of ``pmap``."""
    names = tuple(prop.name for prop in pmap.properties)
    name_index = {name: i for i, name in enumerate(names)}
    codes = np.asarray(pmap.codes)
    sample_sig, sample_da = next(iter(blocks.items()))
    n_prop_dims = len(sample_sig) if sample_sig else 1
    leading_shape = sample_da.shape[:-n_prop_dims]
    out = np.zeros(leading_shape + (pmap.size,), dtype=np.asarray(sample_da.values).dtype)
    for i, row in enumerate(codes):
        sig = presence_signature(row, names)
        da = blocks[sig]
        if not sig:
            continue
        loc = tuple(int(row[name_index[name]]) for name in sig)
        out[..., i] = np.asarray(da.data)[(..., *loc)]
    return PropertyData(pmap, jnp.asarray(out), variant=variant)
