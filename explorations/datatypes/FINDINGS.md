# PropertyData spike findings

Measured on macOS arm64, JAX 0.6.2, xarray 2026.7.0, 2026-09-09.
Times are median of a host loop with `block_until_ready`.

## Recommendations

| Question | Answer |
| --- | --- |
| Canonical representation | **Our own `PropertyData`** (JAX array + static `PropertyMap`). |
| Hashing strategy | **Digest (variant A)**. Identity hashing retraces on equal rebuilt maps. |
| xarray | **Optional export only.** Do not take it as a core dependency. Vendor pytree registration if a jit-able DataArray boundary is needed. |
| Lazy ops | **No.** XLA already fuses elementwise chains. Memoising static one-hot / segment ids is enough. |

## 1. Pytree dispatch and hashing

`PropertyMap` is `@dataclass(frozen=True, eq=False)` with a handwritten `__eq__`, so `__hash__` is `None`. It cannot be pytree aux until that is fixed.

Two wrappers were compared:

- **A / digest** — blake2 of `codes` (+ `parent_row`) plus `properties` / `history`. Structurally equal maps share a treedef.
- **B / identity** — `id(pmap)`. Dispatch is O(1); a new equal instance retraces.

| Workload | raw `*2` | digest `*2` | identity `*2` | digest equal clone | identity clone first |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k | 4.6 µs | 5.3 µs | 5.0 µs | 5.4 µs | **7.4 ms** |
| 100k | 16 µs | 13 µs | 19 µs | 14 µs | **10.0 ms** |

Digest `__eq__` is lost in jit-dispatch noise. Identity's first call on a rebuilt-but-equal map costs ~1000× a dispatch. Use digest.

`tree_map` / `unflatten` pass tracers and sentinels: `__init__` must not check `data.shape`. Validate at the Python edge (`PropertyData.check` / `wrap`).

## 2. Minimum algebra

The following is enough for the modelling calls we care about. Indices come from `PropertyMap.select` / `codes` at trace time and become compile-time constants.

- elementwise `+ - * /` with a scalar or a same-map `PropertyData`
- `d[sel]` gather, `d.subset(sel)` (filtered map), `d.partition(prop)`
- `d.at[sel].set/add/mul`, `d.where(sel, other)`
- `d.sum_over(prop)` → `PropertyMap.from_property(prop)`
- `d.broadcast_over(prop, values)` (absent → 0)
- `d.normalise_within(prop)` = broadcast of `sum_over`

Scoped update matches raw `arr.at[idx].add` (1k: 3.8 µs vs 3.3 µs; 100k: 22 µs vs 22 µs).

`sum_over` via a static one-hot matmul beat `jax.ops.segment_sum` (100k: 71 µs vs 321 µs). Prefer the matmul when `n_traits` is small. float32 reduction order differs (~4e-5 relative); that is expected.

## 3. xarray

### Three routes

1. **PyPI `xarray-jax` 0.0.5** — pins `jax<0.5.0` and `xarray<2025.0.0`. Confirmed on PyPI. Unusable here.
2. **`gdm-xarray-jax`** — GitHub only (`google-deepmind/xarray_jax`), 404 on PyPI, wants `xarray>=2026.1.0`. Same idea as a 40-line flatten. Not worth a git dependency.
3. **Vendored registration** — jax 0.6.2 arrays expose `__array_namespace__`; xarray holds them natively. Register `Variable` / `DataArray` / `Dataset` with coords as hashable aux. This worked: `jit(lambda da: da * 2)` returned a labelled `DataArray`.

### Encodings

**Flat** (`compartment` dim + non-dim property coords) is a lossless labelled view. It does **not** give xarray algebra:

- `.sel()` is not how non-dim coords are queried.
- `severity != "mild"` **includes** absent rows. Kleene `~sev["mild"]` **excludes** unknown. This is the load-bearing mismatch.
- `sev.absent()` can be faked with an empty-string sentinel, but that is not first-class NA.

Kleene queries must stay on `PropertyMap.mask` / `select`.

**Presence-signature blocks** split a ragged map into dense cubes (here `(state, age)` for S/R and `(state, age, severity)` for I). That is the encoding where named dims and `.sum("age")` mean something. Costs on the 12-compartment example: eager `to_blocks` 308 µs vs `to_dataarray` 80 µs. DataTree slash-paths collide when signatures share a prefix (`state/age` vs `state/age/severity`); use flat node names.

### jit cost

| Call | median |
| --- | ---: |
| jit `PropertyData * 2` | 3.5 µs |
| jit `DataArray * 2` (vendored) | 98 µs |
| jit flat round-trip `PD → DA → PD` | 3.1 µs |

Round-trip under jit is free (the buffer is the only leaf). A DataArray as the *working* type is ~30× a PropertyData dispatch on this machine, from flatten/unflatten of coords. Use xarray at I/O and inspection boundaries, not inside the stepper.

## 4. Lazy operations

100k chained `((x*1.1)+0.2)*0.9-0.05` vs hand-fused `x*0.99+0.13`: ratio **1.0–1.6** across runs. XLA already fuses elementwise chains. A Python lazy graph would add API surface for no runtime win.

Worth keeping (static, not lazy):

- memoise selector index arrays (already in `PropertyMap._cache`)
- memoise per-property one-hot / segment-id vectors on the map
- do not introduce a delayed-op graph

## Follow-up `src/summer4` work

No public API should land from this spike as-is. The next feature branch should:

1. **Make `PropertyMap` hashable** with a cached digest of `codes` / `parent_row` plus `properties` and `history`. Keep the existing `__eq__` or make it digest-first so it stays consistent with `__hash__`.
2. Add `PropertyData` (name TBD) as a JAX pytree: last axis = compartments, `pmap` static aux, no shape check in `__init__`.
3. Implement the algebra above; use one-hot matmul for `sum_over` when `n_traits` is small.
4. Keep the taxonomy layer NumPy-only. `PropertyData` lives in a JAX-using module (or extra).
5. Do **not** add xarray (or either xarray-jax) as a core dependency. If results need labelled arrays, add an optional `summer4.xarray` extra with flat + presence-block exporters and a vendored pytree registration.
6. Do **not** build a lazy op graph.

Prototype code: [`prototype.py`](prototype.py), [`xr_bridge.py`](xr_bridge.py), notebooks `01-propertydata.ipynb` and `02-xarray-comparison.ipynb`.
