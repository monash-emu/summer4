# Releases

summer4 is distributed as **git tags**, not yet on PyPI. Each tagged alpha is
installable; the API is not stable. Prefer the latest tag unless a downstream
pin requires an older one.

The repository root [`CHANGELOG.md`](https://github.com/monash-emu/summer4/blob/main/CHANGELOG.md)
is the machine-friendly history. The pages below are the narrative notes for
each release.

```{toctree}
:maxdepth: 1

v0.2.0a4
rate-array-dispatch
v0.2.0a3
v0.2.0a2
v0.2.0a1
```

| Tag | Date | One-line summary |
| --- | --- | --- |
| [`v0.2.0a4`](v0.2.0a4.md) | 2026-09-22 | TB-port surface: ageing, generalised FOI, Output / OutputSet |
| [`rate-array-dispatch`](rate-array-dispatch.md) | 2026-09-21 | NumPy ufuncs build rate nodes; `ndarray * rate` is one node (in a4) |
| [`v0.2.0a3`](v0.2.0a3.md) | 2026-09-21 | `prepare()` boxes float params for Diffrax JIT cache |
| [`v0.2.0a2`](v0.2.0a2.md) | 2026-09-20 | Diffrax `run()` JIT cache; rate math; table interp |
| [`v0.2.0a1`](v0.2.0a1.md) | 2026-09-18 | First pinnable flows stack on `main` |
