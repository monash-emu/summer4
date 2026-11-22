"""How far fusion moves results: fused vs term-by-term on the 36-cell model.

Prints the largest relative difference in the final state and in the gradient, for Euler
(2000 steps) and the default adaptive Tsit5, plus the vector field at the initial state. Run it
under each XLA:CPU runtime to compare those too (``SUMMER4_CPU_RUNTIME=thunk``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import weighted_reduce as wr  # noqa: E402


def rel(a: object, b: object) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(x), np.abs(y))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(scale > 0, np.abs(x - y) / scale, 0.0)
    return float(np.max(r))


def main() -> None:
    import jax
    import jax.numpy as jnp

    import summer4

    info = summer4.runtime_info()
    print(f"jax {jax.__version__}; cpu runtime {info.cpu_runtime} ({info.cpu_runtime_source})")
    for solver in ("euler", "tsit5"):
        fused = wr.make_fns("terms", solver)
        plain = wr.make_fns("terms", solver, compile_kw={"fuse_reductions": False})
        theta = fused["theta"]
        line = [f"{solver:6s}"]
        if solver == "euler":
            p_f = fused["cm"].prepare(dict(wr.PARAMS))
            p_p = plain["cm"].prepare(dict(wr.PARAMS))
            vf_f = fused["cm"].vector_field(0.0, fused["y0"], p_f)
            vf_p = plain["cm"].vector_field(0.0, plain["y0"], p_p)
            line.append(f"vf_rel={rel(vf_f, vf_p):.2e}")
        line.append(f"prevalence_rel={rel(fused['solve'](theta), plain['solve'](theta)):.2e}")
        line.append(f"grad_rel={rel(fused['grad'](theta), plain['grad'](theta)):.2e}")
        print(" ".join(line))
    _ = jnp


if __name__ == "__main__":
    main()
