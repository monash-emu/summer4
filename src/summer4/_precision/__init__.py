"""Default floating-point width: float64, unless the user chose otherwise.

JAX's ``jax_enable_x64`` flag is process-wide. summer4 turns it on when imported, as summer2
did, unless ``JAX_ENABLE_X64`` is set in the environment: then JAX has already applied the
user's choice. A user who wants float32 without the environment variable calls
``jax.config.update("jax_enable_x64", False)`` after importing summer4 and before compiling.
"""

import os
from typing import Any, cast

import jax

if "JAX_ENABLE_X64" not in os.environ:
    cast(Any, jax.config).update("jax_enable_x64", True)
