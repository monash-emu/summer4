"""Deprecation warnings for renamed public spellings."""

from __future__ import annotations

import warnings


def warn_renamed(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Warn that ``old`` is deprecated in favour of ``new`` (removed in a later release)."""
    warnings.warn(
        f"{old} is deprecated; use {new} instead. It will be removed in a later release.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
