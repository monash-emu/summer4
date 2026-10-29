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


def warn_cumulative(old: str, prefix: str = "", *, stacklevel: int = 3) -> None:
    """Warn that ``cumulative()`` is deprecated, naming both replacements.

    ``prefix`` qualifies the replacements, e.g. ``"Output."``.
    """
    warnings.warn(
        f"{old} is deprecated: it is a plain running sum, which is a cumulative count only "
        f"for amounts per interval. Use {prefix}cumsum() for those (integrate_intervals() "
        f"output, midpoint() series, counts) or {prefix}cumulative_integral() to integrate "
        "a rate such as a FlowMass. It will be removed in a later release.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
