"""Shared StrEnum coercion for public mode arguments."""

from __future__ import annotations

from enum import StrEnum


def coerce_strenum[E: StrEnum](cls: type[E], value: E | str, *, what: str) -> E:
    """Accept an enum member or its string value; prefer the enum at call sites."""
    if isinstance(value, cls):
        return value
    if isinstance(value, str):
        try:
            return cls(value)
        except ValueError as exc:
            known = ", ".join(repr(member.value) for member in cls)
            raise ValueError(f"Unknown {what} {value!r}. Known: {known}.") from exc
    raise TypeError(f"{what} must be {cls.__name__} or str, got {type(value).__name__}.")
