"""A small synthetic contact survey, for tests and hand-computed expectations.

Nothing here is real survey data. The matrix is deliberately **asymmetric**
(so a transposition changes every answer) and only partly reciprocal against
:data:`POPULATION` (bands 0 and 1 balance; band 2 does not), so reciprocity
checks have something to find.
"""

from __future__ import annotations

import math

import numpy as np

from summer4.epi import ContactMatrix, SettingStack

BANDS = (0.0, 5.0, 15.0, math.inf)
"""Three bands: ``[0,5)``, ``[5,15)``, ``[15,inf)``."""

VALUES = np.array(
    [
        [2.0, 1.0, 0.5],
        [0.5, 3.0, 1.0],
        [0.25, 1.0, 1.5],
    ]
)
"""Participant-by-contact contacts per person per day."""

POPULATION = np.array([100.0, 200.0, 300.0])
"""The synthetic survey population per band."""

TOTAL = np.array([3.5, 4.5, 2.75])
"""Hand-computed row sums of :data:`VALUES`."""

MEAN_CONTACTS = (100.0 * 3.5 + 200.0 * 4.5 + 300.0 * 2.75) / 600.0
"""Hand-computed population-weighted mean contacts."""

RECIPROCITY = np.array(
    [
        [0.0, 0.0, -25.0],
        [0.0, 0.0, -100.0],
        [25.0, 100.0, 0.0],
    ]
)
"""Hand-computed $c_{ij} N_i - c_{ji} N_j$ for :data:`VALUES` and :data:`POPULATION`."""

HOME = np.array(
    [
        [1.0, 0.5, 0.25],
        [0.25, 1.0, 0.5],
        [0.25, 0.5, 1.0],
    ]
)
SCHOOL = np.array(
    [
        [0.5, 0.25, 0.0],
        [0.25, 1.5, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
WORK = VALUES - HOME - SCHOOL
"""Three synthetic settings that sum to :data:`VALUES`."""


def survey() -> ContactMatrix:
    """The synthetic all-settings matrix with its survey population."""
    return ContactMatrix.from_array(VALUES, BANDS, source_population=POPULATION)


def stack() -> SettingStack:
    """The synthetic per-setting stack; ``stack().total() == survey()`` values."""
    return ContactMatrix.from_settings(
        {
            name: ContactMatrix.from_array(values, BANDS, source_population=POPULATION)
            for name, values in (("home", HOME), ("school", SCHOOL), ("work", WORK))
        }
    )
