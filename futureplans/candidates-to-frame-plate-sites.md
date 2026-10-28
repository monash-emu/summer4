# `Candidates.to_frame` fails on plate sites

**Status:** open. Found in step 47 while writing notebook 29.
**Where:** `src/summer4/epi/calibration/workflow/candidates.py::def to_frame`.

## Problem

`to_frame` builds `pd.DataFrame({site: params[site]})`. For a `NumpyroPosterior`
with a plate site (notebook 28's `z`, shape `(6,)`), `params["z"]` is `(n, 6)` and
pandas raises `ValueError: Per-column arrays must each be 1-dimensional`. Notebook
29 builds its table by hand instead.

## Done looks like

One column per element, named as `LaplaceMetric.element_names` does
(`z[0]` … `z[5]`), and a test on a posterior with a plate site.
