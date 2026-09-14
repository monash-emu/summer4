"""Tests for summer4.time."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from summer4.time import Epoch, TimeAxis


def test_epoch_round_trip_integer_days() -> None:
    epoch = Epoch(date(2020, 1, 1), unit=timedelta(days=1))
    dates = np.array(["2020-01-01", "2020-01-02", "2020-01-10"], dtype="datetime64[D]")
    model = epoch.to_model(dates)
    np.testing.assert_allclose(model, [0.0, 1.0, 9.0])
    back = epoch.from_model(model)
    assert back.dtype == np.dtype("datetime64[ns]")
    assert str(back[0].astype("datetime64[D]")) == "2020-01-01"
    assert str(back[2].astype("datetime64[D]")) == "2020-01-10"


def test_epoch_custom_unit() -> None:
    from datetime import datetime

    epoch = Epoch(date(2020, 1, 1), unit=timedelta(hours=1))
    assert float(epoch.to_model(datetime(2020, 1, 1, 0, 0))) == 0.0
    assert float(epoch.to_model(date(2020, 1, 2))) == 24.0


def test_locate_and_window_accept_dates() -> None:
    epoch = Epoch(date(2021, 1, 1))
    values = np.arange(0.0, 10.0, dtype=np.float64)
    axis = TimeAxis(values=values, epoch=epoch, kind="grid")
    assert axis.locate(date(2021, 1, 5)) == 4
    sl = axis.window(date(2021, 1, 2), date(2021, 1, 5))
    assert sl == slice(1, 5)


def test_weights_for_on_grid_is_exact_gather() -> None:
    axis = TimeAxis(values=np.array([0.0, 1.0, 2.0, 3.0]))
    idx, w = axis.weights_for([1.0, 2.0])
    np.testing.assert_array_equal(idx[:, 0], idx[:, 1])
    np.testing.assert_allclose(w[:, 0], 1.0)
    np.testing.assert_allclose(w[:, 1], 0.0)


def test_weights_for_interpolates_off_grid() -> None:
    axis = TimeAxis(values=np.array([0.0, 2.0, 4.0]))
    idx, w = axis.weights_for([1.0])
    np.testing.assert_array_equal(idx[0], [0, 1])
    np.testing.assert_allclose(w[0], [0.5, 0.5])


def test_grouping_month_and_integer_factor() -> None:
    epoch = Epoch(date(2020, 1, 1))
    # daily for Jan+Feb
    values = np.arange(0.0, 60.0, dtype=np.float64)
    axis = TimeAxis(values=values, epoch=epoch)
    monthly = axis.grouping("ME")
    assert monthly.n_groups == 2
    assert monthly.counts[0] == 31
    assert monthly.counts[1] == 29  # 2020 leap year Feb
    factor = axis.grouping(7)
    assert factor.n_groups == (60 + 6) // 7


def test_unsupported_calendar_rule_points_at_to_pandas() -> None:
    axis = TimeAxis(values=np.arange(5.0), epoch=Epoch(date(2020, 1, 1)))
    with pytest.raises(ValueError, match="to_pandas"):
        axis.grouping("BMS")


def test_host_side_on_traced_axis_raises() -> None:
    import jax
    import jax.numpy as jnp

    def fn(v: object) -> int:
        axis = TimeAxis(values=v)
        return axis.locate(0.0)

    with pytest.raises(TypeError, match="traced"):
        jax.jit(fn)(jnp.arange(5.0))


def test_rolling_spec_counts() -> None:
    axis = TimeAxis(values=np.arange(10.0))
    spec = axis.rolling(3, how="mean", min_periods=2)
    assert spec.window == 3
    assert int(spec.valid_counts[0]) == 1
    assert int(spec.valid_counts[2]) == 3
