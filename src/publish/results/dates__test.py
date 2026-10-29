"""Tests for the dataset end-date label."""

import numpy as np
import pytest
import xarray as xr

from publish.results.dates import latest_dataset_end_label


def test__latest_dataset_end_label__daily_times_in_may__returns_may_2026() -> None:
    times = np.arange("2026-05-01", "2026-06-01", dtype="datetime64[D]")
    dataset = xr.Dataset({"pm25": ("time", np.zeros(len(times)))}, coords={"time": times})

    assert latest_dataset_end_label(dataset) == "May 2026"


def test__latest_dataset_end_label__missing_time__raises() -> None:
    dataset = xr.Dataset({"pm25": ("x", [1.0])})

    with pytest.raises(ValueError, match="no time coordinate"):
        latest_dataset_end_label(dataset)
