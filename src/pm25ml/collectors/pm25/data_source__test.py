"""Tests for CreaMeasurementsApiDataSource.

We mock network access by monkeypatching polars read/scan functions.
"""

from __future__ import annotations

import arrow
import polars as pl
import pytest
from unittest.mock import patch
from polars.testing import assert_frame_equal

from pm25ml.collectors.pm25.data_source import CreaMeasurementsApiDataSource
from pm25ml.setup.temporal_config import TemporalConfig


@pytest.fixture()
def temporal_config_two_months() -> TemporalConfig:
    """Temporal configuration spanning two months (Jan & Feb 2023)."""
    return TemporalConfig(
        start_date=arrow.get("2023-01-01"), end_date=arrow.get("2023-02-28")
    )


def test__fetch_station_stats__aggregates_quantiles_and_caches(
    temporal_config_two_months,
):
    """It should compute per-station q1, q3 and IQR only once (cached on 2nd call)."""

    # Data chosen so quartiles fall exactly on existing values (avoids interpolation ambiguity)
    measurements_df = pl.DataFrame(
        {
            "location_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": [10.0, 10.0, 30.0, 30.0, 5.0, 5.0, 15.0, 15.0],
        }
    )

    with patch(
        "pm25ml.collectors.pm25.data_source.pl.scan_csv",
        return_value=measurements_df.lazy(),
    ) as mock_scan_csv:
        ds = CreaMeasurementsApiDataSource(temporal_config=temporal_config_two_months)

        first = ds.fetch_station_stats()
        second = ds.fetch_station_stats()  # Should use cache

        # Caching assertions
        assert (
            mock_scan_csv.call_count == 1
        ), "scan_csv should be called only once due to caching"
        assert first is second, "Cached DataFrame instance should be reused"

        # Ensure URLs were generated for each month in the temporal config
        (paths_arg,) = mock_scan_csv.call_args.args
        assert isinstance(paths_arg, list)
        assert len(paths_arg) == len(temporal_config_two_months.months)

        # Validate aggregation results (order not guaranteed -> sort)
        actual = first.sort("location_id")
        expected = pl.DataFrame(
            {
                "location_id": [1, 2],
                "station_q1": [10.0, 5.0],
                "station_q3": [30.0, 15.0],
                "station_iqr": [20.0, 10.0],
            }
        ).sort("location_id")

        assert_frame_equal(actual, expected)


def test__fetch_stations_for_india__parses_coordinates_and_caches(
    temporal_config_two_months,
):
    """It should parse coordinate strings into longitude/latitude and cache the result."""

    stations_df = pl.DataFrame(
        {
            "id": [1, 2],
            "coordinates": [
                "{'longitude': 77.10, 'latitude': 28.60}",
                "{'longitude': 72.90, 'latitude': 19.00}",
            ],
            # Extra column to ensure it is dropped by select at the end
            "other": ["x", "y"],
        }
    )

    with patch(
        "pm25ml.collectors.pm25.data_source.pl.read_csv",
        return_value=stations_df,
    ) as mock_read_csv:
        ds = CreaMeasurementsApiDataSource(temporal_config=temporal_config_two_months)

        first = ds.fetch_stations_for_india()
        second = ds.fetch_stations_for_india()

        assert (
            mock_read_csv.call_count == 1
        ), "read_csv should be called only once due to caching"
        assert first is second

        # Columns should be limited to id, longitude, latitude
        assert set(first.columns) == {"id", "longitude", "latitude"}

        # Validate parsed coordinate values
        row1 = first.row(0, named=True)
        row2 = first.row(1, named=True)
        assert pytest.approx(row1["longitude"], rel=1e-6) == 77.10
        assert pytest.approx(row1["latitude"], rel=1e-6) == 28.60
        assert pytest.approx(row2["longitude"], rel=1e-6) == 72.90
        assert pytest.approx(row2["latitude"], rel=1e-6) == 19.00


def test__fetch_station_data__casts_types(temporal_config_two_months):
    """It should cast date to pl.Date and value to Float32 for the requested range."""

    measurements_df = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "value": [12.5, 15.0],
            "location_id": [1, 1],  # extra column is passed through unchanged
        }
    )

    with patch(
        "pm25ml.collectors.pm25.data_source.pl.read_csv",
        return_value=measurements_df,
    ):
        ds = CreaMeasurementsApiDataSource(temporal_config=temporal_config_two_months)

        result = ds.fetch_station_data(arrow.get("2023-01-01"), arrow.get("2023-01-02"))

        # Schema checks
        assert result.schema["date"].__class__.__name__ == "Date"
        assert result.schema["value"].__class__.__name__ == "Float32"

        # Value checks
        assert (
            result.select(pl.col("date").min()).item() == arrow.get("2023-01-01").date()
        )
        assert (
            result.select(pl.col("date").max()).item() == arrow.get("2023-01-02").date()
        )
        assert result.height == 2
