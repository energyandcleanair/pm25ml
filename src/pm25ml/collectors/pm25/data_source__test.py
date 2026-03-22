"""Tests for CreaMeasurementsApiDataSource."""

from __future__ import annotations

import arrow
import polars as pl
import pytest
from unittest.mock import MagicMock, patch
from polars.testing import assert_frame_equal

from pm25ml.collectors.pm25.data_source import (
    CSV_REQUEST_HEADERS,
    CSV_REQUEST_TIMEOUT_SECONDS,
    CreaMeasurementsApiDataSource,
)
from pm25ml.setup.date_params import TemporalConfig


@pytest.fixture()
def temporal_config_two_months() -> TemporalConfig:
    """Temporal configuration spanning two months (Jan & Feb 2023)."""
    return TemporalConfig(start_date=arrow.get("2023-01-01"), end_date=arrow.get("2023-02-28"))


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
        "pm25ml.collectors.pm25.data_source.CreaMeasurementsApiDataSource._read_csvs_from_urls",
        return_value=measurements_df,
    ) as mock_read_csvs:
        ds = CreaMeasurementsApiDataSource(
            temporal_config=temporal_config_two_months,
            source_ids=("cpcb",),
        )

        first = ds.fetch_station_stats()
        second = ds.fetch_station_stats()  # Should use cache

        # Caching assertions
        assert mock_read_csvs.call_count == 1, (
            "monthly CSVs should be fetched only once due to caching"
        )
        assert first is second, "Cached DataFrame instance should be reused"

        # Ensure URLs were generated for each month in the temporal config
        (paths_arg,) = mock_read_csvs.call_args.args
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


def test__fetch_stations__parses_coordinates_and_caches(temporal_config_two_months):
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
        "pm25ml.collectors.pm25.data_source.CreaMeasurementsApiDataSource._read_csv_from_url",
        return_value=stations_df,
    ) as mock_read_csv:
        ds = CreaMeasurementsApiDataSource(
            temporal_config=temporal_config_two_months,
            source_ids=("cpcb",),
        )

        first = ds.fetch_stations()
        second = ds.fetch_stations()

        assert mock_read_csv.call_count == 1, "read_csv should be called only once due to caching"
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
        "pm25ml.collectors.pm25.data_source.CreaMeasurementsApiDataSource._read_csv_from_url",
        return_value=measurements_df,
    ):
        ds = CreaMeasurementsApiDataSource(
            temporal_config=temporal_config_two_months,
            source_ids=("cpcb",),
        )

        result = ds.fetch_station_data(arrow.get("2023-01-01"), arrow.get("2023-01-02"))

        # Schema checks
        assert result.schema["date"].__class__.__name__ == "Date"
        assert result.schema["value"].__class__.__name__ == "Float32"

        # Value checks
        assert result.select(pl.col("date").min()).item() == arrow.get("2023-01-01").date()
        assert result.select(pl.col("date").max()).item() == arrow.get("2023-01-02").date()
        assert result.height == 2


def test__read_csv_from_url__uses_requests_with_browser_like_headers(
    temporal_config_two_months,
):
    """It should fetch CSV content via requests with headers accepted by the CREA API."""

    csv_bytes = b"date,value,location_id\n2023-01-01,12.5,1\n"

    with patch("pm25ml.collectors.pm25.data_source.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = csv_bytes
        mock_get.return_value = mock_response

        ds = CreaMeasurementsApiDataSource(
            temporal_config=temporal_config_two_months,
            source_ids=("cpcb",),
        )

        result = ds._read_csv_from_url("https://api.energyandcleanair.org/test.csv")

        mock_get.assert_called_once_with(
            "https://api.energyandcleanair.org/test.csv",
            headers=CSV_REQUEST_HEADERS,
            timeout=CSV_REQUEST_TIMEOUT_SECONDS,
        )
        mock_response.raise_for_status.assert_called_once_with()

        assert result.shape == (1, 3)
        assert result["location_id"].item() == 1
