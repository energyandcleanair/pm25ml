"""Integration tests for CreaMeasurementsApiDataSource."""

from __future__ import annotations

import arrow
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pm25ml.collectors.pm25.data_source import CreaMeasurementsApiDataSource
from pm25ml.setup.date_params import TemporalConfig

pytestmark = pytest.mark.integration


@pytest.fixture()
def temporal_config_test_month() -> TemporalConfig:
    """Temporal configuration for a small test period."""
    return TemporalConfig(start_date=arrow.get("2023-01-01"), end_date=arrow.get("2023-01-31"))


def test__fetch_station_data__no_duplicate_rows_per_date(
    temporal_config_test_month,
):
    """It should return at most one row per station per date."""
    ds = CreaMeasurementsApiDataSource(
        source_ids=("cpcb",),
    )

    result = ds.fetch_station_data(arrow.get("2023-01-01"), arrow.get("2023-01-31"))

    # Check that there are no duplicate (location_id, date) combinations
    duplicates = (
        result.group_by("location_id", "date").agg(count=pl.len()).filter(pl.col("count") > 1)
    )

    assert duplicates.height == 0, (
        f"Found {duplicates.height} duplicate (location_id, date) combinations. "
        f"Expected at most one row per station per date. "
        f"Duplicates: {duplicates}"
    )

    # Verify data integrity
    assert "location_id" in result.columns, "location_id column should be present"
    assert "date" in result.columns, "date column should be present"
    assert "value" in result.columns, "value column should be present"

    # Verify date range
    if result.height > 0:
        min_date = result.select(pl.col("date").min()).item()
        max_date = result.select(pl.col("date").max()).item()
        assert min_date >= arrow.get("2023-01-01").date(), (
            f"min_date {min_date} should be >= 2023-01-01"
        )
        assert max_date <= arrow.get("2023-01-31").date(), (
            f"max_date {max_date} should be <= 2023-01-31"
        )
