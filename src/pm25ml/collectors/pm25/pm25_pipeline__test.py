"""Tests for `Pm25MeasurementsPipeline`."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from morefs.memory import MemFS
import pytest
from arrow import get as arrow_get, Arrow

from pm25ml.collectors.pm25.pm25_pipeline import (
    Pm25MeasurementsPipelineConstructor,
    Pm25MeasurementFilterMarker,
)
from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.grid import Grid


class _BaseFilter(Pm25MeasurementFilterMarker):  # pragma: no cover - trivial base for tests
    _window_size: int

    def __init__(self, window_size: int) -> None:
        self._window_size = window_size

    @property
    def window_size(self) -> int:  # noqa: D401
        return self._window_size


class DropHighValuesFilter(_BaseFilter):
    """Marks measurements with value > threshold as drop."""

    def __init__(self, threshold: float, window_size: int = 5) -> None:  # noqa: D401
        super().__init__(window_size)
        self.threshold = threshold

    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:  # noqa: D401
        return to_process_df.with_columns(
            pl.when(pl.col("value") > self.threshold)
            .then(pl.lit("drop"))
            .otherwise(pl.col("label"))
            .alias("label"),
        )


class NoopFilter(_BaseFilter):
    """Leaves labels unchanged (used to verify window selection chooses max)."""

    def __init__(self, window_size: int = 2) -> None:  # noqa: D401
        super().__init__(window_size)

    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:  # noqa: D401
        return to_process_df


@dataclass
class _FakeDataSource:
    """A fake data source capturing invocation parameters and returning fixtures."""

    measurement_df: pl.DataFrame
    stations_df: pl.DataFrame
    station_stats_df: pl.DataFrame
    last_fetch_start: Arrow | None = None
    last_fetch_end: Arrow | None = None

    # Interface methods expected by pipeline
    def fetch_station_data(self, start_date: Arrow, end_date: Arrow) -> pl.DataFrame:  # noqa: D401
        self.last_fetch_start = start_date
        self.last_fetch_end = end_date
        return self.measurement_df

    def fetch_stations(self) -> pl.DataFrame:  # noqa: D401
        return self.stations_df

    def fetch_station_stats(self) -> pl.DataFrame:  # noqa: D401
        return self.station_stats_df


def _make_grid() -> Grid:
    """Create a minimal 2-cell grid."""
    df = pl.DataFrame(
        {
            Grid.GRID_ID_COL: [1, 2],
            Grid.GRID_ID_50KM_COL: [101, 102],
            Grid.GEOM_COL: ["POINT (20 10)", "POINT (21 11)"],
            Grid.LAT_COL: [10.0, 11.0],
            Grid.LON_COL: [20.0, 21.0],
            Grid.ORIGINAL_X: [0.0, 1.0],
            Grid.ORIGINAL_Y: [2.0, 3.0],
        },
    )
    return Grid(df)


@pytest.fixture
def start_month() -> Arrow:
    return arrow_get(2023, 1, 1)


@pytest.fixture
def grid() -> Grid:
    return _make_grid()


@pytest.fixture
def stations_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [10, 11],
            "longitude": [20.0, 21.0],
            "latitude": [10.0, 11.0],
        },
    )


@pytest.fixture
def measurement_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [
                arrow_get(2023, 1, 1).date(),
                arrow_get(2023, 1, 1).date(),
                arrow_get(2023, 1, 1).date(),
                arrow_get(2023, 1, 2).date(),
                arrow_get(2023, 1, 3).date(),
            ],
            "location_id": [10, 10, 11, 11, 10],
            "value": [10.0, 20.0, 5.0, 8.0, 250.0],
        },
    )


@pytest.fixture
def station_stats_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "location_id": [10, 11],
            "station_q1": [5.0, 4.0],
            "station_q3": [25.0, 12.0],
            "station_iqr": [20.0, 8.0],
        },
    )


@pytest.fixture
def fake_ds(
    measurement_df: pl.DataFrame,
    stations_df: pl.DataFrame,
    station_stats_df: pl.DataFrame,
) -> _FakeDataSource:  # noqa: D401
    return _FakeDataSource(
        measurement_df=measurement_df,
        stations_df=stations_df,
        station_stats_df=station_stats_df,
    )


@pytest.fixture
def archive_storage() -> IngestArchiveStorage:
    memfs = MemFS()
    return IngestArchiveStorage(memfs, "mem")


@pytest.fixture
def filters() -> list[Pm25MeasurementFilterMarker]:
    return [DropHighValuesFilter(200.0, window_size=5), NoopFilter(window_size=2)]


@pytest.fixture
def constructor(
    grid: Grid,
    fake_ds: _FakeDataSource,
    archive_storage: IngestArchiveStorage,
    filters: list[Pm25MeasurementFilterMarker],
) -> Pm25MeasurementsPipelineConstructor:
    return Pm25MeasurementsPipelineConstructor(
        in_memory_grid=grid,
        crea_ds=fake_ds,  # type: ignore[arg-type]
        archive_storage=archive_storage,
        filters=filters,
    )


@pytest.fixture
def result_subpath() -> str:
    return "country=india/dataset=pm25/month=2023-01"


@pytest.fixture
def pipeline(
    constructor: Pm25MeasurementsPipelineConstructor,
    result_subpath: str,
    start_month: Arrow,
):
    return constructor.construct(result_subpath=result_subpath, month=start_month)


def test__pm25_pipeline__config_metadata__matches_expectations(
    pipeline,  # type: ignore[annotations]
    grid: Grid,
    result_subpath: str,
):
    config = pipeline.get_config_metadata()
    assert config.expected_rows == 31 * grid.n_rows  # Jan
    assert config.id_columns == {"date", "grid_id"}
    assert config.value_columns == {"pm25"}
    assert config.result_subpath == result_subpath


def test__pm25_pipeline__upload_with_filters__writes_full_scaffold_and_applies_filter(
    pipeline,  # type: ignore[annotations]
    archive_storage: IngestArchiveStorage,
    result_subpath: str,
    grid: Grid,
    fake_ds: _FakeDataSource,
):
    pipeline.upload()

    assert archive_storage.does_dataset_exist(result_subpath)
    asset = archive_storage.read_data_asset(result_subpath)
    df = asset.data_frame

    assert df.shape[0] == 31 * grid.n_rows
    assert set(df.columns) == {"grid_id", "date", "pm25"}

    g1d1 = df.filter(pl.col("grid_id") == 1, pl.col("date") == "2023-01-01").select("pm25").item()
    assert g1d1 == 15.0

    g2d2 = df.filter(pl.col("grid_id") == 2, pl.col("date") == "2023-01-02").select("pm25").item()
    assert g2d2 == 8.0

    g1d3 = df.filter(pl.col("grid_id") == 1, pl.col("date") == "2023-01-03").select("pm25").item()
    assert g1d3 is None

    assert df.filter(pl.col("date") == "2023-01-04").shape[0] == 2
    assert (
        df.filter(pl.col("date") == "2023-01-04").select(pl.col("pm25").is_null().all()).item()
        is True
    )

    assert fake_ds.last_fetch_start is not None
    assert fake_ds.last_fetch_end is not None
    assert fake_ds.last_fetch_start.format("YYYY-MM-DD") == "2022-12-27"
    assert fake_ds.last_fetch_end.format("YYYY-MM-DD") == "2023-01-31"


def test__collect_required_data__duplicate_location_id_date__raises_assertion_error(
    constructor: Pm25MeasurementsPipelineConstructor,
    result_subpath: str,
    start_month: Arrow,
    grid: Grid,
    stations_df: pl.DataFrame,
    station_stats_df: pl.DataFrame,
    archive_storage: IngestArchiveStorage,
    filters: list[Pm25MeasurementFilterMarker],
):
    """It should raise AssertionError if there are duplicate (location_id, date) combinations."""
    # Create measurement data with duplicates for the same (location_id, date)
    duplicate_measurement_df = pl.DataFrame(
        {
            "date": [
                arrow_get(2023, 1, 1).date(),
                arrow_get(2023, 1, 1).date(),
                arrow_get(2023, 1, 1).date(),
            ],
            "location_id": [10, 10, 11],  # location_id 10 appears twice on same date
            "value": [10.0, 20.0, 5.0],
        },
    )

    fake_ds_with_duplicates = _FakeDataSource(
        measurement_df=duplicate_measurement_df,
        stations_df=stations_df,
        station_stats_df=station_stats_df,
    )

    pipeline = Pm25MeasurementsPipelineConstructor(
        in_memory_grid=grid,
        crea_ds=fake_ds_with_duplicates,  # type: ignore[arg-type]
        archive_storage=archive_storage,
        filters=filters,
    ).construct(result_subpath=result_subpath, month=start_month)

    with pytest.raises(AssertionError, match="Found 1 duplicate.*location_id.*date"):
        pipeline._collect_required_data()
