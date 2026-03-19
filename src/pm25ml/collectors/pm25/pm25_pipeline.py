"""Pipeline for ingesting CREA measurements data."""

from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from arrow import Arrow
from sklearn.neighbors import BallTree

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig, ValueColumnType
from pm25ml.collectors.grid import Grid
from pm25ml.collectors.pm25.data_source import CreaMeasurementsApiDataSource
from pm25ml.logging import logger


class Pm25MeasurementFilterMarker(ABC):
    """Protocol for marking invalid PM2.5 measurements."""

    @abstractmethod
    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:
        """
        Process the DataFrame and mark invalid measurements.

        This must set the label to "drop" for those which are considered invalid
        by this filter. Otherwise, it must leave the existing labels unchanged.
        """
        ...

    @property
    @abstractmethod
    def window_size(self) -> int:
        """Get the window size for the pipeline."""
        ...


class Pm25MeasurementsPipeline(ExportPipeline):
    """Pipeline for ingesting CREA measurements data."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        in_memory_grid: Grid,
        crea_ds: CreaMeasurementsApiDataSource,
        archive_storage: IngestArchiveStorage,
        filters: list[Pm25MeasurementFilterMarker],
        result_subpath: str,
        start_date: Arrow,
    ) -> None:
        """Initialize the pipeline."""
        self.in_memory_grid = in_memory_grid
        self.crea_ds = crea_ds
        self.archive_storage = archive_storage
        self.filters = filters
        self.result_subpath = result_subpath
        self.start_date = start_date

    def upload(self) -> None:
        """Upload the processed data to the archive storage."""
        to_process_df = self._collect_required_data()

        filtered_df = self._filter_invalid(to_process_df)

        by_grid_date = (
            filtered_df.group_by(["grid_id", "date"])
            .agg(
                value=pl.col("value").mean(),
            )
            .rename({"value": "pm25"})
        )

        with_missing_explicit = self._ensure_missing_explicit(
            by_grid_date,
        ).with_columns(date=pl.col("date").dt.date().cast(pl.String))

        self.archive_storage.write_to_destination(
            with_missing_explicit,
            self.result_subpath,
        )

    def _collect_required_data(self) -> pl.DataFrame:
        """Collect and join the required data together for processing."""
        measurement_df = self.crea_ds.fetch_station_data(
            self._start_date_with_window,
            self._end_date,
        ).select(
            "date",
            "location_id",
            "value",
        )
        station_to_grid_ids_df = self._with_closest_grid_points(
            self.crea_ds.fetch_stations(),
        ).select(
            "id",
            "grid_id",
        )

        station_stats_df = self.crea_ds.fetch_station_stats()

        return measurement_df.join(
            station_to_grid_ids_df,
            left_on="location_id",
            right_on="id",
        ).join(
            station_stats_df,
            on="location_id",
        )

    def get_config_metadata(self) -> PipelineConfig:
        """Get the configuration metadata for the pipeline."""
        return PipelineConfig(
            result_subpath=self.result_subpath,
            id_columns={"date", "grid_id"},
            value_column_type_map={
                "pm25": ValueColumnType.FLOAT,
            },
            expected_rows=self.in_memory_grid.n_rows * self._days_in_month,
        )

    def _filter_invalid(
        self,
        to_process_df: pl.DataFrame,
    ) -> pl.DataFrame:
        processed = to_process_df.with_columns(
            label=pl.lit("keep"),
        )
        for process in self.filters:
            logger.info(f"Applying process: {process.__class__.__name__}")
            processed = process.mark(processed)

        return (
            processed.filter(pl.col("label") == "keep")
            .filter(
                pl.col("date") >= self.start_date.date(),
                pl.col("date") <= self._end_date.date(),
            )
            .drop("label")
        )

    def _ensure_missing_explicit(
        self,
        with_renamed_columns: pl.DataFrame,
    ) -> pl.DataFrame:
        all_dates = (
            pl.date_range(
                start=self.start_date.date(),
                end=self._end_date.date(),
                interval="1d",
                eager=True,
            )
            .cast(pl.Date)
            .alias("date")
            .to_frame()
        )

        all_grid_ids = self.in_memory_grid.df.select("grid_id").unique()

        scaffold = all_grid_ids.join(all_dates, how="cross")

        return scaffold.join(
            with_renamed_columns,
            on=["grid_id", "date"],
            how="left",
        )

    def _with_closest_grid_points(
        self,
        station_data: pl.DataFrame,
    ) -> pl.DataFrame:
        logger.info("Finding closest grid points for configured profile stations")
        df_grid = self.in_memory_grid.df

        grid_latlon_rad = np.deg2rad(
            np.column_stack([df_grid["lat"].to_numpy(), df_grid["lon"].to_numpy()]),
        )
        tree = BallTree(grid_latlon_rad, metric="haversine")

        stn_latlon_rad = np.deg2rad(
            np.column_stack(
                [station_data["latitude"].to_numpy(), station_data["longitude"].to_numpy()],
            ),
        )
        _, nn_idx = tree.query(stn_latlon_rad, k=1)  # shape (n_stn, 1)

        nearest_ids = df_grid["grid_id"].to_numpy()[nn_idx.ravel()]

        return station_data.with_columns(
            pl.Series("grid_id", nearest_ids),
        )

    @property
    def _days_in_month(self) -> int:
        end_date = self._end_date.shift(days=1)
        return (end_date - self.start_date).days

    @property
    def _end_date(self) -> Arrow:
        return self.start_date.shift(months=1).shift(days=-1)

    @property
    def _window(self) -> int:
        return max(filter_.window_size for filter_ in self.filters)

    @property
    def _start_date_with_window(self) -> Arrow:
        return self.start_date.shift(days=-self._window)


class Pm25MeasurementsPipelineConstructor:
    """Pipeline constructor for CREA measurements data."""

    def __init__(
        self,
        *,
        in_memory_grid: Grid,
        crea_ds: CreaMeasurementsApiDataSource,
        archive_storage: IngestArchiveStorage,
        filters: list[Pm25MeasurementFilterMarker],
    ) -> None:
        """Initialize the pipeline constructor."""
        self.crea_ds = crea_ds
        self.in_memory_grid = in_memory_grid
        self.archive_storage = archive_storage
        self.filters = filters

    def construct(
        self,
        result_subpath: str,
        month: Arrow,
    ) -> Pm25MeasurementsPipeline:
        """
        Construct a CreaMeasurementsPipeline with the given parameters.

        :param result_subpath: The subpath in the destination bucket where the results
        will be stored.
        :param month: The month for which the pipeline is being constructed.
        """
        return Pm25MeasurementsPipeline(
            in_memory_grid=self.in_memory_grid,
            crea_ds=self.crea_ds,
            archive_storage=self.archive_storage,
            filters=self.filters,
            result_subpath=result_subpath,
            start_date=month,
        )
