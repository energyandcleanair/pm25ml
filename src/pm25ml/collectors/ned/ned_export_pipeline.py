"""NASA EarthData pipeline for exporting data."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import polars as pl
import xarray
from arrow import Arrow
from polars import DataFrame

from pm25ml.collectors.coverage import CoverageProbeResult, dates_from_arrows
from pm25ml.collectors.export_pipeline import (
    ExportPipeline,
    PipelineConfig,
    PipelineConsumerBehaviour,
    ValueColumnType,
)
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever
from pm25ml.collectors.ned.errors import NedMissingDataError
from pm25ml.logging import logger

if TYPE_CHECKING:
    from datetime import date

    from pm25ml.collectors.archive_storage import IngestArchiveStorage
    from pm25ml.collectors.grid import Grid
    from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
    from pm25ml.collectors.ned.data_retrievers import NedDataRetriever
    from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor


class NedPipelineConstructor:
    """
    Constructs a NED export pipeline.

    This class is used to create an instance of the NED export pipeline with the necessary
    parameters.
    """

    def __init__(
        self,
        *,
        archive_storage: IngestArchiveStorage,
        grid: Grid,
    ) -> None:
        """
        Initialize the pipeline constructor with the archive storage and grid.

        Args:
            archive_storage (IngestArchiveStorage): The storage where the results will be archived.
            grid (GeoDataFrame): The grid to which the data will be regridded.

        """
        self.archive_storage = archive_storage
        self.grid = grid

    def construct(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
        dataset_retriever: NedDataRetriever | None = None,
        dataset_reader: NedDataReader,
        result_subpath: str,
        consumer_behaviour: PipelineConsumerBehaviour | None = None,
    ) -> NedExportPipeline:
        """
        Construct the NED export pipeline with the given parameters.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The descriptor for the dataset
            to be exported.
            dataset_retriever (NedDataRetriever, optional): The retriever for the dataset.
            Defaults to the RawEarthAccessDataRetriever.
            dataset_reader (NedDataReader): The reader for the dataset.
            result_subpath (str): The subpath where the results will be stored.
            consumer_behaviour (PipelineConsumerBehaviour, optional): The behaviour of the
            consumers of the export pipeline. Defaults to None, which uses the default
            consumer behaviour.

        Returns:
            NedExportPipeline: An instance of the NED export pipeline.

        """
        return NedExportPipeline(
            grid=self.grid,
            archive_storage=self.archive_storage,
            dataset_descriptor=dataset_descriptor,
            dataset_reader=dataset_reader,
            dataset_retriever=(
                dataset_retriever
                if dataset_retriever is not None
                else RawEarthAccessDataRetriever()
            ),
            result_subpath=result_subpath,
            consumer_behaviour=consumer_behaviour,
        )


class NedExportPipeline(ExportPipeline):
    """
    ExportPipeline for the NED data.

    This only works where the grid that we want to regrid into has a higher resolution
    than the data fetched.

    This pipeline is responsible for orchestrating the exporting of the NED data from the origin,
    transforming it into the grid format, and uploading it to the underlying storage.
    """

    @staticmethod
    def with_args(
        *,
        grid: Grid,
        archive_storage: IngestArchiveStorage,
    ) -> NedPipelineConstructor:
        """
        Create a NedPipelineConstructor with the given grid and archive storage.

        This allows for a more fluent interface when constructing pipelines.
        """
        return NedPipelineConstructor(
            archive_storage=archive_storage,
            grid=grid,
        )

    def __init__(  # noqa: PLR0913
        self,
        *,
        grid: Grid,
        archive_storage: IngestArchiveStorage,
        dataset_descriptor: NedDatasetDescriptor,
        dataset_retriever: NedDataRetriever,
        dataset_reader: NedDataReader,
        result_subpath: str,
        consumer_behaviour: PipelineConsumerBehaviour | None = None,
    ) -> None:
        """
        Initialize the NED export pipeline.

        Args:
            grid (GeoDataFrame): The grid to which the data will be regridded.
            archive_storage (IngestArchiveStorage): The storage where the results will be archived.
            dataset_descriptor (NedDatasetDescriptor): The descriptor for the dataset to be
            exported.
            dataset_retriever (NedDataRetriever): The retriever for the dataset.
            dataset_reader (NedDataReader): The reader for the dataset.
            result_subpath (str): The subpath where the results will be stored.
            consumer_behaviour (PipelineConsumerBehaviour, optional): The behaviour of the
            consumers of the export pipeline. Defaults to None, which uses the default
            consumer behaviour.

        """
        super().__init__()
        self.dataset_descriptor = dataset_descriptor
        self.dataset_reader = dataset_reader
        self.dataset_retriever = dataset_retriever
        self.result_subpath = result_subpath
        self._grid = grid
        self.archive_storage = archive_storage
        self._lon = xarray.DataArray(
            self._grid.df["lon"],
            dims="points",
        )
        self._lat = xarray.DataArray(
            self._grid.df["lat"],
            dims="points",
        )
        self.consumer_behaviour = (
            consumer_behaviour if consumer_behaviour else PipelineConsumerBehaviour.default()
        )

    def upload(self) -> None:
        """
        Upload the data to the archive storage.

        This method retrieves the data files, regrids them to the specified grid,
        and writes the transformed data to the archive storage.
        """
        partial_dfs = []

        for file in self.dataset_retriever.stream_files(
            dataset_descriptor=self.dataset_descriptor,
        ):
            logger.info(
                "Loading %s for dataset %s",
                file,
                self.dataset_descriptor,
            )
            data = self.dataset_reader.extract_data(
                file=file,
                dataset_descriptor=self.dataset_descriptor,
            )

            logger.info(
                "Regridding data for %s for dataset %s",
                file,
                self.dataset_descriptor,
            )

            transformed_data = self._regrid(data)

            data_out = transformed_data[
                ["grid_id", "date", *self.dataset_descriptor.variable_mapping.keys()]
            ].rename(
                mapping=self.dataset_descriptor.variable_mapping,
            )

            partial_dfs.append(data_out)

        if not partial_dfs:
            msg = f"No data found for dataset {self.dataset_descriptor}."
            raise NedMissingDataError(msg)

        logger.info(
            "Writing result for dataset %s",
            self.dataset_descriptor,
        )
        # Concatenate all partial dataframes
        full_df = pl.concat(partial_dfs)

        full_df = self._add_missing_rows(full_df)

        self.archive_storage.write_to_destination(
            table=full_df,
            result_subpath=self.result_subpath,
        )

    def _add_missing_rows(self, full_df: DataFrame) -> DataFrame:
        dates = [
            date.format("YYYY-MM-DD")
            for date in Arrow.range(
                "day",
                start=self.dataset_descriptor.start_date,
                end=self.dataset_descriptor.end_date,
            )
        ]

        grids = self._grid.df["grid_id"].unique()

        # Get the type of the grid_id column
        grid_id_dtype = self._grid.df["grid_id"].dtype

        full_index = pl.DataFrame(
            product(
                dates,
                grids,
            ),
            schema={
                "date": pl.String,
                "grid_id": grid_id_dtype,
            },
        )

        return full_df.join(
            full_index,
            on=["date", "grid_id"],
            how="full",
            coalesce=True,
        )

    def requested_dates(self) -> frozenset[date]:
        """Return the calendar days this pipeline is configured to collect."""
        return dates_from_arrows(
            self.dataset_descriptor.start_date,
            self.dataset_descriptor.end_date,
        )

    def probe_coverage(self) -> CoverageProbeResult:
        """Search for Earthdata granules without downloading or subsetting."""
        return self.dataset_retriever.probe_coverage(
            dataset_descriptor=self.dataset_descriptor,
        )

    def get_config_metadata(self) -> PipelineConfig:
        """Get the expected result of the export operation."""
        return PipelineConfig(
            result_subpath=self.result_subpath,
            id_columns={"date", "grid_id"},
            value_column_type_map=dict.fromkeys(
                self.dataset_descriptor.variable_mapping.values(),
                ValueColumnType.FLOAT,
            ),
            expected_rows=self._grid.n_rows * self.dataset_descriptor.days_in_range,
            consumer_behaviour=self.consumer_behaviour,
        )

    def _regrid(self, data: NedDayData) -> DataFrame:
        """Regrid the data to the grid."""
        grid_data = self._grid.df
        interpolation_method = self.dataset_descriptor.interpolation_method

        sampled_values = data.data.interp(
            lon=self._lon,
            lat=self._lat,
            method=interpolation_method,  # Use the specified interpolation method
        )

        source_var_names = list(self.dataset_descriptor.variable_mapping.keys())

        new_columns = [
            pl.Series(
                name=source_var_name,
                values=sampled_values[source_var_name].values,
            )
            for source_var_name in source_var_names
        ]

        return grid_data.with_columns(
            pl.lit(data.date).alias("date"),
            *new_columns,
        )
