"""Export pipeline for Google Earth Engine data to the underlying storage."""

from __future__ import annotations

import contextlib
from itertools import product
from time import sleep
from typing import TYPE_CHECKING

from ee.batch import Export, Task
from nanoid import generate
from polars import DataFrame, Float32, Int64, String

from pm25ml.collectors.export_pipeline import (
    ExportPipeline,
    MissingDataError,
    PipelineConfig,
    PipelineConsumerBehaviour,
    ValueColumnType,
)
from pm25ml.logging import logger

if TYPE_CHECKING:
    from pm25ml.collectors.archive_storage import IngestArchiveStorage
    from pm25ml.collectors.gee.feature_planner import FeaturePlan

    from .intermediate_storage import GeeIntermediateStorage


class GeeExportPipeline(ExportPipeline):
    """Handles the export of data from GEE to the specified storage."""

    def __init__(
        self,
        *,
        intermediate_storage: GeeIntermediateStorage,
        archive_storage: IngestArchiveStorage,
        plan: FeaturePlan,
        result_subpath: str,
        pipeline_consumer_behaviour: PipelineConsumerBehaviour | None = None,
    ) -> None:
        """Initialize the GeeExportPipeline with the storage and plan."""
        self.archive_storage = archive_storage
        self.intermediate_storage = intermediate_storage
        self.plan = plan
        self.result_subpath = result_subpath
        self.pipeline_consumer_behaviour = (
            pipeline_consumer_behaviour
            if pipeline_consumer_behaviour
            else PipelineConsumerBehaviour.default()
        )

    def upload(self) -> None:
        """Upload the data from GEE to the underlying storage."""
        temporary_file_prefix = generate(size=10)
        task_name = f"{temporary_file_prefix}__{self.plan.feature_name}"[:100]

        if not self.plan.is_data_available():
            msg = f"Data for feature '{self.plan.feature_name}' is not available in GEE."
            raise MissingDataError(msg)

        # First, we define the task to export the data to GCS, run it, and then wait until it
        # completes.
        logger.info(f"Task {task_name}: starting task")
        task = self._define_task(
            task_name=task_name,
        )
        self._complete_task(task_name=task_name, task=task)

        # Now that the task is complete, we can read the CSV file from GCS.
        logger.debug(f"Task {task_name}: reading task result CSV from GCS")
        raw_table = self.intermediate_storage.get_intermediate_by_id(task_name)

        # After reading the CSV file, we process it.
        logger.debug(f"Task {task_name}: processing raw CSV table for task")
        processed_table = self._process(raw_table)

        # Then we write the processed table to the destination bucket format.
        logger.debug(f"Task {task_name}: writing task processed table to GCS {self.result_subpath}")
        self.archive_storage.write_to_destination(processed_table, self.result_subpath)

        # Finally, we delete the temporary CSV file from the intermediate bucket. This should happen
        # in the future anyway with the bucket lifecycle, but we do it now to clean up the
        # intermediate storage.
        logger.debug(f"Task {task_name}: deleting task old CSV file from GCS")
        self.intermediate_storage.delete_intermediate_by_id(task_name)

    def get_config_metadata(self) -> PipelineConfig:
        """Get the expected result of the export operation."""
        return PipelineConfig(
            result_subpath=self.result_subpath,
            id_columns=self.plan.expected_id_columns,
            value_column_type_map=dict.fromkeys(
                self.plan.expected_value_columns,
                ValueColumnType.FLOAT,
            ),
            expected_rows=self.plan.expected_n_rows,
            consumer_behaviour=self.pipeline_consumer_behaviour,
        )

    def _define_task(self, task_name: str) -> Task:
        exported_properties = self.plan.intermediate_columns
        return Export.table.toCloudStorage(
            description=task_name,
            collection=self.plan.planned_collection,
            bucket=self.intermediate_storage.bucket,
            fileNamePrefix=task_name,
            fileFormat="CSV",
            selectors=exported_properties if not self.plan.ignore_selectors else None,
        )

    def _complete_task(self, *, task_name: str, task: Task) -> None:
        try:
            task.start()
            delay_backoff = 1.0
            growth_factor = 1.5
            max_delay = 10.0
            while task.active():
                logger.debug(
                    f"Task {task_name}: waiting for task to complete ({delay_backoff}s delay)",
                )
                sleep(delay_backoff)
                delay_backoff = min(max_delay, delay_backoff * growth_factor)

            if task.status().get("state") != "COMPLETED":
                logger.warning(f"Task {task_name} failed with status: {task.status()}")
                error_message = task.status().get("error_message", "No error message")
                msg = f"Task {task_name} failed: {error_message}"
                raise RuntimeError(msg)
        finally:
            with contextlib.suppress(Exception):
                task.cancel()

    def _process(self, table: DataFrame) -> DataFrame:
        table = self._validate_and_rename_columns(table)
        table = self._ensure_non_empty_export(table)
        table = self._normalize_grid_id(table)
        table = self._complete_missing_rows(table)
        table = self._coerce_value_columns(table)
        self._validate_non_null_columns(table)
        return self._sort_processed_table(table)

    def _validate_and_rename_columns(self, table: DataFrame) -> DataFrame:
        expected_intermediate_columns = self.plan.intermediate_columns

        missing_columns = [col for col in expected_intermediate_columns if col not in table.columns]
        if missing_columns:
            msg = f"Table is missing expected columns: {', '.join(missing_columns)}"
            raise ValueError(msg)

        extra_columns = [col for col in table.columns if col not in expected_intermediate_columns]
        if extra_columns:
            logger.warning(f"Dropping extra columns from table: {', '.join(extra_columns)}")
            table = table.drop(extra_columns)

        return table.rename(self.plan.column_mappings)

    def _ensure_non_empty_export(self, table: DataFrame) -> DataFrame:
        if table.height == 0:
            msg = f"No rows were exported for feature '{self.plan.feature_name}'."
            raise MissingDataError(msg)
        return table

    def _normalize_grid_id(self, table: DataFrame) -> DataFrame:
        if "grid_id" not in table.columns:
            return table
        return table.with_columns(table["grid_id"].cast(Int64))

    def _complete_missing_rows(self, table: DataFrame) -> DataFrame:
        if "date" not in table.columns or "grid_id" not in table.columns:
            return table

        if not self.plan.dates:
            msg = "Feature plan does not have dates defined but has a date column."
            raise ValueError(msg)

        dates = [date.format("YYYY-MM-DDTHH:mm:ss") for date in self.plan.dates]
        grid_ids = table["grid_id"].unique().to_list()
        full_index = DataFrame(
            product(dates, grid_ids),
            schema={"date": String, "grid_id": Int64},
            orient="row",
        )
        return table.join(
            full_index,
            on=["date", "grid_id"],
            how="full",
            coalesce=True,
        )

    def _coerce_value_columns(self, table: DataFrame) -> DataFrame:
        for value_column in self.plan.expected_value_columns:
            if table[value_column].dtype == Float32():
                continue

            logger.warning(
                f"Coercing column '{value_column}' to float32 from {table[value_column].dtype}",
            )
            table = table.with_columns(table[value_column].cast(Float32(), strict=False))
        return table

    def _validate_non_null_columns(self, table: DataFrame) -> None:
        columns_null_values = [
            col
            for col in self.plan.expected_value_columns.union(self.plan.expected_id_columns)
            if table[col].null_count() == table.height
        ]
        if columns_null_values:
            msg = f"Table has columns with all null values: {', '.join(columns_null_values)}"
            raise ValueError(msg)

    def _sort_processed_table(self, table: DataFrame) -> DataFrame:
        preferred_sort_order = ["date", "grid_id"]
        columns_to_sort = [col for col in preferred_sort_order if col in table.columns]
        if not columns_to_sort:
            return table
        return table.sort(by=columns_to_sort)


class GeePipelineConstructor:
    """
    A constructor for GeeExportPipeline that allows for a more fluent interface.

    Should be used with `GeeExportPipeline.with_storage()`.
    """

    def __init__(
        self,
        *,
        intermediate_storage: GeeIntermediateStorage,
        archive_storage: IngestArchiveStorage,
    ) -> None:
        """Initialize the GeePipelineConstructor with the storage."""
        self.archive_storage = archive_storage
        self.intermediate_storage = intermediate_storage

    def construct(
        self,
        plan: FeaturePlan,
        result_subpath: str,
        pipeline_consumer_behaviour: PipelineConsumerBehaviour | None = None,
    ) -> GeeExportPipeline:
        """
        Construct a GeeExportPipeline with the given plan and result subpath.

        :param plan: The feature plan to use for the export.
        :param result_subpath: The subpath in the destination bucket where the results will be
        stored.
        """
        return GeeExportPipeline(
            archive_storage=self.archive_storage,
            intermediate_storage=self.intermediate_storage,
            plan=plan,
            result_subpath=result_subpath,
            pipeline_consumer_behaviour=pipeline_consumer_behaviour,
        )
