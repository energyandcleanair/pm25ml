"""Manage the spatial imputation of data using a specified imputer."""

from collections import deque
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from arrow import Arrow

from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.imputation.spatial.daily_spatial_interpolator import (
    DailySpatialInterpolator,
)
from pm25ml.logging import logger
from pm25ml.setup.temporal_config import TemporalConfig


class SpatialImputationValidationError(Exception):
    """An error raised when the spatial imputation results do not match expectations."""


class SpatialImputationManager:
    """Manage the spatial imputation of data using a specified imputer."""

    def __init__(
        self,
        combined_storage: CombinedStorage,
        spatial_imputer: DailySpatialInterpolator,
        temporal_config: TemporalConfig,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """
        Initialize the SpatialImputationManager.

        :param combined_storage: The storage where combined data is stored.
        :param spatial_imputer: The imputer used for spatial interpolation.
        :param months: A collection of Arrow objects representing the months to process.
        """
        self.combined_storage = combined_storage
        self.spatial_imputer = spatial_imputer
        self.months = temporal_config.months
        self.months_as_ids = [month.format("YYYY-MM") for month in self.months]
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def impute(self) -> None:
        """Perform spatial imputation for each month."""
        column_regex = self.spatial_imputer.value_column_regex_selector

        logger.info(
            f"Scanning combined monthly data for spatial imputation with regex: {column_regex}",
        )
        combined_dataset = self.combined_storage.scan_stage("combined_monthly").select(
            "month",
            "grid_id",
            "date",
            pl.col(self.spatial_imputer.value_column_regex_selector),
        )

        logger.info(
            "Checking if all expected months are present in the dataset we're going to impute for",
        )
        self._check_all_months_present(combined_dataset)

        expected_columns = combined_dataset.collect_schema().names()
        months_to_upload = self._identify_months_to_upload(expected_columns)

        logger.info(
            f"Found {len(months_to_upload)} months to process for spatial imputation.",
        )
        self._impute_all_months(
            ds=combined_dataset,
            months_to_upload=months_to_upload,
        )

        logger.info(
            "Checking the results of spatial imputation",
        )

        self._validate_all(months_to_upload, expected_columns)

    def _impute_all_months(
        self,
        ds: pl.LazyFrame,
        months_to_upload: Collection[str],
    ) -> None:
        column_regex = self.spatial_imputer.value_column_regex_selector

        def process_month(month: Arrow) -> None:
            """Process spatial imputation for a specific month."""
            logger.debug(f"Spatially imputing data for month: {month}")

            month_df = ds.filter(pl.col("month") == month).collect(engine="streaming")

            expected_length = month_df.select(pl.len()).to_series()[0]

            imputed_month = self.spatial_imputer.impute(month_df).select(
                "grid_id",
                "date",
                pl.col(column_regex),
            )

            actual_length = imputed_month.select(pl.len()).to_series()[0]

            if actual_length != expected_length:
                msg = f"Imputed month {month} has length {actual_length}, expected {expected_length}."
                raise ValueError(msg)

            self.combined_storage.write_to_destination(
                imputed_month,
                self.output_data_artifact.for_month(month.format("YYYY-MM")),
            )

        with ThreadPoolExecutor(8) as executor:
            executor.map(process_month, months_to_upload)

    def _validate_all(
        self,
        months_to_upload: Collection[str],
        expected_columns: Collection[str],
    ) -> None:
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda month: self._validate_result(
                    month=month,
                    expected_columns=expected_columns,
                ),
                months_to_upload,
            )

            deque(results)

    def _identify_months_to_upload(
        self, expected_columns: Collection[str]
    ) -> Collection[str]:
        with ThreadPoolExecutor() as executor:
            months_to_upload = list(
                executor.map(
                    lambda month: (
                        month
                        if self._needs_upload(
                            month=month,
                            expected_columns=expected_columns,
                        )
                        else None
                    ),
                    self.months_as_ids,
                ),
            )
        return [month for month in months_to_upload if month is not None]

    def _needs_upload(self, month: str, expected_columns: Collection[str]) -> bool:
        logger.debug(f"Checking if spatial imputation month {month} needs upload")
        if not self.combined_storage.does_dataset_exist(
            self.output_data_artifact.for_month(month),
        ):
            logger.debug(f"Dataset for month {month} does not exist, needs upload.")
            return True

        try:
            self._validate_result(month=month, expected_columns=expected_columns)
        except SpatialImputationValidationError as exc:
            logger.debug(
                f"Data for month {month} does not match expected schema, needs re-upload: {exc}",
                exc_info=True,
            )
            return True

        return False

    def _validate_result(
        self,
        *,
        month: str,
        expected_columns: Collection[str],
    ) -> None:
        expected_rows = self._days_in_month(month) * VALID_COUNTRIES["india"]

        final_combined_metadata = self.combined_storage.read_dataframe_metadata(
            result_subpath=self.output_data_artifact.for_month(month),
        )

        actual_schema = final_combined_metadata.schema.to_arrow_schema()
        n_rows = final_combined_metadata.num_rows

        as_str = str(actual_schema).replace("\n", " | ")
        logger.debug(
            f"Validating imputed data for month {month}: {n_rows} rows, and schema {as_str}",
        )

        if n_rows != expected_rows:
            msg = (
                f"Expected {expected_rows} rows in the final imputed result for month {month}, "
                f"but found {n_rows} rows."
            )
            raise SpatialImputationValidationError(msg)

        actual_columns = set(actual_schema.names)
        missing = set(expected_columns) - actual_columns - {"month"}
        if missing:
            msg = (
                f"Expected columns {set(expected_columns)} in the final imputed result for month "
                f"{month}, but {missing} were missing."
            )
            raise SpatialImputationValidationError(msg)

    def _check_all_months_present(self, ds: pl.LazyFrame) -> None:
        available_to_process = (
            ds.select("month").unique().collect(engine="streaming").to_series().sort()
        )

        missing_months = set(self.months_as_ids) - set(available_to_process)
        if missing_months:
            missing_months_text = ", ".join(sorted(missing_months))
            msg = f"The following months are not present in the dataset: {missing_months_text}."
            raise ValueError(msg)

    def _days_in_month(self, month: str) -> int:
        """Return the number of days in a given month."""
        month_as_arrow = Arrow.strptime(month, "%Y-%m")
        month_end = month_as_arrow.shift(months=1).replace(day=1)
        return (month_end - month_as_arrow).days
