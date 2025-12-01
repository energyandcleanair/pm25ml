"""Handles recombination of datasets into a single combined dataset."""

from collections import deque
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from arrow import Arrow

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.logging import logger
from pm25ml.setup.temporal_config import TemporalConfig


class Recombiner:
    """
    Recombines datasets from multiple stages and months into a single combined dataset.

    Attributes:
        combined_storage (CombinedStorage): The storage where the combined results will be stored.
        new_stage_name (str): The name of the new stage for the combined dataset.
        months (Collection[Arrow]): The months to recombine.

    """

    def __init__(
        self,
        *,
        combined_storage: CombinedStorage,
        temporal_config: TemporalConfig,
        output_data_artifact: DataArtifactRef,
        max_workers: int,
        force_recombine: bool = False,
    ) -> None:
        """
        Initialize the Recombiner with a storage for combined datasets.

        :param combined_storage: The storage where the combined results will be stored.
        :param new_stage_name: The name of the new stage for the combined dataset.
        """
        self.combined_storage = combined_storage
        self.output_data_artifact = output_data_artifact
        self.months = temporal_config.months
        self.max_workers = max_workers
        self.force_recombine = force_recombine

    def recombine(
        self,
        stages: Collection[DataArtifactRef],
        *,
        overwrite_columns: bool = False,
    ) -> None:
        """
        Recombine datasets from multiple stages and months into a single combined dataset.

        :param stages: The stages to recombine.
        :param overwrite_columns: Whether to overwrite shared columns or raise an error. Overwrites
        with the data in the rightmost dataframe.
        :raises ValueError: If shared columns are detected and overwrite_columns is False.
        """
        filtered_months = self._filter_months_to_update(
            stages=stages,
        )

        self._process_in_parallel(
            stages=stages,
            months=filtered_months,
            overwrite_columns=overwrite_columns,
        )

        self._validate_all(
            stages=stages,
            months=filtered_months,
        )

    def _filter_months_to_update(
        self,
        stages: Collection[DataArtifactRef],
    ) -> Collection[Arrow]:
        if self.force_recombine:
            return self.months

        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda month: (month, self._needs_recombining(month, stages)),
                self.months,
            )
        return [month for month, needs_recombining in results if needs_recombining]

    def _validate_all(
        self,
        stages: Collection[DataArtifactRef],
        months: Collection[Arrow],
    ) -> None:
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda month: self._validate_combined(month, stages),
                months,
            )

            deque(results)

    def _process_in_parallel(
        self,
        stages: Collection[DataArtifactRef],
        months: Collection[Arrow],
        *,
        overwrite_columns: bool = False,
    ) -> None:
        def process_month(month: Arrow) -> None:
            logger.debug(
                f"Recombining {stages} to {self.output_data_artifact.stage} "
                f"for month {month.format('YYYY-MM')}",
            )
            stage_dfs = self._read_dfs_to_merge(stages, month)

            # Combine all DataFrames
            combined_df = self._combine_all(
                stage_dfs, overwrite_columns=overwrite_columns
            )

            # Write the combined DataFrame to storage
            self.combined_storage.write_to_destination(
                combined_df,
                self.output_data_artifact.for_month(month.format("YYYY-MM")),
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            result = executor.map(process_month, months)
            deque(result)

    def _read_dfs_to_merge(
        self,
        stages: Collection[DataArtifactRef],
        month: Arrow,
    ) -> list[pl.DataFrame]:
        return [
            self.combined_storage.read_dataframe(
                stage.for_month(month.format("YYYY-MM")),
            )
            for stage in stages
        ]

    def _combine_all(
        self,
        stage_dfs: list[pl.DataFrame],
        *,
        overwrite_columns: bool,
    ) -> pl.DataFrame:
        id_columns = {"grid_id", "date"}

        combined_df = stage_dfs[0]
        for df in stage_dfs[1:]:
            shared_columns = (set(combined_df.columns) & set(df.columns)) - id_columns

            if shared_columns:
                if overwrite_columns:
                    # Drop shared columns from the left DataFrame
                    logger.debug(
                        f"Dropping shared columns {shared_columns} from the left DataFrame "
                        "to overwrite with the right DataFrame.",
                    )
                    combined_df = combined_df.drop(shared_columns)
                else:
                    error_message = (
                        f"Shared columns detected: {shared_columns}. "
                        "Set overwrite_columns=True to overwrite them."
                    )
                    raise ValueError(error_message)

            shared_id_columns = set(combined_df.columns) & set(df.columns) & id_columns

            combined_df = combined_df.join(
                df,
                on=list(shared_id_columns),
                how="full",
                coalesce=True,
            )
        return combined_df

    def _needs_recombining(
        self, month: Arrow, stages: Collection[DataArtifactRef]
    ) -> bool:
        logger.debug(
            f"Checking if data for month {month.format('YYYY-MM')} needs to be recombined.",
        )

        if not self.combined_storage.does_dataset_exist(
            self.output_data_artifact.for_month(month.format("YYYY-MM")),
        ):
            return True

        try:
            self._validate_combined(month, stages)
        except ValueError as exc:
            logger.debug(
                f"Data doesn't match expected schema for month {month.format('YYYY-MM')}, "
                f"requesting re-combination: {exc}",
                exc_info=True,
            )
            return True

        return False

    def _validate_combined(
        self, month: Arrow, stages: Collection[DataArtifactRef]
    ) -> None:
        month_short = month.format("YYYY-MM")

        # Collect expected columns from all stages
        expected_columns = set()
        for stage in stages:
            metadata = self.combined_storage.read_dataframe_metadata(
                stage.for_month(month_short),
            )
            expected_columns.update(metadata.schema.names)

        first_stage = next(iter(stages))
        expected_rows = self.combined_storage.read_dataframe_metadata(
            first_stage.for_month(month_short),
        ).num_rows

        # Read metadata of the combined dataset
        combined_metadata = self.combined_storage.read_dataframe_metadata(
            self.output_data_artifact.for_month(month_short),
        )
        actual_columns = set(combined_metadata.schema.names)

        logger.debug(
            f"Validating recombined {self.output_data_artifact.stage} for month {month_short}: "
            f"expecting {expected_rows} rows and {len(expected_columns)} columns.",
        )

        if combined_metadata.num_rows != expected_rows:
            msg = (
                f"Expected {expected_rows} rows in the recombined result for month {month_short}, "
                f"but found {combined_metadata.num_rows} rows."
            )
            raise ValueError(msg)

        missing_columns = expected_columns - actual_columns
        if missing_columns:
            msg = (
                f"Expected columns {expected_columns} in the final combined result, "
                f"but {missing_columns} were missing."
            )
            raise ValueError(
                msg,
            )
