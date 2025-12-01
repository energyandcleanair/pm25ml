"""Sampling for data."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.logging import logger
from pm25ml.setup.temporal_config import TemporalConfig


class FullModelSampler:
    """
    Sampler class for data.

    It selects all data where the PM2.5 column has a value.
    """

    def __init__(
        self,
        combined_storage: CombinedStorage,
        temporal_config: TemporalConfig,
        column_name: str,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Initialize the ImputationSampler."""
        self.combined_storage = combined_storage
        self.temporal_config = temporal_config
        self.column_name = column_name
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def sample(self) -> None:
        """Sample the data for a given column to impute."""
        months = self.temporal_config.month_ids

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    self._process_month,
                    month=month,
                )
                for month in months
            ]
            for future in as_completed(futures):
                future.result()

    def _process_month(
        self,
        *,
        month: str,
    ) -> None:
        logger.info(
            f"Sampling for {self.column_name} for month: {month}",
        )
        monthly_data = self.combined_storage.read_dataframe(
            result_subpath=self.input_data_artifact.for_month(month),
        ).filter(
            pl.col(self.column_name).is_not_null(),
        )

        self.combined_storage.write_to_destination(
            monthly_data,
            self.output_data_artifact.for_month(month),
        )
