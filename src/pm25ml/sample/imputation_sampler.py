"""Sampling for data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import polars as pl

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.logging import logger
from pm25ml.setup.temporal_config import TemporalConfig


@dataclass
class ImputationSamplerDefinition:
    """Configuration for imputation sampling."""

    value_column: str
    model_name: str
    percentage_sample: float


class SpatialTemporalImputationSampler:
    """
    Sampler class for data.

    Groups by data by the year-month and the grid_id_50km. It samples the dataset for imputing the
    specified value_column, selecting only the rows where the value_column is not null.
    """

    def __init__(
        self,
        combined_storage: CombinedStorage,
        temporal_config: TemporalConfig,
        imputation_sampler_definition: ImputationSamplerDefinition,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Initialize the ImputationSampler."""
        self.combined_storage = combined_storage
        self.temporal_config = temporal_config
        self.imputation_sampler_definition = imputation_sampler_definition
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
            f"Sampling for {self.imputation_sampler_definition.value_column} for month: {month}",
        )
        monthly_data = self.combined_storage.read_dataframe(
            result_subpath=self.input_data_artifact.for_month(month),
        ).filter(
            pl.col(self.imputation_sampler_definition.value_column).is_not_null(),
        )

        sampled_keys = (
            monthly_data.select(
                "grid_id",
                "date",
                "grid__id_50km",
            )
            .group_by("grid__id_50km")
            .map_groups(
                lambda df: df.sample(
                    fraction=self.imputation_sampler_definition.percentage_sample,
                    with_replacement=False,
                    seed=42,
                ),
            )
            .select(
                "grid_id",
                "date",
            )
            .with_columns(
                pl.lit("training").alias("split"),
            )
        )

        split_dataset = monthly_data.join(
            sampled_keys,
            on=["grid_id", "date"],
            how="left",
            coalesce=True,
        ).with_columns(
            split=pl.col("split").fill_null("test"),
        )

        self.combined_storage.write_to_destination(
            split_dataset,
            self.output_data_artifact.for_month(month),
        )
