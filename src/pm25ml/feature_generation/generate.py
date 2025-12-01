"""
Feature generation for PM2.5 data.

The logic to generate the features themselves is passed in to avoid tying the
generator to the model's implementation.
"""

from typing import Callable

import polars as pl

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.setup.temporal_config import TemporalConfig


class FeatureGenerator:
    """Class to generate features for PM2.5 data."""

    def __init__(
        self,
        combined_storage: CombinedStorage,
        temporal_config: TemporalConfig,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
        generate_for_year: Callable[[TemporalConfig, pl.LazyFrame, int], pl.LazyFrame],
    ) -> None:
        """
        Initialize the FeatureGenerator.

        :param combined_storage: Combined storage instance.
        :param temporal_config: Temporal configuration.
        :param input_data_artifact: Input data artifact reference.
        :param output_data_artifact: Output data artifact reference.
        :param generate_for_year: Function to generate features for a specific year.

        """
        self.combined_storage = combined_storage
        self.temporal_config = temporal_config
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact
        self.generate_for_year = generate_for_year

    def generate(self) -> None:
        """Generate features for PM2.5 data."""
        lf = self.combined_storage.scan_stage(self.input_data_artifact.stage)
        for year in self.temporal_config.years:

            self.combined_storage.sink_stage(
                self.generate_for_year(self.temporal_config, lf, year),
                self.output_data_artifact.stage,
            )
