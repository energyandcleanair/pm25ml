"""Configuration and definition of samplers for PM2.5 ML project."""

from collections.abc import Collection
from dataclasses import dataclass

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.sample.imputation_sampler import (
    ImputationSamplerDefinition,
    SpatialTemporalImputationSampler,
)
from pm25ml.setup.temporal_config import TemporalConfig


@dataclass
class ImputationStep:
    """Configuration for imputation steps."""

    imputation_sampler_definition: ImputationSamplerDefinition


def define_samplers(
    combined_storage: CombinedStorage,
    temporal_config: TemporalConfig,
    imputation_steps: Collection[ImputationStep],
    input_data_artifact: DataArtifactRef,
    output_data_artifact: DataArtifactRef,
) -> Collection[SpatialTemporalImputationSampler]:
    """Define samplers for the PM2.5 ML project."""
    return [
        SpatialTemporalImputationSampler(
            combined_storage=combined_storage,
            temporal_config=temporal_config,
            imputation_sampler_definition=step.imputation_sampler_definition,
            input_data_artifact=input_data_artifact,
            output_data_artifact=output_data_artifact.for_sub_artifact(
                step.imputation_sampler_definition.model_name,
            ),
        )
        for step in imputation_steps
    ]
