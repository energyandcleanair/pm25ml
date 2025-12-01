"""Type helpers for providers."""

from typing import Protocol

from pm25ml.model_reference import ImputationModelReference
from pm25ml.training.imputation_model_pipeline import (
    ImputationModelPipeline,
)


class ModelTrainerFactory(Protocol):
    """
    Protocol for a factory that creates ModelPipeline instances.

    This factory is used to create model trainers for different models.
    """

    def __call__(
        self,
        model_reference: ImputationModelReference,
    ) -> ImputationModelPipeline:
        """Create a ModelPipeline instance for the given model reference."""
        ...
