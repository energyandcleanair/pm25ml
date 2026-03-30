"""Defines the protocol for final statistics writing."""

from typing import Protocol

from pm25ml.training.model_storage import LoadedValidationMetadata


class FinalStatsWriter(Protocol):
    """A protocol for writing final model statistics outputs."""

    def write(self, stats: LoadedValidationMetadata) -> None:
        """Write final model statistics to storage."""
