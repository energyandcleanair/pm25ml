"""ExportPipeline interface for exporting data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pm25ml.hive_path import HivePath

AvailableIdKeys = Literal["date", "grid_id"]
AVAILABLE_ID_KEY_NAMES: list[AvailableIdKeys] = ["date", "grid_id"]


class ExportPipeline(ABC):
    """
    Abstract base class for export pipelines.

    Responsible for orchestrating the exporting the data from the origin, transforming it into the
    grid format, and uploading it to the underlying storage.

    It must raise MissingDataError if the export pipeline is missing a significant amount of data
    (for example a whole month).

    The file uploaded must be in a parquet format and must contain the every grid ID for the map.
    If it includes dates, it must include every date grid ID combination for the month being
    exported.

    The pipeline config metadata must match the expected result of the export operation.
    """

    @abstractmethod
    def upload(self) -> None:
        """Export the given data."""

    @abstractmethod
    def get_config_metadata(self) -> PipelineConfig:
        """
        Get the expected result of the export operation.

        This method should be overridden by subclasses to provide the expected result.
        """


@dataclass(frozen=True)
class PipelineConsumerBehaviour:
    """
    Represents the behaviour of consumers of the export pipeline.

    Pipelines must only use this class to communicate how consumers of the export pipeline
    should behave when consuming the results of the export operation - not to modify their
    own behaviour.
    """

    missing_data_heuristic: MissingDataHeuristic

    @staticmethod
    def default() -> PipelineConsumerBehaviour:
        """
        Get the default consumer behaviour.

        This is used when the export pipeline does not specify a consumer behaviour.
        """
        return PipelineConsumerBehaviour(missing_data_heuristic=MissingDataHeuristic.FAIL)


class MissingDataHeuristic(Enum):
    """
    Represents the heuristic used to determine if data is missing.

    It must not be used by the export pipeline directly. It's used to communicate to consumers of
    the export pipeline how to handle missing data in the export operation.
    """

    FAIL = ("fail", False)
    """
    Fail the export operation if data is missing.
    """

    COPY_LATEST_AVAILABLE_BEFORE = ("copy_latest_available_before", True)
    """
    Copy the latest available data that's before the missing data, if missing.
    """

    def __init__(self, type_name: str, allows_missing: bool) -> None:  # noqa: FBT001
        """
        Initialize the MissingDataHeuristic with a name and allows_missing flag.

        :param type_name: The name of the heuristic.
        :param allows_missing: Whether the heuristic allows missing data.
        """
        self.type_name = type_name
        self.allows_missing = allows_missing


class ValueColumnType(Enum):
    """
    Represents the type of value columns in the export pipeline.

    This is used to specify the type of value columns in the export pipeline.
    """

    FLOAT = "float"
    INT = "int"

    def __str__(self) -> str:
        """Return the string representation of the value column type."""
        return self.value


@dataclass(frozen=True)
class PipelineConfig:
    """
    Represents the pipeline config metadata of an export operation.

    This is standardised across all export pipelines and can be used to check the
    configuration of the export operation and validate the results after the export
    is complete.
    """

    result_subpath: str
    """
    The subpath where the result will be stored.
    """
    id_columns: set[AvailableIdKeys]
    """
    The expected ID columns in the result.
    """
    value_column_type_map: dict[str, ValueColumnType]
    """
    The expected value columns in the result.
    """
    expected_rows: int
    """
    The expected number of rows in the result.
    """
    consumer_behaviour: PipelineConsumerBehaviour = field(
        default_factory=PipelineConsumerBehaviour.default,
    )
    """
    How consumers of the export pipeline should behave.
    """
    constrains_end_month: bool = True
    """
    Whether this pipeline has a missing-data check that can constrain automatic end-month
    discovery. Pipelines without such a check are collected normally after a month is selected.
    """

    @property
    def all_columns(self) -> set[str]:
        """
        Get all columns in the result.

        :return: A set of all columns in the result.
        """
        return self.id_columns.union(self.value_column_type_map.keys())

    @property
    def hive_path(self) -> HivePath:
        """
        Get the HivePath representation of the result subpath.

        :return: A HivePath object representing the result subpath.
        """
        return HivePath(self.result_subpath)

    @property
    def allows_missing_data(self) -> bool:
        """
        Check if the consumer behaviour allows missing data.

        :return: True if the consumer behaviour allows missing data, False otherwise.
        """
        return self.consumer_behaviour.missing_data_heuristic.allows_missing

    @property
    def value_columns(self) -> set[str]:
        """
        Get the value columns in the result.

        :return: A set of value columns in the result.
        """
        return set(self.value_column_type_map.keys())


class MissingDataError(Exception):
    """
    Exception raised when the export pipeline is missing data.

    This exception is raised when the export pipeline cannot find the expected data
    for the export operation.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the MissingDataError with a message.

        :param message: The error message to be displayed.
        """
        super().__init__(message)


class ErrorWhileFetchingDataError(Exception):
    """
    Exception raised when there is an error while fetching data for the export pipeline.

    This exception is raised when the export pipeline encounters an error while trying to
    fetch the data required for the export operation.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the ErrorWhileFetchingDataError with a message.

        :param message: The error message to be displayed.
        """
        super().__init__(message)
