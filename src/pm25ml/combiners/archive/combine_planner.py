"""
Planner for combining data for a specific month.

It abstracts the logic for determining which data paths are needed to be combined
for a given month:
- Finding the paths that need to be merged
- Handling missing data heuristics for datasets missing data

It also provides information needed to validate the combined data by providing
the expected columns and the number of rows that should be present in the final
combined dataset for the month.
"""

from collections.abc import Collection
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal

from arrow import Arrow

from pm25ml.collectors.collector import UploadResult
from pm25ml.collectors.export_pipeline import MissingDataHeuristic
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.hive_path import HivePath
from pm25ml.setup.temporal_config import TemporalConfig


@dataclass(frozen=True)
class CombinePlan:
    """Descriptor for combining data for a specific month."""

    month: Arrow
    """
    The month in YYYY-MM format that this descriptor applies to.
    """

    paths: AbstractSet[HivePath]
    """
    The paths to the data that should be combined for this month.
    """

    expected_columns: AbstractSet[str]
    """
    The columns that are expected in the combined data for this month.
    This includes both ID columns and value columns.
    """

    @property
    def month_id(self) -> str:
        """The month in 'YYYY-MM' format."""
        return _month_format(self.month)

    @property
    def expected_rows(self) -> int:
        """The number of rows expected in the combined data for this month."""
        return VALID_COUNTRIES["india"] * self.days_in_month

    @property
    def days_in_month(self) -> int:
        """The number of days in the month."""
        return (self.month.ceil("month") - self.month.floor("month")).days + 1


class CombinePlanner:
    """Planner for combining data for multiple months."""

    def __init__(self, temporal_config: TemporalConfig) -> None:
        """
        Initialize the CombinePlanner.

        :param months: A collection of Arrow objects representing the months to combine.
        """
        self.months = temporal_config.months

    def plan(
        self,
        results: Collection[UploadResult],
    ) -> Collection[CombinePlan]:
        """Choose what to combine for each month."""
        all_id_columns = {
            column for result in results for column in result.pipeline_config.id_columns
        }

        all_value_columns = {
            f"{result.pipeline_config.hive_path.require_key('dataset')}__{column}"
            for result in results
            for column in result.pipeline_config.value_column_type_map
        }

        all_expected_columns = all_id_columns | all_value_columns

        return [
            CombinePlan(
                month=month,
                paths=set(self._list_paths_to_merge(month, results)),
                expected_columns=all_expected_columns,
            )
            for month in self.months
        ]

    def _list_paths_to_merge(
        self,
        month: Arrow,
        results: Collection[UploadResult],
    ) -> Collection[HivePath]:
        dataset_groups = _DatasetResultGroup.group_by_dataset(results)

        return [group.get_best_matching(month) for group in dataset_groups]


@dataclass(frozen=True)
class _DatasetResultGroup:
    """Group of results for a specific dataset."""

    dataset: str
    results: Collection[UploadResult]

    def get_best_matching(self, month: Arrow) -> HivePath:
        """
        Get the best matching HivePath for the given month.

        Usually, this will return the HivePath that matches "static", "month", or "year"
        based on the dataset key.

        If the dataset is empty and has a heuristic that allows a selection of a fallback dataset,
        it will use the heuristic to find the matching data.
        """
        key = self.dataset_key
        value = self.extract_value(month)

        actual_match = next(
            result
            for result in self.results
            if result.pipeline_config.hive_path.metadata.get(key) == value
        )

        if actual_match.completeness.data_available:
            return actual_match.pipeline_config.hive_path

        missing_data_heuristic = (
            actual_match.pipeline_config.consumer_behaviour.missing_data_heuristic
        )

        if missing_data_heuristic == MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE:
            earlier_available_paths = (
                result.pipeline_config.hive_path
                for result in self.results
                if result.completeness.data_available
                and result.pipeline_config.hive_path.metadata[key] < value
            )

            return max(
                earlier_available_paths,
                key=lambda result: result.metadata[key],
            )

        msg = (
            f"No matching HivePath found for '{self.dataset}' with '{key}'='{value}'. "
            "All results for this dataset are empty or do not match the expected key-value pair."
        )
        raise ValueError(
            msg,
        )

    @property
    def dataset_key(
        self,
    ) -> Literal["type", "month", "year"]:
        if all(
            result.pipeline_config.hive_path.metadata.get("type") == "static"
            for result in self.results
        ):
            return "type"

        if all(
            result.pipeline_config.hive_path.metadata.get("month")
            for result in self.results
        ):
            return "month"

        if all(
            result.pipeline_config.hive_path.metadata.get("year")
            for result in self.results
        ):
            return "year"

        msg = (
            "Cannot determine dataset key for results. "
            "All results must have the same metadata key for 'type', 'month', or 'year'."
        )
        raise ValueError(msg)

    def extract_value(
        self,
        month: Arrow,
    ) -> str:
        if self.dataset_key == "type":
            return "static"
        if self.dataset_key == "month":
            return _month_format(month)
        if self.dataset_key == "year":
            return str(month.year)
        msg = (
            "Cannot extract value for dataset key. "
            "Dataset key must be one of 'type', 'month', or 'year'."
        )
        raise ValueError(msg)

    @staticmethod
    def group_by_dataset(
        results: Collection[UploadResult],
    ) -> Collection["_DatasetResultGroup"]:
        hive_paths = [result.pipeline_config.hive_path for result in results]

        dataset_names = {path.metadata["dataset"] for path in hive_paths}

        return [
            _DatasetResultGroup(
                dataset=dataset,
                results=[
                    result
                    for result in results
                    if result.pipeline_config.hive_path.metadata.get("dataset")
                    == dataset
                ],
            )
            for dataset in dataset_names
        ]


def _month_format(month: Arrow) -> str:
    """
    Format the month in 'YYYY-MM' format.

    :param month: The month to format.
    :return: The formatted month string.
    """
    return month.format("YYYY-MM")
