"""Validates that the expected results from a dataset match the path's expected results."""

from collections.abc import Collection

import arrow

from pm25ml.collectors.export_pipeline import ExportPipeline
from pm25ml.hive_path import HivePath


def validate_configuration(
    processors: Collection[ExportPipeline],
    expected_grid_cell_count: int,
) -> None:
    """
    Perform a preliminary check on the expected result.

    checks that the expected result is correct based off the metadata in the path.

    :param n_grids: The number of grids expected in the result.
    :raises ValueError: If the expected number of rows does not match the number of grids.
    """
    for processor in processors:
        _validate_single_processor_config(processor, expected_grid_cell_count)

    # All result paths are unique
    result_subpaths = {processor.get_config_metadata().result_subpath for processor in processors}
    if len(result_subpaths) != len(processors):
        msg = (
            "Duplicate result subpaths found in processors. "
            "Each processor must have a unique result subpath."
        )
        raise ValueError(msg)


def _validate_single_processor_config(
    processor: ExportPipeline,
    expected_grid_cell_count: int,
) -> None:
    expected_result = processor.get_config_metadata()

    path_metadata = HivePath(
        result_subpath=expected_result.result_subpath,
    )

    # Check that dataset is not empty
    path_metadata.require_key("dataset")
    path_metadata.require_key("country")
    expected_rows = _expected_row_count(
        path_metadata.metadata,
        n_grids=expected_grid_cell_count,
    )
    _assert(
        expected_result.expected_rows == expected_rows,
        (
            f"Expected {expected_rows} rows in {expected_result.result_subpath}, "
            f"but found {expected_result.expected_rows} rows based on the metadata."
        ),
    )

    expected_ids = _expected_ids(
        path_metadata.metadata,
    )
    _assert(
        expected_result.id_columns == expected_ids,
        (
            f"Expected ID columns {expected_ids} in {expected_result.result_subpath}, "
            f"but found {expected_result.id_columns}."
        ),
    )


def _expected_row_count(
    path_metadata: dict[str, str],
    n_grids: int,
) -> int:
    if "month" not in path_metadata:
        return n_grids

    month_value = arrow.get(path_metadata["month"])
    end_of_month = month_value.ceil("month")
    days_in_month = (end_of_month - month_value).days + 1

    return days_in_month * n_grids


def _expected_ids(path_metadata: dict[str, str]) -> set[str]:
    if "month" in path_metadata:
        return {"date", "grid_id"}
    return {"grid_id"}


# We use this assert because asserts are stripped out in production code,
# and we want to ensure that the validation logic is always executed.
def _assert(condition: bool, message: str) -> None:  # noqa: FBT001
    """
    Assert that a condition is true, otherwise raise a ValueError with the given message.

    :param condition: The condition to check.
    :param message: The message to include in the ValueError if the condition is false.
    :raises ValueError: If the condition is false.
    """
    if not condition:
        raise ValueError(message)
