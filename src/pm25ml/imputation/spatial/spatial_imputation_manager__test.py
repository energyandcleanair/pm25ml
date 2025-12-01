import pytest
from unittest.mock import MagicMock
from arrow import Arrow
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.hive_path import HivePath
from pm25ml.imputation.spatial.spatial_imputation_manager import (
    SpatialImputationManager,
)
import polars as pl
from pyarrow.parquet import FileMetaData
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pyarrow import schema

from pm25ml.setup.temporal_config import TemporalConfig

INPUT_DATA_ARTIFACT = DataArtifactRef(stage="combined_monthly")
OUTPUT_DATA_ARTIFACT = DataArtifactRef(stage="era5_spatially_imputed")


@pytest.fixture
def fake_data_with_missing():
    return pl.DataFrame(
        {
            "month": ["2023-01", "2023-01", "2023-02", "2023-02"],
            "grid_id": [1, 2, 1, 2],
            "date": ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
            "value_column_regex": [10, None, None, 40],
        }
    )


@pytest.fixture
def fake_data_result_map():
    return {
        "2023-01": pl.DataFrame(
            {
                "month": ["2023-01", "2023-01"],
                "grid_id": [1, 2],
                "date": ["2023-01-01", "2023-01-02"],
                "value_column_regex": [10, 20],
            }
        ),
        "2023-02": pl.DataFrame(
            {
                "month": ["2023-02", "2023-02"],
                "grid_id": [1, 2],
                "date": ["2023-02-01", "2023-02-02"],
                "value_column_regex": [30, 40],
            }
        ),
    }


@pytest.fixture
def mock_imputer():
    spatial_imputer_mock = MagicMock()
    spatial_imputer_mock.value_column_regex_selector = "value_column_regex"
    return spatial_imputer_mock


@pytest.fixture
def mock_imputer_fills_missing(mock_imputer, fake_data_result_map):
    mock_imputer.impute.side_effect = lambda df: fake_data_result_map[
        df["month"].unique()[0]
    ]
    return mock_imputer


def create_mock_file_metadata(
    result_subpath: HivePath,
):
    month = str(result_subpath).split("/")[-1].split("=")[-1]
    if month == "2023-01":
        days_in_month = 31
    else:
        days_in_month = 28

    schema_mock = MagicMock()
    schema_mock.names = ["month", "grid_id", "date", "value_column_regex"]

    mock = MagicMock(spec=FileMetaData)
    mock.num_rows = days_in_month * VALID_COUNTRIES["india"]
    mock.schema.to_arrow_schema.return_value = schema_mock

    return mock


def calculate_expected_rows(month: str) -> int:
    month_as_arrow = Arrow.strptime(month, "%Y-%m")
    month_end = month_as_arrow.shift(months=1).replace(day=1)
    days_in_month = (month_end - month_as_arrow).days
    return days_in_month * VALID_COUNTRIES["india"]


def test__impute__all_months_available__processes_all_months(
    fake_data_with_missing,
    fake_data_result_map,
    mock_imputer_fills_missing,
):
    # Mock dependencies
    combined_storage_mock = MagicMock()

    combined_storage_mock.scan_stage.return_value = fake_data_with_missing.lazy()
    combined_storage_mock.does_dataset_exist.side_effect = lambda ds_name: False
    combined_storage_mock.read_dataframe_metadata.side_effect = (
        create_mock_file_metadata
    )

    # Instantiate the manager
    manager = SpatialImputationManager(
        combined_storage=combined_storage_mock,
        spatial_imputer=mock_imputer_fills_missing,
        temporal_config=TemporalConfig(
            start_date=Arrow(2023, 1, 1),
            end_date=Arrow(2023, 2, 1),
        ),
        input_data_artifact=INPUT_DATA_ARTIFACT,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
    )

    # Call the method under test
    manager.impute()

    # Assertions
    combined_storage_mock.scan_stage.assert_called_once_with("combined_monthly")
    assert_df_in_calls(
        combined_storage_mock.write_to_destination,
        fake_data_result_map["2023-01"].select(
            "grid_id", "date", pl.col("value_column_regex")
        ),
        HivePath.from_args(
            stage="era5_spatially_imputed",
            month="2023-01",
        ),
    )
    assert_df_in_calls(
        combined_storage_mock.write_to_destination,
        fake_data_result_map["2023-02"].select(
            "grid_id", "date", pl.col("value_column_regex")
        ),
        HivePath.from_args(
            stage="era5_spatially_imputed",
            month="2023-02",
        ),
    )
    assert combined_storage_mock.write_to_destination.call_count == 2


def test__impute__missing_months__raises_value_error(
    fake_data_with_missing,
    mock_imputer,
):
    # Mock dependencies
    combined_storage_mock = MagicMock()

    # Mock data
    months = [
        Arrow(2023, 1, 1),
        Arrow(2023, 2, 1),
        Arrow(2023, 3, 1),
    ]  # Add an extra month

    combined_storage_mock.scan_stage.return_value = fake_data_with_missing.lazy()
    combined_storage_mock.does_dataset_exist.side_effect = lambda ds_name: False

    # Instantiate the manager
    manager = SpatialImputationManager(
        combined_storage=combined_storage_mock,
        spatial_imputer=mock_imputer,
        temporal_config=TemporalConfig(
            start_date=Arrow(2023, 1, 1),
            end_date=Arrow(2023, 3, 1),
        ),
        input_data_artifact=INPUT_DATA_ARTIFACT,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
    )

    # Call the method under test and assert exception
    with pytest.raises(
        ValueError, match="The following months are not present in the dataset: 2023-03"
    ):
        manager.impute()


def test__impute__some_months_already_uploaded__skips_those_months(
    fake_data_with_missing,
    fake_data_result_map,
    mock_imputer_fills_missing,
):
    # Mock dependencies
    combined_storage_mock = MagicMock()

    # Mock data
    months = [Arrow(2023, 1, 1), Arrow(2023, 2, 1)]

    combined_storage_mock.scan_stage.return_value = fake_data_with_missing.lazy()
    combined_storage_mock.does_dataset_exist.side_effect = (
        lambda ds_name: ds_name
        == HivePath.from_args(stage="era5_spatially_imputed", month="2023-01")
    )
    combined_storage_mock.read_dataframe_metadata.side_effect = (
        create_mock_file_metadata
    )

    # Instantiate the manager
    manager = SpatialImputationManager(
        combined_storage=combined_storage_mock,
        spatial_imputer=mock_imputer_fills_missing,
        temporal_config=TemporalConfig(
            start_date=Arrow(2023, 1, 1),
            end_date=Arrow(2023, 2, 1),
        ),
        input_data_artifact=INPUT_DATA_ARTIFACT,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
    )

    # Call the method under test
    manager.impute()

    # Assertions
    combined_storage_mock.scan_stage.assert_called_once_with("combined_monthly")
    assert_df_in_calls(
        combined_storage_mock.write_to_destination,
        fake_data_result_map["2023-02"].select(
            "grid_id", "date", pl.col("value_column_regex")
        ),
        HivePath.from_args(
            stage="era5_spatially_imputed",
            month="2023-02",
        ),
    )
    assert combined_storage_mock.write_to_destination.call_count == 1


def assert_df_in_calls(mock, expected_df, expected_path):
    for calls in mock.call_args_list:
        actual_df, actual_path = calls[0]
        if actual_path == expected_path and actual_df.equals(expected_df):
            return
    raise AssertionError(
        f"Expected DataFrame not found in calls to write_to_destination. "
        f"Expected path: {expected_path}, DataFrame: {expected_df}"
    )
