import pytest
from unittest.mock import MagicMock, ANY
from arrow import Arrow
import polars as pl
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.hive_path import HivePath
from morefs.memory import MemFS

from pm25ml.setup.temporal_config import TemporalConfig

OUTPUT_DATA_ARTIFACT = DataArtifactRef(stage="recombined_stage")
INPUT_STAGE_1_NAME = "stage1"
INPUT_STAGE_2_NAME = "stage2"
INPUT_STAGE_1_ARTIFACT = DataArtifactRef(stage=INPUT_STAGE_1_NAME)
INPUT_STAGE_2_ARTIFACT = DataArtifactRef(stage=INPUT_STAGE_2_NAME)


@pytest.fixture
def in_memory_combined_storage():
    return CombinedStorage(
        filesystem=MemFS(),
        destination_bucket="test-bucket",
    )


@pytest.fixture
def recombiner(in_memory_combined_storage):
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    return Recombiner(
        combined_storage=in_memory_combined_storage,
        temporal_config=temporal_config,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
        max_workers=4,
    )


@pytest.fixture
def mock_combined_storage_with_data(in_memory_combined_storage):
    def write_mock_data(stage, month, data):
        in_memory_combined_storage.write_to_destination(
            pl.DataFrame(data), f"stage={stage}/month={month}"
        )

    # Mock data for January 2023
    write_mock_data(
        "stage1",
        "2023-01",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value1": [10, 20, 30],
        },
    )
    write_mock_data(
        "stage2",
        "2023-01",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value2": [100, 200, 300],
        },
    )

    # Mock data for February 2023
    write_mock_data(
        "stage1",
        "2023-02",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-02-01", "2023-02-02", "2023-02-03"],
            "value1": [40, 50, 60],
        },
    )
    write_mock_data(
        "stage2",
        "2023-02",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-02-01", "2023-02-02", "2023-02-03"],
            "value2": [400, 500, 600],
        },
    )


@pytest.fixture
def mock_combined_storage_with_overlapping_columns(in_memory_combined_storage):
    def write_mock_data(stage, month, data):
        in_memory_combined_storage.write_to_destination(
            pl.DataFrame(data), f"stage={stage}/month={month}"
        )

    # Mock data for January 2023 with overlapping columns
    write_mock_data(
        "stage1",
        "2023-01",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value1": [10, 20, 30],
            "shared_column": [1, 2, 3],
        },
    )
    write_mock_data(
        "stage2",
        "2023-01",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value2": [100, 200, 300],
            "shared_column": [4, 5, 6],
        },
    )

    # Mock data for February 2023 with overlapping columns
    write_mock_data(
        "stage1",
        "2023-02",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-02-01", "2023-02-02", "2023-02-03"],
            "value1": [40, 50, 60],
            "shared_column": [1, 2, 3],
        },
    )
    write_mock_data(
        "stage2",
        "2023-02",
        {
            "grid_id": [1, 2, 3],
            "date": ["2023-02-01", "2023-02-02  ", "2023-02-03"],
            "value2": [400, 500, 600],
            "shared_column": [7, 8, 9],
        },
    )


@pytest.mark.usefixtures("mock_combined_storage_with_data")
def test__recombine__valid_input__combines_data(recombiner, in_memory_combined_storage):
    recombiner.recombine(
        [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT], overwrite_columns=True
    )

    # Validate January 2023
    result_jan = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(stage="recombined_stage", month="2023-01")
    )
    assert set(result_jan.columns) == {"grid_id", "date", "value1", "value2"}
    assert result_jan["value1"].to_list() == [10, 20, 30]
    assert result_jan["value2"].to_list() == [100, 200, 300]

    # Validate February 2023
    result_feb = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(stage="recombined_stage", month="2023-02")
    )
    assert set(result_feb.columns) == {"grid_id", "date", "value1", "value2"}
    assert result_feb["value1"].to_list() == [40, 50, 60]
    assert result_feb["value2"].to_list() == [400, 500, 600]


@pytest.mark.usefixtures("mock_combined_storage_with_overlapping_columns")
def test__recombine__shared_columns_no_overwrite__raises_error(recombiner):
    with pytest.raises(ValueError, match="Shared columns detected"):
        recombiner.recombine(
            [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT], overwrite_columns=False
        )


@pytest.mark.usefixtures("mock_combined_storage_with_data")
def test__recombine__existing_dataset_correct_columns__only_updates_one_month(
    recombiner, in_memory_combined_storage
):
    # Pre-existing dataset with correct columns
    in_memory_combined_storage.write_to_destination(
        pl.DataFrame(
            {
                "grid_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value1": [10, 20, 30],
                "value2": [100, 200, 300],
            }
        ),
        HivePath.from_args(
            stage="recombined_stage",
            month="2023-01",
        ),
    )

    # Mock the write_to_destination method to track calls
    in_memory_combined_storage.write_to_destination = MagicMock(
        wraps=in_memory_combined_storage.write_to_destination
    )

    # Run recombine
    recombiner.recombine(
        [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT], overwrite_columns=False
    )

    # Validate that the dataset remains unchanged
    result = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(stage="recombined_stage", month="2023-01")
    )
    assert set(result.columns) == {"grid_id", "date", "value1", "value2"}
    assert result["value1"].to_list() == [10, 20, 30]
    assert result["value2"].to_list() == [100, 200, 300]

    # Ensure write_to_destination was not called again
    assert in_memory_combined_storage.write_to_destination.call_count == 1
    in_memory_combined_storage.write_to_destination.assert_called_with(
        ANY,
        HivePath.from_args(
            stage="recombined_stage",
            month="2023-02",
        ),
    )
