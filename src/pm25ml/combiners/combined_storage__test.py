import os
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
import pytest
from morefs.memory import MemFS

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.hive_path import HivePath

DESTINATION_BUCKET = "destination_bucket"
TEST_COUNTRY = "india"


@pytest.fixture
def example_table():
    data = {
        "month": ["2023-01", "2023-01", "2023-02"],
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    }
    return pl.DataFrame(data)


@pytest.fixture
def in_memory_filesystem():
    return MemFS()


def test__write_to_destination__valid_input__writes_parquet(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    storage.write_to_destination(example_table, "result_path")

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "data.parquet")
    assert in_memory_filesystem.exists(result_path)


def test__write_to_destination__hivepath_input__writes_parquet(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Use a HivePath format
    hive_path = HivePath.from_args(stage="result_path", test="test")

    # Write the table to the destination using the HivePath
    storage.write_to_destination(example_table, hive_path)

    # Validate that the Parquet file was written to the temp directory
    result_path = os.path.join(DESTINATION_BUCKET, "stage=result_path", "test=test", "data.parquet")
    assert in_memory_filesystem.exists(result_path)


def test__read_dataframe__valid_input__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Read the DataFrame
    dataframe = storage.read_dataframe("result_path")

    # Validate the DataFrame
    assert_frame_equal(dataframe, example_table)


def test__read_dataframe__hivepath_input__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )
    # Use a HivePath format
    hive_path = HivePath.from_args(
        stage="result_path",
    )

    # Write the table to the destination
    storage.write_to_destination(example_table, hive_path)

    # Read the DataFrame using the HivePath
    dataframe = storage.read_dataframe(hive_path)

    # Validate the DataFrame
    assert_frame_equal(dataframe, example_table)


def test__read_dataframe__file_named_0_parquet__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Write the table to a file named 0.parquet
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "0.parquet")
    with in_memory_filesystem.open(result_path, "wb") as file:
        example_table.write_parquet(file)

    # Read the DataFrame
    dataframe = storage.read_dataframe("result_path")

    # Validate the DataFrame
    assert_frame_equal(dataframe, example_table)


def test__read_dataframe__file_named_data_parquet__returns_dataframe(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Write the table to a file named data.parquet
    result_path = os.path.join(DESTINATION_BUCKET, "result_path", "data.parquet")
    with in_memory_filesystem.open(result_path, "wb") as file:
        example_table.write_parquet(file)

    # Read the DataFrame
    dataframe = storage.read_dataframe("result_path")

    # Validate the DataFrame
    assert_frame_equal(dataframe, example_table)


def test__scan_stage__valid_stage__returns_lazyframe(in_memory_filesystem, example_table) -> None:
    with patch(
        "polars.scan_parquet",
    ) as mock_scan:
        storage = CombinedStorage(
            filesystem=in_memory_filesystem,
            destination_bucket=DESTINATION_BUCKET,
            profile_id=TEST_COUNTRY,
        )

        # Mock the scan_parquet method to return a LazyFrame
        mock_lazy_frame = pl.LazyFrame(example_table)
        mock_scan.return_value = mock_lazy_frame

        lazy_frame = storage.scan_stage("valid_stage")

        mock_scan.assert_called_once_with(
            f"gs://{DESTINATION_BUCKET}/country={TEST_COUNTRY}/stage=valid_stage/",
            hive_partitioning=True,
        )

        # Validate that the returned object is a LazyFrame
        assert lazy_frame is mock_lazy_frame


def test__scan_stage__profile_storage__includes_country_partition(
    in_memory_filesystem, example_table
) -> None:
    with patch(
        "polars.scan_parquet",
    ) as mock_scan:
        storage = CombinedStorage(
            filesystem=in_memory_filesystem,
            destination_bucket=DESTINATION_BUCKET,
            profile_id="in+bd",
        )

        mock_lazy_frame = pl.LazyFrame(example_table)
        mock_scan.return_value = mock_lazy_frame

        storage.scan_stage("valid_stage")

        mock_scan.assert_called_once_with(
            f"gs://{DESTINATION_BUCKET}/country=in+bd/stage=valid_stage/",
            hive_partitioning=True,
        )


def test__does_dataset_exist__dataset_written__returns_true(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Initially, the dataset should not exist
    assert not storage.does_dataset_exist("result_path")

    # Write the table to the destination
    storage.write_to_destination(example_table, "result_path")

    # Now, the dataset should exist
    assert storage.does_dataset_exist("result_path")


def test__does_dataset_exist__dataset_not_written__returns_false(in_memory_filesystem) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Ensure the dataset does not exist
    assert not storage.does_dataset_exist("non_existent_path")


def test__does_dataset_exist__hivepath_input__returns_true(
    in_memory_filesystem, example_table
) -> None:
    storage = CombinedStorage(
        filesystem=in_memory_filesystem,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )

    # Use a HivePath format
    hive_path = HivePath.from_args(
        stage="result_path",
    )

    # Initially, the dataset should not exist
    assert not storage.does_dataset_exist(hive_path)

    # Write the table to the destination using the HivePath
    storage.write_to_destination(example_table, hive_path)

    # Now, the dataset should exist
    assert storage.does_dataset_exist(hive_path)
