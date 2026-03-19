"""Unit tests for ArchiveWideCombiner."""

from unittest.mock import MagicMock
from arrow import get
import pytest
from polars import DataFrame
from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig
from pm25ml.combiners.archive.combine_planner import CombinePlan
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.archive.combiner import ArchiveWideCombiner
from morefs.memory import MemFS

from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.hive_path import HivePath

DESTINATION_BUCKET = "test_bucket"
N_GRID_CELLS = 33074
TEST_COUNTRY = "india"


@pytest.fixture
def combined_storage():
    """Fixture for an in-memory filesystem."""
    fs = MemFS()
    return CombinedStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
        profile_id=TEST_COUNTRY,
    )


@pytest.fixture
def archive_storage__no_files():
    """Fixture for an empty archive storage."""
    fs = MemFS()
    return IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )


@pytest.fixture
def archive_storage__with_all_matching_types_and_extra():
    """Fixture for archive storage with multiple files and dataset key."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "date": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                ],
                "col_1": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 5.0, 10.0, 15.0],
            }
        ),
        "dataset=dataset_1/month=2023-01",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "col_2": [40.0, 50.0, 60.0],
            }
        ),
        "dataset=dataset_2/year=2023",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "col_3": [70.0, 80.0, 90.0],
            }
        ),
        "dataset=dataset_3/type=static",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [10, 11, 12],
                "date": ["2023-01-04", "2023-01-05", "2023-01-06"],
                "col_4": [100.0, 110.0, 120.0],
            }
        ),
        "dataset=excluded_dataset/type=different",
    )

    return storage


@pytest.fixture
def archive_storage__with_odd_number_of_datasets():
    """Fixture for archive storage with multiple files and dataset key."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "date": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                ],
                "col_1": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 5.0, 10.0, 15.0],
            }
        ),
        "dataset=dataset_1/month=2023-01",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "col_2": [40.0, 50.0, 60.0],
            }
        ),
        "dataset=dataset_2/year=2023",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "col_3": [70.0, 80.0, 90.0],
            }
        ),
        "dataset=dataset_3/type=static",
    )

    return storage


@pytest.fixture
def archive_storage__no_matching_merge():
    """Fixture for archive storage with no matching merge."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "col_1": [10.0, 20.0, 30.0],
            }
        ),
        "dataset=partial_1/month=2023-01",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [4, 5, 6],
                "col_2": [40.0, 50.0, 60.0],
            }
        ),
        "dataset=partial_2/year=2023",
    )

    return storage


@pytest.fixture
def archive_storage__with_date_and_time_for_one():
    """Fixture for archive storage with date and time for one dataset."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 1, 1],
                "date": ["2023-01-01T00:00:00", "2023-01-01T01:00:00", "2023-01-01T02:00:00"],
                "col_1": [10.0, 11.0, 12.0],
            }
        ),
        "dataset=with_time/month=2023-01",
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 1, 1],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "col_2": [40.0, 41.0, 42.0],
            }
        ),
        "dataset=without_time/year=2023",
    )

    return storage


@pytest.fixture
def archive_storage__with_dataset_key():
    """Fixture for archive storage with dataset key in metadata."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 1, 1],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "col_1": [10.0, 20.0, 30.0],
            }
        ),
        "dataset=test_dataset/month=2023-01",
    )

    return storage


@pytest.fixture
def archive_storage__without_dataset_key():
    """Fixture for archive storage with a dataset missing the dataset key in the path."""
    fs = MemFS()

    storage = IngestArchiveStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )

    storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "col_1": [10.0, 20.0, 30.0],
            }
        ),
        "month=2023-01",
    )

    return storage


def create_mock_export_pipeline_with_path(path: str) -> ExportPipeline:
    pipeline_mock = MagicMock(spec=ExportPipeline)
    pipeline_config_mock = MagicMock(spec=PipelineConfig)
    pipeline_config_mock.hive_path = HivePath(path)
    pipeline_mock.get_config_metadata.return_value = pipeline_config_mock
    return pipeline_mock


def test__combine__no_files__raises_error(
    archive_storage__no_files: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test that combining with no files raises an error."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__no_files,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    processors = []  # No processors provided, so no data to combine

    combine_def = CombinePlan(
        month=get("2023-01"),
        paths=set(),
        expected_columns={"grid_id", "date", "col_1"},
        n_grid_cells=N_GRID_CELLS,
    )

    with pytest.raises(ValueError, match="No data found for month '2023-01'..*"):
        combiner.combine(combine_def)


def test__combine__with_all_matching_types_and_extra_correct_result__successfully_merges(
    archive_storage__with_all_matching_types_and_extra: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test combining with multiple files."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__with_all_matching_types_and_extra,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    combine_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("dataset=dataset_1/month=2023-01"),
            HivePath("dataset=dataset_2/year=2023"),
            HivePath("dataset=dataset_3/type=static"),
        },
        expected_columns={
            "grid_id",
            "date",
            "dataset_1__col_1",
            "dataset_2__col_2",
            "dataset_3__col_3",
        },
        n_grid_cells=N_GRID_CELLS,
    )

    # Combine for January 2023
    combiner.combine(combine_def)

    # Read the combined data
    combined_data = combined_storage.read_dataframe(
        f"country={TEST_COUNTRY}/stage=combined_monthly/month=2023-01"
    )

    # Validate the combined data
    assert combined_data.height == 9
    assert "grid_id" in combined_data.columns
    assert "date" in combined_data.columns
    assert "dataset_1__col_1" in combined_data.columns
    assert "dataset_2__col_2" in combined_data.columns
    assert "dataset_3__col_3" in combined_data.columns


def test__combine__with_odd_number__successfully_merges(
    archive_storage__with_odd_number_of_datasets: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test combining with multiple files."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__with_odd_number_of_datasets,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    combined_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("dataset=dataset_1/month=2023-01"),
            HivePath("dataset=dataset_2/year=2023"),
            HivePath("dataset=dataset_3/type=static"),
        },
        expected_columns={
            "grid_id",
            "date",
            "dataset_1__col_1",
            "dataset_2__col_2",
            "dataset_3__col_3",
        },
        n_grid_cells=N_GRID_CELLS,
    )

    # Combine for January 2023
    combiner.combine(combined_def)

    # Read the combined data
    combined_data = combined_storage.read_dataframe(
        f"country={TEST_COUNTRY}/stage=combined_monthly/month=2023-01"
    )

    # Validate the combined data
    assert combined_data.height == 9
    assert "grid_id" in combined_data.columns
    assert "date" in combined_data.columns
    assert "dataset_1__col_1" in combined_data.columns
    assert "dataset_2__col_2" in combined_data.columns
    assert "dataset_3__col_3" in combined_data.columns


def test__combine__no_matching_merge__empty_dataset(
    archive_storage__no_matching_merge: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test combining with no matching merge."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__no_matching_merge,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    combined_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("dataset=partial_1/month=2023-01"),
            HivePath("dataset=partial_2/year=2023"),
        },
        expected_columns={
            "grid_id",
            "date",
            "partial_1__col_1",
            "partial_2__col_2",
        },
        n_grid_cells=N_GRID_CELLS,
    )

    combiner.combine(combined_def)

    combined_data = combined_storage.read_dataframe(
        f"country={TEST_COUNTRY}/stage=combined_monthly/month=2023-01"
    )

    assert combined_data.height == 0
    assert "grid_id" in combined_data.columns
    assert "date" in combined_data.columns
    assert "partial_1__col_1" in combined_data.columns
    assert "partial_2__col_2" in combined_data.columns


def test__combine__with_date_and_time_for_one__successfully_merges(
    archive_storage__with_date_and_time_for_one: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test combining with date and time for one dataset."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__with_date_and_time_for_one,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )
    combined_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("dataset=with_time/month=2023-01"),
            HivePath("dataset=without_time/year=2023"),
        },
        expected_columns={
            "grid_id",
            "date",
            "with_time__col_1",
            "without_time__col_2",
        },
        n_grid_cells=N_GRID_CELLS,
    )

    combiner.combine(combined_def)

    combined_data = combined_storage.read_dataframe(
        f"country={TEST_COUNTRY}/stage=combined_monthly/month=2023-01"
    )

    assert combined_data.height == 3
    assert "grid_id" in combined_data.columns
    assert "date" in combined_data.columns
    assert "with_time__col_1" in combined_data.columns
    assert "without_time__col_2" in combined_data.columns


def test__combine__renaming_columns__successfully_renames(
    archive_storage__with_all_matching_types_and_extra: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test that columns are renamed correctly."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__with_all_matching_types_and_extra,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    combined_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("dataset=dataset_1/month=2023-01"),
            HivePath("dataset=dataset_2/year=2023"),
            HivePath("dataset=dataset_3/type=static"),
        },
        expected_columns={
            "grid_id",
            "date",
            "dataset_1__col_1",
            "dataset_2__col_2",
            "dataset_3__col_3",
        },
        n_grid_cells=N_GRID_CELLS,
    )

    combiner.combine(combined_def)

    combined_data = combined_storage.read_dataframe(
        f"country={TEST_COUNTRY}/stage=combined_monthly/month=2023-01"
    )

    # Validate renamed columns
    assert "dataset_1__col_1" in combined_data.columns
    assert "dataset_2__col_2" in combined_data.columns
    assert "dataset_3__col_3" in combined_data.columns


def test__combine__without_dataset_key__raises_error(
    archive_storage__without_dataset_key: IngestArchiveStorage,
    combined_storage: CombinedStorage,
) -> None:
    """Test combining when a dataset is missing the dataset key in the path."""
    combiner = ArchiveWideCombiner(
        archive_storage=archive_storage__without_dataset_key,
        combined_storage=combined_storage,
        output_artifact=DataArtifactRef(
            stage="combined_monthly",
            country=TEST_COUNTRY,
        ),
    )

    combined_def = CombinePlan(
        month=get("2023-01"),
        paths={
            HivePath("month=2023-01"),
        },
        expected_columns={"grid_id", "date", "col_1"},
        n_grid_cells=N_GRID_CELLS,
    )

    with pytest.raises(
        ValueError, match="Expected 'dataset' key in month=2023-01, but it is missing."
    ):
        combiner.combine(combined_def)
