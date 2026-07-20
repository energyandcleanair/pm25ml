import calendar
import pytest
from unittest.mock import MagicMock, ANY
from arrow import Arrow
import polars as pl
from polars.testing import assert_frame_equal
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.hive_path import HivePath
from morefs.memory import MemFS

from pm25ml.setup.date_params import TemporalConfig

TEST_COUNTRY = "india"

OUTPUT_DATA_ARTIFACT = DataArtifactRef(stage="recombined_stage", country=TEST_COUNTRY)
INPUT_STAGE_1_NAME = "stage1"
INPUT_STAGE_2_NAME = "stage2"
INPUT_STAGE_1_ARTIFACT = DataArtifactRef(stage=INPUT_STAGE_1_NAME, country=TEST_COUNTRY)
INPUT_STAGE_2_ARTIFACT = DataArtifactRef(stage=INPUT_STAGE_2_NAME, country=TEST_COUNTRY)


def _build_monthly_stage_df(
    month_id: str,
    *,
    n_grid_cells: int,
    value_column: str,
    value_multiplier: int,
    include_shared: bool = False,
    shared_offset: int = 0,
) -> pl.DataFrame:
    year_str, month_str = month_id.split("-")
    year = int(year_str)
    month = int(month_str)
    days_in_month = calendar.monthrange(year, month)[1]

    grid_ids = [
        grid_id for day in range(1, days_in_month + 1) for grid_id in range(1, n_grid_cells + 1)
    ]
    dates = [
        f"{month_id}-{day:02d}" for day in range(1, days_in_month + 1) for _ in range(n_grid_cells)
    ]
    values = [
        (grid_id * value_multiplier) + day
        for day in range(1, days_in_month + 1)
        for grid_id in range(1, n_grid_cells + 1)
    ]

    data = {
        "grid_id": grid_ids,
        "date": dates,
        value_column: values,
    }

    if include_shared:
        data["shared_column"] = [value + shared_offset for value in values]

    return pl.DataFrame(data)


def _expected_merged_month_df(month_id: str, *, n_grid_cells: int) -> pl.DataFrame:
    year_str, month_str = month_id.split("-")
    year = int(year_str)
    month = int(month_str)
    days_in_month = calendar.monthrange(year, month)[1]

    grid_ids = [
        grid_id for day in range(1, days_in_month + 1) for grid_id in range(1, n_grid_cells + 1)
    ]
    dates = [
        f"{month_id}-{day:02d}" for day in range(1, days_in_month + 1) for _ in range(n_grid_cells)
    ]
    value1 = [
        (grid_id * 100) + day
        for day in range(1, days_in_month + 1)
        for grid_id in range(1, n_grid_cells + 1)
    ]
    value2 = [
        (grid_id * 1000) + day
        for day in range(1, days_in_month + 1)
        for grid_id in range(1, n_grid_cells + 1)
    ]

    return pl.DataFrame(
        {
            "grid_id": grid_ids,
            "date": dates,
            "value1": value1,
            "value2": value2,
        }
    )


@pytest.fixture
def in_memory_combined_storage():
    return CombinedStorage(
        filesystem=MemFS(),
        destination_bucket="test-bucket",
        profile_id=TEST_COUNTRY,
    )


@pytest.fixture
def temporal_config():
    return TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )


@pytest.fixture
def recombiner(in_memory_combined_storage):
    return Recombiner(
        combined_storage=in_memory_combined_storage,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
        max_workers=4,
        n_grid_cells=3,
    )


@pytest.fixture
def mock_combined_storage_with_data(in_memory_combined_storage):
    def write_mock_data(stage, month, data):
        in_memory_combined_storage.write_to_destination(
            pl.DataFrame(data), f"country={TEST_COUNTRY}/stage={stage}/month={month}"
        )

    write_mock_data(
        "stage1",
        "2023-01",
        _build_monthly_stage_df(
            "2023-01", n_grid_cells=3, value_column="value1", value_multiplier=100
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage2",
        "2023-01",
        _build_monthly_stage_df(
            "2023-01", n_grid_cells=3, value_column="value2", value_multiplier=1000
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage1",
        "2023-02",
        _build_monthly_stage_df(
            "2023-02", n_grid_cells=3, value_column="value1", value_multiplier=100
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage2",
        "2023-02",
        _build_monthly_stage_df(
            "2023-02", n_grid_cells=3, value_column="value2", value_multiplier=1000
        ).to_dict(as_series=False),
    )


@pytest.fixture
def mock_combined_storage_with_overlapping_columns(in_memory_combined_storage):
    def write_mock_data(stage, month, data):
        in_memory_combined_storage.write_to_destination(
            pl.DataFrame(data), f"country={TEST_COUNTRY}/stage={stage}/month={month}"
        )

    write_mock_data(
        "stage1",
        "2023-01",
        _build_monthly_stage_df(
            "2023-01",
            n_grid_cells=3,
            value_column="value1",
            value_multiplier=100,
            include_shared=True,
            shared_offset=0,
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage2",
        "2023-01",
        _build_monthly_stage_df(
            "2023-01",
            n_grid_cells=3,
            value_column="value2",
            value_multiplier=1000,
            include_shared=True,
            shared_offset=10,
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage1",
        "2023-02",
        _build_monthly_stage_df(
            "2023-02",
            n_grid_cells=3,
            value_column="value1",
            value_multiplier=100,
            include_shared=True,
            shared_offset=0,
        ).to_dict(as_series=False),
    )
    write_mock_data(
        "stage2",
        "2023-02",
        _build_monthly_stage_df(
            "2023-02",
            n_grid_cells=3,
            value_column="value2",
            value_multiplier=1000,
            include_shared=True,
            shared_offset=10,
        ).to_dict(as_series=False),
    )


@pytest.mark.usefixtures("mock_combined_storage_with_data")
def test__recombine__valid_input__combines_data(
    recombiner, in_memory_combined_storage, temporal_config
):
    recombiner.recombine(
        [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT],
        temporal_config,
        overwrite_columns=True,
    )

    # Validate January 2023
    result_jan = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(country=TEST_COUNTRY, stage="recombined_stage", month="2023-01")
    )
    assert result_jan.shape[0] == 93
    assert set(result_jan.columns) == {"grid_id", "date", "value1", "value2"}
    assert_frame_equal(
        result_jan.sort(by=["grid_id", "date"]),
        _expected_merged_month_df("2023-01", n_grid_cells=3).sort(by=["grid_id", "date"]),
        check_column_order=False,
    )

    # Validate February 2023
    result_feb = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(country=TEST_COUNTRY, stage="recombined_stage", month="2023-02")
    )
    assert result_feb.shape[0] == 84
    assert set(result_feb.columns) == {"grid_id", "date", "value1", "value2"}
    assert_frame_equal(
        result_feb.sort(by=["grid_id", "date"]),
        _expected_merged_month_df("2023-02", n_grid_cells=3).sort(by=["grid_id", "date"]),
        check_column_order=False,
    )


@pytest.mark.usefixtures("mock_combined_storage_with_overlapping_columns")
def test__recombine__shared_columns_no_overwrite__raises_error(recombiner, temporal_config):
    with pytest.raises(ValueError, match="Shared columns detected"):
        recombiner.recombine(
            [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT],
            temporal_config,
            overwrite_columns=False,
        )


@pytest.mark.usefixtures("mock_combined_storage_with_data")
def test__recombine__existing_dataset_correct_columns__only_updates_one_month(
    recombiner, in_memory_combined_storage, temporal_config
):
    # Pre-existing dataset with correct columns
    existing_jan = _build_monthly_stage_df(
        "2023-01", n_grid_cells=3, value_column="value1", value_multiplier=100
    ).join(
        _build_monthly_stage_df(
            "2023-01", n_grid_cells=3, value_column="value2", value_multiplier=1000
        ),
        on=["grid_id", "date"],
        how="inner",
    )

    in_memory_combined_storage.write_to_destination(
        existing_jan,
        HivePath.from_args(
            country=TEST_COUNTRY,
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
        [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT],
        temporal_config,
        overwrite_columns=False,
    )

    # Validate that the dataset remains unchanged
    result = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(country=TEST_COUNTRY, stage="recombined_stage", month="2023-01")
    )
    assert result.shape[0] == 93
    assert set(result.columns) == {"grid_id", "date", "value1", "value2"}
    assert_frame_equal(
        result.sort(by=["grid_id", "date"]),
        existing_jan.sort(by=["grid_id", "date"]),
        check_column_order=False,
    )

    # Ensure write_to_destination was not called again
    assert in_memory_combined_storage.write_to_destination.call_count == 1
    in_memory_combined_storage.write_to_destination.assert_called_with(
        ANY,
        HivePath.from_args(
            country=TEST_COUNTRY,
            stage="recombined_stage",
            month="2023-02",
        ),
    )


@pytest.mark.usefixtures("mock_combined_storage_with_data")
def test__recombine__when_grid_size_validation_enabled_and_row_count_mismatch__raises_error(
    in_memory_combined_storage,
):
    strict_recombiner = Recombiner(
        combined_storage=in_memory_combined_storage,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
        max_workers=1,
        n_grid_cells=1,
    )

    with pytest.raises(
        ValueError,
        match="Expected 31 rows in the recombined result",
    ):
        strict_recombiner.recombine(
            [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT],
            TemporalConfig(
                start_date=Arrow(2023, 1, 1),
                end_date=Arrow(2023, 1, 1),
            ),
            overwrite_columns=True,
        )


def test__recombine__when_grid_size_validation_enabled_and_row_count_matches__passes(
    in_memory_combined_storage,
):
    month = "2023-01"

    stage1_df = _build_monthly_stage_df(
        month, n_grid_cells=2, value_column="value1", value_multiplier=100
    )
    stage2_df = _build_monthly_stage_df(
        month, n_grid_cells=2, value_column="value2", value_multiplier=1000
    )

    in_memory_combined_storage.write_to_destination(
        stage1_df,
        HivePath.from_args(country=TEST_COUNTRY, stage="stage1", month=month),
    )
    in_memory_combined_storage.write_to_destination(
        stage2_df,
        HivePath.from_args(country=TEST_COUNTRY, stage="stage2", month=month),
    )

    strict_recombiner = Recombiner(
        combined_storage=in_memory_combined_storage,
        output_data_artifact=OUTPUT_DATA_ARTIFACT,
        max_workers=1,
        n_grid_cells=2,
    )

    strict_recombiner.recombine(
        [INPUT_STAGE_1_ARTIFACT, INPUT_STAGE_2_ARTIFACT],
        TemporalConfig(
            start_date=Arrow(2023, 1, 1),
            end_date=Arrow(2023, 1, 1),
        ),
        overwrite_columns=True,
    )

    result = in_memory_combined_storage.read_dataframe(
        HivePath.from_args(country=TEST_COUNTRY, stage="recombined_stage", month=month),
    )
    assert result.shape[0] == 62
    assert_frame_equal(
        result.sort(by=["grid_id", "date"]),
        _expected_merged_month_df(month, n_grid_cells=2).sort(by=["grid_id", "date"]),
        check_column_order=False,
    )
