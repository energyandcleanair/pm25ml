from polars import DataFrame
import pytest
from unittest.mock import Mock, MagicMock
from arrow import Arrow
from pm25ml.combiners.archive.combiner import ArchiveWideCombiner
from pm25ml.combiners.archive.combine_planner import CombinePlan
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager, MonthlyValidationError
from pm25ml.collectors.export_pipeline import PipelineConfig, ValueColumnType
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.hive_path import HivePath

from morefs.memory import MemFS

TEST_MONTHS = [Arrow(2023, 1, 1), Arrow(2023, 2, 1)]
N_GRID_CELLS = 33074
TEST_COUNTRY = "india"


ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME = "output_stage"
ARBITRARY_OUTPUT_ARTIFACT = DataArtifactRef(
    stage=ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME,
    country=TEST_COUNTRY,
)


@pytest.fixture
def in_memory_combined_storage():
    return CombinedStorage(
        filesystem=MemFS(),
        destination_bucket="test-bucket",
        profile_id=TEST_COUNTRY,
    )


@pytest.fixture
def mock_archived_wide_combiner():
    wide_combiner = MagicMock(
        spec=ArchiveWideCombiner,
    )
    wide_combiner.output_artifact = ARBITRARY_OUTPUT_ARTIFACT
    return wide_combiner


def create_valid_dataframe_for_month(month: Arrow, rows: int) -> DataFrame:
    """Create a valid DataFrame for a given month."""
    return DataFrame(
        {
            "date": [month.format("YYYY-MM-DD")] * rows,
            "grid_id": list(range(rows)),
            "test_dataset_static__value_static": [1.0] * rows,
            "test_dataset_monthly__value_monthly": [2.0] * rows,
            "test_dataset_yearly__value_year": [3.0] * rows,
        }
    )


@pytest.fixture
def mock_succeeding_archived_wide_combiner(in_memory_combined_storage, mock_archived_wide_combiner):
    mock_archived_wide_combiner.combine.side_effect = (
        lambda desc: in_memory_combined_storage.write_to_destination(
            create_valid_dataframe_for_month(desc.month_id, desc.expected_rows),
            result_subpath=(
                f"country={TEST_COUNTRY}/stage={ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME}/"
                f"month={desc.month_id}"
            ),
        )
    )


@pytest.fixture
def mock_missing_columns_archived_wide_combiner(
    in_memory_combined_storage, mock_archived_wide_combiner
):
    mock_archived_wide_combiner.combine.side_effect = (
        lambda desc: in_memory_combined_storage.write_to_destination(
            create_valid_dataframe_for_month(desc.month_id, desc.expected_rows).drop(
                "test_dataset_yearly__value_year"
            ),
            result_subpath=(
                f"country={TEST_COUNTRY}/stage={ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME}/"
                f"month={desc.month_id}"
            ),
        )
    )


@pytest.fixture
def mock_missing_rows_archived_wide_combiner(
    in_memory_combined_storage, mock_archived_wide_combiner
):
    mock_archived_wide_combiner.combine.side_effect = (
        lambda desc: in_memory_combined_storage.write_to_destination(
            create_valid_dataframe_for_month(desc.month_id, desc.expected_rows - 1),
            result_subpath=(
                f"country={TEST_COUNTRY}/stage={ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME}/"
                f"month={desc.month_id}"
            ),
        )
    )


@pytest.fixture
def mock_pipeline_config():
    return PipelineConfig(
        id_columns={"date", "grid_id"},
        value_column_type_map={"value1": ValueColumnType.FLOAT},
        result_subpath="country=india/dataset=test_dataset/month=2023-01",
        expected_rows=33074 * 31,
    )


@pytest.fixture
def monthly_combiner(in_memory_combined_storage, mock_archived_wide_combiner):
    return MonthlyCombinerManager(
        combined_storage=in_memory_combined_storage,
        archived_wide_combiner=mock_archived_wide_combiner,
    )


@pytest.fixture
def combine_defs():
    return [
        CombinePlan(
            month=Arrow(2023, 1, 1),
            paths={
                HivePath("country=india/dataset=test_dataset_static/type=static"),
                HivePath("country=india/dataset=test_dataset_monthly/month=2023-01"),
                HivePath("country=india/dataset=test_dataset_yearly/year=2023"),
            },
            expected_columns={
                "date",
                "grid_id",
                "test_dataset_static__value_static",
                "test_dataset_monthly__value_monthly",
                "test_dataset_yearly__value_year",
            },
            n_grid_cells=N_GRID_CELLS,
        ),
        CombinePlan(
            month=Arrow(2023, 2, 1),
            paths={
                HivePath("country=india/dataset=test_dataset_static/type=static"),
                HivePath("country=india/dataset=test_dataset_monthly/month=2023-02"),
                HivePath("country=india/dataset=test_dataset_yearly/year=2023"),
            },
            expected_columns={
                "date",
                "grid_id",
                "test_dataset_static__value_static",
                "test_dataset_monthly__value_monthly",
                "test_dataset_yearly__value_year",
            },
            n_grid_cells=N_GRID_CELLS,
        ),
    ]


@pytest.fixture
def mock_successful_result_dataframe_2023_01():
    return DataFrame(
        {
            "date": ["2023-01-01"] * (33074 * 31),
            "grid_id": list(range(33074 * 31)),
            "test_dataset_static__value_static": [1.0] * (33074 * 31),
            "test_dataset_monthly__value_monthly": [2.0] * (33074 * 31),
            "test_dataset_yearly__value_year": [3.0] * (33074 * 31),
        },
    )


@pytest.fixture
def mock_successful_result_dataframe_2023_02():
    return DataFrame(
        {
            "date": ["2023-02-01"] * (33074 * 28),
            "grid_id": list(range(33074 * 28)),
            "test_dataset_static__value_static": [1.0] * (33074 * 28),
            "test_dataset_monthly__value_monthly": [2.0] * (33074 * 28),
            "test_dataset_yearly__value_year": [3.0] * (33074 * 28),
        },
    )


@pytest.fixture
def mock_combined_storage_with_both_months(
    in_memory_combined_storage,
    mock_successful_result_dataframe_2023_01,
    mock_successful_result_dataframe_2023_02,
):
    in_memory_combined_storage.write_to_destination(
        mock_successful_result_dataframe_2023_01,
        result_subpath=(
            f"country={TEST_COUNTRY}/stage={ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME}/month=2023-01"
        ),
    )
    in_memory_combined_storage.write_to_destination(
        mock_successful_result_dataframe_2023_02,
        result_subpath=(
            f"country={TEST_COUNTRY}/stage={ARBITRARY_OUTPUT_ARTIFACT_OUTPUT_NAME}/month=2023-02"
        ),
    )


@pytest.mark.usefixtures(
    "mock_succeeding_archived_wide_combiner",
)
def test__combine_for_months__valid_input__combines_data(
    monthly_combiner,
    mock_archived_wide_combiner,
    combine_defs,
):
    monthly_combiner.combine_for_months(combine_defs)

    mock_archived_wide_combiner.combine.assert_any_call(
        combine_defs[0],
    )
    mock_archived_wide_combiner.combine.assert_any_call(
        combine_defs[1],
    )


@pytest.mark.usefixtures(
    "mock_combined_storage_with_both_months",
)
def test__combine_for_months__no_combining_needed__skips_combining(
    monthly_combiner,
    mock_archived_wide_combiner,
    combine_defs,
):
    monthly_combiner.combine_for_months(combine_defs)

    mock_archived_wide_combiner.combine.assert_not_called()


@pytest.mark.usefixtures(
    "mock_missing_columns_archived_wide_combiner",
)
def test__validate_combined__missing_columns__raises_error(monthly_combiner, combine_defs):
    with pytest.raises(MonthlyValidationError, match="Expected columns"):
        monthly_combiner.combine_for_months(combine_defs)


@pytest.mark.usefixtures(
    "mock_missing_rows_archived_wide_combiner",
)
def test__validate_combined__incorrect_row_count__raises_error(monthly_combiner, combine_defs):
    with pytest.raises(MonthlyValidationError, match="Expected 1025294 rows"):
        monthly_combiner.combine_for_months(combine_defs)
