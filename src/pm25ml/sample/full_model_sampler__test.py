"""Unit tests for FullModelSampler."""

from arrow import get
import pytest
import polars as pl
from polars import DataFrame
from polars.testing import assert_frame_equal

from morefs.memory import MemFS

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.sample.full_model_sampler import FullModelSampler
from pm25ml.setup.temporal_config import TemporalConfig


DESTINATION_BUCKET = "test_bucket"

ORIGIN_ARTIFACT_NAME = "origin_stage"
RESULT_ARTIFACT_NAME = "result_stage"
ORIGIN_ARTIFACT = DataArtifactRef(stage=ORIGIN_ARTIFACT_NAME)
RESULT_ARTIFACT = DataArtifactRef(stage=RESULT_ARTIFACT_NAME)


@pytest.fixture
def combined_storage():
    """In-memory CombinedStorage backed by MemFS."""
    fs = MemFS()
    return CombinedStorage(filesystem=fs, destination_bucket=DESTINATION_BUCKET)


@pytest.fixture
def temporal_config_one_month():
    return TemporalConfig(start_date=get("2023-01-01"), end_date=get("2023-01-31"))


@pytest.fixture
def temporal_config_two_months():
    return TemporalConfig(start_date=get("2023-01-01"), end_date=get("2023-02-28"))


def test__full_model_sampler__single_month__filters_nulls(
    combined_storage, temporal_config_one_month
):
    """It should write only non-null rows for the column for a single month."""
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3, 4],
                "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
                "col_1": [10.0, None, 30.0, None],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )

    sampler = FullModelSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_one_month,
        column_name="col_1",
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )

    sampler.sample()

    result = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-01"
    )
    assert_frame_equal(
        result.sort(["grid_id", "date"]),
        DataFrame(
            {
                "grid_id": [1, 3],
                "date": ["2023-01-01", "2023-01-02"],
                "col_1": [10.0, 30.0],
            }
        ).sort(["grid_id", "date"]),
    )


def test__full_model_sampler__multiple_months__processes_all(
    combined_storage, temporal_config_two_months
):
    """It should process every month in the temporal config via the thread pool."""
    # Month 1
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3],
                "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
                "col_1": [10.0, None, 30.0],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )
    # Month 2
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [4, 5, 6],
                "date": ["2023-02-01", "2023-02-01", "2023-02-02"],
                "col_1": [None, 50.0, 60.0],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-02",
    )

    sampler = FullModelSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_two_months,
        column_name="col_1",
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )
    sampler.sample()

    jan = combined_storage.read_dataframe(f"stage={RESULT_ARTIFACT_NAME}/month=2023-01")
    feb = combined_storage.read_dataframe(f"stage={RESULT_ARTIFACT_NAME}/month=2023-02")

    assert jan.height == 2  # 10.0 & 30.0
    assert feb.height == 2  # 50.0 & 60.0
    assert set(jan["col_1"].to_list()) == {10.0, 30.0}
    assert set(feb["col_1"].to_list()) == {50.0, 60.0}


def test__full_model_sampler__all_null_column__writes_empty_dataset(
    combined_storage, temporal_config_one_month
):
    """If the target column is entirely null, an empty dataset should be written (0 rows)."""
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2],
                "date": ["2023-01-01", "2023-01-02"],
                "col_1": [None, None],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )

    sampler = FullModelSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_one_month,
        column_name="col_1",
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )
    sampler.sample()

    result = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-01"
    )
    assert result.height == 0
    # Ensure schema preserved (same columns present)
    assert result.columns == ["grid_id", "date", "col_1"]
