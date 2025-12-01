"""Unit tests for ImputationSampler."""

import pytest
from polars import DataFrame
import polars as pl
from polars.testing import assert_frame_equal
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.sample.imputation_sampler import (
    SpatialTemporalImputationSampler,
    ImputationSamplerDefinition,
)
from pm25ml.setup.temporal_config import TemporalConfig
from morefs.memory import MemFS
from arrow import get

DESTINATION_BUCKET = "test_bucket"

RESULT_ARTIFACT_NAME = "result_stage"
ORIGIN_ARTIFACT_NAME = "origin_stage"
RESULT_ARTIFACT = DataArtifactRef(stage=RESULT_ARTIFACT_NAME)
ORIGIN_ARTIFACT = DataArtifactRef(stage=ORIGIN_ARTIFACT_NAME)


@pytest.fixture
def combined_storage():
    """Fixture for an in-memory filesystem."""
    fs = MemFS()
    return CombinedStorage(
        fs,
        destination_bucket=DESTINATION_BUCKET,
    )


@pytest.fixture
def temporal_config_one_month():
    """Fixture for temporal configuration."""
    return TemporalConfig(start_date=get("2023-01-01"), end_date=get("2023-01-31"))


@pytest.fixture
def temporal_config_two_months():
    """Fixture for temporal configuration with multiple months."""
    return TemporalConfig(start_date=get("2023-01-01"), end_date=get("2023-02-28"))


def test__imputation_sampler__process_month__correct_sampling(
    combined_storage, temporal_config_one_month
):
    """Test that ImputationSampler correctly samples data for a single month."""
    # Create sample data
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3, 4, 5, 6],
                "date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-02",
                ],
                "grid__id_50km": [1, 1, 1, 1, 1, 1],
                "col_1": [10.0, None, 30.0, 40.0, None, 60.0],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )

    # Create sampler
    sampler = SpatialTemporalImputationSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_one_month,
        imputation_sampler_definition=ImputationSamplerDefinition(
            value_column="col_1",
            model_name="mean",
            percentage_sample=0.5,
        ),
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )

    # Process a single month
    sampler.sample()

    # Read the sampled data
    sampled_data = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-01"
    )

    assert_frame_equal(
        sampled_data,
        DataFrame(
            {
                "grid_id": [1, 3, 4, 6],
                "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
                "grid__id_50km": [1, 1, 1, 1],
                "col_1": [10.0, 30.0, 40.0, 60.0],
                "split": [
                    "training",
                    "training",
                    "test",
                    "test",
                ],  # We've calculated this once based on the seeded sample.
            }
        ),
    )


def test__imputation_sampler__process_month_multiple_grids__correct_sampling(
    combined_storage, temporal_config_one_month
):
    """Test that ImputationSampler correctly samples data for a single month."""
    # Create sample data
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "date": ["2023-01-01"] * 9,
                "grid__id_50km": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "col_1": [None, 20.0, 30.0, 40.0, None, 60.0, 70.0, 80.0, None],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )

    # Create sampler
    sampler = SpatialTemporalImputationSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_one_month,
        imputation_sampler_definition=ImputationSamplerDefinition(
            value_column="col_1",
            model_name="mean",
            percentage_sample=0.5,
        ),
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )

    # Process a single month
    sampler.sample()

    # Read the sampled data
    sampled_data = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-01"
    )

    assert_frame_equal(
        sampled_data,
        DataFrame(
            {
                "grid_id": [2, 3, 4, 6, 7, 8],
                "date": ["2023-01-01"] * 6,
                "grid__id_50km": [2, 3, 1, 3, 1, 2],
                "col_1": [20.0, 30.0, 40.0, 60.0, 70.0, 80.0],
                "split": [
                    "training",
                    "training",
                    "training",
                    "test",
                    "test",
                    "test",
                ],  # We've calculated this once based on the seeded sample.
            }
        ),
    )


def test__imputation_sampler__process_month_multiple_months(
    combined_storage, temporal_config_two_months
):
    """Test that ImputationSampler correctly samples data for multiple months."""
    # Create sample data for two months
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3, 4, 5, 6],
                "date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                ],
                "grid__id_50km": [1, 1, 1, 2, 2, 2],
                "col_1": [10.0, None, 30.0, 40.0, None, 60.0],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-01",
    )
    combined_storage.write_to_destination(
        DataFrame(
            {
                "grid_id": [1, 2, 3, 4, 5, 6],
                "date": [
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-01",
                ],
                "grid__id_50km": [1, 1, 1, 2, 2, 2],
                "col_1": [10.0, None, 30.0, 40.0, None, 60.0],
            }
        ),
        f"stage={ORIGIN_ARTIFACT_NAME}/month=2023-02",
    )

    # Create sampler
    sampler = SpatialTemporalImputationSampler(
        combined_storage=combined_storage,
        temporal_config=temporal_config_two_months,
        imputation_sampler_definition=ImputationSamplerDefinition(
            value_column="col_1",
            model_name="mean",
            percentage_sample=0.5,
        ),
        input_data_artifact=ORIGIN_ARTIFACT,
        output_data_artifact=RESULT_ARTIFACT,
    )

    # Process the months
    sampler.sample()

    # Read the sampled data for January
    sampled_data_jan = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-01"
    )
    sampled_data_feb = combined_storage.read_dataframe(
        f"stage={RESULT_ARTIFACT_NAME}/month=2023-02"
    )

    assert sampled_data_jan.height == 4
    assert sampled_data_feb.height == 4

    assert sampled_data_jan.filter(pl.col("split") == "training").height == 2
    assert sampled_data_jan.filter(pl.col("split") == "test").height == 2

    assert sampled_data_feb.filter(pl.col("split") == "training").height == 2
    assert sampled_data_feb.filter(pl.col("split") == "test").height == 2
