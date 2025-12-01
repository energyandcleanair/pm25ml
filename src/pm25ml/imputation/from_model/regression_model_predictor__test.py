import pytest
import polars as pl
import numpy as np
from unittest.mock import MagicMock, Mock
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.imputation.from_model.regression_model_predictor import (
    RegressionModelPredictor,
)
from pm25ml.model_reference import ImputationModelReference
from pm25ml.training.model_storage import LoadedValidatedModel
from pm25ml.setup.temporal_config import TemporalConfig
from pm25ml.combiners.combined_storage import CombinedStorage
from morefs.memory import MemFS
from polars.testing import assert_frame_equal

ARBITRARY_MODEL_NAME = "aod"

ARBRITARY_INPUT_ARTIFACT = DataArtifactRef(stage="input_stage")

OUTPUT_ARTIFACT_STAGE = DataArtifactRef(stage=f"imputed+{ARBITRARY_MODEL_NAME}")


@pytest.fixture
def mock_model_reference():
    return ImputationModelReference(
        model_name=ARBITRARY_MODEL_NAME,  # Use a valid model_name literal
        predictor_cols=["feature1", "feature2"],
        target_col="target_col",
        grouper_col="group_col",
        model_builder=Mock(),
        extra_sampler=Mock(),
        min_r2_score=0.7,
        max_r2_score=0.9,
    )


@pytest.fixture
def mock_loaded_validated_model():
    model = MagicMock(spec=LoadedValidatedModel)
    model.cv_results = {"test_r2": np.array([0.8, 0.85, 0.9])}
    mock_predictor = MagicMock()
    mock_predictor.predict = MagicMock(
        return_value=np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    )
    model.model = mock_predictor
    return model


@pytest.fixture
def mock_temporal_config():
    temporal_config = MagicMock(spec=TemporalConfig)
    temporal_config.months = ["2023-01", "2023-02"]
    return temporal_config


@pytest.fixture
def combined_storage_with_data(tmp_path):
    fs = MemFS()
    storage = CombinedStorage(
        filesystem=fs,
        destination_bucket=str(tmp_path),
    )

    # Prepare the required files using CombinedStorage's write_to_destination
    for month in ["2023-01", "2023-02"]:
        result_subpath = ARBRITARY_INPUT_ARTIFACT.for_month(month)
        storage.write_to_destination(
            table=pl.DataFrame(
                {
                    "grid_id": [1, 2, 1, 2, 1, 2],
                    "date": [
                        f"{month}-01",
                        f"{month}-01",
                        f"{month}-02",
                        f"{month}-02",
                        f"{month}-03",
                        f"{month}-03",
                    ],
                    "target_col": [None, None, 1.0, 1.0, None, 2.0],
                    # Unused but needed.
                    "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "feature2": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                }
            ),
            result_subpath=result_subpath,
        )

    return storage


@pytest.fixture
def regression_model_imputer_with_data(
    mock_model_reference,
    mock_loaded_validated_model,
    mock_temporal_config,
    combined_storage_with_data,
):
    return RegressionModelPredictor(
        model_ref=mock_model_reference,
        model=mock_loaded_validated_model,
        temporal_config=mock_temporal_config,
        combined_storage=combined_storage_with_data,
        input_data_artifact=ARBRITARY_INPUT_ARTIFACT,
        output_data_artifact=OUTPUT_ARTIFACT_STAGE,
    )


def test__impute__raises_error_on_low_score(
    regression_model_imputer_with_data, mock_model_reference
):
    # Adjust the model's R2 score range to trigger validation logic
    mock_model_reference.min_r2_score = 0.9
    mock_model_reference.max_r2_score = 1.0

    # Expect an error due to low average CV score
    with pytest.raises(ValueError, match="too low"):
        regression_model_imputer_with_data.predict()


def test__impute__raises_error_on_high_score(
    regression_model_imputer_with_data, mock_model_reference
):
    # Adjust the range to trigger high score validation
    mock_model_reference.min_r2_score = 0.7
    mock_model_reference.max_r2_score = 0.8

    # Expect an error due to high average CV score
    with pytest.raises(ValueError, match="unusually high"):
        regression_model_imputer_with_data.predict()


def test__impute__passes_on_valid_score(
    regression_model_imputer_with_data, mock_model_reference
):
    # Set a valid range and ensure no exceptions are raised
    mock_model_reference.min_r2_score = 0.7
    mock_model_reference.max_r2_score = 0.9
    regression_model_imputer_with_data.predict()


def test__impute__writes_results_correctly_with_data(
    regression_model_imputer_with_data,
):
    regression_model_imputer_with_data.predict()

    # Verify the results are written by reading them using CombinedStorage
    combined_storage = regression_model_imputer_with_data.combined_storage
    # Manually define expected results for two months
    expected_results = {
        "2023-01": pl.DataFrame(
            {
                "grid_id": [1, 2, 1, 2, 1, 2],
                "date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-03",
                ],
                "target_col__imputed_flag": pl.Series(
                    [1, 1, 0, 0, 1, 0], dtype=pl.Int32
                ),
                "target_col__predicted": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "target_col__imputed": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                "target_col__score": [
                    1.0 * 0.85,
                    2.0 * 0.85,
                    1.0,
                    1.0,
                    1.0 * 0.85,
                    2.0,
                ],
                "target_col__share_imputed_across_all_grids": [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                ],
                "target_col__imputed_r7d": [
                    1.0,
                    2.0,
                    1.0,
                    1.5,
                    1.0,
                    1.6666666666666667,
                ],
            }
        ),
        "2023-02": pl.DataFrame(
            {
                "grid_id": [1, 2, 1, 2, 1, 2],
                "date": [
                    "2023-02-01",
                    "2023-02-01",
                    "2023-02-02",
                    "2023-02-02",
                    "2023-02-03",
                    "2023-02-03",
                ],
                "target_col__imputed_flag": pl.Series(
                    [1, 1, 0, 0, 1, 0], dtype=pl.Int32
                ),
                "target_col__predicted": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "target_col__imputed": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                "target_col__score": [
                    1.0 * 0.85,
                    2.0 * 0.85,
                    1.0,
                    1.0,
                    1.0 * 0.85,
                    2.0,
                ],
                "target_col__share_imputed_across_all_grids": [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                ],
                # Second month has different imputed_r7d values, as it includes rolling imputation from the first month
                "target_col__imputed_r7d": [
                    1.0,
                    1.75,
                    1.0,
                    1.6,
                    1.0,
                    1.6666666666666667,
                ],
            }
        ),
    }

    for month, expected_df in expected_results.items():
        result_subpath = OUTPUT_ARTIFACT_STAGE.for_month(month)
        result_df = combined_storage.read_dataframe(
            result_subpath=result_subpath,
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)
