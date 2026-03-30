from pathlib import Path

import pandas as pd
from morefs.memory import MemFS

from pm25ml.results.cv_summary_result_writer import CvSummaryResultWriter
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.training.model_storage import LoadedValidationMetadata

DESTINATION_BUCKET = "test_bucket"
TEST_COUNTRY = "india"


def _sample_validation_metadata() -> LoadedValidationMetadata:
    return LoadedValidationMetadata(
        model_run_ref="2026-03-29+10-00-00",
        cv_results=pd.DataFrame(
            {
                "fit_time": [1.0, 3.0],
                "score_time": [2.0, 4.0],
                "test_neg_root_mean_squared_error": [-1.5, -2.5],
                "train_neg_root_mean_squared_error": [-0.5, -1.5],
                "test_r2": [0.8, 0.9],
                "train_r2": [0.85, 0.95],
            },
        ),
        test_metrics={"r2": 0.91, "rmse": 1.85},
    )


def test__write__writes_cv_summary_csv_to_expected_location_with_expected_columns() -> None:
    model_run_ref = "v2.0.0"
    storage = FinalResultStorage(
        filesystem=MemFS(),
        destination_bucket=DESTINATION_BUCKET,
        output_path=f"country={TEST_COUNTRY}/run={model_run_ref}",
    )
    writer = CvSummaryResultWriter(
        model_run_ref=model_run_ref,
        output_storage=storage,
    )

    writer.write(_sample_validation_metadata())

    expected_dir = f"{DESTINATION_BUCKET}/country={TEST_COUNTRY}/run={model_run_ref}"
    files = storage.filesystem.ls(expected_dir)
    assert len(files) == 1

    file_name = Path(files[0]).name
    assert file_name == "v2.0.0_cv-summary.csv"

    with storage.filesystem.open(files[0], "rb") as f:
        summary_df = pd.read_csv(f)

    assert list(summary_df.columns) == [
        "model_name",
        "model_run_ref",
        "metric",
        "cv_mean",
        "cv_std",
        "cv_min",
        "cv_max",
    ]

    assert set(summary_df["metric"]) == {
        "fit_time",
        "score_time",
        "test_rmse",
        "train_rmse",
        "test_r2",
        "train_r2",
    }

    test_rmse_row = summary_df.loc[summary_df["metric"] == "test_rmse"].iloc[0]
    assert test_rmse_row["cv_mean"] == -2.0
    assert test_rmse_row["cv_min"] == -2.5
    assert test_rmse_row["cv_max"] == -1.5
    assert all(summary_df["model_name"] == "full_pm25")
    assert all(summary_df["model_run_ref"] == "2026-03-29+10-00-00")
