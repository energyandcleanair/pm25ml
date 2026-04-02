"""Writer for outputting final-stage CV summary results as CSV."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pm25ml.results.final_result_storage import FinalResultStorage
    from pm25ml.training.model_storage import LoadedValidationMetadata


class CvSummaryResultWriter:
    """Write a CSV summary for CV diagnostics from a specific model run."""

    def __init__(
        self,
        model_run_ref: str,
        output_storage: FinalResultStorage,
    ) -> None:
        """Initialize the CV summary writer."""
        self.model_run_ref = model_run_ref
        self.output_storage = output_storage

    def write(self, stats: LoadedValidationMetadata) -> None:
        """Write CV summary metadata to CSV in final result storage."""
        output_df = self._summarise_cv_results(stats.cv_results)

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")

        filename = f"pm25-cv-summary_{self.model_run_ref}.csv"

        self.output_storage.write(
            BytesIO(csv_bytes),
            file_name=filename,
        )

    def _summarise_cv_results(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        if "test_r2" not in cv_results.columns:
            msg = "Missing required CV column: test_r2"
            raise ValueError(msg)

        if "test_neg_root_mean_squared_error" in cv_results.columns:
            rmse_scores = -cv_results["test_neg_root_mean_squared_error"]
        elif "test_rmse" in cv_results.columns:
            rmse_scores = cv_results["test_rmse"].abs()
        else:
            msg = "Missing required CV column: test_neg_root_mean_squared_error or test_rmse"
            raise ValueError(msg)

        summary = {
            "r2": float(cv_results["test_r2"].mean()),
            "rmse_ugm3": float(rmse_scores.mean()),
        }

        return pd.DataFrame([summary]).astype(
            {
                "r2": "float64",
                "rmse_ugm3": "float64",
            },
            copy=False,
        )
