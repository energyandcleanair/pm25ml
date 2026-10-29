"""Writer for outputting final-stage CV summary results as CSV."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pm25ml.results.final_result_storage import FinalResultStorage
    from pm25ml.training.model_storage import LoadedValidationMetadata


def cv_summary_filename(model_run_ref: str) -> str:
    """Return the CV summary CSV filename for a run."""
    return f"pm25-cv-summary_{model_run_ref}.csv"


def mean_cv_r2_and_rmse(cv_results: pd.DataFrame) -> tuple[float, float]:
    """Return mean cross-validation r2 and RMSE from sklearn CV columns."""
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

    return float(cv_results["test_r2"].mean()), float(rmse_scores.mean())


def format_performance_summary(model_name: str, stats: LoadedValidationMetadata) -> str:
    """Format CV and test metrics for a trained model."""
    r2, rmse = mean_cv_r2_and_rmse(stats.cv_results)
    lines = [
        f"## {model_name} performance ##",
        f"CV r2={r2:.3f}  RMSE={rmse:.3f}",
    ]
    test_metrics = stats.test_metrics or {}
    test_r2 = test_metrics.get("r2")
    test_rmse = test_metrics.get("rmse")
    if test_r2 is not None and test_rmse is not None:
        lines.append(f"Test r2={float(test_r2):.3f}  RMSE={float(test_rmse):.3f}")
    return "\n".join(lines)


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

        filename = cv_summary_filename(self.model_run_ref)

        self.output_storage.write(
            BytesIO(csv_bytes),
            file_name=filename,
        )

    def _summarise_cv_results(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        r2, rmse = mean_cv_r2_and_rmse(cv_results)
        summary = {
            "r2": r2,
            "rmse_ugm3": rmse,
        }

        return pd.DataFrame([summary]).astype(
            {
                "r2": "float64",
                "rmse_ugm3": "float64",
            },
            copy=False,
        )
