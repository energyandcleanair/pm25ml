"""Writer for outputting final-stage CV summary results as CSV."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from pm25ml.training.full_model_pipeline import MODEL_NAME

if TYPE_CHECKING:
    import pandas as pd

    from pm25ml.results.final_result_storage import FinalResultStorage
    from pm25ml.training.model_storage import LoadedValidationMetadata
    from pm25ml.training.types import ModelName


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
        self.model_name: ModelName = MODEL_NAME

    def write(self, stats: LoadedValidationMetadata) -> None:
        """Write CV summary metadata to CSV in final result storage."""
        cv_results = self._normalise_cv_columns(stats.cv_results)
        summary = self._summarise_cv_results(cv_results)

        summary_with_context = summary.assign(
            model_name=self.model_name,
            model_run_ref=stats.model_run_ref,
        )

        output_df = summary_with_context[
            [
                "model_name",
                "model_run_ref",
                "metric",
                "cv_mean",
                "cv_std",
                "cv_min",
                "cv_max",
            ]
        ]

        csv_bytes = output_df.to_csv(index=False).encode("utf-8")

        filename = f"{self.model_run_ref}_cv-summary.csv"

        self.output_storage.write(
            BytesIO(csv_bytes),
            file_name=filename,
        )

    def _normalise_cv_columns(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        return cv_results.rename(
            columns={
                "test_neg_root_mean_squared_error": "test_rmse",
                "train_neg_root_mean_squared_error": "train_rmse",
            },
        )

    def _summarise_cv_results(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        numeric_cv_results = cv_results.select_dtypes(include="number")
        if numeric_cv_results.empty:
            msg = "No numeric cross-validation columns found to summarise."
            raise ValueError(msg)

        summary = (
            numeric_cv_results.agg(["mean", "std", "min", "max"])
            .transpose()
            .reset_index()
            .rename(
                columns={
                    "index": "metric",
                    "mean": "cv_mean",
                    "std": "cv_std",
                    "min": "cv_min",
                    "max": "cv_max",
                },
            )
        )
        return summary.astype(
            {
                "metric": "string",
                "cv_mean": "float64",
                "cv_std": "float64",
                "cv_min": "float64",
                "cv_max": "float64",
            },
            copy=False,
        )
