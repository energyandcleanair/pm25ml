"""Configuration for the final result writers."""

from pm25ml.results.cv_summary_result_writer import CvSummaryResultWriter
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.results.final_result_writer import FinalResultWriter
from pm25ml.results.final_stats_writer import FinalStatsWriter
from pm25ml.results.netcdf_final_result_writer import NetCdfResultWriter


def define_result_writers(
    storage: FinalResultStorage,
    model_run_ref: str,
) -> list[FinalResultWriter]:
    """Build the result writers for the application."""
    return [
        NetCdfResultWriter(
            model_run_ref=model_run_ref,
            output_storage=storage,
        ),
    ]


def define_stats_writers(
    storage: FinalResultStorage,
    model_run_ref: str,
) -> list[FinalStatsWriter]:
    """Build the final statistics writers for the application."""
    return [
        CvSummaryResultWriter(
            model_run_ref=model_run_ref,
            output_storage=storage,
        ),
    ]
