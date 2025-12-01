"""Configuration for the final result writers."""

from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.results.final_result_writer import FinalResultWriter
from pm25ml.results.netcdf_final_result_writer import NetCdfResultWriter


def define_result_writers(
    storage: FinalResultStorage,
) -> list[FinalResultWriter]:
    """Build the result writers for the application."""
    return [
        NetCdfResultWriter(
            output_ref=DataArtifactRef(
                stage="netcdf",
            ),
            output_storage=storage,
            file_prefix="pm25_final",
        ),
    ]
