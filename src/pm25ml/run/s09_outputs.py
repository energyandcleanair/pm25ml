"""Script to output data in useful formats."""

import polars as pl
from dependency_injector.wiring import Provide, inject

from pm25ml.collectors.grid import Grid
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.results.final_result_writer import FinalResultWriter
from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)
from pm25ml.setup.temporal_config import TemporalConfig


@inject
def _main(
    final_data_artifact: DataArtifactRef = Provide[
        Pm25mlContainer.data_artifacts_container.final_prediction
    ],
    combined_storage: CombinedStorage = Provide[Pm25mlContainer.combined_storage],
    temporal_config: TemporalConfig = Provide[Pm25mlContainer.temporal_config],
    grid: Grid = Provide[Pm25mlContainer.in_memory_grid],
    final_result_writers: list[FinalResultWriter] = Provide[
        Pm25mlContainer.final_result_writers
    ],
) -> None:
    data_from_storage = (
        combined_storage.scan_stage(final_data_artifact.stage)
        .filter(
            pl.col("month").is_in(temporal_config.month_ids),
        )
        .drop("month")
        .rename(
            {
                "pm25__pm25__predicted": "pm25",
            },
        )
        .collect()
    )

    result = grid.to_xarray_with_data(data_from_storage)

    for outputter in final_result_writers:
        outputter.write(result)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
