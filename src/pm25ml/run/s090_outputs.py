"""Script to output data in useful formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from dependency_injector.wiring import Provide, inject

from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)
from pm25ml.training.full_model_pipeline import MODEL_NAME

if TYPE_CHECKING:
    from pm25ml.collectors.grid import Grid
    from pm25ml.combiners.combined_storage import CombinedStorage
    from pm25ml.combiners.data_artifact import DataArtifactRef
    from pm25ml.results.final_result_writer import FinalResultWriter
    from pm25ml.results.final_stats_writer import FinalStatsWriter
    from pm25ml.setup.date_params import TemporalConfig
    from pm25ml.training.model_storage import ModelStorage


@inject
def _main(  # noqa: PLR0913
    final_data_artifact: DataArtifactRef = Provide[
        Pm25mlContainer.data_artifacts_container.final_prediction
    ],
    combined_storage: CombinedStorage = Provide[Pm25mlContainer.combined_storage],
    temporal_config: TemporalConfig = Provide[Pm25mlContainer.temporal_config],
    grid: Grid = Provide[Pm25mlContainer.in_memory_grid],
    model_storage: ModelStorage = Provide[Pm25mlContainer.model_store],
    model_run_ref: str = Provide[Pm25mlContainer.model_run_ref],
    final_result_writers: list[FinalResultWriter] = Provide[Pm25mlContainer.final_result_writers],
    final_stats_writers: list[FinalStatsWriter] = Provide[Pm25mlContainer.final_stats_writers],
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
    validation_metadata = model_storage.load_validation_metadata(
        MODEL_NAME,
        model_run_ref,
    )

    for result_writer in final_result_writers:
        result_writer.write(result)

    for stats_writer in final_stats_writers:
        stats_writer.write(validation_metadata)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
