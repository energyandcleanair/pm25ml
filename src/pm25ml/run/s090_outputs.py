"""Script to output data in useful formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest
from pm25ml.training.full_model_pipeline import MODEL_NAME

if TYPE_CHECKING:
    from pm25ml.collectors.grid import Grid
    from pm25ml.combiners.combined_storage import CombinedStorage
    from pm25ml.combiners.data_artifact import DataArtifactRef
    from pm25ml.results.final_result_writer import FinalResultWriter
    from pm25ml.results.final_stats_writer import FinalStatsWriter
    from pm25ml.setup.date_params import TemporalConfig
    from pm25ml.training.model_storage import ModelStorage


def _main(  # noqa: PLR0913
    final_data_artifact: DataArtifactRef,
    combined_storage: CombinedStorage,
    temporal_config: TemporalConfig,
    grid: Grid,
    model_storage: ModelStorage,
    model_run_ref: str,
    final_result_writers: list[FinalResultWriter],
    final_stats_writers: list[FinalStatsWriter],
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
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(
        final_data_artifact=container.data_artifacts().final_prediction,
        combined_storage=container.combined_storage(),
        temporal_config=temporal_config,
        grid=container.in_memory_grid(),
        model_storage=container.model_store(),
        model_run_ref=container.model_run_ref(),
        final_result_writers=container.final_result_writers(),
        final_stats_writers=container.final_stats_writers(),
    )
