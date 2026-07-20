"""Runner to get the data from a variety of sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.collectors.validate_configuration import validate_configuration
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.collectors.collector import RawDataCollector
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager
    from pm25ml.combiners.archive.combine_planner import CombinePlanner
    from pm25ml.combiners.recombiner.recombiner import Recombiner
    from pm25ml.imputation.spatial.spatial_imputation_manager import SpatialImputationManager
    from pm25ml.setup.date_params import TemporalConfig


def _main(  # noqa: PLR0913
    processors: Collection[ExportPipeline],
    collector: RawDataCollector,
    grid_cell_count: int,
    monthly_combiner: MonthlyCombinerManager,
    combine_planner: CombinePlanner,
    spatial_imputation_manager: SpatialImputationManager,
    spatial_interpolation_recombiner: Recombiner,
    temporal_config: TemporalConfig,
) -> None:
    """Collect, combine, and spatially impute the configured month range."""
    logger.info("Validating export pipeline config")
    validate_configuration(processors, grid_cell_count)

    logger.info("Collect data from processors and store in the ingest archive")
    results = collector.collect(processors)

    logger.info("Combining results from the archive storage")
    monthly_combiner.combine_for_months(combine_planner.plan(results, temporal_config))

    logger.info("Imputing spatial data")
    spatial_imputation_manager.impute(temporal_config)
    logger.info("Recombining combined data with spatial interpolation")
    spatial_interpolation_recombiner.recombine(
        stages=[
            monthly_combiner.archived_wide_combiner.output_artifact,
            spatial_imputation_manager.output_data_artifact,
        ],
        temporal_config=temporal_config,
        overwrite_columns=True,
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.gee_auth()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(
        processors=list(container.pipelines(temporal_config=temporal_config)),
        collector=container.collector(),
        grid_cell_count=container.settings().grid_cell_count,
        monthly_combiner=container.monthly_combiner(),
        combine_planner=container.combine_planner(),
        spatial_imputation_manager=container.spatial_imputation_manager(),
        spatial_interpolation_recombiner=container.spatial_interpolation_recombiner(),
        temporal_config=temporal_config,
    )
