"""Runner to get the data from a variety of sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from pm25ml.collectors.validate_configuration import validate_configuration
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.collectors.collector import RawDataCollector
    from pm25ml.collectors.export_pipeline import (
        ExportPipeline,
    )
    from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager
    from pm25ml.combiners.archive.combine_planner import CombinePlanner
    from pm25ml.combiners.recombiner.recombiner import Recombiner
    from pm25ml.imputation.spatial.spatial_imputation_manager import SpatialImputationManager


@inject
def _main(  # noqa: PLR0913
    processors: Collection[ExportPipeline] = Provide[Pm25mlContainer.pipelines],
    collector: RawDataCollector = Provide[Pm25mlContainer.collector],
    monthly_combiner: MonthlyCombinerManager = Provide[Pm25mlContainer.monthly_combiner],
    combine_planner: CombinePlanner = Provide[Pm25mlContainer.combine_planner],
    grid_cell_count: int = Provide[Pm25mlContainer.config.profile.grid_cell_count],
    spatial_imputation_manager: SpatialImputationManager = Provide[
        Pm25mlContainer.spatial_imputation_manager
    ],
    spatial_interpolation_recombiner: Recombiner = Provide[
        Pm25mlContainer.spatial_interpolation_recombiner
    ],
) -> None:
    logger.info("Validating export pipeline config")
    validate_configuration(processors, grid_cell_count)

    logger.info("Collect data from processors and store in the ingest archive")
    results = collector.collect(processors)

    logger.info("Combining results from the archive storage")
    monthly_combiner.combine_for_months(combine_planner.plan(results))

    logger.info("Imputing spatial data")
    spatial_imputation_manager.impute()
    logger.info("Recombining combined data with spatial interpolation")
    spatial_interpolation_recombiner.recombine(
        stages=[
            monthly_combiner.archived_wide_combiner.output_artifact,
            spatial_imputation_manager.output_data_artifact,
        ],
        overwrite_columns=True,
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
