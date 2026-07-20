"""Runner to get the data from a variety of sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.collectors.validate_configuration import validate_configuration
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import init_dependencies_from_env

if TYPE_CHECKING:
    from pm25ml.setup.dependency_injection import Pm25mlContainer


def _main(container: Pm25mlContainer) -> None:
    collector = container.collector()
    processors = list(container.pipelines())
    grid_cell_count = container.config.profile.grid_cell_count()

    logger.info("Validating export pipeline config")
    validate_configuration(processors, grid_cell_count)

    logger.info("Collect data from processors and store in the ingest archive")
    results = collector.collect(processors)

    logger.info("Combining results from the archive storage")
    monthly_combiner = container.monthly_combiner()
    combine_planner = container.combine_planner()
    monthly_combiner.combine_for_months(combine_planner.plan(results))

    logger.info("Imputing spatial data")
    spatial_imputation_manager = container.spatial_imputation_manager()
    spatial_imputation_manager.impute()
    logger.info("Recombining combined data with spatial interpolation")
    spatial_interpolation_recombiner = container.spatial_interpolation_recombiner()
    spatial_interpolation_recombiner.recombine(
        stages=[
            monthly_combiner.archived_wide_combiner.output_artifact,
            spatial_imputation_manager.output_data_artifact,
        ],
        overwrite_columns=True,
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    _main(container)
