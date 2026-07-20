"""Tests for the explicit collection runner boundary."""

from unittest.mock import Mock

import arrow

from pm25ml.run.s010_fetch_and_combine import _main
from pm25ml.setup.date_params import TemporalConfig


def test__main__runs_collection_and_preparation_in_order() -> None:
    processors = []
    collector = Mock()
    collector.collect.return_value = ["upload"]
    combine_planner = Mock()
    combine_planner.plan.return_value = ["plan"]
    monthly_combiner = Mock()
    monthly_combiner.archived_wide_combiner.output_artifact = "combined"
    spatial_imputation_manager = Mock()
    spatial_imputation_manager.output_data_artifact = "spatial"
    spatial_interpolation_recombiner = Mock()
    temporal_config = TemporalConfig(
        start_date=arrow.get("2024-01-01"),
        end_date=arrow.get("2024-02-01"),
    )

    _main(
        processors=processors,
        collector=collector,
        grid_cell_count=10,
        monthly_combiner=monthly_combiner,
        combine_planner=combine_planner,
        spatial_imputation_manager=spatial_imputation_manager,
        spatial_interpolation_recombiner=spatial_interpolation_recombiner,
        temporal_config=temporal_config,
    )

    collector.collect.assert_called_once_with(processors)
    combine_planner.plan.assert_called_once_with(["upload"], temporal_config)
    monthly_combiner.combine_for_months.assert_called_once_with(["plan"])
    spatial_imputation_manager.impute.assert_called_once_with(temporal_config)
    spatial_interpolation_recombiner.recombine.assert_called_once_with(
        stages=["combined", "spatial"],
        temporal_config=temporal_config,
        overwrite_columns=True,
    )
