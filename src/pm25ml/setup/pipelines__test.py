"""Tests for pipeline construction helpers."""

from unittest.mock import Mock

import arrow

from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.pipelines import define_pipelines


def test__define_pipelines__always_includes_pm25_for_monthly_ranges() -> None:
    pm25_pipeline = Mock()
    pm25_pipeline_constructor = Mock()
    pm25_pipeline_constructor.construct.return_value = pm25_pipeline
    temporal_config = TemporalConfig(
        start_date=arrow.get("2026-05-01"),
        end_date=arrow.get("2026-05-01"),
    )

    pipelines = define_pipelines(
        gee_pipeline_constructor=Mock(),
        ned_pipeline_constructor=Mock(),
        pm25_pipeline_constructor=pm25_pipeline_constructor,
        in_memory_grid=Mock(),
        archive_storage=Mock(),
        feature_planner=Mock(),
        temporal_config=temporal_config,
        profile_id="india",
        include_non_monthly=False,
    )

    pm25_pipeline_constructor.construct.assert_called_once_with(
        result_subpath="country=india/dataset=pm25/month=2026-05",
        month=arrow.get("2026-05-01"),
        temporal_config=temporal_config,
    )
    assert pm25_pipeline in pipelines
