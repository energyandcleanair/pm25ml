"""Tests for pipeline construction helpers."""

from unittest.mock import Mock, patch

import arrow

from pm25ml.collectors.export_pipeline import (
    ExportPipeline,
    MissingDataHeuristic,
    PipelineConfig,
    PipelineConsumerBehaviour,
)
from pm25ml.setup.pipelines import EndMonthCandidatePipelineFactory


def _processor(
    path: str,
    *,
    constrains_end_month: bool = True,
    allows_missing: bool = False,
) -> Mock:
    processor = Mock(spec=ExportPipeline)
    processor.get_config_metadata.return_value = PipelineConfig(
        result_subpath=path,
        id_columns={"grid_id"},
        value_column_type_map={},
        expected_rows=0,
        constrains_end_month=constrains_end_month,
        consumer_behaviour=PipelineConsumerBehaviour(
            missing_data_heuristic=(
                MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
                if allows_missing
                else MissingDataHeuristic.FAIL
            ),
        ),
    )
    return processor


def test__candidate_pipeline_factory__builds_only_constraining_exact_month() -> None:
    required = _processor("country=india/dataset=required/month=2026-05")
    wrong_month = _processor("country=india/dataset=required/month=2026-04")
    nonconstraining = _processor(
        "country=india/dataset=pm25/month=2026-05",
        constrains_end_month=False,
    )
    allowed_missing = _processor(
        "country=india/dataset=optional/month=2026-05",
        allows_missing=True,
    )
    factory = EndMonthCandidatePipelineFactory(
        gee_pipeline_constructor=Mock(),
        ned_pipeline_constructor=Mock(),
        in_memory_grid=Mock(),
        archive_storage=Mock(),
        feature_planner=Mock(),
        profile_id="india",
    )

    with patch(
        "pm25ml.setup.pipelines.define_pipelines",
        return_value=[required, wrong_month, nonconstraining, allowed_missing],
    ) as define:
        result = factory.build(arrow.get("2026-05-01"))

    assert result == [required]
    assert define.call_args.kwargs["pm25_pipeline_constructor"] is None
    assert define.call_args.kwargs["include_non_monthly"] is False
    assert define.call_args.kwargs["temporal_config"].start_date == arrow.get("2026-05-01")
    assert define.call_args.kwargs["temporal_config"].end_date == arrow.get("2026-05-01")
