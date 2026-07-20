"""Tests for automatic end-month selection."""

from unittest.mock import Mock

import arrow
import pytest

from pm25ml.collectors.collector import DataCompleteness, UploadResult
from pm25ml.collectors.end_month_selector import (
    AutomaticEndMonthSelector,
    EndMonthCoordinator,
)
from pm25ml.collectors.export_pipeline import (
    ExportPipeline,
    MissingDataHeuristic,
    PipelineConfig,
    PipelineConsumerBehaviour,
)
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.end_month import EndMonthStore, StaleDataError


def _processor(
    dataset: str,
    month: str,
    *,
    allows_missing: bool = False,
    constrains_end_month: bool = True,
) -> Mock:
    processor = Mock(spec=ExportPipeline)
    processor.get_config_metadata.return_value = PipelineConfig(
        result_subpath=f"country=india/dataset={dataset}/month={month}",
        id_columns={"date", "grid_id"},
        value_column_type_map={},
        expected_rows=0,
        consumer_behaviour=PipelineConsumerBehaviour(
            missing_data_heuristic=(
                MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
                if allows_missing
                else MissingDataHeuristic.FAIL
            ),
        ),
        constrains_end_month=constrains_end_month,
    )
    return processor


def _result(processor: Mock, completeness: DataCompleteness) -> UploadResult:
    return UploadResult(processor.get_config_metadata(), completeness)


def test__select__newest_candidate_complete__selects_it() -> None:
    june = [_processor("a", "2026-06"), _processor("b", "2026-06")]
    collector = Mock()
    collector.collect.return_value = [
        _result(june[0], DataCompleteness.COMPLETE),
        _result(june[1], DataCompleteness.ALREADY_UPLOADED),
    ]

    selected = AutomaticEndMonthSelector(collector).select(
        candidate_pipeline_factory=lambda _candidate: june,
        latest_candidate=arrow.get("2026-06-01"),
        max_data_lag_months=3,
    )

    assert selected == arrow.get("2026-06-01")
    collector.collect.assert_called_once_with(june, allow_missing_required=True)


def test__select__incomplete_candidate_and_gap__tries_exact_earlier_months() -> None:
    june = _processor("satellite", "2026-06")
    may = _processor("satellite", "2026-05")
    april = _processor("satellite", "2026-04")
    collector = Mock()
    collector.collect.side_effect = [
        [_result(june, DataCompleteness.EMPTY)],
        [_result(may, DataCompleteness.EMPTY)],
        [_result(april, DataCompleteness.COMPLETE)],
    ]
    by_month = {"2026-06": [june], "2026-05": [may], "2026-04": [april]}

    selected = AutomaticEndMonthSelector(collector).select(
        candidate_pipeline_factory=lambda candidate: by_month[candidate.format("YYYY-MM")],
        latest_candidate=arrow.get("2026-06-01"),
        max_data_lag_months=2,
    )

    assert selected == arrow.get("2026-04-01")
    assert collector.collect.call_count == 3


def test__select__no_complete_candidate_within_inclusive_lag__raises_details() -> None:
    candidates = [_processor("satellite", month) for month in ("2026-06", "2026-05")]
    collector = Mock()
    collector.collect.side_effect = [
        [_result(processor, DataCompleteness.EMPTY)] for processor in candidates
    ]
    by_month = {
        processor.get_config_metadata().hive_path.require_key("month"): [processor]
        for processor in candidates
    }

    with pytest.raises(StaleDataError, match=r"2026-06: satellite.*2026-05: satellite"):
        AutomaticEndMonthSelector(collector).select(
            candidate_pipeline_factory=lambda candidate: by_month[candidate.format("YYYY-MM")],
            latest_candidate=arrow.get("2026-06-01"),
            max_data_lag_months=1,
        )


def test__select__operational_failure__is_not_treated_as_missing_data() -> None:
    processor = _processor("satellite", "2026-06")
    collector = Mock()
    collector.collect.side_effect = RuntimeError("authentication failed")

    with pytest.raises(RuntimeError, match="authentication failed"):
        AutomaticEndMonthSelector(collector).select(
            candidate_pipeline_factory=lambda _candidate: [processor],
            latest_candidate=arrow.get("2026-06-01"),
            max_data_lag_months=3,
        )


def test__coordinator__stored_source__returns_no_op_resolution() -> None:
    collector = Mock()
    candidate_pipeline_factory = Mock()
    store = Mock(spec=EndMonthStore)
    coordinator = EndMonthCoordinator(
        collector=collector,
        candidate_pipeline_factory=candidate_pipeline_factory,
        start_date=arrow.get("2020-01-01"),
        configured_end_month=arrow.get("2026-06-01"),
        source="stored",
        max_data_lag_months=3,
        store=store,
    )

    resolution = coordinator.resolve()
    resolution.persist()

    assert resolution.temporal_config == TemporalConfig(
        start_date=arrow.get("2020-01-01"),
        end_date=arrow.get("2026-06-01"),
    )
    collector.collect.assert_not_called()
    candidate_pipeline_factory.build.assert_not_called()
    store.write.assert_not_called()


def test__coordinator__explicit_source__persists_before_later_stages() -> None:
    store = Mock(spec=EndMonthStore)
    coordinator = EndMonthCoordinator(
        collector=Mock(),
        candidate_pipeline_factory=Mock(),
        start_date=arrow.get("2020-01-01"),
        configured_end_month=arrow.get("2026-06-01"),
        source="explicit",
        max_data_lag_months=3,
        store=store,
    )

    resolution = coordinator.resolve()
    resolution.persist()

    store.write.assert_called_once_with(arrow.get("2026-06-01"))


def test__coordinator__provisional_source__returns_new_config_and_defers_persistence() -> None:
    june = _processor("satellite", "2026-06")
    may = _processor("satellite", "2026-05")
    collector = Mock()
    collector.collect.side_effect = [
        [_result(june, DataCompleteness.EMPTY)],
        [_result(may, DataCompleteness.COMPLETE)],
    ]
    candidate_pipeline_factory = Mock()
    candidate_pipeline_factory.build.side_effect = lambda month: (
        [june] if month.format("YYYY-MM") == "2026-06" else [may]
    )
    store = Mock(spec=EndMonthStore)
    coordinator = EndMonthCoordinator(
        collector=collector,
        candidate_pipeline_factory=candidate_pipeline_factory,
        start_date=arrow.get("2020-01-01"),
        configured_end_month=arrow.get("2026-06-01"),
        source="provisional",
        max_data_lag_months=3,
        store=store,
    )

    resolution = coordinator.resolve()

    assert resolution.temporal_config == TemporalConfig(
        start_date=arrow.get("2020-01-01"),
        end_date=arrow.get("2026-05-01"),
    )
    store.write.assert_not_called()

    resolution.persist()

    store.write.assert_called_once_with(arrow.get("2026-05-01"))
