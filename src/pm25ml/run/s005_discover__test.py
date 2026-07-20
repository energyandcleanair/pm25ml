"""Tests for the explicit end-month discovery runner boundary."""

from unittest.mock import Mock

import arrow

from pm25ml.collectors.end_month_selector import EndMonthCoordinator
from pm25ml.run.s005_discover import _candidate_pipelines, _main
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.settings import TemporalConfigRequest


def _request() -> TemporalConfigRequest:
    return TemporalConfigRequest(
        start_month=arrow.get("2024-01-01"),
        explicit_end_month=None,
        max_data_lag_months=3,
    )


def test__main__resolves_and_returns_the_persisted_temporal_config() -> None:
    temporal_config = TemporalConfig(
        start_date=arrow.get("2024-01-01"),
        end_date=arrow.get("2024-06-01"),
    )
    coordinator = Mock(spec=EndMonthCoordinator)
    coordinator.resolve.return_value = temporal_config
    candidate_pipeline_factory = Mock()
    request = _request()

    result = _main(coordinator, request, candidate_pipeline_factory)

    assert result == temporal_config
    coordinator.resolve.assert_called_once_with(
        request,
        candidate_pipeline_factory=candidate_pipeline_factory,
    )


def test__candidate_pipelines__uses_normal_monthly_pipeline_factory() -> None:
    container = Mock()
    container.pipelines.return_value = ["pipeline"]
    month = arrow.get("2024-06-01")

    result = _candidate_pipelines(container, month)

    assert result == ["pipeline"]
    container.pipelines.assert_called_once_with(
        temporal_config=TemporalConfig(start_date=month, end_date=month),
        include_non_monthly=False,
    )
