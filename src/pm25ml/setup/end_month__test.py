"""Tests for end-month storage and configuration helpers."""

import arrow
import pytest
from morefs.memory import MemFS

from pm25ml.setup.end_month import (
    EndMonthStore,
    UnresolvedEndMonthError,
    latest_completed_month,
    load_persisted_temporal_config,
    parse_max_data_lag_months,
)
from pm25ml.setup.settings import TemporalConfigRequest


def _store(filesystem: MemFS | None = None) -> EndMonthStore:
    return EndMonthStore(
        filesystem=filesystem if filesystem is not None else MemFS(),
        bucket="combined",
        profile_id="india",
        model_run_ref="run-1",
    )


def test__latest_completed_month__uses_previous_utc_calendar_month() -> None:
    now = arrow.get("2026-07-20T10:00:00+05:30")

    assert latest_completed_month(now=now) == arrow.get("2026-06-01T00:00:00+00:00")


def test__store__write_and_read__normalizes_to_month() -> None:
    store = _store()
    store.write(arrow.get("2026-05-18"))

    assert store.read_required() == arrow.get("2026-05-01")


def test__store__missing_required_value__raises_actionable_error() -> None:
    with pytest.raises(UnresolvedEndMonthError, match="Run s005_discover first"):
        _store().read_required()


def test__load_persisted_temporal_config__combines_request_and_stored_end() -> None:
    store = _store()
    store.write(arrow.get("2026-05-18"))
    request = TemporalConfigRequest(
        start_month=arrow.get("2020-01-01"),
        explicit_end_month=arrow.get("2024-01-01"),
        max_data_lag_months=3,
    )

    temporal_config = load_persisted_temporal_config(request, store)

    assert temporal_config.start_date == arrow.get("2020-01-01")
    assert temporal_config.end_date == arrow.get("2026-05-01")


def test__parse_max_data_lag_months__default_and_boundary() -> None:
    assert parse_max_data_lag_months(None) == 3
    assert parse_max_data_lag_months("0") == 0


def test__parse_max_data_lag_months__negative__raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parse_max_data_lag_months("-1")
