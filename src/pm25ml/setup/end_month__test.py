"""Tests for end-month resolution and persistence."""

import arrow
import pytest
from morefs.memory import MemFS

from pm25ml.setup.end_month import (
    EndMonthStore,
    UnresolvedEndMonthError,
    latest_completed_month,
    parse_max_data_lag_months,
    resolve_end_month,
)


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


def test__resolve_end_month__stored_value__takes_priority_over_explicit_input() -> None:
    store = _store()
    store.write(arrow.get("2026-05-01"))

    result = resolve_end_month(
        explicit_value="2020-02-29",
        store=store,
        allow_provisional=False,
    )

    assert result.month == arrow.get("2026-05-01")
    assert result.source == "stored"


def test__resolve_end_month__explicit_input_without_stored_value__is_selected() -> None:
    result = resolve_end_month(
        explicit_value="2020-02-29",
        store=_store(),
        allow_provisional=True,
    )

    assert result.month == arrow.get("2020-02-29")
    assert result.source == "explicit"


def test__resolve_end_month__stored_value__is_reused() -> None:
    filesystem = MemFS()
    _store(filesystem).write(arrow.get("2026-05-18"))

    result = resolve_end_month(
        explicit_value=None,
        store=_store(filesystem),
        allow_provisional=False,
    )

    assert result.month == arrow.get("2026-05-01")
    assert result.source == "stored"


def test__resolve_end_month__missing_value_for_later_stage__raises() -> None:
    with pytest.raises(UnresolvedEndMonthError, match="Run s005_discover first"):
        resolve_end_month(
            explicit_value=None,
            store=_store(),
            allow_provisional=False,
        )


def test__resolve_end_month__collection_stage__uses_provisional_previous_month() -> None:
    result = resolve_end_month(
        explicit_value=None,
        store=_store(),
        allow_provisional=True,
        now=arrow.get("2026-07-20T00:00:00Z"),
    )

    assert result.month == arrow.get("2026-06-01T00:00:00Z")
    assert result.source == "provisional"


def test__parse_max_data_lag_months__default_and_boundary() -> None:
    assert parse_max_data_lag_months(None) == 3
    assert parse_max_data_lag_months("0") == 0


def test__parse_max_data_lag_months__negative__raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parse_max_data_lag_months("-1")
