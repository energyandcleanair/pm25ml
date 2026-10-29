"""Tests for contiguous coverage helpers."""

from datetime import date

import arrow

from pm25ml.collectors.coverage import (
    CoverageProbeResult,
    calendar_dates,
    latest_contiguous_date,
)


def test__latest_contiguous_date__interior_gap__stops_before_gap() -> None:
    covered = calendar_dates(date(2026, 6, 1), date(2026, 6, 10)) | calendar_dates(
        date(2026, 6, 12),
        date(2026, 6, 30),
    )

    assert latest_contiguous_date(covered) == date(2026, 6, 10)


def test__latest_contiguous_date__later_isolated_day__does_not_extend() -> None:
    covered = {date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 30)}

    assert latest_contiguous_date(covered) == date(2026, 6, 2)


def test__latest_contiguous_date__full_month__returns_last_day() -> None:
    covered = calendar_dates(date(2026, 6, 1), date(2026, 6, 30))

    assert latest_contiguous_date(covered) == date(2026, 6, 30)


def test__latest_contiguous_date__empty__returns_none() -> None:
    assert latest_contiguous_date([]) is None


def test__covers_month__missing_first_day__is_incomplete() -> None:
    requested = calendar_dates(date(2026, 6, 1), date(2026, 6, 30))
    covered = calendar_dates(date(2026, 6, 2), date(2026, 6, 30))
    result = CoverageProbeResult.known(requested, covered)

    assert result.covers_month(arrow.get("2026-06-01")) is False


def test__covers_month__unknown__returns_none() -> None:
    result = CoverageProbeResult.unknown(calendar_dates(date(2026, 6, 1), date(2026, 6, 30)))

    assert result.covers_month(arrow.get("2026-06-01")) is None
