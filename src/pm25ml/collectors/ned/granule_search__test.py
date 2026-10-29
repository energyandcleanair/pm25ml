"""Tests for NASA granule metadata coverage."""

from datetime import date

from pm25ml.collectors.coverage import calendar_dates
from pm25ml.collectors.ned.granule_search import covered_dates_from_granules, granule_temporal_range


def _granule(start: str, end: str) -> dict:
    return {
        "umm": {
            "TemporalExtent": {
                "RangeDateTime": {
                    "BeginningDateTime": start,
                    "EndingDateTime": end,
                },
            },
        },
    }


def test__granule_temporal_range__daily_granule__covers_that_day() -> None:
    start, end = granule_temporal_range(
        _granule("2026-09-01T00:00:00.000Z", "2026-09-01T23:59:59.000Z"),
    )

    assert start == date(2026, 9, 1)
    assert end == date(2026, 9, 1)


def test__granule_temporal_range__exclusive_midnight_end__does_not_include_next_day() -> None:
    start, end = granule_temporal_range(
        _granule("2026-09-01T00:00:00.000Z", "2026-09-02T00:00:00.000Z"),
    )

    assert start == date(2026, 9, 1)
    assert end == date(2026, 9, 1)


def test__covered_dates_from_granules__unparseable_extent__returns_none() -> None:
    requested = calendar_dates(date(2026, 9, 1), date(2026, 9, 2))
    granules = [_granule("2026-09-01T00:00:00.000Z", "2026-09-01T23:59:59.000Z"), {"umm": {}}]

    assert covered_dates_from_granules(granules, requested) is None


def test__covered_dates_from_granules__gap__omits_missing_day() -> None:
    requested = calendar_dates(date(2026, 9, 1), date(2026, 9, 3))
    granules = [
        _granule("2026-09-01T00:00:00.000Z", "2026-09-01T23:59:59.000Z"),
        _granule("2026-09-03T00:00:00.000Z", "2026-09-03T23:59:59.000Z"),
    ]

    assert covered_dates_from_granules(granules, requested) == {
        date(2026, 9, 1),
        date(2026, 9, 3),
    }
