"""Shared NASA Earthdata granule search used by collection and availability probes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, cast

import arrow
import earthaccess

from pm25ml.collectors.coverage import calendar_dates

if TYPE_CHECKING:
    from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

EARTHACCESS_SEARCH_DATE_FORMAT = "YYYY-MM-DD"


def search_earthaccess_granules(
    dataset_descriptor: NedDatasetDescriptor,
) -> list[earthaccess.DataGranule]:
    """Search Earthdata for granules matching the dataset descriptor."""
    return list(
        earthaccess.search_data(
            short_name=dataset_descriptor.dataset_name,
            temporal=(
                dataset_descriptor.start_date.format(EARTHACCESS_SEARCH_DATE_FORMAT),
                dataset_descriptor.end_date.format(EARTHACCESS_SEARCH_DATE_FORMAT),
            ),
            count=-1,
            version=dataset_descriptor.dataset_version,
        ),
    )


def covered_dates_from_granules(
    granules: list[earthaccess.DataGranule],
    requested_dates: frozenset[date],
) -> frozenset[date] | None:
    """
    Return requested days overlapped by granule temporal extents.

    ``None`` means at least one granule had no parseable temporal extent, so coverage
    cannot be determined from metadata.
    """
    covered: set[date] = set()
    for granule in granules:
        extent = granule_temporal_range(granule)
        if extent is None:
            return None
        start_day, end_day = extent
        covered.update(calendar_dates(start_day, end_day) & requested_dates)
    return frozenset(covered)


def granule_temporal_range(granule: object) -> tuple[date, date] | None:
    """Return the inclusive UTC date range covered by a granule, if parseable."""
    getter = getattr(granule, "get", None)
    umm = _mapping(getter("umm") if callable(getter) else None)
    if not umm:
        umm = _mapping(getattr(granule, "umm", None))
    temporal = _mapping(umm.get("TemporalExtent"))
    if not temporal:
        return None

    range_dt = _mapping(temporal.get("RangeDateTime"))
    if range_dt:
        start = _parse_datetime(range_dt.get("BeginningDateTime"))
        end = _parse_datetime(range_dt.get("EndingDateTime"))
        if start is None or end is None:
            return None
        start_day = start.date()
        end_day = _inclusive_end_date(end)
        if end_day < start_day:
            return None
        return start_day, end_day

    single = _parse_datetime(temporal.get("SingleDateTime"))
    if single is None:
        return None
    day = single.date()
    return day, day


def _inclusive_end_date(end: datetime) -> date:
    """Treat an exclusive midnight end timestamp as the previous calendar day."""
    if end.time() == datetime.min.time():
        return (end - timedelta(microseconds=1)).date()
    return end.date()


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = arrow.get(cast("str | datetime", value)).to("UTC").datetime
    except (TypeError, ValueError, OSError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
