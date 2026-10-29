"""Metadata-only coverage results shared by discovery and the availability check."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arrow import Arrow

    from pm25ml.collectors.export_pipeline import PipelineConfig


CoverageStatus = Literal["known", "unknown"]


def calendar_dates(start: date, end: date) -> frozenset[date]:
    """Return every calendar day from ``start`` through ``end`` inclusive."""
    if end < start:
        return frozenset()
    return frozenset(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def dates_from_arrows(start: Arrow, end: Arrow) -> frozenset[date]:
    """Return every calendar day between two Arrow values, inclusive."""
    return calendar_dates(start.date(), end.date())


def month_last_date(month: Arrow) -> date:
    """Return the last calendar day of the given month."""
    return month.floor("month").shift(months=1).shift(days=-1).date()


def latest_contiguous_date(covered: Iterable[date]) -> date | None:
    """
    Return the last day of the first contiguous run starting at the earliest covered day.

    A gap before the first covered day is ignored. Days after the first missing day do
    not extend the result.
    """
    days = sorted(set(covered))
    if not days:
        return None

    current = days[0]
    expected = current
    for day in days:
        if day != expected:
            break
        current = day
        expected = day + timedelta(days=1)
    return current


@dataclass(frozen=True)
class CoverageProbeResult:
    """Coverage of a pipeline's requested dates from a metadata-only probe."""

    status: CoverageStatus
    requested_dates: frozenset[date]
    covered_dates: frozenset[date]
    detail: str | None = None

    @classmethod
    def known(
        cls,
        requested_dates: frozenset[date],
        covered_dates: frozenset[date],
        *,
        detail: str | None = None,
    ) -> CoverageProbeResult:
        """Build a conclusive coverage result."""
        return cls(
            status="known",
            requested_dates=requested_dates,
            covered_dates=covered_dates,
            detail=detail,
        )

    @classmethod
    def unknown(
        cls,
        requested_dates: frozenset[date],
        detail: str | None = None,
    ) -> CoverageProbeResult:
        """Build an inconclusive coverage result."""
        return cls(
            status="unknown",
            requested_dates=requested_dates,
            covered_dates=frozenset(),
            detail=detail,
        )

    def covers_month(self, month: Arrow) -> bool | None:
        """
        Return whether this result covers every day of ``month``.

        ``None`` means the probe was inconclusive.
        """
        if self.status == "unknown":
            return None
        needed = calendar_dates(month.floor("month").date(), month_last_date(month))
        if self.requested_dates:
            needed &= self.requested_dates
        return needed <= self.covered_dates


def constrains_end_month(config: PipelineConfig) -> bool:
    """Return whether missing data on this pipeline can block automatic end-month selection."""
    return config.constrains_end_month and not config.allows_missing_data
