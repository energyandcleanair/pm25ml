"""Assess latest usable months from metadata-only pipeline coverage probes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from arrow import Arrow

from pm25ml.collectors.coverage import (
    CoverageProbeResult,
    calendar_dates,
    constrains_end_month,
    month_last_date,
)
from pm25ml.collectors.gee.gee_export_pipeline import GeeExportPipeline
from pm25ml.setup.end_month import latest_completed_month

_PROBE_MAX_WORKERS = 32

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig


@dataclass(frozen=True)
class DatasetAvailability:
    """Latest contiguous coverage for one configured dataset."""

    dataset: str
    required: bool
    unknown: bool
    latest_contiguous_date: date | None
    covered_dates: frozenset[date] = frozenset()
    unknown_dates: frozenset[date] = frozenset()
    detail: str | None = None


@dataclass(frozen=True)
class AvailabilityReport:
    """Summary of upstream availability for a pipeline profile."""

    profile_id: str
    latest_available_month: Arrow | None
    data_available_through: date | None
    next_month: Arrow
    next_month_in_progress: bool
    blockers: tuple[DatasetAvailability, ...]
    datasets: tuple[DatasetAvailability, ...]


def probe_pipelines(
    processors: Collection[ExportPipeline],
) -> list[tuple[PipelineConfig, CoverageProbeResult]]:
    """Run metadata-only coverage probes, parallelising non-GEE sources."""
    if not processors:
        return []

    gee_processors = [
        processor for processor in processors if isinstance(processor, GeeExportPipeline)
    ]
    other_processors = [
        processor for processor in processors if not isinstance(processor, GeeExportPipeline)
    ]
    results: list[tuple[PipelineConfig, CoverageProbeResult]] = []
    if other_processors:
        with ThreadPoolExecutor(max_workers=_PROBE_MAX_WORKERS) as executor:
            results.extend(executor.map(_probe_one, other_processors))
    # Earth Engine's client is not safe for concurrent getInfo calls.
    results.extend(_probe_one(processor) for processor in gee_processors)
    return results


def assess_availability(
    *,
    profile_id: str,
    candidate_pipeline_factory: Callable[[Arrow], Collection[ExportPipeline]],
    max_data_lag_months: int,
    now: Arrow | None = None,
) -> AvailabilityReport:
    """Probe configured monthly pipelines across the lag window and current month."""
    current = (now or Arrow.utcnow()).to("UTC")
    latest_completed = latest_completed_month(now=current)
    window_start = latest_completed.shift(months=-max_data_lag_months).floor("month")
    current_month = current.floor("month")
    months = list(Arrow.range("month", window_start, current_month))

    processors: list[ExportPipeline] = []
    dataset_order: list[str] = []
    for month in months:
        for processor in candidate_pipeline_factory(month):
            processors.append(processor)
            dataset = _dataset_id(processor.get_config_metadata())
            if dataset not in dataset_order:
                dataset_order.append(dataset)

    probed = probe_pipelines(processors)

    datasets = _dataset_availabilities(probed, dataset_order)
    required = [item for item in datasets if item.required]
    latest_available_month = _latest_passing_month(required, months, latest_completed)
    next_month = (
        latest_available_month.shift(months=1)
        if latest_available_month is not None
        else latest_completed
    )
    next_month_in_progress = next_month.floor("month") == current_month
    blockers = (
        ()
        if next_month_in_progress
        else tuple(item for item in required if _blocks_following_month(item, next_month))
    )
    data_available_through = (
        month_last_date(latest_available_month) if latest_available_month is not None else None
    )
    return AvailabilityReport(
        profile_id=profile_id,
        latest_available_month=latest_available_month,
        data_available_through=data_available_through,
        next_month=next_month.floor("month"),
        next_month_in_progress=next_month_in_progress,
        blockers=blockers,
        datasets=tuple(datasets),
    )


def format_availability_report(report: AvailabilityReport) -> str:
    """Render a human-readable availability report."""
    lines = [f"{report.profile_id} PM2.5 data availability", ""]
    if report.latest_available_month is None or report.data_available_through is None:
        lines.append("Latest available month: unknown")
        lines.append("Data available through: unknown")
    else:
        lines.append(
            f"Latest available month: {report.latest_available_month.format('MMMM YYYY')}",
        )
        lines.append(f"Data available through: {report.data_available_through.isoformat()}")

    if not report.next_month_in_progress:
        lines.append("")
        lines.append(
            f"Datasets preventing {report.next_month.format('MMMM YYYY')} availability:",
        )
        if report.blockers:
            lines.extend(
                f"  - {blocker.dataset} (latest: {_format_dataset_date(blocker)})"
                for blocker in report.blockers
            )
        else:
            lines.append("  none")

    lines.extend(["", "Latest available data by source:"])
    width = max((len(item.dataset) for item in report.datasets), default=0)
    lines.extend(
        f"  {item.dataset:<{width}}  {_format_dataset_date(item)}" for item in report.datasets
    )

    unknown_required = [item for item in report.datasets if item.required and item.unknown]
    if unknown_required:
        names = ", ".join(item.dataset for item in unknown_required)
        lines.extend(["", f"Unknown required coverage: {names}"])

    lines.extend(
        [
            "",
            "Note: Dates reflect upstream data availability, not verified",
            "completeness or successful processing. The pipeline may",
            "encounter missing observations or other validation issues.",
        ],
    )
    return "\n".join(lines) + "\n"


def _probe_one(
    processor: ExportPipeline,
) -> tuple[PipelineConfig, CoverageProbeResult]:
    return processor.get_config_metadata(), processor.probe_coverage()


def _dataset_id(config: PipelineConfig) -> str:
    return config.hive_path.require_key("dataset")


def _dataset_availabilities(
    probed: Sequence[tuple[PipelineConfig, CoverageProbeResult]],
    dataset_order: Sequence[str],
) -> list[DatasetAvailability]:
    grouped: dict[str, list[tuple[PipelineConfig, CoverageProbeResult]]] = {}
    for config, result in probed:
        grouped.setdefault(_dataset_id(config), []).append((config, result))

    return [
        _combine_dataset(dataset, grouped[dataset])
        for dataset in dataset_order
        if dataset in grouped
    ]


def _combine_dataset(
    dataset: str,
    parts: Sequence[tuple[PipelineConfig, CoverageProbeResult]],
) -> DatasetAvailability:
    required = any(constrains_end_month(config) for config, _result in parts)
    unknown_days: set[date] = set()
    covered_days: set[date] = set()
    requested_days: set[date] = set()
    details: list[str] = []
    for _config, result in parts:
        requested_days.update(result.requested_dates)
        if result.detail:
            details.append(result.detail)
        if result.status == "unknown":
            unknown_days.update(result.requested_dates)
        else:
            covered_days.update(result.covered_dates)

    unknown, latest = _contiguous_end_with_unknown(
        covered_days=covered_days,
        unknown_days=unknown_days,
        requested_days=requested_days,
    )
    return DatasetAvailability(
        dataset=dataset,
        required=required,
        unknown=unknown,
        latest_contiguous_date=latest,
        covered_dates=frozenset(covered_days),
        unknown_dates=frozenset(unknown_days),
        detail="; ".join(details) if details else None,
    )


def _contiguous_end_with_unknown(
    *,
    covered_days: set[date],
    unknown_days: set[date],
    requested_days: set[date],
) -> tuple[bool, date | None]:
    if not requested_days:
        return True, None

    end = max(requested_days)
    day = min(requested_days)
    while day <= end and day not in covered_days and day not in unknown_days:
        day += timedelta(days=1)
    if day > end:
        return False, None
    if day in unknown_days:
        return True, None

    current = day
    expected = day + timedelta(days=1)
    while expected <= end:
        if expected in unknown_days:
            return False, current
        if expected not in covered_days:
            return False, current
        current = expected
        expected += timedelta(days=1)
    return False, current


def _latest_passing_month(
    required: Sequence[DatasetAvailability],
    months: Sequence[Arrow],
    latest_completed: Arrow,
) -> Arrow | None:
    completed_months = [
        month.floor("month")
        for month in months
        if month.floor("month") <= latest_completed.floor("month")
    ]
    for month in reversed(completed_months):
        if all(_dataset_covers_month(item, month) for item in required):
            return month
    return None


def _dataset_covers_month(item: DatasetAvailability, month: Arrow) -> bool:
    needed = calendar_dates(month.floor("month").date(), month_last_date(month))
    if needed & item.unknown_dates:
        return False
    return needed <= item.covered_dates


def _blocks_following_month(
    item: DatasetAvailability,
    next_month: Arrow,
) -> bool:
    if item.unknown:
        return True
    latest = item.latest_contiguous_date
    if latest is None:
        return True
    return latest < month_last_date(next_month)


def _format_dataset_date(item: DatasetAvailability) -> str:
    if item.unknown or item.latest_contiguous_date is None:
        return "unknown"
    return item.latest_contiguous_date.isoformat()
