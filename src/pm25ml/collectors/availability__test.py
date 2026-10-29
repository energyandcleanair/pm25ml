"""Tests for availability assessment from pipeline probes."""

from datetime import date
from unittest.mock import Mock

import arrow

from pm25ml.collectors.availability import (
    assess_availability,
    format_availability_report,
)
from pm25ml.collectors.coverage import CoverageProbeResult, calendar_dates
from pm25ml.collectors.export_pipeline import (
    ExportPipeline,
    MissingDataHeuristic,
    PipelineConfig,
    PipelineConsumerBehaviour,
)


def _month_dates(month: str) -> frozenset[date]:
    start = arrow.get(f"{month}-01")
    end = start.shift(months=1).shift(days=-1)
    return calendar_dates(start.date(), end.date())


def _processor(
    dataset: str,
    month: str,
    coverage: CoverageProbeResult,
    *,
    required: bool = True,
) -> Mock:
    processor = Mock(spec=ExportPipeline)
    processor.get_config_metadata.return_value = PipelineConfig(
        result_subpath=f"country=india/dataset={dataset}/month={month}",
        id_columns={"date", "grid_id"},
        value_column_type_map={},
        expected_rows=0,
        consumer_behaviour=PipelineConsumerBehaviour(
            missing_data_heuristic=(
                MissingDataHeuristic.FAIL
                if required
                else MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
            ),
        ),
        constrains_end_month=required,
    )
    processor.probe_coverage.return_value = coverage
    return processor


def _full(month: str) -> CoverageProbeResult:
    requested = _month_dates(month)
    return CoverageProbeResult.known(requested, requested)


def _through(month: str, last_day: int) -> CoverageProbeResult:
    start = arrow.get(f"{month}-01")
    requested = _month_dates(month)
    covered = calendar_dates(start.date(), date(start.year, start.month, last_day))
    return CoverageProbeResult.known(requested, covered)


def test__assess_availability__required_gap_blocks_month_optional_does_not() -> None:
    def factory(month: arrow.Arrow) -> list[Mock]:
        month_id = month.format("YYYY-MM")
        if month_id == "2026-09":
            return [
                _processor("merra_aot", month_id, _full(month_id)),
                _processor("pm25", month_id, _through(month_id, 10), required=False),
            ]
        if month_id == "2026-10":
            return [
                _processor("merra_aot", month_id, _through(month_id, 15)),
                _processor("pm25", month_id, _through(month_id, 20), required=False),
            ]
        return [
            _processor("merra_aot", month_id, _full(month_id)),
            _processor("pm25", month_id, _full(month_id), required=False),
        ]

    report = assess_availability(
        profile_id="india",
        candidate_pipeline_factory=factory,
        max_data_lag_months=1,
        now=arrow.get("2026-10-20T00:00:00Z"),
    )

    assert report.latest_available_month == arrow.get("2026-09-01")
    assert report.data_available_through == date(2026, 9, 30)
    by_name = {item.dataset: item for item in report.datasets}
    assert by_name["merra_aot"].latest_contiguous_date == date(2026, 10, 15)
    assert by_name["pm25"].required is False
    assert report.next_month_in_progress is True
    assert report.blockers == ()
    assert "Datasets preventing" not in format_availability_report(report)


def test__assess_availability__unknown_required__does_not_select_that_month() -> None:
    def factory(month: arrow.Arrow) -> list[Mock]:
        month_id = month.format("YYYY-MM")
        coverage = (
            CoverageProbeResult.unknown(_month_dates(month_id), "gee failed")
            if month_id == "2026-09"
            else _full(month_id)
        )
        return [_processor("era5_land", month_id, coverage)]

    report = assess_availability(
        profile_id="india",
        candidate_pipeline_factory=factory,
        max_data_lag_months=1,
        now=arrow.get("2026-10-05T00:00:00Z"),
    )

    assert report.latest_available_month == arrow.get("2026-08-01")
    assert {item.dataset for item in report.datasets if item.unknown} == set()


def test__assess_availability__interior_gap__does_not_use_later_days() -> None:
    def factory(month: arrow.Arrow) -> list[Mock]:
        month_id = month.format("YYYY-MM")
        requested = _month_dates(month_id)
        if month_id == "2026-09":
            covered = calendar_dates(date(2026, 9, 1), date(2026, 9, 10)) | {
                date(2026, 9, 30),
            }
            coverage = CoverageProbeResult.known(requested, frozenset(covered))
        else:
            coverage = _full(month_id)
        return [_processor("modis_aod", month_id, coverage)]

    report = assess_availability(
        profile_id="india",
        candidate_pipeline_factory=factory,
        max_data_lag_months=0,
        now=arrow.get("2026-10-05T00:00:00Z"),
    )

    assert report.latest_available_month is None
    by_name = {item.dataset: item for item in report.datasets}
    assert by_name["modis_aod"].latest_contiguous_date == date(2026, 9, 10)


def test__format_availability_report__completed_following_month__lists_blockers() -> None:
    def factory(month: arrow.Arrow) -> list[Mock]:
        month_id = month.format("YYYY-MM")
        if month_id == "2026-10":
            return [
                _processor(
                    "merra_aot",
                    month_id,
                    CoverageProbeResult.known(_month_dates(month_id), frozenset()),
                ),
                _processor("era5_land", month_id, _through(month_id, 21)),
            ]
        return [
            _processor("merra_aot", month_id, _full(month_id)),
            _processor("era5_land", month_id, _full(month_id)),
        ]

    report = assess_availability(
        profile_id="india",
        candidate_pipeline_factory=factory,
        max_data_lag_months=1,
        now=arrow.get("2026-11-05T00:00:00Z"),
    )
    text = format_availability_report(report)

    assert "Latest available month: September 2026" in text
    assert "Data available through: 2026-09-30" in text
    assert "Datasets preventing October 2026 availability:" in text
    assert "- merra_aot (latest: 2026-09-30)" in text
    assert "- era5_land (latest: 2026-10-21)" in text
    assert report.next_month_in_progress is False
    assert "Note: Dates reflect upstream data availability" in text
