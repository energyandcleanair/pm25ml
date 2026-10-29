"""Tests for the independent data availability command."""

from datetime import date
from unittest.mock import Mock

import arrow

from pm25ml.collectors.availability import AvailabilityReport, DatasetAvailability
from pm25ml.collectors.availability import format_availability_report
from pm25ml.collectors.coverage import CoverageProbeResult, calendar_dates
from pm25ml.collectors.export_pipeline import ExportPipeline, PipelineConfig
from pm25ml.run.check_data_availability import _candidate_pipelines, _main
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.settings import TemporalConfigRequest


def test__main__probes_without_collecting_or_persisting() -> None:
    requested = calendar_dates(date(2026, 9, 1), date(2026, 9, 30))
    processor = Mock(spec=ExportPipeline)
    processor.get_config_metadata.return_value = PipelineConfig(
        result_subpath="country=india/dataset=merra_aot/month=2026-09",
        id_columns={"date", "grid_id"},
        value_column_type_map={},
        expected_rows=0,
    )
    processor.probe_coverage.return_value = CoverageProbeResult.known(requested, requested)
    factory = Mock(
        side_effect=lambda month: [processor] if month.format("YYYY-MM") == "2026-09" else [],
    )
    collector = Mock()
    store = Mock()

    report = _main(
        profile_id="india",
        request=TemporalConfigRequest(
            start_month=arrow.get("2018-08-01"),
            explicit_end_month=None,
            max_data_lag_months=0,
        ),
        candidate_pipeline_factory=factory,
        now=arrow.get("2026-10-05T00:00:00Z"),
    )

    assert report.latest_available_month == arrow.get("2026-09-01")
    collector.collect.assert_not_called()
    store.write.assert_not_called()
    store.read.assert_not_called()


def test__main__formats_report_from_probe_result() -> None:
    report = AvailabilityReport(
        profile_id="india",
        latest_available_month=arrow.get("2026-09-01"),
        data_available_through=date(2026, 9, 30),
        next_month=arrow.get("2026-10-01"),
        next_month_in_progress=True,
        blockers=(
            DatasetAvailability(
                dataset="merra_aot",
                required=True,
                unknown=False,
                latest_contiguous_date=date(2026, 9, 30),
            ),
        ),
        datasets=(
            DatasetAvailability(
                dataset="merra_aot",
                required=True,
                unknown=False,
                latest_contiguous_date=date(2026, 9, 30),
            ),
            DatasetAvailability(
                dataset="pm25",
                required=False,
                unknown=True,
                latest_contiguous_date=None,
            ),
        ),
    )

    text = format_availability_report(report)

    assert text.startswith("india PM2.5 data availability")
    assert "Latest available month: September 2026" in text
    assert "Datasets preventing" not in text
    assert "pm25" in text
    assert "unknown" in text


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
