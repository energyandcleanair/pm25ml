"""Report the latest month the pipeline could generate from upstream availability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.collectors.availability import assess_availability, format_availability_report
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from arrow import Arrow

    from pm25ml.collectors.availability import AvailabilityReport
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.setup.dependency_injection import Pm25mlContainer


def _candidate_pipelines(
    container: Pm25mlContainer,
    month: Arrow,
) -> Collection[ExportPipeline]:
    """Build the normal monthly pipelines for one availability candidate."""
    return container.pipelines(
        temporal_config=TemporalConfig(start_date=month, end_date=month),
        include_non_monthly=False,
    )


def _main(
    *,
    profile_id: str,
    request: TemporalConfigRequest,
    candidate_pipeline_factory: Callable[[Arrow], Collection[ExportPipeline]],
    now: Arrow | None = None,
) -> AvailabilityReport:
    """Probe upstream availability without collecting, persisting, or modelling."""
    return assess_availability(
        profile_id=profile_id,
        candidate_pipeline_factory=candidate_pipeline_factory,
        max_data_lag_months=request.max_data_lag_months,
        now=now,
    )


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.gee_auth()
    report = _main(
        profile_id=container.settings().profile_id,
        request=TemporalConfigRequest.from_env(),
        candidate_pipeline_factory=lambda month: _candidate_pipelines(container, month),
    )
    print(format_availability_report(report), end="")  # noqa: T201
