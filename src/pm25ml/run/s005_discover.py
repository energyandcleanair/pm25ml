"""Discover and persist the immutable end month for a pipeline run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from arrow import Arrow

    from pm25ml.collectors.end_month_selector import EndMonthCoordinator
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.setup.dependency_injection import Pm25mlContainer


def _candidate_pipelines(
    container: Pm25mlContainer,
    month: Arrow,
) -> Collection[ExportPipeline]:
    """Build the normal monthly pipelines for one discovery candidate."""
    return container.pipelines(
        temporal_config=TemporalConfig(start_date=month, end_date=month),
        include_non_monthly=False,
    )


def _main(
    coordinator: EndMonthCoordinator,
    request: TemporalConfigRequest,
    candidate_pipeline_factory: Callable[[Arrow], Collection[ExportPipeline]],
) -> TemporalConfig:
    """Resolve and persist the runtime month range for downstream stages."""
    temporal_config = coordinator.resolve(
        request,
        candidate_pipeline_factory=candidate_pipeline_factory,
    )
    logger.info(
        "Using persisted END_MONTH=%s",
        temporal_config.end_date.format("YYYY-MM-DD"),
    )
    return temporal_config


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.gee_auth()
    _main(
        coordinator=container.end_month_coordinator(),
        request=TemporalConfigRequest.from_env(),
        candidate_pipeline_factory=lambda month: _candidate_pipelines(container, month),
    )
