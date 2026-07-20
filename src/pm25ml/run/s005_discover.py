"""Discover and persist the immutable end month for a pipeline run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import init_dependencies_from_env

if TYPE_CHECKING:
    from pm25ml.setup.dependency_injection import Pm25mlContainer


def _main(container: Pm25mlContainer) -> None:
    resolution = container.end_month_coordinator().resolve()
    resolution.persist()
    logger.info(
        "Using persisted END_MONTH=%s",
        resolution.temporal_config.end_date.format("YYYY-MM-DD"),
    )


if __name__ == "__main__":
    container = init_dependencies_from_env(end_month_mode="discovery")
    _main(container)
