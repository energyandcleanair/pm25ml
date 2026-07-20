"""Entry point for imputation sampling in the PM2.5 ML project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.sample.imputation_sampler import SpatialTemporalImputationSampler
    from pm25ml.setup.date_params import TemporalConfig


def _main(
    samplers: Collection[SpatialTemporalImputationSampler],
    temporal_config: TemporalConfig,
) -> None:
    for sampler in samplers:
        logger.info(f"Starting sampling for {sampler.imputation_sampler_definition.model_name}")
        sampler.sample(temporal_config)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(container.imputation_samplers(), temporal_config)
