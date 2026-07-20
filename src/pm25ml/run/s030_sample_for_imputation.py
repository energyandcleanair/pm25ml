"""Entry point for imputation sampling in the PM2.5 ML project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.sample.imputation_sampler import SpatialTemporalImputationSampler


@inject
def _main(
    samplers: Collection[SpatialTemporalImputationSampler] = Provide[
        Pm25mlContainer.imputation_samplers
    ],
) -> None:
    for sampler in samplers:
        logger.info(f"Starting sampling for {sampler.imputation_sampler_definition.model_name}")
        sampler.sample()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
