"""Script to impute missing data using a regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)

if TYPE_CHECKING:
    from pm25ml.sample.full_model_sampler import FullModelSampler


@inject
def _main(
    full_model_sampler: FullModelSampler = Provide[Pm25mlContainer.full_model_sampler],
) -> None:
    full_model_sampler.sample()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
