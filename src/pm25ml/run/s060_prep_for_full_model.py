"""Script to impute missing data using a regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from pm25ml.sample.full_model_sampler import FullModelSampler
    from pm25ml.setup.date_params import TemporalConfig


def _main(
    full_model_sampler: FullModelSampler,
    temporal_config: TemporalConfig,
) -> None:
    full_model_sampler.sample(temporal_config)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(container.full_model_sampler(), temporal_config)
