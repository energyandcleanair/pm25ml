"""Script to impute missing data using a regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest

if TYPE_CHECKING:
    from pm25ml.imputation.from_model.imputation_controller import (
        ImputationController,
    )
    from pm25ml.setup.date_params import TemporalConfig


def _main(
    regression_model_imputer_controller: ImputationController,
    temporal_config: TemporalConfig,
) -> None:
    regression_model_imputer_controller.impute(temporal_config)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(container.regression_model_imputer_controller(), temporal_config)
