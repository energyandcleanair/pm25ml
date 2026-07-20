"""Script to impute missing data using a regression model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)

if TYPE_CHECKING:
    from pm25ml.imputation.from_model.imputation_controller import (
        ImputationController,
    )


@inject
def _main(
    regression_model_imputer_controller: ImputationController = Provide[
        Pm25mlContainer.regression_model_imputer_controller
    ],
) -> None:
    regression_model_imputer_controller.impute()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
