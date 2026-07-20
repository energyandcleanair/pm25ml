"""Script to predict data for the full model."""

from dependency_injector.wiring import Provide, inject

from pm25ml.imputation.from_model.full_predict_controller import FinalPredictionController
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env


@inject
def _main(
    prediction_controller: FinalPredictionController = Provide[
        Pm25mlContainer.final_predict_controller
    ],
) -> None:
    prediction_controller.predict()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
