"""Script to train the AOD model."""

from dependency_injector.wiring import Provide, Provider, inject

from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)
from pm25ml.setup.types import ModelTrainerFactory


@inject
def _main(
    model_defs: dict = Provide[Pm25mlContainer.ml_model_defs],
    ml_model_trainer_factory: ModelTrainerFactory = Provider[
        Pm25mlContainer.ml_model_trainer_factory
    ],
) -> None:
    no2_trainer = ml_model_trainer_factory(model_reference=model_defs["no2"])

    no2_trainer.train_model()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
