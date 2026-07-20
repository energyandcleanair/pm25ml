"""Script to train the full model."""

from dependency_injector.wiring import Provide, inject

from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env
from pm25ml.training.full_model_pipeline import FullModelPipeline


@inject
def _main(
    full_model_trainer: FullModelPipeline = Provide[Pm25mlContainer.full_model_pipeline,],
) -> None:
    full_model_trainer.train_model()


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
