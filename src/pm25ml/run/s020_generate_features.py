"""Entry point for generating features in the PM2.5 ML project."""

from dependency_injector.wiring import Provide, inject

from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env


@inject
def _main(
    feature_generator: FeatureGenerator = Provide[Pm25mlContainer.feature_generator],
) -> None:
    feature_generator.generate()
    logger.info("Feature generation completed successfully.")


if __name__ == "__main__":
    container = init_dependencies_from_env()
    container.wire(modules=[__name__])
    _main()
