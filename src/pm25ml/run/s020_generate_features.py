"""Entry point for generating features in the PM2.5 ML project."""

from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest


def _main(feature_generator: FeatureGenerator, temporal_config: TemporalConfig) -> None:
    feature_generator.generate(temporal_config)
    logger.info("Feature generation completed successfully.")


if __name__ == "__main__":
    container = init_dependencies_from_env()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(container.feature_generator(), temporal_config)
