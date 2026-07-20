"""Script to predict data for the full model."""

from pm25ml.imputation.from_model.full_predict_controller import FinalPredictionController
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.end_month import load_persisted_temporal_config
from pm25ml.setup.settings import TemporalConfigRequest


def _main(
    prediction_controller: FinalPredictionController,
    temporal_config: TemporalConfig,
) -> None:
    prediction_controller.predict(temporal_config)


if __name__ == "__main__":
    container = init_dependencies_from_env()
    temporal_config = load_persisted_temporal_config(
        TemporalConfigRequest.from_env(),
        container.end_month_store(),
    )
    _main(container.final_predict_controller(), temporal_config)
