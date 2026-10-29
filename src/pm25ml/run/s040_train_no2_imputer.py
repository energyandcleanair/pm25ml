"""Script to train the AOD model."""

from pm25ml.model_reference import ImputationModelReference
from pm25ml.run._stage_report import report_production_stage
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.injection_type_helpers import ModelTrainerFactory

STAGE_TITLE = "train_no2"


def _main(
    model_reference: ImputationModelReference,
    ml_model_trainer_factory: ModelTrainerFactory,
) -> None:
    no2_trainer = ml_model_trainer_factory(model_reference=model_reference)

    no2_trainer.train_model()


if __name__ == "__main__":
    with report_production_stage(STAGE_TITLE):
        container = init_dependencies_from_env()
        _main(container.ml_model_defs()["no2"], container.ml_model_trainer_factory)
