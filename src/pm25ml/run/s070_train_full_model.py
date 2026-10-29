"""Script to train the full model."""

from pm25ml.run._stage_report import report_production_stage
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.training.full_model_pipeline import FullModelPipeline

STAGE_TITLE = "full_model_train"


def _main(
    full_model_trainer: FullModelPipeline,
) -> None:
    full_model_trainer.train_model()


if __name__ == "__main__":
    with report_production_stage(STAGE_TITLE):
        container = init_dependencies_from_env()
        _main(container.full_model_pipeline())
