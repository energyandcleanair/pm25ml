"""Script to train the AOD model."""

from pm25ml.model_reference import ImputationModelReference
from pm25ml.results.cv_summary_result_writer import format_performance_summary
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.injection_type_helpers import ModelTrainerFactory
from pm25ml.setup.logging_injection import init_logging_from_env

STAGE_TITLE = "train_no2"


def _main(
    model_reference: ImputationModelReference,
    ml_model_trainer_factory: ModelTrainerFactory,
) -> None:
    no2_trainer = ml_model_trainer_factory(model_reference=model_reference)

    no2_trainer.train_model()


if __name__ == "__main__":
    logging = init_logging_from_env()
    slack = logging.production_slack()
    with slack.report_stage(STAGE_TITLE):
        container = init_dependencies_from_env()
        _main(container.ml_model_defs()["no2"], container.ml_model_trainer_factory)
        stats = container.model_store().load_validation_metadata(
            "no2",
            container.model_run_ref(),
        )
        slack.post_info(format_performance_summary("no2", stats))
