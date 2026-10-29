"""Script to train the full model."""

from pm25ml.results.cv_summary_result_writer import format_performance_summary
from pm25ml.setup.dependency_injection import init_dependencies_from_env
from pm25ml.setup.logging_injection import init_logging_from_env
from pm25ml.training.full_model_pipeline import MODEL_NAME, FullModelPipeline

STAGE_TITLE = "full_model_train"


def _main(
    full_model_trainer: FullModelPipeline,
) -> None:
    full_model_trainer.train_model()


if __name__ == "__main__":
    logging = init_logging_from_env()
    slack = logging.production_slack()
    with slack.report_stage(STAGE_TITLE):
        container = init_dependencies_from_env()
        _main(container.full_model_pipeline())
        stats = container.model_store().load_validation_metadata(
            MODEL_NAME,
            container.model_run_ref(),
        )
        slack.post_info(format_performance_summary(MODEL_NAME, stats))
