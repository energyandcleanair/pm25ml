"""Contract tests for dependency injection stage wiring."""

from __future__ import annotations

from dependency_injector import providers

from pm25ml.setup.dependency_injection import Pm25mlContainer


def _stage(provider: providers.ProvidedInstance) -> str:
    stage = provider.provides.kwargs["stage"]
    assert isinstance(stage, str)
    return stage


def test__stage_wiring__sampling_and_training__share_sampled_stage() -> None:
    """Sampled data must be produced by sampling and consumed by imputation training."""
    container = Pm25mlContainer()

    sampler_input = container.imputation_samplers.kwargs["input_data_artifact"]
    sampler_output = container.imputation_samplers.kwargs["output_data_artifact"]
    trainer_input = container.ml_model_trainer_factory.kwargs["input_data_artifact"]

    assert _stage(sampler_input) == "generated_features"
    assert _stage(sampler_output) == "sampled"
    assert _stage(trainer_input) == "sampled"


def test__stage_wiring__imputation_and_full_model__use_imputed_stage() -> None:
    """Model imputation output must feed later full-model sampling and prediction."""
    container = Pm25mlContainer()

    imputer_input = container.regression_model_imputer_controller.kwargs["input_data_artifact"]
    imputer_output = container.regression_model_imputer_controller.kwargs["output_data_artifact"]
    full_model_sampler_input = container.full_model_sampler.kwargs["input_data_artifact"]
    final_predict_input = container.final_predict_controller.kwargs["input_data_artifact"]

    assert _stage(imputer_input) == "generated_features"
    assert _stage(imputer_output) == "imputed"
    assert _stage(full_model_sampler_input) == "imputed"
    assert _stage(final_predict_input) == "imputed"


def test__stage_wiring__final_stages__are_connected() -> None:
    """Full-model sample output should feed full-model training and final output stage."""
    container = Pm25mlContainer()

    full_model_sampler_output = container.full_model_sampler.kwargs["output_data_artifact"]
    full_model_training_input = container.full_model_pipeline.kwargs["input_data_artifact"]
    final_predict_output = container.final_predict_controller.kwargs["output_data_artifact"]

    assert _stage(full_model_sampler_output) == "full_model_sample"
    assert _stage(full_model_training_input) == "full_model_sample"
    assert _stage(final_predict_output) == "final_prediction"
