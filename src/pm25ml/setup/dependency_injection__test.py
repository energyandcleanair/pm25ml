"""Behavioral contract tests for the dependency composition root."""

from unittest.mock import Mock, patch

from dependency_injector import providers

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.collectors.end_month_selector import EndMonthCoordinator
from pm25ml.setup.dependency_injection import Pm25mlContainer, init_dependencies_from_env
from pm25ml.setup.end_month import EndMonthStore
from pm25ml.setup.settings import PipelineSettings
from pm25ml.training.model_storage import ModelStorage


def _settings() -> PipelineSettings:
    return PipelineSettings(
        gcp_project="project",
        csv_bucket="csv",
        archive_bucket="archive",
        combined_bucket="combined",
        model_storage_bucket="models",
        final_result_bucket="results",
        grid_asset_path="grid",
        profile_id="india",
        grid_cell_count=10,
        pm25_source_ids=("cpcb",),
        max_parallel_tasks=2,
        take_mini_training_sample=False,
        spatial_computation_value_column_regex="^era5",
        model_run_ref="run-1",
    )


def _container() -> Pm25mlContainer:
    container = Pm25mlContainer()
    container.settings.override(providers.Object(_settings()))
    container.combined_storage.override(providers.Object(Mock(spec=CombinedStorage)))
    return container


def test__data_artifacts__define_the_pipeline_stage_contract() -> None:
    artifacts = _container().data_artifacts()

    assert artifacts.combined.stage == "combined_monthly"
    assert artifacts.spatially_imputed_era5.stage == "era5_spatially_imputed"
    assert artifacts.spatially_imputed.stage == "combined_with_spatial_interpolation"
    assert artifacts.generated_features.stage == "generated_features"
    assert artifacts.sampled.stage == "sampled"
    assert artifacts.imputed.stage == "imputed"
    assert artifacts.full_model_sample.stage == "full_model_sample"
    assert artifacts.final_prediction.stage == "final_prediction"
    assert {artifact.country for artifact in artifacts.__dict__.values()} == {"india"}


def test__initialization__does_not_resolve_temporal_or_external_resources() -> None:
    with (
        patch.object(PipelineSettings, "from_env", return_value=_settings()),
        patch.object(EndMonthStore, "read_required") as read_required,
        patch.object(EndMonthCoordinator, "resolve") as resolve,
        patch("pm25ml.setup.dependency_injection.initialize_gee") as initialize_gee,
    ):
        container = init_dependencies_from_env()

    assert container.settings() == _settings()
    read_required.assert_not_called()
    resolve.assert_not_called()
    initialize_gee.assert_not_called()


def test__feature_generator__uses_adjacent_artifacts_without_temporal_state() -> None:
    container = _container()

    generator = container.feature_generator()

    assert not hasattr(container, "temporal_config")
    assert not hasattr(generator, "temporal_config")
    assert generator.input_data_artifact == container.data_artifacts().spatially_imputed
    assert generator.output_data_artifact == container.data_artifacts().generated_features


def test__full_model_sampler__connects_imputation_to_training_sample() -> None:
    container = _container()

    sampler = container.full_model_sampler()

    assert sampler.input_data_artifact == container.data_artifacts().imputed
    assert sampler.output_data_artifact == container.data_artifacts().full_model_sample


def test__run_reference__comes_from_typed_settings() -> None:
    container = _container()

    assert container.model_run_ref() == "run-1"


def test__training_and_prediction__share_artifacts_and_run_reference() -> None:
    container = _container()
    container.model_store.override(providers.Object(Mock(spec=ModelStorage)))

    trainer = container.ml_model_trainer_factory(
        model_reference=container.ml_model_defs()["aod"],
    )
    full_model_trainer = container.full_model_pipeline()
    predictor = container.final_predict_controller()

    assert trainer.input_data_artifact.stage == "sampled+aod"
    assert full_model_trainer.input_data_artifact == container.data_artifacts().full_model_sample
    assert predictor.input_data_artifact == container.data_artifacts().imputed
    assert predictor.output_data_artifact == container.data_artifacts().final_prediction
    assert {trainer.model_run_ref, full_model_trainer.model_run_ref, predictor.model_run_ref} == {
        "run-1",
    }
