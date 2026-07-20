"""Application composition root for the PM2.5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import ee
import google.auth
from dependency_injector import containers, providers
from ee.featurecollection import FeatureCollection
from gcsfs import GCSFileSystem

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator
from pm25ml.collectors.collector import RawDataCollector
from pm25ml.collectors.end_month_selector import EndMonthCoordinator
from pm25ml.collectors.gee.feature_planner import GriddedFeatureCollectionPlanner
from pm25ml.collectors.gee.gee_export_pipeline import GeePipelineConstructor
from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.grid import Grid, load_grid_from_files
from pm25ml.collectors.ned.ned_export_pipeline import NedPipelineConstructor
from pm25ml.collectors.pm25.data_source import CreaMeasurementsApiDataSource
from pm25ml.collectors.pm25.pm25_pipeline import Pm25MeasurementsPipelineConstructor
from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager
from pm25ml.combiners.archive.combine_planner import CombinePlanner
from pm25ml.combiners.archive.combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.imputation.from_model.full_predict_controller import FinalPredictionController
from pm25ml.imputation.from_model.imputation_controller import ImputationController
from pm25ml.imputation.spatial.daily_spatial_interpolator import DailySpatialInterpolator
from pm25ml.imputation.spatial.spatial_imputation_manager import SpatialImputationManager
from pm25ml.logging import logger
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.sample.full_model_sampler import FullModelSampler
from pm25ml.sample.imputation_sampler import ImputationSamplerDefinition
from pm25ml.setup.end_month import EndMonthStore
from pm25ml.setup.pipelines import define_pipelines
from pm25ml.setup.pm25_filters import define_filters
from pm25ml.setup.result_writers import define_result_writers, define_stats_writers
from pm25ml.setup.samplers import ImputationStep, define_samplers
from pm25ml.setup.settings import PipelineSettings
from pm25ml.setup.training import build_model_ref
from pm25ml.setup.training_full import build_full_model_ref
from pm25ml.training.full_model_pipeline import FullModelPipeline
from pm25ml.training.imputation_model_pipeline import ImputationModelPipeline
from pm25ml.training.model_storage import ModelStorage

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from pm25ml.model_reference import ImputationModelReference
    from pm25ml.training.types import ModelName

T = TypeVar("T")


@dataclass(frozen=True)
class DataArtifacts:
    """Named data artifacts exchanged between pipeline stages."""

    combined: DataArtifactRef
    spatially_imputed_era5: DataArtifactRef
    spatially_imputed: DataArtifactRef
    generated_features: DataArtifactRef
    sampled: DataArtifactRef
    imputed: DataArtifactRef
    full_model_sample: DataArtifactRef
    final_prediction: DataArtifactRef


def _identity(value: T) -> T:
    return value


def initialize_gee(gcp_project: str) -> None:
    """Authenticate with and initialize Earth Engine for the current process."""
    logger.debug("Initializing GEE with project: %s", gcp_project)
    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(project=gcp_project, credentials=credentials)


def _profile_asset_dir(profile_id: str) -> Path:
    return Path("./assets") / profile_id


def _load_grid_reference_asset(
    grid_asset_path: str,
    expected_grid_cell_count: int,
    profile_id: str,
) -> FeatureCollection:
    logger.debug("Loading grid reference asset for %s from: %s", profile_id, grid_asset_path)
    grid_reference = FeatureCollection(grid_asset_path)
    grid_reference_size = grid_reference.size().getInfo()
    if grid_reference_size != expected_grid_cell_count:
        msg = (
            f"Expected {expected_grid_cell_count} features in the GEE grid for {profile_id}, "
            f"but found {grid_reference_size}."
        )
        raise ValueError(msg)
    return grid_reference


def _load_in_memory_grid(profile_id: str) -> Grid:
    asset_dir = _profile_asset_dir(profile_id)
    grid_zip_path = asset_dir / "grid_10km_shapefiles.zip"
    grid_50km_mapping_csv_path = asset_dir / "grid_intersect_with_50km.csv"
    grid_region_parquet_path = asset_dir / "grid_region.parquet"
    logger.debug("Loading in-memory grid for %s from: %s", profile_id, grid_zip_path)
    return load_grid_from_files(
        path_to_shapefile_zip=grid_zip_path,
        path_to_50km_csv=grid_50km_mapping_csv_path,
        path_to_region_parquet=grid_region_parquet_path,
    )


def _build_data_artifacts(country: str) -> DataArtifacts:
    return DataArtifacts(
        combined=DataArtifactRef(stage="combined_monthly", country=country),
        spatially_imputed_era5=DataArtifactRef(stage="era5_spatially_imputed", country=country),
        spatially_imputed=DataArtifactRef(
            stage="combined_with_spatial_interpolation",
            country=country,
        ),
        generated_features=DataArtifactRef(stage="generated_features", country=country),
        sampled=DataArtifactRef(stage="sampled", country=country),
        imputed=DataArtifactRef(stage="imputed", country=country),
        full_model_sample=DataArtifactRef(stage="full_model_sample", country=country),
        final_prediction=DataArtifactRef(stage="final_prediction", country=country),
    )


def _define_imputation_steps() -> tuple[ImputationStep, ...]:
    return (
        ImputationStep(
            ImputationSamplerDefinition(
                value_column="modis_aod__Optical_Depth_055",
                model_name="aod",
                percentage_sample=0.03,
            ),
        ),
        ImputationStep(
            ImputationSamplerDefinition(
                value_column="s5p_no2__tropospheric_NO2_column_number_density",
                model_name="no2",
                percentage_sample=0.02,
            ),
        ),
        ImputationStep(
            ImputationSamplerDefinition(
                value_column="s5p_co__CO_column_number_density",
                model_name="co",
                percentage_sample=0.02,
            ),
        ),
    )


def _sample_every(frame: pl.LazyFrame, interval: int) -> pl.LazyFrame:
    return frame.gather_every(interval)


def _unchanged(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame


def _build_extra_sampler(
    *,
    take_mini_training_sample: bool,
    interval: int,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    if not take_mini_training_sample:
        return _unchanged
    return lambda frame: _sample_every(frame, interval)


def _build_model_definitions(
    *,
    extra_sampler: Callable[[pl.LazyFrame], pl.LazyFrame],
    take_mini_training_sample: bool,
) -> dict[ModelName, ImputationModelReference]:
    model_names: tuple[ModelName, ...] = ("aod", "no2", "co")
    return {
        model_name: build_model_ref(
            ref=model_name,
            extra_sampler=extra_sampler,
            take_mini_training_sample=take_mini_training_sample,
        )
        for model_name in model_names
    }


def _build_imputation_model_pipeline(  # noqa: PLR0913
    *,
    model_reference: ImputationModelReference,
    combined_storage: CombinedStorage,
    model_store: ModelStorage,
    model_run_ref: str,
    n_jobs: int,
    input_data_artifact: DataArtifactRef,
) -> ImputationModelPipeline:
    return ImputationModelPipeline(
        combined_storage=combined_storage,
        data_ref=model_reference,
        model_store=model_store,
        model_run_ref=model_run_ref,
        n_jobs=n_jobs,
        input_data_artifact=input_data_artifact.for_sub_artifact(model_reference.model_name),
    )


def _final_output_path(profile_id: str, model_run_ref: str) -> str:
    return f"country={profile_id}/run={model_run_ref}"


class Pm25mlContainer(containers.DeclarativeContainer):
    """Compose reusable application services from static typed settings."""

    settings = providers.Dependency(instance_of=PipelineSettings)

    data_artifacts = providers.Singleton(
        _build_data_artifacts,
        country=settings.provided.profile_id,
    )
    model_run_ref = providers.Singleton(_identity, settings.provided.model_run_ref)
    gee_auth = providers.Singleton(initialize_gee, gcp_project=settings.provided.gcp_project)
    gee_grid_reference = providers.Callable(
        _load_grid_reference_asset,
        grid_asset_path=settings.provided.grid_asset_path,
        expected_grid_cell_count=settings.provided.grid_cell_count,
        profile_id=settings.provided.profile_id,
    )
    feature_planner = providers.Singleton(
        GriddedFeatureCollectionPlanner,
        grid=gee_grid_reference,
    )
    gcs_filesystem: providers.Provider[GCSFileSystem] = providers.Singleton(GCSFileSystem)
    intermediate_storage = providers.Singleton(
        GeeIntermediateStorage,
        filesystem=gcs_filesystem,
        bucket=settings.provided.csv_bucket,
    )
    archive_storage = providers.Singleton(
        IngestArchiveStorage,
        filesystem=gcs_filesystem,
        destination_bucket=settings.provided.archive_bucket,
    )
    metadata_validator = providers.Singleton(
        ArchivedFileValidator,
        archive_storage=archive_storage,
    )
    gee_pipeline_constructor = providers.Singleton(
        GeePipelineConstructor,
        archive_storage=archive_storage,
        intermediate_storage=intermediate_storage,
    )
    in_memory_grid = providers.Singleton(
        _load_in_memory_grid,
        profile_id=settings.provided.profile_id,
    )
    ned_pipeline_constructor = providers.Singleton(
        NedPipelineConstructor,
        archive_storage=archive_storage,
        grid=in_memory_grid,
    )
    pm25_data_source = providers.Singleton(
        CreaMeasurementsApiDataSource,
        source_ids=settings.provided.pm25_source_ids,
    )
    pm25_filters = providers.Singleton(define_filters)
    pm25_pipeline_constructor = providers.Singleton(
        Pm25MeasurementsPipelineConstructor,
        in_memory_grid=in_memory_grid,
        crea_ds=pm25_data_source,
        archive_storage=archive_storage,
        filters=pm25_filters,
    )
    combined_storage = providers.Singleton(
        CombinedStorage,
        filesystem=gcs_filesystem,
        destination_bucket=settings.provided.combined_bucket,
        profile_id=settings.provided.profile_id,
    )
    end_month_store = providers.Singleton(
        EndMonthStore,
        filesystem=gcs_filesystem,
        bucket=settings.provided.combined_bucket,
        profile_id=settings.provided.profile_id,
        model_run_ref=model_run_ref,
    )
    archived_wide_combiner = providers.Singleton(
        ArchiveWideCombiner,
        archive_storage=archive_storage,
        combined_storage=combined_storage,
        output_artifact=data_artifacts.provided.combined,
    )
    monthly_combiner = providers.Singleton(
        MonthlyCombinerManager,
        combined_storage=combined_storage,
        archived_wide_combiner=archived_wide_combiner,
    )
    collector = providers.Singleton(RawDataCollector, metadata_validator=metadata_validator)
    end_month_coordinator = providers.Singleton(
        EndMonthCoordinator,
        collector=collector,
        store=end_month_store,
    )
    pipelines = providers.Factory(
        define_pipelines,
        gee_pipeline_constructor=gee_pipeline_constructor,
        ned_pipeline_constructor=ned_pipeline_constructor,
        pm25_pipeline_constructor=pm25_pipeline_constructor,
        in_memory_grid=in_memory_grid,
        archive_storage=archive_storage,
        feature_planner=feature_planner,
        profile_id=settings.provided.profile_id,
    )
    combine_planner = providers.Singleton(
        CombinePlanner,
        n_grid_cells=settings.provided.grid_cell_count,
    )
    daily_spatial_interpolator = providers.Singleton(
        DailySpatialInterpolator,
        grid=in_memory_grid,
        value_column_regex_selector=settings.provided.spatial_computation_value_column_regex,
    )
    spatial_imputation_manager = providers.Singleton(
        SpatialImputationManager,
        combined_storage=combined_storage,
        spatial_imputer=daily_spatial_interpolator,
        input_data_artifact=data_artifacts.provided.combined,
        output_data_artifact=data_artifacts.provided.spatially_imputed_era5,
        n_grid_cells=settings.provided.grid_cell_count,
    )
    spatial_interpolation_recombiner = providers.Singleton(
        Recombiner,
        combined_storage=combined_storage,
        output_data_artifact=data_artifacts.provided.spatially_imputed,
        max_workers=8,
        n_grid_cells=settings.provided.grid_cell_count,
    )
    feature_generator = providers.Singleton(
        FeatureGenerator,
        combined_storage=combined_storage,
        input_data_artifact=data_artifacts.provided.spatially_imputed,
        output_data_artifact=data_artifacts.provided.generated_features,
    )
    imputation_samplers = providers.Singleton(
        define_samplers,
        combined_storage=combined_storage,
        imputation_steps=providers.Callable(_define_imputation_steps),
        input_data_artifact=data_artifacts.provided.generated_features,
        output_data_artifact=data_artifacts.provided.sampled,
    )
    model_store = providers.Singleton(
        ModelStorage,
        filesystem=gcs_filesystem,
        bucket_name=settings.provided.model_storage_bucket,
        profile_id=settings.provided.profile_id,
    )
    extra_sampler = providers.Singleton(
        _build_extra_sampler,
        take_mini_training_sample=settings.provided.take_mini_training_sample,
        interval=500,
    )
    ml_model_defs = providers.Singleton(
        _build_model_definitions,
        extra_sampler=extra_sampler,
        take_mini_training_sample=settings.provided.take_mini_training_sample,
    )
    ml_model_trainer_factory = providers.Factory(
        _build_imputation_model_pipeline,
        combined_storage=combined_storage,
        model_store=model_store,
        model_run_ref=model_run_ref,
        n_jobs=settings.provided.max_parallel_tasks,
        input_data_artifact=data_artifacts.provided.sampled,
    )
    imputer_recombiner = providers.Singleton(
        Recombiner,
        combined_storage=combined_storage,
        output_data_artifact=data_artifacts.provided.imputed,
        max_workers=4,
        n_grid_cells=settings.provided.grid_cell_count,
        force_recombine=True,
    )
    regression_model_imputer_controller = providers.Factory(
        ImputationController,
        model_store=model_store,
        model_run_ref=model_run_ref,
        combined_storage=combined_storage,
        model_refs=ml_model_defs,
        recombiner=imputer_recombiner,
        input_data_artifact=data_artifacts.provided.generated_features,
        output_data_artifact=data_artifacts.provided.imputed,
    )
    full_model_sampler = providers.Singleton(
        FullModelSampler,
        combined_storage=combined_storage,
        input_data_artifact=data_artifacts.provided.imputed,
        output_data_artifact=data_artifacts.provided.full_model_sample,
        column_name="pm25__pm25",
    )
    extra_sampler_full = providers.Singleton(
        _build_extra_sampler,
        take_mini_training_sample=settings.provided.take_mini_training_sample,
        interval=10,
    )
    full_model_ref = providers.Singleton(
        build_full_model_ref,
        extra_sampler=extra_sampler_full,
        take_mini_training_sample=settings.provided.take_mini_training_sample,
    )
    full_model_pipeline = providers.Singleton(
        FullModelPipeline,
        combined_storage=combined_storage,
        data_ref=full_model_ref,
        model_store=model_store,
        model_run_ref=model_run_ref,
        n_jobs=settings.provided.max_parallel_tasks,
        input_data_artifact=data_artifacts.provided.full_model_sample,
    )
    final_predict_controller = providers.Singleton(
        FinalPredictionController,
        model_store=model_store,
        model_run_ref=model_run_ref,
        combined_storage=combined_storage,
        model_ref=full_model_ref,
        input_data_artifact=data_artifacts.provided.imputed,
        output_data_artifact=data_artifacts.provided.final_prediction,
    )
    final_result_storage = providers.Singleton(
        FinalResultStorage,
        filesystem=gcs_filesystem,
        destination_bucket=settings.provided.final_result_bucket,
        output_path=providers.Callable(
            _final_output_path,
            profile_id=settings.provided.profile_id,
            model_run_ref=model_run_ref,
        ),
    )
    final_result_writers = providers.Singleton(
        define_result_writers,
        storage=final_result_storage,
        model_run_ref=model_run_ref,
    )
    final_stats_writers = providers.Singleton(
        define_stats_writers,
        storage=final_result_storage,
        model_run_ref=model_run_ref,
    )


def _container(settings: PipelineSettings) -> Pm25mlContainer:
    container = Pm25mlContainer()
    container.settings.override(providers.Object(settings))
    return container


def init_dependencies_from_env() -> Pm25mlContainer:
    """Build static application dependencies from environment settings."""
    settings = PipelineSettings.from_env()
    container = _container(settings)
    logger.info("Using MODEL_RUN_REF: %s", settings.model_run_ref)
    return container
