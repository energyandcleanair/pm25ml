"""A module for setting up the PM2.5 ML project using Dependency Injection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import arrow
import ee
import google.auth
from dependency_injector import containers, providers
from ee.featurecollection import FeatureCollection
from gcsfs import GCSFileSystem

from pm25ml.collectors.archive_storage import IngestArchiveStorage
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator
from pm25ml.collectors.collector import RawDataCollector
from pm25ml.collectors.gee.feature_planner import GriddedFeatureCollectionPlanner
from pm25ml.collectors.gee.gee_export_pipeline import GeePipelineConstructor
from pm25ml.collectors.gee.intermediate_storage import GeeIntermediateStorage
from pm25ml.collectors.grid import Grid, load_grid_from_files
from pm25ml.collectors.ned.ned_export_pipeline import NedPipelineConstructor
from pm25ml.collectors.pm25.data_source import CreaMeasurementsApiDataSource
from pm25ml.collectors.pm25.pm25_pipeline import Pm25MeasurementsPipelineConstructor
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.archive.combine_manager import MonthlyCombinerManager
from pm25ml.combiners.archive.combine_planner import CombinePlanner
from pm25ml.combiners.archive.combiner import ArchiveWideCombiner
from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.imputation.from_model.full_predict_controller import (
    FinalPredictionController,
)
from pm25ml.imputation.from_model.imputation_controller import (
    ImputationController,
)
from pm25ml.imputation.spatial.daily_spatial_interpolator import (
    DailySpatialInterpolator,
)
from pm25ml.imputation.spatial.spatial_imputation_manager import (
    SpatialImputationManager,
)
from pm25ml.logging import logger
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.sample.full_model_sampler import FullModelSampler
from pm25ml.sample.imputation_sampler import ImputationSamplerDefinition
from pm25ml.setup.ingestion.pipelines import define_pipelines
from pm25ml.setup.ingestion.pm25_filters import define_filters
from pm25ml.setup.models.feature_generator import generate_for_year
from pm25ml.setup.models.samplers import ImputationStep, define_samplers
from pm25ml.setup.models.training_full import build_full_model_ref
from pm25ml.setup.models.training_impute import build_model_ref
from pm25ml.setup.results.result_writers import define_result_writers
from pm25ml.setup.temporal_config import TemporalConfig
from pm25ml.training.full_model_pipeline import FullModelPipeline
from pm25ml.training.imputation_model_pipeline import ImputationModelPipeline
from pm25ml.training.model_storage import ModelStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    type BooleanSelector = Literal["true", "false"]

LOCAL_GRID_ZIP_PATH = Path("./assets/grid_india_10km_shapefiles.zip")
LOCAL_GRID_50KM_MAPPING_CSV_PATH = Path("./assets/grid_intersect_with_50km.csv")
LOCAL_GRID_REGION_PARQUET_PATH = Path("./assets/grid_region.parquet")

NO_OP = lambda x: x  # noqa: E731


def _boolean_selector_to_bool(selector: BooleanSelector) -> bool:
    """Convert a BooleanSelector to a boolean."""
    return selector == "true"


@contextmanager
def _init_gee(
    gcp_project: str,
) -> Generator[None, None, None]:
    logger.debug("Initializing GEE with project: %s", gcp_project)
    creds, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(project=gcp_project, credentials=creds)
    yield


def _load_india_grid_reference_asset(india_shapefile_asset: str) -> FeatureCollection:
    """
    Initialize the GeeIndiaGridReferenceResource with the GEE asset path.

    Args:
        india_shapefile_asset (str): The GEE asset path for the India shapefile.

    Returns:
        ee.FeatureCollection: The initialized FeatureCollection.

    """
    logger.debug("Loading India grid reference asset from: %s", india_shapefile_asset)
    gee_india_grid_reference = FeatureCollection(india_shapefile_asset)
    gee_india_grid_reference_size = gee_india_grid_reference.size().getInfo()
    if gee_india_grid_reference_size != VALID_COUNTRIES["india"]:
        msg = (
            f"Expected {VALID_COUNTRIES['india']} features in the GEE India grid, "
            f"but found {gee_india_grid_reference_size}."
        )
        raise ValueError(
            msg,
        )

    return gee_india_grid_reference


def _load_in_memory_grid() -> Grid:
    logger.debug("Loading in-memory grid from local zip file: %s", LOCAL_GRID_ZIP_PATH)
    return load_grid_from_files(
        path_to_shapefile_zip=LOCAL_GRID_ZIP_PATH,
        path_to_50km_csv=LOCAL_GRID_50KM_MAPPING_CSV_PATH,
        path_to_region_parquet=LOCAL_GRID_REGION_PARQUET_PATH,
    )


class DataArtifactProvider(containers.DeclarativeContainer):
    """
    Provides DataArtifact references for various stages of the PM2.5 ML project.

    This container defines the stages used in the project, allowing for easy access
    and management of data artifacts throughout the pipeline.
    """

    combined_stage = providers.Singleton(
        DataArtifactRef,
        stage="combined_monthly",
    )

    spatially_imputed_era5_stage = providers.Singleton(
        DataArtifactRef,
        stage="era5_spatially_imputed",
    )

    spatially_imputed_stage = providers.Singleton(
        DataArtifactRef,
        stage="combined_with_spatial_interpolation",
    )

    generated_features_stage = providers.Singleton(
        DataArtifactRef,
        stage="generated_features",
    )

    ml_imputer_sampled_super_stage = providers.Singleton(
        DataArtifactRef,
        stage="sampled",
    )

    ml_imputed_super_stage = providers.Singleton(
        DataArtifactRef,
        stage="imputed",
    )

    ml_full_model_sample_stage = providers.Singleton(
        DataArtifactRef,
        stage="full_model_sample",
    )

    final_prediction = providers.Singleton(
        DataArtifactRef,
        stage="final_prediction",
    )


class Pm25mlContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the PM2.5 ML project.

    This container manages the configuration and resources required for the project,
    """

    config = providers.Configuration(strict=True)

    data_artifacts_container = providers.Container(
        DataArtifactProvider,
    )

    temporal_config = providers.Singleton(
        TemporalConfig,
        start_date=config.start_month,
        end_date=config.end_month,
    )

    gee_auth = providers.Resource(
        _init_gee,
        gcp_project=config.gcp.gcp_project,
    )

    gee_india_grid_reference = providers.Callable(
        _load_india_grid_reference_asset,
        india_shapefile_asset=config.gcp.gee.india_shapefile_asset,
    )

    feature_planner = providers.Singleton(
        GriddedFeatureCollectionPlanner,
        grid=gee_india_grid_reference,
    )

    gcs_filesystem: providers.Provider[GCSFileSystem] = providers.Singleton(
        GCSFileSystem,
    )

    intermediate_storage = providers.Singleton(
        GeeIntermediateStorage,
        filesystem=gcs_filesystem,
        bucket=config.gcp.csv_bucket,
    )

    archive_storage = providers.Singleton(
        IngestArchiveStorage,
        filesystem=gcs_filesystem,
        destination_bucket=config.gcp.archive_bucket,
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
    )

    ned_pipeline_constructor = providers.Singleton(
        NedPipelineConstructor,
        archive_storage=archive_storage,
        grid=in_memory_grid,
    )

    pm25_data_source = providers.Singleton(
        CreaMeasurementsApiDataSource,
        temporal_config=temporal_config,
    )

    pm25_filters = providers.Singleton(
        define_filters,
    )

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
        destination_bucket=config.gcp.combined_bucket,
    )

    archived_wide_combiner = providers.Singleton(
        ArchiveWideCombiner,
        archive_storage=archive_storage,
        combined_storage=combined_storage,
        output_artifact=data_artifacts_container.combined_stage.provided,
    )

    monthly_combiner = providers.Singleton(
        MonthlyCombinerManager,
        combined_storage=combined_storage,
        archived_wide_combiner=archived_wide_combiner,
    )

    collector = providers.Singleton(
        RawDataCollector,
        metadata_validator=metadata_validator,
    )

    pipelines = providers.Singleton(
        define_pipelines,
        gee_pipeline_constructor=gee_pipeline_constructor,
        ned_pipeline_constructor=ned_pipeline_constructor,
        pm25_pipeline_constructor=pm25_pipeline_constructor,
        in_memory_grid=in_memory_grid,
        archive_storage=archive_storage,
        feature_planner=feature_planner,
        temporal_config=temporal_config,
    )

    combine_planner = providers.Singleton(
        CombinePlanner,
        temporal_config=temporal_config,
    )

    daily_spatial_interpolator = providers.Singleton(
        DailySpatialInterpolator,
        grid=in_memory_grid,
        value_column_regex_selector=config.spatial_computation_value_column_regex,
    )

    spatial_imputation_manager = providers.Singleton(
        SpatialImputationManager,
        combined_storage=combined_storage,
        spatial_imputer=daily_spatial_interpolator,
        temporal_config=temporal_config,
        input_data_artifact=data_artifacts_container.combined_stage.provided,
        output_data_artifact=data_artifacts_container.spatially_imputed_era5_stage.provided,
    )

    spatial_interpolation_recombiner = providers.Singleton(
        Recombiner,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        output_data_artifact=data_artifacts_container.spatially_imputed_stage.provided,
        max_workers=8,
    )

    feature_generator = providers.Singleton(
        FeatureGenerator,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        input_data_artifact=data_artifacts_container.spatially_imputed_stage.provided,
        output_data_artifact=data_artifacts_container.generated_features_stage.provided,
        generate_for_year=generate_for_year,
    )

    imputation_samplers = providers.Singleton(
        define_samplers,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        imputation_steps=config.imputation_steps,
        input_data_artifact=data_artifacts_container.generated_features_stage.provided,
        output_data_artifact=data_artifacts_container.ml_imputed_super_stage.provided,
    )

    extra_sampler = providers.Selector(
        config.take_mini_training_sample_selector,
        true=providers.Object(
            lambda x: x.gather_every(500),
        ),
        false=providers.Object(NO_OP),
    )

    model_store = providers.Singleton(
        ModelStorage,
        filesystem=gcs_filesystem,
        bucket_name=config.gcp.model_storage_bucket,
    )

    ml_model_def_factory = providers.Factory(
        build_model_ref,
        extra_sampler=extra_sampler,
        take_mini_training_sample=config.take_mini_training_sample_bool,
    )

    ml_model_defs = providers.Dict(
        {
            "aod": providers.Callable(ml_model_def_factory, ref="aod"),
            "no2": providers.Callable(ml_model_def_factory, ref="no2"),
            "co": providers.Callable(ml_model_def_factory, ref="co"),
        },
    )

    ml_model_trainer_factory = providers.Factory(
        lambda *, model_reference, combined_storage, model_store, n_jobs, input_data_artifact: ImputationModelPipeline(
            combined_storage=combined_storage,
            data_ref=model_reference,
            model_store=model_store,
            n_jobs=n_jobs,
            input_data_artifact=input_data_artifact.for_sub_artifact(
                model_reference.model_name,
            ),
        ),
        combined_storage=combined_storage,
        model_store=model_store,
        n_jobs=config.max_parallel_tasks,
        input_data_artifact=data_artifacts_container.ml_imputer_sampled_super_stage.provided,
    )

    imputer_recombiner = providers.Singleton(
        Recombiner,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        output_data_artifact=data_artifacts_container.ml_imputed_super_stage.provided,
        max_workers=4,
        force_recombine=True,
    )

    regression_model_imputer_controller = providers.Factory(
        ImputationController,
        model_store=model_store,
        temporal_config=temporal_config,
        combined_storage=combined_storage,
        model_refs=ml_model_defs,
        recombiner=imputer_recombiner,
        input_data_artifact=data_artifacts_container.generated_features_stage.provided,
        output_data_artifact=data_artifacts_container.ml_imputed_super_stage.provided,
    )

    full_model_sampler = providers.Singleton(
        FullModelSampler,
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        input_data_artifact=data_artifacts_container.ml_imputed_super_stage.provided,
        output_data_artifact=data_artifacts_container.ml_full_model_sample_stage.provided,
        column_name="pm25__pm25",
    )

    extra_sampler_full = providers.Selector(
        config.take_mini_training_sample_selector,
        true=providers.Object(
            lambda x: x.gather_every(10),
        ),
        false=providers.Object(NO_OP),
    )

    full_model_ref = providers.Singleton(
        build_full_model_ref,
        extra_sampler=extra_sampler_full,
        take_mini_training_sample=config.take_mini_training_sample_bool,
    )

    full_model_pipeline = providers.Singleton(
        FullModelPipeline,
        combined_storage=combined_storage,
        data_ref=full_model_ref,
        model_store=model_store,
        n_jobs=config.max_parallel_tasks,
        input_data_artifact=data_artifacts_container.ml_full_model_sample_stage.provided,
    )

    final_predict_controller = providers.Singleton(
        FinalPredictionController,
        model_store=model_store,
        temporal_config=temporal_config,
        combined_storage=combined_storage,
        model_ref=full_model_ref,
        input_data_artifact=data_artifacts_container.ml_imputed_super_stage.provided,
        output_data_artifact=data_artifacts_container.final_prediction.provided,
    )

    final_result_storage = providers.Singleton(
        FinalResultStorage,
        filesystem=gcs_filesystem,
        destination_bucket=config.gcp.final_result_bucket,
    )

    final_result_writers = providers.Singleton(
        define_result_writers,
        storage=final_result_storage,
    )


def init_dependencies_from_env() -> Pm25mlContainer:
    """
    Create a container instance with configuration loaded from environment variables.

    Returns:
        Container: An instance of the Container class with configuration set.

    """
    container = Pm25mlContainer()

    container.config.gcp.gcp_project.from_env("GCP_PROJECT")
    container.config.gcp.csv_bucket.from_env("CSV_BUCKET_NAME")
    container.config.gcp.archive_bucket.from_env("INGEST_ARCHIVE_BUCKET_NAME")
    container.config.gcp.combined_bucket.from_env("COMBINED_BUCKET_NAME")
    container.config.gcp.model_storage_bucket.from_env(
        "MODEL_STORAGE_BUCKET_NAME",
    )
    container.config.gcp.final_result_bucket.from_env("FINAL_RESULT_BUCKET_NAME")

    container.config.gcp.gee.india_shapefile_asset.from_env("INDIA_SHAPEFILE_ASSET")

    container.config.max_parallel_tasks.from_env(
        "MAX_PARALLEL_TASKS",
        as_=lambda x: int(x),
        default=str(os.cpu_count() or 1),
    )

    container.config.take_mini_training_sample_selector.from_value(
        _parse_bool_env_var(
            os.getenv("TAKE_MINI_TRAINING_SAMPLE") or "false",
        ),
    )
    container.config.take_mini_training_sample_bool.from_value(
        _boolean_selector_to_bool(
            container.config.take_mini_training_sample_selector(),
        ),
    )

    logger.info(
        f"Using local training: {container.config.take_mini_training_sample_selector()}"
    )

    container.config.start_month.from_env(
        "START_MONTH", as_=lambda x: arrow.get(x, "YYYY-MM-DD")
    )
    container.config.end_month.from_env(
        "END_MONTH", as_=lambda x: arrow.get(x, "YYYY-MM-DD")
    )

    container.config.spatial_computation_value_column_regex.from_env(
        "SPATIAL_COMPUTATION_VALUE_COLUMN_REGEX",
    )

    container.config.imputation_steps.from_value(
        [
            # AOD
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="modis_aod__Optical_Depth_055",
                    model_name="aod",
                    percentage_sample=0.03,
                ),
            ),
            # Tropomi NO2
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="s5p_no2__tropospheric_NO2_column_number_density",
                    model_name="no2",
                    percentage_sample=0.02,
                ),
            ),
            # Tropomi CO
            ImputationStep(
                imputation_sampler_definition=ImputationSamplerDefinition(
                    value_column="s5p_co__CO_column_number_density",
                    model_name="co",
                    percentage_sample=0.02,
                ),
            ),
        ],
    )

    container.init_resources()

    return container


def _parse_bool_env_var(value: str) -> BooleanSelector:
    result = str(value).strip().lower() in ("1", "true", "yes", "on")
    return "true" if result else "false"
