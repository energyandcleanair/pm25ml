"""Tests for typed environment settings."""

import re

import arrow
import pytest

from pm25ml.setup.settings import PipelineSettings, PreflightSettings, TemporalConfigRequest


def _environment() -> dict[str, str]:
    return {
        "GCP_PROJECT": "project",
        "CSV_BUCKET_NAME": "csv",
        "GEE_STAGING_BUCKET_NAME": "staging",
        "INGEST_ARCHIVE_BUCKET_NAME": "archive",
        "COMBINED_BUCKET_NAME": "combined",
        "MODEL_STORAGE_BUCKET_NAME": "models",
        "FINAL_RESULT_BUCKET_NAME": "results",
        "GEE_GRID_ASSET_PATH": "projects/project/assets/grid",
        "PIPELINE_PROFILE": "india",
        "PROFILE_GRID_CELL_COUNT": "33074",
        "START_MONTH": "2024-01-01",
        "SPATIAL_COMPUTATION_VALUE_COLUMN_REGEX": "^era5_land__.*$",
        "MODEL_RUN_REF": " run-1 ",
    }


def test__pipeline_settings__parses_values_and_defaults() -> None:
    settings = PipelineSettings.from_env(_environment())

    assert settings.profile_id == "india"
    assert settings.grid_cell_count == 33074
    assert settings.pm25_source_ids == ("cpcb",)
    assert settings.model_run_ref == "run-1"
    assert not settings.take_mini_training_sample


def test__pipeline_settings__parses_optional_values() -> None:
    environment = _environment() | {
        "PM25_SOURCE_IDS": "cpcb, embassy",
        "MAX_PARALLEL_TASKS": "7",
        "TAKE_MINI_TRAINING_SAMPLE": "yes",
        "END_MONTH": "2024-06-30",
        "MAX_DATA_LAG_MONTHS": "0",
    }

    settings = PipelineSettings.from_env(environment)

    assert settings.pm25_source_ids == ("cpcb", "embassy")
    assert settings.max_parallel_tasks == 7
    assert settings.take_mini_training_sample


def test__temporal_config_request__parses_values_and_defaults() -> None:
    default_request = TemporalConfigRequest.from_env(_environment())
    explicit_request = TemporalConfigRequest.from_env(
        _environment() | {"END_MONTH": "2024-06-30", "MAX_DATA_LAG_MONTHS": "0"},
    )

    assert default_request.start_month == arrow.get("2024-01-01")
    assert default_request.explicit_end_month is None
    assert default_request.max_data_lag_months == 3
    assert explicit_request.explicit_end_month == arrow.get("2024-06-30")
    assert explicit_request.max_data_lag_months == 0


def test__temporal_config_request__requires_start_month() -> None:
    environment = _environment()
    del environment["START_MONTH"]

    with pytest.raises(ValueError, match="START_MONTH"):
        TemporalConfigRequest.from_env(environment)


def test__pipeline_settings__generates_missing_model_run_reference() -> None:
    environment = _environment()
    del environment["MODEL_RUN_REF"]

    settings = PipelineSettings.from_env(environment)

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\+\d{2}-\d{2}-\d{2}", settings.model_run_ref)


def test__pipeline_settings__missing_required_value__names_variable() -> None:
    environment = _environment()
    del environment["COMBINED_BUCKET_NAME"]

    with pytest.raises(ValueError, match="COMBINED_BUCKET_NAME"):
        PipelineSettings.from_env(environment)


@pytest.mark.parametrize(
    ("name", "value"),
    [("PROFILE_GRID_CELL_COUNT", "0"), ("MAX_PARALLEL_TASKS", "-1")],
)
def test__pipeline_settings__non_positive_counts__raise(name: str, value: str) -> None:
    environment = _environment() | {name: value}

    with pytest.raises(ValueError, match=name):
        PipelineSettings.from_env(environment)


def test__preflight_settings__requires_only_bootstrap_values() -> None:
    settings = PreflightSettings.from_env(_environment())

    assert settings.gcp_project == "project"
    assert settings.gee_staging_bucket == "staging"
    assert settings.grid_cell_count == 33074
