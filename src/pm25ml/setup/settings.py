"""Typed application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import arrow
from arrow import Arrow

from pm25ml.setup.end_month import parse_max_data_lag_months

if TYPE_CHECKING:
    from collections.abc import Mapping


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return value


def _parse_positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return parsed


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_run_ref(value: str | None) -> str:
    if value is None or not value.strip():
        return arrow.utcnow().format("YYYY-MM-DD+HH-mm-ss")
    return value.strip()


@dataclass(frozen=True)
class PreflightSettings:
    """Settings needed to create and validate the Earth Engine grid asset."""

    gcp_project: str
    gee_staging_bucket: str
    grid_asset_path: str
    profile_id: str
    grid_cell_count: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PreflightSettings:
        """Load preflight settings from an environment mapping."""
        values = os.environ if environ is None else environ
        return cls(
            gcp_project=_required(values, "GCP_PROJECT"),
            gee_staging_bucket=_required(values, "GEE_STAGING_BUCKET_NAME"),
            grid_asset_path=_required(values, "GEE_GRID_ASSET_PATH"),
            profile_id=_required(values, "PIPELINE_PROFILE"),
            grid_cell_count=_parse_positive_int(
                _required(values, "PROFILE_GRID_CELL_COUNT"),
                "PROFILE_GRID_CELL_COUNT",
            ),
        )


@dataclass(frozen=True)
class PipelineSettings:
    """Validated settings shared by the pipeline dependency graph."""

    gcp_project: str
    csv_bucket: str
    archive_bucket: str
    combined_bucket: str
    model_storage_bucket: str
    final_result_bucket: str
    grid_asset_path: str
    profile_id: str
    grid_cell_count: int
    pm25_source_ids: tuple[str, ...]
    max_parallel_tasks: int
    take_mini_training_sample: bool
    spatial_computation_value_column_regex: str
    model_run_ref: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PipelineSettings:
        """Load and validate pipeline settings from an environment mapping."""
        values = os.environ if environ is None else environ
        source_ids = tuple(
            part.strip()
            for part in values.get("PM25_SOURCE_IDS", "cpcb").split(",")
            if part.strip()
        )
        if not source_ids:
            msg = "PM25_SOURCE_IDS must contain at least one source ID"
            raise ValueError(msg)

        return cls(
            gcp_project=_required(values, "GCP_PROJECT"),
            csv_bucket=_required(values, "CSV_BUCKET_NAME"),
            archive_bucket=_required(values, "INGEST_ARCHIVE_BUCKET_NAME"),
            combined_bucket=_required(values, "COMBINED_BUCKET_NAME"),
            model_storage_bucket=_required(values, "MODEL_STORAGE_BUCKET_NAME"),
            final_result_bucket=_required(values, "FINAL_RESULT_BUCKET_NAME"),
            grid_asset_path=_required(values, "GEE_GRID_ASSET_PATH"),
            profile_id=_required(values, "PIPELINE_PROFILE"),
            grid_cell_count=_parse_positive_int(
                _required(values, "PROFILE_GRID_CELL_COUNT"),
                "PROFILE_GRID_CELL_COUNT",
            ),
            pm25_source_ids=source_ids,
            max_parallel_tasks=_parse_positive_int(
                values.get("MAX_PARALLEL_TASKS", str(os.cpu_count() or 1)),
                "MAX_PARALLEL_TASKS",
            ),
            take_mini_training_sample=_parse_bool(
                values.get("TAKE_MINI_TRAINING_SAMPLE", "false"),
            ),
            spatial_computation_value_column_regex=_required(
                values,
                "SPATIAL_COMPUTATION_VALUE_COLUMN_REGEX",
            ),
            model_run_ref=_resolve_model_run_ref(values.get("MODEL_RUN_REF")),
        )


@dataclass(frozen=True)
class TemporalConfigRequest:
    """Environment inputs used to resolve a runtime temporal configuration."""

    start_month: Arrow
    explicit_end_month: Arrow | None
    max_data_lag_months: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TemporalConfigRequest:
        """Load unresolved temporal inputs from an environment mapping."""
        values = os.environ if environ is None else environ
        explicit_end_month = values.get("END_MONTH", "").strip()
        return cls(
            start_month=arrow.get(_required(values, "START_MONTH"), "YYYY-MM-DD"),
            explicit_end_month=(
                arrow.get(explicit_end_month, "YYYY-MM-DD") if explicit_end_month else None
            ),
            max_data_lag_months=parse_max_data_lag_months(
                values.get("MAX_DATA_LAG_MONTHS"),
            ),
        )
