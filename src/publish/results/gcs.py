"""Download finished pipeline outputs from GCS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gcsfs import GCSFileSystem

if TYPE_CHECKING:
    from pathlib import Path

    from fsspec import AbstractFileSystem

NETCDF_PREFIX = "pm25-full"
CV_SUMMARY_PREFIX = "pm25-cv-summary"


def netcdf_filename(model_run_ref: str) -> str:
    """Return the NetCDF filename for a run."""
    return f"{NETCDF_PREFIX}_{model_run_ref}.nc"


def cv_summary_filename(model_run_ref: str) -> str:
    """Return the CV summary filename for a run."""
    return f"{CV_SUMMARY_PREFIX}_{model_run_ref}.csv"


def asset_prefix(netcdf_path: Path) -> str:
    """Return the website GCS prefix from the NetCDF basename."""
    return netcdf_path.name.removesuffix(".nc")


def result_object_path(bucket: str, profile_id: str, model_run_ref: str, file_name: str) -> str:
    """Return the gcsfs path for a finished result object."""
    return f"{bucket}/country={profile_id}/run={model_run_ref}/{file_name}"


def download_result_file(  # noqa: PLR0913
    *,
    filesystem: AbstractFileSystem,
    bucket: str,
    profile_id: str,
    model_run_ref: str,
    file_name: str,
    destination: Path,
) -> Path:
    """Copy one result object from GCS to ``destination``."""
    source = result_object_path(bucket, profile_id, model_run_ref, file_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with filesystem.open(source, "rb") as src, destination.open("wb") as dst:
        dst.write(src.read())
    return destination


def gcs_filesystem() -> GCSFileSystem:
    """Return the default GCS filesystem."""
    return GCSFileSystem()
