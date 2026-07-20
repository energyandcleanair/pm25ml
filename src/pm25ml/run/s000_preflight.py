"""Ensure the configured Earth Engine grid asset exists and is valid."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ee
import ee.data
from ee.ee_exception import EEException
from ee.featurecollection import FeatureCollection

from pm25ml.logging import logger
from pm25ml.setup.dependency_injection import (
    Pm25mlContainer,
    init_dependencies_from_env,
)

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


@dataclass(frozen=True)
class GridAssetConfig:
    """Configuration for ensuring the grid asset exists in Earth Engine."""

    profile_id: str
    grid_asset_path: str
    expected_grid_cell_count: int
    local_shapefile_zip: Path
    gee_staging_bucket_name: str
    upload_gcs_uri: str | None


def _asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
    except EEException:
        return False

    return True


def _validate_asset_size(asset_id: str, expected_grid_cell_count: int) -> None:
    feature_count = FeatureCollection(asset_id).size().getInfo()
    if feature_count != expected_grid_cell_count:
        msg = (
            f"Grid asset '{asset_id}' has {feature_count} features, "
            f"expected {expected_grid_cell_count}."
        )
        raise ValueError(msg)


def _file_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def _safe_asset_slug(asset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", asset_id).strip("_")


def _derive_default_gcs_uri(
    gee_staging_bucket_name: str,
    profile_id: str,
    asset_id: str,
    local_shapefile_zip: Path,
) -> str:
    content_sha = _file_sha256(local_shapefile_zip)
    asset_slug = _safe_asset_slug(asset_id)
    object_name = f"grid_assets/{profile_id}/{asset_slug}/{content_sha}/{local_shapefile_zip.name}"
    return f"gs://{gee_staging_bucket_name}/{object_name}"


def _split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        msg = f"Expected gs:// URI, got: {gcs_uri}"
        raise ValueError(msg)

    without_scheme = gcs_uri.removeprefix("gs://")
    bucket, _, path = without_scheme.partition("/")
    if not bucket or not path:
        msg = f"Invalid gs:// URI: {gcs_uri}"
        raise ValueError(msg)

    return bucket, path


def _upload_local_zip_to_gcs(
    filesystem: AbstractFileSystem,
    local_path: Path,
    gcs_uri: str,
) -> str:
    if not local_path.exists():
        msg = (
            f"Local shapefile zip not found at '{local_path}'. "
            "Set GRID_SHAPEFILE_ZIP_PATH or GEE_GRID_UPLOAD_GCS_URI."
        )
        raise FileNotFoundError(msg)

    bucket, path = _split_gcs_uri(gcs_uri)
    filesystem.put(str(local_path), f"{bucket}/{path}")
    logger.info("Uploaded local shapefile zip to %s", gcs_uri)
    return gcs_uri


def _ensure_parent_folders(asset_id: str) -> None:
    if "/assets/" not in asset_id:
        return

    root, _, suffix = asset_id.partition("/assets/")
    parts = [part for part in suffix.split("/") if part]
    if len(parts) <= 1:
        return

    parent_parts = parts[:-1]
    current = f"{root}/assets"
    for part in parent_parts:
        current = f"{current}/{part}"
        with contextlib.suppress(Exception):
            ee.data.createAsset({"type": "FOLDER"}, current)


def _start_table_ingestion(asset_id: str, gcs_uri: str) -> str:
    task = ee.data.startTableIngestion(
        None,  # type: ignore[arg-type]
        {
            "name": asset_id,
            "sources": [{"uris": [gcs_uri]}],
        },
    )

    task_id = task["id"]
    logger.info("Started EE table ingestion task %s for %s", task_id, asset_id)
    return task_id


def _wait_for_task(task_id: str, timeout_seconds: int = 1800) -> None:
    start = time.monotonic()
    while True:
        status = ee.data.getTaskStatus(task_id)
        state = status[0]["state"] if status else "UNKNOWN"

        if state == "COMPLETED":
            return

        if state == "FAILED":
            error_msg = status[0].get("error_message", "Unknown error") if status else "Unknown"
            msg = f"EE ingestion task {task_id} failed: {error_msg}"
            raise RuntimeError(msg)

        elapsed = time.monotonic() - start
        if elapsed > timeout_seconds:
            msg = f"Timed out waiting for EE ingestion task {task_id}"
            raise TimeoutError(msg)

        time.sleep(5)


def _ensure_grid_asset(config: GridAssetConfig, gcs_filesystem: AbstractFileSystem) -> None:
    if _asset_exists(config.grid_asset_path):
        logger.info("Grid asset already exists: %s", config.grid_asset_path)
        _validate_asset_size(config.grid_asset_path, config.expected_grid_cell_count)
        return

    logger.warning("Grid asset does not exist yet: %s", config.grid_asset_path)

    source_gcs_uri = config.upload_gcs_uri or _derive_default_gcs_uri(
        gee_staging_bucket_name=config.gee_staging_bucket_name,
        profile_id=config.profile_id,
        asset_id=config.grid_asset_path,
        local_shapefile_zip=config.local_shapefile_zip,
    )

    if config.upload_gcs_uri is None:
        _upload_local_zip_to_gcs(gcs_filesystem, config.local_shapefile_zip, source_gcs_uri)
    else:
        logger.info("Using pre-staged shapefile zip from %s", source_gcs_uri)

    _ensure_parent_folders(config.grid_asset_path)

    task_id = _start_table_ingestion(config.grid_asset_path, source_gcs_uri)
    _wait_for_task(task_id)

    _validate_asset_size(config.grid_asset_path, config.expected_grid_cell_count)
    logger.info("Grid asset ensured and validated: %s", config.grid_asset_path)


def _build_local_shapefile_zip_path(profile_id: str) -> Path:
    local_default = Path("assets") / profile_id / "grid_10km_shapefiles.zip"
    return Path(os.getenv("GRID_SHAPEFILE_ZIP_PATH", str(local_default)))


def _main(container: Pm25mlContainer) -> None:
    # Ensure GEE auth/init resource has been created before EE API calls.
    _ = container.gee_auth()

    profile_id = container.config.profile.id()
    config = GridAssetConfig(
        profile_id=profile_id,
        grid_asset_path=container.config.gcp.gee.grid_asset_path(),
        expected_grid_cell_count=container.config.profile.grid_cell_count(),
        local_shapefile_zip=_build_local_shapefile_zip_path(profile_id),
        gee_staging_bucket_name=container.config.gcp.gee_staging_bucket(),
        upload_gcs_uri=os.getenv("GEE_GRID_UPLOAD_GCS_URI"),
    )
    _ensure_grid_asset(config, container.gcs_filesystem())


if __name__ == "__main__":
    container = init_dependencies_from_env(end_month_mode="none")
    _main(container)
