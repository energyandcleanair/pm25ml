"""Prepare website assets, a Zenodo draft, and a website pull request."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from publish.manifest import PublishManifest, default_manifest_path, write_manifest
from publish.model_run_ref import require_production_model_run_ref
from publish.results import (
    asset_prefix,
    cv_summary_filename,
    download_result_file,
    gcs_filesystem,
    latest_dataset_end_label,
    netcdf_filename,
)
from publish.website import GitHubWebsiteClient, generate_and_upload_map_assets
from publish.zenodo import ZenodoClient, load_deposit_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fsspec import AbstractFileSystem


def prepare(  # noqa: PLR0913
    model_run_ref: str,
    *,
    netcdf_path: Path | None = None,
    cv_summary_path: Path | None = None,
    manifest_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    filesystem: AbstractFileSystem | None = None,
    zenodo_client: ZenodoClient | None = None,
    github_client: GitHubWebsiteClient | None = None,
) -> PublishManifest:
    """
    Download results, upload map assets, and open unpublished release pieces.

    Safe to re-run for the same production ref: map assets overwrite the same
    prefix, the unpublished Zenodo draft is reused, and the website PR branch is
    updated in place.
    """
    values = os.environ if environ is None else environ
    ref = require_production_model_run_ref(model_run_ref)
    fs = filesystem or gcs_filesystem()
    zenodo = zenodo_client or ZenodoClient.from_env(values)
    github = github_client or GitHubWebsiteClient.from_env()

    with tempfile.TemporaryDirectory(prefix="pm25ml_publish_") as tmp:
        work = Path(tmp)
        resolved_netcdf = netcdf_path or download_result_file(
            filesystem=fs,
            bucket=_required(values, "FINAL_RESULT_BUCKET_NAME"),
            profile_id=_profile_id(values),
            model_run_ref=ref,
            file_name=netcdf_filename(ref),
            destination=work / netcdf_filename(ref),
        )
        resolved_cv = cv_summary_path or download_result_file(
            filesystem=fs,
            bucket=_required(values, "FINAL_RESULT_BUCKET_NAME"),
            profile_id=_profile_id(values),
            model_run_ref=ref,
            file_name=cv_summary_filename(ref),
            destination=work / cv_summary_filename(ref),
        )
        prefix = asset_prefix(resolved_netcdf)
        dataset = xr.open_dataset(resolved_netcdf, decode_coords="all")
        try:
            end_label = latest_dataset_end_label(dataset)
        finally:
            dataset.close()

        generate_and_upload_map_assets(
            netcdf_path=resolved_netcdf,
            asset_prefix=prefix,
            filesystem=fs,
            work_dir=work / "map",
        )
        draft = zenodo.ensure_draft(
            version=ref,
            files=(resolved_netcdf, resolved_cv),
            metadata=load_deposit_metadata(version=ref),
        )
        pull_request = github.ensure_metadata_pull_request(
            model_run_ref=ref,
            dataset_end_label=end_label,
        )

    manifest = PublishManifest(
        model_run_ref=ref,
        asset_prefix=prefix,
        zenodo_draft_id=draft.record_id,
        zenodo_draft_url=draft.html_url,
        pr_url=pull_request.html_url,
        pr_number=pull_request.number,
        dataset_end_label=end_label,
    )
    destination = manifest_path or default_manifest_path(ref)
    write_manifest(destination, manifest)
    return manifest


def _profile_id(environ: Mapping[str, str]) -> str:
    return environ.get("PIPELINE_PROFILE", "india").strip() or "india"


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        msg = f"{name} is required to download finished results."
        raise RuntimeError(msg)
    return value
