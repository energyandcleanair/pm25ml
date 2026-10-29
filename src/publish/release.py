"""Publish the prepared Zenodo draft, map index, and website pull request."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from publish.manifest import default_manifest_path, read_manifest
from publish.model_run_ref import require_production_model_run_ref
from publish.results import gcs_filesystem
from publish.website import GitHubWebsiteClient, switch_live_index
from publish.zenodo import ZenodoClient

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fsspec import AbstractFileSystem


def release(  # noqa: PLR0913
    model_run_ref: str,
    *,
    manifest_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    filesystem: AbstractFileSystem | None = None,
    zenodo_client: ZenodoClient | None = None,
    github_client: GitHubWebsiteClient | None = None,
) -> None:
    """Publish the three prepared components, stopping at the first failure."""
    values = os.environ if environ is None else environ
    ref = require_production_model_run_ref(model_run_ref)
    path = manifest_path or default_manifest_path(ref)
    if not path.is_file():
        msg = f"Publish manifest not found: {path}"
        raise FileNotFoundError(msg)
    manifest = read_manifest(path)
    if manifest.model_run_ref != ref:
        msg = f"Manifest {path} is for {manifest.model_run_ref}, not the requested {ref}."
        raise ValueError(msg)

    zenodo = zenodo_client or ZenodoClient.from_env(values)
    github = github_client or GitHubWebsiteClient.from_env()
    fs = filesystem or gcs_filesystem()
    completed: list[str] = []
    try:
        zenodo.publish_draft(manifest.zenodo_draft_id)
        completed.append("Zenodo record published")
        switch_live_index(filesystem=fs, asset_prefix=manifest.asset_prefix)
        completed.append("Website map index switched")
        github.merge_pull_request(manifest.pr_number)
        completed.append("Website pull request merged")
    except Exception as error:
        done = "; ".join(completed) if completed else "none"
        msg = f"Release failed after completing: {done}."
        raise RuntimeError(msg) from error
