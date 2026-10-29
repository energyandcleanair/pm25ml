"""Tests for the prepare command."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from publish.manifest import read_manifest
from publish.model_run_ref import ModelRunRefError
from publish.prepare import prepare
from publish.website import WebsitePullRequest
from publish.zenodo import ZenodoDraft

REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def _netcdf(path: Path) -> Path:
    times = np.arange("2026-05-01", "2026-06-01", dtype="datetime64[D]")
    xr.Dataset({"pm25": ("time", np.zeros(len(times)))}, coords={"time": times}).to_netcdf(path)
    return path


def test__prepare__production_ref__writes_manifest_and_calls_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    netcdf = _netcdf(tmp_path / f"pm25-full_{REF}.nc")
    cv_path = tmp_path / f"pm25-cv-summary_{REF}.csv"
    cv_path.write_text("r2,rmse\n0.6,27\n", encoding="utf-8")
    zenodo = Mock()
    zenodo.ensure_draft.return_value = ZenodoDraft("222", "https://zenodo.org/uploads/222")
    github = Mock()
    github.ensure_metadata_pull_request.return_value = WebsitePullRequest(
        number=4,
        html_url="https://github.com/example/pr/4",
        branch=REF,
    )
    generate = Mock()
    monkeypatch.setattr("publish.prepare.generate_and_upload_map_assets", generate)
    manifest_path = tmp_path / f"publish-{REF}.json"

    result = prepare(
        REF,
        netcdf_path=netcdf,
        cv_summary_path=cv_path,
        manifest_path=manifest_path,
        environ={"FINAL_RESULT_BUCKET_NAME": "bucket"},
        filesystem=Mock(),
        zenodo_client=zenodo,
        github_client=github,
    )

    generate.assert_called_once()
    zenodo.ensure_draft.assert_called_once()
    draft_files = [path.name for path in zenodo.ensure_draft.call_args.kwargs["files"]]
    assert f"pm25-full_{REF}.nc" in draft_files
    assert "dataset.html" not in draft_files
    assert "versioning.html" not in draft_files
    metadata = zenodo.ensure_draft.call_args.kwargs["metadata"]
    assert metadata["version"] == REF
    assert metadata["title"] == "South Asia PM2.5 map: India | Data"
    assert "<h1>" in metadata["description"]
    assert "Disclaimer" in metadata["description"]
    assert "Data versioning" in metadata["notes"]
    assert "Disclaimer" in metadata["notes"]
    assert metadata["creators"][0]["name"] == "Rushwood, Panda"
    github.ensure_metadata_pull_request.assert_called_once_with(
        model_run_ref=REF,
        dataset_end_label="May 2026",
    )
    assert result.dataset_end_label == "May 2026"
    assert read_manifest(manifest_path).pr_number == 4


def test__prepare__local_ref__raises() -> None:
    with pytest.raises(ModelRunRefError):
        prepare("rel-7f3a9c2e_s-1.0_mb-local-abc")
