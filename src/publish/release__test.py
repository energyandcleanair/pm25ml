"""Tests for the release command."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from publish.manifest import PublishManifest, write_manifest
from publish.release import release
from publish.website import BUCKET_ID

REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def _manifest(path: Path) -> Path:
    write_manifest(
        path,
        PublishManifest(
            model_run_ref=REF,
            asset_prefix=f"pm25-full_{REF}",
            zenodo_draft_id="222",
            zenodo_draft_url="https://zenodo.org/uploads/222",
            pr_url="https://github.com/example/pr/4",
            pr_number=4,
            dataset_end_label="May 2026",
        ),
    )
    return path


def test__release__all_steps__publishes_switches_and_merges(tmp_path: Path) -> None:
    path = _manifest(tmp_path / f"publish-{REF}.json")
    zenodo = Mock()
    github = Mock()
    filesystem = MemoryFileSystem()
    prefix = f"pm25-full_{REF}"
    filesystem.pipe(f"{BUCKET_ID}/{prefix}/metadata.json", b"{}")
    filesystem.pipe(f"{BUCKET_ID}/{prefix}/color_scale.json", b"{}")

    release(
        REF,
        manifest_path=path,
        filesystem=filesystem,
        zenodo_client=zenodo,
        github_client=github,
    )

    zenodo.publish_draft.assert_called_once_with("222")
    github.merge_pull_request.assert_called_once_with(4)
    assert filesystem.cat(f"{BUCKET_ID}/metadata.json") == b"{}"
    assert filesystem.cat(f"{BUCKET_ID}/color_scale.json") == b"{}"


def test__release__zenodo_fails__does_not_switch_index(tmp_path: Path) -> None:
    path = _manifest(tmp_path / f"publish-{REF}.json")
    zenodo = Mock()
    zenodo.publish_draft.side_effect = RuntimeError("zenodo down")
    github = Mock()
    filesystem = Mock()

    with pytest.raises(RuntimeError, match="completing: none"):
        release(
            REF,
            manifest_path=path,
            filesystem=filesystem,
            zenodo_client=zenodo,
            github_client=github,
        )

    github.merge_pull_request.assert_not_called()
    filesystem.open.assert_not_called()


def test__release__index_fails_after_zenodo__reports_completed_zenodo(tmp_path: Path) -> None:
    path = _manifest(tmp_path / f"publish-{REF}.json")
    zenodo = Mock()
    github = Mock()
    filesystem = Mock()
    filesystem.open.side_effect = OSError("bucket")

    with pytest.raises(RuntimeError, match="Zenodo record published"):
        release(
            REF,
            manifest_path=path,
            filesystem=filesystem,
            zenodo_client=zenodo,
            github_client=github,
        )

    github.merge_pull_request.assert_not_called()
