"""Tests for the publish manifest."""

from pathlib import Path

from publish.manifest import PublishManifest, default_manifest_path, read_manifest, write_manifest

REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def test__write_and_read_manifest__round_trip__preserves_fields(tmp_path: Path) -> None:
    path = tmp_path / f"publish-{REF}.json"
    original = PublishManifest(
        model_run_ref=REF,
        asset_prefix=f"pm25-full_{REF}",
        zenodo_draft_id="99",
        zenodo_draft_url="https://zenodo.org/uploads/99",
        pr_url="https://github.com/energyandcleanair/southasia-pm2-5/pull/3",
        pr_number=3,
        dataset_end_label="May 2026",
    )

    write_manifest(path, original)

    assert read_manifest(path) == original


def test__default_manifest_path__given_directory__uses_ref_in_filename(tmp_path: Path) -> None:
    assert default_manifest_path(REF, tmp_path) == tmp_path / f"publish-{REF}.json"
