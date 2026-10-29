"""Tests for downloading finished result files."""

from pathlib import Path

from fsspec.implementations.memory import MemoryFileSystem

from publish.results.gcs import asset_prefix, download_result_file, result_object_path

REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def test__download_result_file__object_in_memory_fs__writes_destination(tmp_path: Path) -> None:
    filesystem = MemoryFileSystem()
    source = result_object_path("bucket", "india", REF, f"pm25-full_{REF}.nc")
    filesystem.pipe(source, b"netcdf-bytes")
    destination = tmp_path / f"pm25-full_{REF}.nc"

    download_result_file(
        filesystem=filesystem,
        bucket="bucket",
        profile_id="india",
        model_run_ref=REF,
        file_name=f"pm25-full_{REF}.nc",
        destination=destination,
    )

    assert destination.read_bytes() == b"netcdf-bytes"


def test__asset_prefix__netcdf_path__strips_extension() -> None:
    assert asset_prefix(Path(f"pm25-full_{REF}.nc")) == f"pm25-full_{REF}"
