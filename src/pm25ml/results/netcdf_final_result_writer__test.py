import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
from morefs.memory import MemFS

from pm25ml.collectors.geo_time_grid_dataset import GeoTimeGridDataset, as_geo_time_grid
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.results.netcdf_final_result_writer import NetCdfResultWriter


DESTINATION_BUCKET = "test_bucket"
TEST_COUNTRY = "india"
MODEL_RUN_REF = "release-2026.03"


@pytest.fixture()
def mem_storage() -> FinalResultStorage:
    """Final storage backed by MemFS."""
    fs = MemFS()
    return FinalResultStorage(
        filesystem=fs,
        destination_bucket=DESTINATION_BUCKET,
        output_path=f"country={TEST_COUNTRY}/run={MODEL_RUN_REF}",
    )


def _make_dataset(time_len: int = 16, y_len: int = 82, x_len: int = 72) -> GeoTimeGridDataset:
    """Create a CF-friendly dataset with dims (time,y,x) and coords x,y in meters.

    The sizes align with the writer's default chunk sizes (16,82,72).
    """
    # Time: 16 daily timestamps
    time = np.arange("2023-01-01", "2023-01-17", dtype="datetime64[D]")
    assert time.size == time_len

    # Grid: simple regular spacing in meters
    x = np.arange(x_len, dtype="int64") * 1000  # 1 km spacing
    y = np.arange(y_len, dtype="int64") * 1000  # 1 km spacing

    # Data variable: deterministic values for stability
    # Shape: (time, y, x)
    base = (
        np.arange(time_len, dtype="float32")[:, None, None]
        + np.arange(y_len, dtype="float32")[None, :, None]
        + np.arange(x_len, dtype="float32")[None, None, :]
    )
    ds = xr.Dataset(
        data_vars={
            # xarray expects (dims, data) for variable specification
            "pm25": (("time", "y", "x"), base),
        },
        coords={
            "time": time,
            "y": y,
            "x": x,
        },
        attrs={
            "title": "Synthetic PM2.5 dataset for testing",
        },
    )
    # Validate and tag as GeoTimeGridDataset
    return as_geo_time_grid(ds)


def test__netcdf_writer__writes_to_memfs_and_preserves_cf_attrs(
    mem_storage: FinalResultStorage,
) -> None:
    writer = NetCdfResultWriter(
        model_run_ref=MODEL_RUN_REF,
        output_storage=mem_storage,
    )

    ds: GeoTimeGridDataset = _make_dataset()
    writer.write(ds)

    # Verify a file exists in MemFS at the expected path.
    rel_dir = f"country={TEST_COUNTRY}/run={MODEL_RUN_REF}"
    expected_dir = f"{DESTINATION_BUCKET}/{rel_dir}"
    files = mem_storage.filesystem.ls(expected_dir)
    assert len(files) == 1, f"Expected exactly one file in {expected_dir}, found: {files}"

    # Verify filename matches expected run-ref and suffix pattern.
    file_path = files[0]
    file_name = Path(file_path).name
    assert file_name == "release-2026.03_pm25-full.nc"

    full_path = file_path

    # Read back via a local temp copy for xarray/h5netcdf
    with (
        mem_storage.filesystem.open(full_path, "rb") as src,
        tempfile.NamedTemporaryFile(suffix=".nc") as tmp,
    ):
        data = src.read()
        if isinstance(data, str):
            data = data.encode()
        tmp.write(cast(bytes, data))
        tmp.flush()
        read_ds = xr.open_dataset(Path(tmp.name), engine="h5netcdf")

        try:
            # Basic variables and dims
            assert "pm25" in read_ds.data_vars
            assert read_ds.sizes == {"time": 16, "y": 82, "x": 72}

            # CF/global attrs
            assert read_ds.attrs.get("Conventions") == "CF-1.8"
            assert "GeoTransform" in read_ds.attrs

            # Coordinate attrs
            assert read_ds["x"].attrs.get("axis") == "X"
            assert read_ds["y"].attrs.get("axis") == "Y"
            assert read_ds["time"].attrs.get("axis") == "T"
            assert read_ds["time"].attrs.get("standard_name") == "time"

            # Projection / grid mapping
            assert "spatial_ref" in read_ds.variables
            assert read_ds["pm25"].attrs.get("grid_mapping") == "spatial_ref"

            # Coordinates: some engines may not persist the 'coordinates' attr; dims must match
            assert read_ds["pm25"].dims == ("time", "y", "x")
        finally:
            read_ds.close()
