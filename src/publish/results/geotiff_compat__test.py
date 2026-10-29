"""Fail the build if Rasterio or write_cog cannot produce a valid GeoTIFF."""

from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from crea_graphics.geotiff import write_cog
from rasterio.transform import from_origin


def test__rasterio__roundtrip_geotiff__preserves_values_crs_and_transform(
    tmp_path: Path,
) -> None:
    path = tmp_path / "roundtrip.tif"
    transform = from_origin(77.0, 29.0, 0.1, 0.1)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.transform == transform
        np.testing.assert_array_equal(src.read(1), data)


def test__write_cog__small_grid__tiled_deflate_uint8_epsg3857(tmp_path: Path) -> None:
    path = tmp_path / "cog.tif"
    data = xr.DataArray(
        np.linspace(10, 80, 32 * 32, dtype=np.float32).reshape(32, 32),
        dims=("y", "x"),
        coords={"x": np.linspace(77.0, 77.3, 32), "y": np.linspace(28.3, 28.0, 32)},
        name="pm25",
    )
    data = data.rio.write_crs("EPSG:4326")
    data = data.rio.set_spatial_dims(x_dim="x", y_dim="y")
    write_cog(data, vmin=0, vmax=90, output_path=str(path), no_alpha=False)

    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint8"
        assert src.crs.to_epsg() == 3857
        assert src.profile["tiled"] is True
        assert src.compression is not None
        assert src.compression.name.upper() == "DEFLATE"
        assert src.overviews(1)
        assert src.count >= 3
