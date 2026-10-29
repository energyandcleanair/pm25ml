"""Build website map assets and switch the live index."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from crea_graphics.geotiff import write_cog
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

if TYPE_CHECKING:
    from pathlib import Path

    from fsspec import AbstractFileSystem
    from matplotlib.colors import Colormap
    from xarray import DataArray, Dataset

URL_PREFIX = "https://storage.googleapis.com/crea-pm25ml-project-website-assets"
BUCKET_ID = "crea-pm25ml-project-website-assets"
INDEX_FILES = ("metadata.json", "color_scale.json")


def generate_and_upload_map_assets(
    *,
    netcdf_path: Path,
    asset_prefix: str,
    filesystem: AbstractFileSystem,
    work_dir: Path,
    bucket_id: str = BUCKET_ID,
) -> Path:
    """
    Write monthly COGs and JSON, then upload them under ``asset_prefix``.

    This does not replace the live index files at the bucket root.

    Args:
        netcdf_path: Finished ``pm25-full_*.nc`` file.
        asset_prefix: Version directory name in the website bucket.
        filesystem: Filesystem used to upload objects.
        work_dir: Local directory for generated rasters and JSON.
        bucket_id: Website assets bucket.

    Returns:
        The local directory that was uploaded.

    """
    dataset = xr.open_dataset(netcdf_path, decode_coords="all")
    try:
        output_dir = _write_local_assets(dataset, asset_prefix=asset_prefix, work_dir=work_dir)
    finally:
        dataset.close()

    _upload_directory(
        filesystem=filesystem,
        local_directory=output_dir,
        bucket_id=bucket_id,
        asset_prefix=asset_prefix,
    )
    return output_dir


def switch_live_index(
    *,
    filesystem: AbstractFileSystem,
    asset_prefix: str,
    bucket_id: str = BUCKET_ID,
) -> None:
    """Copy a version's index files over the live files at the bucket root."""
    for name in INDEX_FILES:
        source = f"{bucket_id}/{asset_prefix}/{name}"
        destination = f"{bucket_id}/{name}"
        with filesystem.open(source, "rb") as src, filesystem.open(destination, "wb") as dst:
            dst.write(src.read())


def custom_colormap(sample_count: int = 250) -> LinearSegmentedColormap:
    """
    Create a custom continuous colormap adapted from CREA colors.

    Args:
        sample_count: Number of colors in the continuous colormap.

    Returns:
        Continuous colormap.

    """
    n_discrete = 12
    vals = np.ones((n_discrete, 4)) * 256
    vals[0, 0:-1] = [39, 165, 156]
    vals[1, 0:-1] = [205, 224, 180]
    vals[2, 0:-1] = [255, 229, 153]
    vals[3, 0:-1] = [246, 178, 107]
    vals[4, 0:-1] = [230, 93, 48]
    vals[5, 0:-1] = [204, 0, 0]
    vals[6, 0:-1] = [153, 0, 0]
    vals[7, 0:-1] = [116, 27, 71]
    vals[8, 0:-1] = [83, 32, 100]
    vals[9, 0:-1] = [53, 28, 117]
    vals[10, 0:-1] = [27, 14, 59]
    vals[11, 0:-1] = [0, 0, 0]
    vals = vals / 256

    discrete_cmap = ListedColormap(vals)
    cmaplist = [discrete_cmap(index) for index in range(discrete_cmap.N)]
    return LinearSegmentedColormap.from_list("crea_custom_cont", cmaplist, sample_count)


def color_properties(
    color_map: Colormap | str,
    color_bounds: tuple[float, float],
    units: str,
    sample_count: int = 20,
) -> dict[str, object]:
    """Return JSON metadata for the website colour scale."""
    resolved = plt.get_cmap(color_map) if isinstance(color_map, str) else color_map
    color_scale = [
        mcolors.rgb2hex(resolved(value))
        for value in np.linspace(0, resolved.N, sample_count).astype(int)
    ]
    return {
        "minval": round(color_bounds[0]),
        "maxval": round(color_bounds[1]),
        "unit": units,
        "scale": color_scale,
    }


def _write_local_assets(dataset: Dataset, *, asset_prefix: str, work_dir: Path) -> Path:
    monthly = dataset.resample(time="MS").mean()
    vmin = 0.0
    vmax = float(monthly["pm25"].max().values) * 0.9
    color_map = custom_colormap()
    rasters_dir = work_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)

    data_dicts: list[dict[str, str]] = []
    for month in monthly["time"]:
        out_name = f"pm25_{month.dt.strftime('%Y_%m').item()}"
        data: DataArray = monthly["pm25"].sel(time=month)
        write_cog(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=color_map,
            output_path=str(rasters_dir / f"{out_name}.tif"),
            no_alpha=False,
        )
        data_dicts.append(
            {
                "year": str(int(month.dt.year.item())),
                "month": str(int(month.dt.month.item())),
                "url": f"{URL_PREFIX}/{asset_prefix}/rasters/{out_name}.tif",
            },
        )

    (work_dir / "metadata.json").write_text(json.dumps(data_dicts, indent=4), encoding="utf-8")
    color_props = color_properties(color_map, (vmin, vmax), "µg/m³")
    (work_dir / "color_scale.json").write_text(json.dumps(color_props, indent=4), encoding="utf-8")
    return work_dir


def _upload_directory(
    *,
    filesystem: AbstractFileSystem,
    local_directory: Path,
    bucket_id: str,
    asset_prefix: str,
) -> None:
    for path in sorted(local_directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_directory).as_posix()
        destination = f"{bucket_id}/{asset_prefix}/{relative}"
        with path.open("rb") as src, filesystem.open(destination, "wb") as dst:
            dst.write(src.read())
