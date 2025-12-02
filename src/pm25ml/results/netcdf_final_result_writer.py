"""Writer for putting final results to a NetCDF file."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
from xarray import DataArray, Dataset

from pm25ml.collectors.geo_time_grid_dataset import GeoTimeGridDataset
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.results.final_result_writer import FinalResultWriter


class NetCdfResultWriter(FinalResultWriter):
    """
    Write a Dataset as NetCDF, then upload it to the configured final storage.

    This writer ensures compatibility by writing to a local temporary file first
    (using the h5netcdf engine) and then streaming the file to the destination
    filesystem provided by ``FinalResultStorage``.
    """

    def __init__(
        self,
        output_ref: DataArtifactRef,
        file_prefix: str,
        output_storage: FinalResultStorage,
    ) -> None:
        """
        Initialize the NetCdfResultWriter.

        Args:
            output_ref (DataArtifactRef): The reference to the output data artifact.
            file_prefix (str): The prefix for the output file name.
            output_storage (FinalResultStorage): Storage to upload the final file to.

        """
        self.output_ref = output_ref
        self.file_prefix = file_prefix
        self.output_storage = output_storage

    def write(self, result: GeoTimeGridDataset) -> None:
        """
        Write the given result to NetCDF and upload to final storage.

        The destination key will be ``{output_ref.initial_path}/{file_prefix}.nc``.
        """
        ds: Dataset = result.copy()

        ds["x"].attrs.update(
            {
                "standard_name": "projection_x_coordinate",
                "long_name": "Easting",
                "units": "m",
            },
        )
        ds["y"].attrs.update(
            {
                "standard_name": "projection_y_coordinate",
                "long_name": "Northing",
                "units": "m",
            },
        )
        ds["x"].attrs.setdefault("axis", "X")
        ds["y"].attrs.setdefault("axis", "Y")

        for value in ds.data_vars:
            ds[value].attrs["coordinates"] = "time y x"

        ds.attrs.setdefault("Conventions", "CF-1.8")

        ds = self._fix_time_dimension(ds)
        ds = self._add_projection_info(ds)

        # Write to a temporary NetCDF file, then upload that file to the final storage.
        # Direct streaming to remote filesystems isn't supported by h5netcdf/netcdf4 engines.
        # Using a local temp file ensures compatibility and low memory overhead.
        with tempfile.TemporaryDirectory(prefix="pm25ml_netcdf_") as tmpdir:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{self.file_prefix}_{timestamp}.nc"
            tmp_path = Path(tmpdir) / filename

            compression_args: dict[str, Any] = {
                "zlib": True,
                "complevel": 5,
                "chunksizes": (16, 82, 72),
                "shuffle": True,
            }
            encoding: dict[str, dict[str, Any]] = dict.fromkeys(
                ds.data_vars, compression_args,
            )

            # Persist to NetCDF using the h5netcdf engine (netCDF4-compatible)
            ds.to_netcdf(
                str(tmp_path),
                engine="h5netcdf",
                encoding=encoding,
            )

            # Stream the file to final storage
            with tmp_path.open("rb") as fh:
                self.output_storage.write(
                    fh,
                    path=str(self.output_ref.initial_path),
                    file_name=filename,
                )

    def _fix_time_dimension(self, ds: Dataset) -> Dataset:
        ds["time"].encoding.update(
            {
                "units": "days since 2000-01-01 00:00:00",
                "calendar": "gregorian",
            },
        )
        ds["time"].attrs.update(
            {
                "standard_name": "time",
                "axis": "T",
            },
        )

        return ds

    def _add_projection_info(self, ds: Dataset) -> Dataset:
        crs = pyproj.CRS.from_epsg(7755)
        ds["spatial_ref"] = DataArray(
            0,
            attrs=crs.to_cf(),
        )
        for v in ds.data_vars:
            ds[v].attrs["grid_mapping"] = "spatial_ref"

        x = ds.x.values
        y = ds.y.values

        dx = float(np.diff(x).mean())
        dy = abs(float(np.diff(y).mean()))

        gt0 = x[0] - dx / 2.0
        gt1 = dx
        gt2 = 0.0
        gt3 = y[0] + dy / 2.0
        gt4 = 0.0
        gt5 = -dy
        gt_str = f"{gt0} {gt1} {gt2} {gt3} {gt4} {gt5}"

        ds.attrs["GeoTransform"] = gt_str

        return ds
