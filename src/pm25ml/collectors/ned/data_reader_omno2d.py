"""Data reader for OMI NO2 Collection 4 (OMNO2d v004) data."""

from ast import literal_eval
from typing import IO, TYPE_CHECKING, cast

import arrow
import numpy as np
import xarray
from numpy.typing import NDArray

from pm25ml.collectors.ned.coord_types import Lat, Lon, NpLat, NpLon
from pm25ml.collectors.ned.data_readers import NedDataReader, NedDayData
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

if TYPE_CHECKING:
    from xarray.core.types import ReadBuffer


class Omno2dReader(NedDataReader):
    """
    Data reader for OMI NO2 Collection 4 (OMNO2d v004) data.

    This class reads OMNO2d v004 data from an HDF5 file and extracts the relevant
    data for a given date range and geographical bounds.
    """

    GRID_NAME = "ColumnAmountNO2"

    def __init__(self) -> None:
        """Initialize the OMNO2d v004 data reader."""
        super().__init__()

    def extract_data(
        self,
        file: IO[bytes],
        dataset_descriptor: NedDatasetDescriptor,
    ) -> NedDayData:
        """
        Extract the data from an OMI NO2 file for a dataset.

        Args:
            file (IO[bytes]): The file containing the OMI NO2 data.
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
            on how to extract the data.

        Returns:
            NedDayData: An object containing the results of the extraction.

        """
        date = self._extract_date(file)

        lon, lat = self._build_coords(file)

        ds = xarray.open_dataset(
            cast("ReadBuffer", file),
            group=f"HDFEOS/GRIDS/{self.GRID_NAME}/Data Fields",
            phony_dims="access",
        )

        var_names = list(dataset_descriptor.variable_mapping.keys())

        min_lon = dataset_descriptor.filter_min_lon
        max_lon = dataset_descriptor.filter_max_lon
        min_lat = dataset_descriptor.filter_min_lat
        max_lat = dataset_descriptor.filter_max_lat

        ds = ds.rename({"phony_dim_0": "lat", "phony_dim_1": "lon"})
        ds = ds.assign_coords(lat=("lat", lat), lon=("lon", lon))
        ds = ds[var_names]
        ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

        dataset = ds

        return NedDayData(dataset=dataset, date=date)

    def _extract_date(self, file: IO[bytes]) -> str:
        file_attributes = xarray.open_dataset(
            file,
            group="HDFEOS/ADDITIONAL/FILE_ATTRIBUTES",
            phony_dims="access",
        )

        year_str = file_attributes.attrs["GranuleYear"].item()
        month_str = file_attributes.attrs["GranuleMonth"].item()
        day_str = file_attributes.attrs["GranuleDay"].item()

        return arrow.get(int(year_str), int(month_str), int(day_str)).format("YYYY-MM-DD")

    def _build_coords(self, file: IO[bytes]) -> tuple[np.ndarray, np.ndarray]:
        grid_info = xarray.open_dataset(
            file,
            group=f"HDFEOS/GRIDS/{self.GRID_NAME}",
            phony_dims="access",
        )
        bounds: tuple[Lon, Lon, Lat, Lat] = literal_eval(grid_info.attrs["GridSpan"])
        resolution: tuple[Lon, Lat] = literal_eval(grid_info.attrs["GridSpacing"])
        min_lon, max_lon, min_lat, max_lat = bounds

        lon, lat = self._define_coords(
            lon_bounds=(min_lon, max_lon),
            lat_bounds=(min_lat, max_lat),
            resolution=resolution,
        )

        lat_len = grid_info.attrs["NumberOfLatitudesInGrid"].item()
        lon_len = grid_info.attrs["NumberOfLongitudesInGrid"].item()

        if lat_len != len(lat):
            msg = f"lat length {lat_len} does not match generated grid length {len(lat)}"
            raise ValueError(msg)
        if lon_len != len(lon):
            msg = f"lon length {lon_len} does not match generated grid length {len(lon)}"
            raise ValueError(msg)

        return lon, lat

    def _define_coords(
        self,
        *,
        lat_bounds: tuple[Lat, Lat],
        lon_bounds: tuple[Lon, Lon],
        resolution: tuple[Lon, Lat],
    ) -> tuple[NDArray[NpLon], NDArray[NpLat]]:
        """
        Create a meshgrid of points between bounding coordinates.

        Uses latitude bounds, longitude bounds, and the data product
        resolution to create a grid of points.

        Args:
            lat_bounds (tuple[Lat, Lat]): latitude bounds as a tuple.
            lon_bounds (tuple[Lon, Lon]): longitude bounds as a tuple.
            resolution (tuple[Lon, Lat]): data resolution as a tuple.

        Returns:
            lon (NDArray[NpLon]): x (longitude) coordinates.
            lat (NDArray[NpLat]): y (latitude) coordinates.

        """
        # See existing real-world examples of how to use this:
        # - https://drivendata.co/blog/predict-no2-benchmark
        # - https://github.com/quintusdias/hdfeos_python_zoo/blob/master/zoo/gesdisc/omi/OMI_L3_ColumnAmountO3.py

        lon_resolution = resolution[0]
        lat_resolution = resolution[1]

        lon_centre_adjustment = lon_resolution / 2.0
        lat_centre_adjustment = lat_resolution / 2.0

        lon = cast(
            "NDArray[NpLon]",
            np.arange(lon_bounds[0], lon_bounds[1], lon_resolution) + lon_centre_adjustment,
        )
        lat = cast(
            "NDArray[NpLat]",
            np.arange(lat_bounds[0], lat_bounds[1], lat_resolution) + lat_centre_adjustment,
        )

        return lon, lat
