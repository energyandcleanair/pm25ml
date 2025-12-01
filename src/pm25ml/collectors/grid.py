"""Load the grid from a shapefile zip file."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl
import shapefile
from polars import DataFrame
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.ops import transform
from shapely.wkt import loads as load_wkt
from xarray import Dataset

from pm25ml.collectors.geo_time_grid_dataset import as_geo_time_grid
from pm25ml.collectors.ned.coord_types import Lat, Lon
from pm25ml.logging import logger

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from pm25ml.collectors.geo_time_grid_dataset import GeoTimeGridDataset


class Grid:
    """A class representing a grid for the NED dataset."""

    ORIGINAL_GEOM_COL = "original_geometry_wkt"
    ORIGINAL_X = "original_x"
    ORIGINAL_Y = "original_y"

    GEOM_COL = "geometry_wkt"
    LAT_COL = "lat"
    LON_COL = "lon"
    REGION_COL = "k_region"
    GRID_ID_COL = "grid_id"
    GRID_ID_50KM_COL = "id_50km"

    ACTUAL_COLUMNS: ClassVar[set[str]] = {
        GRID_ID_COL,
        GRID_ID_50KM_COL,
        GEOM_COL,
        LAT_COL,
        LON_COL,
        REGION_COL,
    }

    ORIGINAL_COLUMNS: ClassVar[set[str]] = {
        GRID_ID_COL,
        GRID_ID_50KM_COL,
        ORIGINAL_GEOM_COL,
        ORIGINAL_X,
        ORIGINAL_Y,
        REGION_COL,
    }

    BOUNDS_BORDER: float = 1.0

    _bounds_cache: tuple[Lon, Lat, Lon, Lat] | None = None
    _expanded_bounds_cache: tuple[Lon, Lat, Lon, Lat] | None = None

    def __init__(self, df: DataFrame) -> None:
        """
        Initialize the Grid with a DataFrame.

        Args:
            df (DataFrame): The DataFrame containing grid data.

        """
        for col in [Grid.LON_COL, Grid.LAT_COL, Grid.ORIGINAL_X, Grid.ORIGINAL_Y]:
            if df[col].dtype != pl.Float32 and df[col].dtype != pl.Float64:
                msg = f"Column {col} is not a float32 or float64"
                raise ValueError(msg)

        self.df = df.select(
            [pl.col(col) for col in self.ACTUAL_COLUMNS if col in df.columns],
        )
        self.df_original = df.select(
            [pl.col(col) for col in self.ORIGINAL_COLUMNS if col in df.columns],
        )

    @property
    def bounds(self) -> tuple[Lon, Lat, Lon, Lat]:
        """Get the bounds of the grid."""
        if self._bounds_cache is not None:
            return self._bounds_cache
        bounds = [load_wkt(wkt).bounds for wkt in self.df[self.GEOM_COL]]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        self._bounds_cache = (Lon(minx), Lat(miny), Lon(maxx), Lat(maxy))
        return self._bounds_cache

    @property
    def expanded_bounds(
        self,
    ) -> tuple[Lon, Lat, Lon, Lat]:
        """Get the bounds of the grid with an additional border."""
        if self._expanded_bounds_cache is not None:
            return self._expanded_bounds_cache

        minx, miny, maxx, maxy = self.bounds
        self._expanded_bounds_cache = (
            Lon(minx - Grid.BOUNDS_BORDER),
            Lat(miny - Grid.BOUNDS_BORDER),
            Lon(maxx + Grid.BOUNDS_BORDER),
            Lat(maxy + Grid.BOUNDS_BORDER),
        )
        return self._expanded_bounds_cache

    @property
    def n_rows(self) -> int:
        """Get the number of rows in the grid."""
        return self.df.shape[0]

    def to_xarray_with_data(self, data_df: DataFrame) -> GeoTimeGridDataset:
        """
        Convert input to xarray dataset with the provided data.

        Input data must have a column for "grid_id" and "date". Any remaining data columns will be
        treated as variables.

        Returns a 3D dataset, with time, y, and x dimensions in the original grid's CRS.
        """
        actual_columns = set(data_df.columns)
        incoming_df_expected_id_cols = {"grid_id", "date"}
        missing_id_columns = incoming_df_expected_id_cols - actual_columns
        if missing_id_columns:
            msg = f"Missing ID columns in DataFrame: {missing_id_columns}"
            raise ValueError(msg)

        # Check that date column is of type date
        if data_df["date"].dtype != pl.Date:
            data_df = data_df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
            )

        joined_df = (
            self.df_original.select(
                [
                    self.ORIGINAL_X,
                    self.ORIGINAL_Y,
                    self.GRID_ID_COL,
                ],
            )
            .join(
                data_df,
                on="grid_id",
                how="outer",
                coalesce=True,
            )
            .rename(
                {
                    "original_x": "x",
                    "original_y": "y",
                    "date": "time",
                },
            )
            .to_pandas()
        )

        joined_id_cols = ["x", "y", "time", "grid_id"]

        value_cols = [c for c in joined_df.columns if c not in joined_id_cols]
        # We need it in float32 to avoid a roundoff error causing extra points in the y-axis.
        joined_df["x"] = joined_df["x"].astype(np.float32)
        joined_df["y"] = joined_df["y"].astype(np.float32)
        for col in value_cols:
            joined_df[col] = joined_df[col].astype(np.float32)
        indexed_df = joined_df.set_index(["time", "y", "x"]).sort_index()

        return as_geo_time_grid(
            Dataset.from_dataframe(
                indexed_df[value_cols],
            ).transpose(
                "time",
                "y",
                "x",
            ),
        )


def load_grid_from_files(
    *,
    path_to_shapefile_zip: Path,
    path_to_50km_csv: Path,
    path_to_region_parquet: Path,
) -> Grid:
    """Load the grid from a file."""
    # Extract ZIP to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.debug("Extracting and reading grid from shapefile")
        grid_df = _load_from_zip(
            tmp_dir,
            path_to_shapefile_zip,
        )

        logger.debug("Loading 50km grid IDs from CSV")
        grid_id_to_50km_grids = pl.read_csv(
            path_to_50km_csv,
            has_header=True,
        ).select(
            pl.col("grid_id_10km").alias(Grid.GRID_ID_COL),
            pl.col("grid_id_50km").alias(Grid.GRID_ID_50KM_COL),
        )

        logger.debug("Loading grid regions from Parquet")
        grid_id_to_regions = pl.read_parquet(
            path_to_region_parquet,
        ).select(
            pl.col("grid_id").alias(Grid.GRID_ID_COL),
            pl.col("k_region").alias(Grid.REGION_COL),
        )

        logger.debug("Joining grid data with 50km grid IDs and regions")
        # Load into polars
        return Grid(
            grid_df.with_columns(
                [
                    pl.col(Grid.ORIGINAL_X).round(0).cast(float),
                    pl.col(Grid.ORIGINAL_Y).round(0).cast(float),
                ],
            )
            .join(
                grid_id_to_50km_grids,
                on=Grid.GRID_ID_COL,
                how="left",
                coalesce=True,
            )
            .join(
                grid_id_to_regions,
                on=Grid.GRID_ID_COL,
                how="left",
                coalesce=True,
            ),
        )


def _load_from_zip(
    tmp_dir: str,
    path_to_shapefile_zip: Path,
) -> DataFrame:
    # The shapefile zip contains the grid for the NED dataset.
    # It has a directory structure like this:
    # - grid_india_10km/
    #   - grid_india_10km.shp
    #   - grid_india_10km.shx
    #   - grid_india_10km.dbf
    #   - grid_india_10km.prj
    with zipfile.ZipFile(path_to_shapefile_zip, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    tmpdir_path = Path(tmp_dir)

    # Find .shp and .prj files recursively
    shp_path = next(tmpdir_path.rglob("*.shp"), None)
    prj_path = next(tmpdir_path.rglob("*.prj"), None)

    if not shp_path:
        msg = "Shapefile (.shp) not found in the ZIP archive."
        raise ValueError(msg)

    if not prj_path:
        msg = "Projection file (.prj) not found in the ZIP archive."
        raise ValueError(msg)

    # Load CRS
    with prj_path.open() as f:
        wkt = f.read()
    input_crs = CRS.from_wkt(wkt)

    output_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

    def reproject_geom(geom: BaseGeometry) -> BaseGeometry:
        return transform(transformer.transform, geom)

    # Read shapefile
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]  # skip deletion flag

    records = []
    for sr in reader.shapeRecords():
        attrs = dict(zip(fields, sr.record))
        # Convert grid_id to int if present
        if "grid_id" not in attrs:
            msg = "grid_id not found in shapefile attributes."
            raise ValueError(msg)

        attrs[Grid.GRID_ID_COL] = int(attrs["grid_id"])
        geom = shape(sr.shape.__geo_interface__)
        geom_reproj = reproject_geom(geom)
        attrs[Grid.GEOM_COL] = geom_reproj.wkt

        # Extract centroid coordinates for lon and lat
        centroid = geom_reproj.centroid
        attrs[Grid.LON_COL] = centroid.x
        attrs[Grid.LAT_COL] = centroid.y

        # Extract original centroid
        original_centroid = geom.centroid
        attrs[Grid.ORIGINAL_GEOM_COL] = geom.wkt
        attrs[Grid.ORIGINAL_X] = original_centroid.x
        attrs[Grid.ORIGINAL_Y] = original_centroid.y

        records.append(attrs)

    return DataFrame(records)
