from polars import Float64, Int64, String
from pathlib import Path

import pytest
from pm25ml.collectors.grid import load_grid_from_files, Grid

from shapely.wkt import loads as load_wkt
from shapely.geometry import Polygon
from typing import cast

pytestmark = pytest.mark.integration

SAMPLE_GRID_ID = 61788
EXPECTED_COVNERTED_POLYGON = (
    "POLYGON (("
    "92.91545192768046 23.55959233529598, "
    "92.92460218988784 23.65128722473793, "
    "93.02413560355933 23.642819806575076, "
    "93.01491567514682 23.551130887157257, "
    "92.91545192768046 23.55959233529598))"
)
EXPECTED_CENTROID_LON = 92.96977688113475
EXPECTED_CENTROID_LAT = 23.601212914119746

EXPECTED_N_ROWS = 33074

MAX_ERROR_SIZE = 1e-10


EXPECTED_ORIGINAL_POLYGON = (
    "POLYGON ((5291162.5154203195 4011391.913685304, "
    "5291162.5154203195 4021391.913685304, "
    "5301162.5154203195 4021391.913685304, "
    "5301162.5154203195 4011391.913685304, "
    "5291162.5154203195 4011391.913685304))"
)
EXPECTED_CENTROID_X = 5296162.5154203195
EXPECTED_CENTROID_Y = 4016391.9136853046

EXPECTED_50KM_GRID_ID = 2494


def test__load_grid_from_zip__valid_shapefile_zip__grid_loaded():
    """Test loading a grid from a valid shapefile zip file."""
    # Arrange
    path_to_shapefile_zip = Path("./assets/india/grid_10km_shapefiles.zip")
    path_to_csv_50km = Path("./assets/india/grid_intersect_with_50km.csv")
    path_to_region_parquet = Path("./assets/india/grid_region.parquet")

    # Act
    grid = load_grid_from_files(
        path_to_shapefile_zip=path_to_shapefile_zip,
        path_to_50km_csv=path_to_csv_50km,
        path_to_region_parquet=path_to_region_parquet,
    )

    assert isinstance(grid, Grid)

    ### Test the converted loaded grid matches expected values

    # Assert the shape and types of the data frame
    assert grid.df.shape[0] == EXPECTED_N_ROWS
    assert "geometry_wkt" in grid.df.columns
    assert grid.df["geometry_wkt"].dtype == String
    assert "lon" in grid.df.columns
    assert grid.df["lon"].dtype == Float64
    assert "lat" in grid.df.columns
    assert grid.df["lat"].dtype == Float64
    assert "grid_id" in grid.df.columns
    assert grid.df["grid_id"].dtype == Int64
    # Region column should exist and be an int
    assert "k_region" in grid.df.columns
    assert grid.df["k_region"].dtype == Int64

    # Make sure that all of the lat and lon values are within expected ranges
    lon_min = grid.df["lon"].min()
    lon_max = grid.df["lon"].max()
    assert isinstance(lon_min, (float, int)) and lon_min >= -180
    assert isinstance(lon_max, (float, int)) and lon_max <= 180
    lat_min = grid.df["lat"].min()
    lat_max = grid.df["lat"].max()
    assert isinstance(lat_min, (float, int)) and lat_min >= -90
    assert isinstance(lat_max, (float, int)) and lat_max <= 90

    # Check that the sampled GRID ID has a polygon close to the expected one
    sample_row = grid.df.filter(grid.df["grid_id"] == SAMPLE_GRID_ID)

    grid_id_50km = sample_row["id_50km"].item()
    assert grid_id_50km == EXPECTED_50KM_GRID_ID, (
        f"Expected 50km grid ID {EXPECTED_50KM_GRID_ID}, "
        f"but got {grid_id_50km} for grid ID {SAMPLE_GRID_ID}"
    )

    polygon_wkt = sample_row["geometry_wkt"].item()
    expected_wkt: Polygon = cast(Polygon, load_wkt(EXPECTED_COVNERTED_POLYGON))
    actual_wkt = load_wkt(polygon_wkt)
    assert isinstance(actual_wkt, Polygon), "Sample geometry is not a Polygon"
    assert len(list(expected_wkt.exterior.coords)) == len(list(actual_wkt.exterior.coords)), (
        "Number of points in polygon does not match"
    )
    for expected_coord, actual_coord in zip(
        expected_wkt.exterior.coords, actual_wkt.exterior.coords
    ):
        assert abs(expected_coord[0] - actual_coord[0]) < MAX_ERROR_SIZE, (
            "Longitude coordinates do not match on shape"
        )
        assert abs(expected_coord[1] - actual_coord[1]) < MAX_ERROR_SIZE, (
            "Latitude coordinates do not match on shape"
        )

    # Check that the sampled GRID ID has a centroid close to the expected one
    actual_converted_centroid = actual_wkt.centroid
    assert abs(actual_converted_centroid.x - EXPECTED_CENTROID_LON) < MAX_ERROR_SIZE, (
        "Centroid longitude does not match"
    )
    assert abs(actual_converted_centroid.y - EXPECTED_CENTROID_LAT) < MAX_ERROR_SIZE, (
        "Centroid latitude does not match"
    )

    ### Test the original loaded grid matches expected values

    df_original = grid.df_original
    assert df_original.shape[0] == EXPECTED_N_ROWS
    assert "original_geometry_wkt" in df_original.columns
    assert df_original["original_geometry_wkt"].dtype == String
    assert "original_x" in df_original.columns
    assert df_original["original_x"].dtype == Float64
    assert "original_y" in df_original.columns
    assert df_original["original_y"].dtype == Float64

    original_grid_id_50km = df_original.filter(df_original["grid_id"] == SAMPLE_GRID_ID)[
        "id_50km"
    ].item()
    assert original_grid_id_50km == EXPECTED_50KM_GRID_ID, (
        f"Expected 50km grid ID {EXPECTED_50KM_GRID_ID}, "
        f"but got {original_grid_id_50km} for grid ID {SAMPLE_GRID_ID}"
    )

    # Check that the original polygon matches the expected one
    original_polygon_wkt = df_original.filter(df_original["grid_id"] == SAMPLE_GRID_ID)[
        "original_geometry_wkt"
    ].item()
    original_polygon: Polygon = cast(Polygon, load_wkt(original_polygon_wkt))
    expected_original_polygon: Polygon = cast(Polygon, load_wkt(EXPECTED_ORIGINAL_POLYGON))
    assert len(list(expected_original_polygon.exterior.coords)) == len(
        list(original_polygon.exterior.coords)
    ), "Number of points in original polygon does not match"
    for expected_coord, actual_coord in zip(
        expected_original_polygon.exterior.coords, original_polygon.exterior.coords
    ):
        assert abs(expected_coord[0] - actual_coord[0]) < MAX_ERROR_SIZE, (
            "Original polygon longitude coordinates do not match on shape"
        )
        assert abs(expected_coord[1] - actual_coord[1]) < MAX_ERROR_SIZE, (
            "Original polygon latitude coordinates do not match on shape"
        )

    actual_converted_centroid = original_polygon.centroid
    assert abs(actual_converted_centroid.x - EXPECTED_CENTROID_X) < MAX_ERROR_SIZE, (
        "Original polygon centroid longitude does not match"
    )
    assert abs(actual_converted_centroid.y - EXPECTED_CENTROID_Y) < MAX_ERROR_SIZE, (
        "Original polygon centroid latitude does not match"
    )

    # Region column also present in original df and matches expected for sample
    assert "k_region" in df_original.columns
    sample_region = df_original.filter(df_original["grid_id"] == SAMPLE_GRID_ID)["k_region"].item()
    assert isinstance(sample_region, (int,))
    assert sample_region == 2
