# This script is a modification of the code provided by:
# Kawano, A., Kelp, M., Qiu, M., Singh, K., Chaturvedi, E., DAHIYA, S., 
# Azevedo, I., & Burke, M. (2024). High-Quality Daily PM2.5 Datasets 
# for India at 10 km Resolution (Version 2) [Data set]. Science Advances. 
# https://doi.org/10.5281/zenodo.13694585
#
# Modifications: instead of saving the grid as a shapefile, it is returned.

# create_grid_50_by_50.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import geometry

# Define constants
coordinates = [(68.0, 6.6), (68.0, 37.0), (97.0, 37.0), (97.0, 6.6)]
square_size = 50000  # polygon size

def create_polygon(coords, polygon_name):
    polygon = Polygon(coordinates)
    gdf = gpd.GeoDataFrame(crs={'init':'epsg:4326'}, geometry=[polygon])
    gdf.loc[0,'name'] = polygon_name
    return gdf

def main():
    zone = create_polygon(coordinates, 'India')
    zone = zone.to_crs(epsg=7755)

    total_bounds = zone.total_bounds
    minX, minY, maxX, maxY = total_bounds

    x, y = minX, minY
    geom_array = []
    while y <= maxY:
        while x <= maxX:
            geom = geometry.Polygon([(x,y), (x, y+square_size), (x+square_size, y+square_size), (x+square_size, y), (x, y)])
            geom_array.append(geom)
            x += square_size
        x = minX
        y += square_size

    grid_india_50km = gpd.GeoDataFrame(geom_array, columns=['geometry']).set_crs('EPSG:7755')
    grid_india_50km['grid_id'] = list(grid_india_50km.index)
    
    return grid_india_50km  