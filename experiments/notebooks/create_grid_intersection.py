# This script is a modification of the code provided by:
# Kawano, A., Kelp, M., Qiu, M., Singh, K., Chaturvedi, E., DAHIYA, S., 
# Azevedo, I., & Burke, M. (2024). High-Quality Daily PM2.5 Datasets 
# for India at 10 km Resolution (Version 2) [Data set]. Science Advances. 
# https://doi.org/10.5281/zenodo.13694585
#
# Modifications: instead reading shapefiles the grids are given as inputs 
# to the function & the result is returned instead of being saved to a file.


# create_grid_intersection.py
import pandas as pd
import geopandas as gpd

def main(grid_india_10km, grid_india_50km):

    grid_india_10km['grid_id'] = grid_india_10km['grid_id'].astype(str)
    grid_india_50km['grid_id'] = grid_india_50km['grid_id'].astype(str)
    
    intersect = gpd.sjoin(left_df=grid_india_10km, right_df=grid_india_50km, how='left')
    intersect = intersect.drop(columns = ['geometry', 'index_right'])
    intersect = intersect.rename(columns = {'grid_id_left':'grid_id_10km', 'grid_id_right':'grid_id_50km'})

    # drop duplicated grid_id_10km values
    intersect = intersect.drop_duplicates(subset = 'grid_id_10km', keep = 'first')
    
    return intersect
if __name__ == "__main__":
    main()