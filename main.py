import json
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import numpy as np
import os
import xarray as xr

from crea_graphics.geotiff import write_cog


def main(input_file: str, output_directory: str,
         url_prefix: str = "https://storage.googleapis.com/crea-pm25ml-project-website-assets"):

    # 1. Read in the nc file
    ds = xr.open_dataset(input_file, decode_coords="all")

    # 2. Calculate monthly averages
    monthly_ds = ds.resample(time='MS').mean()

    # 3. Save monthly files as COG
    vmin = 0
    vmax = monthly_ds['pm25'].max().values*0.9
    color_map = custom_colormap()

    # create output dir
    os.makedirs(os.path.join(output_directory, 'rasters'), exist_ok=True)

    # create a list of dictionaries with metadata info: month, year, and filename
    data_dicts = []

    for month in monthly_ds['time']:
        outfilename = f'pm25_{str(month.dt.strftime("%Y_%m").values)}'
        data = monthly_ds['pm25'].sel(time=month)

        write_cog(data, vmin=vmin, vmax=vmax, cmap=color_map,
                  output_path=os.path.join(
                      output_directory, 'rasters', f'{outfilename}.tif'),
                  no_alpha=False)

        data_dicts.append({
            'year': str(int(month.dt.year.values)),
            'month': str(int(month.dt.month.values)),
            'url': f'{url_prefix}/rasters/{outfilename}.tif'
        })

    # 4. Generate json reference file
    # save the data_dicts as a json file

    with open(os.path.join(output_directory, 'metadata.json'), 'w') as f:
        json.dump(data_dicts, f, indent=4)

    # also make a json file for the color scale
    color_props = color_properties(color_map, (vmin, vmax), 'µg/m³')
    with open(os.path.join(output_directory, 'color_scale.json'), 'w') as f:
        json.dump(color_props, f, indent=4)


def custom_colormap(N: int = 250) -> LinearSegmentedColormap:
    """Create a custom continuous colormap adapted from CREA colors.

    Parameters
    ----------
    N : int
        Number of colors in the continuous colormap.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Continuous colormap.
    """
    # colormap, adapted from crea colors
    n_discrete = 12
    vals = np.ones((n_discrete, 4))*256
    vals[0, 0:-1] = [39, 165, 156]   # #27a59c
    vals[1, 0:-1] = [205, 224, 180]
    vals[2, 0:-1] = [255, 229, 153]  # FFE599 (instead of fff2cc)
    vals[3, 0:-1] = [246, 178, 107]  # f6b26b
    vals[4, 0:-1] = [230, 93, 48]
    vals[5, 0:-1] = [204, 0, 0]     # #cc0000
    vals[6, 0:-1] = [153, 0, 0]     # #990000
    vals[7, 0:-1] = [116, 27, 71]     # #741b47
    vals[8, 0:-1] = [83, 32, 100]  # 532064
    vals[9, 0:-1] = [53, 28, 117]  # #351c75
    vals[10, 0:-1] = [27, 14, 59]     # #1b0e3b
    vals[11, 0:-1] = [0, 0, 0]     # #000000

    vals = vals/256

    # Create discrete colormap and convert to continuous
    discrete_cmap = ListedColormap(vals)
    cmaplist = [discrete_cmap(i) for i in range(discrete_cmap.N)]

    return LinearSegmentedColormap.from_list(
        "crea_custom_cont", cmaplist, N)


def color_properties(color_map, color_bounds, units: str, N_sample: int = 20):

    # generate colorscale for metadata, as a list of hex strings
    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)

    # sample 20 colors from the colormap
    # (need to use integeres, so that color_map returns color
    # based on index in colormap and not color assigned to value)
    color_scale = [color_map(x) for x in
                   np.linspace(0, color_map.N, N_sample).astype(int)]
    # convert to hex
    color_scale = [mcolors.rgb2hex(c) for c in color_scale]

    # create metadata dictionary
    metadata = {
        'minval': round(color_bounds[0]),
        'maxval': round(color_bounds[1]),
        'unit': units,
        'scale': color_scale,
    }

    return metadata


if __name__ == "__main__":
    main(input_file='/home/work/Documents/projects/202508_india_satellite/input/pm25_final.nc',
         output_directory='data')
