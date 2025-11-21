from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import xarray as xr
import numpy as np
from crea_graphics.geotiff import write_cog


def main(input_file: str, output_directory: str):

    # 1. Read in the nc file
    ds = xr.open_dataset(input_file, decode_coords="all")

    # 2. Calculate monthly averages
    monthly_ds = ds.resample(time='MS').mean()

    # 3. Save monthly files as COG
    vmin = 0  # monthly_ds['pm25'].min().values
    vmax = monthly_ds['pm25'].max().values*0.9
    color_map = custom_colormap()  # 'viridis'

    # create output dir
    os.makedirs(os.path.join(output_directory, 'rasters'), exist_ok=True)

    for month in monthly_ds['time']:
        outfilename = f'pm25_{str(month.dt.strftime("%Y_%m").values)}'
        data = monthly_ds['pm25'].sel(time=month)

        write_cog(data, vmin=vmin, vmax=vmax, cmap=color_map,
                  output_path=os.path.join(
                      output_directory, 'rasters', f'{outfilename}.tif'),
                  no_alpha=False)

    # 4. Generate json reference file

    pass


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


if __name__ == "__main__":
    main(input_file='/home/work/Documents/projects/202508_india_satellite/input/pm25_final.nc',
         output_directory='data')
