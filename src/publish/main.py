import json
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import numpy as np
import os
import xarray as xr
import sys

from crea_graphics.geotiff import write_cog

from google.cloud import storage

URL_PREFIX = "https://storage.googleapis.com/crea-pm25ml-project-website-assets"
PROJECT_ID = "vital-form-455413-p8"
BUCKET_ID = "crea-pm25ml-project-website-assets"

OUTPUT_DIR = "data/output"


def main(input_file: str, version_number: str):

    base_dir = os.path.join(OUTPUT_DIR, version_number)

    # 1. Read in the nc file
    ds = xr.open_dataset(input_file, decode_coords="all")

    # 2. Calculate monthly averages
    monthly_ds = ds.resample(time="MS").mean()

    # 3. Save monthly files as COG
    vmin = 0
    vmax = monthly_ds["pm25"].max().values * 0.9
    color_map = custom_colormap()

    # create output dir
    os.makedirs(os.path.join(base_dir, "rasters"), exist_ok=True)

    # create a list of dictionaries with metadata info: month, year, and filename
    data_dicts = []

    for month in monthly_ds["time"]:
        outfilename = f'pm25_{str(month.dt.strftime("%Y_%m").values)}'
        data = monthly_ds["pm25"].sel(time=month)

        write_cog(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=color_map,
            output_path=os.path.join(base_dir, "rasters", f"{outfilename}.tif"),
            no_alpha=False,
        )

        data_dicts.append(
            {
                "year": str(int(month.dt.year.values)),
                "month": str(int(month.dt.month.values)),
                "url": f"{URL_PREFIX}/{version_number}/rasters/{outfilename}.tif",
            }
        )

    # 4. Generate json reference file
    # save the data_dicts as a json file

    with open(
        os.path.join(base_dir, "metadata.json"), "w"
    ) as f:
        json.dump(data_dicts, f, indent=4)

    # also make a json file for the color scale
    color_props = color_properties(color_map, (vmin, vmax), "µg/m³")
    with open(
        os.path.join(base_dir, "color_scale.json"), "w"
    ) as f:
        json.dump(color_props, f, indent=4)


    upload_files_to_gcs(
        local_directory=base_dir,
        bucket_id=BUCKET_ID,
        version_number=version_number,
    )


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
    vals = np.ones((n_discrete, 4)) * 256
    vals[0, 0:-1] = [39, 165, 156]  # #27a59c
    vals[1, 0:-1] = [205, 224, 180]
    vals[2, 0:-1] = [255, 229, 153]  # FFE599 (instead of fff2cc)
    vals[3, 0:-1] = [246, 178, 107]  # f6b26b
    vals[4, 0:-1] = [230, 93, 48]
    vals[5, 0:-1] = [204, 0, 0]  # #cc0000
    vals[6, 0:-1] = [153, 0, 0]  # #990000
    vals[7, 0:-1] = [116, 27, 71]  # #741b47
    vals[8, 0:-1] = [83, 32, 100]  # 532064
    vals[9, 0:-1] = [53, 28, 117]  # #351c75
    vals[10, 0:-1] = [27, 14, 59]  # #1b0e3b
    vals[11, 0:-1] = [0, 0, 0]  # #000000

    vals = vals / 256

    # Create discrete colormap and convert to continuous
    discrete_cmap = ListedColormap(vals)
    cmaplist = [discrete_cmap(i) for i in range(discrete_cmap.N)]

    return LinearSegmentedColormap.from_list("crea_custom_cont", cmaplist, N)


def color_properties(color_map, color_bounds, units: str, N_sample: int = 20):

    # generate colorscale for metadata, as a list of hex strings
    if isinstance(color_map, str):
        color_map = plt.get_cmap(color_map)

    # sample 20 colors from the colormap
    # (need to use integeres, so that color_map returns color
    # based on index in colormap and not color assigned to value)
    color_scale = [
        color_map(x) for x in np.linspace(0, color_map.N, N_sample).astype(int)
    ]
    # convert to hex
    color_scale = [mcolors.rgb2hex(c) for c in color_scale]

    # create metadata dictionary
    metadata = {
        "minval": round(color_bounds[0]),
        "maxval": round(color_bounds[1]),
        "unit": units,
        "scale": color_scale,
    }

    return metadata

def upload_files_to_gcs(local_directory: str, bucket_id: str, version_number: str):
    """Upload files from a local directory to a GCS bucket.

    Parameters
    ----------
    local_directory : str
        Path to the local directory containing the files to upload.
    bucket_id : str
        ID of the GCS bucket to upload to.
    version_number : str
        Version number to use as a prefix for the uploaded files in the bucket.
    """
    storage_client = storage.Client(
        project=PROJECT_ID
    )
    bucket = storage_client.bucket(bucket_id)

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            blob_path = f"{version_number}/{relative_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_id}/{blob_path}")
    

if __name__ == "__main__":

    # Take the first argument from the script arguments this was called with as the input file path.
    # Example would be like "data/input/pm25-full_rel-2025-11-r0_s-1.0_mb-1.0-r0.nc"

    def print_usage():
        print("Usage: python main.py <input_file>")
        print("Version number will be extracted from the input file name, so it should be in the format: pm25-full_rel-2025-11-r0_s-1.0_mb-1.0-r0.nc")

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    if len(sys.argv) > 2:
        print_usage()
        sys.exit(1)

    input_file = sys.argv[1]
    version_number = input_file.split("/")[-1].replace(".nc", "")

    main(
        input_file=input_file,
        version_number=version_number,
    )
