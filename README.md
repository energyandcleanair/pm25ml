# southasia-pm2-5-website-data-generator

Generates and uploads the COG and index files required for the [South Asia
PM2.5] website to display the PM2.5 data.

This tool:
1. Takes an input file
2. Generate the needed cloud optimised geotiffs for the widget on the
 website
3. Uploads them to a new directory in the bucket

> ![IMPORTANT]
>
> This will not replace the active index and colour map files. To do this, you
> must copy the latest files over the existing ones.
>
> To do this, use the release script as defined below

## Usage

Requirements:
- Install python
- Install poetry
- Install dependencies (`poetry install`)
- Be authenticated with Google Cloud

Then run the following, where the `<input_file>` is the resulting NetCDF file
that you want to generate the results for.
```
poetry run python main.py <input_file>
```

Once you're ready for release, use the `release` script:
```
./release <version_number>
```

[South Asia PM2.5]: https://southasia-pm2-5.energyandcleanair.org
