High-resolution surface PM2.5 concentration for India, built from satellite products, surface
monitors, and machine learning pipelines. This dataset is produced using a reimplementation of
*[Improved daily PM2.5 estimates in India reveal inequalities in recent enhancement of air
quality](https://www.science.org/doi/10.1126/sciadv.adq1071)* (Kawano, et al., 2025) built by the
[Centre for Research on Energy and Clean Air](https://energyandcleanair.org/) (CREA). See
[southasia-pm2-5.energyandcleanair.org](http://southasia-pm2-5.energyandcleanair.org) for more
details.

You can download daily PM2.5 data from the original authors for 1 January 2005 to 30 September 2023
from <https://zenodo.org/records/13694585>.

## Structure of the dataset

The dataset is provided in CF-1.8–compliant NetCDF format, containing daily surface PM2.5
concentrations (µg/m³) at 10 km resolution over India. It includes `time`, `x`, and `y` dimensions,
with the primary variable `pm25(time, y, x)`. The data uses the WGS 84 / India NSF Lambert
Conformal Conic projection (EPSG: 7755) to define spatial coordinates.

Filenames and the meaning of each version field are described in Technical info on this record.

## Model accuracy

We provide the results of the spatial cross-validation in a CSV file alongside the results. This has
the headers `r2` and `rmse` (in µg/m³). We use the same cross-validation metrics and methodology for
daily values as for the final full model by Kawano et al. (2025).

## Using the data

When using this data, please reference:

- Kawano, Ayako, et al. "Improved daily PM2.5 estimates in India reveal inequalities in recent
  enhancement of air quality." Science Advances 11.4 (2025): eadq1071.
  <https://doi.org/10.1126/sciadv.adq1071>
- The version of the data used from Zenodo.
