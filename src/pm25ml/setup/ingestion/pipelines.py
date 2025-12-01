"""A module for defining export pipelines for the PM2.5 ML project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arrow import Arrow

from pm25ml.collectors.export_pipeline import (
    MissingDataHeuristic,
    PipelineConsumerBehaviour,
)
from pm25ml.collectors.misc.grid_export_pipeline import GridExportPipeline
from pm25ml.collectors.ned.data_reader_merra import MerraDataReader
from pm25ml.collectors.ned.data_reader_omno2d import Omno2dReader
from pm25ml.collectors.ned.data_retriever_harmony import HarmonySubsetterDataRetriever
from pm25ml.collectors.ned.data_retriever_raw import RawEarthAccessDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from pm25ml.collectors.archive_storage import IngestArchiveStorage
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.collectors.gee.feature_planner import GriddedFeatureCollectionPlanner
    from pm25ml.collectors.gee.gee_export_pipeline import GeePipelineConstructor
    from pm25ml.collectors.grid import Grid
    from pm25ml.collectors.ned.ned_export_pipeline import NedPipelineConstructor
    from pm25ml.collectors.pm25.pm25_pipeline import Pm25MeasurementsPipelineConstructor
    from pm25ml.setup.temporal_config import TemporalConfig


MODIS_LAND_ALLOW_MISSING_FROM_YEAR = Arrow.now().year - 2


def define_pipelines(  # noqa: PLR0913
    *,
    gee_pipeline_constructor: GeePipelineConstructor,
    ned_pipeline_constructor: NedPipelineConstructor,
    pm25_pipeline_constructor: Pm25MeasurementsPipelineConstructor,
    in_memory_grid: Grid,
    archive_storage: IngestArchiveStorage,
    feature_planner: GriddedFeatureCollectionPlanner,
    temporal_config: TemporalConfig,
) -> Collection[ExportPipeline]:
    """Define export pipelines for the PM2.5 ML project."""

    def _static_pipelines() -> Iterable[ExportPipeline]:
        """Fetch static datasets that do not change over time."""
        return [
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_static_feature(
                    image_name="USGS/SRTMGL1_003",
                    selected_bands=["elevation"],
                ),
                result_subpath="country=india/dataset=srtm_elevation/type=static",
            ),
            GridExportPipeline(
                grid=in_memory_grid,
                archive_storage=archive_storage,
                result_subpath="country=india/dataset=grid/type=static",
            ),
        ]

    def _yearly_pipelines(year: int) -> Iterable[ExportPipeline]:
        """Fetch datasets that are aggregated yearly."""
        return [
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_summarise_annual_classified_pixels(
                    collection_name="MODIS/061/MCD12Q1",
                    classification_band="LC_Type1",
                    output_names_to_class_values={
                        "forest": [1, 2, 3, 4, 5],
                        "shrub": [6, 7],
                        "savanna": [9],
                        "urban": [13],
                        "water": [17],
                    },
                    year=year,
                ),
                result_subpath=f"country=india/dataset=modis_land_cover/year={year}",
                pipeline_consumer_behaviour=(
                    PipelineConsumerBehaviour(
                        missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE,
                    )
                    if year > MODIS_LAND_ALLOW_MISSING_FROM_YEAR
                    else PipelineConsumerBehaviour.default()
                ),
            ),
        ]

    def _monthly_pipelines(month_start: Arrow) -> Iterable[ExportPipeline]:
        """Fetch datasets that are aggregated monthly."""
        month_end = month_start.shift(months=1).shift(days=-1)

        dates_in_month: list[Arrow] = list(
            Arrow.range("day", start=month_start, end=month_end),
        )

        month_short = month_start.format("YYYY-MM")

        return [
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="COPERNICUS/S5P/OFFL/L3_CO",
                    selected_bands=["CO_column_number_density"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=s5p_co/month={month_short}",
            ),
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="COPERNICUS/S5P/OFFL/L3_NO2",
                    selected_bands=["tropospheric_NO2_column_number_density"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=s5p_no2/month={month_short}",
            ),
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="ECMWF/ERA5_LAND/DAILY_AGGR",
                    selected_bands=[
                        "temperature_2m",
                        "dewpoint_temperature_2m",
                        "u_component_of_wind_10m",
                        "v_component_of_wind_10m",
                        "total_precipitation_sum",
                        "surface_net_thermal_radiation_sum",
                        "surface_pressure",
                        "leaf_area_index_high_vegetation",
                        "leaf_area_index_low_vegetation",
                    ],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=era5_land/month={month_short}",
            ),
            gee_pipeline_constructor.construct(
                plan=feature_planner.plan_daily_average(
                    collection_name="MODIS/061/MCD19A2_GRANULES",
                    selected_bands=["Optical_Depth_047", "Optical_Depth_055"],
                    dates=dates_in_month,
                ),
                result_subpath=f"country=india/dataset=modis_aod/month={month_short}",
            ),
            ned_pipeline_constructor.construct(
                dataset_descriptor=NedDatasetDescriptor(
                    # https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary
                    dataset_name="M2T1NXAER",
                    dataset_version="5.12.4",
                    start_date=month_start,
                    end_date=month_end,
                    filter_bounds=in_memory_grid.expanded_bounds,
                    variable_mapping={
                        "TOTEXTTAU": "aot",
                    },
                    level=None,
                ),
                dataset_reader=MerraDataReader(),
                dataset_retriever=HarmonySubsetterDataRetriever(),
                result_subpath=f"country=india/dataset=merra_aot/month={month_short}",
            ),
            ned_pipeline_constructor.construct(
                dataset_descriptor=NedDatasetDescriptor(
                    # https://cmr.earthdata.nasa.gov/search/concepts/C1276812901-GES_DISC.html
                    dataset_name="M2I3NVCHM",
                    dataset_version="5.12.4",
                    start_date=month_start,
                    end_date=month_end,
                    filter_bounds=in_memory_grid.expanded_bounds,
                    variable_mapping={
                        "CO": "co",
                    },
                    level=-1,
                ),
                dataset_reader=MerraDataReader(),
                dataset_retriever=HarmonySubsetterDataRetriever(),
                result_subpath=f"country=india/dataset=merra_co/month={month_short}",
            ),
            ned_pipeline_constructor.construct(
                dataset_descriptor=NedDatasetDescriptor(
                    # https://cmr.earthdata.nasa.gov/search/concepts/C1276812901-GES_DISC.html
                    dataset_name="M2I3NVCHM",
                    dataset_version="5.12.4",
                    start_date=month_start,
                    end_date=month_end,
                    filter_bounds=in_memory_grid.expanded_bounds,
                    variable_mapping={
                        "CO": "co",
                    },
                    level=0,
                ),
                dataset_reader=MerraDataReader(),
                dataset_retriever=HarmonySubsetterDataRetriever(),
                result_subpath=f"country=india/dataset=merra_co_top/month={month_short}",
            ),
            ned_pipeline_constructor.construct(
                dataset_descriptor=NedDatasetDescriptor(
                    # https://cmr.earthdata.nasa.gov/searCch/concepts/C1266136111-GES_DISC.html
                    dataset_name="OMNO2d",
                    dataset_version="003",
                    start_date=month_start,
                    end_date=month_end,
                    filter_bounds=in_memory_grid.expanded_bounds,
                    variable_mapping={
                        "ColumnAmountNO2": "no2",
                    },
                    level=None,
                    interpolation_method="linear",
                ),
                dataset_reader=Omno2dReader(),
                dataset_retriever=RawEarthAccessDataRetriever(),
                result_subpath=f"country=india/dataset=omi_no2/month={month_short}",
            ),
            pm25_pipeline_constructor.construct(
                result_subpath=f"country=india/dataset=pm25/month={month_short}",
                month=month_start,
            ),
        ]

    yearly_pipelines = [
        pipeline
        for year in temporal_config.years
        for pipeline in _yearly_pipelines(year)
    ]
    monthly_pipelines = [
        pipeline
        for month in temporal_config.months
        for pipeline in _monthly_pipelines(month)
    ]
    static_pipelines = _static_pipelines()

    return [
        *reversed(yearly_pipelines),
        *reversed(monthly_pipelines),
        *static_pipelines,
    ]
