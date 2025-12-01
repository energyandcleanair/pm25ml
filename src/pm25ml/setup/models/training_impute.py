"""Builds the model references for training data."""

from typing import Callable

import polars as pl
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from pm25ml.model_reference import ImputationModelReference
from pm25ml.training.types import ModelName

SHARED_IMPUTATION_PREDICTOR_COLS = [
    "merra_aot__aot",
    "merra_co_top__co",
    "merra_co__co",
    "era5_land__v_component_of_wind_10m",
    "era5_land__u_component_of_wind_10m",
    "era5_land__total_precipitation_sum",
    "era5_land__temperature_2m",
    "era5_land__surface_pressure",
    "era5_land__surface_net_thermal_radiation_sum",
    "era5_land__leaf_area_index_low_vegetation",
    "era5_land__leaf_area_index_high_vegetation",
    "era5_land__dewpoint_temperature_2m",
    "srtm_elevation__elevation",
    "modis_land_cover__water",
    "modis_land_cover__shrub",
    "modis_land_cover__urban",
    "modis_land_cover__forest",
    "modis_land_cover__savanna",
    "month_of_year",
    "day_of_year",
    "cos_day_of_year",
    "monsoon_season",
    "grid__lon",
    "grid__lat",
    "era5_land__wind_degree_computed",
    "era5_land__relative_humidity_computed",
    "merra_aot__aot__mean_r7d",
    "merra_co_top__co__mean_r7d",
    "omi_no2__no2__mean_r7d",
    "era5_land__v_component_of_wind_10m__mean_r7d",
    "era5_land__u_component_of_wind_10m__mean_r7d",
    "era5_land__total_precipitation_sum__mean_r7d",
    "era5_land__temperature_2m__mean_r7d",
    "era5_land__wind_degree_computed__mean_r7d",
    "era5_land__relative_humidity_computed__mean_r7d",
    "era5_land__surface_net_thermal_radiation_sum__mean_r7d",
    "era5_land__dewpoint_temperature_2m__mean_r7d",
    "merra_aot__aot__mean_year",
    "merra_co_top__co__mean_year",
    "omi_no2__no2__mean_year",
    "era5_land__v_component_of_wind_10m__mean_year",
    "era5_land__u_component_of_wind_10m__mean_year",
    "era5_land__total_precipitation_sum__mean_year",
    "era5_land__surface_net_thermal_radiation_sum__mean_year",
    "era5_land__leaf_area_index_low_vegetation__mean_year",
    "era5_land__leaf_area_index_high_vegetation__mean_year",
    "era5_land__dewpoint_temperature_2m__mean_year",
    "era5_land__wind_degree_computed__mean_year",
    "era5_land__relative_humidity_computed__mean_year",
    "merra_co_top__co__mean_all",
]

SHARED_IMPUTATION_GROUPER_COL = "grid__id_50km"


def build_model_ref(
    *,
    ref: ModelName,
    extra_sampler: Callable[[pl.LazyFrame], pl.LazyFrame],
    take_mini_training_sample: bool,
) -> ImputationModelReference:
    """Build a training data reference for the given ref."""
    if ref == "aod":
        return ImputationModelReference(
            model_name="aod",
            predictor_cols=SHARED_IMPUTATION_PREDICTOR_COLS,
            target_col="modis_aod__Optical_Depth_055",
            grouper_col=SHARED_IMPUTATION_GROUPER_COL,
            model_builder=lambda: XGBRegressor(
                eta=0.1,
                gamma=0.8,
                max_depth=20,
                min_child_weight=1,
                subsample=0.8,
                reg_lambda=100,
                n_estimators=1000,
                booster="gbtree",
            ),
            extra_sampler=extra_sampler,
            min_r2_score=0.4 if take_mini_training_sample else 0.8,
            max_r2_score=0.9,
        )
    if ref == "no2":
        return ImputationModelReference(
            model_name="no2",
            predictor_cols=SHARED_IMPUTATION_PREDICTOR_COLS,
            target_col="s5p_no2__tropospheric_NO2_column_number_density",
            grouper_col=SHARED_IMPUTATION_GROUPER_COL,
            model_builder=lambda: LGBMRegressor(
                boosting="gbdt",
                lambda_l2=10,
                learning_rate=0.1,
                max_bin=500,
                max_depth=10,
                min_data_in_leaf=10,
                num_iterations=3000,
                num_leaves=1500,
                objective="regression",
            ),
            extra_sampler=extra_sampler,
            min_r2_score=-0.1 if take_mini_training_sample else 0.4,
            max_r2_score=0.6,
        )
    if ref == "co":
        return ImputationModelReference(
            model_name="co",
            predictor_cols=SHARED_IMPUTATION_PREDICTOR_COLS,
            target_col="s5p_co__CO_column_number_density",
            grouper_col=SHARED_IMPUTATION_GROUPER_COL,
            model_builder=lambda: LGBMRegressor(
                boosting="gbdt",
                lambda_l2=10,
                learning_rate=0.1,
                max_bin=1000,
                max_depth=10,
                min_data_in_leaf=10,
                num_iterations=3000,
                num_leaves=1500,
                objective="regression",
            ),
            extra_sampler=extra_sampler,
            min_r2_score=-0.4 if take_mini_training_sample else 0.9,
            max_r2_score=0.97,
        )

    msg = f"Unknown model reference: {ref}"
    raise ValueError(msg)
