"""Data quality filters for PM2.5."""

import polars as pl

from pm25ml.collectors.pm25.pm25_pipeline import Pm25MeasurementFilterMarker

_REPEATING_DAYS_REQUIREMENT = 5

_REPEATING_IS_DUPLICATE_THRESHOLD = 0.05
_ANOMALY_IQR_TOO_HIGH_MULTIPLE = 15
_MAX_ALLOWABLE_VALUE = 999.99


def define_filters() -> list[Pm25MeasurementFilterMarker]:
    """Define the filters for the PM2.5 measurements pipeline."""
    return [
        RepeatingValuesMarker(),
        AnomalyMarker(),
        MaxValueMarker(),
    ]


class RepeatingValuesMarker(Pm25MeasurementFilterMarker):
    """Marker for repeating values in PM2.5 data."""

    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:
        """Mark repeating values with the label "dropped"."""
        rolling_average = (
            pl.col("value")
            .over("location_id")
            .rolling_mean(window_size=_REPEATING_DAYS_REQUIREMENT)
        )
        repeats_for_too_long = (
            pl.col("value") - rolling_average
        ).abs() < _REPEATING_IS_DUPLICATE_THRESHOLD

        return to_process_df.with_columns(
            label=_mark_matching_as_dropped(repeats_for_too_long),
        )

    @property
    def window_size(self) -> int:
        """Get the window size for the pipeline."""
        return _REPEATING_DAYS_REQUIREMENT


class AnomalyMarker(Pm25MeasurementFilterMarker):
    """Marker for anomalies in PM2.5 data."""

    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:
        """Mark anomalies with the label "dropped" using IQR."""
        is_too_high = pl.col("value") > pl.col("station_iqr") * _ANOMALY_IQR_TOO_HIGH_MULTIPLE

        return to_process_df.with_columns(
            label=_mark_matching_as_dropped(is_too_high),
        )

    @property
    def window_size(self) -> int:
        """Get the window size for the pipeline."""
        return 1


class MaxValueMarker(Pm25MeasurementFilterMarker):
    """Marker for max values in PM2.5 data."""

    def mark(self, to_process_df: pl.DataFrame) -> pl.DataFrame:
        """Mark max values with the label "dropped"."""
        is_too_high = pl.col("value") >= _MAX_ALLOWABLE_VALUE

        return to_process_df.with_columns(
            label=_mark_matching_as_dropped(is_too_high),
        )

    @property
    def window_size(self) -> int:
        """Get the window size for the pipeline."""
        return 1


def _mark_matching_as_dropped(expression: pl.Expr) -> pl.Expr:
    return pl.when(expression).then(pl.lit("drop")).otherwise(pl.col("label"))
