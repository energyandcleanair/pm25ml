"""Data source for the measurements and stations."""

import ast
import threading

import polars as pl
from arrow import Arrow

from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig

BASE_URI = "https://api.energyandcleanair.org"


class CreaMeasurementsApiDataSource:
    """Data source for CREA measurements and stations."""

    def __init__(
        self,
        temporal_config: TemporalConfig,
        source_ids: tuple[str, ...],
    ) -> None:
        """Initialize the data source."""
        self._station_stats_cache: pl.DataFrame | None = None
        self._stations_cache: pl.DataFrame | None = None

        self.temporal_config = temporal_config
        self.source_ids = source_ids
        self._station_stats_lock = threading.Lock()
        self._stations_lock = threading.Lock()

    def fetch_station_stats(self) -> pl.DataFrame:
        """
        Fetch station statistics for a given date range.

        The results will be cached in memory for the instance for subsequent calls.
        """
        with self._station_stats_lock:
            if self._station_stats_cache is not None:
                logger.info(
                    "Using cached station stats for sources %s",
                    self._source_query_value,
                )
                return self._station_stats_cache

            logger.info("Building station stats for sources %s", self._source_query_value)

            # Generate a URL per month between min_date and max_date. The date_to value
            # is inclusive, not exclusive

            month_ranges = [
                (
                    m.format("YYYY-MM-DD"),
                    m.shift(months=1).shift(days=-1).format("YYYY-MM-DD"),
                )
                for m in self.temporal_config.months
            ]

            measurements_urls = [
                (
                    f"{BASE_URI}/v1/measurements"
                    "?format=csv"
                    "&process_id=station_day_mad"
                    f"&date_from={start}"
                    f"&date_to={end}"
                    f"&source={self._source_query_value}"
                    "&pollutant=pm25"
                )
                for start, end in month_ranges
            ]

            # We want the q1 per station, and q3 per station, along with the IQR.
            results = (
                pl.scan_csv(measurements_urls)
                .select(
                    "location_id",
                    "value",
                )
                .group_by("location_id")
                .agg(
                    [
                        pl.col("value").quantile(0.25).alias("station_q1"),
                        pl.col("value").quantile(0.75).alias("station_q3"),
                    ],
                )
                .with_columns((pl.col("station_q3") - pl.col("station_q1")).alias("station_iqr"))
            )

            self._station_stats_cache = results.collect()
            return self._station_stats_cache

    def fetch_stations(self) -> pl.DataFrame:
        """
        Fetch station information for the configured profile.

        The results will be cached in memory for the instance for subsequent calls.
        """
        with self._stations_lock:
            if self._stations_cache is not None:
                logger.info("Using cached stations for sources %s", self._source_query_value)
                return self._stations_cache

            logger.info("Fetching stations for sources %s", self._source_query_value)

            url = (
                f"{BASE_URI}/stations?format=csv&source={self._source_query_value}"
                "&with_data_only=false"
            )

            station_data = pl.read_csv(url)

            # Safely parse the 'coordinates' column
            if "coordinates" in station_data.columns:
                station_data = station_data.with_columns(
                    coordinates=pl.col("coordinates").map_elements(
                        ast.literal_eval,
                        return_dtype=pl.Struct(
                            [
                                pl.Field("longitude", pl.Float64),
                                pl.Field("latitude", pl.Float64),
                            ],
                        ),
                    ),
                ).with_columns(
                    longitude=pl.col("coordinates").struct.field("longitude"),
                    latitude=pl.col("coordinates").struct.field("latitude"),
                )

            self._stations_cache = station_data.select(
                "id",
                "longitude",
                "latitude",
            )
            return self._stations_cache

    def fetch_station_data(self, start_date: Arrow, end_date: Arrow) -> pl.DataFrame:
        """Fetch station data for a given date range."""
        logger.info("Fetching station data for sources %s", self._source_query_value)

        start_formatted = start_date.format("YYYY-MM-DD")
        end_formatted = end_date.format("YYYY-MM-DD")

        measurements_url = (
            f"{BASE_URI}/v1/measurements"
            "?format=csv"
            "&process_id=station_day_mad"
            f"&date_from={start_formatted}"
            f"&date_to={end_formatted}"
            f"&source={self._source_query_value}"
            "&pollutant=pm25"
        )

        logger.info(
            "Fetching station data from URL: %s",
            measurements_url,
        )

        return pl.read_csv(measurements_url).with_columns(
            date=pl.col("date").cast(pl.Date),
            value=pl.col("value").cast(pl.Float32),
        )

    @property
    def _source_query_value(self) -> str:
        return ",".join(self.source_ids)
