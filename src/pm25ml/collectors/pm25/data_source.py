"""Data source for the measurements and stations."""

import ast
import threading
from collections.abc import Callable
from io import BytesIO
from urllib.parse import urlparse

import polars as pl
import requests
from arrow import Arrow

from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig

BASE_URI = "https://api.energyandcleanair.org"
CSV_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; pm25ml/0.1; +https://api.energyandcleanair.org)",
    "Accept": "text/csv,*/*;q=0.8",
}
CSV_REQUEST_TIMEOUT_SECONDS = 60


class CreaMeasurementsApiDataSource:
    """Data source for CREA measurements and stations."""

    def __init__(
        self,
        source_ids: tuple[str, ...],
    ) -> None:
        """Initialize the data source."""
        self.source_ids = source_ids

        self._station_stats_cache: dict[tuple[str, ...], pl.DataFrame] = {}
        self._station_stats_lock = threading.Lock()
        self._stations_cache: pl.DataFrame | None = None
        self._stations_lock = threading.Lock()

    def fetch_station_stats(self, temporal_config: TemporalConfig) -> pl.DataFrame:
        """
        Fetch station statistics for a given date range.

        The results will be cached in memory for the instance for subsequent calls.
        """

        def fetch_and_process() -> pl.DataFrame:
            # Generate a URL per month between min_date and max_date. The date_to value
            # is inclusive, not exclusive

            month_ranges = [
                (
                    m.format("YYYY-MM-DD"),
                    m.shift(months=1).shift(days=-1).format("YYYY-MM-DD"),
                )
                for m in temporal_config.months
            ]

            measurements_urls = [
                self._build_measurements_url(start, end) for start, end in month_ranges
            ]

            # We want the q1 per station, and q3 per station, along with the IQR.
            return (
                self._read_csvs_from_urls(measurements_urls)
                .lazy()
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
                .collect()
            )

        cache_key = tuple(temporal_config.month_ids)
        with self._station_stats_lock:
            if cache_key not in self._station_stats_cache:
                logger.info("Fetching station stats")
                self._station_stats_cache[cache_key] = fetch_and_process()
            return self._station_stats_cache[cache_key]

    def fetch_stations(self) -> pl.DataFrame:
        """
        Fetch station information for the configured profile.

        The results will be cached in memory for the instance for subsequent calls.
        """

        def fetch_and_process() -> pl.DataFrame:
            url = (
                f"{BASE_URI}/stations?format=csv&source={self._source_query_value}"
                "&with_data_only=false"
            )

            station_data = self._read_csv_from_url(url)

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

            return station_data.select(
                "id",
                "longitude",
                "latitude",
            )

        return self._get_cached_data(
            cache_attr="_stations_cache",
            lock=self._stations_lock,
            fetch_fn=fetch_and_process,
            cache_name="stations",
        )

    def fetch_station_data(self, start_date: Arrow, end_date: Arrow) -> pl.DataFrame:
        """Fetch station data for a given date range."""
        logger.info("Fetching station data for sources %s", self._source_query_value)

        start_formatted = start_date.format("YYYY-MM-DD")
        end_formatted = end_date.format("YYYY-MM-DD")

        measurements_url = self._build_measurements_url(start_formatted, end_formatted)

        logger.info(
            "Fetching station data from URL: %s",
            measurements_url,
        )

        return self._read_csv_from_url(measurements_url).with_columns(
            date=pl.col("date").cast(pl.Date),
            value=pl.col("value").cast(pl.Float32),
        )

    def _build_measurements_url(self, date_from: str, date_to: str) -> str:
        """
        Build a CREA API URL for measurements data.

        Args:
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format

        Returns:
            Full URL for measurements API endpoint

        """
        return (
            f"{BASE_URI}/v1/measurements"
            "?format=csv"
            "&process_id=station_day_mad"
            f"&date_from={date_from}"
            f"&date_to={date_to}"
            f"&source={self._source_query_value}"
            "&pollutant=pm25"
        )

    def _read_csvs_from_urls(self, urls: list[str]) -> pl.DataFrame:
        frames = [self._read_csv_from_url(url) for url in urls]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed")

    @staticmethod
    def _read_csv_from_url(url: str) -> pl.DataFrame:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "api.energyandcleanair.org":
            msg = f"Unsupported CREA API URL: {url}"
            raise ValueError(msg)

        response = requests.get(
            url,
            headers=CSV_REQUEST_HEADERS,
            timeout=CSV_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return pl.read_csv(BytesIO(response.content))

    @property
    def _source_query_value(self) -> str:
        return ",".join(self.source_ids)

    def _get_cached_data(
        self,
        cache_attr: str,
        lock: threading.Lock,
        fetch_fn: Callable[[], pl.DataFrame],
        cache_name: str,
    ) -> pl.DataFrame:
        """
        Fetch data with caching and thread safety.

        Args:
            cache_attr: Name of the cache attribute (e.g., "_station_stats_cache")
            lock: Threading lock to use for synchronization
            fetch_fn: Callable that performs data fetching and transformation
            cache_name: Human-readable name for logging (e.g., "station stats")

        Returns:
            Cached or freshly fetched DataFrame

        """
        with lock:
            cached_data = getattr(self, cache_attr)
            if cached_data is not None:
                logger.info(
                    "Using cached %s for sources %s",
                    cache_name,
                    self._source_query_value,
                )
                return cached_data

            logger.info("Fetching %s for sources %s", cache_name, self._source_query_value)

            data = fetch_fn()
            setattr(self, cache_attr, data)
            return data
