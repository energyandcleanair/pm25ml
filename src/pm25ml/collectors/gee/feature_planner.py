"""Feature planning for gridded feature collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from ee.ee_date import Date
from ee.ee_list import List
from ee.image import Image
from ee.imagecollection import ImageCollection
from ee.reducer import Reducer

from pm25ml.collectors.constants import INDIA_CRS
from pm25ml.collectors.export_pipeline import AVAILABLE_ID_KEY_NAMES, AvailableIdKeys

if TYPE_CHECKING:
    from arrow import Arrow
    from ee.computedobject import ComputedObject
    from ee.ee_number import Number
    from ee.element import Element
    from ee.featurecollection import FeatureCollection

ISO8601_WITHOUT_TZ = "YYYY-MM-DDTHH:mm:ss"
ISO8601_DATE_ONLY = "YYYY-MM-DD"


class GriddedFeatureCollectionPlanner:
    """
    A planner for creating gridded feature collections from Earth Engine data.

    This class provides methods to process Earth Engine data into gridded feature collections
    for various use cases, such as daily averages, static features, and annual summaries of
    classified pixels.

    These may skip days if no data is available for that day, so the
    resulting feature collection may not have a row for every day in the month.

    Reference: how the "scale" arguments works in Earth Engine: https://developers.google.com/earth-engine/guides/scale
    """

    def __init__(self, grid: FeatureCollection) -> None:
        """
        Initialize the planner with a grid.

        :param grid: The grid to which the features will be mapped.
        :type grid: FeatureCollection
        """
        self.grid = grid
        self._n_grids: int | None = None

    def plan_daily_average(
        self,
        *,
        collection_name: str,
        selected_bands: list[str],
        dates: list[Arrow],
    ) -> FeaturePlan:
        """
        Create a daily average feature plan.

        Aggregate pixel-wise mean values for each day and mapping them to the grid.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :param selected_bands: The bands to include in the aggregation.
        :type selected_bands: list[str]
        :param dates: The list of dates for which daily averages are computed.
        :type dates: list[Arrow]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        if not dates:
            msg = "Expected at least one date for daily averaging."
            raise ValueError(msg)

        original_raster_scale = self._get_collection_scale(collection_name, selected_bands)

        ids = ["date", "grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = dict(zip(exported_properties, wanted_properties))

        date_strings = [date.format(ISO8601_DATE_ONLY) for date in dates]
        gee_dates = List(date_strings)
        date_window_start = Date(min(date_strings))
        date_window_end = Date(max(date_strings)).advance(1, "day")
        grid_geometry = self.grid.geometry()

        # Prefilter once by location and date window so daily map work starts from a smaller set.
        collection = (
            ImageCollection(collection_name)
            .select(selected_bands)
            .filterBounds(grid_geometry)
            .filterDate(date_window_start, date_window_end)
        )

        # Annotate each image with a day key, then deduplicate days before
        # collecting available dates to reduce metadata transfer.
        collection_by_day = collection.map(
            lambda im: im.set(
                "day",
                Date(im.get("system:time_start")).format(ISO8601_DATE_ONLY),
            )
        )

        # Keep only requested dates that actually have source imagery.
        available_dates = List(collection_by_day.distinct("day").aggregate_array("day")).sort()
        # ee.List has no intersection() in Python; emulate requested ∩ available while
        # preserving the original requested-date order.
        unavailable_dates = gee_dates.removeAll(available_dates)
        dates_to_process = gee_dates.removeAll(unavailable_dates)

        def daily_mean_image(date_string: ComputedObject) -> Image:
            start = Date(date_string)
            end = start.advance(1, "day")

            return (
                collection.filterDate(start, end)
                # Single value per pixel for the day for each band.
                .reduce(Reducer.mean())
                # We set the date property to the date to carry through
                # to the final export.
                .set("date", start)
            )

        # We create an ImageCollection of daily composites for the month, each
        # the pixel-wise mean value for the day.
        images = ImageCollection.fromImages(
            # This does a server side map operation to create an image for each date.
            dates_to_process.map(daily_mean_image),
        )

        # We then average the values for each grid cell for each date.
        def average_grid_value_for_date(im: Image) -> FeatureCollection:
            image_date = im.get("date")

            def carry_date_through(f: Image) -> Element:
                return f.set("date", image_date)

            return im.reduceRegions(
                collection=self.grid,
                # Single value per grid cell for the day.
                reducer=Reducer.mean(),
                crs=INDIA_CRS,
                scale=original_raster_scale,
            ).map(carry_date_through)

        processed_images: FeatureCollection = images.map(average_grid_value_for_date).flatten()

        date_summary = self._common_granularity(dates)

        return FeaturePlan(
            feature_name=self._generate_clean_name(
                "grid-daily-average",
                collection_name,
                date_summary,
            ),
            planned_collection=processed_images,
            column_mappings=column_mappings,
            expected_n_rows=self._get_n_grids() * len(dates),
            dates=dates,
        )

    def plan_static_feature(
        self,
        *,
        image_name: str,
        selected_bands: list[str],
    ) -> FeaturePlan:
        """
        Create a static feature plan by regridding a single image to the grid.

        :param image_name: The name of the Earth Engine image.
        :type image_name: str
        :param selected_bands: The bands to include in the regridding.
        :type selected_bands: list[str]
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_image_scale(image_name, selected_bands)

        ids = ["grid_id"]
        transformed_band_names = (
            [f"{band}_mean" for band in selected_bands] if len(selected_bands) > 1 else ["mean"]
        )
        exported_properties = ids + transformed_band_names
        wanted_properties = ids + selected_bands
        column_mappings = dict(zip(exported_properties, wanted_properties))

        image = Image(image_name).select(selected_bands)
        # For a single image, reduce directly instead of wrapping in a one-element collection.
        processed_image: FeatureCollection = image.reduceRegions(
            collection=self.grid,
            reducer=Reducer.mean(),
            crs=INDIA_CRS,
            scale=original_raster_scale,
        )

        return FeaturePlan(
            feature_name=self._generate_clean_name("single-image-grid", image_name),
            planned_collection=processed_image,
            column_mappings=column_mappings,
            expected_n_rows=self._get_n_grids(),
        )

    def plan_summarise_annual_classified_pixels(
        self,
        *,
        collection_name: str,
        classification_band: str,
        output_names_to_class_values: dict[str, list[int]],
        year: int,
    ) -> FeaturePlan:
        """
        Create a feature plan for summarizing annual classified pixel data.

        Aggregate by creating boolean bands for each class and taking the mean
        over the year.

        This will give, for each grid cell in the specified year, the percentage
        of time that the pixels in that cell were classified as each of the
        specified classes.

        :param collection_name: The name of the Earth Engine image collection.
        :type collection_name: str
        :param classification_band: The band containing classification data.
        :type classification_band: str
        :param output_names_to_class_values: A mapping of output names to class values.
        :type output_names_to_class_values: dict[str, int]
        :param year: The year for which the summary is computed.
        :type year: int
        :return: A feature plan containing the processed collection and column mappings.
        :rtype: FeaturePlan
        """
        original_raster_scale = self._get_collection_scale(collection_name, [classification_band])

        if not output_names_to_class_values:
            msg = "Expected at least one output class mapping."
            raise ValueError(msg)

        expected_column_names = list(output_names_to_class_values.keys())
        class_output_items = list(output_names_to_class_values.items())
        year_start = f"{year}-01-01T00:00:00"
        year_end = f"{year + 1}-01-01T00:00:00"
        grid_geometry = self.grid.geometry()

        def add_classes_as_boolean_bands(original_image: Image) -> Image:
            band_column = original_image.select(classification_band)

            first_output_name, first_class_values = class_output_items[0]
            derived_image = band_column.remap(
                first_class_values,
                [1] * len(first_class_values),
                0,
                classification_band,
            ).rename(first_output_name)

            for output_name, class_values in class_output_items[1:]:
                derived_image = derived_image.addBands(
                    band_column.remap(
                        class_values,
                        [1] * len(class_values),
                        0,
                        classification_band,
                    ).rename(output_name),
                    [output_name],
                )

            # Keep only the derived boolean class bands to reduce memory pressure.
            return derived_image

        images = (
            ImageCollection(collection_name)
            .filterBounds(grid_geometry)
            .filterDate(year_start, year_end)
            .select(classification_band)
        )

        image = Image(
            images.map(add_classes_as_boolean_bands)
            .select(expected_column_names)
            .reduce(Reducer.mean()),
        )

        # We create an image for the year by filtering the collection
        # and then taking the mean

        collection_for_year = image.reduceRegions(
            collection=self.grid,
            reducer=Reducer.mean(),
            crs=INDIA_CRS,
            scale=original_raster_scale,
        )

        flattened = collection_for_year

        id_columns = ["grid_id"]
        wanted_columns = id_columns + expected_column_names
        exported_columns = id_columns + [
            f"{output_name}_mean" for output_name in output_names_to_class_values
        ]

        column_mappings = dict(zip(exported_columns, wanted_columns))

        def availability_checker() -> bool:
            """
            Check if the feature collection is available for the specified year.

            :return: True if the feature collection is available, False otherwise.
            :rtype: bool
            """
            return images.size().getInfo() >= 1

        return FeaturePlan(
            feature_name=self._generate_clean_name(
                "annual-classified-pixels",
                collection_name,
                str(year),
            ),
            planned_collection=flattened,
            column_mappings=column_mappings,
            expected_n_rows=self._get_n_grids(),
            availability_checker=availability_checker,
        )

    @staticmethod
    def _get_collection_scale(collection_name: str, selected_bands: list[str]) -> Number:
        return (
            ImageCollection(collection_name)
            .select(selected_bands)
            .first()
            .projection()
            .nominalScale()
        )

    @staticmethod
    def _get_image_scale(image_name: str, selected_bands: list[str]) -> Number:
        return Image(image_name).select(selected_bands).projection().nominalScale()

    @staticmethod
    def _generate_clean_name(*args: str) -> str:
        def clean_name_part(name: str) -> str:
            return name.replace(" ", "-").replace("/", "-").replace("_", "-").lower()

        return "__".join(clean_name_part(arg) for arg in args)

    @staticmethod
    def _common_granularity(dates: list[Arrow]) -> str:
        same_year = all(d.year == dates[0].year for d in dates)
        if not same_year:
            return "x"

        same_month = all(d.month == dates[0].month for d in dates)
        if not same_month:
            return dates[0].format("YYYY")

        same_day = all(d.day == dates[0].day for d in dates)
        if not same_day:
            return dates[0].format("YYYY-MM")

        return dates[0].format("YYYY-MM-DD")

    def _get_n_grids(
        self,
    ) -> int:
        if self._n_grids is None:
            self._n_grids = cast("int", self.grid.size().getInfo())
        return self._n_grids


@dataclass(
    frozen=True,
)
class FeaturePlan:
    """
    Represents a plan for processing and exporting features.

    This class encapsulates the proposed feature collection, column mappings,
    and metadata.
    """

    feature_name: str
    planned_collection: FeatureCollection
    column_mappings: dict[str, str]
    expected_n_rows: int
    ignore_selectors: bool = False
    dates: list[Arrow] | None = None
    availability_checker: Callable[[], bool] | None = None

    @property
    def intermediate_columns(self) -> list[str]:
        """
        Return the columns that will be exported to the intermediate storage.

        :return: The keys of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.keys())

    @property
    def wanted_columns(self) -> list[str]:
        """
        Return the columns that are wanted in the final export.

        :return: The values of the column_mappings dictionary.
        :rtype: list[str]
        """
        return list(self.column_mappings.values())

    @property
    def expected_id_columns(self) -> set[AvailableIdKeys]:
        """
        Return the expected ID columns in the result.

        :return: A set of expected ID columns.
        :rtype: set[AvailableIdKeys]
        """
        return {
            cast("AvailableIdKeys", key)
            for key in self.column_mappings
            if key in AVAILABLE_ID_KEY_NAMES
        }

    @property
    def expected_value_columns(self) -> set[str]:
        """
        Return the expected value columns in the result.

        :return: A set of expected value columns.
        :rtype: set[str]
        """
        return {
            value
            for key, value in self.column_mappings.items()
            if key not in AVAILABLE_ID_KEY_NAMES
        }

    def is_data_available(self) -> bool:
        """
        Check if the feature collection is available.

        If an availability checker is defined, it will be used to check
        the availability of the feature collection.

        :return: True if the feature collection is available, False otherwise.
        :rtype: bool
        """
        if self.availability_checker:
            return self.availability_checker()
        return True
