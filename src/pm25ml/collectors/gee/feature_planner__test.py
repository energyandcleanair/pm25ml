from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from arrow import get

from pm25ml.collectors.gee.feature_planner import (
    FeaturePlan,
    GriddedFeatureCollectionPlanner,
)

DUMMY_N_ROWS_VALUE = 100


def test__FeaturePlan_intermediate_columns__correct_keys__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
    )

    assert feature_plan.intermediate_columns == ["key1", "key2"]


def test__FeaturePlan_wanted_columns__correct_values__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"key1": "value1", "key2": "value2"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
    )

    assert feature_plan.wanted_columns == ["value1", "value2"]


def test__FeaturePlan_expected_id_columns__correct_values__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"date": "date", "grid_id": "grid_id", "key1": "value1"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
    )

    assert feature_plan.expected_id_columns == {"date", "grid_id"}


def test__FeaturePlan_expected_value_columns__correct_values__returned() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"date": "date", "grid_id": "grid_id", "key1": "value1"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
    )

    assert feature_plan.expected_value_columns == {"value1"}


def test__FeaturePlan_is_data_available__delegated_func_returns_true__true() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"date": "date", "grid_id": "grid_id", "key1": "value1"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
        availability_checker=MagicMock(return_value=True),
    )

    assert feature_plan.is_data_available() is True


def test__FeaturePlan_is_data_available__delegated_func_returns_false__false() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"date": "date", "grid_id": "grid_id", "key1": "value1"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
        availability_checker=MagicMock(return_value=False),
    )

    assert feature_plan.is_data_available() is False


def test__FeaturePlan_is_data_available__no_availability_checker__true() -> None:
    mock_feature_collection = MagicMock()
    column_mappings = {"date": "date", "grid_id": "grid_id", "key1": "value1"}
    feature_plan = FeaturePlan(
        feature_name="test-type",
        planned_collection=mock_feature_collection,
        column_mappings=column_mappings,
        expected_n_rows=DUMMY_N_ROWS_VALUE,
        availability_checker=None,
    )

    assert feature_plan.is_data_available() is True


@pytest.fixture
def mock_gee_for_daily_average() -> Iterator[dict[str, MagicMock]]:
    with (
        patch("pm25ml.collectors.gee.feature_planner.Algorithms") as MockAlgorithms,
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
        patch("pm25ml.collectors.gee.feature_planner.List") as MockList,
        patch("pm25ml.collectors.gee.feature_planner.Date") as MockDate,
    ):
        mock_ic_instance = MagicMock()
        MockImageCollection.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance

        fake_image = MagicMock()
        mock_ic_instance.filterDate.return_value = fake_image
        mock_ic_instance.size.return_value.gt.return_value = True
        fake_image.reduce.return_value = fake_image
        fake_image.set.return_value = fake_image
        fake_image.filterBounds.return_value = fake_image
        MockAlgorithms.If.return_value = fake_image

        from_images_mock = MagicMock()
        from_images_mock.map.return_value.flatten.return_value = MagicMock()
        MockImageCollection.fromImages.return_value = from_images_mock

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        mock_list_instance = MagicMock()
        MockList.return_value = mock_list_instance
        mock_list_instance.map.return_value = fake_image

        mock_date_instance_given = MagicMock()
        mock_date_instance_advanced = MagicMock()
        MockDate.return_value = mock_date_instance_given
        mock_date_instance_given.advance.return_value = mock_date_instance_advanced

        yield {
            "MockImageCollection": MockImageCollection,
            "MockAlgorithms": MockAlgorithms,
            "MockReducer": MockReducer,
            "MockImage": MockImage,
            "MockList": MockList,
            "mock_ic_instance": mock_ic_instance,
            "fake_image": fake_image,
            "from_images_mock": from_images_mock,
            "fake_mean": fake_mean,
            "mock_list_instance": mock_list_instance,
            "mock_date_instance_given": mock_date_instance_given,
            "mock_date_instance_advanced": mock_date_instance_advanced,
        }


def test__GriddedFeatureCollectionPlanner_plan_daily_average__columns_specified__correct_columns_suggested(
    mock_gee_for_daily_average,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    selected_bands = ["AOD", "PM25"]
    date = get("2023-04-01")

    planner.plan_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=selected_bands,
        dates=[date],
    )

    # Check that we selected the correct bands in all calls
    mock_gee_for_daily_average["mock_ic_instance"].select.assert_called_with(
        selected_bands,
    )


@pytest.mark.parametrize(
    ("bands", "expected_column_mappings"),
    [
        (
            ["PM25"],
            {"date": "date", "grid_id": "grid_id", "mean": "PM25"},
        ),
        (
            ["NO2", "SO2"],
            {
                "date": "date",
                "grid_id": "grid_id",
                "NO2_mean": "NO2",
                "SO2_mean": "SO2",
            },
        ),
    ],
    ids=["single_band", "multiple_bands"],
)
@pytest.mark.usefixtures("mock_gee_for_daily_average")
def test__GriddedFeatureCollectionPlanner_plan_daily_average__with_bands__correct_column_mappings_specified(
    bands,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    result = planner.plan_daily_average(
        collection_name="ANY",
        selected_bands=bands,
        dates=[get("2023-01-01", "YYYY-MM-DD")],
    )

    assert result.column_mappings == expected_column_mappings


def test__GriddedFeatureCollectionPlanner_plan_daily_average__missing_days__drops_none_images(
    mock_gee_for_daily_average,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)

    planner.plan_daily_average(
        collection_name="FAKE/COLLECTION",
        selected_bands=["AOD"],
        dates=[get("2023-04-01")],
    )

    fake_image = mock_gee_for_daily_average["fake_image"]
    fake_image.removeAll.assert_called_once_with([None])
    mock_gee_for_daily_average["MockImageCollection"].fromImages.assert_called_once_with(
        fake_image.removeAll.return_value,
    )


@pytest.fixture
def mock_gee_for_static_feature():
    with (
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
    ):
        mock_image_instance = MagicMock()
        MockImage.return_value = mock_image_instance
        mock_image_instance.select.return_value = mock_image_instance

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        mock_feature_collection_instance = MagicMock()
        mock_image_instance.reduceRegions.return_value = mock_feature_collection_instance

        mock_image_collection_instance = MagicMock()
        MockImageCollection.fromImages.return_value = mock_image_collection_instance

        yield {
            "MockImage": MockImage,
            "MockReducer": MockReducer,
            "MockImageCollection": MockImageCollection,
            "mock_image_instance": mock_image_instance,
            "mock_image_collection_instance": mock_image_collection_instance,
            "mock_feature_collection_instance": mock_feature_collection_instance,
            "fake_mean": fake_mean,
        }


def test__GriddedFeatureCollectionPlanner_static_feature__columns_specified__correct_columns_suggested(
    mock_gee_for_static_feature,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    selected_bands = ["AOD", "PM25"]

    planner.plan_static_feature(image_name="FAKE/IMAGE", selected_bands=selected_bands)

    # Check that we selected the correct bands
    mock_gee_for_static_feature["mock_image_instance"].select.assert_called_with(
        selected_bands,
    )


@pytest.mark.parametrize(
    ("selected_bands", "expected_column_mappings"),
    [
        (
            ["NO2"],
            {"grid_id": "grid_id", "mean": "NO2"},
        ),
        (
            ["NO2", "SO2"],
            {"grid_id": "grid_id", "NO2_mean": "NO2", "SO2_mean": "SO2"},
        ),
    ],
    ids=["single_band", "multiple_bands"],
)
def test__GriddedFeatureCollectionPlanner_static_feature__column_mappings__correct_mappings_specified(
    mock_gee_for_static_feature,
    selected_bands,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)

    result = planner.plan_static_feature(
        image_name="FAKE/IMAGE",
        selected_bands=selected_bands,
    )

    assert result.column_mappings == expected_column_mappings


@pytest.fixture
def mock_gee_for_annual_classified_pixels():
    with (
        patch("pm25ml.collectors.gee.feature_planner.ImageCollection") as MockImageCollection,
        patch("pm25ml.collectors.gee.feature_planner.Reducer") as MockReducer,
        patch("pm25ml.collectors.gee.feature_planner.Image") as MockImage,
    ):
        mock_ic_instance = MagicMock()
        MockImageCollection.return_value = mock_ic_instance
        mock_ic_instance.select.return_value = mock_ic_instance
        mock_ic_instance.filterBounds.return_value = mock_ic_instance
        mock_ic_instance.filterDate.return_value = mock_ic_instance

        fake_image = MagicMock()
        mock_ic_instance.map.return_value = fake_image
        fake_image.reduce.return_value = fake_image
        fake_image.reduceRegions.return_value = MagicMock()

        fake_mean = MagicMock()
        MockReducer.mean.return_value = fake_mean

        yield {
            "MockImageCollection": MockImageCollection,
            "MockReducer": MockReducer,
            "MockImage": MockImage,
            "mock_ic_instance": mock_ic_instance,
            "fake_image": fake_image,
            "fake_mean": fake_mean,
        }


def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__columns_specified__correct_columns_suggested(
    mock_gee_for_annual_classified_pixels,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    classification_band = "land_cover"
    output_names_to_class_values = {"forest": [1], "urban": [2]}
    year = 2023

    planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=year,
    )

    # Check that we selected the correct classification band
    mock_gee_for_annual_classified_pixels["mock_ic_instance"].select.assert_called_with(
        classification_band,
    )


def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__gridding__gridding_logic_is_handled_correctly(
    mock_gee_for_annual_classified_pixels,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)
    classification_band = "land_cover"
    output_names_to_class_values = {"forest": [1], "urban": [2]}
    year = 2023

    planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=year,
    )

    # Capture the function passed into the map call
    captured_map_function = mock_gee_for_annual_classified_pixels["mock_ic_instance"].map.call_args[
        0
    ][0]

    # Mock an image with a classification band
    mock_image = MagicMock()
    mock_image.select.return_value.remap.return_value.rename.return_value = mock_image
    mock_image.addBands.return_value = mock_image

    # Call the captured map function directly
    result = captured_map_function(mock_image)

    # Verify that addBands was called with the correct parameters
    mock_image.addBands.assert_called()

    # Verify the result
    assert result == mock_image


@pytest.mark.parametrize(
    ("classification_band", "output_names_to_class_values", "expected_column_mappings"),
    [
        (
            "land_cover",
            {"forest": [1], "urban": [2]},
            {"grid_id": "grid_id", "forest_mean": "forest", "urban_mean": "urban"},
        ),
        (
            "land_use",
            {"agriculture": [3], "water": [4]},
            {
                "grid_id": "grid_id",
                "agriculture_mean": "agriculture",
                "water_mean": "water",
            },
        ),
    ],
    ids=["land_cover", "land_use"],
)
@pytest.mark.usefixtures("mock_gee_for_annual_classified_pixels")
def test__GriddedFeatureCollectionPlanner_annual_classified_pixels__column_mappings__correct_mappings_specified(
    classification_band,
    output_names_to_class_values,
    expected_column_mappings,
) -> None:
    grid = MagicMock()
    planner = GriddedFeatureCollectionPlanner(grid=grid)

    result = planner.plan_summarise_annual_classified_pixels(
        collection_name="FAKE/COLLECTION",
        classification_band=classification_band,
        output_names_to_class_values=output_names_to_class_values,
        year=2023,
    )

    assert result.column_mappings == expected_column_mappings
