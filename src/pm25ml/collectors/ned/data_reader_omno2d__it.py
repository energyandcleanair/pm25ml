"""
Integration test for Omno2dReader.
"""

import pytest
from pathlib import Path
from pm25ml.collectors.ned.data_reader_omno2d import Omno2dReader
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.coord_types import Lon, Lat
import arrow

pytestmark = pytest.mark.integration


@pytest.fixture
def example_file_path() -> Path:
    """
    Fixture to provide the path to the example OMI NO2 file.
    """
    return (
        Path(__file__).parent
        / "data_reader_omno2d__it_assets"
        / "OMI-Aura_L3-OMNO2d_2023m0111_v004-2025m1103t113244.he5"
    )


@pytest.fixture
def dataset_descriptor() -> NedDatasetDescriptor:
    """
    Fixture to provide a dataset descriptor for the test.
    """
    return NedDatasetDescriptor(
        dataset_name="OMNO2d",
        dataset_version="004",
        start_date=arrow.get("2023-01-01"),
        end_date=arrow.get("2023-01-01"),
        filter_bounds=(Lon(70.0), Lat(10.0), Lon(90.0), Lat(30.0)),
        variable_mapping={
            "ColumnAmountNO2": "NO2",
        },
    )


@pytest.fixture
def dataset_descriptor_trop() -> NedDatasetDescriptor:
    """
    Fixture to provide a dataset descriptor for the test.
    """
    return NedDatasetDescriptor(
        dataset_name="OMNO2d",
        dataset_version="004",
        start_date=arrow.get("2023-01-01"),
        end_date=arrow.get("2023-01-01"),
        filter_bounds=(Lon(70.0), Lat(10.0), Lon(90.0), Lat(30.0)),
        variable_mapping={
            "ColumnAmountNO2Trop": "NO2",
        },
    )


def test__omno2dreader__read_example_file__extracts_data(example_file_path, dataset_descriptor):
    """
    Test that Omno2dReader can read an example file and extract data.
    """
    reader = Omno2dReader()

    with example_file_path.open("rb") as file:
        ned_day_data = reader.extract_data(file, dataset_descriptor)

        assert ned_day_data.data is not None
        assert ned_day_data.data.sizes == {"lat": 80, "lon": 80}
        assert ned_day_data.data.coords["lon"].min() == 70.125
        assert ned_day_data.data.coords["lon"].max() == 89.875
        assert ned_day_data.data.coords["lat"].min() == 10.125
        assert ned_day_data.data.coords["lat"].max() == 29.875
        assert ned_day_data.date == "2023-01-11"

        # Has the expected variable
        assert "ColumnAmountNO2" in ned_day_data.data.data_vars
        assert "ColumnAmountNO2Trop" not in ned_day_data.data.data_vars


def test__omno2dreader__read_example_file__extracts_data_trop(
    example_file_path, dataset_descriptor_trop
):
    """
    Test that Omno2dReader can read an example file and extract data.
    """
    reader = Omno2dReader()

    with example_file_path.open("rb") as file:
        ned_day_data = reader.extract_data(file, dataset_descriptor_trop)

        assert ned_day_data.data is not None
        assert ned_day_data.data.sizes == {"lat": 80, "lon": 80}
        assert ned_day_data.data.coords["lon"].min() == 70.125
        assert ned_day_data.data.coords["lon"].max() == 89.875
        assert ned_day_data.data.coords["lat"].min() == 10.125
        assert ned_day_data.data.coords["lat"].max() == 29.875
        assert ned_day_data.date == "2023-01-11"

        # Has the expected variable, and not others
        assert "ColumnAmountNO2Trop" in ned_day_data.data.data_vars
        assert "ColumnAmountNO2" not in ned_day_data.data.data_vars
