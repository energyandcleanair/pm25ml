import pytest
from pm25ml.collectors.validate_configuration import validate_configuration


class MockExportPipeline:
    def __init__(self, result_subpath, expected_rows, id_columns):
        self.result_subpath = result_subpath
        self.expected_rows = expected_rows
        self.id_columns = id_columns

    def get_config_metadata(self):
        return self


@pytest.fixture
def valid_processor_months():
    return MockExportPipeline(
        result_subpath="country=india/dataset=test/month=2023-01",
        expected_rows=33074 * 31,  # 31 days in January
        id_columns={"date", "grid_id"},
    )


@pytest.fixture
def valid_processor_years():
    return MockExportPipeline(
        result_subpath="country=india/dataset=test/year=2023",
        expected_rows=33074,
        id_columns={"grid_id"},
    )


@pytest.fixture
def valid_processor_months_no_time_qualifier():
    return MockExportPipeline(
        result_subpath="country=india/dataset=test",
        expected_rows=33074,
        id_columns={"grid_id"},
    )


@pytest.fixture
def invalid_country_processor():
    return MockExportPipeline(
        result_subpath="country=invalid/dataset=test/month=2023-01",
        expected_rows=33074 * 31,
        id_columns={"date", "grid_id"},
    )


@pytest.fixture
def missing_dataset_key_processor():
    return MockExportPipeline(
        result_subpath="month=2023-01",
        expected_rows=33074 * 31,
        id_columns={"date", "grid_id"},
    )


@pytest.fixture
def missing_country_key_processor():
    return MockExportPipeline(
        result_subpath="dataset=test/month=2023-01",
        expected_rows=33074 * 31,
        id_columns={"date", "grid_id"},
    )


def test__validate_configuration__valid_processor_months__no_error(valid_processor_months):
    validate_configuration([valid_processor_months], 33074)


def test__validate_configuration__valid_processor_years__no_error(valid_processor_years):
    validate_configuration([valid_processor_years], 33074)


def test__validate_configuration__valid_processor_months_no_time_qualifier__no_error(
    valid_processor_months_no_time_qualifier,
):
    validate_configuration([valid_processor_months_no_time_qualifier], 33074)


def test__validate_configuration__invalid_country_processor__no_error_if_rows_match(
    invalid_country_processor,
):
    validate_configuration([invalid_country_processor], 33074)


def test__validate_configuration__missing_dataset_key_processor__raises_value_error(
    missing_dataset_key_processor,
):
    with pytest.raises(
        ValueError, match="Expected 'dataset' key in month=2023-01, but it is missing."
    ):
        validate_configuration([missing_dataset_key_processor], 33074)


def test__validate_configuration__missing_country_key_processor__raises_value_error(
    missing_country_key_processor,
):
    with pytest.raises(
        ValueError, match="Expected 'country' key in dataset=test/month=2023-01, but it is missing."
    ):
        validate_configuration([missing_country_key_processor], 33074)
