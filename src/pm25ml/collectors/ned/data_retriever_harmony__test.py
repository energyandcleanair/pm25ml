import collections
import pytest
import responses
from responses import matchers, RequestsMock
from unittest.mock import call, patch, MagicMock
from pm25ml.collectors.ned.data_retriever_harmony import HarmonySubsetterDataRetriever
from pm25ml.collectors.ned.coord_types import Lon, Lat
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
import arrow

from pm25ml.collectors.ned.errors import NedMissingDataError

EXPECTED_ACCESS_TOKEN = "an-arbitrary-token"

EXPECTED_AUTH_HEADER_MATCHER = matchers.header_matcher(
    {"Authorization": f"Bearer {EXPECTED_ACCESS_TOKEN}"}
)


@pytest.fixture
def mock_dataset_descriptor():
    """Mock dataset descriptor."""
    return NedDatasetDescriptor(
        dataset_name="mock_dataset",
        dataset_version="1.0",
        start_date=arrow.get("2025-06-01"),
        end_date=arrow.get("2025-06-15"),
        filter_bounds=(Lon(-10.0), Lat(-10.0), Lon(10.0), Lat(10.0)),
        variable_mapping={
            "mock_var": "mock_target",
        },
    )


@pytest.fixture()
def mock_earth_access__data_available():
    with (
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_datasets"
        ) as mock_search_datasets,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_data"
        ) as mock_search_data,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.get_edl_token"
        ) as mock_get_edl_token,
    ):
        # Mock dataset search
        mock_search_datasets.return_value = [MagicMock(concept_id=lambda: "mock_collection_id")]

        # Mock granule search
        mock_search_data.return_value = [
            MagicMock() for _ in range(15)
        ]  # Adjust to match days_in_range

        mock_get_edl_token.return_value = {"access_token": EXPECTED_ACCESS_TOKEN}
        yield


@pytest.fixture
def mock_earth_access__no_dataset():
    with (
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_datasets"
        ) as mock_search_datasets,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_data"
        ) as mock_search_data,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.get_edl_token"
        ) as mock_get_edl_token,
    ):
        # Mock dataset search to return no datasets
        mock_search_datasets.return_value = []

        # Mock granule search to return no data
        mock_search_data.return_value = []

        mock_get_edl_token.return_value = {"access_token": EXPECTED_ACCESS_TOKEN}
        yield


@pytest.fixture
def mock_earth_access__too_many_datasets():
    with (
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_datasets"
        ) as mock_search_datasets,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_data"
        ) as mock_search_data,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.get_edl_token"
        ) as mock_get_edl_token,
    ):
        mock_search_datasets.return_value = [
            MagicMock(concept_id=lambda: "mock_collection_id1"),
            MagicMock(concept_id=lambda: "mock_collection_id2"),
        ]

        mock_search_data.return_value = []

        mock_get_edl_token.return_value = {"access_token": EXPECTED_ACCESS_TOKEN}
        yield


@pytest.fixture()
def mock_earth_access__dataset_available_wrong_number_of_granules():
    with (
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_datasets"
        ) as mock_search_datasets,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_data"
        ) as mock_search_data,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.get_edl_token"
        ) as mock_get_edl_token,
    ):
        # Mock dataset search
        mock_search_datasets.return_value = [MagicMock(concept_id=lambda: "mock_collection_id")]

        # Mock granule search
        mock_search_data.return_value = [
            MagicMock() for _ in range(5)
        ]  # Adjust to match days_in_range

        mock_get_edl_token.return_value = {"access_token": EXPECTED_ACCESS_TOKEN}
        yield


@pytest.fixture()
def mock_earth_access__dataset_available_no_granules():
    with (
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_datasets"
        ) as mock_search_datasets,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.search_data"
        ) as mock_search_data,
        patch(
            "pm25ml.collectors.ned.data_retriever_harmony.earthaccess.get_edl_token"
        ) as mock_get_edl_token,
    ):
        # Mock dataset search
        mock_search_datasets.return_value = [MagicMock(concept_id=lambda: "mock_collection_id")]

        # Mock granule search
        mock_search_data.return_value = []

        mock_get_edl_token.return_value = {"access_token": EXPECTED_ACCESS_TOKEN}
        yield


@pytest.fixture
def mock_files():
    return [MagicMock(), MagicMock()]


@pytest.fixture
def mock_https_filesystem__opens_files(mock_files):
    mock_https_filesystem = MagicMock()
    mock_https_filesystem.open.side_effect = mock_files
    return mock_https_filesystem


@pytest.fixture
def mock_ffspec__create_filesystem_which_opens_files(mock_https_filesystem__opens_files):
    with patch(
        "pm25ml.collectors.ned.data_retriever_harmony.fsspec.filesystem"
    ) as mocked_filesystem:
        mocked_filesystem.return_value = mock_https_filesystem__opens_files
        yield mocked_filesystem


@pytest.fixture
def mock_response__job_submit_success():
    responses.add(
        responses.GET,
        (
            "https://harmony.earthdata.nasa.gov/"
            "mock_collection_id/ogc-api-coverages/1.0.0/"
            "collections/parameter_vars/coverage/rangeset"
        ),
        json={"jobID": "mock_job_id"},
        match=[
            matchers.query_param_matcher(
                {
                    "format": "application/x-netcdf4",
                    "variable": "mock_var",
                    "subset": [
                        "lon(-10.0:10.0)",
                        "lat(-10.0:10.0)",
                        'time("2025-06-01T00:00:00+0000":"2025-06-15T23:59:59+0000")',
                    ],
                    "maxResults": 31,
                }
            ),
            EXPECTED_AUTH_HEADER_MATCHER,
        ],
        status=200,
    )


@pytest.fixture
def mock_response__job_complete():
    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={
            "status": "successful",
            "progress": 100,
            "links": [
                {"href": "https://example.com/mock_file_1.nc"},
                {"href": "https://example.com/mock_file_2.nc"},
            ],
        },
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )


@pytest.fixture
def mock_response__job_fails():
    """Mock a failed job response."""
    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={"status": "failed", "progress": 0, "error": "Job failed due to an error."},
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )


@pytest.fixture
def mock_response__job_3x_running_then_success():
    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={
            "status": "running",
            "progress": 0,
        },
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )
    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={
            "status": "running",
            "progress": 50,
        },
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )
    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={
            "status": "running",
            "progress": 80,
        },
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )

    responses.add(
        responses.GET,
        "https://harmony.earthdata.nasa.gov/jobs/mock_job_id",
        json={
            "status": "successful",
            "progress": 100,
            "links": [
                {"href": "https://example.com/mock_file_1.nc"},
                {"href": "https://example.com/mock_file_2.nc"},
            ],
        },
        match=[EXPECTED_AUTH_HEADER_MATCHER],
        status=200,
    )


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__data_available",
    "mock_ffspec__create_filesystem_which_opens_files",
    "mock_response__job_submit_success",
    "mock_response__job_complete",
)
def test__HarmonySubsetterDataRetriever_stream_files__found_dataset_and_processing_succeeded__returns_files(
    mock_dataset_descriptor, mock_files, mock_https_filesystem__opens_files
):
    retriever = HarmonySubsetterDataRetriever()
    files = list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))

    mock_https_filesystem__opens_files.open.asset_has_calls(
        [call("https://example.com/mock_file_1.nc"), call("https://example.com/mock_file_2.nc")]
    )
    assert len(files) == 2
    assert files[0] == mock_files[0]
    assert files[1] == mock_files[1]


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__data_available",
    "mock_ffspec__create_filesystem_which_opens_files",
    "mock_response__job_submit_success",
    "mock_response__job_3x_running_then_success",
)
@patch("time.sleep", return_value=None)
def test__HarmonySubsetterDataRetriever_stream_files__takes_3_requests_before_complete__returns_files(
    _sleep, mock_dataset_descriptor, mock_https_filesystem__opens_files
):
    retriever = HarmonySubsetterDataRetriever()
    files = list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))

    mock_https_filesystem__opens_files.open.asset_has_calls(
        [call("https://example.com/mock_file_1.nc"), call("https://example.com/mock_file_2.nc")]
    )
    assert len(files) == 2


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__data_available",
    "mock_https_filesystem__opens_files",
    "mock_response__job_submit_success",
    "mock_response__job_complete",
)
def test__HarmonySubsetterDataRetriever_stream_files__happy_path__inits_filesystem_correctly(
    mock_dataset_descriptor, mock_ffspec__create_filesystem_which_opens_files
):
    retriever = HarmonySubsetterDataRetriever()
    # We need to consume the generator to trigger the filesystem creation.
    collections.deque(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))

    mock_ffspec__create_filesystem_which_opens_files.assert_called_once_with(
        "https",
        client_kwargs={
            "headers": {"Authorization": f"Bearer {EXPECTED_ACCESS_TOKEN}"},
            "trust_env": False,
        },
    )


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__data_available",
    "mock_ffspec__create_filesystem_which_opens_files",
    "mock_response__job_submit_success",
    "mock_response__job_fails",
)
def test__HarmonySubsetterDataRetriever_stream_files__job_failed__raises_exception(
    mock_dataset_descriptor, mock_files, mock_https_filesystem__opens_files
):
    with pytest.raises(
        NedMissingDataError,
        match="Job mock_job_id failed with status: failed. Please check the Harmony Subsetter API for more details.",
    ):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__no_dataset", "mock_ffspec__create_filesystem_which_opens_files"
)
def test__HarmonySubsetterDataRetriever_stream_files__no_dataset_found__raises_exception(
    mock_dataset_descriptor,
):
    with pytest.raises(NedMissingDataError, match="No datasets found for mock_dataset."):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__too_many_datasets", "mock_ffspec__create_filesystem_which_opens_files"
)
def test__HarmonySubsetterDataRetriever_stream_files__too_many_datasets_found__raises_exception(
    mock_dataset_descriptor,
):
    with pytest.raises(
        NedMissingDataError,
        match="Multiple datasets found for mock_dataset. Please specify a more precise dataset name.",
    ):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__dataset_available_wrong_number_of_granules",
    "mock_ffspec__create_filesystem_which_opens_files",
)
def test__HarmonySubsetterDataRetriever_stream_files__wrong_n_granules__raises_exception(
    mock_dataset_descriptor,
):
    with pytest.raises(
        NedMissingDataError,
        match=r"We require 14 or 15 \(for 15 days\) granules for dataset .* but found 5.",
    ):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__dataset_available_no_granules",
    "mock_ffspec__create_filesystem_which_opens_files",
)
def test__HarmonySubsetterDataRetriever_stream_files__no_granules__raises_exception(
    mock_dataset_descriptor,
):
    with pytest.raises(NedMissingDataError, match="No granules found for dataset .*."):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))


@responses.activate
@pytest.mark.usefixtures(
    "mock_earth_access__data_available",
    "mock_ffspec__create_filesystem_which_opens_files",
    "mock_response__job_submit_success",
    "mock_response__job_fails",
)
def test__HarmonySubsetterDataRetriever_stream_files__multiple_variables__raises_exception(
    mock_dataset_descriptor,
):
    # Modify the mock dataset descriptor to include multiple variables
    mock_dataset_descriptor.variable_mapping = {
        "var1": "target1",
        "var2": "target2",
    }

    with pytest.raises(
        ValueError,
        match="Harmony Subsetter API only supports one variable for retrieval. Provided variables:.*",
    ):
        retriever = HarmonySubsetterDataRetriever()
        list(retriever.stream_files(dataset_descriptor=mock_dataset_descriptor))
