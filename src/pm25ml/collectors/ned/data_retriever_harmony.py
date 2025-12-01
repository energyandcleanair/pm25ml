"""
Data retriever for Harmony Subsetter API.

References:
    https://harmony.earthdata.nasa.gov/

"""

import time
from collections.abc import Iterable
from typing import IO, TypedDict, Union, cast
from urllib import parse

import earthaccess
import fsspec
import requests
from arrow import Arrow
from requests.auth import AuthBase

from pm25ml.collectors.ned.data_retriever_raw import EARTH_ENGINE_SEARCH_DATE_FORMAT
from pm25ml.collectors.ned.data_retrievers import (
    NedDataRetriever,
)
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.errors import NedMissingDataError
from pm25ml.logging import logger

# We define helper types to make the code more readable and maintainable.
_JSONObject = dict[str, "_JSONType"]

_JSONType = Union[
    str,
    int,
    float,
    bool,
    None,
    _JSONObject,
    list["_JSONType"],
]

_JobResultLink = dict[str, str]
_JobResultLinks = list[_JobResultLink]


class _JobInitResponse(TypedDict):
    """Represents the response from the Harmony Subsetter API when initializing a job."""

    jobID: str


class _StatusResponse(TypedDict):
    """Represents the status response from the Harmony Subsetter API."""

    status: str
    progress: int
    links: _JobResultLinks


HARMONY_DATE_FILTER_FORMAT = "YYYY-MM-DDTHH:mm:ssZ"


class HarmonySubsetterDataRetriever(NedDataRetriever):
    """
    Retrieves data using the Harmony Subsetter API.

    This subsets data from NASA Earthdata collections, where possible, based on the
    spatial and temporal filters defined in the dataset descriptor.
    """

    ogc_api_coverages_version = "1.0.0"
    harmony_root = "https://harmony.earthdata.nasa.gov"

    job_complete_percentage = 100

    def __init__(
        self,
    ) -> None:
        """Initialize the data retriever with the source."""

    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[IO[bytes]]:
        """
        Stream data using the Harmony Subsetter API.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The descriptor containing dataset details and
            processing instructions.

        Returns:
            Iterable[FileLike]: An iterable of file-like objects containing the subsetted
            data.

        """
        logger.info(
            "Starting data retrieval for dataset %s",
            dataset_descriptor,
        )

        logger.debug("Searching for datasets for %s", dataset_descriptor)
        datasets = earthaccess.search_datasets(
            short_name=dataset_descriptor.dataset_name,
        )
        self._check_expected_dataset(datasets, dataset_descriptor)
        collection_id = datasets[0].concept_id()

        logger.debug("Found dataset with collection ID %s", collection_id)

        logger.debug("Searching for granules for dataset %s", dataset_descriptor)
        granules: list[earthaccess.DataGranule] = earthaccess.search_data(
            short_name=dataset_descriptor.dataset_name,
            temporal=(
                dataset_descriptor.start_date.format(EARTH_ENGINE_SEARCH_DATE_FORMAT),
                dataset_descriptor.end_date.format(EARTH_ENGINE_SEARCH_DATE_FORMAT),
            ),
            count=-1,
            version=dataset_descriptor.dataset_version,
        )

        self._check_expected_granules(granules, dataset_descriptor)

        logger.debug("Starting subsetting job for collection ID %s", collection_id)
        job_response = self._init_subsetting_job(collection_id, dataset_descriptor)
        job_id = job_response.get("jobID")
        if not job_id:
            msg = f"Unable to start job: {job_response}"
            raise NedMissingDataError(msg)

        logger.debug("Subsetting job started with ID %s", job_id)
        links_details = self._await_download_url_results(job_id)

        logger.debug(
            "Job %s completed successfully with %d links",
            job_id,
            len(links_details),
        )

        https_file_system = fsspec.filesystem(
            "https",
            client_kwargs={
                "headers": {
                    "Authorization": f"Bearer {self._get_earthdata_token()}",
                },
                "trust_env": False,
            },
        )

        for link_details in links_details:
            href = link_details.get("href")
            if not href:
                msg = f"Link details missing 'href': {link_details}"
                raise NedMissingDataError(msg)
            yield https_file_system.open(href)

    def _await_download_url_results(self, job_id: str) -> _JobResultLinks:
        job_status_response = self._fetch_job_status(job_id)

        while self._is_job_running(job_status_response):
            logger.debug(
                "Job %s is still running: %s%% complete",
                job_id,
                job_status_response["progress"],
            )
            time.sleep(10)
            job_status_response = self._fetch_job_status(job_id)

        if self._has_job_succeeded(job_status_response):
            return [
                link for link in job_status_response["links"] if link.get("rel", "data") == "data"
            ]

        msg = (
            f"Job {job_id} failed with status: {job_status_response['status']}. "
            "Please check the Harmony Subsetter API for more details."
        )
        raise NedMissingDataError(msg)

    def _init_subsetting_job(
        self,
        collection_id: str,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> _JobInitResponse:
        job_init_url = self._build_subsetting_url(collection_id, dataset_descriptor)

        return cast(
            "_JobInitResponse",
            self._make_json_request(job_init_url),
        )

    def _fetch_job_status(
        self,
        job_id: str,
    ) -> _StatusResponse:
        job_url = self.harmony_root + f"/jobs/{job_id}"

        return cast(
            "_StatusResponse",
            self._make_json_request(job_url),
        )

    def _check_expected_dataset(
        self,
        datasets: list[earthaccess.DataCollection],
        dataset_descriptor: NedDatasetDescriptor,
    ) -> None:
        if not datasets:
            msg = f"No datasets found for {dataset_descriptor.dataset_name}."
            raise NedMissingDataError(msg)
        if len(datasets) > 1:
            msg = (
                f"Multiple datasets found for {dataset_descriptor.dataset_name}. "
                "Please specify a more precise dataset name."
            )
            raise NedMissingDataError(msg)

    def _build_subsetting_url(
        self,
        collection_id: str,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> str:
        west, south, east, north = dataset_descriptor.filter_bounds

        api_path = (
            f"/{collection_id}/ogc-api-coverages/{HarmonySubsetterDataRetriever.ogc_api_coverages_version}"
            f"/collections/parameter_vars/coverage/rangeset"
        )

        start_time = dataset_descriptor.start_date.format(HARMONY_DATE_FILTER_FORMAT)

        def to_end_of_day(date: Arrow) -> Arrow:
            return date.replace(hour=23, minute=59, second=59)

        end_time = to_end_of_day(dataset_descriptor.end_date).format(
            HARMONY_DATE_FILTER_FORMAT,
        )

        if len(dataset_descriptor.variable_mapping) != 1:
            msg = (
                "Harmony Subsetter API only supports one variable for retrieval. "
                f"Provided variables: {dataset_descriptor.variable_mapping.keys()}"
            )
            raise ValueError(msg)

        variable_name = next(iter(dataset_descriptor.variable_mapping.keys()))

        # Build query string
        query_params: list[tuple[str, str | int]] = [
            ("format", "application/x-netcdf4"),
            ("variable", variable_name),
            ("subset", f"lon({west}:{east})"),
            ("subset", f"lat({south}:{north})"),
            ("subset", f'time("{start_time}":"{end_time}")'),
            ("maxResults", 31),
        ]
        query_string = parse.urlencode(query_params)

        return HarmonySubsetterDataRetriever.harmony_root + api_path + "?" + query_string

    def _make_json_request(
        self,
        url: str,
    ) -> _JSONObject:
        """Make a JSON request to the Harmony Subsetter API."""
        auth = _BearerToken(self._get_earthdata_token())
        response = requests.get(url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.json()

    def _check_expected_granules(
        self,
        granules: list[earthaccess.DataGranule],
        dataset_descriptor: NedDatasetDescriptor,
    ) -> None:
        if len(granules) == 0:
            msg = f"No granules found for dataset {dataset_descriptor}."
            raise NedMissingDataError(msg)

        expected_days = dataset_descriptor.days_in_range
        if len(granules) != expected_days:
            logger.warning(
                "Expected %d granules for dataset %s, but found %d.",
                expected_days,
                dataset_descriptor,
                len(granules),
            )

        if len(granules) > expected_days:
            msg = (
                f"Found {len(granules)} granules for dataset {dataset_descriptor}, "
                f"but expected only {expected_days}. This may indicate an issue with the dataset."
            )
            raise NedMissingDataError(msg)

        if len(granules) < expected_days - 1:
            msg = (
                f"We require {expected_days - 1} or {expected_days} (for {expected_days} days) "
                f"granules for dataset {dataset_descriptor}, but found {len(granules)}."
            )
            raise NedMissingDataError(
                msg,
            )

        logger.debug(
            "Found %d granules for dataset %s",
            len(granules),
            dataset_descriptor,
        )

    @staticmethod
    def _get_earthdata_token() -> str:
        """Get the Earthdata token for authentication."""
        token = cast("dict[str, str]", earthaccess.get_edl_token())
        return token["access_token"]

    @staticmethod
    def _is_job_running(
        job_status: _StatusResponse,
    ) -> bool:
        return (
            job_status["status"] == "running"
            and job_status["progress"] < HarmonySubsetterDataRetriever.job_complete_percentage
        )

    @staticmethod
    def _has_job_succeeded(
        job_status: _StatusResponse,
    ) -> bool:
        return (
            job_status["status"] == "successful"
            and job_status["progress"] == HarmonySubsetterDataRetriever.job_complete_percentage
        )


class _BearerToken(AuthBase):
    def __init__(self, token: str) -> None:
        self.token = token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r
