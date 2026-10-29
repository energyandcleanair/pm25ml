"""Retrieves raw Earth Access data."""

from collections.abc import Iterable
from typing import IO, cast

import earthaccess

from pm25ml.collectors.ned.data_retrievers import NedDataRetriever
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.errors import NedMissingDataError
from pm25ml.collectors.ned.granule_search import search_earthaccess_granules
from pm25ml.logging import logger


class RawEarthAccessDataRetriever(NedDataRetriever):
    """
    Retrieves raw Earth Access data.

    This class streams files from the raw Earth Access source based on the dataset descriptor.
    It searches for granules matching the dataset name and date range, and yields the files
    containing the data for the dataset.

    It does not perform any subsetting or filtering of the data before yielding the files.
    """

    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[IO[bytes]]:
        """
        Stream data from the raw Earth Access source.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
                and processing instructions.

        Returns:
            Iterable[IO[bytes]]: An iterable of files containing the data for the
            dataset.

        """
        logger.info("Searching for granules for dataset %s", dataset_descriptor)
        granules: list[earthaccess.DataGranule] = search_earthaccess_granules(dataset_descriptor)

        self._check_expected_granules(
            dataset_descriptor=dataset_descriptor,
            granules=granules,
        )

        for granule in granules:
            [file] = earthaccess.open([granule])
            # earthaccess misreports the return type of the file as an AbstractFileSystem
            # but it is actually a file-like object.
            yield cast("IO[bytes]", file)

    def _check_expected_granules(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
        granules: list[earthaccess.DataGranule],
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
                f"We require {expected_days - 1} (for {expected_days} days) granules for dataset "
                f"{dataset_descriptor}, but found {len(granules)}."
            )
            raise NedMissingDataError(
                msg,
            )

        logger.info(
            "Found %d granules for dataset %s",
            len(granules),
            dataset_descriptor,
        )
