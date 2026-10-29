"""Data retriever for NASA Earthdata sources."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import IO

from pm25ml.collectors.coverage import CoverageProbeResult, dates_from_arrows
from pm25ml.collectors.ned.dataset_descriptor import NedDatasetDescriptor
from pm25ml.collectors.ned.granule_search import (
    covered_dates_from_granules,
    search_earthaccess_granules,
)


class NedDataRetriever(ABC):
    """
    Retrieves data from a NASA source.

    Streams files from the source based on the dataset descriptor. Optionally, this may allow for
    server-side subsetting operations based on the dataset descriptor.
    """

    @abstractmethod
    def stream_files(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> Iterable[IO[bytes]]:
        """
        Stream files from the source.

        Optionally, this may allow for server-side subsetting operations based on the
        dataset descriptor.

        Args:
            dataset_descriptor (NedDatasetDescriptor): The dataset descriptor containing metadata
                and processing instructions.

        Returns:
            Iterable[IO[bytes]]: An iterable of files containing the data for the
            dataset.

        """

    def probe_coverage(
        self,
        *,
        dataset_descriptor: NedDatasetDescriptor,
    ) -> CoverageProbeResult:
        """Search for granules without downloading files."""
        requested_dates = dates_from_arrows(
            dataset_descriptor.start_date,
            dataset_descriptor.end_date,
        )
        try:
            granules = search_earthaccess_granules(dataset_descriptor)
            covered_dates = covered_dates_from_granules(granules, requested_dates)
        except Exception as exc:  # noqa: BLE001
            return CoverageProbeResult.unknown(requested_dates, str(exc))
        if covered_dates is None:
            return CoverageProbeResult.unknown(
                requested_dates,
                "granule temporal extent was not parseable",
            )
        return CoverageProbeResult.known(requested_dates, covered_dates)
