"""Handles the collection of raw data from various sources to the archive storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pm25ml.collectors.export_pipeline import (
    ErrorWhileFetchingDataError,
    ExportPipeline,
    MissingDataError,
    PipelineConfig,
)
from pm25ml.logging import logger

if TYPE_CHECKING:
    from collections.abc import Collection

    from pm25ml.collectors.archived_file_validator import ArchivedFileValidator


class DataCompleteness(Enum):
    """Enum representing the completeness status of data."""

    ALREADY_UPLOADED = ("already_uploaded", True, True)
    """
    Data has already been uploaded and is available, it was skipped for this run.
    """
    COMPLETE = ("complete", True, True)
    """
    Data is complete and available.
    """
    EMPTY = ("empty", True, False)
    """
    Data is empty and not available but is allowed.
    """
    ERROR = ("error", False, False)
    """
    An error occurred during the upload process.
    """

    def __init__(self, type_name: str, successful: bool, data_available: bool) -> None:  # noqa: FBT001
        """
        Initialize the DataCompleteness with a name and successful flag.

        :param type_name: The name of the completeness status.
        :param successful: Whether the completeness status was successful.
        """
        self.type_name = type_name
        self.successful = successful
        self.data_available = data_available


@dataclass(frozen=True)
class UploadResult:
    """Represents the result of an upload operation."""

    pipeline_config: PipelineConfig
    completeness: DataCompleteness
    error: Exception | None = None

    @property
    def successful(self) -> bool:
        """
        Check if the upload was successful.

        :return: True if the upload was successful, False otherwise.
        """
        return self.completeness.successful


class RawDataCollector:
    """Collects raw data from various sources to be uploaded to the archive."""

    def __init__(self, metadata_validator: ArchivedFileValidator) -> None:
        """
        Initialize the RawDataCollector with the metadata validator.

        :param metadata_validator: The validator to check if the results need to be uploaded.
        """
        self.metadata_validator = metadata_validator

    def collect(
        self,
        processors: Collection[ExportPipeline],
        *,
        allow_missing_required: bool = False,
    ) -> Collection[UploadResult]:
        """
        Collect data from the given processors and upload them to the archive storage.

        It will attempt to upload only those datasets that have not been uploaded before.

        :param processors: A collection of ExportPipeline instances to collect data from.
        :param allow_missing_required: Return recognized missing-data outcomes for discovery
            instead of failing the collection.
        """
        logger.info("Filtering down to only those datasets that need to be uploaded")
        # Now we only want to download the results if we never have before.
        # We can check with the archive_storage if the results already exist.
        # Evaluate which processors need upload in parallel for speed.
        filtered_processors = self._filter_processors_needing_upload(processors)

        logger.info(f"Downloading {len(filtered_processors)} datasets")
        results = self._run_pipelines_in_parallel(
            filtered_processors,
            allow_missing_required=allow_missing_required,
        )

        # Check all results were uploaded successfully, not just the ones we
        # downloaded this time.
        logger.info("Validating all recent results")
        results_to_validate = [
            result
            for result in results
            if result.completeness.data_available or result.pipeline_config.allows_missing_data
        ]
        self.metadata_validator.validate_all_results(results_to_validate)

        skipped_results = [
            UploadResult(
                processor.get_config_metadata(),
                DataCompleteness.ALREADY_UPLOADED,
            )
            for processor in processors
            if processor not in filtered_processors
        ]

        return results + skipped_results

    def _filter_processors_needing_upload(
        self,
        processors: Collection[ExportPipeline],
    ) -> list[ExportPipeline]:
        with ThreadPoolExecutor() as executor:
            needs_upload_results = list(
                executor.map(
                    lambda processor: self.metadata_validator.needs_upload(
                        expected_result=processor.get_config_metadata(),
                    ),
                    processors,
                ),
            )

        return [
            processor
            for processor, needs_upload in zip(processors, needs_upload_results)
            if needs_upload
        ]

    def _run_pipelines_in_parallel(
        self,
        filtered_processors: Collection[ExportPipeline],
        *,
        allow_missing_required: bool = False,
    ) -> list[UploadResult]:
        if not filtered_processors:
            logger.info("No processors to run, skipping upload.")
            return []

        with ThreadPoolExecutor() as executor:

            def _upload_processor(
                processor: ExportPipeline,
            ) -> UploadResult:
                config = processor.get_config_metadata()
                result_subpath = config.result_subpath

                allows_missing_data = config.allows_missing_data

                try:
                    logger.info(
                        f"Starting upload for processor {result_subpath}",
                    )
                    processor.upload()
                except MissingDataError as e:
                    if allows_missing_data or allow_missing_required:
                        logger.warning(
                            f"Missing data for processor {result_subpath}",
                            exc_info=True,
                            stack_info=True,
                        )
                        return UploadResult(config, DataCompleteness.EMPTY, e)

                    logger.error(
                        f"Missing data for processor {result_subpath} but not allowed",
                        exc_info=True,
                        stack_info=True,
                    )
                    return UploadResult(config, DataCompleteness.ERROR, e)
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Failed to upload processor {result_subpath}",
                        exc_info=True,
                        stack_info=True,
                    )
                    return UploadResult(config, DataCompleteness.ERROR, e)
                else:
                    return UploadResult(
                        config,
                        DataCompleteness.COMPLETE,
                    )

            results = executor.map(_upload_processor, filtered_processors)

            completed_results = list(results)

            failure_results = [
                result for result in completed_results if not result.completeness.successful
            ]

            if failure_results:
                error_count = len(failure_results)
                total_count = len(completed_results)
                logger.error(
                    f"{error_count}/{total_count} processors failed to upload: {failure_results}",
                )
                msg = (
                    f"{error_count}/{total_count} processors failed to upload. "
                    "Check the logs for more details."
                )
                raise ErrorWhileFetchingDataError(
                    msg,
                )

            return completed_results
