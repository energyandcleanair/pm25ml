from assertpy import assert_that
import pytest
from unittest.mock import create_autospec
from pm25ml.collectors.collector import DataCompleteness, RawDataCollector, UploadResult
from pm25ml.collectors.export_pipeline import (
    PipelineConsumerBehaviour,
    ExportPipeline,
    MissingDataError,
    MissingDataHeuristic,
    PipelineConfig,
    ValueColumnType,
)
from pm25ml.collectors.archived_file_validator import ArchivedFileValidator


@pytest.fixture
def mock_metadata_validator():
    return create_autospec(ArchivedFileValidator)


@pytest.fixture
def mock_processors():
    return [create_autospec(ExportPipeline) for _ in range(3)]


@pytest.fixture
def collector(mock_metadata_validator):
    return RawDataCollector(metadata_validator=mock_metadata_validator)


def test__collect__validates_all_results__uploads_all_processors(
    collector, mock_metadata_validator, mock_processors
):
    for processor in mock_processors:
        processor.get_config_metadata.return_value = PipelineConfig(
            result_subpath="mock_path",
            id_columns=set(),
            value_column_type_map={},
            expected_rows=0,
        )
        processor.upload.return_value = None

    mock_metadata_validator.needs_upload.return_value = True
    mock_metadata_validator.validate_all_results.return_value = None

    results = collector.collect(mock_processors)

    for processor in mock_processors:
        processor.upload.assert_called_once()

    assert assert_that(results).contains_only(
        *[
            UploadResult(processor.get_config_metadata(), DataCompleteness.COMPLETE, None)
            for processor in mock_processors
        ]
    )


def test__collect__filters_processors_needing_upload__uploads_only_required_processors(
    collector, mock_metadata_validator, mock_processors
):
    for i, processor in enumerate(mock_processors):
        processor.get_config_metadata.return_value = PipelineConfig(
            result_subpath=f"mock_path_{i}",
            id_columns=set(),
            value_column_type_map={},
            expected_rows=0,
        )
        processor.upload.return_value = None

    mock_metadata_validator.needs_upload.side_effect = [True, False, True]

    results = collector.collect(mock_processors)

    for i, processor in enumerate(mock_processors):
        if i == 1:  # This processor should not be uploaded
            processor.upload.assert_not_called()
        else:  # These processors should be uploaded
            processor.upload.assert_called_once()

    assert assert_that(results).contains_only(
        *[
            UploadResult(mock_processors[0].get_config_metadata(), DataCompleteness.COMPLETE, None),
            UploadResult(
                mock_processors[1].get_config_metadata(), DataCompleteness.ALREADY_UPLOADED, None
            ),
            UploadResult(mock_processors[2].get_config_metadata(), DataCompleteness.COMPLETE, None),
        ]
    )


def test__run_pipelines_in_parallel__handles_success_and_failure__raises_exception_on_failure(
    collector, mock_processors
):
    for i, processor in enumerate(mock_processors):
        processor.get_config_metadata.return_value = PipelineConfig(
            result_subpath=f"mock_path_{i}",
            id_columns=set(),
            value_column_type_map={},
            expected_rows=0,
        )

    for processor in mock_processors:
        if processor.get_config_metadata().result_subpath == "mock_path_1":
            processor.upload.side_effect = Exception("Mock failure")
        else:
            processor.upload.return_value = None

    with pytest.raises(Exception):
        collector._run_pipelines_in_parallel(mock_processors)

    for processor in mock_processors:
        processor.upload.assert_called_once()


def test__run_pipelines_in_parallel__allows_missing_error__runs_pipeline_successfully(collector):
    process_1 = create_autospec(ExportPipeline)
    process_1.get_config_metadata.return_value = PipelineConfig(
        result_subpath="mock_path_1",
        id_columns=set(),
        value_column_type_map={"value": ValueColumnType.FLOAT},
        expected_rows=0,
        consumer_behaviour=PipelineConsumerBehaviour(
            missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE,
        ),
    )

    process_2 = create_autospec(ExportPipeline)
    process_2.get_config_metadata.return_value = PipelineConfig(
        result_subpath="mock_path_2",
        id_columns=set(),
        value_column_type_map={"value": ValueColumnType.FLOAT},
        expected_rows=0,
    )

    mock_processors = [process_1, process_2]

    expected_error = MissingDataError("Mock missing data error")

    for processor in mock_processors:
        if processor.get_config_metadata().result_subpath == "mock_path_1":
            processor.upload.side_effect = expected_error
        else:
            processor.upload.return_value = None

    results = collector._run_pipelines_in_parallel(mock_processors)

    assert assert_that(results).contains_only(
        *[
            UploadResult(process_1.get_config_metadata(), DataCompleteness.EMPTY, expected_error),
            UploadResult(process_2.get_config_metadata(), DataCompleteness.COMPLETE, None),
        ]
    )
