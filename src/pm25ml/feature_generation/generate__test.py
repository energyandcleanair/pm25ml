"""Unit tests for FeatureGenerator."""

from __future__ import annotations

from unittest.mock import Mock, call, create_autospec

from arrow import get
import polars as pl

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.feature_generation.generate import FeatureGenerator
from pm25ml.setup.temporal_config import TemporalConfig


def test__feature_generator__multiple_years__scans_once_and_sinks_per_year():
    """It should scan once, generate per year, and sink once per year in order."""
    # Arrange inputs and mocks
    temporal_config = TemporalConfig(
        start_date=get("2022-01-01"), end_date=get("2023-12-31")
    )
    input_artifact = DataArtifactRef(stage="input_stage")
    output_artifact = DataArtifactRef(stage="output_stage")

    combined_storage = create_autospec(CombinedStorage, instance=True)

    # The LazyFrame returned by scan_stage
    lf_scanned = pl.DataFrame({"a": [1]}).lazy()
    combined_storage.scan_stage.return_value = lf_scanned

    # Distinct LazyFrames to be returned for each year by generate_for_year
    lf_2022 = pl.DataFrame({"year": [2022]}).lazy()
    lf_2023 = pl.DataFrame({"year": [2023]}).lazy()

    generate_for_year = Mock(side_effect=[lf_2022, lf_2023])

    generator = FeatureGenerator(
        combined_storage=combined_storage,
        temporal_config=temporal_config,
        input_data_artifact=input_artifact,
        output_data_artifact=output_artifact,
        generate_for_year=generate_for_year,
    )

    # Act
    generator.generate()

    # Assert scan only once with the input stage
    combined_storage.scan_stage.assert_called_once_with(input_artifact.stage)

    # Assert generate_for_year called once per year with (temporal_config, lf, year)
    assert generate_for_year.call_count == 2
    assert generate_for_year.mock_calls == [
        call(temporal_config, lf_scanned, 2022),
        call(temporal_config, lf_scanned, 2023),
    ]

    # Assert sink_stage called with the corresponding frames per year and output stage
    assert combined_storage.sink_stage.call_count == 2
    combined_storage.sink_stage.assert_has_calls(
        [
            call(lf_2022, output_artifact.stage),
            call(lf_2023, output_artifact.stage),
        ],
        any_order=False,
    )
