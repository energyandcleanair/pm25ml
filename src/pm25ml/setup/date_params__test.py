"""Tests for temporal configuration."""

import arrow
import pytest
from attr.exceptions import FrozenInstanceError

from pm25ml.setup.date_params import TemporalConfig


def test__temporal_config__is_immutable() -> None:
    config = TemporalConfig(
        start_date=arrow.get("2020-01-01"),
        end_date=arrow.get("2026-05-01"),
    )

    with pytest.raises(FrozenInstanceError):
        config.end_date = arrow.get("2026-06-01")


def test__temporal_config__start_after_end__raises() -> None:
    with pytest.raises(ValueError, match="START_MONTH must not be later"):
        TemporalConfig(
            start_date=arrow.get("2026-06-01"),
            end_date=arrow.get("2026-05-01"),
        )
