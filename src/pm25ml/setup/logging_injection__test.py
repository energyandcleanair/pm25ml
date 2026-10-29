"""Tests for the standalone production logging container."""

from pm25ml.setup.logging_injection import init_logging_from_env
from pm25ml.setup.slack import ProductionSlack


def test__init_logging_from_env__builds_slack_without_pipeline_settings() -> None:
    logging = init_logging_from_env()

    slack = logging.production_slack()

    assert isinstance(slack, ProductionSlack)
    assert logging.production_slack() is slack
