"""Minimal production logging graph, usable before the full pipeline container."""

from __future__ import annotations

from dependency_injector import containers, providers

from pm25ml.setup.slack import ProductionSlack


class LoggingContainer(containers.DeclarativeContainer):
    """Compose Slack logging without pipeline settings or cloud clients."""

    production_slack = providers.Singleton(ProductionSlack.from_env)


def init_logging_from_env() -> LoggingContainer:
    """Build the logging graph from the process environment."""
    return LoggingContainer()
