"""Post production pipeline events to Slack."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import requests

from pm25ml.logging import logger
from pm25ml.setup.model_run_ref import BUNDLE_SOURCE_RELEASE, bundle_source

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

SLACK_CHANNEL: Final = "#log-pm25ml"
CHAT_POST_MESSAGE_URL: Final = "https://slack.com/api/chat.postMessage"
STEP_COLOR: Final = "#1447E0"
INFO_COLOR: Final = "good"
ERROR_COLOR: Final = "#E91E63"


def slack_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether production Slack posts should be sent."""
    values = os.environ if environ is None else environ
    return bundle_source(values) == BUNDLE_SOURCE_RELEASE and bool(
        values.get("SLACK_API_TOKEN", "").strip(),
    )


class ProductionSlack:
    """Fail-open Slack client for production pipeline events."""

    def __init__(
        self,
        *,
        token: str,
        enabled: bool,
        environ: Mapping[str, str],
    ) -> None:
        """Store token and enablement without logging the token."""
        self._token = token
        self._enabled = enabled
        self._environ = environ

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProductionSlack:
        """Build from process environment without putting the token in settings."""
        values = os.environ if environ is None else environ
        token = values.get("SLACK_API_TOKEN", "").strip()
        return cls(
            token=token,
            enabled=slack_enabled(values),
            environ=values,
        )

    def _context_lines(self) -> str:
        lines: list[str] = []
        model_run_ref = self._environ.get("MODEL_RUN_REF", "").strip()
        profile = self._environ.get("PIPELINE_PROFILE", "").strip()
        if model_run_ref:
            lines.append(f"MODEL_RUN_REF={model_run_ref}")
        if profile:
            lines.append(f"PIPELINE_PROFILE={profile}")
        return "\n".join(lines)

    def _message_text(self, headline: str) -> str:
        extra = self._context_lines()
        if extra:
            return f"{headline}\n{extra}"
        return headline

    def _post(
        self,
        headline: str,
        *,
        title: str,
        color: str,
    ) -> None:
        if not self._enabled:
            return
        payload = {
            "channel": SLACK_CHANNEL,
            "text": headline,
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": self._message_text(headline),
                    "mrkdwn_in": ["text"],
                },
            ],
        }
        try:
            response = requests.post(
                CHAT_POST_MESSAGE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                logger.warning(
                    "Slack rejected production log: %s",
                    body.get("error", "unknown_error"),
                )
        except requests.RequestException:
            logger.warning("Failed to post production log to Slack", exc_info=True)

    def post_step(self, message: str) -> None:
        """Post a stage-start production message. Failures are logged only."""
        self._post(message, title="STEP", color=STEP_COLOR)

    def post_info(self, message: str) -> None:
        """Post an informational production message. Failures are logged only."""
        self._post(message, title="INFO", color=INFO_COLOR)

    def post_error(self, message: str) -> None:
        """Post an error production message. Failures are logged only."""
        self._post(message, title="ERROR", color=ERROR_COLOR)

    @contextmanager
    def report_stage(self, stage_name: str) -> Iterator[None]:
        """Log a production stage start, duration, and failure to Slack when enabled."""
        self.post_step(f"## Running step: {stage_name} ##")
        started = time.monotonic()
        try:
            yield
        except Exception as exc:
            minutes = (time.monotonic() - started) / 60
            self.post_error(
                f"Error in step {stage_name} (after running for {minutes:.1f} minutes): {exc}",
            )
            raise
        else:
            minutes = (time.monotonic() - started) / 60
            self.post_info(
                f'Step "{stage_name}" completed successfully after {minutes:.1f} minutes.',
            )
