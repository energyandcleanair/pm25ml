"""Report production pipeline stages to Slack using FST-style timing messages."""

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
_STEP_COLOR: Final = "#1447E0"
_INFO_COLOR: Final = "good"
_ERROR_COLOR: Final = "#E91E63"


def _slack_enabled(environ: Mapping[str, str]) -> bool:
    return bundle_source(environ) == BUNDLE_SOURCE_RELEASE and bool(
        environ.get("SLACK_API_TOKEN", "").strip(),
    )


def _message_text(headline: str, environ: Mapping[str, str]) -> str:
    lines = [headline]
    model_run_ref = environ.get("MODEL_RUN_REF", "").strip()
    profile = environ.get("PIPELINE_PROFILE", "").strip()
    if model_run_ref:
        lines.append(f"MODEL_RUN_REF={model_run_ref}")
    if profile:
        lines.append(f"PIPELINE_PROFILE={profile}")
    return "\n".join(lines)


def _post_stage(
    environ: Mapping[str, str],
    *,
    title: str,
    color: str,
    headline: str,
) -> None:
    token = environ.get("SLACK_API_TOKEN", "").strip()
    if not token:
        return
    payload = {
        "channel": SLACK_CHANNEL,
        "text": headline,
        "attachments": [
            {
                "color": color,
                "title": title,
                "text": _message_text(headline, environ),
                "mrkdwn_in": ["text"],
            },
        ],
    }
    try:
        response = requests.post(
            CHAT_POST_MESSAGE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            logger.warning(
                "Slack rejected production stage log: %s",
                body.get("error", "unknown_error"),
            )
    except requests.RequestException:
        logger.warning("Failed to post production stage to Slack", exc_info=True)


@contextmanager
def report_production_stage(stage_name: str) -> Iterator[None]:
    """Log a production stage start, duration, and failure to Slack when enabled."""
    enabled = _slack_enabled(os.environ)
    if enabled:
        _post_stage(
            os.environ,
            title="STEP",
            color=_STEP_COLOR,
            headline=f"## Running step: {stage_name} ##",
        )
    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        minutes = (time.monotonic() - started) / 60
        if enabled:
            _post_stage(
                os.environ,
                title="ERROR",
                color=_ERROR_COLOR,
                headline=(
                    f"Error in step {stage_name} (after running for {minutes:.1f} minutes): {exc}"
                ),
            )
        raise
    else:
        minutes = (time.monotonic() - started) / 60
        if enabled:
            _post_stage(
                os.environ,
                title="INFO",
                color=_INFO_COLOR,
                headline=(
                    f'Step "{stage_name}" completed successfully after {minutes:.1f} minutes.'
                ),
            )
