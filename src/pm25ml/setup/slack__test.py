"""Tests for production Slack reporting."""

from __future__ import annotations

import json

import pytest
import responses

from pm25ml.setup.slack import CHAT_POST_MESSAGE_URL, SLACK_CHANNEL, ProductionSlack

TOKEN = "xoxb-test-token"
MODEL_RUN_REF = "rel-2026-10-r0_s-1.0_mb-1.2.0"


def _release_environ() -> dict[str, str]:
    return {
        "PM25ML_BUNDLE_SOURCE": "release",
        "SLACK_API_TOKEN": TOKEN,
        "MODEL_RUN_REF": MODEL_RUN_REF,
        "PIPELINE_PROFILE": "india",
    }


def _slack(environ: dict[str, str] | None = None) -> ProductionSlack:
    return ProductionSlack.from_env(environ or _release_environ())


def _body(call: responses.Call) -> dict:
    raw = call.request.body
    text = raw.decode() if isinstance(raw, bytes) else raw
    return json.loads(text)


@responses.activate
def test__report_stage__posts_start_and_success_in_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    monkeypatch.setattr("pm25ml.setup.slack.time.monotonic", lambda: 60.0)
    slack = _slack()

    with slack.report_stage("preflight"):
        monkeypatch.setattr("pm25ml.setup.slack.time.monotonic", lambda: 180.0)

    assert len(responses.calls) == 2
    start = responses.calls[0].request
    start_payload = _body(responses.calls[0])
    done_payload = _body(responses.calls[1])
    assert start.headers["Authorization"] == f"Bearer {TOKEN}"
    assert start_payload["channel"] == SLACK_CHANNEL
    assert start_payload["text"] == "## Running step: preflight ##"
    assert start_payload["attachments"][0]["title"] == "STEP"
    assert start_payload["attachments"][0]["color"] == "#1447E0"
    assert "MODEL_RUN_REF=rel-2026-10-r0_s-1.0_mb-1.2.0" in start_payload["attachments"][0]["text"]
    assert "PIPELINE_PROFILE=india" in start_payload["attachments"][0]["text"]
    assert done_payload["attachments"][0]["title"] == "INFO"
    assert done_payload["text"] == 'Step "preflight" completed successfully after 2.0 minutes.'


@responses.activate
def test__report_stage__posts_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    times = iter([10.0, 70.0])
    monkeypatch.setattr("pm25ml.setup.slack.time.monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="boom"), _slack().report_stage("impute"):
        raise RuntimeError("boom")

    assert len(responses.calls) == 2
    error_payload = _body(responses.calls[1])
    assert error_payload["attachments"][0]["title"] == "ERROR"
    assert error_payload["attachments"][0]["color"] == "#E91E63"
    assert error_payload["text"] == "Error in step impute (after running for 1.0 minutes): boom"


@responses.activate
def test__report_stage__commit_does_not_post() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    slack = _slack(
        {
            "PM25ML_BUNDLE_SOURCE": "commit",
            "SLACK_API_TOKEN": TOKEN,
        },
    )

    with slack.report_stage("preflight"):
        pass

    assert len(responses.calls) == 0


@responses.activate
def test__report_stage__release_without_token_does_not_post() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    slack = _slack({"PM25ML_BUNDLE_SOURCE": "release"})

    with slack.report_stage("preflight"):
        pass

    assert len(responses.calls) == 0


@responses.activate
def test__report_stage__slack_ok_false_does_not_raise() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": False, "error": "not_in_channel"}, status=200)

    with _slack().report_stage("preflight"):
        pass

    assert len(responses.calls) == 2


@responses.activate
def test__report_stage__http_5xx_does_not_raise() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": False}, status=503)

    with _slack().report_stage("preflight"):
        pass

    assert len(responses.calls) == 2


@responses.activate
def test__post_step__posts_caller_message_when_release() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack().post_step("## Running step: preflight ##")

    payload = _body(responses.calls[0])
    assert payload["text"] == "## Running step: preflight ##"
    assert payload["attachments"][0]["title"] == "STEP"
    assert payload["attachments"][0]["color"] == "#1447E0"


@responses.activate
def test__post_info__posts_caller_message_when_release() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack().post_info("## Using MODEL_RUN_REF ##\nrel-1")

    assert len(responses.calls) == 1
    payload = _body(responses.calls[0])
    assert payload["text"] == "## Using MODEL_RUN_REF ##\nrel-1"
    assert payload["attachments"][0]["title"] == "INFO"
    assert payload["attachments"][0]["color"] == "good"


@responses.activate
def test__post_error__posts_caller_message_when_release() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack().post_error("Error in step impute: boom")

    payload = _body(responses.calls[0])
    assert payload["text"] == "Error in step impute: boom"
    assert payload["attachments"][0]["title"] == "ERROR"
    assert payload["attachments"][0]["color"] == "#E91E63"


@responses.activate
def test__post_info__commit_does_not_post() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack({"PM25ML_BUNDLE_SOURCE": "commit", "SLACK_API_TOKEN": TOKEN}).post_info("hello")

    assert len(responses.calls) == 0
