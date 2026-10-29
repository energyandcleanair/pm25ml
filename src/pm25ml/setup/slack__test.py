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
    assert "text" not in start_payload
    start_attachment = start_payload["attachments"][0]
    assert start_attachment["title"] == "STEP"
    assert start_attachment["footer"] == "STEP"
    assert start_attachment["color"] == "#1447E0"
    assert start_attachment["fallback"] == "Running step: preflight"
    assert start_attachment["text"].startswith(f"_{MODEL_RUN_REF} | india_\n")
    assert "Running step: preflight" in start_attachment["text"]
    assert "PM25ML" not in start_attachment["text"]
    assert isinstance(start_attachment["ts"], int)
    assert done_payload["attachments"][0]["title"] == "INFO"
    assert done_payload["attachments"][0]["footer"] == "INFO"
    assert "text" not in done_payload
    assert (
        'Step "preflight" completed successfully after 2.0 minutes.'
        in done_payload["attachments"][0]["text"]
    )


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
    assert error_payload["attachments"][0]["footer"] == "ERROR"
    assert error_payload["attachments"][0]["color"] == "#E91E63"
    assert "text" not in error_payload
    assert (
        "Error in step impute (after running for 1.0 minutes): boom"
        in error_payload["attachments"][0]["text"]
    )


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

    _slack().post_step("Running step: preflight")

    payload = _body(responses.calls[0])
    assert "text" not in payload
    assert payload["attachments"][0]["title"] == "STEP"
    assert payload["attachments"][0]["color"] == "#1447E0"
    assert payload["attachments"][0]["text"].startswith(f"_{MODEL_RUN_REF} | india_\n")
    assert "Running step: preflight" in payload["attachments"][0]["text"]


@responses.activate
def test__post_info__posts_caller_message_when_release() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack().post_info("Using MODEL_RUN_REF\nrel-1")

    assert len(responses.calls) == 1
    payload = _body(responses.calls[0])
    assert "text" not in payload
    assert payload["attachments"][0]["text"].startswith(f"_{MODEL_RUN_REF} | india_\n")
    assert "Using MODEL_RUN_REF" in payload["attachments"][0]["text"]
    assert "rel-1" in payload["attachments"][0]["text"]
    assert payload["attachments"][0]["title"] == "INFO"
    assert payload["attachments"][0]["color"] == "good"


@responses.activate
def test__post_error__posts_caller_message_when_release() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack().post_error("Error in step impute: boom")

    payload = _body(responses.calls[0])
    assert "text" not in payload
    assert "Error in step impute: boom" in payload["attachments"][0]["text"]
    assert payload["attachments"][0]["title"] == "ERROR"
    assert payload["attachments"][0]["color"] == "#E91E63"


@responses.activate
@pytest.mark.parametrize(
    ("environ", "expected_prefix"),
    [
        ({}, ""),
        ({"MODEL_RUN_REF": MODEL_RUN_REF}, f"_{MODEL_RUN_REF}_\n"),
        ({"PIPELINE_PROFILE": "india"}, "_india_\n"),
    ],
)
def test__post_info__italic_context_line_uses_whichever_fields_are_set(
    environ: dict[str, str],
    expected_prefix: str,
) -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    _slack(
        {
            "PM25ML_BUNDLE_SOURCE": "release",
            "SLACK_API_TOKEN": TOKEN,
            **environ,
        },
    ).post_info("hello")

    text = _body(responses.calls[0])["attachments"][0]["text"]
    assert text == f"{expected_prefix}hello"


@responses.activate
def test__post_info__commit_does_not_post() -> None:
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    _slack({"PM25ML_BUNDLE_SOURCE": "commit", "SLACK_API_TOKEN": TOKEN}).post_info("hello")

    assert len(responses.calls) == 0
