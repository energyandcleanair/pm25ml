"""Tests for production Slack stage reporting."""

from __future__ import annotations

import json

import pytest
import responses

from pm25ml.run._stage_report import (
    CHAT_POST_MESSAGE_URL,
    SLACK_CHANNEL,
    report_production_stage,
)

TOKEN = "xoxb-test-token"


def _release_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PM25ML_BUNDLE_SOURCE", "release")
    monkeypatch.setenv("SLACK_API_TOKEN", TOKEN)
    monkeypatch.setenv("MODEL_RUN_REF", "rel-2026-10-r0_s-1.0_mb-1.2.0")
    monkeypatch.setenv("PIPELINE_PROFILE", "india")


def _body(call: responses.Call) -> dict:
    raw = call.request.body
    text = raw.decode() if isinstance(raw, bytes) else raw
    return json.loads(text)


@responses.activate
def test__report_production_stage__posts_start_and_success_in_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_env(monkeypatch)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    monkeypatch.setattr("pm25ml.run._stage_report.time.monotonic", lambda: 60.0)

    with report_production_stage("preflight"):
        monkeypatch.setattr("pm25ml.run._stage_report.time.monotonic", lambda: 180.0)

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
def test__report_production_stage__posts_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_env(monkeypatch)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)
    times = iter([10.0, 70.0])
    monkeypatch.setattr("pm25ml.run._stage_report.time.monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="boom"), report_production_stage("impute"):
        raise RuntimeError("boom")

    assert len(responses.calls) == 2
    error_payload = _body(responses.calls[1])
    assert error_payload["attachments"][0]["title"] == "ERROR"
    assert error_payload["attachments"][0]["color"] == "#E91E63"
    assert error_payload["text"] == "Error in step impute (after running for 1.0 minutes): boom"


@responses.activate
def test__report_production_stage__commit_does_not_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PM25ML_BUNDLE_SOURCE", "commit")
    monkeypatch.setenv("SLACK_API_TOKEN", TOKEN)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    with report_production_stage("preflight"):
        pass

    assert len(responses.calls) == 0


@responses.activate
def test__report_production_stage__release_without_token_does_not_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PM25ML_BUNDLE_SOURCE", "release")
    monkeypatch.delenv("SLACK_API_TOKEN", raising=False)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": True}, status=200)

    with report_production_stage("preflight"):
        pass

    assert len(responses.calls) == 0


@responses.activate
def test__report_production_stage__slack_ok_false_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_env(monkeypatch)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": False, "error": "not_in_channel"}, status=200)

    with report_production_stage("preflight"):
        pass

    assert len(responses.calls) == 2


@responses.activate
def test__report_production_stage__http_5xx_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_env(monkeypatch)
    responses.post(CHAT_POST_MESSAGE_URL, json={"ok": False}, status=503)

    with report_production_stage("preflight"):
        pass

    assert len(responses.calls) == 2
