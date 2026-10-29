"""Tests for the website metadata pull-request client."""

import base64
import json

import pytest
import responses

from publish.website.github import GitHubWebsiteClient, GitHubWebsiteError, gh_auth_token

API = "https://github.test"
REPO = "energyandcleanair/southasia-pm2-5"
REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def _client() -> GitHubWebsiteClient:
    return GitHubWebsiteClient(token="token", repo=REPO, api_url=API)


@responses.activate
def test__ensure_metadata_pull_request__new_branch__creates_file_and_pr() -> None:
    responses.get(
        f"{API}/repos/{REPO}/git/ref/heads/{REF}",
        status=404,
        json={"message": "Not Found"},
    )
    responses.get(
        f"{API}/repos/{REPO}/git/ref/heads/main",
        json={"object": {"sha": "abc"}},
    )
    responses.post(f"{API}/repos/{REPO}/git/refs", json={})
    responses.get(
        f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json",
        json={"sha": "file-sha"},
    )
    responses.put(f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json", json={})
    responses.get(f"{API}/repos/{REPO}/pulls", json=[])
    responses.post(
        f"{API}/repos/{REPO}/pulls",
        json={"number": 4, "html_url": "https://github.com/example/pr/4"},
    )

    result = _client().ensure_metadata_pull_request(
        model_run_ref=REF,
        dataset_end_label="May 2026",
    )

    assert result.number == 4
    assert result.branch == REF
    created = next(
        call
        for call in responses.calls
        if call.request.url.endswith("/pulls") and call.request.method == "POST"
    )
    body = json.loads(created.request.body)
    assert body["title"] == REF
    assert body["head"] == REF
    file_put = next(
        call
        for call in responses.calls
        if call.request.method == "PUT" and "datasetMetadata.json" in call.request.url
    )
    encoded = json.loads(file_put.request.body)
    decoded = base64.b64decode(encoded["content"]).decode("utf-8")
    assert json.loads(decoded) == {"latestDatasetEndDate": "May 2026"}
    assert encoded["message"] == "Update latestDatasetEndDate to May 2026"


@responses.activate
def test__ensure_metadata_pull_request__existing_open_pr__reuses_pr() -> None:
    responses.get(
        f"{API}/repos/{REPO}/git/ref/heads/{REF}",
        json={"object": {"sha": "def"}},
    )
    responses.get(
        f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json",
        json={"sha": "file-sha"},
    )
    responses.put(f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json", json={})
    responses.get(
        f"{API}/repos/{REPO}/pulls",
        json=[{"number": 4, "html_url": "https://github.com/example/pr/4"}],
    )

    result = _client().ensure_metadata_pull_request(
        model_run_ref=REF,
        dataset_end_label="May 2026",
    )

    assert result.number == 4
    assert not any(
        call.request.method == "POST" and call.request.url.endswith("/pulls")
        for call in responses.calls
    )


@responses.activate
def test__ensure_metadata_pull_request__unchanged_file__skips_put() -> None:
    contents = json.dumps({"latestDatasetEndDate": "May 2026"}, indent=2) + "\n"
    encoded = base64.b64encode(contents.encode("utf-8")).decode("ascii")
    responses.get(
        f"{API}/repos/{REPO}/git/ref/heads/{REF}",
        json={"object": {"sha": "def"}},
    )
    responses.get(
        f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json",
        json={"sha": "file-sha", "content": encoded, "encoding": "base64"},
    )
    responses.get(
        f"{API}/repos/{REPO}/pulls",
        json=[{"number": 4, "html_url": "https://github.com/example/pr/4", "state": "open"}],
    )

    result = _client().ensure_metadata_pull_request(
        model_run_ref=REF,
        dataset_end_label="May 2026",
    )

    assert result.number == 4
    assert not any(call.request.method == "PUT" for call in responses.calls)


@responses.activate
def test__ensure_metadata_pull_request__merged_pr__reuses_without_new_pr() -> None:
    contents = json.dumps({"latestDatasetEndDate": "May 2026"}, indent=2) + "\n"
    encoded = base64.b64encode(contents.encode("utf-8")).decode("ascii")
    responses.get(
        f"{API}/repos/{REPO}/git/ref/heads/{REF}",
        json={"object": {"sha": "def"}},
    )
    responses.get(
        f"{API}/repos/{REPO}/contents/src/data/datasetMetadata.json",
        json={"sha": "file-sha", "content": encoded, "encoding": "base64"},
    )
    responses.get(
        f"{API}/repos/{REPO}/pulls",
        json=[{"number": 4, "html_url": "https://github.com/example/pr/4", "state": "closed"}],
    )

    result = _client().ensure_metadata_pull_request(
        model_run_ref=REF,
        dataset_end_label="May 2026",
    )

    assert result.number == 4
    assert not any(
        call.request.method == "POST" and call.request.url.endswith("/pulls")
        for call in responses.calls
    )


@responses.activate
def test__merge_pull_request__success__puts_merge() -> None:
    responses.put(f"{API}/repos/{REPO}/pulls/4/merge", json={"merged": True})

    _client().merge_pull_request(4)


@responses.activate
def test__merge_pull_request__api_error__raises() -> None:
    responses.put(f"{API}/repos/{REPO}/pulls/4/merge", status=405, json={"message": "nope"})

    with pytest.raises(GitHubWebsiteError, match="405"):
        _client().merge_pull_request(4)


def test__gh_auth_token__github_token_set__returns_stripped_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "  ghp_example  ")

    assert gh_auth_token() == "ghp_example"


def test__gh_auth_token__missing_token__raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(GitHubWebsiteError, match="GITHUB_TOKEN"):
        gh_auth_token()
