"""Open and merge the website metadata pull request through the GitHub API."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import requests

WEBSITE_REPO = "energyandcleanair/southasia-pm2-5"
METADATA_PATH = "src/data/datasetMetadata.json"
GITHUB_API_URL = "https://api.github.com"


class GitHubWebsiteError(RuntimeError):
    """Raised when the website repository API rejects a request."""


@dataclass(frozen=True)
class WebsitePullRequest:
    """An open pull request that updates the dataset end date."""

    number: int
    html_url: str
    branch: str


TokenProvider = Callable[[], str]


def gh_auth_token() -> str:
    """Return the token from ``GITHUB_TOKEN``."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        msg = "GITHUB_TOKEN is required to open or merge the website pull request."
        raise GitHubWebsiteError(msg)
    return token


class GitHubWebsiteClient:
    """Create or update the dataset-metadata pull request without a checkout."""

    def __init__(
        self,
        *,
        token: str,
        repo: str = WEBSITE_REPO,
        api_url: str = GITHUB_API_URL,
        session: requests.Session | None = None,
    ) -> None:
        """Store credentials and the website repository."""
        self._token = token
        self._repo = repo
        self._api_url = api_url.rstrip("/")
        self._session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        *,
        token_provider: TokenProvider = gh_auth_token,
        session: requests.Session | None = None,
        repo: str = WEBSITE_REPO,
        api_url: str = GITHUB_API_URL,
    ) -> GitHubWebsiteClient:
        """Build a client using ``GITHUB_TOKEN``."""
        return cls(token=token_provider(), repo=repo, api_url=api_url, session=session)

    def ensure_metadata_pull_request(
        self,
        *,
        model_run_ref: str,
        dataset_end_label: str,
    ) -> WebsitePullRequest:
        """Create or update a PR whose branch and title are ``model_run_ref``."""
        contents = json.dumps({"latestDatasetEndDate": dataset_end_label}, indent=2) + "\n"
        message = f"Update latestDatasetEndDate to {dataset_end_label}"
        if not self._ref_exists(f"heads/{model_run_ref}"):
            self._create_ref(model_run_ref, self._ref_sha("heads/main"))
        self._ensure_file(
            path=METADATA_PATH,
            content=contents,
            branch=model_run_ref,
            message=message,
        )
        existing = self._existing_pull_request(model_run_ref)
        if existing is not None:
            return existing
        try:
            created = self._json_object(
                self._request(
                    "POST",
                    f"/repos/{self._repo}/pulls",
                    json={
                        "title": model_run_ref,
                        "head": model_run_ref,
                        "base": "main",
                        "body": message,
                    },
                ),
            )
        except GitHubWebsiteError:
            existing = self._existing_pull_request(model_run_ref)
            if existing is not None:
                return existing
            raise
        return WebsitePullRequest(
            number=int(created["number"]),
            html_url=str(created["html_url"]),
            branch=model_run_ref,
        )

    def merge_pull_request(self, number: int) -> None:
        """Merge an open website pull request."""
        self._request("PUT", f"/repos/{self._repo}/pulls/{number}/merge")

    def _existing_pull_request(self, head: str) -> WebsitePullRequest | None:
        owner = self._repo.split("/", maxsplit=1)[0]
        pulls = self._request(
            "GET",
            f"/repos/{self._repo}/pulls",
            params={"head": f"{owner}:{head}", "state": "all"},
        )
        if not isinstance(pulls, list):
            return None
        chosen: dict[str, object] | None = None
        for entry in pulls:
            if not isinstance(entry, dict):
                continue
            if entry.get("state") == "open":
                chosen = entry
                break
            if chosen is None:
                chosen = entry
        if chosen is None:
            return None
        return WebsitePullRequest(
            number=int(chosen["number"]),
            html_url=str(chosen["html_url"]),
            branch=head,
        )

    def _ref_exists(self, ref: str) -> bool:
        response = self._session.request(
            "GET",
            f"{self._api_url}/repos/{self._repo}/git/ref/{ref}",
            headers=self._headers(),
        )
        if response.status_code == requests.codes.not_found:
            return False
        if not response.ok:
            self._raise(response)
        return True

    def _ref_sha(self, ref: str) -> str:
        payload = self._json_object(self._request("GET", f"/repos/{self._repo}/git/ref/{ref}"))
        git_object = payload["object"]
        if not isinstance(git_object, dict):
            msg = f"GitHub ref {ref} did not include an object sha."
            raise GitHubWebsiteError(msg)
        return str(git_object["sha"])

    def _create_ref(self, branch: str, sha: str) -> None:
        self._request(
            "POST",
            f"/repos/{self._repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )

    def _ensure_file(
        self,
        *,
        path: str,
        content: str,
        branch: str,
        message: str,
    ) -> None:
        payload = self._json_object(
            self._request(
                "GET",
                f"/repos/{self._repo}/contents/{path}",
                params={"ref": branch},
            ),
        )
        current = _decode_contents_file(payload)
        if current is not None and current == content:
            return
        self._put_file(
            path=path,
            content=content,
            branch=branch,
            message=message,
            sha=str(payload["sha"]),
        )

    def _put_file(
        self,
        *,
        path: str,
        content: str,
        branch: str,
        message: str,
        sha: str,
    ) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        self._request(
            "PUT",
            f"/repos/{self._repo}/contents/{path}",
            json={
                "message": message,
                "content": encoded,
                "branch": branch,
                "sha": sha,
            },
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: Mapping[str, str] | None = None,
    ) -> object:
        response = self._session.request(
            method,
            f"{self._api_url}{path}",
            headers=self._headers(),
            json=json,
            params=params,
        )
        if not response.ok:
            self._raise(response)
        if not response.content:
            return {}
        return response.json()

    def _json_object(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            msg = "GitHub returned a non-object JSON payload."
            raise GitHubWebsiteError(msg)
        return payload

    def _raise(self, response: requests.Response) -> None:
        msg = (
            f"GitHub {response.request.method} {response.url} failed "
            f"({response.status_code}): {response.text}"
        )
        raise GitHubWebsiteError(msg)


def _decode_contents_file(payload: dict[str, object]) -> str | None:
    raw = payload.get("content")
    if not isinstance(raw, str):
        return None
    return base64.b64decode("".join(raw.split())).decode("utf-8")
