"""Create and publish dataset versions on Zenodo."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

CONCEPT_RECORD_ID = "17573562"
DEFAULT_API_URL = "https://zenodo.org/api"
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
_REQUEST_ATTEMPTS = 4


class ZenodoError(RuntimeError):
    """Raised when the Zenodo API rejects a publish request."""


@dataclass(frozen=True)
class ZenodoDraft:
    """An unpublished new version of the dataset concept."""

    record_id: str
    html_url: str


class ZenodoClient:
    """Client for Zenodo record drafts and publishing."""

    def __init__(
        self,
        *,
        token: str,
        api_url: str = DEFAULT_API_URL,
        session: requests.Session | None = None,
        concept_id: str = CONCEPT_RECORD_ID,
    ) -> None:
        """Store credentials and the concept record to version."""
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._session = session or requests.Session()
        self._concept_id = concept_id

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        session: requests.Session | None = None,
    ) -> ZenodoClient:
        """Build from ``ZENODO_TOKEN`` and optional ``ZENODO_API_URL``."""
        values = os.environ if environ is None else environ
        token = values.get("ZENODO_TOKEN", "").strip()
        if not token:
            msg = "ZENODO_TOKEN is required to prepare or release a Zenodo record."
            raise ZenodoError(msg)
        api_url = values.get("ZENODO_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL
        return cls(token=token, api_url=api_url, session=session)

    def ensure_draft(
        self,
        *,
        version: str,
        files: Iterable[Path],
        metadata: Mapping[str, Any] | None = None,
    ) -> ZenodoDraft:
        """
        Create or reuse a draft for ``version`` and replace its files.

        Metadata and HTML docs are written only on this draft, the latest
        unpublished version. Already-published versions are not updated.
        """
        latest = self._get(f"/records/{self._concept_id}")
        draft_id = self._unpublished_deposition_id()
        if draft_id is None and (latest.get("metadata") or {}).get("version") == version:
            return ZenodoDraft(
                record_id=str(latest["id"]),
                html_url=_draft_html_url(latest, str(latest["id"])),
            )
        if draft_id is None:
            draft_id = self._create_version(str(latest["id"]))
        self._put_version_metadata(draft_id, version=version, metadata=metadata)
        self._replace_files(draft_id, files)
        draft = self._get(f"/records/{draft_id}/draft")
        return ZenodoDraft(record_id=draft_id, html_url=_draft_html_url(draft, draft_id))

    def publish_draft(self, record_id: str, *, publication_date: date | None = None) -> None:
        """Set today's publication date and publish the draft."""
        day = publication_date or datetime.now(UTC).date()
        metadata = self._writable_deposit_metadata(record_id)
        metadata["publication_date"] = day.isoformat()
        self._put(f"/deposit/depositions/{record_id}", json={"metadata": metadata})
        self._request("POST", f"/deposit/depositions/{record_id}/actions/publish")

    def _unpublished_deposition_id(self) -> str | None:
        listed = self._request(
            "GET",
            f"/deposit/depositions?q=conceptrecid:{self._concept_id}",
        )
        items: list[Any]
        if isinstance(listed, list):
            items = listed
        elif isinstance(listed, dict):
            items = list((listed.get("hits") or {}).get("hits") or [])
        else:
            items = []
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            if item.get("submitted") is False or item.get("state") == "unsubmitted":
                return str(item["id"])
        return None

    def _create_version(self, latest_id: str) -> str:
        try:
            created = self._request("POST", f"/records/{latest_id}/versions")
        except ZenodoError:
            draft_id = self._unpublished_deposition_id()
            if draft_id is not None:
                return draft_id
            raise
        if not isinstance(created, dict) or "id" not in created:
            msg = "Zenodo did not return a new version id."
            raise ZenodoError(msg)
        return str(created["id"])

    def _put_version_metadata(
        self,
        record_id: str,
        *,
        version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(metadata or {})
        payload.pop("doi", None)
        payload.pop("prereserve_doi", None)
        payload["version"] = version
        self._put(f"/deposit/depositions/{record_id}", json={"metadata": payload})

    def _writable_deposit_metadata(self, record_id: str) -> dict[str, Any]:
        deposit = self._get(f"/deposit/depositions/{record_id}")
        metadata = dict(deposit.get("metadata") or {})
        for key in ("doi", "prereserve_doi"):
            metadata.pop(key, None)
        return metadata

    def _replace_files(self, record_id: str, files: Iterable[Path]) -> None:
        listed = self._get(f"/records/{record_id}/draft/files")
        for entry in listed.get("entries") or listed.get("hits", {}).get("hits") or []:
            key = entry.get("key") or entry.get("filename")
            if key:
                self._request("DELETE", f"/records/{record_id}/draft/files/{key}")
        for path in files:
            self._upload_file(record_id, path)

    def _upload_file(self, record_id: str, path: Path) -> None:
        self._request(
            "POST",
            f"/records/{record_id}/draft/files",
            json=[{"key": path.name}],
        )
        with path.open("rb") as handle:
            self._request(
                "PUT",
                f"/records/{record_id}/draft/files/{path.name}/content",
                data=handle,
                content_type="application/octet-stream",
            )
        self._request("POST", f"/records/{record_id}/draft/files/{path.name}/commit")

    def _get(self, path: str) -> dict[str, Any]:
        payload = self._request("GET", path)
        if not isinstance(payload, dict):
            msg = f"Zenodo returned a non-object payload for GET {path}."
            raise ZenodoError(msg)
        return payload

    def _put(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        payload = self._request("PUT", path, json=json)
        if not isinstance(payload, dict):
            return {}
        return payload

    def _request(  # noqa: PLR0913
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        data: object | None = None,
        content_type: str | None = None,
        absolute: bool = False,
    ) -> object:
        url = path if absolute else f"{self._api_url}{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        elif json is not None:
            headers["Content-Type"] = "application/json"
        attempts = _REQUEST_ATTEMPTS if data is None else 1
        last_error = ""
        for attempt in range(attempts):
            response = self._session.request(method, url, headers=headers, json=json, data=data)
            if response.ok:
                if not response.content:
                    return {}
                return response.json()
            last_error = f"Zenodo {method} {url} failed ({response.status_code}): {response.text}"
            if response.status_code not in _RETRYABLE_STATUS or attempt == attempts - 1:
                raise ZenodoError(last_error)
            time.sleep(2**attempt)
        raise ZenodoError(last_error)


def _draft_html_url(draft: dict[str, Any], record_id: str) -> str:
    links = draft.get("links") or {}
    return str(
        links.get("self_html") or links.get("html") or f"https://zenodo.org/uploads/{record_id}"
    )
