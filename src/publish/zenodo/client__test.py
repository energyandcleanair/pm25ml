"""Tests for the Zenodo draft client."""

import json
from pathlib import Path

import pytest
import responses

from publish.zenodo.client import CONCEPT_RECORD_ID, ZenodoClient, ZenodoError

API = "https://zenodo.test/api"
REF = "rel-2026-07-r0_s-1.0_mb-1.3.0"
DEPOSITIONS_QUERY = f"{API}/deposit/depositions?q=conceptrecid:{CONCEPT_RECORD_ID}"


def _client() -> ZenodoClient:
    return ZenodoClient(token="token", api_url=API)


def _metadata(**fields: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "South Asia PM2.5 map: India | Data",
        "upload_type": "dataset",
    }
    payload.update(fields)
    return payload


@responses.activate
def test__ensure_draft__new_version__uploads_files_and_sets_version(tmp_path: Path) -> None:
    netcdf = tmp_path / f"pm25-full_{REF}.nc"
    csv_path = tmp_path / f"pm25-cv-summary_{REF}.csv"
    netcdf.write_bytes(b"nc")
    csv_path.write_text("r2,rmse\n0.6,27\n", encoding="utf-8")

    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={"id": 111, "links": {}},
    )
    responses.get(DEPOSITIONS_QUERY, json=[])
    responses.post(f"{API}/records/111/versions", json={"id": 222})
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": [{"key": "old.nc"}]})
    responses.delete(f"{API}/records/222/draft/files/old.nc", status=204)
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/{netcdf.name}/content", json={})
    responses.post(f"{API}/records/222/draft/files/{netcdf.name}/commit", json={})
    responses.put(f"{API}/records/222/draft/files/{csv_path.name}/content", json={})
    responses.post(f"{API}/records/222/draft/files/{csv_path.name}/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={
            "id": 222,
            "metadata": {"version": REF},
            "links": {"self_html": "https://zenodo.org/uploads/222"},
        },
    )

    draft = _client().ensure_draft(version=REF, files=(netcdf, csv_path), metadata=_metadata())

    assert draft.record_id == "222"
    assert draft.html_url == "https://zenodo.org/uploads/222"
    metadata_puts = [
        call
        for call in responses.calls
        if call.request.method == "PUT" and call.request.url == f"{API}/deposit/depositions/222"
    ]
    assert json.loads(metadata_puts[0].request.body)["metadata"]["version"] == REF
    assert "description" not in json.loads(metadata_puts[0].request.body)["metadata"]


@responses.activate
def test__ensure_draft__existing_matching_draft__reuses_draft(tmp_path: Path) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={"id": 111, "links": {"draft": f"{API}/records/222/draft"}},
    )
    responses.get(
        DEPOSITIONS_QUERY,
        json=[{"id": 222, "submitted": False, "state": "unsubmitted"}],
    )
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    draft = _client().ensure_draft(version=REF, files=(netcdf,), metadata=_metadata())

    assert draft.record_id == "222"
    assert not any(
        call.request.method == "POST" and call.request.url.endswith("/versions")
        for call in responses.calls
    )


@responses.activate
def test__ensure_draft__existing_draft_with_previous_version__reuses_draft(
    tmp_path: Path,
) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={"id": 111, "links": {"draft": f"{API}/records/222/draft"}},
    )
    responses.get(
        DEPOSITIONS_QUERY,
        json=[{"id": 222, "submitted": False, "state": "unsubmitted"}],
    )
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    draft = _client().ensure_draft(version=REF, files=(netcdf,), metadata=_metadata())

    assert draft.record_id == "222"
    assert not any(
        call.request.method == "POST" and call.request.url.endswith("/versions")
        for call in responses.calls
    )


@responses.activate
def test__ensure_draft__already_published_this_version__does_not_create_or_replace(
    tmp_path: Path,
) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={
            "id": 222,
            "metadata": {"version": REF},
            "links": {"self_html": "https://zenodo.org/records/222"},
        },
    )
    responses.get(DEPOSITIONS_QUERY, json=[])

    draft = _client().ensure_draft(version=REF, files=(netcdf,))

    assert draft.record_id == "222"
    assert not any(call.request.method == "POST" for call in responses.calls)
    assert not any(call.request.method == "PUT" for call in responses.calls)


@responses.activate
def test__ensure_draft__published_record_in_edit_mode__creates_new_version(
    tmp_path: Path,
) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={
            "id": 111,
            "submitted": True,
            "state": "done",
            "metadata": {"version": "rel-2026-07-r0_s-1.0_mb-1.3-r0"},
            "links": {"draft": f"{API}/records/111/draft"},
        },
    )
    responses.get(DEPOSITIONS_QUERY, json=[{"id": 111, "submitted": True, "state": "done"}])
    responses.post(f"{API}/records/111/versions", json={"id": 222})
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    draft = _client().ensure_draft(version=REF, files=(netcdf,), metadata=_metadata())

    assert draft.record_id == "222"
    assert any(
        call.request.method == "POST" and call.request.url.endswith("/versions")
        for call in responses.calls
    )


@responses.activate
def test__ensure_draft__create_version_conflict__reuses_existing_draft(
    tmp_path: Path,
) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={"id": 111, "links": {}},
    )
    responses.get(DEPOSITIONS_QUERY, json=[])
    responses.post(f"{API}/records/111/versions", status=400, json={"message": "draft exists"})
    responses.get(
        DEPOSITIONS_QUERY,
        json=[{"id": 222, "submitted": False, "state": "unsubmitted"}],
    )
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    draft = _client().ensure_draft(version=REF, files=(netcdf,), metadata=_metadata())

    assert draft.record_id == "222"


@responses.activate
def test__publish_draft__sets_publication_date_then_publishes() -> None:
    from datetime import date

    responses.get(
        f"{API}/deposit/depositions/222",
        json={"metadata": {"version": REF}},
    )
    responses.put(f"{API}/deposit/depositions/222", json={})
    responses.post(f"{API}/deposit/depositions/222/actions/publish", json={"id": 222})

    _client().publish_draft("222", publication_date=date(2026, 7, 22))

    put = next(call for call in responses.calls if call.request.method == "PUT")
    assert json.loads(put.request.body)["metadata"]["publication_date"] == "2026-07-22"


@responses.activate
def test__ensure_draft__description_and_html_files__updates_only_this_draft(
    tmp_path: Path,
) -> None:
    netcdf = tmp_path / "a.nc"
    html = tmp_path / "dataset.html"
    netcdf.write_bytes(b"nc")
    html.write_text("<h1>Dataset</h1>", encoding="utf-8")
    responses.get(
        f"{API}/records/{CONCEPT_RECORD_ID}",
        json={"id": 111, "links": {}},
    )
    responses.get(DEPOSITIONS_QUERY, json=[])
    responses.post(f"{API}/records/111/versions", json={"id": 222})
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.put(f"{API}/records/222/draft/files/dataset.html/content", json={})
    responses.post(f"{API}/records/222/draft/files/dataset.html/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    _client().ensure_draft(
        version=REF,
        files=(netcdf, html),
        metadata=_metadata(description="<h1>Dataset</h1>"),
    )

    metadata_put = next(
        call
        for call in responses.calls
        if call.request.method == "PUT" and call.request.url == f"{API}/deposit/depositions/222"
    )
    payload = json.loads(metadata_put.request.body)["metadata"]
    assert payload["version"] == REF
    assert payload["description"] == "<h1>Dataset</h1>"
    uploaded = [
        call.request.url
        for call in responses.calls
        if call.request.method == "PUT" and "/content" in call.request.url
    ]
    assert any(url.endswith("/dataset.html/content") for url in uploaded)


@responses.activate
def test__ensure_draft__additional_descriptions__puts_rdm_sections(tmp_path: Path) -> None:
    netcdf = tmp_path / "a.nc"
    netcdf.write_bytes(b"nc")
    additional = [
        {
            "description": "<h3>Data versioning</h3>",
            "type": {"id": "technical-info"},
            "lang": {"id": "eng"},
        },
        {
            "description": "<h3>Disclaimer</h3>",
            "type": {"id": "notes"},
            "lang": {"id": "eng"},
        },
    ]
    responses.get(f"{API}/records/{CONCEPT_RECORD_ID}", json={"id": 111, "links": {}})
    responses.get(DEPOSITIONS_QUERY, json=[])
    responses.post(f"{API}/records/111/versions", json={"id": 222})
    responses.put(f"{API}/deposit/depositions/222", json={"id": 222})
    responses.get(f"{API}/records/222/draft", json={"id": 222, "metadata": {"title": "Title"}})
    responses.put(f"{API}/records/222/draft", json={"id": 222})
    responses.get(f"{API}/records/222/draft/files", json={"entries": []})
    responses.post(f"{API}/records/222/draft/files", json={})
    responses.put(f"{API}/records/222/draft/files/a.nc/content", json={})
    responses.post(f"{API}/records/222/draft/files/a.nc/commit", json={})
    responses.get(
        f"{API}/records/222/draft",
        json={"id": 222, "links": {"self_html": "https://zenodo.org/uploads/222"}},
    )

    _client().ensure_draft(
        version=REF,
        files=(netcdf,),
        metadata=_metadata(additional_descriptions=additional, description="<p>Dataset</p>"),
    )

    deposit_put = next(
        call
        for call in responses.calls
        if call.request.method == "PUT" and call.request.url == f"{API}/deposit/depositions/222"
    )
    deposit_payload = json.loads(deposit_put.request.body)["metadata"]
    assert "additional_descriptions" not in deposit_payload
    rdm_put = next(
        call
        for call in responses.calls
        if call.request.method == "PUT" and call.request.url == f"{API}/records/222/draft"
    )
    rdm_payload = json.loads(rdm_put.request.body)["metadata"]
    assert rdm_payload["additional_descriptions"] == additional
    assert rdm_payload["title"] == "Title"


def test__from_env__missing_token__raises() -> None:
    with pytest.raises(ZenodoError, match="ZENODO_TOKEN"):
        ZenodoClient.from_env(environ={})
