"""Tests for version-controlled Zenodo deposit metadata."""

from pathlib import Path

import pytest

from publish.zenodo import load_deposit_metadata

REF = "rel-2026-10-r1_s-1.0_mb-1.5.1"


def test__load_deposit_metadata__repo_docs__sets_description_and_sections() -> None:
    metadata = load_deposit_metadata(version=REF)

    assert metadata["version"] == REF
    assert "High-resolution surface PM2.5" in metadata["description"]
    assert "Data versioning" not in metadata["description"]
    assert "Disclaimer" not in metadata["description"]
    assert "<h3>Disclaimer</h3>" in metadata["notes"]
    assert "legal status of any country" in metadata["notes"]
    types = [item["type"]["id"] for item in metadata["additional_descriptions"]]
    assert types == ["technical-info", "notes"]
    technical = metadata["additional_descriptions"][0]["description"]
    notes = metadata["additional_descriptions"][1]["description"]
    assert "<h3>Data versioning</h3>" in technical
    assert "rel-YYYY-MM-rT" in technical
    assert notes == metadata["notes"]
    assert metadata["title"] == "South Asia PM2.5 map: India | Data"
    assert metadata["upload_type"] == "dataset"
    assert metadata["license"] == "cc-by-4.0"
    assert metadata["imprint_publisher"] == "Centre for Research on Energy and Clean Air"
    names = [creator["name"] for creator in metadata["creators"]]
    assert names == ["Rushwood, Panda", "Gierens, Rosa", "Kawano, Ayako", "Thieriot, Hubert"]
    assert metadata["creators"][1]["orcid"] == "0000-0002-3879-3099"
    assert len(metadata["references"]) == 14
    assert (
        metadata["custom"]["code:codeRepository"] == "https://github.com/energyandcleanair/pm25ml"
    )
    assert "publication_date" not in metadata
    assert "doi" not in metadata


def test__load_deposit_metadata__missing_required_field__raises(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "zenodo.yaml").write_text("title: Only a title\n", encoding="utf-8")
    (docs_dir / "dataset.md").write_text("Dataset body\n", encoding="utf-8")
    (docs_dir / "versioning.md").write_text("### Versioning\n", encoding="utf-8")
    (docs_dir / "disclaimer.md").write_text("### Disclaimer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="upload_type"):
        load_deposit_metadata(version=REF, docs_dir=docs_dir)
