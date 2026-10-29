"""Load version-controlled Zenodo deposit metadata from the repo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from publish.zenodo.markdown import (
    DATASET_MARKDOWN,
    DISCLAIMER_MARKDOWN,
    DOCS_DIR,
    VERSIONING_MARKDOWN,
    render_markdown_file,
)

if TYPE_CHECKING:
    from pathlib import Path

ZENODO_YAML = DOCS_DIR / "zenodo.yaml"
_DOC_OWNED_KEYS = frozenset(
    {"doi", "prereserve_doi", "version", "description", "notes", "publication_date"}
)
_REQUIRED_KEYS = ("title", "upload_type", "access_right", "license", "creators")


def load_deposit_metadata(
    *,
    version: str,
    docs_dir: Path = DOCS_DIR,
) -> dict[str, Any]:
    """
    Build deposit metadata from docs in the repo.

    ``docs/dataset.md`` is the description, ``docs/versioning.md`` is the
    additional notes, and ``docs/disclaimer.md`` is appended to both.
    ``docs/zenodo.yaml`` supplies the remaining fields. Version and
    publication date are not taken from those files.
    """
    source = docs_dir / ZENODO_YAML.name
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"{source} must contain a mapping of Zenodo deposit fields."
        raise TypeError(msg)
    missing = [key for key in _REQUIRED_KEYS if not loaded.get(key)]
    if missing:
        msg = f"{source} is missing required fields: {', '.join(missing)}."
        raise ValueError(msg)

    metadata = {key: value for key, value in loaded.items() if key not in _DOC_OWNED_KEYS}
    disclaimer_html = render_markdown_file(docs_dir / DISCLAIMER_MARKDOWN.name)
    metadata["version"] = version
    metadata["description"] = _join_html(
        render_markdown_file(docs_dir / DATASET_MARKDOWN.name),
        disclaimer_html,
    )
    metadata["notes"] = _join_html(
        render_markdown_file(docs_dir / VERSIONING_MARKDOWN.name),
        disclaimer_html,
    )
    return metadata


def _join_html(*parts: str) -> str:
    return "\n".join(part.rstrip() for part in parts if part.strip())
