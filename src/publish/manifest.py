"""Persist the identifiers created by ``prepare`` for ``release``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishManifest:
    """Identifiers needed to finish a dataset publish."""

    model_run_ref: str
    asset_prefix: str
    zenodo_draft_id: str
    zenodo_draft_url: str
    pr_url: str
    pr_number: int
    dataset_end_label: str


def default_manifest_path(model_run_ref: str, directory: Path | None = None) -> Path:
    """Return ``publish-<ref>.json`` in ``directory`` or the current directory."""
    root = Path.cwd() if directory is None else directory
    return root / f"publish-{model_run_ref}.json"


def write_manifest(path: Path, manifest: PublishManifest) -> None:
    """Write the manifest as JSON."""
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> PublishManifest:
    """Load a previously written manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PublishManifest(
        model_run_ref=payload["model_run_ref"],
        asset_prefix=payload["asset_prefix"],
        zenodo_draft_id=str(payload["zenodo_draft_id"]),
        zenodo_draft_url=payload["zenodo_draft_url"],
        pr_url=payload["pr_url"],
        pr_number=int(payload["pr_number"]),
        dataset_end_label=payload["dataset_end_label"],
    )
