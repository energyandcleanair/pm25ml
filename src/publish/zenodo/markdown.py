"""Render version-controlled Markdown into Zenodo HTML fragments."""

from __future__ import annotations

from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
DATASET_MARKDOWN = DOCS_DIR / "dataset.md"
VERSIONING_MARKDOWN = DOCS_DIR / "versioning.md"
DISCLAIMER_MARKDOWN = DOCS_DIR / "disclaimer.md"


def markdown_to_html_fragment(source: str) -> str:
    """Convert Markdown to an HTML fragment suitable for Zenodo."""
    return markdown.markdown(
        source,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html",
    )


def render_markdown_file(path: Path) -> str:
    """Read a Markdown file and return an HTML fragment."""
    return markdown_to_html_fragment(path.read_text(encoding="utf-8"))
