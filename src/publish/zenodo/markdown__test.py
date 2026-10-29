"""Tests for Markdown-to-HTML dataset documentation."""

from publish.zenodo.markdown import markdown_to_html_fragment, render_markdown_file


def test__markdown_to_html_fragment__heading_and_link__renders_html() -> None:
    html = markdown_to_html_fragment("# Title\n\nSee [CREA](https://energyandcleanair.org/).")

    assert "<h1>Title</h1>" in html
    assert 'href="https://energyandcleanair.org/"' in html


def test__render_markdown_file__reads_path(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Data versioning\n\n`rel-YYYY-MM-rT`\n", encoding="utf-8")

    html = render_markdown_file(path)

    assert "<h1>Data versioning</h1>" in html
    assert "<code>rel-YYYY-MM-rT</code>" in html
