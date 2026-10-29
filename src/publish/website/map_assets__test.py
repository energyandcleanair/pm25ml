"""Tests for map asset helpers that do not require write_cog."""

from fsspec.implementations.memory import MemoryFileSystem

from publish.website.map_assets import color_properties, custom_colormap, switch_live_index


def test__custom_colormap__default_samples__has_requested_length() -> None:
    assert custom_colormap(32).N == 32


def test__color_properties__crea_map__includes_units_and_scale() -> None:
    props = color_properties(custom_colormap(), (0.0, 100.4), "µg/m³")

    assert props["minval"] == 0
    assert props["maxval"] == 100
    assert props["unit"] == "µg/m³"
    assert isinstance(props["scale"], list)
    assert str(props["scale"][0]).startswith("#")


def test__switch_live_index__versioned_files__copied_to_bucket_root() -> None:
    filesystem = MemoryFileSystem()
    filesystem.pipe("bucket/pm25-full_rel/metadata.json", b'{"ok": true}')
    filesystem.pipe("bucket/pm25-full_rel/color_scale.json", b'{"scale": []}')

    switch_live_index(filesystem=filesystem, asset_prefix="pm25-full_rel", bucket_id="bucket")

    assert filesystem.cat("bucket/metadata.json") == b'{"ok": true}'
    assert filesystem.cat("bucket/color_scale.json") == b'{"scale": []}'
