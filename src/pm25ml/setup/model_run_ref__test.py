"""Tests for model-run reference allocation and validation."""

from __future__ import annotations

from subprocess import CompletedProcess
from typing import TYPE_CHECKING

import arrow
import pytest
from morefs.memory import MemFS

from pm25ml.results.netcdf_final_result_writer import SCHEMA_VERSION
from pm25ml.setup.model_run_ref import (
    CombinedRunStorage,
    ModelRunRefError,
    ProcessVersions,
    allocate_production_model_run_ref,
    list_existing_run_refs,
    load_model_run_ref_for_process,
    next_production_rel,
    parse_model_run_ref,
    resolve_model_bundle,
    resolve_model_run_ref,
    write_allocated_model_run_ref,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _git_run(commands: Mapping[tuple[str, ...], CompletedProcess[str]]):
    def _run(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        return commands[tuple(args)]

    return _run


def test__parse_model_run_ref__accepts_production_and_local_forms() -> None:
    production = parse_model_run_ref("rel-2026-10-r0_s-1.0_mb-1.2.0")
    local = parse_model_run_ref("rel-7f3a9c2e_s-1.0_mb-local-abc123-dirty")

    assert production.rel == "rel-2026-10-r0"
    assert production.schema == "1.0"
    assert production.bundle == "mb-1.2.0"
    assert local.bundle == "mb-local-abc123-dirty"


def test__parse_model_run_ref__rejects_legacy_bundle() -> None:
    with pytest.raises(ModelRunRefError, match="retired bundle"):
        parse_model_run_ref("rel-2026-10-r0_s-1.0_mb-1.2-r0")


def test__next_production_rel__uses_numeric_counter_for_current_month() -> None:
    existing = (
        "rel-2026-10-r9_s-1.0_mb-1.2.0",
        "rel-2026-10-r10_s-1.0_mb-1.2.0",
        "rel-2026-09-r50_s-1.0_mb-1.2.0",
        "rel-7f3a9c2e_s-1.0_mb-local-abc",
    )

    assert next_production_rel(existing, now=arrow.get("2026-10-29T12:00:00+00:00")) == (
        "rel-2026-10-r11"
    )


def test__allocate_production_model_run_ref__skips_claimed_counter() -> None:
    filesystem = MemFS()
    filesystem.makedirs("combined/country=india/run=rel-2026-10-r0_s-1.0_mb-1.2.0", exist_ok=True)
    with filesystem.open(
        "combined/country=india/run=rel-2026-10-r0_s-1.0_mb-1.2.0/.reserved",
        "wt",
    ) as handle:
        handle.write("")

    storage = CombinedRunStorage(filesystem, "combined", "india")
    allocated = allocate_production_model_run_ref(
        storage,
        ProcessVersions(schema="1.0", bundle="mb-1.2.0"),
        now=arrow.get("2026-10-29T12:00:00+00:00"),
    )

    assert allocated == "rel-2026-10-r1_s-1.0_mb-1.2.0"
    assert set(list_existing_run_refs(storage)) == {
        "rel-2026-10-r0_s-1.0_mb-1.2.0",
        "rel-2026-10-r1_s-1.0_mb-1.2.0",
    }
    with filesystem.open(
        "combined/country=india/run=rel-2026-10-r1_s-1.0_mb-1.2.0/.reserved",
        "rt",
    ) as handle:
        assert handle.read() == ""


def test__resolve_model_bundle__local_includes_sha_and_dirty() -> None:
    run = _git_run(
        {
            ("git", "rev-parse", "HEAD"): CompletedProcess(
                ["git", "rev-parse", "HEAD"],
                0,
                "abc123def\n",
                "",
            ),
            ("git", "status", "--porcelain"): CompletedProcess(
                ["git", "status", "--porcelain"],
                0,
                " M src/pm25ml/setup/model_run_ref.py\n",
                "",
            ),
        },
    )

    assert resolve_model_bundle({}, run=run) == "mb-local-abc123def-dirty"


def test__resolve_model_bundle__commit_uses_embedded_sha() -> None:
    bundle = resolve_model_bundle(
        {"PM25ML_BUNDLE_SOURCE": "commit", "PM25ML_COMMIT_SHA": "deadbeef"},
    )

    assert bundle == "mb-local-deadbeef"


def test__resolve_model_run_ref__release_rejects_schema_mismatch() -> None:
    filesystem = MemFS()
    ref = "rel-2026-10-r0_s-0.9_mb-1.2.0"
    filesystem.makedirs(f"combined/country=india/run={ref}", exist_ok=True)
    with filesystem.open(
        f"combined/country=india/run={ref}/.reserved",
        "wt",
    ) as handle:
        handle.write("")

    with pytest.raises(ModelRunRefError, match="schema"):
        resolve_model_run_ref(
            {
                "PM25ML_BUNDLE_SOURCE": "release",
                "MODEL_BUNDLE": "1.2.0",
                "MODEL_RUN_REF": ref,
                "COMBINED_BUCKET_NAME": "combined",
                "PIPELINE_PROFILE": "india",
            },
            filesystem,
        )


def test__resolve_model_run_ref__local_overwrite_ignores_current_sha() -> None:
    filesystem = MemFS()
    run = _git_run(
        {
            ("git", "rev-parse", "HEAD"): CompletedProcess(
                ["git", "rev-parse", "HEAD"],
                0,
                "aaaaaaaa\n",
                "",
            ),
            ("git", "status", "--porcelain"): CompletedProcess(
                ["git", "status", "--porcelain"],
                0,
                "",
                "",
            ),
        },
    )
    explicit = f"rel-7f3a9c2e_s-{SCHEMA_VERSION}_mb-local-bbbbbbbb-dirty"

    resolved = resolve_model_run_ref(
        {"MODEL_RUN_REF": explicit},
        filesystem,
        run=run,
    )

    assert resolved == explicit


def test__resolve_model_run_ref__local_generates_short_rel() -> None:
    filesystem = MemFS()
    run = _git_run(
        {
            ("git", "rev-parse", "HEAD"): CompletedProcess(
                ["git", "rev-parse", "HEAD"],
                0,
                "abc123\n",
                "",
            ),
            ("git", "status", "--porcelain"): CompletedProcess(
                ["git", "status", "--porcelain"],
                0,
                "",
                "",
            ),
        },
    )

    resolved = resolve_model_run_ref(
        {},
        filesystem,
        run=run,
        token_hex=lambda _nbytes: "7f3a9c2e",
    )

    assert resolved == f"rel-7f3a9c2e_s-{SCHEMA_VERSION}_mb-local-abc123"


def test__load_model_run_ref_for_process__reads_allocation_object() -> None:
    filesystem = MemFS()
    write_allocated_model_run_ref(
        filesystem,
        "combined",
        "exec-1",
        "rel-7f3a9c2e_s-1.0_mb-local-abc123",
    )

    loaded = load_model_run_ref_for_process(
        {
            "PM25ML_ALLOCATION_ID": "exec-1",
            "COMBINED_BUCKET_NAME": "combined",
        },
        filesystem,
    )

    assert loaded == "rel-7f3a9c2e_s-1.0_mb-local-abc123"
