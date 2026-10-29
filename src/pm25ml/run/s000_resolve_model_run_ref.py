"""Allocate or reuse MODEL_RUN_REF before any other pipeline stage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem

from pm25ml.setup.logging_injection import init_logging_from_env
from pm25ml.setup.model_run_ref import (
    BUNDLE_SOURCE_RELEASE,
    apply_model_run_ref_to_environ,
    bundle_source,
    format_resolved_model_run_ref,
    log_resolved_model_run_ref,
    persist_resolved_model_run_ref,
    resolve_model_run_ref,
)

if TYPE_CHECKING:
    from pm25ml.setup.slack import ProductionSlack

STAGE_TITLE = "resolve_model_run_ref"


def _filesystem() -> GCSFileSystem | LocalFileSystem:
    needs_shared_storage = bundle_source(os.environ) == BUNDLE_SOURCE_RELEASE or bool(
        os.environ.get("PM25ML_ALLOCATION_ID", "").strip(),
    )
    if needs_shared_storage:
        return GCSFileSystem()
    return LocalFileSystem()


def _main(slack: ProductionSlack) -> None:
    filesystem = _filesystem()
    model_run_ref = resolve_model_run_ref(os.environ, filesystem)
    persist_resolved_model_run_ref(os.environ, filesystem, model_run_ref)
    apply_model_run_ref_to_environ(os.environ, model_run_ref)
    log_resolved_model_run_ref(model_run_ref)
    slack.post_info(format_resolved_model_run_ref(model_run_ref))


if __name__ == "__main__":
    slack = init_logging_from_env().production_slack()
    with slack.report_stage(STAGE_TITLE):
        _main(slack)
