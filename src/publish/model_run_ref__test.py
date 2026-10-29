"""Tests for production model-run reference checks."""

import pytest

from publish.model_run_ref import ModelRunRefError, require_production_model_run_ref

PRODUCTION = "rel-2026-07-r0_s-1.0_mb-1.3.0"


def test__require_production_model_run_ref__production_ref__returns_stripped_value() -> None:
    assert require_production_model_run_ref(f"  {PRODUCTION}  ") == PRODUCTION


def test__require_production_model_run_ref__local_hex_ref__raises() -> None:
    with pytest.raises(ModelRunRefError, match="production model-run reference"):
        require_production_model_run_ref("rel-7f3a9c2e_s-1.0_mb-local-abc123")
