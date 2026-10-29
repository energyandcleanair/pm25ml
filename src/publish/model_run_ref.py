"""Validate production model-run references for publishing."""

from __future__ import annotations

import re

_PRODUCTION_REF = re.compile(
    r"^rel-\d{4}-\d{2}-r\d+_s-\d+\.\d+_mb-\d+\.\d+\.\d+$",
)


class ModelRunRefError(ValueError):
    """Raised when a model-run reference cannot be published."""


def require_production_model_run_ref(value: str) -> str:
    """
    Return a production ``rel-YYYY-MM-rT_s-X.Y_mb-X.Y.Z`` ref.

    Args:
        value: Candidate model-run reference.

    Returns:
        The stripped production reference.

    Raises:
        ModelRunRefError: If the value is empty, local, or otherwise invalid.

    """
    ref = value.strip()
    if not _PRODUCTION_REF.fullmatch(ref):
        msg = (
            "Publishing requires a production model-run reference of the form "
            f"rel-YYYY-MM-rT_s-X.Y_mb-X.Y.Z, got {value!r}."
        )
        raise ModelRunRefError(msg)
    return ref
