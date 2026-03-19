"""
Run the PM2.5 pipeline locally with an optional continue_from step.

Behavior: run the specified step and all subsequent steps; skip those before.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from runpy import run_module


@dataclass(frozen=True)
class Step:
    key: str
    module: str


def _ordered_steps() -> list[Step]:
    """Return ordered list of (canonical_step_key, module_path)."""
    base = "pm25ml.run"
    return [
        Step("s00_preflight", f"{base}.s00_preflight"),
        Step("s01_fetch_and_combine", f"{base}.s01_fetch_and_combine"),
        Step("s02_generate_features", f"{base}.s02_generate_features"),
        Step("s03_sample_for_imputation", f"{base}.s03_sample_for_imputation"),
        Step("s04_train_aod_imputer", f"{base}.s04_train_aod_imputer"),
        Step("s04_train_co_imputer", f"{base}.s04_train_co_imputer"),
        Step("s04_train_no2_imputer", f"{base}.s04_train_no2_imputer"),
        Step("s05_impute", f"{base}.s05_impute"),
        Step("s06_prep_for_full_model", f"{base}.s06_prep_for_full_model"),
        Step("s07_train_full_model", f"{base}.s07_train_full_model"),
        Step("s08_predict_full_model", f"{base}.s08_predict_full_model"),
    ]


_steps = tuple(_ordered_steps())
_aliases: dict[str, str] = {
    # s00
    "preflight": "s00_preflight",
    "s00_preflight": "s00_preflight",
    # s01
    "fetch_and_combine": "s01_fetch_and_combine",
    "s01_fetch_and_combine": "s01_fetch_and_combine",
    # s02
    "generate_features": "s02_generate_features",
    "s02_generate_features": "s02_generate_features",
    # s03
    "sample_for_imputation": "s03_sample_for_imputation",
    "s03_sample_for_imputation": "s03_sample_for_imputation",
    # s04
    "train_aod_imputer": "s04_train_aod_imputer",
    "s04_train_aod_imputer": "s04_train_aod_imputer",
    "train_co_imputer": "s04_train_co_imputer",
    "s04_train_co_imputer": "s04_train_co_imputer",
    "train_no2_imputer": "s04_train_no2_imputer",
    "s04_train_no2_imputer": "s04_train_no2_imputer",
    # s05
    "impute": "s05_impute",
    "s05_impute": "s05_impute",
    # s06
    "prep_for_full_model": "s06_prep_for_full_model",
    "s06_prep_for_full_model": "s06_prep_for_full_model",
    # s07
    "train_full_model": "s07_train_full_model",
    "s07_train_full_model": "s07_train_full_model",
    # s08
    "predict_full_model": "s08_predict_full_model",
    "s08_predict_full_model": "s08_predict_full_model",
}


def _main(continue_from: str) -> None:
    start_key = _resolve_continue_from(continue_from)
    _run_steps_from(start_key)


def _resolve_continue_from(continue_from: str) -> str:
    key = (continue_from or "preflight").strip()
    canonical = _aliases.get(key)
    if not canonical:
        valid = ", ".join(sorted(_aliases.keys()))
        msg = "Unknown continue_from '" + str(continue_from) + "'. Valid values: " + valid
        raise ValueError(msg)
    return canonical


def _run_steps_from(step_key: str) -> None:
    start_index = _identify_start_index(step_key)

    for step in _steps[start_index:]:
        run_module(step.module, run_name="__main__")


def _identify_start_index(step_key: str) -> int:
    start_index: int | None = None
    for i, step in enumerate(_steps):
        if step.key == step_key:
            start_index = i
            break
    if start_index is None:
        msg = "Internal error: step not found: " + str(step_key)
        raise RuntimeError(msg)
    return start_index


if __name__ == "__main__":
    continue_from = sys.argv[1] if len(sys.argv) > 1 else "preflight"
    _main(continue_from)
