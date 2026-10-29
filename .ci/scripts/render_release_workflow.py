"""Render commit and release Cloud Workflow and Cloud Run specs.

``infra/workflow.yaml`` and ``infra/run.yaml`` are templates. The only values
that change between deploys are the ``__PM25ML_*__`` tokens named at the top of
those files. This script substitutes them; it does not rewrite job names or
image paths by search-and-replace.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATTERN = re.compile(r"__PM25ML_[A-Z0-9_]+__")

# Keys must match the __PM25ML_*__ tokens in infra/workflow.yaml and infra/run.yaml.
COMMIT_VALUES = {
    "PM25ML_JOB_NAME": "pm25ml",
    "PM25ML_IMAGE_TAG": "latest",
    "PM25ML_BUNDLE_SOURCE": "commit",
    "PM25ML_MODEL_BUNDLE": "",
}


def release_values(version: str) -> dict[str, str]:
    """Values for the published X.Y.Z tag deploy."""
    return {
        "PM25ML_JOB_NAME": "pm25ml-release",
        "PM25ML_IMAGE_TAG": version,
        "PM25ML_BUNDLE_SOURCE": "release",
        "PM25ML_MODEL_BUNDLE": f"mb-{version}",
    }


def values_for(variant: str, version: str | None) -> dict[str, str]:
    """Return the token map for a deploy variant."""
    if variant == "commit":
        if version:
            msg = "commit variant does not take --version"
            raise ValueError(msg)
        return dict(COMMIT_VALUES)
    if not version:
        msg = "release variant requires --version X.Y.Z"
        raise ValueError(msg)
    return release_values(version)


def render(source: str, values: Mapping[str, str]) -> str:
    """Replace ``__PM25ML_*__`` tokens present in ``source``.

    Tokens that appear in the file must have a value. Extra values are ignored
    so ``run.yaml`` can omit bundle tokens that only ``workflow.yaml`` uses.
    """
    present = set(TOKEN_PATTERN.findall(source))
    known = {f"__{key}__" for key in values}
    missing = present - known
    if missing:
        msg = f"no value for tokens: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    rendered = source
    for key, value in values.items():
        rendered = rendered.replace(f"__{key}__", value)
    leftover = TOKEN_PATTERN.findall(rendered)
    if leftover:
        msg = f"unsubstituted tokens: {', '.join(sorted(set(leftover)))}"
        raise ValueError(msg)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render infra/workflow.yaml and infra/run.yaml for commit or release.",
    )
    parser.add_argument("variant", choices=("commit", "release"))
    parser.add_argument("--version", help="X.Y.Z tag; required for the release variant")
    parser.add_argument("--workflow-out", type=Path, required=True)
    parser.add_argument("--job-out", type=Path, required=True)
    args = parser.parse_args()

    values = values_for(args.variant, args.version)
    args.workflow_out.write_text(
        render((ROOT / "infra" / "workflow.yaml").read_text(), values),
    )
    args.job_out.write_text(render((ROOT / "infra" / "run.yaml").read_text(), values))


if __name__ == "__main__":
    main()
