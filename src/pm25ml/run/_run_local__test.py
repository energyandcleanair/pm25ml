"""Tests for local pipeline stage routing."""

import sys
from pathlib import Path

from pm25ml.run._run_local import _aliases, _ordered_steps, _resolve_continue_from

REPO_ROOT = Path(__file__).parents[3]


def test__ordered_steps__resolver_precedes_preflight_and_discovery() -> None:
    step_keys = [step.key for step in _ordered_steps()]

    assert step_keys[:4] == [
        "s000_resolve_model_run_ref",
        "s001_preflight",
        "s005_discover",
        "s010_fetch_and_combine",
    ]
    assert step_keys[-1] == "s090_outputs"


def test__resolve_continue_from__discovery_aliases_route_to_discovery() -> None:
    assert _resolve_continue_from("discover") == "s005_discover"
    assert _resolve_continue_from("s005_discover") == "s005_discover"
    assert _aliases["discover_and_collect"] == "s005_discover"
    assert _aliases["preflight"] == "s001_preflight"
    assert _aliases["s000_preflight"] == "s001_preflight"


def test__cloud_workflow__routes_through_the_discovery_job() -> None:
    workflow = (REPO_ROOT / "infra" / "workflow.yaml").read_text()

    assert "pm25ml.run.s000_resolve_model_run_ref" in workflow
    assert "pm25ml.run.s001_preflight" in workflow
    assert "pm25ml.run.s005_discover" in workflow
    assert '- condition: ${ continue_from == "discover" }\n            next: discover' in workflow
    assert "pm25ml.run.s000_resolve_model_run_ref" in workflow
    assert "next: dispatch_continue_from" in workflow


def test__infra_templates__render_commit_and_release_from_yaml_tokens() -> None:
    sys.path.append(str(REPO_ROOT / ".ci" / "scripts"))
    from render_release_workflow import render, values_for

    workflow = (REPO_ROOT / "infra" / "workflow.yaml").read_text()
    job = (REPO_ROOT / "infra" / "run.yaml").read_text()

    commit_workflow = render(workflow, values_for("commit", None))
    commit_job = render(job, values_for("commit", None))
    release_workflow = render(workflow, values_for("release", "1.2.0"))
    release_job = render(job, values_for("release", "1.2.0"))

    assert "__PM25ML_" not in commit_workflow
    assert "__PM25ML_" not in release_workflow
    assert "PM25ML_BUNDLE_SOURCE: commit" in commit_workflow
    assert 'MODEL_BUNDLE: ""' in commit_workflow
    assert "jobs/pm25ml\n" in commit_workflow or commit_workflow.count("/jobs/pm25ml") >= 1
    assert "pm25ml/pm25ml:latest" in commit_workflow
    assert "name: pm25ml\n" in commit_job
    assert "PM25ML_BUNDLE_SOURCE: release" in release_workflow
    assert 'MODEL_BUNDLE: "mb-1.2.0"' in release_workflow
    assert "jobs/pm25ml-release" in release_workflow
    assert "pm25ml/pm25ml:v1.2.0" in release_workflow
    assert "name: pm25ml-release" in release_job
    assert "pm25ml-feature-" in release_workflow
    assert "name: SLACK_API_TOKEN" in commit_job
    assert "name: SLACK_API_TOKEN" in release_job
    assert "secrets/SLACK_API_TOKEN/versions/latest" in commit_workflow
    assert "secrets/SLACK_API_TOKEN/versions/latest" in release_workflow
