"""Tests for local pipeline stage routing and Cloud Workflow structure."""

import sys
from pathlib import Path

from pm25ml.run._run_local import _aliases, _ordered_steps, _resolve_continue_from

REPO_ROOT = Path(__file__).parents[3]
SCRIPTS = REPO_ROOT / ".ci" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from render_release_workflow import render, values_for
from render_test_workflow import JOB_CALLS, load_workflow_yaml, render_test_workflow
from validate_test_workflow import TRAIN_STAGES, match_stages


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


EXPECTED_GROUPS = [
    "initialisation",
    "data_preparation",
    "features_and_imputation",
    "full_model",
    "output",
]
EXPECTED_STAGE_INDEX = {
    "preflight": 0,
    "discover": 1,
    "collect_data_and_combine": 2,
    "generate_features": 3,
    "sample": 4,
    "train": 5,
    "impute": 6,
    "full_model_sample": 7,
    "full_model_train": 8,
    "full_model_predict": 9,
    "output_results": 10,
}
PIPELINE_MODULES = [
    "pm25ml.run.s000_resolve_model_run_ref",
    "pm25ml.run.s001_preflight",
    "pm25ml.run.s005_discover",
    "pm25ml.run.s010_fetch_and_combine",
    "pm25ml.run.s020_generate_features",
    "pm25ml.run.s030_sample_for_imputation",
    "pm25ml.run.s050_impute",
    "pm25ml.run.s060_prep_for_full_model",
    "pm25ml.run.s070_train_full_model",
    "pm25ml.run.s080_predict_full_model",
    "pm25ml.run.s090_outputs",
]


def test__cloud_workflow__groups_stages_and_guards_resume() -> None:
    source = (REPO_ROOT / "infra" / "workflow.yaml").read_text()
    workflow = load_workflow_yaml(source)
    groups = [_step_name(step) for step in workflow["main"]["steps"]]

    assert groups == EXPECTED_GROUPS
    assert "dispatch_continue_from" not in source
    assert "run_dummy_stage" not in source
    assert "fail_at" not in source
    for module in PIPELINE_MODULES:
        assert module in source
    assert '${"pm25ml.run.s040_train_" + model + "_imputer"}' in source
    assert "n4-standard-8" in source
    assert "n4-standard-16" in source
    assert "c2d-highcpu-112" in source
    assert "stage_index.discover >= start_index" in source
    assert "map.get(stage_index, continue_from) == null" in source

    init = _named(_named(workflow["main"]["steps"], "initialisation")["steps"], "init_env")
    profiles = _assign_value(init["assign"], "profiles")
    stage_index = _assign_value(init["assign"], "stage_index")
    assert stage_index == EXPECTED_STAGE_INDEX
    assert profiles["india"] == {
        "GEE_GRID_ASSET_PATH": "projects/vital-form-455413-p8/assets/grid_india_10km_shapefile",
        "PROFILE_GRID_CELL_COUNT": "33074",
        "PM25_SOURCE_IDS": "cpcb",
    }
    assert profiles["in-bd"] == {
        "GEE_GRID_ASSET_PATH": "projects/vital-form-455413-p8/assets/grid_in-bd_10km_shapefile",
        "PROFILE_GRID_CELL_COUNT": "34357",
        "PM25_SOURCE_IDS": "cpcb",
    }


def test__generated_test_workflow__replaces_job_calls_with_dummy_runner() -> None:
    source = (REPO_ROOT / "infra" / "workflow.yaml").read_text()
    generated = render_test_workflow(source)
    workflow = load_workflow_yaml(generated)

    assert "run_dummy_stage" in workflow
    assert "fail_at" in generated
    for call in JOB_CALLS:
        assert call not in generated
    for module in PIPELINE_MODULES:
        assert module not in generated
    assert '${"train_" + model}' in generated
    assert workflow["main"]["steps"][0]
    init = _named(_named(workflow["main"]["steps"], "initialisation")["steps"], "init_env")
    assert _assign_value(init["assign"], "fail_at") == '${ default(map.get(args, "fail_at"), "") }'


def test__match_stages__allows_unordered_train_group() -> None:
    match_stages(
        [
            "resolve_model_run_ref",
            "sample",
            "train_co",
            "train_aod",
            "train_no2",
            "impute",
        ],
        ["resolve_model_run_ref", "sample", TRAIN_STAGES, "impute"],
    )


def test__infra_templates__render_commit_and_release_from_yaml_tokens() -> None:
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
    assert "secretKeyRef" in commit_job
    assert "secretVariables: ${common_secret_env}" in commit_workflow
    assert "secretVariables: ${common_secret_env}" in release_workflow
    assert (
        commit_workflow.count(
            "projects/vital-form-455413-p8/secrets/SLACK_API_TOKEN/versions/latest",
        )
        == 1
    )
    assert "load_slack_api_token" not in commit_workflow
    assert "load_slack_api_token" not in release_workflow


def _step_name(step: dict) -> str:
    return next(iter(step))


def _named(steps: list, name: str) -> dict:
    for step in steps:
        if name in step:
            return step[name]
    raise KeyError(name)


def _assign_value(assign: list, key: str):
    for item in assign:
        if key in item:
            return item[key]
    raise KeyError(key)
