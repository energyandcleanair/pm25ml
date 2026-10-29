"""Deploy and exercise the dummy PM25ML orchestrator workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from render_test_workflow import render_test_workflow

WORKFLOW_NAME = "pm25ml-orchestrator-test"
LOCATION = "europe-west1"
PROJECT = "vital-form-455413-p8"
SERVICE_ACCOUNT = "pm25ml-orchestrator@vital-form-455413-p8.iam.gserviceaccount.com"

TRAIN_STAGES = frozenset({"train_aod", "train_co", "train_no2"})
AFTER_SAMPLE = [
    "impute",
    "full_model_sample",
    "full_model_train",
    "full_model_predict",
    "output_results",
]
FULL_PATTERN: list[str | frozenset[str]] = [
    "resolve_model_run_ref",
    "preflight",
    "discover",
    "collect_data_and_combine",
    "generate_features",
    "sample",
    TRAIN_STAGES,
    *AFTER_SAMPLE,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Validate against the already deployed test workflow.",
    )
    args = parser.parse_args()
    if not args.skip_deploy:
        deploy_test_workflow()
    failures = run_scenarios()
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("All dummy workflow scenarios passed.")


def deploy_test_workflow() -> None:
    source = (ROOT / "infra" / "workflow.yaml").read_text()
    rendered = render_test_workflow(source)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(rendered)
        path = handle.name
    _gcloud(
        [
            "workflows",
            "deploy",
            WORKFLOW_NAME,
            f"--project={PROJECT}",
            f"--source={path}",
            f"--location={LOCATION}",
            f"--service-account={SERVICE_ACCOUNT}",
            "--call-log-level=LOG_ALL_CALLS",
            "--execution-history-level=execution-history-detailed",
            "--quiet",
        ],
        check=True,
    )


def run_scenarios() -> list[str]:
    failures: list[str] = []
    cases = [
        ("full_india", lambda: _check_success({"profile": "india"}, FULL_PATTERN)),
        (
            "resume_discover",
            lambda: _check_success(
                {"profile": "india", "continue_from": "discover"},
                [
                    "resolve_model_run_ref",
                    "discover",
                    "collect_data_and_combine",
                    "generate_features",
                    "sample",
                    TRAIN_STAGES,
                    *AFTER_SAMPLE,
                ],
            ),
        ),
        (
            "resume_impute",
            lambda: _check_success(
                {"profile": "india", "continue_from": "impute"},
                ["resolve_model_run_ref", *AFTER_SAMPLE],
            ),
        ),
        (
            "resume_output",
            lambda: _check_success(
                {"profile": "india", "continue_from": "output_results"},
                ["resolve_model_run_ref", "output_results"],
            ),
        ),
        (
            "skip_training_full",
            lambda: _check_success(
                {"profile": "india", "do_training": False},
                [
                    "resolve_model_run_ref",
                    "preflight",
                    "discover",
                    "collect_data_and_combine",
                    "generate_features",
                    "sample",
                    *AFTER_SAMPLE,
                ],
            ),
        ),
        (
            "skip_training_from_train",
            lambda: _check_success(
                {"profile": "india", "continue_from": "train", "do_training": False},
                ["resolve_model_run_ref", *AFTER_SAMPLE],
            ),
        ),
        (
            "in_bd_output",
            lambda: _check_success(
                {"profile": "in-bd", "continue_from": "output_results"},
                ["resolve_model_run_ref", "output_results"],
            ),
        ),
        (
            "missing_profile",
            lambda: _check_failure({}, "Missing required workflow arg: profile", []),
        ),
        (
            "invalid_profile",
            lambda: _check_failure(
                {"profile": "nope"},
                "Invalid profile value: nope. Expected one of: india, in-bd",
                [],
            ),
        ),
        (
            "invalid_continue_from",
            lambda: _check_failure(
                {"profile": "india", "continue_from": "nope"},
                "Invalid continue_from value: nope",
                [],
            ),
        ),
        (
            "fail_at_impute",
            lambda: _check_failure(
                {"profile": "india", "fail_at": "impute"},
                "Simulated failure at stage: impute",
                [
                    "resolve_model_run_ref",
                    "preflight",
                    "discover",
                    "collect_data_and_combine",
                    "generate_features",
                    "sample",
                    TRAIN_STAGES,
                ],
            ),
        ),
    ]
    for name, action in cases:
        try:
            action()
            print(f"PASS {name}")
        except AssertionError as error:
            failures.append(f"FAIL {name}: {error}")
            print(f"FAIL {name}: {error}")
    return failures


def match_stages(
    actual: Sequence[str],
    pattern: Sequence[str | frozenset[str]],
) -> None:
    """Assert actual dummy stages follow pattern, with unordered train groups."""
    index = 0
    for item in pattern:
        if isinstance(item, frozenset):
            got = set(actual[index : index + len(item)])
            if got != set(item):
                msg = f"expected parallel stages {set(item)} at {index}, got {actual[index : index + len(item)]}"
                raise AssertionError(msg)
            index += len(item)
            continue
        if index >= len(actual) or actual[index] != item:
            msg = f"expected {item!r} at {index}, got {actual[index:]!r}"
            raise AssertionError(msg)
        index += 1
    if index != len(actual):
        msg = f"unexpected extra stages: {list(actual[index:])}"
        raise AssertionError(msg)


def dummy_stages_from_entries(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract successful dummy stage names from execution step entries."""
    stages: list[str] = []
    for entry in entries:
        if not _is_succeeded(entry):
            continue
        if entry.get("routine") != "run_dummy_stage":
            continue
        if entry.get("step") != "succeed":
            continue
        stage = _variable(entry, "stage")
        if not isinstance(stage, str) or not stage:
            msg = f"dummy succeed step missing stage: {entry}"
            raise AssertionError(msg)
        stages.append(stage)
    return stages


def _check_success(
    data: Mapping[str, Any],
    pattern: Sequence[str | frozenset[str]],
) -> None:
    execution = _execute(data, expect_success=True)
    stages = dummy_stages_from_entries(_step_entries(execution["name"]))
    match_stages(stages, pattern)


def _check_failure(
    data: Mapping[str, Any],
    message: str,
    pattern: Sequence[str | frozenset[str]],
) -> None:
    execution = _execute(data, expect_success=False)
    error_text = _error_text(execution)
    if message not in error_text:
        msg = f"expected error {message!r}, got {error_text!r}"
        raise AssertionError(msg)
    stages = dummy_stages_from_entries(_step_entries(execution["name"]))
    match_stages(stages, pattern)


def _execute(data: Mapping[str, Any], *, expect_success: bool) -> dict[str, Any]:
    completed = _gcloud(
        [
            "workflows",
            "run",
            WORKFLOW_NAME,
            f"--project={PROJECT}",
            f"--location={LOCATION}",
            f"--data={json.dumps(data)}",
            "--format=json",
        ],
        check=False,
    )
    if completed.returncode != 0 and not completed.stdout.strip():
        msg = completed.stderr.strip() or "gcloud workflows run failed"
        raise AssertionError(msg)
    execution = json.loads(completed.stdout)
    state = str(execution.get("state", ""))
    succeeded = state in {"SUCCEEDED", "STATE_SUCCEEDED"}
    if expect_success and not succeeded:
        msg = f"expected success, got {state}: {_error_text(execution)}"
        raise AssertionError(msg)
    if not expect_success and succeeded:
        msg = f"expected failure, execution succeeded: {execution}"
        raise AssertionError(msg)
    return execution


def _step_entries(execution_name: str) -> list[dict[str, Any]]:
    token = _gcloud(["auth", "print-access-token"], check=True).stdout.strip()
    entries: list[dict[str, Any]] = []
    page_token = ""
    while True:
        query = urllib.parse.urlencode(
            {"pageSize": 500, **({"pageToken": page_token} if page_token else {})},
        )
        request = urllib.request.Request(
            f"https://workflowexecutions.googleapis.com/v1/{execution_name}/stepEntries?{query}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        entries.extend(payload.get("stepEntries") or [])
        page_token = str(payload.get("nextPageToken") or "")
        if not page_token:
            return entries


def _error_text(execution: Mapping[str, Any]) -> str:
    error = execution.get("error") or {}
    parts = [str(error.get("payload") or ""), str(error.get("context") or "")]
    return " ".join(part for part in parts if part)


def _is_succeeded(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("state", "")) in {"SUCCEEDED", "STATE_SUCCEEDED"}


def _variable(entry: Mapping[str, Any], key: str) -> Any:
    data = entry.get("variableData") or {}
    variables = data.get("variables") if isinstance(data, dict) else None
    if isinstance(variables, dict) and key in variables:
        return variables[key]
    payload = entry.get("payload")
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    return None


def _gcloud(args: Sequence[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["gcloud", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        msg = f"gcloud {' '.join(args)} failed: {detail}"
        raise RuntimeError(msg)
    return completed


if __name__ == "__main__":
    main()
