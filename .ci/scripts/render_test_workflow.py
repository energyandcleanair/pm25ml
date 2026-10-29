"""Derive a dummy Cloud Workflow from the production orchestrator template.

The test workflow keeps the production control flow (groups, resume guards,
profile lookup, parallel imputer training) but replaces Cloud Run and Cloud
Batch job calls with ``run_dummy_stage``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from render_release_workflow import render, values_for

RUN_CALL = "googleapis.run.v1.namespaces.jobs.run"
BATCH_CALL = "googleapis.batch.v1.projects.locations.jobs.create"
JOB_CALLS = {RUN_CALL, BATCH_CALL}

MODULE_TO_STAGE = {
    "pm25ml.run.s000_resolve_model_run_ref": "resolve_model_run_ref",
    "pm25ml.run.s001_preflight": "preflight",
    "pm25ml.run.s005_discover": "discover",
    "pm25ml.run.s010_fetch_and_combine": "collect_data_and_combine",
    "pm25ml.run.s020_generate_features": "generate_features",
    "pm25ml.run.s030_sample_for_imputation": "sample",
    "pm25ml.run.s050_impute": "impute",
    "pm25ml.run.s060_prep_for_full_model": "full_model_sample",
    "pm25ml.run.s070_train_full_model": "full_model_train",
    "pm25ml.run.s080_predict_full_model": "full_model_predict",
    "pm25ml.run.s090_outputs": "output_results",
}

TRAIN_MODULE_EXPR = '${"pm25ml.run.s040_train_" + model + "_imputer"}'
TRAIN_STAGE_EXPR = '${"train_" + model}'
FAIL_AT_EXPR = '${ default(map.get(args, "fail_at"), "") }'

DUMMY_SUBWORKFLOW = {
    "params": ["stage", "fail_at"],
    "steps": [
        {
            "maybe_fail": {
                "switch": [
                    {
                        "condition": '${ fail_at != "" and fail_at == stage }',
                        "steps": [
                            {
                                "fail": {
                                    "raise": (
                                        '${ "Simulated failure at stage: " + string(stage) }'
                                    ),
                                },
                            },
                        ],
                    },
                ],
            },
        },
        {"succeed": {"return": "${stage}"}},
    ],
}


def protect_expressions(text: str) -> tuple[str, list[str]]:
    """Replace Workflows ``${...}`` values with placeholders for YAML parsing."""
    expressions: list[str] = []
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("${", i):
            end = _expression_end(text, i)
            expressions.append(text[i:end])
            out.append(f"__EXPR_{len(expressions) - 1}__")
            i = end
            continue
        out.append(text[i])
        i += 1
    return "".join(out), expressions


def restore_expressions(value: Any, expressions: list[str]) -> Any:
    """Put original ``${...}`` strings back into a parsed YAML tree."""
    if isinstance(value, str):
        match = re.fullmatch(r"__EXPR_(\d+)__", value)
        if match:
            return expressions[int(match.group(1))]
        return value
    if isinstance(value, list):
        return [restore_expressions(item, expressions) for item in value]
    if isinstance(value, dict):
        return {key: restore_expressions(item, expressions) for key, item in value.items()}
    return value


def load_workflow_yaml(text: str) -> dict[str, Any]:
    """Parse a Cloud Workflows YAML document, preserving expressions."""
    protected, expressions = protect_expressions(text)
    loaded = yaml.safe_load(protected)
    restored = restore_expressions(loaded, expressions)
    if not isinstance(restored, dict):
        msg = "workflow YAML must be a mapping"
        raise TypeError(msg)
    return restored


def dump_workflow_yaml(data: Mapping[str, Any]) -> str:
    """Serialize a workflow mapping, quoting expression strings."""
    return yaml.dump(
        data,
        Dumper=_WorkflowDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        width=120,
    )


def extract_stage(call_node: Mapping[str, Any]) -> str:
    """Return the dummy stage name for a Cloud Run or Batch job call."""
    modules: list[Any] = []
    _collect_modules(call_node.get("args"), modules)
    if len(modules) != 1:
        msg = f"expected one python -m module in job call, found {modules!r}"
        raise ValueError(msg)
    module = modules[0]
    if isinstance(module, str) and "s040_train_" in module and "model" in module:
        return TRAIN_STAGE_EXPR
    stage = MODULE_TO_STAGE.get(module)
    if stage is None:
        msg = f"unknown pipeline module: {module!r}"
        raise ValueError(msg)
    return stage


def replace_job_calls(node: Any) -> Any:
    """Replace Cloud Run and Batch calls with ``run_dummy_stage``."""
    if isinstance(node, dict):
        if node.get("call") in JOB_CALLS:
            return {
                "call": "run_dummy_stage",
                "args": {
                    "stage": extract_stage(node),
                    "fail_at": "${fail_at}",
                },
            }
        return {key: replace_job_calls(value) for key, value in node.items()}
    if isinstance(node, list):
        return [replace_job_calls(item) for item in node]
    return node


def inject_fail_at(workflow: dict[str, Any]) -> None:
    """Record the optional ``fail_at`` argument during initialisation."""
    init_env = _find_named_step(workflow["main"]["steps"], "initialisation")
    init_steps = init_env["steps"]
    assign_step = _find_named_step(init_steps, "init_env")
    assign_step["assign"].append({"fail_at": FAIL_AT_EXPR})


def render_test_workflow(source: str) -> str:
    """Render the commit production workflow as a dummy test workflow."""
    rendered = render(source, values_for("commit", None))
    workflow = load_workflow_yaml(rendered)
    inject_fail_at(workflow)
    transformed = replace_job_calls(workflow)
    transformed["run_dummy_stage"] = DUMMY_SUBWORKFLOW
    return dump_workflow_yaml(transformed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a dummy test workflow from infra/workflow.yaml.",
    )
    parser.add_argument("--workflow-out", type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / "infra" / "workflow.yaml").read_text()
    args.workflow_out.write_text(render_test_workflow(source))


def _collect_modules(node: Any, modules: list[Any]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            if item == "-m" and index + 1 < len(node):
                modules.append(node[index + 1])
        for item in node:
            _collect_modules(item, modules)
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect_modules(value, modules)


def _find_named_step(steps: list[Any], name: str) -> dict[str, Any]:
    for item in steps:
        if isinstance(item, dict) and name in item:
            found = item[name]
            if not isinstance(found, dict):
                msg = f"step {name!r} is not a mapping"
                raise TypeError(msg)
            return found
    msg = f"step not found: {name}"
    raise KeyError(msg)


def _expression_end(text: str, start: int) -> int:
    depth = 0
    i = start
    in_string: str | None = None
    while i < len(text):
        char = text[i]
        if in_string:
            if char == "\\" and in_string == '"':
                i += 2
                continue
            if char == in_string:
                in_string = None
            i += 1
            continue
        if char in {'"', "'"}:
            in_string = char
            i += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    msg = f"unclosed Workflows expression at offset {start}"
    raise ValueError(msg)


class _WorkflowDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    if data.startswith("${") or any(char in data for char in ":{}#[]&*!|>%@`"):
        style = "'" if "'" not in data else '"'
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_WorkflowDumper.add_representer(str, _represent_str)


if __name__ == "__main__":
    main()
