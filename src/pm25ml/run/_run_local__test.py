"""Tests for local pipeline stage routing."""

from pathlib import Path

from pm25ml.run._run_local import _aliases, _ordered_steps, _resolve_continue_from


def test__ordered_steps__discovery_precedes_collection() -> None:
    step_keys = [step.key for step in _ordered_steps()]

    assert step_keys[:3] == ["s000_preflight", "s005_discover", "s010_fetch_and_combine"]


def test__resolve_continue_from__discovery_aliases_route_to_discovery() -> None:
    assert _resolve_continue_from("discover") == "s005_discover"
    assert _resolve_continue_from("s005_discover") == "s005_discover"
    assert _aliases["discover_and_collect"] == "s005_discover"


def test__cloud_workflow__routes_through_the_discovery_job() -> None:
    workflow = (Path(__file__).parents[3] / "infra" / "workflow.yaml").read_text()

    assert "pm25ml.run.s005_discover" in workflow
    assert '- condition: ${ continue_from == "discover" }\n            next: discover' in workflow
