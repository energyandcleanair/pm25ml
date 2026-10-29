"""Tests for output-stage Slack result folder messaging."""

from unittest.mock import Mock

from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.run.s090_outputs import _final_results_message

MODEL_RUN_REF = "rel-2026-10-r0_s-1.0_mb-1.2.0"


def test__final_results_message__includes_console_folder_and_gs_prefix() -> None:
    storage = FinalResultStorage(
        filesystem=Mock(),
        destination_bucket="crea-pm25ml-final-results",
        output_path=f"country=india/run={MODEL_RUN_REF}",
    )

    text = _final_results_message(storage, gcp_project="vital-form-455413-p8")

    folder = f"country=india/run={MODEL_RUN_REF}"
    assert "## Final results ##" in text
    assert (
        "https://console.cloud.google.com/storage/browser/"
        f"crea-pm25ml-final-results/{folder}?project=vital-form-455413-p8"
    ) in text
    assert f"gs://crea-pm25ml-final-results/{folder}" in text
    assert "storage.cloud.google.com" not in text
