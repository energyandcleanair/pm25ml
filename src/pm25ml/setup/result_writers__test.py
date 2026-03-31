from typing import cast

from pm25ml.results.cv_summary_result_writer import CvSummaryResultWriter
from pm25ml.results.final_result_storage import FinalResultStorage
from pm25ml.results.netcdf_final_result_writer import NetCdfResultWriter
from pm25ml.setup.result_writers import define_result_writers, define_stats_writers


class DummyStorage:
    pass


def test__define_result_writers__returns_netcdf_writer() -> None:
    writers = define_result_writers(
        storage=cast("FinalResultStorage", DummyStorage()),
        model_run_ref="v2026.03",
    )

    assert len(writers) == 1
    assert isinstance(writers[0], NetCdfResultWriter)

    netcdf_writer = writers[0]
    assert netcdf_writer.model_run_ref == "v2026.03"


def test__define_stats_writers__returns_cv_summary_writer() -> None:
    writers = define_stats_writers(
        storage=cast("FinalResultStorage", DummyStorage()),
        model_run_ref="v2026.03",
    )

    assert len(writers) == 1
    assert isinstance(writers[0], CvSummaryResultWriter)

    cv_summary_writer = writers[0]
    assert cv_summary_writer.model_run_ref == "v2026.03"
