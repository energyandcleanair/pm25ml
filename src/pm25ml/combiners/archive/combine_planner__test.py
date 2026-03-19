from arrow import Arrow
from assertpy import assert_that
from pm25ml.collectors.export_pipeline import (
    MissingDataHeuristic,
    PipelineConsumerBehaviour,
    ValueColumnType,
)
from pm25ml.combiners.archive.combine_planner import CombinePlan, CombinePlanner
from pm25ml.collectors.collector import DataCompleteness, UploadResult, PipelineConfig
from pm25ml.hive_path import HivePath
from collections.abc import Collection
from dataclasses import dataclass

from pm25ml.setup.date_params import TemporalConfig


N_GRID_CELLS = 33074


def _value_column_type_map(*column_names: str) -> dict[str, ValueColumnType]:
    return {column_name: ValueColumnType.FLOAT for column_name in column_names}


def test__CombinePlan__month_id():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
        n_grid_cells=N_GRID_CELLS,
    )

    assert desc.month_id == "2023-01", "Month ID should be formatted as 'YYYY-MM'"


def test__CombinePlan__expected_rows():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
        n_grid_cells=N_GRID_CELLS,
    )

    assert desc.expected_rows == N_GRID_CELLS * 31, (
        "Expected rows should be equal to the number of days in the month"
    )


def test__CombinePlan__days_in_month():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
        n_grid_cells=N_GRID_CELLS,
    )

    assert desc.days_in_month == 31, "Days in month should be 31 for January"


def test__plan__valid_results__returns_combine_plans():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config, n_grid_cells=N_GRID_CELLS)

    results: Collection[UploadResult] = [
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=static_dataset/type=static",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("s1", "s2"),
                expected_rows=N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2023",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("y1"),
                expected_rows=N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d1v1", "d1v2"),
                expected_rows=31 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d2v1"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d1v1", "d1v2"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d2v1"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
    ]

    plans = list(planner.plan(results))

    assert_that(plans).contains_only(
        *[
            CombinePlan(
                month=Arrow(2023, 1, 1),
                paths={
                    HivePath("country=india/dataset=static_dataset/type=static"),
                    HivePath("country=india/dataset=yearly_dataset/year=2023"),
                    HivePath("country=india/dataset=monthly_dataset1/month=2023-01"),
                    HivePath("country=india/dataset=monthly_dataset2/month=2023-01"),
                },
                expected_columns={
                    "date",
                    "grid_id",
                    "monthly_dataset1__d1v1",
                    "monthly_dataset1__d1v2",
                    "monthly_dataset2__d2v1",
                    "static_dataset__s1",
                    "static_dataset__s2",
                    "yearly_dataset__y1",
                },
                n_grid_cells=N_GRID_CELLS,
            ),
            CombinePlan(
                month=Arrow(2023, 2, 1),
                paths={
                    HivePath("country=india/dataset=static_dataset/type=static"),
                    HivePath("country=india/dataset=yearly_dataset/year=2023"),
                    HivePath("country=india/dataset=monthly_dataset1/month=2023-02"),
                    HivePath("country=india/dataset=monthly_dataset2/month=2023-02"),
                },
                expected_columns={
                    "date",
                    "grid_id",
                    "monthly_dataset1__d1v1",
                    "monthly_dataset1__d1v2",
                    "monthly_dataset2__d2v1",
                    "static_dataset__s1",
                    "static_dataset__s2",
                    "yearly_dataset__y1",
                },
                n_grid_cells=N_GRID_CELLS,
            ),
        ]
    )


def test__plan__empty_results__returns_empty_plans():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config, n_grid_cells=N_GRID_CELLS)

    results: Collection[UploadResult] = []

    plans = list(planner.plan(results))

    assert_that(plans).contains_only(
        *[
            CombinePlan(
                month=Arrow(2023, 1, 1),
                paths=set(),
                expected_columns=set(),
                n_grid_cells=N_GRID_CELLS,
            ),
            CombinePlan(
                month=Arrow(2023, 2, 1),
                paths=set(),
                expected_columns=set(),
                n_grid_cells=N_GRID_CELLS,
            ),
        ]
    )


def test__plan__missing_yearly_dataset__returns_last_previously_available():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config, n_grid_cells=N_GRID_CELLS)

    results: Collection[UploadResult] = [
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=static_dataset/type=static",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("s1", "s2"),
                expected_rows=N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2021",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("y1"),
                expected_rows=N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2022",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("y1"),
                expected_rows=N_GRID_CELLS,
                consumer_behaviour=PipelineConsumerBehaviour(
                    missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
                ),
            ),
            completeness=DataCompleteness.EMPTY,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2023",
                id_columns={"grid_id"},
                value_column_type_map=_value_column_type_map("y1"),
                expected_rows=N_GRID_CELLS,
                consumer_behaviour=PipelineConsumerBehaviour(
                    missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
                ),
            ),
            completeness=DataCompleteness.EMPTY,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2022-12",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d1v1", "d1v2"),
                expected_rows=31 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d1v1", "d1v2"),
                expected_rows=31 * N_GRID_CELLS,
                consumer_behaviour=PipelineConsumerBehaviour(
                    missing_data_heuristic=MissingDataHeuristic.COPY_LATEST_AVAILABLE_BEFORE
                ),
            ),
            completeness=DataCompleteness.EMPTY,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d2v1"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d1v1", "d1v2"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map=_value_column_type_map("d2v1"),
                expected_rows=28 * N_GRID_CELLS,
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
    ]

    plans = list(planner.plan(results))

    assert_that(plans).contains_only(
        *[
            CombinePlan(
                month=Arrow(2023, 1, 1),
                paths={
                    HivePath("country=india/dataset=static_dataset/type=static"),
                    HivePath("country=india/dataset=yearly_dataset/year=2021"),
                    HivePath("country=india/dataset=monthly_dataset1/month=2022-12"),
                    HivePath("country=india/dataset=monthly_dataset2/month=2023-01"),
                },
                expected_columns={
                    "date",
                    "grid_id",
                    "monthly_dataset1__d1v1",
                    "monthly_dataset1__d1v2",
                    "monthly_dataset2__d2v1",
                    "static_dataset__s1",
                    "static_dataset__s2",
                    "yearly_dataset__y1",
                },
                n_grid_cells=N_GRID_CELLS,
            ),
            CombinePlan(
                month=Arrow(2023, 2, 1),
                paths={
                    HivePath("country=india/dataset=static_dataset/type=static"),
                    HivePath("country=india/dataset=yearly_dataset/year=2021"),
                    HivePath("country=india/dataset=monthly_dataset1/month=2023-02"),
                    HivePath("country=india/dataset=monthly_dataset2/month=2023-02"),
                },
                expected_columns={
                    "date",
                    "grid_id",
                    "monthly_dataset1__d1v1",
                    "monthly_dataset1__d1v2",
                    "monthly_dataset2__d2v1",
                    "static_dataset__s1",
                    "static_dataset__s2",
                    "yearly_dataset__y1",
                },
                n_grid_cells=N_GRID_CELLS,
            ),
        ]
    )
