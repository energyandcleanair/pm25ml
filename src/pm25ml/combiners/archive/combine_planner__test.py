from arrow import Arrow
from assertpy import assert_that
from pm25ml.collectors.export_pipeline import (
    MissingDataHeuristic,
    PipelineConsumerBehaviour,
)
from pm25ml.collectors.validate_configuration import VALID_COUNTRIES
from pm25ml.combiners.archive.combine_planner import CombinePlan, CombinePlanner
from pm25ml.collectors.collector import DataCompleteness, UploadResult, PipelineConfig
from pm25ml.hive_path import HivePath
from collections.abc import Collection
from dataclasses import dataclass

from pm25ml.setup.temporal_config import TemporalConfig


def test__CombinePlan__month_id():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
    )

    assert desc.month_id == "2023-01", "Month ID should be formatted as 'YYYY-MM'"


def test__CombinePlan__expected_rows():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
    )

    assert (
        desc.expected_rows == VALID_COUNTRIES["india"] * 31
    ), "Expected rows should be equal to the number of days in the month"


def test__CombinePlan__days_in_month():
    desc = CombinePlan(
        month=Arrow(2023, 1, 1),
        paths=set(),
        expected_columns={"col1", "col2"},
    )

    assert desc.days_in_month == 31, "Days in month should be 31 for January"


def test__plan__valid_results__returns_combine_plans():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config)

    results: Collection[UploadResult] = [
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=static_dataset/type=static",
                id_columns={"grid_id"},
                value_column_type_map={"s1", "s2"},
                expected_rows=VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2023",
                id_columns={"grid_id"},
                value_column_type_map={"y1"},
                expected_rows=VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d1v1", "d1v2"},
                expected_rows=31 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d2v1"},
                expected_rows=28 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d1v1", "d1v2"},
                expected_rows=28 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d2v1"},
                expected_rows=28 * VALID_COUNTRIES["india"],
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
            ),
        ]
    )


def test__plan__empty_results__returns_empty_plans():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config)

    results: Collection[UploadResult] = []

    plans = list(planner.plan(results))

    assert_that(plans).contains_only(
        *[
            CombinePlan(
                month=Arrow(2023, 1, 1),
                paths=set(),
                expected_columns=set(),
            ),
            CombinePlan(
                month=Arrow(2023, 2, 1),
                paths=set(),
                expected_columns=set(),
            ),
        ]
    )


def test__plan__missing_yearly_dataset__returns_last_previously_available():
    temporal_config = TemporalConfig(
        start_date=Arrow(2023, 1, 1),
        end_date=Arrow(2023, 2, 1),
    )
    planner = CombinePlanner(temporal_config)

    results: Collection[UploadResult] = [
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=static_dataset/type=static",
                id_columns={"grid_id"},
                value_column_type_map={"s1", "s2"},
                expected_rows=VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2021",
                id_columns={"grid_id"},
                value_column_type_map={"y1"},
                expected_rows=VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=yearly_dataset/year=2022",
                id_columns={"grid_id"},
                value_column_type_map={"y1"},
                expected_rows=VALID_COUNTRIES["india"],
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
                value_column_type_map={"y1"},
                expected_rows=VALID_COUNTRIES["india"],
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
                value_column_type_map={"d1v1", "d1v2"},
                expected_rows=31 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-01",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d1v1", "d1v2"},
                expected_rows=31 * VALID_COUNTRIES["india"],
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
                value_column_type_map={"d2v1"},
                expected_rows=28 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset1/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d1v1", "d1v2"},
                expected_rows=28 * VALID_COUNTRIES["india"],
            ),
            completeness=DataCompleteness.COMPLETE,
        ),
        UploadResult(
            pipeline_config=PipelineConfig(
                result_subpath="country=india/dataset=monthly_dataset2/month=2023-02",
                id_columns={"date", "grid_id"},
                value_column_type_map={"d2v1"},
                expected_rows=28 * VALID_COUNTRIES["india"],
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
            ),
        ]
    )
