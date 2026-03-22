import contextlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import arrow
import ee
import ee.data
from google.cloud import storage

import nanoid
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pm25ml.collectors.gee.feature_planner import (
    FeaturePlan,
    GriddedFeatureCollectionPlanner,
)

pytestmark = pytest.mark.integration

# ### NOTE ####
# See pm25ml/collectors/gee/feature_planner__it_assets/README.md for details on
# the structure of the assets used in these tests.

TEST_UNIQUE_ID = nanoid.generate()

BUCKET_NAME = os.environ.get("IT_GEE_ASSET_BUCKET_NAME")
GEE_IT_ASSET_ROOT = os.environ.get("IT_GEE_ASSET_ROOT")

GEE_IT_TEST_INSTANCE_ASSETS_ROOT = f"{GEE_IT_ASSET_ROOT}/{TEST_UNIQUE_ID}"

GEE_IMAGE_COLLECTION_ROOT = f"{GEE_IT_TEST_INSTANCE_ASSETS_ROOT}/dummy_data"
GEE_ORBIT_IMAGE_COLLECTION_ROOT = f"{GEE_IT_TEST_INSTANCE_ASSETS_ROOT}/dummy_orbit_data"
GEE_GRID_LOCATION = f"{GEE_IT_TEST_INSTANCE_ASSETS_ROOT}/grid"

ASSET_DIR = Path(__file__).parent / "feature_planner__it_assets"


@pytest.fixture(scope="module")
def check_env():
    if not BUCKET_NAME:
        raise ValueError(
            "Environment variable IT_GEE_ASSET_BUCKET_NAME must be set for integration tests.",
        )
    if not GEE_IT_ASSET_ROOT:
        raise ValueError(
            "Environment variable IT_GEE_ASSET_ROOT must be set for integration tests.",
        )


@pytest.fixture(scope="module")
def initialize_gee(check_env):
    if os.environ.get("GITHUB_ACTIONS"):
        secret_contents = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
        service_account = os.environ.get("GOOGLE_SERVICE_ACCOUNT_NAME")
        project_id = os.environ.get("GCP_PROJECT")
        if not secret_contents:
            raise ValueError(
                "Environment variable GEE_SERVICE_ACCOUNT_KEY must be set for integration tests.",
            )
        if not service_account:
            raise ValueError(
                "Environment variable GOOGLE_SERVICE_ACCOUNT_NAME must be set for integration tests.",
            )
        if not project_id:
            raise ValueError(
                "Environment variable GCP_PROJECT must be set for integration tests.",
            )
        creds = ee.ServiceAccountCredentials(service_account, key_data=secret_contents)  # type: ignore
        ee.Initialize(
            credentials=creds,
            project=project_id,
        )
    else:
        ee.Initialize()

    check_ee_initialised()


def check_ee_initialised() -> None:
    ee.data.getAssetRoots()


@pytest.fixture(scope="module")
def gcs_bucket():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return bucket


@pytest.fixture(scope="module")
def cleanup(initialize_gee, gcs_bucket):
    def delete_gee_assets_recursively(asset_id):
        """Recursively delete all assets under the given asset_id."""
        assets = ee.data.listAssets({"parent": asset_id})
        for asset in assets["assets"]:
            if asset["type"] == "FOLDER" or asset["type"] == "IMAGE_COLLECTION":
                delete_gee_assets_recursively(asset["id"])
            else:
                ee.data.deleteAsset(asset["id"])

        ee.data.deleteAsset(asset_id)

    def clear_test_gcs_assets():
        # Empty the bucket before uploading new files
        blobs = gcs_bucket.list_blobs(prefix=TEST_UNIQUE_ID)
        for blob in blobs:
            blob.delete()

    clear_test_gcs_assets()
    with contextlib.suppress(Exception):
        delete_gee_assets_recursively(GEE_IT_TEST_INSTANCE_ASSETS_ROOT)
    yield
    clear_test_gcs_assets()
    with contextlib.suppress(Exception):
        delete_gee_assets_recursively(GEE_IT_TEST_INSTANCE_ASSETS_ROOT)


@pytest.fixture(scope="module", autouse=True)
def upload_dummy_tiffs(cleanup, gcs_bucket) -> dict[str, str]:
    # We define these files manually up front as it's easier to manage than to
    # read from the directories and pull metadata out.
    assets_to_upload = [
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2022-12-30.tif"),
            "date": datetime(2022, 12, 30, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_data/dummy_image_2022-12-30.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_data/dummy_image_2022-12-30.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2022-12-30",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2022-12-31.tif"),
            "date": datetime(2022, 12, 31, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_data/dummy_image_2022-12-31.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_data/dummy_image_2022-12-31.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2022-12-31",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-01.tif"),
            "date": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_data/dummy_image_2023-01-01.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_data/dummy_image_2023-01-01.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2023-01-01",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-02.tif"),
            "date": datetime(2023, 1, 2, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_data/dummy_image_2023-01-02.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_data/dummy_image_2023-01-02.tif",
            "asset_path": f"{GEE_IMAGE_COLLECTION_ROOT}/2023-01-02",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-01.tif"),
            "date": datetime(2023, 1, 1, 7, 44, 9, tzinfo=timezone.utc),
            "end_date": datetime(2023, 1, 1, 9, 25, 39, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-01_a.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-01_a.tif",
            "asset_path": f"{GEE_ORBIT_IMAGE_COLLECTION_ROOT}/2023-01-01-a",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-01.tif"),
            "date": datetime(2023, 1, 1, 9, 25, 39, tzinfo=timezone.utc),
            "end_date": datetime(2023, 1, 1, 11, 0, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-01_b.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-01_b.tif",
            "asset_path": f"{GEE_ORBIT_IMAGE_COLLECTION_ROOT}/2023-01-01-b",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-02.tif"),
            "date": datetime(2023, 1, 2, 6, 6, 43, tzinfo=timezone.utc),
            "end_date": datetime(2023, 1, 2, 7, 48, 13, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-02_a.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-02_a.tif",
            "asset_path": f"{GEE_ORBIT_IMAGE_COLLECTION_ROOT}/2023-01-02-a",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "dummy_image", "2023-01-02.tif"),
            "date": datetime(2023, 1, 2, 7, 48, 13, tzinfo=timezone.utc),
            "end_date": datetime(2023, 1, 2, 9, 30, tzinfo=timezone.utc),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-02_b.tif",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_orbit_data/dummy_image_2023-01-02_b.tif",
            "asset_path": f"{GEE_ORBIT_IMAGE_COLLECTION_ROOT}/2023-01-02-b",
            "type": "image",
        },
        {
            "local_path": Path(ASSET_DIR, "test-grid.zip"),
            "gcs_path": f"gs://{BUCKET_NAME}/{TEST_UNIQUE_ID}/dummy_data/test-grid.zip",
            "bucket_path": f"{TEST_UNIQUE_ID}/dummy_data/test-grid.zip",
            "asset_path": GEE_GRID_LOCATION,
            "type": "table",
        },
    ]

    def upload_to_gcs(
        bucket,
        assets: list,
    ):
        for upload in assets:
            blob = bucket.blob(upload["bucket_path"])
            blob.upload_from_filename(upload["local_path"])

    def upload_to_ee(
        assets_to_upload: list,
    ):
        # Then we need to create the folders and collections we need.
        with contextlib.suppress(Exception):
            ee.data.createAsset({"type": "FOLDER"}, GEE_IT_ASSET_ROOT)
        ee.data.createAsset({"type": "FOLDER"}, GEE_IT_TEST_INSTANCE_ASSETS_ROOT)
        ee.data.createAsset({"type": "IMAGE_COLLECTION"}, GEE_IMAGE_COLLECTION_ROOT)
        ee.data.createAsset({"type": "IMAGE_COLLECTION"}, GEE_ORBIT_IMAGE_COLLECTION_ROOT)

        tasks = []

        # Then we can start the ingestion tasks for each asset.
        for upload in assets_to_upload:
            if upload["type"] == "image":
                task = ee.data.startIngestion(
                    None,
                    {
                        "name": upload["asset_path"],
                        "tilesets": [
                            {
                                "sources": [
                                    {
                                        "uris": [upload["gcs_path"]],
                                    },
                                ],
                            },
                        ],
                        "startTime": upload["date"].isoformat(),
                        "endTime": upload.get(
                            "end_date",
                            upload["date"] + timedelta(days=1),
                        ).isoformat(),
                    },
                )
            else:
                task = ee.data.startTableIngestion(
                    None,  # type: ignore
                    {
                        "name": upload["asset_path"],
                        "sources": [
                            {
                                "uris": [upload["gcs_path"]],
                            },
                        ],
                    },
                )

            tasks.append(task)  # type: ignore

        # Finally, we can wait for the tasks to complete.
        incomplete_task_ids = [task["id"] for task in tasks]
        while incomplete_task_ids:
            for task in incomplete_task_ids[:]:
                status = ee.data.getTaskStatus(task)
                if status and status[0]["state"] == "COMPLETED":
                    incomplete_task_ids.remove(task)
                elif status and status[0]["state"] == "FAILED":
                    raise RuntimeError(f"Task {task} failed: {status[0]['error_message']}")
            if incomplete_task_ids:
                print(f"Waiting for {len(incomplete_task_ids)} tasks to complete...")
                sleep(5)

    upload_to_gcs(gcs_bucket, assets_to_upload)
    upload_to_ee(assets_to_upload)

    return {
        "image_collection": GEE_IMAGE_COLLECTION_ROOT,
        "orbit_image_collection": GEE_ORBIT_IMAGE_COLLECTION_ROOT,
        "grid": GEE_GRID_LOCATION,
    }


@pytest.fixture(scope="module")
def feature_planner(upload_dummy_tiffs):
    return GriddedFeatureCollectionPlanner(
        grid=ee.FeatureCollection(upload_dummy_tiffs["grid"]),  # type: ignore
    )


def test_plan_daily_average(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    daily_average_plan = feature_planner.plan_daily_average(
        collection_name=upload_dummy_tiffs["image_collection"],
        selected_bands=["b1", "b2"],
        dates=[
            arrow.get("2023-01-01"),
            arrow.get("2023-01-02"),
        ],
    )

    actual_df = execute_plan_to_dataframe(daily_average_plan)

    first_date = arrow.get("2023-01-01").date()
    second_date = arrow.get("2023-01-02").date()

    expected_df = pl.DataFrame(
        {
            "date": [
                first_date,
                first_date,
                first_date,
                first_date,
                second_date,
                second_date,
                second_date,
                second_date,
            ],
            "b1_mean": [
                14.5,
                14.5,
                6.5,
                6.5,
                15.5,
                15.5,
                7.5,
                7.5,
            ],
            "b2_mean": [
                114.5,
                114.5,
                106.5,
                106.5,
                115.5,
                115.5,
                107.5,
                107.5,
            ],
            "grid_id": [
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.2,
    )


def test_plan_daily_average__orbit_style_collection(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    daily_average_plan = feature_planner.plan_daily_average(
        collection_name=upload_dummy_tiffs["orbit_image_collection"],
        selected_bands=["b1", "b2"],
        dates=[
            arrow.get("2023-01-01"),
            arrow.get("2023-01-02"),
            arrow.get("2023-01-03"),
        ],
    )

    actual_df = execute_plan_to_dataframe(daily_average_plan)

    expected_df = pl.DataFrame(
        {
            "date": [
                arrow.get("2023-01-01").date(),
                arrow.get("2023-01-01").date(),
                arrow.get("2023-01-01").date(),
                arrow.get("2023-01-01").date(),
                arrow.get("2023-01-02").date(),
                arrow.get("2023-01-02").date(),
                arrow.get("2023-01-02").date(),
                arrow.get("2023-01-02").date(),
            ],
            "b1_mean": [
                14.5,
                14.5,
                6.5,
                6.5,
                15.5,
                15.5,
                7.5,
                7.5,
            ],
            "b2_mean": [
                114.5,
                114.5,
                106.5,
                106.5,
                115.5,
                115.5,
                107.5,
                107.5,
            ],
            "grid_id": [0, 1, 2, 3, 0, 1, 2, 3],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.2,
    )


def test_plan_static_feature(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    image_name = upload_dummy_tiffs["image_collection"] + "/2023-01-01"
    static_feature_plan = feature_planner.plan_static_feature(
        image_name=image_name,
        selected_bands=["b1", "b2"],
    )

    actual_df = execute_plan_to_dataframe(static_feature_plan)
    expected_df = pl.DataFrame(
        {
            "b1": [
                14.5,
                14.5,
                6.5,
                6.5,
            ],
            "b2": [
                114.5,
                114.5,
                106.5,
                106.5,
            ],
            "grid_id": [
                0,
                1,
                2,
                3,
            ],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.2,
    )


def test_plan_summarise_annual_classified_pixels(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    collection_name = upload_dummy_tiffs["image_collection"]
    category_band = "b11"
    year = 2023

    # "day 1" refers to 2023-01-01 and "day 2" refers to 2023-01-02

    # fmt: off
    # "day 1"'s 4x4 grid has the following categories:
    day_1_full_grid = [
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
    ]
    # fmt: on

    # fmt: off
    # "day 2"'s 4x4 grid has the following categories:
    day_2_full_grid = [
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
    ]
    # fmt: on

    # fmt: off
    x_grid_mask = [
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
    ]
    top_left_grid_mask = [
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # fmt: on

    def filter_by_mask(grid, mask):
        """Filter grid values by a mask."""
        return [val for val, m in zip(grid, mask) if m]

    output_names_to_class_values = {
        "half_day_1_categories": filter_by_mask(day_1_full_grid, x_grid_mask),
        "half_day_2_categories": filter_by_mask(day_2_full_grid, x_grid_mask),
        "half_both_day_categories": filter_by_mask(day_1_full_grid, x_grid_mask)
        + filter_by_mask(day_2_full_grid, x_grid_mask),
        "top_left_grid_categories": filter_by_mask(
            day_1_full_grid,
            top_left_grid_mask,
        )
        + filter_by_mask(
            day_2_full_grid,
            top_left_grid_mask,
        ),
        "invalid_categories": list(
            range(
                32,
            )
        ),
    }

    annual_classified_plan = feature_planner.plan_summarise_annual_classified_pixels(
        collection_name=collection_name,
        classification_band=category_band,
        output_names_to_class_values=output_names_to_class_values,
        year=year,
    )

    actual_df = execute_plan_to_dataframe(annual_classified_plan)

    expected_df = pl.DataFrame(
        {
            "half_day_1_categories_mean": [0.25, 0.25, 0.25, 0.25],
            "half_day_2_categories_mean": [0.25, 0.25, 0.25, 0.25],
            "half_both_day_categories_mean": [0.5, 0.5, 0.5, 0.5],
            # the top left grid has the ID 2
            "top_left_grid_categories_mean": [0.0, 0.0, 1.0, 0.0],
            "invalid_categories_mean": [0.0, 0.0, 0.0, 0.0],
            "grid_id": [0, 1, 2, 3],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
        rtol=0,
        atol=0.05,
    )


def execute_plan_to_dataframe(feature_plan: FeaturePlan):
    """Extract features from the planned collection into a Polars DataFrame."""

    def convert_record(record):
        # Extract the date and grid_id from the record and then the any remaining properties
        date = (
            arrow.get(record["properties"]["date"]["value"]).date()
            if "date" in record["properties"]
            else None
        )
        properties = {k: v for k, v in record["properties"].items() if k not in ["date"]}
        # Flatten the properties into a single dictionary
        result = {
            "date": date,
            **properties,
        }
        result = {k: v for k, v in result.items() if v is not None}
        return result

    plan = feature_plan.planned_collection

    features = plan.getInfo()["features"]  # type: ignore
    rows = [convert_record(f) for f in features]
    return pl.DataFrame(rows)


def test_plan_daily_average__no_bands__returns_nulls(
    feature_planner: GriddedFeatureCollectionPlanner,
    upload_dummy_tiffs,
):
    """
    Test that if there are no bands for an image, the plan does not throw an exception
    and returns nulls for that grid.
    """
    daily_average_plan = feature_planner.plan_daily_average(
        collection_name=upload_dummy_tiffs["image_collection"],
        selected_bands=["b1"],
        dates=[
            arrow.get("2023-01-02"),
            arrow.get("2023-01-03"),
        ],
    )

    actual_df = execute_plan_to_dataframe(daily_average_plan)

    second_date = arrow.get("2023-01-02").date()

    expected_df = pl.DataFrame(
        {
            "date": [
                second_date,
                second_date,
                second_date,
                second_date,
            ],
            "mean": [
                15.5,
                15.5,
                7.5,
                7.5,
            ],
            "grid_id": [
                0,
                1,
                2,
                3,
            ],
        },
    )

    assert_frame_equal(
        actual_df,
        expected_df,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        rtol=0,
        atol=0.2,
    )
