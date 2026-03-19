import warnings
from arrow import Arrow
import arrow
import numpy as np
import pytest
import pandas as pd
import json
from morefs.memory import MemFS
from xgboost import XGBRegressor
from lightgbm import Booster, LGBMRegressor
from pm25ml.training.model_storage import ModelStorage, ValidatedModel


TEST_COUNTRY = "india"


@pytest.fixture
def in_memory_filesystem():
    return MemFS()


@pytest.fixture(scope="module")
def sample_data():
    # Simple dataset for training
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [1, 2, 3, 4]
    return X, y


@pytest.fixture()
def model_storage(in_memory_filesystem):
    return ModelStorage(in_memory_filesystem, "test_bucket", profile_id=TEST_COUNTRY)


@pytest.fixture(scope="module")
def trained_xgb_model(sample_data):
    X, y = sample_data
    model = XGBRegressor()
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def trained_lgbm_model(sample_data):
    X, y = sample_data
    model = LGBMRegressor(min_data_in_leaf=1)
    model.fit(X, y)
    return model


EXAMPLE_DATE = arrow.get("2023-10-01+12-00-00", "YYYY-MM-DD+HH-mm-ss")
EXAMPLE_DATE_LATER = arrow.get("2023-10-02+12-00-00", "YYYY-MM-DD+HH-mm-ss")


def test__save_model__xgb_model_with_validated_model__files_exist(model_storage, trained_xgb_model):
    validated_model = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    model_storage.save_model("xgb_model", EXAMPLE_DATE, validated_model)

    expected_date_path = EXAMPLE_DATE.format("YYYY-MM-DD+HH-mm-ss")

    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/xgb_model/{expected_date_path}/model+XGBRegressor.gz"
    )
    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/xgb_model/{expected_date_path}/cv_results.parquet"
    )
    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/xgb_model/{expected_date_path}/test_metrics.json"
    )


def test__save_model__lgbm_model_with_validated_model__files_exist(
    model_storage, trained_lgbm_model
):
    validated_model = ValidatedModel(
        model=trained_lgbm_model,
        cv_results=pd.DataFrame({"metric": [0.3, 0.4]}),
        test_metrics={"mae": 0.25},
    )

    model_storage.save_model("lgbm_model", EXAMPLE_DATE, validated_model)

    expected_date_path = EXAMPLE_DATE.format("YYYY-MM-DD+HH-mm-ss")

    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/lgbm_model/{expected_date_path}/model+LGBMRegressor.gz"
    )
    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/lgbm_model/{expected_date_path}/cv_results.parquet"
    )
    assert model_storage.filesystem.exists(
        f"test_bucket/country={TEST_COUNTRY}/lgbm_model/{expected_date_path}/test_metrics.json"
    )


def test__save_model__statistics_model_with_validated_model__files_and_content_correct(
    model_storage, sample_data
):
    X, y = sample_data
    model = XGBRegressor()
    model.fit(X, y)

    validated_model = ValidatedModel(
        model=model,
        cv_results=pd.DataFrame({"metric": [0.5, 0.6]}),
        test_metrics={"r2": 0.85},
    )

    model_storage.save_model("stats_model", EXAMPLE_DATE, validated_model)

    expected_date_path = EXAMPLE_DATE.format("YYYY-MM-DD+HH-mm-ss")

    with model_storage.filesystem.open(
        f"test_bucket/country={TEST_COUNTRY}/stats_model/{expected_date_path}/cv_results.parquet"
    ) as f:
        cv_results = pd.read_csv(f)
        assert "metric" in cv_results.columns

    with model_storage.filesystem.open(
        f"test_bucket/country={TEST_COUNTRY}/stats_model/{expected_date_path}/test_metrics.json"
    ) as f:
        test_metrics = json.load(f)
        assert test_metrics["r2"] == 0.85


def test__load_model__xgb_model_saved_and_loaded__model_and_predictions_match(
    model_storage, trained_xgb_model
):
    validated_model = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    model_storage.save_model("xgb_model", EXAMPLE_DATE, validated_model)

    loaded_model_wrapper = model_storage.load_model("xgb_model", EXAMPLE_DATE)

    loaded_model = loaded_model_wrapper.model
    to_predict = np.array([[1, 2]], dtype=np.float32)
    assert loaded_model.predict(to_predict) == trained_xgb_model.predict(to_predict)


def test__load_model__lgbm_model_saved_and_loaded__model_and_predictions_match(
    model_storage, trained_lgbm_model
):
    validated_model = ValidatedModel(
        model=trained_lgbm_model,
        cv_results=pd.DataFrame({"metric": [0.3, 0.4]}),
        test_metrics={"mae": 0.25},
    )

    model_storage.save_model("lgbm_model", EXAMPLE_DATE, validated_model)

    loaded_model = model_storage.load_model("lgbm_model", EXAMPLE_DATE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        to_predict = np.array([[1, 2]], dtype=np.float64)
        assert np.allclose(
            loaded_model.model.predict(to_predict), trained_lgbm_model.predict(to_predict)
        )


def test__load_latest_model__xgb_model_with_multiple_runs__latest_model_loaded(
    model_storage, trained_xgb_model
):
    validated_model_1 = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    validated_model_2 = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.3, 0.4]}),
        test_metrics={"rmse": 0.10},
    )

    model_storage.save_model("xgb_model", EXAMPLE_DATE, validated_model_1)
    model_storage.save_model("xgb_model", EXAMPLE_DATE_LATER, validated_model_2)

    loaded_model_wrapper = model_storage.load_latest_model("xgb_model")

    assert loaded_model_wrapper.test_metrics["rmse"] == 0.10


def test__save_model__profile_storage__writes_under_country_partition(
    in_memory_filesystem, trained_xgb_model
):
    model_storage = ModelStorage(in_memory_filesystem, "test_bucket", profile_id="in+bd")
    validated_model = ValidatedModel(
        model=trained_xgb_model,
        cv_results=pd.DataFrame({"metric": [0.1, 0.2]}),
        test_metrics={"rmse": 0.15},
    )

    model_storage.save_model("xgb_model", EXAMPLE_DATE, validated_model)

    expected_date_path = EXAMPLE_DATE.format("YYYY-MM-DD+HH-mm-ss")

    assert model_storage.filesystem.exists(
        f"test_bucket/country=in+bd/xgb_model/{expected_date_path}/model+XGBRegressor.gz"
    )
