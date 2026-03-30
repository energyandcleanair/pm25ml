"""Trains regression models for pm25ml."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate

from pm25ml.logging import logger
from pm25ml.training.model_storage import ModelStorage, ValidatedModel

if TYPE_CHECKING:
    from pm25ml.combiners.combined_storage import CombinedStorage
    from pm25ml.combiners.data_artifact import DataArtifactRef
    from pm25ml.model_reference import ImputationModelReference
    from pm25ml.training.types import Pm25mlCompatibleModel


class ImputationModelPipeline:
    """
    Trains regression models for pm25ml.

    Including imputation models and the final model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        combined_storage: CombinedStorage,
        data_ref: ImputationModelReference,
        model_store: ModelStorage,
        model_run_ref: str,
        n_jobs: int,
        input_data_artifact: DataArtifactRef,
    ) -> None:
        """Initialize the ModelTrainer."""
        self.data = data_ref
        self.combined_storage = combined_storage
        self.model_store = model_store
        self.model_run_ref = model_run_ref
        self.n_jobs = n_jobs
        self.input_data_artifact = input_data_artifact

    def train_model(self) -> None:
        """Run imputation ML model."""
        # 1. Sampling
        logger.info(f"Loading and sampling training data for {self.data.model_name} imputation")
        df_sampled = self.load_training_data()
        self._check_clean(df_sampled)

        # 2. Create folds
        # outer_cv is a list where each item contains a tuple with indices
        # of training and validation sets for each fold
        logger.info("Cross validate model")
        model = self.data.model_builder()
        cv_results = self.cross_validate_with_stratification(model, df_sampled)
        self.log_cv_results(cv_results)

        logger.info("Training model on the sampled data")
        trained_model = self.train_model_on_sample(model, df_sampled)

        del df_sampled

        logger.info("Loading test data for evaluation")
        df_test = self.load_test_data()
        self._check_clean(df_test)

        # 5. Evaluate model on the test data
        logger.info("Evaluating model on the test data")
        test_metrics = self.evaluate_model(trained_model, df_test)

        logger.info(f"Test metrics: {test_metrics}")

        # 6. Save the model and diagnostics
        # Create a temporary directory to save diagnostics
        logger.info("Saving model and diagnostics")
        self.model_store.save_model(
            model_name=self.data.model_name,
            model_run_ref=self.model_run_ref,
            model=ValidatedModel(
                model=trained_model,
                cv_results=cv_results,
                test_metrics=test_metrics,
            ),
        )

    def train_model_on_sample(
        self,
        model: Pm25mlCompatibleModel,
        df_sampled: pd.DataFrame,
    ) -> Pm25mlCompatibleModel:
        """
        Train the XGBRegressor model on the sampled data.

        Args:
            model (XGBRegressor): The XGBRegressor model to be trained.
            df_sampled (pd.DataFrame): Sampled data for training.

        Returns:
            XGBRegressor: The trained XGBRegressor model.

        """
        target = df_sampled[self.data.target_col]
        predictors = df_sampled[self.data.predictor_cols]

        model.set_params(n_jobs=self.n_jobs)
        model.fit(predictors, target)

        return model

    def load_training_data(self) -> pd.DataFrame:
        """Load the sampled data imputation from GCS."""
        results = self.combined_storage.scan_stage(
            stage=self.input_data_artifact.stage,
        )

        return (
            self.data.extra_sampler(results)
            .filter(pl.col("split") == "training")
            .with_columns(
                month_of_year=pl.col("date").dt.month(),
            )
            .select(self.data.all_cols)
            .collect(engine="streaming")
            .to_pandas()
        )

    def load_test_data(self) -> pd.DataFrame:
        """Load the test data for imputation from GCS."""
        results = self.combined_storage.scan_stage(
            stage=self.input_data_artifact.stage,
        )

        return (
            self.data.extra_sampler(results)
            .filter(pl.col("split") == "test")
            .with_columns(
                month_of_year=pl.col("date").dt.month(),
            )
            .select(self.data.all_cols)
            .collect(engine="streaming")
            .to_pandas()
        )

    def cross_validate_with_stratification(
        self,
        model: Pm25mlCompatibleModel,
        df_sampled: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform cross-validation with stratification on the sampled data.

        Args:
            model (XGBRegressor): The XGBRegressor model to be trained.
            df_sampled (pd.DataFrame): Sampled data for training.

        Returns:
            XGBRegressor: The trained XGBRegressor model.

        """
        target = df_sampled[self.data.target_col]
        predictors = df_sampled[self.data.predictor_cols]
        grouper = df_sampled[self.data.grouper_col]

        n_splits = 10
        cpus_per_model = int(self.n_jobs / n_splits)

        model.set_params(n_jobs=cpus_per_model)

        selector = GroupKFold(n_splits=n_splits)

        scores = cross_validate(
            model,  # pyright: ignore[reportArgumentType]
            predictors,
            target,
            cv=selector,
            groups=grouper,
            scoring=["neg_root_mean_squared_error", "r2"],
            n_jobs=n_splits,
            return_train_score=True,
        )

        return pd.DataFrame(scores)

    def log_cv_results(self, cv_results: pd.DataFrame) -> None:
        """Log the cross-validation results."""
        cv_results = cv_results.rename(
            columns={
                "test_neg_root_mean_squared_error": "test_rmse",
                "train_neg_root_mean_squared_error": "train_rmse",
            },
        )
        logger.info(f"Cross-validation scores:\n{cv_results.to_string()}")
        cv_results_agg = cv_results.aggregate(["mean", "std", "min", "max"])
        logger.info(f"Cross-validation scores aggregated:\n{cv_results_agg.to_string()}")

    def evaluate_model(self, model: Pm25mlCompatibleModel, df_rest: pd.DataFrame) -> dict:
        """
        Evaluate the model on the rest of the data.

        This function uses the trained model to predict values
        for the rest of the data (test set not used for training) and returns
        the predictions.

        Args:
            model (XGBRegressor): The trained XGBRegressor model.
            df_rest (pd.DataFrame): Dataframe with the rest of the data for evaluation.

        """
        # Check if the model is trained
        if not hasattr(model, "feature_importances_") and not hasattr(model, "booster_"):
            msg = "Model is not trained yet."
            raise ValueError(msg)

        predicted: Any = model.predict(df_rest[self.data.predictor_cols])
        target = df_rest[self.data.target_col]

        # Check if prediction length matches the number of rows in rest_df
        if len(predicted) != df_rest.shape[0]:
            msg = "Prediction length does not match the number of rows in rest_df"
            raise TypeError(msg)

        # Calculate metrics for evaluation
        r2 = r2_score(target, predicted)
        rmse = math.sqrt(mean_squared_error(target, predicted))

        return {"r2": r2, "rmse": rmse}

    def _check_clean(self, df: pd.DataFrame) -> None:
        """Check if the DataFrame is clean."""
        must_not_be_null_cols = self.data.predictor_cols

        if df[must_not_be_null_cols].isna().any().any():
            msg = (
                f"DataFrame contains null values in columns: {must_not_be_null_cols}. "
                "Please ensure that the data is clean before training stage."
            )
            raise ValueError(msg)
