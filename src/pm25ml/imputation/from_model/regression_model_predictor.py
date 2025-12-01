"""Imputers using pre-trained regression models from previous stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pm25ml.logging import logger

if TYPE_CHECKING:
    from numpy import ndarray

    from pm25ml.combiners.combined_storage import CombinedStorage
    from pm25ml.combiners.data_artifact import DataArtifactRef
    from pm25ml.model_reference import ModelReference
    from pm25ml.setup.temporal_config import TemporalConfig
    from pm25ml.training.model_storage import LoadedValidatedModel


class RegressionModelPredictor:
    """Imputes missing data using a regression model."""

    def __init__(  # noqa: PLR0913
        self,
        model_ref: ModelReference,
        model: LoadedValidatedModel,
        temporal_config: TemporalConfig,
        combined_storage: CombinedStorage,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Initialize the RegressionModelImputer."""
        self.model_ref = model_ref
        self.model = model
        self.temporal_config = temporal_config
        self.combined_storage = combined_storage
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def predict(self, *, include_stats: bool = True) -> None:
        """Impute missing data and imputation relevant stats to the DataFrame."""
        trainer_data_def = self.model_ref

        average_cv_score = self.model.cv_results["test_r2"].mean()

        self._check_model_quality(
            average_cv_score=average_cv_score,
        )

        predictor = self.model.model

        # This keeps the previous month's results to calculate rolling mean.
        previous_result: pl.DataFrame | None = None
        for month in self.temporal_config.months:
            logger.debug(f"Processing month: {month}")

            month_id = month.format("YYYY-MM")

            logger.debug(f"Loading data for month: {month_id}")
            data_for_month = self.combined_storage.read_dataframe(
                self.input_data_artifact.for_month(month_id),
            )

            logger.debug(f"Starting prediction for month: {month_id}.")
            predicted_data = predictor.predict(
                data_for_month.select(
                    trainer_data_def.predictor_cols,
                ).to_pandas(),
            )

            logger.debug(f"Adding imputation details for month: {month_id}.")
            result_df = self._add_imputed_details_to_df(
                average_cv_score,
                data_for_month,
                predicted_data,
                previous_result,
                include_stats=include_stats,
            )

            previous_result = result_df

            if len(data_for_month) != len(result_df):
                logger.warning(
                    f"Data length mismatch for month {month_id}: "
                    f"{len(data_for_month)} != {len(result_df)}",
                )

            logger.debug(f"Writing results for month: {month_id}.")
            self.combined_storage.write_to_destination(
                result_df.select(
                    pl.col("grid_id"),
                    pl.col("date"),
                    pl.col(f"^{trainer_data_def.target_col}__.*$"),
                ).sort(
                    [
                        "date",
                        "grid_id",
                    ],
                ),
                result_subpath=self.output_data_artifact.for_month(month_id),
            )

    def _check_model_quality(self, *, average_cv_score: float) -> None:
        model_name = self.model_ref.model_name
        min_score = self.model_ref.min_r2_score
        max_score = self.model_ref.max_r2_score
        if average_cv_score < min_score:
            msg = (
                f"Average CV R2 score for the {model_name} "
                f"model is too low: {average_cv_score:.2f}."
            )
            raise ValueError(msg)

        if average_cv_score > max_score:
            msg = (
                f"Average CV R2 score for the {model_name} "
                f"model is unusually high: {average_cv_score:.2f}."
            )
            raise ValueError(msg)

        logger.debug(
            f"CV R2 score for the {model_name} model is within expected range: "
            f"{average_cv_score:.2f}.",
            extra={
                "min_cv_r2_score": min_score,
                "max_cv_r2_score": max_score,
                "average_cv_score": average_cv_score,
            },
        )

    def _add_imputed_details_to_df(
        self,
        average_cv_score: float,
        data_for_month: pl.DataFrame,
        predicted_data: ndarray,
        previous_result: pl.DataFrame | None,
        *,
        include_stats: bool,
    ) -> pl.DataFrame:
        target_col_name = self.model_ref.target_col

        predicted_col_name = f"{target_col_name}__predicted"

        with_predicted_results = data_for_month.with_columns(
            pl.Series(name=predicted_col_name, values=predicted_data),
        )
        if not include_stats:
            return with_predicted_results

        imputed_flag_col_name = f"{target_col_name}__imputed_flag"
        imputed_col_name = f"{target_col_name}__imputed"
        rolling_imputed_col_name = f"{target_col_name}__imputed_r7d"
        score_col_name = f"{target_col_name}__score"
        share_imputed_across_all_grids_col_name = (
            f"{target_col_name}__share_imputed_across_all_grids"
        )

        target_col = pl.col(target_col_name)

        with_predicted_results = with_predicted_results.with_columns(
            # This allows us to use "is_imputed" in the next step
            **{
                imputed_flag_col_name: pl.when(target_col.is_null())
                .then(pl.lit(1))
                .otherwise(pl.lit(0)),
            },
        )

        is_imputed = pl.col(imputed_flag_col_name) == 1

        predicted_col = pl.col(predicted_col_name)
        imputed_col = pl.col(imputed_col_name)

        with_aggregates = with_predicted_results.with_columns(
            **{
                imputed_col_name: pl.when(is_imputed)
                .then(predicted_col)
                .otherwise(target_col),
                score_col_name: pl.when(is_imputed)
                .then(predicted_col * average_cv_score)
                .otherwise(target_col),
                share_imputed_across_all_grids_col_name: is_imputed.mean().over(
                    "date",
                ),
            },
        )

        both_df = (
            pl.concat(
                [
                    with_aggregates.with_columns(
                        _which_df=pl.lit("current"),
                    ),
                    previous_result.drop(
                        rolling_imputed_col_name,
                    ).with_columns(
                        _which_df=pl.lit("previous"),
                    ),
                ],
            )
            if previous_result is not None
            else with_aggregates.with_columns(
                _which_df=pl.lit("current"),
            )
        )

        return (
            both_df.sort(
                [
                    "grid_id",
                    "date",
                ],
            )
            .with_columns(
                **{
                    rolling_imputed_col_name: imputed_col.rolling_mean(
                        window_size=7,
                        min_samples=1,
                    ).over(
                        "grid_id",
                    ),
                },
            )
            .filter(
                pl.col("_which_df") == "current",
            )
            .drop(
                "_which_df",
            )
        )
