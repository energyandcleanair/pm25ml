"""Controller for imputation using regression models."""

import gc

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.imputation.from_model.regression_model_predictor import (
    RegressionModelPredictor,
)
from pm25ml.logging import logger
from pm25ml.model_reference import FullModelReference
from pm25ml.setup.temporal_config import TemporalConfig
from pm25ml.training.model_storage import ModelStorage


class FinalPredictionController:
    """
    Imputes for each of the project's imputation models.

    Uses the latest available regression model for each and combines to a .
    """

    def __init__(  # noqa: PLR0913
        self,
        model_store: ModelStorage,
        temporal_config: TemporalConfig,
        combined_storage: CombinedStorage,
        model_ref: FullModelReference,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Build a RegressionModelImputer instance."""
        self.model_store = model_store
        self.temporal_config = temporal_config
        self.combined_storage = combined_storage
        self.model_ref = model_ref
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def predict(self) -> None:
        """
        Impute the data using the latest regression model for each model.

        Do this for the time period specified by the temporal config.
        """
        self._impute_for_model()

    def _impute_for_model(
        self,
    ) -> DataArtifactRef:
        model_ref = self.model_ref
        model_name = self.model_ref.model_name
        logger.info(f"Imputing for model: {model_name}")

        logger.debug(f"Loading model reference: {model_ref}")
        latest_model = self.model_store.load_latest_model(model_name)
        gc.collect()

        regression_model_imputer = RegressionModelPredictor(
            model_ref=model_ref,
            model=latest_model,
            temporal_config=self.temporal_config,
            combined_storage=self.combined_storage,
            input_data_artifact=self.input_data_artifact,
            output_data_artifact=self.output_data_artifact,
        )

        logger.debug(f"Imputing for model: {model_name}")
        regression_model_imputer.predict(include_stats=False)
        return self.output_data_artifact
