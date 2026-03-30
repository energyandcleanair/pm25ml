"""Controller for imputation using regression models."""

import gc

from pm25ml.combiners.combined_storage import CombinedStorage
from pm25ml.combiners.data_artifact import DataArtifactRef
from pm25ml.combiners.recombiner.recombiner import Recombiner
from pm25ml.imputation.from_model.regression_model_predictor import (
    RegressionModelPredictor,
)
from pm25ml.logging import logger
from pm25ml.model_reference import ImputationModelReference
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.training.model_storage import ModelStorage
from pm25ml.training.types import ModelName


class ImputationController:
    """
    Imputes for each of the project's imputation models.

    Uses the latest available regression model for each and combines to a .
    """

    def __init__(  # noqa: PLR0913
        self,
        model_store: ModelStorage,
        model_run_ref: str,
        temporal_config: TemporalConfig,
        combined_storage: CombinedStorage,
        model_refs: dict[ModelName, ImputationModelReference],
        recombiner: Recombiner,
        input_data_artifact: DataArtifactRef,
        output_data_artifact: DataArtifactRef,
    ) -> None:
        """Build a RegressionModelImputer instance."""
        self.model_store = model_store
        self.model_run_ref = model_run_ref
        self.temporal_config = temporal_config
        self.combined_storage = combined_storage
        self.model_refs = model_refs
        self.recombiner = recombiner
        self.input_data_artifact = input_data_artifact
        self.output_data_artifact = output_data_artifact

    def impute(self) -> None:
        """
        Impute the data using regression models for the configured run reference.

        Do this for the time period specified by the temporal config.
        """
        to_add_stage_names = self._impute_for_all()

        logger.info("Combining imputed data with generated features")
        self.recombiner.recombine(
            stages=[
                self.input_data_artifact,
                *to_add_stage_names,
            ],
        )

    def _impute_for_all(self) -> list[DataArtifactRef]:
        to_add_stage_names: list[DataArtifactRef] = []

        for model_name, model_ref in self.model_refs.items():
            sub_artifact = self._impute_for_model(model_name, model_ref)
            to_add_stage_names.append(sub_artifact)
            # Do an explicit garbage collect to ensure memory is freed
            gc.collect()
        return to_add_stage_names

    def _impute_for_model(
        self,
        model_name: ModelName,
        model_ref: ImputationModelReference,
    ) -> DataArtifactRef:
        logger.info(f"Imputing for model: {model_name}")

        logger.debug(f"Loading model reference: {model_ref}")
        latest_model = self.model_store.load_model(model_name, self.model_run_ref)
        gc.collect()

        logger.debug(f"Selecting data for model: {model_ref}")
        sub_artifact = self.output_data_artifact.for_sub_artifact(model_name)

        regression_model_imputer = RegressionModelPredictor(
            model_ref=model_ref,
            model=latest_model,
            temporal_config=self.temporal_config,
            combined_storage=self.combined_storage,
            input_data_artifact=self.input_data_artifact,
            output_data_artifact=sub_artifact,
        )

        logger.debug(f"Imputing for model: {model_name}")
        regression_model_imputer.predict()
        return sub_artifact
