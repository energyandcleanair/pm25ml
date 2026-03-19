"""Storage for ML models."""

import json
import tempfile
from dataclasses import dataclass
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import daal4py
import lightgbm
import pandas as pd
from arrow import Arrow
from daal4py.mb import GBTDAALModel
from fsspec import AbstractFileSystem
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from pm25ml.logging import logger
from pm25ml.training.types import ModelName, Pm25mlCompatibleModel

if TYPE_CHECKING:
    from io import BufferedReader, BufferedWriter


@dataclass
class ModelStats:
    """Statistics for a model run."""

    cv_results: pd.DataFrame
    test_metrics: dict


@dataclass
class ValidatedModel(ModelStats):
    """A validated model with its metadata."""

    model: Pm25mlCompatibleModel

    @property
    def type(self) -> str:
        """Return the type of the model."""
        return self.model.__class__.__name__


@dataclass
class LoadedValidatedModel(ModelStats):
    """
    A validated model loaded from storage.

    This abstracts the type of model away when it's loaded as not all models can be loaded as the
    whole input. For example, the LGBMRegressor can't be loaded fully.
    """

    model: GBTDAALModel


type ModelRunRef = Arrow

type ModelType = Literal["XGBRegressor", "LGBMRegressor"]


class ModelStorage:
    """
    Storage for ML models.

    The models are stored temporarily on the local filesystem during writing and loading
    as they do not support direct streaming from cloud storage.
    """

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        bucket_name: str,
        profile_id: str,
    ) -> None:
        """
        Initialize the model storage.

        Args:
            filesystem (AbstractFileSystem): The filesystem to use for storage.
            bucket_name (str): The name of the bucket where models will be stored.
            profile_id (str): The profile/country identifier for partitioned storage.

        """
        self.filesystem = filesystem
        self.bucket_name = bucket_name
        self.profile_id = profile_id

    def save_model(
        self,
        model_name: ModelName,
        model_run_ref: ModelRunRef,
        model: ValidatedModel,
    ) -> None:
        """
        Save the validated model to the storage.

        Args:
            model_name (str): The name of the model to save.
            model_run_ref (Arrow): A reference for the model run.
            model (ValidatedModel): The validated model to save.

        """
        base_path = Path(
            self._models_root(),
            model_name,
            model_run_ref.format("YYYY-MM-DD+HH-mm-ss"),
        )

        self.filesystem.mkdir(str(base_path), create_parents=True, exist_ok=True)

        model_path = str(base_path / f"model+{model.type}.gz")
        with (
            cast("BufferedWriter", self.filesystem.open(model_path, "wb")) as f,
            GzipFile(fileobj=f, mode="wb") as gz_f,
        ):
            gz_f.write(self._serialised_to_bytes(model.model))

        cv_results_path = str(base_path / "cv_results.parquet")
        with self.filesystem.open(cv_results_path, "wb") as f:
            model.cv_results.to_csv(f, index=False)

        test_metrics_path = str(base_path / "test_metrics.json")
        with self.filesystem.open(test_metrics_path, "w") as f:
            json.dump(model.test_metrics, f)

    def _serialised_to_bytes(
        self,
        model: Pm25mlCompatibleModel,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            """Save the model to a temporary directory."""
            if isinstance(model, XGBRegressor):
                model_path = Path(tmp_dir, "model.json")
                model.save_model(model_path)
                with model_path.open("rb") as f:
                    return f.read()
            elif isinstance(model, LGBMRegressor):
                model_path = Path(tmp_dir, "model.txt")
                model.booster_.save_model(model_path)
                with model_path.open("rb") as f:
                    return f.read()
            else:
                msg = "Unsupported model type for saving."
                raise TypeError(msg)

    def load_model(
        self,
        model_name: ModelName,
        model_run_ref: ModelRunRef,
    ) -> LoadedValidatedModel:
        """
        Load a validated model from the storage.

        Args:
            model_name (str): The name of the model to load.
            model_run_ref (ModelRef): A reference for the model run, typically a timestamp.

        Returns:
            ValidatedModel: The loaded validated model.

        """
        return self._load_from_str_ref(
            model_name,
            model_run_ref.format("YYYY-MM-DD+HH-mm-ss"),
        )

    def load_latest_model(self, model_name: ModelName) -> LoadedValidatedModel:
        """
        Load the latest validated model for a given model name.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            LoadedValidatedModel: The loaded validated model.

        """
        base_path = Path(self._models_root(), model_name)

        # Find the latest model run reference
        model_run_refs: list[str] = [
            Path(path).name
            for path in cast("list[str]", self.filesystem.glob(str(base_path / "*")))
            if self.filesystem.isdir(path)
        ]
        if not model_run_refs:
            msg = f"No model runs found for model: {model_name}"
            raise FileNotFoundError(msg)

        latest_run_ref: str = max(model_run_refs)
        logger.debug(f"Latest model run reference for {model_name}: {latest_run_ref}")
        # Delegate to the existing load_model method
        return self._load_from_str_ref(model_name, latest_run_ref)

    def _load_from_str_ref(
        self,
        model_name: ModelName,
        model_run_ref: str,
    ) -> LoadedValidatedModel:
        base_path = Path(
            self._models_root(),
            model_name,
            model_run_ref,
        )

        model_type_path = self._find_model_path(base_path)
        model_type = self._identify_model_type(model_type_path)
        predictor = self._load_model_from_remote_path(model_type_path, model_type)

        cv_results = self._load_cv_metrics(base_path)

        test_metrics = self._load_test_metrics(base_path)

        return LoadedValidatedModel(
            model=predictor,
            cv_results=cv_results,
            test_metrics=test_metrics,
        )

    def _find_model_path(self, base_path: Path) -> str:
        model_type_path = next(
            iter(self.filesystem.glob(str(base_path / "model+*"))),
            None,
        )
        if not model_type_path:
            model_type_not_found_msg = "Model type file not found."
            raise FileNotFoundError(model_type_not_found_msg)
        return model_type_path

    def _identify_model_type(self, model_type_path: str) -> ModelType:
        model_type = model_type_path.split("+")[-1].replace(".gz", "")
        if model_type not in {"XGBRegressor", "LGBMRegressor"}:
            msg = f"Unsupported model type: {model_type}"
            raise ValueError(msg)

        return cast("ModelType", model_type)

    def _load_model_from_remote_path(
        self,
        model_type_path: str,
        model_type: ModelType,
    ) -> GBTDAALModel:
        # When we run this in the cloud, anything written to disk is stored in RAM
        # and so we want to make sure that we remove the local copy of the model
        # after loading it.
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.debug(f"Loading bytes for model type: {model_type_path}")
            model_bytes = self._read_bytes_from_model_path(model_type_path)

            logger.debug(
                f"Writing bytes to temporary dir for model type: {model_type_path}",
            )
            tmp_model_path = Path(
                tmp_dir,
                "model.json" if model_type == "XGBRegressor" else "model.txt",
            )

            with tmp_model_path.open("wb") as tmp_file:
                tmp_file.write(model_bytes)

            # The model's bytes can be pretty significant in size so we want to
            # make sure that we don't keep it in memory before loading it from
            # the temporary file.
            del model_bytes

            return self._load_model_from_local_file(model_type, tmp_model_path)

    def _load_model_from_local_file(
        self,
        model_type: ModelType,
        model_path: Path,
    ) -> GBTDAALModel:
        logger.debug(f"Loading model from temporary path: {model_path}")
        predictor: GBTDAALModel
        match model_type:
            case "XGBRegressor":
                xgbmodel = XGBRegressor()
                xgbmodel.load_model(model_path)
                predictor = cast("GBTDAALModel", daal4py.mb.convert_model(xgbmodel))
            case "LGBMRegressor":
                # We can cast this as, when a booster is used with a data frame, it will return
                # an ndarray - and so will behave like a Predictor.
                lightgbm_booster = lightgbm.Booster(model_file=str(model_path))
                predictor = cast(
                    "GBTDAALModel",
                    daal4py.mb.convert_model(lightgbm_booster),
                )
            case _:
                msg = f"Unsupported model type: {model_type}"
                raise ValueError(msg)
        return predictor

    def _read_bytes_from_model_path(self, model_type_path: str) -> bytes:
        with (
            cast("BufferedReader", self.filesystem.open(model_type_path, "rb")) as f,
            GzipFile(fileobj=f, mode="rb") as gz_f,
        ):
            return cast("bytes", gz_f.read())

    def _models_root(self) -> Path:
        return Path(self.bucket_name, f"country={self.profile_id}")

    def _load_cv_metrics(self, base_path: Path) -> pd.DataFrame:
        cv_results_path = str(base_path / "cv_results.parquet")
        logger.debug(f"Loading CV results from: {cv_results_path}")
        with self.filesystem.open(cv_results_path, "rb") as f:
            return pd.read_csv(f)

    def _load_test_metrics(self, base_path: Path) -> dict:
        test_metrics_path = str(base_path / "test_metrics.json")
        logger.debug(f"Loading test metrics from: {test_metrics_path}")
        with self.filesystem.open(test_metrics_path, "r") as f:
            return json.load(f)
