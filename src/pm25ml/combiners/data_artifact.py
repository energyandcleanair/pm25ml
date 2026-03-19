"""Data artifact for combined storage."""

from __future__ import annotations

from attr import dataclass

from pm25ml.hive_path import HivePath


@dataclass
class DataArtifactRef:
    """Data artifact for combined storage."""

    stage: str
    country: str

    @property
    def initial_path(self) -> HivePath:
        """
        Get the initial HivePath for the artifact.

        :return: The initial HivePath for the artifact.
        """
        return HivePath.from_args(country=self.country, stage=self.stage)

    def for_sub_artifact(self, subartifact_name: str) -> DataArtifactRef:
        """
        Return a new DataArtifact for a sub-artifact.

        :param subartifact_name: The name of the sub-artifact.

        :return: A new DataArtifact with the sub-artifact name appended to the stage.

        """
        return DataArtifactRef(stage=f"{self.stage}+{subartifact_name}", country=self.country)

    def for_month(self, month: str) -> HivePath:
        """
        Get the HivePath for the artifact for a specific month.

        :param month: The month to get the HivePath for.
        :return: The HivePath for the artifact for the specified month.
        """
        return self.initial_path.with_args(month=month)
