"""Stores results."""

from typing import BinaryIO

from fsspec import AbstractFileSystem


class FinalResultStorage:
    """Final result storage for various output formats."""

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        destination_bucket: str,
        output_path: str,
    ) -> None:
        """Initialize the FinalResultStorage."""
        self.filesystem = filesystem
        self.destination_bucket = destination_bucket
        self.output_path = output_path

    def write(self, data: BinaryIO, file_name: str) -> None:
        """
        Write the data to the destination bucket.

        :param data: The data to write.
        """
        dir_path = f"{self.destination_bucket}/{self.output_path}"
        self.filesystem.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/{file_name}"
        with self.filesystem.open(file_path, "wb") as file:
            file.write(data.read())  # pyright: ignore[reportArgumentType]
            file.flush()
