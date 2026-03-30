import io
from morefs.memory import MemFS
import pytest

from pm25ml.results.final_result_storage import FinalResultStorage


@pytest.fixture
def in_memory_filesystem() -> MemFS:
    return MemFS()


@pytest.fixture
def storage(in_memory_filesystem: MemFS) -> FinalResultStorage:
    return FinalResultStorage(
        in_memory_filesystem,
        "test_bucket",
        "country=india/run=v1",
    )


def test__write__new_nested_dir__file_written_with_same_content(
    storage: FinalResultStorage,
):
    data = b"hello world\nthis is binary"
    bio = io.BytesIO(data)

    storage.write(bio, file_name="final.bin")

    fs = storage.filesystem
    file_path = "test_bucket/country=india/run=v1/final.bin"

    assert fs.exists(file_path)
    with fs.open(file_path, "rb") as f:
        assert f.read() == data


def test__write__existing_file__content_is_overwritten(storage: FinalResultStorage):
    initial = io.BytesIO(b"first")
    updated = io.BytesIO(b"second-version")

    name = "file.dat"

    storage.write(initial, file_name=name)
    storage.write(updated, file_name=name)

    fs = storage.filesystem
    file_path = f"test_bucket/country=india/run=v1/{name}"

    with fs.open(file_path, "rb") as f:
        assert f.read() == b"second-version"
