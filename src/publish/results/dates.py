"""Format the latest month present in a published NetCDF file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from xarray import Dataset


def latest_dataset_end_label(dataset: Dataset) -> str:
    """
    Return the latest ``time`` month as ``May 2026``.

    Args:
        dataset: Published NetCDF dataset with a ``time`` coordinate.

    Returns:
        English month name and four-digit year of the latest timestamp.

    Raises:
        ValueError: If the dataset has no ``time`` coordinate or it is empty.

    """
    if "time" not in dataset.coords and "time" not in dataset.variables:
        msg = "NetCDF dataset has no time coordinate."
        raise ValueError(msg)

    times = pd.to_datetime(dataset["time"].values)
    if len(times) == 0:
        msg = "NetCDF dataset time coordinate is empty."
        raise ValueError(msg)

    latest = times.max()
    return f"{latest.month_name()} {latest.year}"
