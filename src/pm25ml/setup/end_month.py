"""Resolve and persist the temporal end month used by a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import arrow
from arrow import Arrow

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

EndMonthSource = Literal["explicit", "stored", "provisional"]


class UnresolvedEndMonthError(RuntimeError):
    """Raised when an automatic end month has not been resolved for a later stage."""


class StaleDataError(RuntimeError):
    """Raised when no sufficiently recent complete month is available."""


@dataclass(frozen=True)
class ResolvedEndMonth:
    """An end month together with the source used to resolve it."""

    month: Arrow
    source: EndMonthSource


class EndMonthStore:
    """Store the automatically selected end month for a profile and model run."""

    def __init__(
        self,
        *,
        filesystem: AbstractFileSystem,
        bucket: str,
        profile_id: str,
        model_run_ref: str,
    ) -> None:
        """Initialize storage for one profile and model run."""
        self.filesystem = filesystem
        self.path = f"{bucket}/country={profile_id}/run={model_run_ref}/resolved_end_month.txt"

    def read(self) -> Arrow | None:
        """Read the stored month, returning ``None`` when it has not been written."""
        if not self.filesystem.exists(self.path):
            return None
        with self.filesystem.open(self.path, "rt") as file:
            return arrow.get(file.read().strip(), "YYYY-MM-DD").floor("month")

    def write(self, month: Arrow) -> None:
        """Persist a month in a stable, human-readable format."""
        with self.filesystem.open(self.path, "wt") as file:
            file.write(month.floor("month").format("YYYY-MM-DD"))


def latest_completed_month(*, now: Arrow | None = None) -> Arrow:
    """Return the first day of the latest fully completed UTC calendar month."""
    current = now or arrow.utcnow()
    return current.to("UTC").floor("month").shift(months=-1)


def resolve_end_month(
    *,
    explicit_value: str | None,
    store: EndMonthStore,
    allow_provisional: bool,
    now: Arrow | None = None,
) -> ResolvedEndMonth:
    """Resolve an explicit, stored, or provisional end month in priority order."""
    stored = store.read()
    if stored is not None:
        return ResolvedEndMonth(month=stored, source="stored")

    if explicit_value and explicit_value.strip():
        return ResolvedEndMonth(
            month=arrow.get(explicit_value.strip(), "YYYY-MM-DD"),
            source="explicit",
        )

    if allow_provisional:
        return ResolvedEndMonth(month=latest_completed_month(now=now), source="provisional")

    msg = (
        "END_MONTH is not set and no automatically resolved end month exists for this "
        "PIPELINE_PROFILE and MODEL_RUN_REF. Run s005_discover first."
    )
    raise UnresolvedEndMonthError(msg)


def parse_max_data_lag_months(value: str | None) -> int:
    """Parse the automatic-selection lag, defaulting to three months."""
    lag = int(value or "3")
    if lag < 0:
        msg = "MAX_DATA_LAG_MONTHS must be a non-negative integer"
        raise ValueError(msg)
    return lag
