"""Provides type checked definitions for the temporal aspects of the PM2.5 ML pipeline."""

from arrow import Arrow
from attr import dataclass


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for the temporal aspects of the PM2.5 ML pipeline."""

    start_date: Arrow
    """
    The start date of the pipeline, inclusive of whole month.
    """
    end_date: Arrow
    """
    The end date of the pipeline, inclusive of whole month.
    """

    def __attrs_post_init__(self) -> None:
        """Validate that the configured month range is ordered."""
        if self.start_date.floor("month") > self.end_date.floor("month"):
            msg = "START_MONTH must not be later than END_MONTH"
            raise ValueError(msg)

    @property
    def end_date_exclusive(self) -> Arrow:
        """Returns the end date of the pipeline, exclusive of whole month."""
        return self.end_date.shift(months=1)

    @property
    def years(self) -> list[int]:
        """Returns a list of years covered by the pipeline."""
        return list(range(self.start_date.year, self.end_date.year + 1))

    @property
    def months(self) -> list[Arrow]:
        """Returns a list of months covered by the pipeline."""
        return list(Arrow.range("month", start=self.start_date, end=self.end_date))

    @property
    def month_ids(self) -> list[str]:
        """Returns a list of month IDs in 'YYYY-MM' format."""
        return [month.format("YYYY-MM") for month in self.months]
