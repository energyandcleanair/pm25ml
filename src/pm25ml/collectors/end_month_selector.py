"""Select the newest sufficiently recent month with complete required data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.end_month import EndMonthStore, StaleDataError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from arrow import Arrow

    from pm25ml.collectors.collector import RawDataCollector, UploadResult
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.setup.end_month import EndMonthSource
    from pm25ml.setup.pipelines import EndMonthCandidatePipelineFactory


@dataclass(frozen=True)
class EndMonthResolution:
    """An immutable temporal configuration with optional deferred persistence."""

    temporal_config: TemporalConfig
    store: EndMonthStore | None = None

    def persist(self) -> None:
        """Persist newly resolved dates; previously stored resolutions are no-ops."""
        if self.store is None:
            return
        self.store.write(self.temporal_config.end_date)
        logger.info(
            "Persisted END_MONTH=%s",
            self.temporal_config.end_date.format("YYYY-MM-DD"),
        )


class AutomaticEndMonthSelector:
    """Run existing collector checks against exact candidate months."""

    def __init__(self, collector: RawDataCollector) -> None:
        """Initialize the selector with the collector whose checks it will reuse."""
        self.collector = collector

    def select(
        self,
        *,
        candidate_pipeline_factory: Callable[[Arrow], Collection[ExportPipeline]],
        latest_candidate: Arrow,
        max_data_lag_months: int,
    ) -> Arrow:
        """Build and check exact candidates lazily from newest to oldest."""
        attempted: dict[str, list[str]] = {}

        for lag in range(max_data_lag_months + 1):
            candidate = latest_candidate.shift(months=-lag).floor("month")
            candidate_id = candidate.format("YYYY-MM")
            candidate_processors = list(candidate_pipeline_factory(candidate))
            if not candidate_processors:
                attempted[candidate_id] = ["no required monthly pipelines were configured"]
                continue

            logger.info("Checking automatic END_MONTH candidate %s", candidate_id)
            results = self.collector.collect(
                candidate_processors,
                allow_missing_required=True,
            )
            blockers = self._blocking_datasets(results)
            if not blockers:
                logger.info("Selected automatic END_MONTH=%s", candidate.format("YYYY-MM-DD"))
                return candidate
            attempted[candidate_id] = blockers
            logger.warning(
                "Automatic END_MONTH candidate %s is incomplete for: %s",
                candidate_id,
                ", ".join(blockers),
            )

        attempted_text = "; ".join(
            f"{month}: {', '.join(blockers)}" for month, blockers in attempted.items()
        )
        msg = (
            "No complete automatic END_MONTH was found within MAX_DATA_LAG_MONTHS="
            f"{max_data_lag_months}. Attempted {attempted_text}"
        )
        raise StaleDataError(msg)

    @staticmethod
    def _blocking_datasets(results: Collection[UploadResult]) -> list[str]:
        return sorted(
            {
                result.pipeline_config.hive_path.require_key("dataset")
                for result in results
                if not result.completeness.data_available
                and not result.pipeline_config.allows_missing_data
            },
        )


class EndMonthCoordinator:
    """Resolve the immutable temporal configuration for the discovery stage."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        collector: RawDataCollector,
        candidate_pipeline_factory: EndMonthCandidatePipelineFactory,
        start_date: Arrow,
        configured_end_month: Arrow,
        source: EndMonthSource,
        max_data_lag_months: int,
        store: EndMonthStore,
    ) -> None:
        """Initialize end-month coordination dependencies."""
        self.selector = AutomaticEndMonthSelector(collector)
        self.candidate_pipeline_factory = candidate_pipeline_factory
        self.start_date = start_date
        self.configured_end_month = configured_end_month
        self.source = source
        self.max_data_lag_months = max_data_lag_months
        self.store = store

    def resolve(self) -> EndMonthResolution:
        """Resolve and return a new immutable temporal configuration."""
        if self.source != "provisional":
            return EndMonthResolution(
                temporal_config=TemporalConfig(
                    start_date=self.start_date,
                    end_date=self.configured_end_month,
                ),
                store=self.store if self.source == "explicit" else None,
            )

        selected_end_month = self.selector.select(
            candidate_pipeline_factory=self.candidate_pipeline_factory.build,
            latest_candidate=self.configured_end_month,
            max_data_lag_months=self.max_data_lag_months,
        )
        return EndMonthResolution(
            temporal_config=TemporalConfig(
                start_date=self.start_date,
                end_date=selected_end_month,
            ),
            store=self.store,
        )
