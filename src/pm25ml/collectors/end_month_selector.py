"""Select the newest sufficiently recent month with complete required data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pm25ml.logging import logger
from pm25ml.setup.date_params import TemporalConfig
from pm25ml.setup.end_month import EndMonthStore, StaleDataError, latest_completed_month

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from arrow import Arrow

    from pm25ml.collectors.collector import RawDataCollector, UploadResult
    from pm25ml.collectors.export_pipeline import ExportPipeline
    from pm25ml.setup.settings import TemporalConfigRequest


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
            candidate_processors = [
                processor
                for processor in candidate_pipeline_factory(candidate)
                if self._constrains_candidate(processor, candidate_id)
            ]
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
    def _constrains_candidate(processor: ExportPipeline, candidate_id: str) -> bool:
        config = processor.get_config_metadata()
        return (
            config.hive_path.metadata.get("month") == candidate_id
            and not config.allows_missing_data
            and config.constrains_end_month
        )

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
    """Resolve and persist the immutable temporal configuration for collection."""

    def __init__(
        self,
        *,
        collector: RawDataCollector,
        store: EndMonthStore,
    ) -> None:
        """Initialize end-month coordination dependencies."""
        self.selector = AutomaticEndMonthSelector(collector)
        self.store = store

    def resolve(
        self,
        request: TemporalConfigRequest,
        *,
        candidate_pipeline_factory: Callable[[Arrow], Collection[ExportPipeline]],
        now: Arrow | None = None,
    ) -> TemporalConfig:
        """Resolve, persist, and return the collection month range."""
        stored_end_month = self.store.read()
        if stored_end_month is not None:
            return TemporalConfig(start_date=request.start_month, end_date=stored_end_month)

        selected_end_month = request.explicit_end_month
        if selected_end_month is None:
            selected_end_month = self.selector.select(
                candidate_pipeline_factory=candidate_pipeline_factory,
                latest_candidate=latest_completed_month(now=now),
                max_data_lag_months=request.max_data_lag_months,
            )

        selected_end_month = selected_end_month.floor("month")
        temporal_config = TemporalConfig(
            start_date=request.start_month,
            end_date=selected_end_month,
        )
        self.store.write(selected_end_month)
        logger.info("Persisted END_MONTH=%s", selected_end_month.format("YYYY-MM-DD"))
        return temporal_config
