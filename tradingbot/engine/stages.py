"""Chronological five-year history partitioning for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from engine.settings import HISTORY_STAGE_YEARS, TIMEFRAME_SECONDS
except ImportError:
    from .settings import HISTORY_STAGE_YEARS, TIMEFRAME_SECONDS


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

STAGE_SCHEMA_VERSION = 1
DEFAULT_STAGE_COUNT = int(HISTORY_STAGE_YEARS)
DEFAULT_STAGE_YEARS = 1
DEFAULT_CONTEXT_BARS = 0
UNASSIGNED_STAGE_ID = "UNASSIGNED"


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class StageError(RuntimeError):
    """Base exception for history-stage planning and partitioning."""


class StageValidationError(StageError, ValueError):
    """Raised when a stage plan or record timestamp is invalid."""


@dataclass(frozen=True)
class HistoryStage:
    """One chronological stage with half-open core and context intervals.

    ``start_time`` and ``end_time_exclusive`` define the interval used to assign
    records to this stage. Context bounds may include candles outside that core
    interval for pattern formation and confirmation across stage boundaries.
    Context records are not reassigned to the stage that supplied the context.
    """

    stage_id: str
    stage_index: int
    label: str
    nominal_start_time: int
    nominal_end_time_exclusive: int
    start_time: Optional[int]
    end_time_exclusive: Optional[int]
    context_start_time: Optional[int]
    context_end_time_exclusive: Optional[int]
    calendar_start_year: int
    calendar_end_year: int

    @property
    def is_active(self) -> bool:
        """Return whether the stage overlaps the supplied source history."""
        return (
            self.start_time is not None
            and self.end_time_exclusive is not None
            and self.start_time < self.end_time_exclusive
        )

    @property
    def duration_seconds(self) -> int:
        """Return the number of source-history seconds assigned to this stage."""
        if not self.is_active:
            return 0
        return int(self.end_time_exclusive) - int(self.start_time)

    def contains(self, timestamp: Any) -> bool:
        """Return whether a timestamp belongs to this stage's core interval."""
        if not self.is_active:
            return False
        value = normalize_timestamp(timestamp, "timestamp")
        return int(self.start_time) <= value < int(self.end_time_exclusive)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible stage description."""
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "label": self.label,
            "nominal_start_time": self.nominal_start_time,
            "nominal_start_utc": timestamp_to_iso(self.nominal_start_time),
            "nominal_end_time_exclusive": self.nominal_end_time_exclusive,
            "nominal_end_utc_exclusive": timestamp_to_iso(
                self.nominal_end_time_exclusive
            ),
            "start_time": self.start_time,
            "start_utc": (
                timestamp_to_iso(self.start_time)
                if self.start_time is not None else None
            ),
            "end_time_exclusive": self.end_time_exclusive,
            "end_utc_exclusive": (
                timestamp_to_iso(self.end_time_exclusive)
                if self.end_time_exclusive is not None else None
            ),
            "context_start_time": self.context_start_time,
            "context_start_utc": (
                timestamp_to_iso(self.context_start_time)
                if self.context_start_time is not None else None
            ),
            "context_end_time_exclusive": self.context_end_time_exclusive,
            "context_end_utc_exclusive": (
                timestamp_to_iso(self.context_end_time_exclusive)
                if self.context_end_time_exclusive is not None else None
            ),
            "calendar_start_year": self.calendar_start_year,
            "calendar_end_year": self.calendar_end_year,
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class HistoryStagePlan:
    """A fixed rolling history window divided into chronological year stages."""

    source_start_time: int
    source_end_time_exclusive: int
    analysis_start_time: int
    analysis_end_time_exclusive: int
    stage_count: int
    stage_years: int
    context_before_seconds: int
    context_after_seconds: int
    stages: Tuple[HistoryStage, ...]
    excluded_before_start_time: Optional[int] = None
    excluded_before_end_time_exclusive: Optional[int] = None

    @property
    def active_stages(self) -> Tuple[HistoryStage, ...]:
        """Return only stages that contain source-history records."""
        return tuple(stage for stage in self.stages if stage.is_active)

    @property
    def active_stage_count(self) -> int:
        """Return the number of stages with source-history coverage."""
        return len(self.active_stages)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible stage-plan description."""
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "source_start_time": self.source_start_time,
            "source_start_utc": timestamp_to_iso(self.source_start_time),
            "source_end_time_exclusive": self.source_end_time_exclusive,
            "source_end_utc_exclusive": timestamp_to_iso(
                self.source_end_time_exclusive
            ),
            "analysis_start_time": self.analysis_start_time,
            "analysis_start_utc": timestamp_to_iso(self.analysis_start_time),
            "analysis_end_time_exclusive": self.analysis_end_time_exclusive,
            "analysis_end_utc_exclusive": timestamp_to_iso(
                self.analysis_end_time_exclusive
            ),
            "stage_count": self.stage_count,
            "stage_years": self.stage_years,
            "context_before_seconds": self.context_before_seconds,
            "context_after_seconds": self.context_after_seconds,
            "active_stage_count": self.active_stage_count,
            "excluded_before_start_time": self.excluded_before_start_time,
            "excluded_before_end_time_exclusive": self.excluded_before_end_time_exclusive,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class StagePartition:
    """Records grouped by stage, with records outside the plan kept separate."""

    groups: Mapping[str, Tuple[Any, ...]]
    unassigned: Tuple[Any, ...]
    timestamp_field: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a compact partition summary without serializing source records."""
        return {
            "timestamp_field": self.timestamp_field,
            "groups": {
                stage_id: len(records)
                for stage_id, records in self.groups.items()
            },
            "unassigned_count": len(self.unassigned),
            "total_count": sum(len(records) for records in self.groups.values())
            + len(self.unassigned),
        }


# ============================================================================
# TIMESTAMP NORMALIZATION
# ============================================================================

def normalize_timestamp(value: Any, field_name: str = "timestamp") -> int:
    """Normalize a UTC Unix-seconds timestamp or timezone-aware datetime."""
    if isinstance(value, bool) or value is None:
        raise StageValidationError(f"{field_name} must be a valid UTC timestamp.")

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise StageValidationError(
                f"{field_name} datetime must include a timezone."
            )
        return int(value.astimezone(timezone.utc).timestamp())

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise StageValidationError(f"{field_name} must not be empty.")
        try:
            numeric = float(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as error:
                raise StageValidationError(
                    f"{field_name} must be Unix seconds or an ISO timestamp."
                ) from error
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise StageValidationError(
                    f"{field_name} ISO timestamp must include a timezone."
                )
            return int(parsed.astimezone(timezone.utc).timestamp())
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise StageValidationError(
                f"{field_name} must be a valid Unix-seconds timestamp."
            ) from error

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise StageValidationError(
            f"{field_name} must be a finite whole number of Unix seconds."
        )
    return int(numeric)


def timestamp_to_iso(timestamp: Any) -> str:
    """Format a Unix-seconds timestamp as an explicit UTC ISO-8601 value."""
    value = normalize_timestamp(timestamp)
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except (OverflowError, OSError, ValueError) as error:
        raise StageValidationError("Timestamp is outside the supported UTC range.") from error


def _add_calendar_years(timestamp: int, years: int) -> int:
    """Shift a UTC timestamp by calendar years, clamping leap days safely."""
    current = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    target_year = current.year + int(years)
    if target_year < 1 or target_year > 9999:
        raise StageValidationError("Calendar stage boundary is out of range.")
    target_day = min(current.day, calendar.monthrange(target_year, current.month)[1])
    shifted = current.replace(year=target_year, day=target_day)
    return int(shifted.timestamp())


def _positive_integer(value: Any, field_name: str) -> int:
    """Validate a positive integer configuration value."""
    if isinstance(value, bool):
        raise StageValidationError(f"{field_name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StageValidationError(
            f"{field_name} must be a positive integer."
        ) from error
    if result <= 0 or str(value).strip() not in {str(result), f"+{result}"}:
        raise StageValidationError(f"{field_name} must be a positive integer.")
    return result


def _non_negative_integer(value: Any, field_name: str) -> int:
    """Validate a non-negative integer configuration value."""
    if isinstance(value, bool):
        raise StageValidationError(f"{field_name} must be a non-negative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StageValidationError(
            f"{field_name} must be a non-negative integer."
        ) from error
    if result < 0 or str(value).strip() not in {str(result), f"+{result}"}:
        raise StageValidationError(
            f"{field_name} must be a non-negative integer."
        )
    return result


# ============================================================================
# STAGE PLAN CREATION
# ============================================================================

def build_stage_plan(
    start_time: Any,
    end_time_exclusive: Any,
    *,
    stage_count: int = DEFAULT_STAGE_COUNT,
    stage_years: int = DEFAULT_STAGE_YEARS,
    context_before_seconds: int = 0,
    context_after_seconds: int = 0,
) -> HistoryStagePlan:
    """Build chronological annual stages for the latest configured history.

    The requested analysis window is anchored at ``end_time_exclusive`` and
    extends backwards by ``stage_count * stage_years`` calendar years. Data
    older than that rolling window is reported as excluded rather than silently
    assigned to one of the stages. If source history is shorter than the rolling
    window, older stages remain present but inactive.

    The end timestamp is exclusive. Each stage uses a half-open interval so a
    timestamp exactly on a year boundary belongs to the following stage.
    """
    source_start = normalize_timestamp(start_time, "start_time")
    source_end = normalize_timestamp(end_time_exclusive, "end_time_exclusive")
    if source_end <= source_start:
        raise StageValidationError(
            "end_time_exclusive must be later than start_time."
        )

    count = _positive_integer(stage_count, "stage_count")
    years_per_stage = _positive_integer(stage_years, "stage_years")
    context_before = _non_negative_integer(
        context_before_seconds, "context_before_seconds"
    )
    context_after = _non_negative_integer(
        context_after_seconds, "context_after_seconds"
    )

    rolling_start = _add_calendar_years(source_end, -(count * years_per_stage))
    analysis_start = max(source_start, rolling_start)
    analysis_end = source_end

    stages: List[HistoryStage] = []
    for index in range(count):
        nominal_start = _add_calendar_years(
            rolling_start,
            index * years_per_stage,
        )
        nominal_end = _add_calendar_years(
            rolling_start,
            (index + 1) * years_per_stage,
        )
        clipped_start = max(source_start, nominal_start)
        clipped_end = min(source_end, nominal_end)

        if clipped_start >= clipped_end:
            active_start: Optional[int] = None
            active_end: Optional[int] = None
            context_start: Optional[int] = None
            context_end: Optional[int] = None
        else:
            active_start = clipped_start
            active_end = clipped_end
            context_start = max(source_start, clipped_start - context_before)
            context_end = min(source_end, clipped_end + context_after)

        nominal_start_year = datetime.fromtimestamp(
            nominal_start, tz=timezone.utc
        ).year
        nominal_end_year = datetime.fromtimestamp(
            nominal_end - 1, tz=timezone.utc
        ).year

        stage_index = index + 1
        stages.append(
            HistoryStage(
                stage_id=f"YEAR_{stage_index}",
                stage_index=stage_index,
                label=f"YEAR {stage_index}",
                nominal_start_time=nominal_start,
                nominal_end_time_exclusive=nominal_end,
                start_time=active_start,
                end_time_exclusive=active_end,
                context_start_time=context_start,
                context_end_time_exclusive=context_end,
                calendar_start_year=nominal_start_year,
                calendar_end_year=nominal_end_year,
            )
        )

    excluded_start: Optional[int] = None
    excluded_end: Optional[int] = None
    if source_start < rolling_start:
        excluded_start = source_start
        excluded_end = rolling_start

    return HistoryStagePlan(
        source_start_time=source_start,
        source_end_time_exclusive=source_end,
        analysis_start_time=analysis_start,
        analysis_end_time_exclusive=analysis_end,
        stage_count=count,
        stage_years=years_per_stage,
        context_before_seconds=context_before,
        context_after_seconds=context_after,
        stages=tuple(stages),
        excluded_before_start_time=excluded_start,
        excluded_before_end_time_exclusive=excluded_end,
    )


def build_history_stages(
    start_time: Any,
    end_time_exclusive: Any,
    *,
    stage_count: int = DEFAULT_STAGE_COUNT,
    stage_years: int = DEFAULT_STAGE_YEARS,
    context_before_seconds: int = 0,
    context_after_seconds: int = 0,
) -> List[HistoryStage]:
    """Return the stage windows from a rolling chronological history plan."""
    plan = build_stage_plan(
        start_time,
        end_time_exclusive,
        stage_count=stage_count,
        stage_years=stage_years,
        context_before_seconds=context_before_seconds,
        context_after_seconds=context_after_seconds,
    )
    return list(plan.stages)


def build_stage_plan_from_candles(
    candles: Sequence[Any],
    timeframe: str,
    *,
    stage_count: int = DEFAULT_STAGE_COUNT,
    stage_years: int = DEFAULT_STAGE_YEARS,
    context_before_bars: int = 0,
    context_after_bars: int = 0,
) -> HistoryStagePlan:
    """Build a stage plan from candle opening timestamps and timeframe duration."""
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in TIMEFRAME_SECONDS:
        raise StageValidationError(f"Unknown timeframe: {normalized_timeframe!r}.")
    if not candles:
        raise StageValidationError("At least one candle is required to build a plan.")

    times: List[int] = []
    for index, candle in enumerate(candles):
        raw_time = _read_field(candle, "time")
        if raw_time is None:
            raise StageValidationError(
                f"Candle at index {index} has no opening timestamp."
            )
        times.append(normalize_timestamp(raw_time, f"candles[{index}].time"))

    if times != sorted(times):
        raise StageValidationError("Candle timestamps must be chronological.")

    duration = int(TIMEFRAME_SECONDS[normalized_timeframe])
    start_time = times[0]
    end_time = times[-1] + duration
    before_seconds = _non_negative_integer(
        context_before_bars, "context_before_bars"
    ) * duration
    after_seconds = _non_negative_integer(
        context_after_bars, "context_after_bars"
    ) * duration

    return build_stage_plan(
        start_time,
        end_time,
        stage_count=stage_count,
        stage_years=stage_years,
        context_before_seconds=before_seconds,
        context_after_seconds=after_seconds,
    )


# ============================================================================
# STAGE LOOKUP AND RECORD PARTITIONING
# ============================================================================

def find_stage(
    timestamp: Any,
    stages: Sequence[HistoryStage] | HistoryStagePlan,
) -> Optional[HistoryStage]:
    """Return the single stage whose core interval contains the timestamp."""
    value = normalize_timestamp(timestamp)
    selected = stages.stages if isinstance(stages, HistoryStagePlan) else stages

    for stage in selected:
        if stage.contains(value):
            return stage
    return None


def stage_id_for_timestamp(
    timestamp: Any,
    stages: Sequence[HistoryStage] | HistoryStagePlan,
) -> Optional[str]:
    """Return the stage ID for a timestamp, or None outside the plan."""
    stage = find_stage(timestamp, stages)
    return stage.stage_id if stage is not None else None


def _read_field(record: Any, field_name: str) -> Any:
    """Read one record value from a mapping or object attribute."""
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def partition_records_by_stage(
    records: Iterable[Any],
    stages: Sequence[HistoryStage] | HistoryStagePlan,
    *,
    timestamp_field: str = "timestamp",
    include_unassigned: bool = False,
    sort_records: bool = True,
) -> StagePartition:
    """Assign records by an explicit event-time field using half-open windows.

    For pattern occurrences, use ``timestamp_field="recognized_at"`` when the
    goal is to group data by when the pattern became knowable. For trade records,
    use their entry or decision timestamp rather than their close timestamp.
    """
    field_name = str(timestamp_field).strip()
    if not field_name:
        raise StageValidationError("timestamp_field must not be empty.")

    selected_stages = stages.stages if isinstance(stages, HistoryStagePlan) else tuple(stages)
    groups: Dict[str, List[Tuple[int, int, Any]]] = {
        stage.stage_id: [] for stage in selected_stages
    }
    unassigned: List[Tuple[int, int, Any]] = []

    for sequence_index, record in enumerate(records):
        raw_timestamp = _read_field(record, field_name)
        if raw_timestamp is None:
            unassigned.append((0, sequence_index, record))
            continue

        timestamp = normalize_timestamp(
            raw_timestamp,
            f"record.{field_name}",
        )
        stage = find_stage(timestamp, selected_stages)
        if stage is None:
            unassigned.append((timestamp, sequence_index, record))
        else:
            groups[stage.stage_id].append((timestamp, sequence_index, record))

    if sort_records:
        for values in groups.values():
            values.sort(key=lambda item: (item[0], item[1]))
        unassigned.sort(key=lambda item: (item[0], item[1]))

    normalized_groups: Dict[str, Tuple[Any, ...]] = {
        stage_id: tuple(item[2] for item in values)
        for stage_id, values in groups.items()
    }
    normalized_unassigned = tuple(item[2] for item in unassigned)

    if include_unassigned:
        normalized_groups[UNASSIGNED_STAGE_ID] = normalized_unassigned
        normalized_unassigned = tuple()

    return StagePartition(
        groups=normalized_groups,
        unassigned=normalized_unassigned,
        timestamp_field=field_name,
    )


def partition_situations_by_recognition_time(
    situations: Iterable[Any],
    stages: Sequence[HistoryStage] | HistoryStagePlan,
    *,
    timestamp_field: str = "recognized_at",
) -> StagePartition:
    """Partition situation records by their point-in-time recognition timestamp."""
    return partition_records_by_stage(
        situations,
        stages,
        timestamp_field=timestamp_field,
    )


def partition_trades_by_entry_time(
    trades: Iterable[Any],
    stages: Sequence[HistoryStage] | HistoryStagePlan,
    *,
    timestamp_field: str = "entry_time",
) -> StagePartition:
    """Partition trade records by the decision or entry timestamp."""
    return partition_records_by_stage(
        trades,
        stages,
        timestamp_field=timestamp_field,
    )


# ============================================================================
# CANDLE WINDOWS AND STAGE COVERAGE
# ============================================================================

def slice_candles_for_stage(
    candles: Sequence[Any],
    stage: HistoryStage,
    timeframe: str,
    *,
    context_bars: int = DEFAULT_CONTEXT_BARS,
    include_context_after: bool = False,
) -> List[Any]:
    """Return candles in one stage interval with optional boundary context.

    Candle timestamps represent opening times. With ``include_context_after``
    enabled, later candles may be included to confirm structures that began
    near the stage boundary. Such future context must only be used to establish
    a later recognition timestamp; it must not be fed into an earlier decision.
    """
    if not isinstance(stage, HistoryStage):
        raise TypeError("stage must be a HistoryStage instance.")
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in TIMEFRAME_SECONDS:
        raise StageValidationError(f"Unknown timeframe: {normalized_timeframe!r}.")
    bars = _non_negative_integer(context_bars, "context_bars")
    if not stage.is_active:
        return []

    duration = int(TIMEFRAME_SECONDS[normalized_timeframe])
    normalized_candles: List[Tuple[int, int, Any]] = []
    for index, candle in enumerate(candles):
        raw_time = _read_field(candle, "time")
        if raw_time is None:
            continue
        candle_time = normalize_timestamp(raw_time, f"candles[{index}].time")
        normalized_candles.append((candle_time, index, candle))

    if not normalized_candles:
        return []

    available_start = min(item[0] for item in normalized_candles)
    available_end = max(item[0] + duration for item in normalized_candles)
    planned_before = (
        max(int(stage.start_time) - int(stage.context_start_time), 0)
        if stage.context_start_time is not None else 0
    )
    effective_before = max(bars * duration, planned_before)
    start_bound = max(available_start, int(stage.start_time) - effective_before)

    end_bound = int(stage.end_time_exclusive)
    if include_context_after:
        planned_after = (
            max(int(stage.context_end_time_exclusive) - end_bound, 0)
            if stage.context_end_time_exclusive is not None else 0
        )
        effective_after = max(bars * duration, planned_after)
        end_bound = min(available_end, end_bound + effective_after)
    else:
        end_bound = min(available_end, end_bound)

    selected: List[Tuple[int, int, Any]] = []
    for candle_time, index, candle in normalized_candles:
        candle_close = candle_time + duration
        if candle_time < start_bound or candle_time >= end_bound:
            continue
        if candle_close > end_bound:
            continue
        selected.append((candle_time, index, candle))

    selected.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in selected]


def iter_stage_candle_windows(
    candles: Sequence[Any],
    stages: Sequence[HistoryStage] | HistoryStagePlan,
    timeframe: str,
    *,
    context_bars: int = DEFAULT_CONTEXT_BARS,
    include_context_after: bool = False,
) -> Iterator[Tuple[HistoryStage, List[Any]]]:
    """Yield stage windows from oldest to newest with configured context."""
    selected_stages = stages.stages if isinstance(stages, HistoryStagePlan) else stages
    for stage in selected_stages:
        if not stage.is_active:
            continue
        yield stage, slice_candles_for_stage(
            candles,
            stage,
            timeframe,
            context_bars=context_bars,
            include_context_after=include_context_after,
        )


def summarize_stage_coverage(
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    stages: Sequence[HistoryStage] | HistoryStagePlan,
) -> Dict[str, Any]:
    """Summarize candle coverage for each stage and loaded timeframe."""
    selected_stages = stages.stages if isinstance(stages, HistoryStagePlan) else stages
    summary: Dict[str, Any] = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "stage_count": len(selected_stages),
        "timeframes": {},
        "stages": {},
    }

    normalized: Dict[str, List[Tuple[int, Any]]] = {}
    for raw_timeframe, candles in candles_by_timeframe.items():
        timeframe = str(raw_timeframe).strip().upper()
        if timeframe not in TIMEFRAME_SECONDS:
            raise StageValidationError(f"Unknown timeframe: {timeframe!r}.")
        values: List[Tuple[int, Any]] = []
        for index, candle in enumerate(candles):
            raw_time = _read_field(candle, "time")
            if raw_time is None:
                continue
            values.append((normalize_timestamp(raw_time, f"{timeframe}[{index}].time"), candle))
        values.sort(key=lambda item: item[0])
        normalized[timeframe] = values
        summary["timeframes"][timeframe] = {
            "candle_count": len(values),
            "first_candle_time": values[0][0] if values else None,
            "last_candle_time": values[-1][0] if values else None,
        }

    for stage in selected_stages:
        stage_data: Dict[str, Any] = {
            "stage": stage.to_dict(),
            "timeframes": {},
        }
        for timeframe, values in normalized.items():
            if not stage.is_active:
                selected_values: List[Tuple[int, Any]] = []
            else:
                start = int(stage.start_time)
                end = int(stage.end_time_exclusive)
                selected_values = [
                    item for item in values if start <= item[0] < end
                ]
            stage_data["timeframes"][timeframe] = {
                "candle_count": len(selected_values),
                "first_candle_time": (
                    selected_values[0][0] if selected_values else None
                ),
                "last_candle_time": (
                    selected_values[-1][0] if selected_values else None
                ),
            }
        summary["stages"][stage.stage_id] = stage_data

    return summary


# ============================================================================
# CHRONOLOGICAL PROCESSING HELPERS
# ============================================================================

def iter_stages(
    stages: Sequence[HistoryStage] | HistoryStagePlan,
    *,
    newest_first: bool = False,
    active_only: bool = True,
) -> Iterator[HistoryStage]:
    """Iterate stages in chronological or reverse chronological order."""
    selected = list(stages.stages if isinstance(stages, HistoryStagePlan) else stages)
    if active_only:
        selected = [stage for stage in selected if stage.is_active]
    if newest_first:
        selected.reverse()
    return iter(selected)


def stage_sequence_metadata(
    stages: Sequence[HistoryStage] | HistoryStagePlan,
) -> List[Dict[str, Any]]:
    """Return chronological stage metadata without assigning training weights."""
    selected = stages.stages if isinstance(stages, HistoryStagePlan) else stages
    active_ids = {stage.stage_id for stage in selected if stage.is_active}
    count = len(selected)
    result: List[Dict[str, Any]] = []
    for stage in selected:
        result.append(
            {
                **stage.to_dict(),
                "is_active": stage.stage_id in active_ids,
                "chronological_position": stage.stage_index,
                "recency_order": count - stage.stage_index + 1,
            }
        )
    return result


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "STAGE_SCHEMA_VERSION",
    "DEFAULT_STAGE_COUNT",
    "DEFAULT_STAGE_YEARS",
    "DEFAULT_CONTEXT_BARS",
    "UNASSIGNED_STAGE_ID",
    "StageError",
    "StageValidationError",
    "HistoryStage",
    "HistoryStagePlan",
    "StagePartition",
    "normalize_timestamp",
    "timestamp_to_iso",
    "build_stage_plan",
    "build_history_stages",
    "build_stage_plan_from_candles",
    "find_stage",
    "stage_id_for_timestamp",
    "partition_records_by_stage",
    "partition_situations_by_recognition_time",
    "partition_trades_by_entry_time",
    "slice_candles_for_stage",
    "iter_stage_candle_windows",
    "summarize_stage_coverage",
    "iter_stages",
    "stage_sequence_metadata",
]
