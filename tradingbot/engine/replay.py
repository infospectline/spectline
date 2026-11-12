"""Point-in-time chronological replay for Spectline historical market data."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from client import normalize_market as client_normalize_market
except ImportError:
    try:
        from ..client import normalize_market as client_normalize_market
    except ImportError:
        client_normalize_market = None

try:
    from engine.settings import ALL_TIMEFRAMES, SUPPORTED_TIMEFRAMES, TIMEFRAME_SECONDS
    from engine.stages import HistoryStage, HistoryStagePlan
except ImportError:
    from .settings import ALL_TIMEFRAMES, SUPPORTED_TIMEFRAMES, TIMEFRAME_SECONDS
    from .stages import HistoryStage, HistoryStagePlan


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

REPLAY_SCHEMA_VERSION = 1
DEFAULT_MAX_HISTORY_BARS: Optional[int] = None
DEFAULT_INCLUDE_PATTERN_OCCURRENCES = True
DEFAULT_INCLUDE_MARKET_EVENTS = True
DEFAULT_INCLUDE_TRADE_RECORDS = False

_EVENT_PRIORITY = {
    "CANDLE_CLOSED": 10,
    "PATTERN_RECOGNIZED": 20,
    "MARKET_EVENT_RECOGNIZED": 30,
    "TRADE_PREDICTION_AVAILABLE": 40,
    "TRADE_OPENED": 50,
    "TRADE_RESULT_KNOWN": 60,
    "CUSTOM": 90,
}


# ============================================================================
# ERRORS
# ============================================================================

class ReplayError(RuntimeError):
    """Base exception for chronological replay failures."""


class ReplayValidationError(ReplayError, ValueError):
    """Raised when replay inputs or event timestamps are invalid."""


class ReplayCallbackError(ReplayError):
    """Raised when a user-supplied replay callback fails."""


# ============================================================================
# CONFIGURATION AND DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class ReplayConfig:
    """Immutable settings for one point-in-time historical replay."""

    start_time: Optional[int] = None
    end_time_exclusive: Optional[int] = None
    timeframes: Optional[Tuple[str, ...]] = None
    max_history_bars_per_timeframe: Optional[int] = DEFAULT_MAX_HISTORY_BARS
    include_pattern_occurrences: bool = DEFAULT_INCLUDE_PATTERN_OCCURRENCES
    include_market_events: bool = DEFAULT_INCLUDE_MARKET_EVENTS
    include_trade_records: bool = DEFAULT_INCLUDE_TRADE_RECORDS
    strict_candle_validation: bool = True
    strict_event_validation: bool = True
    include_stage_context: bool = True

    def __post_init__(self) -> None:
        start = _optional_timestamp(self.start_time, "start_time")
        end = _optional_timestamp(self.end_time_exclusive, "end_time_exclusive")
        if start is not None and end is not None and end <= start:
            raise ReplayValidationError("end_time_exclusive must be later than start_time.")

        selected: Optional[Tuple[str, ...]] = None
        if self.timeframes is not None:
            selected = _normalize_timeframes(self.timeframes)
            if not selected:
                raise ReplayValidationError("At least one timeframe must be selected.")

        limit = self.max_history_bars_per_timeframe
        if limit is not None:
            if isinstance(limit, bool):
                raise ReplayValidationError("max_history_bars_per_timeframe must be a positive integer.")
            try:
                limit = int(limit)
            except (TypeError, ValueError, OverflowError) as error:
                raise ReplayValidationError(
                    "max_history_bars_per_timeframe must be a positive integer."
                ) from error
            if limit < 1:
                raise ReplayValidationError("max_history_bars_per_timeframe must be positive.")

        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time_exclusive", end)
        object.__setattr__(self, "timeframes", selected)
        object.__setattr__(self, "max_history_bars_per_timeframe", limit)


@dataclass(frozen=True)
class ReplayEvent:
    """One item becoming observable at a specific replay timestamp."""

    event_id: str
    event_type: str
    market: str
    event_time: int
    available_at: int
    timeframe: Optional[str]
    payload: Mapping[str, Any]
    source: str
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "market": self.market,
            "event_time": self.event_time,
            "available_at": self.available_at,
            "timeframe": self.timeframe,
            "payload": dict(self.payload),
            "source": self.source,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class ReplayStep:
    """All information that becomes available at one shared UTC timestamp."""

    market: str
    current_time: int
    events: Tuple[ReplayEvent, ...]
    new_candles_by_timeframe: Mapping[str, Tuple[Any, ...]]
    candles_by_timeframe: Mapping[str, Tuple[Any, ...]]
    known_pattern_occurrences: Tuple[Mapping[str, Any], ...]
    known_market_events: Tuple[Mapping[str, Any], ...]
    known_trade_records: Tuple[Mapping[str, Any], ...]
    candle_counts: Mapping[str, int]
    stage_id: Optional[str] = None

    @property
    def new_candle_count(self) -> int:
        return sum(len(values) for values in self.new_candles_by_timeframe.values())

    def to_dict(self, include_history: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "market": self.market,
            "current_time": self.current_time,
            "stage_id": self.stage_id,
            "events": [event.to_dict() for event in self.events],
            "new_candles_by_timeframe": {
                timeframe: len(values)
                for timeframe, values in self.new_candles_by_timeframe.items()
            },
            "candle_counts": dict(self.candle_counts),
            "known_pattern_occurrence_count": len(self.known_pattern_occurrences),
            "known_market_event_count": len(self.known_market_events),
            "known_trade_record_count": len(self.known_trade_records),
        }
        if include_history:
            result["candles_by_timeframe"] = {
                timeframe: [_record_to_dict(candle) for candle in values]
                for timeframe, values in self.candles_by_timeframe.items()
            }
        return result


@dataclass(frozen=True)
class ReplaySummary:
    """Counters and time bounds collected while a replay is running."""

    market: str
    step_count: int
    event_count: int
    candle_count: int
    pattern_occurrence_count: int
    market_event_count: int
    trade_event_count: int
    first_available_at: Optional[int]
    last_available_at: Optional[int]
    stopped_early: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "market": self.market,
            "step_count": self.step_count,
            "event_count": self.event_count,
            "candle_count": self.candle_count,
            "pattern_occurrence_count": self.pattern_occurrence_count,
            "market_event_count": self.market_event_count,
            "trade_event_count": self.trade_event_count,
            "first_available_at": self.first_available_at,
            "last_available_at": self.last_available_at,
            "stopped_early": self.stopped_early,
        }


@dataclass(frozen=True)
class _ScheduledItem:
    available_at: int
    priority: int
    timeframe_order: int
    sequence: int
    event: ReplayEvent
    candle: Any = None


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def _read_field(record: Any, field_name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ReplayValidationError(f"{field_name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReplayValidationError(f"{field_name} must be a finite number.") from error
    if not math.isfinite(number):
        raise ReplayValidationError(f"{field_name} must be a finite number.")
    return number


def _timestamp(value: Any, field_name: str) -> int:
    number = _finite_number(value, field_name)
    if not number.is_integer() or number < 0:
        raise ReplayValidationError(f"{field_name} must be a non-negative Unix-seconds integer.")
    return int(number)


def _optional_timestamp(value: Any, field_name: str) -> Optional[int]:
    return None if value is None else _timestamp(value, field_name)


def _normalize_market(market: str) -> str:
    raw_value = str(market).strip()
    if not raw_value or raw_value in {".", ".."}:
        raise ReplayValidationError("market must be a non-empty market identifier.")
    if callable(client_normalize_market):
        try:
            normalized = str(client_normalize_market(raw_value)).strip()
        except Exception as error:
            raise ReplayValidationError(f"Invalid market identifier: {raw_value!r}.") from error
        if not normalized:
            raise ReplayValidationError("market must normalize to a non-empty identifier.")
        return normalized
    return raw_value


def _timeframe_order() -> Tuple[str, ...]:
    ordered = tuple(ALL_TIMEFRAMES)
    if not ordered:
        ordered = tuple(SUPPORTED_TIMEFRAMES)
    return ordered


def _normalize_timeframes(timeframes: Iterable[str]) -> Tuple[str, ...]:
    values = (timeframes,) if isinstance(timeframes, str) else tuple(timeframes)
    requested = {str(value).strip().upper() for value in values if str(value).strip()}
    unknown = requested - set(TIMEFRAME_SECONDS)
    if unknown:
        raise ReplayValidationError(
            "Unknown timeframe(s): " + ", ".join(sorted(unknown))
        )
    unavailable = requested - set(SUPPORTED_TIMEFRAMES)
    if unavailable:
        raise ReplayValidationError(
            "Timeframe(s) unavailable in the current runtime: "
            + ", ".join(sorted(unavailable))
        )
    return tuple(frame for frame in _timeframe_order() if frame in requested)


def _candle_to_dict(candle: Any) -> Dict[str, Any]:
    return {
        "time": _read_field(candle, "time"),
        "open": _read_field(candle, "open"),
        "high": _read_field(candle, "high"),
        "low": _read_field(candle, "low"),
        "close": _read_field(candle, "close"),
        "volume": _read_field(candle, "volume"),
    }


def _record_to_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "to_dict") and callable(record.to_dict):
        return dict(record.to_dict())
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    raise ReplayValidationError(
        f"Unsupported replay record type: {type(record).__name__}."
    )


def _normalize_candle_stream(
    market: str,
    timeframe: str,
    candles: Sequence[Any],
    strict: bool,
) -> List[Tuple[int, int, Any]]:
    duration = int(TIMEFRAME_SECONDS[timeframe])
    normalized: List[Tuple[int, int, Any]] = []
    previous_time: Optional[int] = None

    for index, candle in enumerate(candles):
        candle_time = _timestamp(
            _read_field(candle, "time"),
            f"{market}.{timeframe}.candles[{index}].time",
        )
        values = {
            name: _finite_number(
                _read_field(candle, name),
                f"{market}.{timeframe}.candles[{index}].{name}",
            )
            for name in ("open", "high", "low", "close")
        }
        if (
            values["high"] < values["low"]
            or values["high"] < max(values["open"], values["close"])
            or values["low"] > min(values["open"], values["close"])
        ):
            raise ReplayValidationError(
                f"Invalid OHLC range in {timeframe} candle at {candle_time}."
            )
        if previous_time is not None and candle_time <= previous_time:
            if strict:
                raise ReplayValidationError(
                    f"{timeframe} candles must be unique and chronologically ordered."
                )
        previous_time = candle_time
        normalized.append((candle_time, candle_time + duration, candle))

    if not strict:
        by_time: Dict[int, Tuple[int, int, Any]] = {}
        for item in normalized:
            by_time[item[0]] = item
        normalized = [by_time[key] for key in sorted(by_time)]

    return normalized


def _validate_stage(stage: Optional[HistoryStage | HistoryStagePlan]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[str]]:
    if stage is None:
        return None, None, None, None, None

    if isinstance(stage, HistoryStagePlan):
        if stage.analysis_end_time_exclusive <= stage.analysis_start_time:
            raise ReplayValidationError("HistoryStagePlan has an invalid analysis interval.")
        return (
            int(stage.analysis_start_time),
            int(stage.analysis_end_time_exclusive),
            int(stage.source_start_time),
            int(stage.source_end_time_exclusive),
            "STAGE_PLAN",
        )

    if not isinstance(stage, HistoryStage):
        raise TypeError("stage must be a HistoryStage or HistoryStagePlan instance.")
    if not stage.is_active:
        raise ReplayValidationError("Cannot replay an inactive history stage.")
    return (
        int(stage.start_time),
        int(stage.end_time_exclusive),
        int(stage.context_start_time) if stage.context_start_time is not None else None,
        int(stage.context_end_time_exclusive) if stage.context_end_time_exclusive is not None else None,
        stage.stage_id,
    )


# ============================================================================
# EVENT BUILDERS
# ============================================================================

def _build_market_event(
    market: str,
    record: Any,
    sequence: int,
    event_type: str,
    source: str,
    timeframe_order: int,
) -> _ScheduledItem:
    payload = _record_to_dict(record)
    record_market = payload.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise ReplayValidationError("Replay record market does not match the replay market.")
    timeframe_value = str(payload.get("timeframe", "")).strip().upper()
    if not timeframe_value:
        raise ReplayValidationError("Market replay records must include a timeframe.")
    timeframe = _normalize_timeframes((timeframe_value,))[0]

    if event_type == "PATTERN_RECOGNIZED":
        event_time = _timestamp(payload.get("occurrence_start"), "occurrence_start")
        available_at = _timestamp(payload.get("recognized_at"), "recognized_at")
        if available_at < event_time:
            raise ReplayValidationError("Pattern recognized_at cannot precede occurrence_start.")
    else:
        event_time = _timestamp(payload.get("event_time"), "event_time")
        available_at = _timestamp(payload.get("recognized_at", event_time), "recognized_at")
        if available_at < event_time:
            raise ReplayValidationError("Market event recognized_at cannot precede event_time.")

    event_id = str(
        payload.get("record_id")
        or payload.get("occurrence_id")
        or payload.get("event_id")
        or f"{source}_{sequence}_{event_time}_{available_at}"
    )
    event = ReplayEvent(
        event_id=event_id,
        event_type=event_type,
        market=market,
        event_time=event_time,
        available_at=available_at,
        timeframe=timeframe,
        payload=payload,
        source=source,
        sequence=sequence,
    )
    return _ScheduledItem(
        available_at=available_at,
        priority=_EVENT_PRIORITY[event_type],
        timeframe_order=timeframe_order,
        sequence=sequence,
        event=event,
    )


def _build_trade_items(
    market: str,
    trade: Any,
    sequence: int,
) -> List[_ScheduledItem]:
    record = _record_to_dict(trade)
    record_market = record.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise ReplayValidationError("Trade market does not match the replay market.")
    trade_id = str(record.get("trade_id", record.get("setup_id", f"TRADE_{sequence}")))
    entry_time = _timestamp(record.get("entry_time"), "trade.entry_time")
    exit_time = _timestamp(record.get("exit_time"), "trade.exit_time")
    outcome_known_at = _timestamp(
        record.get("outcome_known_at", exit_time),
        "trade.outcome_known_at",
    )
    if exit_time < entry_time or outcome_known_at < exit_time:
        raise ReplayValidationError("Trade timestamps must satisfy entry_time <= exit_time <= outcome_known_at.")

    items: List[_ScheduledItem] = []
    prediction = record.get("predicted_probability_at_entry")
    if prediction is not None:
        probability = _finite_number(prediction, "predicted_probability_at_entry")
        if not 0.0 <= probability <= 1.0:
            raise ReplayValidationError("predicted_probability_at_entry must be between zero and one.")
        prediction_time = _timestamp(record.get("predicted_at", entry_time), "predicted_at")
        if prediction_time > entry_time:
            raise ReplayValidationError("Trade prediction cannot be available after entry_time.")
        prediction_payload = {
            "trade_id": trade_id,
            "predicted_at": prediction_time,
            "predicted_probability_at_entry": probability,
        }
        prediction_event = ReplayEvent(
            event_id=f"{trade_id}:PREDICTION",
            event_type="TRADE_PREDICTION_AVAILABLE",
            market=market,
            event_time=prediction_time,
            available_at=prediction_time,
            timeframe=None,
            payload=prediction_payload,
            source="trade",
            sequence=sequence,
        )
        items.append(
            _ScheduledItem(
                prediction_time,
                _EVENT_PRIORITY["TRADE_PREDICTION_AVAILABLE"],
                len(_timeframe_order()),
                sequence,
                prediction_event,
            )
        )

    safe_trade_payload = {
        key: value
        for key, value in record.items()
        if key not in {
            "actual_outcome",
            "outcome_known_at",
            "exit_time",
            "predicted_probability_at_entry",
            "predicted_at",
        }
    }
    safe_trade_payload["trade_id"] = trade_id
    entry_event = ReplayEvent(
        event_id=f"{trade_id}:OPENED",
        event_type="TRADE_OPENED",
        market=market,
        event_time=entry_time,
        available_at=entry_time,
        timeframe=None,
        payload=safe_trade_payload,
        source="trade",
        sequence=sequence,
    )
    items.append(
        _ScheduledItem(
            entry_time,
            _EVENT_PRIORITY["TRADE_OPENED"],
            len(_timeframe_order()),
            sequence,
            entry_event,
        )
    )

    outcome = record.get("actual_outcome")
    if isinstance(outcome, bool) or outcome not in (0, 1):
        raise ReplayValidationError("Finalized trade actual_outcome must be 0 or 1.")
    result_payload = {
        "trade_id": trade_id,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "outcome_known_at": outcome_known_at,
        "actual_outcome": int(outcome),
    }
    result_event = ReplayEvent(
        event_id=f"{trade_id}:RESULT",
        event_type="TRADE_RESULT_KNOWN",
        market=market,
        event_time=outcome_known_at,
        available_at=outcome_known_at,
        timeframe=None,
        payload=result_payload,
        source="trade",
        sequence=sequence,
    )
    items.append(
        _ScheduledItem(
            outcome_known_at,
            _EVENT_PRIORITY["TRADE_RESULT_KNOWN"],
            len(_timeframe_order()),
            sequence,
            result_event,
        )
    )
    return items


# ============================================================================
# REPLAY ENGINE
# ============================================================================

class HistoricalReplay:
    """Replay candle closes and recognized records in strict chronological order.

    A candle becomes available at its closing timestamp, calculated as its
    opening timestamp plus the configured timeframe duration. Pattern records
    become available at ``recognized_at`` rather than their occurrence start.
    Finalized trade outcomes become available only at ``outcome_known_at``.
    """

    def __init__(
        self,
        market: str,
        candles_by_timeframe: Mapping[str, Sequence[Any]],
        *,
        pattern_occurrences: Iterable[Any] = (),
        market_events: Iterable[Any] = (),
        trade_records: Iterable[Any] = (),
        config: Optional[ReplayConfig] = None,
        stage: Optional[HistoryStage | HistoryStagePlan] = None,
    ) -> None:
        if not isinstance(candles_by_timeframe, Mapping):
            raise TypeError("candles_by_timeframe must be a timeframe-to-candles mapping.")

        self.market = _normalize_market(market)
        self.config = config or ReplayConfig()
        if not isinstance(self.config, ReplayConfig):
            raise TypeError("config must be a ReplayConfig instance.")
        self.stage = stage
        stage_start, stage_end, context_start, context_end, stage_id = _validate_stage(stage)
        self.stage_id = stage_id

        selected_frames = (
            self.config.timeframes
            if self.config.timeframes is not None
            else _normalize_timeframes(tuple(candles_by_timeframe.keys()))
        )
        if not selected_frames:
            raise ReplayValidationError("No usable timeframe archives were supplied.")

        self.timeframes = tuple(selected_frames)
        self._candle_streams: Dict[str, List[Tuple[int, int, Any]]] = {}
        for timeframe in self.timeframes:
            raw_candles = candles_by_timeframe.get(timeframe, ())
            self._candle_streams[timeframe] = _normalize_candle_stream(
                self.market,
                timeframe,
                raw_candles,
                self.config.strict_candle_validation,
            )

        effective_start = self.config.start_time
        effective_end = self.config.end_time_exclusive
        if stage_start is not None:
            effective_start = stage_start if effective_start is None else max(effective_start, stage_start)
        if stage_end is not None:
            effective_end = stage_end if effective_end is None else min(effective_end, stage_end)
        if effective_start is not None and effective_end is not None and effective_end <= effective_start:
            raise ReplayValidationError("Replay time window does not overlap the selected stage.")
        self.start_time = effective_start
        self.end_time_exclusive = effective_end

        self._context_start = context_start
        self._context_end = context_end
        self._warmup: Dict[str, List[Any]] = {timeframe: [] for timeframe in self.timeframes}
        self._items: List[_ScheduledItem] = []
        self._build_schedule(
            pattern_occurrences=pattern_occurrences,
            market_events=market_events,
            trade_records=trade_records,
        )

    def _build_schedule(
        self,
        pattern_occurrences: Iterable[Any],
        market_events: Iterable[Any],
        trade_records: Iterable[Any],
    ) -> None:
        timeframe_order = {name: index for index, name in enumerate(_timeframe_order())}
        sequence = 0

        for timeframe in self.timeframes:
            duration = int(TIMEFRAME_SECONDS[timeframe])
            for candle_time, available_at, candle in self._candle_streams[timeframe]:
                if self.start_time is not None and available_at < self.start_time:
                    if (
                        self.config.include_stage_context
                        and self._context_start is not None
                        and candle_time >= self._context_start
                        and (self._context_end is None or available_at <= self._context_end)
                    ):
                        self._warmup[timeframe].append(candle)
                    continue
                if self.end_time_exclusive is not None and available_at >= self.end_time_exclusive:
                    continue
                sequence += 1
                event = ReplayEvent(
                    event_id=f"CANDLE:{timeframe}:{candle_time}",
                    event_type="CANDLE_CLOSED",
                    market=self.market,
                    event_time=candle_time,
                    available_at=available_at,
                    timeframe=timeframe,
                    payload=_candle_to_dict(candle),
                    source="candle",
                    sequence=sequence,
                )
                self._items.append(
                    _ScheduledItem(
                        available_at=available_at,
                        priority=_EVENT_PRIORITY["CANDLE_CLOSED"],
                        timeframe_order=timeframe_order.get(timeframe, len(timeframe_order)),
                        sequence=sequence,
                        event=event,
                        candle=candle,
                    )
                )

        if self.config.include_pattern_occurrences:
            for record in pattern_occurrences:
                sequence += 1
                item = _build_market_event(
                    self.market,
                    record,
                    sequence,
                    "PATTERN_RECOGNIZED",
                    "pattern_occurrence",
                    len(timeframe_order),
                )
                if self._in_replay_window(item.available_at):
                    self._items.append(item)

        if self.config.include_market_events:
            for record in market_events:
                sequence += 1
                item = _build_market_event(
                    self.market,
                    record,
                    sequence,
                    "MARKET_EVENT_RECOGNIZED",
                    "market_event",
                    len(timeframe_order) + 1,
                )
                if self._in_replay_window(item.available_at):
                    self._items.append(item)

        if self.config.include_trade_records:
            for record in trade_records:
                sequence += 1
                for item in _build_trade_items(self.market, record, sequence):
                    if self._in_replay_window(item.available_at):
                        self._items.append(item)

        self._items.sort(
            key=lambda item: (
                item.available_at,
                item.priority,
                item.timeframe_order,
                item.sequence,
                item.event.event_id,
            )
        )

    def _in_replay_window(self, timestamp: int) -> bool:
        if self.start_time is not None and timestamp < self.start_time:
            return False
        if self.end_time_exclusive is not None and timestamp >= self.end_time_exclusive:
            return False
        return True

    def iter_steps(self) -> Iterator[ReplayStep]:
        """Yield one snapshot for each timestamp at which new information arrives."""
        histories: Dict[str, List[Any]] = {
            timeframe: list(self._warmup.get(timeframe, ()))
            for timeframe in self.timeframes
        }
        visible_patterns: List[Mapping[str, Any]] = []
        visible_market_events: List[Mapping[str, Any]] = []
        visible_trades: List[Mapping[str, Any]] = []

        index = 0
        while index < len(self._items):
            current_time = self._items[index].available_at
            batch: List[_ScheduledItem] = []
            while index < len(self._items) and self._items[index].available_at == current_time:
                batch.append(self._items[index])
                index += 1

            new_candles: Dict[str, List[Any]] = {timeframe: [] for timeframe in self.timeframes}
            events: List[ReplayEvent] = []
            new_patterns: List[Mapping[str, Any]] = []
            new_market_events: List[Mapping[str, Any]] = []
            new_trades: List[Mapping[str, Any]] = []

            for item in batch:
                events.append(item.event)
                if item.event.event_type == "CANDLE_CLOSED":
                    timeframe = str(item.event.timeframe)
                    new_candles[timeframe].append(item.candle)
                elif item.event.event_type == "PATTERN_RECOGNIZED":
                    new_patterns.append(item.event.payload)
                elif item.event.event_type == "MARKET_EVENT_RECOGNIZED":
                    new_market_events.append(item.event.payload)
                elif item.event.event_type in {
                    "TRADE_PREDICTION_AVAILABLE",
                    "TRADE_OPENED",
                    "TRADE_RESULT_KNOWN",
                }:
                    new_trades.append(item.event.payload)

            for timeframe, candles in new_candles.items():
                if candles:
                    histories[timeframe].extend(candles)
                    limit = self.config.max_history_bars_per_timeframe
                    if limit is not None and len(histories[timeframe]) > limit:
                        histories[timeframe] = histories[timeframe][-limit:]

            visible_patterns.extend(new_patterns)
            visible_market_events.extend(new_market_events)
            visible_trades.extend(new_trades)

            yield ReplayStep(
                market=self.market,
                current_time=current_time,
                events=tuple(events),
                new_candles_by_timeframe={
                    timeframe: tuple(values)
                    for timeframe, values in new_candles.items()
                },
                candles_by_timeframe={
                    timeframe: tuple(values)
                    for timeframe, values in histories.items()
                },
                known_pattern_occurrences=tuple(visible_patterns),
                known_market_events=tuple(visible_market_events),
                known_trade_records=tuple(visible_trades),
                candle_counts={
                    timeframe: len(values)
                    for timeframe, values in histories.items()
                },
                stage_id=self.stage_id,
            )

    def run(
        self,
        callback: Optional[Callable[[ReplayStep], Any]] = None,
    ) -> ReplaySummary:
        """Run the replay and optionally invoke a callback for every time step.

        Returning ``False`` from the callback stops replay after the current
        step. Callback exceptions are wrapped in ReplayCallbackError.
        """
        step_count = 0
        event_count = 0
        candle_count = 0
        pattern_count = 0
        market_event_count = 0
        trade_event_count = 0
        first_time: Optional[int] = None
        last_time: Optional[int] = None
        stopped_early = False

        for step in self.iter_steps():
            step_count += 1
            event_count += len(step.events)
            candle_count += step.new_candle_count
            pattern_count += sum(
                1 for event in step.events
                if event.event_type == "PATTERN_RECOGNIZED"
            )
            market_event_count += sum(
                1 for event in step.events
                if event.event_type == "MARKET_EVENT_RECOGNIZED"
            )
            trade_event_count += sum(
                1 for event in step.events
                if event.event_type.startswith("TRADE_")
            )
            first_time = step.current_time if first_time is None else first_time
            last_time = step.current_time

            if callback is not None:
                try:
                    keep_going = callback(step)
                except Exception as error:
                    raise ReplayCallbackError(
                        f"Replay callback failed at {step.current_time}: {error}"
                    ) from error
                if keep_going is False:
                    stopped_early = True
                    break

        return ReplaySummary(
            market=self.market,
            step_count=step_count,
            event_count=event_count,
            candle_count=candle_count,
            pattern_occurrence_count=pattern_count,
            market_event_count=market_event_count,
            trade_event_count=trade_event_count,
            first_available_at=first_time,
            last_available_at=last_time,
            stopped_early=stopped_early,
        )


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def iter_market_replay(
    market: str,
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    **kwargs: Any,
) -> Iterator[ReplayStep]:
    """Convenience iterator over a single market's historical replay."""
    engine = HistoricalReplay(market, candles_by_timeframe, **kwargs)
    return engine.iter_steps()


def replay_market_history(
    market: str,
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    *,
    callback: Optional[Callable[[ReplayStep], Any]] = None,
    **kwargs: Any,
) -> ReplaySummary:
    """Run one historical market replay and return its processing summary."""
    engine = HistoricalReplay(market, candles_by_timeframe, **kwargs)
    return engine.run(callback=callback)


def load_market_replay(
    market: str,
    *,
    timeframes: Optional[Sequence[str]] = None,
    pattern_occurrences: Iterable[Any] = (),
    market_events: Iterable[Any] = (),
    trade_records: Iterable[Any] = (),
    config: Optional[ReplayConfig] = None,
    stage: Optional[HistoryStage | HistoryStagePlan] = None,
    load_stored_events: bool = False,
    identity: Any = None,
) -> HistoricalReplay:
    """Load chart archives and optionally load previously stored events.

    Shared market events may be loaded automatically from storage. Customer
    trade results are loaded only when an exact TradingBot identity is supplied.
    """
    try:
        from engine.history import load_candles_for_timeframes
    except ImportError:
        from .history import load_candles_for_timeframes

    if config is not None and config.timeframes is not None:
        selected_timeframes = tuple(config.timeframes)
    else:
        selected_timeframes = tuple(timeframes or SUPPORTED_TIMEFRAMES)
    candles = load_candles_for_timeframes(
        market,
        selected_timeframes,
        identity=identity,
    )

    patterns = list(pattern_occurrences)
    observations = list(market_events)
    trades = list(trade_records)
    if load_stored_events:
        try:
            from engine.storage import (
                iter_customer_trade_results,
                iter_market_events,
                iter_pattern_occurrences,
            )
        except ImportError:
            from .storage import (
                iter_customer_trade_results,
                iter_market_events,
                iter_pattern_occurrences,
            )

        if not patterns:
            patterns = list(iter_pattern_occurrences(market))
        if not observations:
            observations = list(iter_market_events(market))
        if identity is not None and not trades:
            trades = list(iter_customer_trade_results(identity))

    resolved_config = config or ReplayConfig(timeframes=selected_timeframes)
    return HistoricalReplay(
        market,
        candles,
        pattern_occurrences=patterns,
        market_events=observations,
        trade_records=trades,
        config=resolved_config,
        stage=stage,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "ReplayError",
    "ReplayValidationError",
    "ReplayCallbackError",
    "ReplayConfig",
    "ReplayEvent",
    "ReplayStep",
    "ReplaySummary",
    "HistoricalReplay",
    "iter_market_replay",
    "replay_market_history",
    "load_market_replay",
]
