"""Chronological market and customer event timeline for Spectline."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from client import ClientIdentity, normalize_market
except ImportError:
    try:
        from ..client import ClientIdentity, normalize_market
    except ImportError:
        ClientIdentity = Any

        def normalize_market(value: str) -> str:
            normalized = str(value).strip().upper()
            if not normalized:
                raise ValueError("Market must not be empty.")
            return normalized


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

TIMELINE_SCHEMA_VERSION = 1
DEFAULT_PATTERN_TRADE_LOOKBACK_SECONDS = 0
DEFAULT_ORDER_BY = "event_time"
VALID_ORDER_MODES = frozenset({"event_time", "available_at"})

_EVENT_PRIORITY = {
    "PATTERN_OCCURRENCE": 10,
    "MARKET_EVENT": 20,
    "SETUP_RECORDED": 30,
    "TRADE_PREDICTION_AVAILABLE": 40,
    "TRADE_OPENED": 50,
    "TRADE_CLOSED": 60,
}

_IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class TimelineError(RuntimeError):
    """Base exception for timeline operations."""


class TimelineValidationError(TimelineError, ValueError):
    """Raised when a timeline event or configuration is invalid."""


class TimelineConflictError(TimelineError):
    """Raised when one event ID is associated with conflicting event data."""


@dataclass(frozen=True)
class TimelineConfig:
    """Immutable configuration for one chronological timeline view."""

    start_time: Optional[int] = None
    end_time_exclusive: Optional[int] = None
    as_of_time: Optional[int] = None
    order_by: str = DEFAULT_ORDER_BY
    window_field: str = "event_time"
    include_patterns: bool = True
    include_market_events: bool = True
    include_setups: bool = True
    include_open_trades: bool = True
    include_trade_lifecycle_events: bool = True
    build_pattern_trade_links: bool = True
    pattern_trade_lookback_seconds: int = DEFAULT_PATTERN_TRADE_LOOKBACK_SECONDS

    def __post_init__(self) -> None:
        start = _optional_timestamp(self.start_time, "start_time")
        end = _optional_timestamp(self.end_time_exclusive, "end_time_exclusive")
        as_of = _optional_timestamp(self.as_of_time, "as_of_time")

        if start is not None and end is not None and end <= start:
            raise TimelineValidationError(
                "end_time_exclusive must be later than start_time."
            )

        order_by = str(self.order_by).strip().lower()
        if order_by not in VALID_ORDER_MODES:
            raise TimelineValidationError(
                "order_by must be 'event_time' or 'available_at'."
            )

        window_field = str(self.window_field).strip().lower()
        if window_field not in {"event_time", "available_at"}:
            raise TimelineValidationError(
                "window_field must be 'event_time' or 'available_at'."
            )

        lookback = _non_negative_integer(
            self.pattern_trade_lookback_seconds,
            "pattern_trade_lookback_seconds",
        )

        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time_exclusive", end)
        object.__setattr__(self, "as_of_time", as_of)
        object.__setattr__(self, "order_by", order_by)
        object.__setattr__(self, "window_field", window_field)
        object.__setattr__(self, "pattern_trade_lookback_seconds", lookback)


@dataclass(frozen=True)
class TimelineEvent:
    """One market or customer event with separate occurrence and availability times."""

    event_id: str
    event_type: str
    market: str
    event_time: int
    available_at: int
    timeframe: Optional[str]
    source: str
    source_id: Optional[str]
    payload: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "market": self.market,
            "event_time": self.event_time,
            "available_at": self.available_at,
            "timeframe": self.timeframe,
            "source": self.source,
            "source_id": self.source_id,
            "payload": _json_safe(dict(self.payload)),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class PatternTradeLink:
    """A chronological relationship between one pattern occurrence and one setup."""

    id: int
    occurrence_id: str
    pattern_id: str
    relationship: str
    occurrence_start: int
    occurrence_end: int
    recognized_at: int
    trade_entry_time: int
    trade_end_time: Optional[int]
    available_at_entry: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class TimelineResult:
    """A stable, sorted timeline and optional trade-to-pattern relationships."""

    market: str
    events: Tuple[TimelineEvent, ...]
    pattern_trade_links: Tuple[PatternTradeLink, ...] = field(default_factory=tuple)
    start_time: Optional[int] = None
    end_time_exclusive: Optional[int] = None
    as_of_time: Optional[int] = None
    order_by: str = DEFAULT_ORDER_BY
    source_counts: Mapping[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def known_at(self, timestamp: Any) -> "TimelineResult":
        """Return only events that were available by the supplied UTC timestamp."""
        cutoff = normalize_timestamp(timestamp, "timestamp")
        visible = tuple(
            event for event in self.events if event.available_at <= cutoff
        )
        return TimelineResult(
            market=self.market,
            events=visible,
            pattern_trade_links=tuple(
                link
                for link in self.pattern_trade_links
                if link.trade_entry_time <= cutoff and link.recognized_at <= cutoff
            ),
            start_time=self.start_time,
            end_time_exclusive=self.end_time_exclusive,
            as_of_time=cutoff,
            order_by=self.order_by,
            source_counts=_count_sources(visible),
        )

    def between(
        self,
        start_time: Any,
        end_time_exclusive: Any,
        *,
        window_field: str = "event_time",
    ) -> "TimelineResult":
        """Return a half-open time slice using event time or availability time."""
        start = normalize_timestamp(start_time, "start_time")
        end = normalize_timestamp(end_time_exclusive, "end_time_exclusive")
        if end <= start:
            raise TimelineValidationError(
                "end_time_exclusive must be later than start_time."
            )
        field_name = str(window_field).strip().lower()
        if field_name not in {"event_time", "available_at"}:
            raise TimelineValidationError(
                "window_field must be 'event_time' or 'available_at'."
            )
        selected = tuple(
            event
            for event in self.events
            if start <= int(getattr(event, field_name)) < end
        )
        return TimelineResult(
            market=self.market,
            events=selected,
            pattern_trade_links=self.pattern_trade_links,
            start_time=start,
            end_time_exclusive=end,
            as_of_time=self.as_of_time,
            order_by=self.order_by,
            source_counts=_count_sources(selected),
        )

    def to_dict(self, include_payloads: bool = True) -> Dict[str, Any]:
        event_records = []
        for event in self.events:
            item = event.to_dict()
            if not include_payloads:
                item.pop("payload", None)
            event_records.append(item)
        return {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "market": self.market,
            "start_time": self.start_time,
            "end_time_exclusive": self.end_time_exclusive,
            "as_of_time": self.as_of_time,
            "order_by": self.order_by,
            "event_count": len(self.events),
            "source_counts": dict(self.source_counts),
            "events": event_records,
            "pattern_trade_links": [
                link.to_dict() for link in self.pattern_trade_links
            ],
        }


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _record_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "to_dict") and callable(record.to_dict):
        value = record.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    raise TimelineValidationError("Timeline source records must be mappings or objects.")


def normalize_timestamp(value: Any, field_name: str = "timestamp") -> int:
    """Normalize a UTC Unix-seconds timestamp or an aware datetime."""
    if isinstance(value, bool) or value is None:
        raise TimelineValidationError(f"{field_name} must be a valid UTC timestamp.")

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TimelineValidationError(
                f"{field_name} datetime must include a timezone."
            )
        return int(value.astimezone(timezone.utc).timestamp())

    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TimelineValidationError(
            f"{field_name} must be an integer Unix timestamp in seconds."
        ) from error

    if str(value).strip() not in {str(number), f"+{number}"}:
        raise TimelineValidationError(
            f"{field_name} must be an integer Unix timestamp in seconds."
        )
    if number < 0:
        raise TimelineValidationError(f"{field_name} cannot be negative.")
    return number


def _optional_timestamp(value: Any, field_name: str) -> Optional[int]:
    return None if value is None else normalize_timestamp(value, field_name)


def _non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TimelineValidationError(f"{field_name} must be a non-negative integer.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TimelineValidationError(
            f"{field_name} must be a non-negative integer."
        ) from error
    if number < 0 or str(value).strip() not in {str(number), f"+{number}"}:
        raise TimelineValidationError(f"{field_name} must be a non-negative integer.")
    return number


def _normalize_market(market: str) -> str:
    try:
        normalized = str(normalize_market(market)).strip()
    except Exception as error:
        raise TimelineValidationError(f"Invalid market: {market!r}.") from error
    if not normalized:
        raise TimelineValidationError("Market must not be empty.")
    return normalized


def _normalize_timeframe(value: Any) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    timeframe = str(value).strip().upper()
    if len(timeframe) > 16:
        raise TimelineValidationError("timeframe is too long.")
    return timeframe


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TimelineValidationError("Naive datetime values are not supported.")
        return int(value.astimezone(timezone.utc).timestamp())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _stable_hash(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _json_safe(dict(value)),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:24]


def _source_identifier(record: Mapping[str, Any], candidate_fields: Sequence[str]) -> Optional[str]:
    for field_name in candidate_fields:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None

def _setup_archive_id(record: Mapping[str, Any]) -> int:
    raw_id = record.get("id")

    if isinstance(raw_id, bool) or raw_id is None:
        raise TimelineValidationError("Setup record requires a positive integer id.")

    try:
        archive_id = int(raw_id)
    except (TypeError, ValueError, OverflowError) as error:
        raise TimelineValidationError(
            "Setup id must be a positive integer."
        ) from error

    if archive_id <= 0 or str(raw_id).strip() not in {
        str(archive_id),
        f"+{archive_id}",
    }:
        raise TimelineValidationError("Setup id must be a positive integer.")

    return archive_id


def _setup_entry_time(record: Mapping[str, Any]) -> int:
    candidates = [
        record.get("entry_time"),
        record.get("entry_candle_time"),
        record.get("decision_time"),
    ]

    entry_candle = record.get("entry_candle")
    if isinstance(entry_candle, Mapping):
        candidates.append(entry_candle.get("time"))
    else:
        candidates.append(entry_candle)

    geometry = record.get("trade_geometry")
    if isinstance(geometry, Mapping):
        entry_point = geometry.get("entry")
        if isinstance(entry_point, Mapping):
            candidates.append(entry_point.get("time"))

    candidates.append(record.get("breakout_time"))

    for candidate in candidates:
        if candidate is not None:
            return normalize_timestamp(candidate, "setup.entry_time")

    raise TimelineValidationError(
        "Setup record does not contain a valid entry time."
    )

def _event_identifier(
    source: str,
    event_type: str,
    source_id: Optional[str],
    payload: Mapping[str, Any],
) -> str:
    if source_id:
        return f"{source.upper()}:{source_id}:{event_type}"
    return f"{source.upper()}:{event_type}:{_stable_hash(payload)}"


def _make_event(
    *,
    market: str,
    event_type: str,
    event_time: Any,
    available_at: Any,
    source: str,
    source_id: Optional[str],
    timeframe: Any = None,
    payload: Optional[Mapping[str, Any]] = None,
    event_id: Optional[str] = None,
) -> TimelineEvent:
    normalized_event_time = normalize_timestamp(event_time, "event_time")
    normalized_available_at = normalize_timestamp(available_at, "available_at")
    if normalized_available_at < normalized_event_time:
        raise TimelineValidationError(
            "available_at cannot precede event_time in the chronological timeline."
        )

    normalized_type = str(event_type).strip().upper()
    if not normalized_type or len(normalized_type) > 64:
        raise TimelineValidationError("event_type must be a non-empty short string.")

    normalized_payload = _json_safe(dict(payload or {}))
    identifier = str(event_id).strip() if event_id is not None else ""
    if not identifier:
        identifier = _event_identifier(
            source,
            normalized_type,
            source_id,
            normalized_payload,
        )

    return TimelineEvent(
        event_id=identifier,
        event_type=normalized_type,
        market=market,
        event_time=normalized_event_time,
        available_at=normalized_available_at,
        timeframe=_normalize_timeframe(timeframe),
        source=str(source).strip().lower(),
        source_id=source_id,
        payload=normalized_payload,
        priority=int(_EVENT_PRIORITY.get(normalized_type, 100)),
    )


def _count_sources(events: Sequence[TimelineEvent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.source] = counts.get(event.source, 0) + 1
    return dict(sorted(counts.items()))


# ============================================================================
# SOURCE EVENT CONVERSION
# ============================================================================

def _pattern_event(market: str, raw_record: Any) -> TimelineEvent:
    record = _record_dict(raw_record)
    record_market = record.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise TimelineValidationError("Pattern market does not match timeline market.")

    occurrence_start = normalize_timestamp(
        record.get("occurrence_start"), "occurrence_start"
    )
    occurrence_end = normalize_timestamp(
        record.get("occurrence_end"), "occurrence_end"
    )
    recognized_at = normalize_timestamp(record.get("recognized_at"), "recognized_at")
    if occurrence_end < occurrence_start:
        raise TimelineValidationError("occurrence_end cannot precede occurrence_start.")
    if recognized_at < occurrence_start:
        raise TimelineValidationError("recognized_at cannot precede occurrence_start.")

    pattern_id = str(record.get("pattern_id", "")).strip()
    if len(pattern_id) != 4 or not pattern_id.isdigit():
        raise TimelineValidationError("Pattern occurrence requires a four-digit pattern_id.")

    timeframe = _normalize_timeframe(record.get("timeframe"))
    if timeframe is None:
        raise TimelineValidationError("Pattern occurrence requires a timeframe.")
    source_id = _source_identifier(record, ("occurrence_id", "record_id"))
    payload = {
        "pattern_id": pattern_id,
        "occurrence_start": occurrence_start,
        "occurrence_end": occurrence_end,
        "recognized_at": recognized_at,
        "timeframe": record.get("timeframe"),
        "catalog_version": record.get("catalog_version"),
        "direction": record.get("direction"),
        "status": record.get("status"),
        "geometry": record.get("geometry", {}),
        "features": record.get("features", {}),
    }
    return _make_event(
        market=market,
        event_type="PATTERN_OCCURRENCE",
        event_time=occurrence_start,
        available_at=recognized_at,
        source="pattern",
        source_id=source_id,
        timeframe=timeframe,
        payload=payload,
        event_id=record.get("timeline_event_id"),
    )


def _market_event(market: str, raw_record: Any) -> TimelineEvent:
    record = _record_dict(raw_record)
    record_market = record.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise TimelineValidationError("Market event market does not match timeline market.")

    event_time = normalize_timestamp(record.get("event_time"), "event_time")
    recognized_at = normalize_timestamp(
        record.get("recognized_at", event_time), "recognized_at"
    )
    if recognized_at < event_time:
        raise TimelineValidationError("recognized_at cannot precede event_time.")

    timeframe = _normalize_timeframe(record.get("timeframe"))
    if timeframe is None:
        raise TimelineValidationError("Market event requires a timeframe.")
    source_id = _source_identifier(record, ("event_id", "record_id"))
    payload = {
        "event_type": str(record.get("event_type", "")).strip().upper(),
        "event_time": event_time,
        "recognized_at": recognized_at,
        "timeframe": record.get("timeframe"),
        "features": record.get("features", {}),
    }
    return _make_event(
        market=market,
        event_type="MARKET_EVENT",
        event_time=event_time,
        available_at=recognized_at,
        source="market_event",
        source_id=source_id,
        timeframe=timeframe,
        payload=payload,
        event_id=record.get("timeline_event_id"),
    )


def _setup_event(market: str, raw_record: Any) -> TimelineEvent:
    record = _record_dict(raw_record)

    record_market = record.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise TimelineValidationError("Setup market does not match timeline market.")

    archive_id = _setup_archive_id(record)
    entry_time = _setup_entry_time(record)

    safe_fields = (
        "direction",
        "setup_timeframe",
        "entry_timeframe",
        "zones",
        "trade_geometry",
        "anchor_price",
        "breakout",
        "entry",
        "sl",
        "tp",
        "rr_mode",
        "actual_outcome",
        "end_time",
    )

    payload = {
        "id": archive_id,
        **{
            key: record[key]
            for key in safe_fields
            if key in record
        },
    }
    payload["entry_time"] = entry_time

    return _make_event(
        market=market,
        event_type="SETUP_RECORDED",
        event_time=entry_time,
        available_at=entry_time,
        source="setup",
        source_id=str(archive_id),
        timeframe=record.get("entry_timeframe") or record.get("setup_timeframe"),
        payload=payload,
        event_id=f"SETUP:{archive_id}:RECORDED",
    )

def _trade_events(
    market: str,
    raw_record: Any,
    include_open_trade: bool,
) -> List[TimelineEvent]:
    record = _record_dict(raw_record)

    record_market = record.get("market")
    if record_market is not None and _normalize_market(str(record_market)) != market:
        raise TimelineValidationError("Setup market does not match timeline market.")

    archive_id = _setup_archive_id(record)
    entry_time = _setup_entry_time(record)

    actual_outcome = record.get("actual_outcome")
    if actual_outcome not in {None, "TP", "SL", "AMBIGUOUS"}:
        raise TimelineValidationError(
            "actual_outcome must be TP, SL, AMBIGUOUS or None."
        )

    end_time_raw = record.get("end_time")

    if (end_time_raw is None) != (actual_outcome is None):
        raise TimelineValidationError(
            "A finalized setup must contain both end_time and actual_outcome."
        )

    events: List[TimelineEvent] = []

    prediction = record.get("predicted_probability_at_entry")
    if prediction is not None:
        if isinstance(prediction, bool):
            raise TimelineValidationError(
                "predicted_probability_at_entry must be between zero and one."
            )

        try:
            probability = float(prediction)
        except (TypeError, ValueError, OverflowError) as error:
            raise TimelineValidationError(
                "predicted_probability_at_entry must be numeric."
            ) from error

        if not 0.0 <= probability <= 1.0:
            raise TimelineValidationError(
                "predicted_probability_at_entry must be between zero and one."
            )

        predicted_at = normalize_timestamp(
            record.get("predicted_at", entry_time),
            "predicted_at",
        )
        if predicted_at > entry_time:
            raise TimelineValidationError(
                "predicted_at cannot be later than entry_time."
            )

        events.append(
            _make_event(
                market=market,
                event_type="TRADE_PREDICTION_AVAILABLE",
                event_time=predicted_at,
                available_at=predicted_at,
                source="setup",
                source_id=str(archive_id),
                payload={
                    "id": archive_id,
                    "predicted_at": predicted_at,
                    "predicted_probability_at_entry": probability,
                    "probability_target": "TP_BEFORE_SL",
                },
                event_id=f"SETUP:{archive_id}:PREDICTION",
            )
        )

    if end_time_raw is None and not include_open_trade:
        return events

    opening_fields = (
        "direction",
        "entry",
        "sl",
        "tp",
        "setup_timeframe",
        "entry_timeframe",
        "zones",
        "trade_geometry",
    )

    opening_payload = {
        key: record[key]
        for key in opening_fields
        if key in record
    }
    opening_payload["id"] = archive_id
    opening_payload["entry_time"] = entry_time

    events.append(
        _make_event(
            market=market,
            event_type="TRADE_OPENED",
            event_time=entry_time,
            available_at=entry_time,
            source="setup",
            source_id=str(archive_id),
            timeframe=record.get("entry_timeframe"),
            payload=opening_payload,
            event_id=f"SETUP:{archive_id}:OPENED",
        )
    )

    if end_time_raw is None:
        return events

    end_time = normalize_timestamp(end_time_raw, "setup.end_time")
    if end_time < entry_time:
        raise TimelineValidationError("end_time cannot precede entry_time.")

    close_payload = {
        "id": archive_id,
        "entry_time": entry_time,
        "end_time": end_time,
        "actual_outcome": actual_outcome,
    }

    events.append(
        _make_event(
            market=market,
            event_type="TRADE_CLOSED",
            event_time=end_time,
            available_at=end_time,
            source="setup",
            source_id=str(archive_id),
            timeframe=record.get("entry_timeframe"),
            payload=close_payload,
            event_id=f"SETUP:{archive_id}:CLOSED",
        )
    )

    return events

# ============================================================================
# LINKING AND ORDERING
# ============================================================================

def link_patterns_to_trades(
    pattern_occurrences: Iterable[Any],
    setup_records: Iterable[Any],
    *,
    pre_entry_lookback_seconds: int = DEFAULT_PATTERN_TRADE_LOOKBACK_SECONDS,
) -> Tuple[PatternTradeLink, ...]:
    """Link market pattern occurrences to the current setup archive IDs."""

    lookback = _non_negative_integer(
        pre_entry_lookback_seconds,
        "pre_entry_lookback_seconds",
    )

    patterns = [_record_dict(item) for item in pattern_occurrences]
    setups = [_record_dict(item) for item in setup_records]
    links: List[PatternTradeLink] = []

    for setup in setups:
        archive_id = _setup_archive_id(setup)
        entry_time = _setup_entry_time(setup)

        end_raw = setup.get("end_time")
        end_time = (
            normalize_timestamp(end_raw, "setup.end_time")
            if end_raw is not None
            else None
        )

        if end_time is not None and end_time < entry_time:
            raise TimelineValidationError("Setup end_time cannot precede entry_time.")

        comparison_end = end_time if end_time is not None else entry_time

        for pattern in patterns:
            pattern_market = pattern.get("market")
            setup_market = setup.get("market")

            if (
                pattern_market is not None
                and setup_market is not None
                and _normalize_market(str(pattern_market))
                != _normalize_market(str(setup_market))
            ):
                continue

            occurrence_start = normalize_timestamp(
                pattern.get("occurrence_start"),
                "pattern.occurrence_start",
            )
            occurrence_end = normalize_timestamp(
                pattern.get("occurrence_end"),
                "pattern.occurrence_end",
            )
            recognized_at = normalize_timestamp(
                pattern.get("recognized_at"),
                "pattern.recognized_at",
            )

            if occurrence_end < occurrence_start:
                raise TimelineValidationError(
                    "Pattern occurrence has an invalid time range."
                )
            if recognized_at < occurrence_start:
                raise TimelineValidationError(
                    "Pattern recognized_at cannot precede occurrence_start."
                )

            pattern_id = str(pattern.get("pattern_id", "")).strip()
            if len(pattern_id) != 4 or not pattern_id.isdigit():
                raise TimelineValidationError(
                    "Pattern occurrence requires a four-digit pattern_id."
                )

            raw_linked_ids = setup.get("linked_pattern_ids", ())
            if raw_linked_ids is None:
                raw_linked_ids = ()
            if isinstance(raw_linked_ids, (str, bytes)):
                raw_linked_ids = (raw_linked_ids,)

            explicitly_linked = pattern_id in {
                str(value).strip() for value in raw_linked_ids
            }

            overlaps_setup = (
                occurrence_start <= comparison_end
                and occurrence_end >= entry_time
            )
            is_recent_pre_entry = (
                lookback > 0
                and occurrence_end < entry_time
                and occurrence_end >= entry_time - lookback
            )

            if not overlaps_setup and not is_recent_pre_entry and not explicitly_linked:
                continue

            available_at_entry = recognized_at <= entry_time

            if available_at_entry:
                relationship = "KNOWN_AT_ENTRY"
            elif occurrence_start < entry_time:
                relationship = "FORMED_BEFORE_ENTRY_RECOGNIZED_DURING_TRADE"
            else:
                relationship = "OCCURRED_DURING_TRADE"

            occurrence_id = _source_identifier(
                pattern,
                ("occurrence_id", "record_id"),
            ) or _stable_hash(pattern)

            links.append(
                PatternTradeLink(
                    id=archive_id,
                    occurrence_id=occurrence_id,
                    pattern_id=pattern_id,
                    relationship=relationship,
                    occurrence_start=occurrence_start,
                    occurrence_end=occurrence_end,
                    recognized_at=recognized_at,
                    trade_entry_time=entry_time,
                    trade_end_time=end_time,
                    available_at_entry=available_at_entry,
                )
            )

    links.sort(
        key=lambda item: (
            item.trade_entry_time,
            item.occurrence_start,
            item.recognized_at,
            item.id,
            item.pattern_id,
        )
    )

    return tuple(links)

def _sort_events(events: Iterable[TimelineEvent], order_by: str) -> Tuple[TimelineEvent, ...]:
    if order_by == "available_at":
        key = lambda event: (
            event.available_at,
            event.priority,
            event.event_time,
            event.event_id,
        )
    else:
        key = lambda event: (
            event.event_time,
            event.priority,
            event.available_at,
            event.event_id,
        )
    return tuple(sorted(events, key=key))


def _deduplicate_events(events: Iterable[TimelineEvent]) -> Tuple[TimelineEvent, ...]:
    by_id: Dict[str, TimelineEvent] = {}
    for event in events:
        existing = by_id.get(event.event_id)
        if existing is None:
            by_id[event.event_id] = event
            continue
        if existing.to_dict() != event.to_dict():
            raise TimelineConflictError(
                f"Conflicting timeline records share event_id {event.event_id!r}."
            )
    return tuple(by_id.values())


def _apply_config_window(
    events: Iterable[TimelineEvent],
    config: TimelineConfig,
) -> Tuple[TimelineEvent, ...]:
    selected: List[TimelineEvent] = []
    for event in events:
        window_time = (
            event.event_time if config.window_field == "event_time" else event.available_at
        )
        if config.start_time is not None and window_time < config.start_time:
            continue
        if config.end_time_exclusive is not None and window_time >= config.end_time_exclusive:
            continue
        if config.as_of_time is not None and event.available_at > config.as_of_time:
            continue
        selected.append(event)
    return _sort_events(selected, config.order_by)


# ============================================================================
# TIMELINE CONSTRUCTION
# ============================================================================

def build_timeline(
    market: str,
    *,
    pattern_occurrences: Iterable[Any] = (),
    market_events: Iterable[Any] = (),
    setup_records: Iterable[Any] = (),
    config: Optional[TimelineConfig] = None,
) -> TimelineResult:
    """Build a timeline from market memory and the current setup archive."""

    normalized_market = _normalize_market(market)
    resolved_config = config or TimelineConfig()

    if not isinstance(resolved_config, TimelineConfig):
        raise TypeError("config must be a TimelineConfig instance.")

    patterns = (
        list(pattern_occurrences)
        if resolved_config.include_patterns
        else []
    )
    observations = (
        list(market_events)
        if resolved_config.include_market_events
        else []
    )
    setups = (
        list(setup_records)
        if resolved_config.include_setups
        or resolved_config.include_trade_lifecycle_events
        or resolved_config.build_pattern_trade_links
        else []
    )

    events: List[TimelineEvent] = []

    for item in patterns:
        events.append(_pattern_event(normalized_market, item))

    for item in observations:
        events.append(_market_event(normalized_market, item))

    for item in setups:
        if resolved_config.include_setups:
            events.append(_setup_event(normalized_market, item))

        if resolved_config.include_trade_lifecycle_events:
            events.extend(
                _trade_events(
                    normalized_market,
                    item,
                    include_open_trade=resolved_config.include_open_trades,
                )
            )

    unique_events = _deduplicate_events(events)
    selected_events = _apply_config_window(
        unique_events,
        resolved_config,
    )

    links: Tuple[PatternTradeLink, ...] = tuple()

    if resolved_config.build_pattern_trade_links and patterns and setups:
        links = link_patterns_to_trades(
            patterns,
            setups,
            pre_entry_lookback_seconds=resolved_config.pattern_trade_lookback_seconds,
        )

        if resolved_config.as_of_time is not None:
            links = tuple(
                link
                for link in links
                if link.recognized_at <= resolved_config.as_of_time
            )

    return TimelineResult(
        market=normalized_market,
        events=selected_events,
        pattern_trade_links=links,
        start_time=resolved_config.start_time,
        end_time_exclusive=resolved_config.end_time_exclusive,
        as_of_time=resolved_config.as_of_time,
        order_by=resolved_config.order_by,
        source_counts=_count_sources(selected_events),
    )

# ============================================================================
# STORAGE INTEGRATION
# ============================================================================

def _storage_method(storage: Any, method_name: str):
    method = getattr(storage, method_name, None)
    if not callable(method):
        raise TypeError(f"storage must provide {method_name}().")
    return method


def build_timeline_from_storage(
    market: str,
    *,
    identity: Any = None,
    timeframe: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time_exclusive: Optional[int] = None,
    as_of_time: Optional[int] = None,
    order_by: str = DEFAULT_ORDER_BY,
    storage: Any = None,
    include_patterns: bool = True,
    include_market_events: bool = True,
    include_setups: bool = True,
    include_trade_lifecycle_events: bool = True,
    include_open_trades: bool = False,
    pattern_trade_lookback_seconds: int = DEFAULT_PATTERN_TRADE_LOOKBACK_SECONDS,
) -> TimelineResult:
    """Build the timeline from market memory and the authoritative setup archive."""

    normalized_market = _normalize_market(market)

    if identity is not None:
        identity_market = _read_field(identity, "market")
        if (
            identity_market is None
            or _normalize_market(str(identity_market)) != normalized_market
        ):
            raise TimelineValidationError(
                "The supplied TradingBot identity must belong to the requested market."
            )

    if storage is None:
        try:
            from engine.storage import HistoricalStorage
        except ImportError:
            from .storage import HistoricalStorage

        storage_service = HistoricalStorage()
    else:
        storage_service = storage

    pattern_records: List[Dict[str, Any]] = []
    market_event_records: List[Dict[str, Any]] = []
    setup_records: List[Dict[str, Any]] = []

    query_end = (
        None
        if end_time_exclusive is None
        else max(int(end_time_exclusive) - 1, 0)
    )

    if include_patterns:
        iterator = _storage_method(
            storage_service,
            "iter_pattern_occurrences",
        )
        pattern_records = list(
            iterator(
                normalized_market,
                timeframe=timeframe,
                start_time=start_time,
                end_time=query_end,
            )
        )

    if include_market_events:
        iterator = _storage_method(
            storage_service,
            "iter_market_events",
        )
        market_event_records = list(
            iterator(
                normalized_market,
                timeframe=timeframe,
                start_time=start_time,
                end_time=query_end,
            )
        )

    if identity is not None:
        try:
            from setups import load_learning_setups
        except ImportError:
            from ..setups import load_learning_setups

        setup_records = load_learning_setups(identity)

    config = TimelineConfig(
        start_time=start_time,
        end_time_exclusive=end_time_exclusive,
        as_of_time=as_of_time,
        order_by=order_by,
        include_patterns=include_patterns,
        include_market_events=include_market_events,
        include_setups=include_setups,
        include_trade_lifecycle_events=include_trade_lifecycle_events,
        include_open_trades=include_open_trades,
        pattern_trade_lookback_seconds=pattern_trade_lookback_seconds,
    )

    return build_timeline(
        normalized_market,
        pattern_occurrences=pattern_records,
        market_events=market_event_records,
        setup_records=setup_records,
        config=config,
    )

# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "TIMELINE_SCHEMA_VERSION",
    "TimelineError",
    "TimelineValidationError",
    "TimelineConflictError",
    "TimelineConfig",
    "TimelineEvent",
    "PatternTradeLink",
    "TimelineResult",
    "normalize_timestamp",
    "link_patterns_to_trades",
    "build_timeline",
    "build_timeline_from_storage",
]
