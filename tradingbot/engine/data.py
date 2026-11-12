"""Identity-isolated Learning setup normalization and point-in-time data assembly."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import importlib
import inspect
import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from client import (
        ClientIdentity,
        get_learning_setups_path,
        normalize_market,
        resolve_identity,
    )
    from engine.settings import (
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )
except ImportError:
    from ..client import (
        ClientIdentity,
        get_learning_setups_path,
        normalize_market,
        resolve_identity,
    )
    from .settings import (
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

DATA_BUNDLE_SCHEMA_VERSION = 2
_SAFE_FOLDER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALID_ZONE_TYPES = frozenset({"SUPPORT", "RESISTANCE"})
_VALID_OUTCOMES = frozenset({"TP", "SL"})
RESOLUTION_TIMEFRAME_FIELD = "resolution_timeframe"
PREDICTION_TIME_FIELD = "predicted_at"
PREDICTED_PROBABILITY_FIELD = "predicted_probability_at_entry"
ACTUAL_OUTCOME_FIELD = "actual_outcome"
END_TIME_FIELD = "end_time"
DEFAULT_POINT_IN_TIME_CANDLE_LIMIT = 240

_PATTERN_INPUT_FIELDS = (
    "occurrence_id",
    "record_id",
    "market",
    "timeframe",
    "pattern_id",
    "variant_id",
    "occurrence_start",
    "occurrence_end",
    "recognized_at",
    "catalog_version",
    "geometry",
    "features",
)

_MARKET_EVENT_INPUT_FIELDS = (
    "event_id",
    "record_id",
    "market",
    "timeframe",
    "event_type",
    "event_time",
    "recognized_at",
    "features",
)

# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class DataBundleError(RuntimeError):
    """Raised when a requested data bundle cannot be assembled safely."""


@dataclass(frozen=True)
class Candle:
    """Normalized OHLCV candle with an opening-time Unix timestamp."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation of the candle."""
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class ManualZone:
    """One user-drawn Support or Resistance zone from a Learning setup."""
 
    zone_id: Optional[Any]
    zone_type: str
    start_time: Optional[int]
    end_time: Optional[int]
    low: Optional[float]
    high: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class UserSetup:
    """Normalized Learning setup with prediction metadata and labels kept separate."""

    id: int
    direction: str
    setup_timeframe: str
    entry_timeframe: str
    anchor_time: Optional[int]
    anchor_price: Optional[float]
    breakout_time: Optional[int]
    breakout: Optional[float]
    entry_candle_time: Optional[int]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    rr_mode: Optional[str]
    resolution_timeframe: Optional[str]
    zones: List[ManualZone]
    trade_geometry: Dict[str, Any]
    raw: Dict[str, Any]
    validation_errors: List[str] = field(default_factory=list)
    prediction_time: Optional[int] = None
    predicted_probability_at_entry: Optional[float] = None
    end_time: Optional[int] = None
    actual_outcome: Optional[str] = None
    outcome_status: str = "UNKNOWN"
    label_validation_errors: List[str] = field(default_factory=list)
    trade_id: Optional[str] = None

    @property
    def decision_time(self) -> Optional[int]:
        """Return the point-in-time cutoff used for entry prediction features."""
        return (
            self.prediction_time
            if self.prediction_time is not None
            else self.entry_candle_time
        )

    @property
    def entry_time(self) -> Optional[int]:
        """Return the recorded trade entry time."""
        return self.entry_candle_time

    @property
    def is_valid(self) -> bool:
        """Return whether the setup contains valid input fields for alignment."""
        return not self.validation_errors

    @property
    def is_open(self) -> bool:
        return self.end_time is None

    @property
    def is_closed(self) -> bool:
        return self.end_time is not None

    @property
    def has_known_outcome(self) -> bool:
        return (
            self.actual_outcome in {"TP", "SL"}
            and self.end_time is not None
            and not self.label_validation_errors
        )


# ============================================================================
# IDENTITY AND PATH HELPERS
# ============================================================================

def safe_market_folder(market: str) -> str:
    """Return the canonical market folder name."""
    normalized = str(normalize_market(market)).strip()
    if not normalized or not _SAFE_FOLDER_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid market folder name.")
    return normalized


def safe_user_folder(user_id: Optional[str]) -> str:
    """Validate a customer identifier before it is used in a path."""
    if user_id is None:
        raise ValueError("user_id is required in WebMode.")

    value = str(user_id).strip()
    if not value or not _SAFE_FOLDER_PATTERN.fullmatch(value):
        raise ValueError("Invalid user_id.")
    return value


def _identity_to_dict(identity: ClientIdentity) -> Dict[str, str]:
    """Return the four identity dimensions in a stable dictionary."""
    if hasattr(identity, "as_dict") and callable(identity.as_dict):
        payload = identity.as_dict()
    else:
        payload = {
            "user_id": getattr(identity, "user_id", None),
            "market": getattr(identity, "market", None),
            "strategy_id": getattr(identity, "strategy_id", None),
            "tradingbot_id": getattr(identity, "tradingbot_id", None),
        }

    required_fields = ("user_id", "market", "strategy_id", "tradingbot_id")
    normalized: Dict[str, str] = {}
    for field_name in required_fields:
        value = payload.get(field_name)
        if value is None or not str(value).strip():
            raise ValueError(f"Identity is missing {field_name}.")
        normalized[field_name] = str(value).strip()
    normalized["market"] = safe_market_folder(normalized["market"])
    return normalized


def _resolve_ai_identity(
    market: str,
    identity: Optional[ClientIdentity] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Resolve and validate the exact customer, market, strategy and bot identity."""
    normalized_market = normalize_market(market)

    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        identity_market = normalize_market(identity.market)
        if identity_market != normalized_market:
            raise ValueError(
                f"Identity market {identity_market} does not match "
                f"requested market {normalized_market}."
            )
        _identity_to_dict(identity)
        return identity

    resolved = resolve_identity(
        market=normalized_market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    _identity_to_dict(resolved)
    return resolved


def setup_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the Learning setup JSON path for one exact TradingBot identity."""
    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return Path(get_learning_setups_path(resolved))


# ============================================================================
# VALUE AND TIMESTAMP NORMALIZATION
# ============================================================================

def to_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float, returning None for invalid values."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert Unix, millisecond, datetime or ISO-8601 values into UTC seconds."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        try:
            return int(parsed.astimezone(timezone.utc).timestamp())
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if abs(number) > 10_000_000_000:
            number /= 1000.0
        return int(number)

    text = str(value).strip()
    if not text:
        return None

    numeric_text = text.replace(",", "")
    try:
        number = float(numeric_text)
        if math.isfinite(number):
            if abs(number) > 10_000_000_000:
                number /= 1000.0
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass

    normalized = text.replace(" UTC", "+00:00")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.astimezone(timezone.utc).timestamp())
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _strip_legacy_focus_fields(value: Any) -> Any:
    """Remove retired Focus 1/2 fields from a deep in-memory copy."""
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key.startswith(("focus1", "focus2")):
                continue
            cleaned[key] = _strip_legacy_focus_fields(child)
        return cleaned

    if isinstance(value, list):
        return [_strip_legacy_focus_fields(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_strip_legacy_focus_fields(item) for item in value)

    return copy.deepcopy(value)


def _parse_id(value: Any, fallback: int) -> Tuple[int, bool]:
    """Parse a stable integer setup ID and report whether the value was valid."""
    if isinstance(value, bool):
        return fallback, False

    if isinstance(value, str) and not value.strip().isdigit():
        return fallback, False

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback, False

    if isinstance(value, float) and not value.is_integer():
        return fallback, False
    if parsed <= 0:
        return fallback, False
    return parsed, True


def _normalize_timeframe(value: Any, field_name: str, errors: List[str]) -> str:
    """Normalize a timeframe and record values unsupported in the current mode."""
    normalized = str(value or "").strip().upper()
    if not normalized:
        errors.append(f"Missing {field_name}.")
        return ""

    if normalized not in SUPPORTED_TIMEFRAMES:
        errors.append(
            f"Unsupported {field_name} '{normalized}' for the current runtime mode."
        )
    return normalized


def _first_timestamp(source: Mapping[str, Any], field_names: Sequence[str]) -> Optional[int]:
    """Return the first valid timestamp from an ordered set of object fields."""
    for field_name in field_names:
        timestamp = parse_timestamp(source.get(field_name))
        if timestamp is not None:
            return timestamp
    return None


def _normalize_probability(value: Any) -> Optional[float]:
    """Normalize a probability supplied as a fraction or a percentage."""
    number = to_float(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _normalize_outcome(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        normalized = value.strip().upper()

        aliases = {
            "TP": "TP",
            "TP_FIRST": "TP",
            "TP BEFORE SL": "TP",
            "TP_BEFORE_SL": "TP",
            "SL": "SL",
            "SL_FIRST": "SL",
            "SL BEFORE TP": "SL",
            "SL_BEFORE_TP": "SL",
            "AMBIGUOUS": None,
            "UNRESOLVED": None,
            "NULL": None,
            "NONE": None,
        }

        return aliases.get(normalized)

    if isinstance(value, (int, float)) and value in (0, 1):
        return "TP" if value == 1 else "SL"

    return None


def _normalize_trade_status(value: Any) -> str:
    """Normalize a trade lifecycle status without inferring a binary label."""
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if "AMBIGUOUS" in normalized or "UNRESOLVED" in normalized:
        return "AMBIGUOUS"
    if normalized in {"CLOSED", "CLOSE", "EXITED", "FINISHED"}:
        return "CLOSED_UNLABELLED"
    if normalized in {"OPEN", "ACTIVE", "PENDING"}:
        return "OPEN"
    return "UNKNOWN"


# ============================================================================
# CANDLE NORMALIZATION AND POINT-IN-TIME SLICING
# ============================================================================

def _read_object_field(value: Any, field_name: str, default: Any = None) -> Any:
    """Read one field from either a mapping or an attribute-based object."""
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _normalize_candle(value: Any) -> Optional[Candle]:
    """Convert a mapping or Candle-like object into a validated candle."""
    timestamp = parse_timestamp(_read_object_field(value, "time"))
    open_value = to_float(_read_object_field(value, "open"))
    high_value = to_float(_read_object_field(value, "high"))
    low_value = to_float(_read_object_field(value, "low"))
    close_value = to_float(_read_object_field(value, "close"))
    volume_value = to_float(_read_object_field(value, "volume", 0.0))

    if None in (timestamp, open_value, high_value, low_value, close_value):
        return None
    if high_value < max(open_value, close_value, low_value):
        return None
    if low_value > min(open_value, close_value, high_value):
        return None
    if volume_value is None or volume_value < 0.0:
        volume_value = 0.0

    return Candle(
        time=int(timestamp),
        open=float(open_value),
        high=float(high_value),
        low=float(low_value),
        close=float(close_value),
        volume=float(volume_value),
    )


def normalize_candles(
    candles: Iterable[Any],
) -> Tuple[List[Candle], Dict[str, int]]:
    """Normalize, sort and de-duplicate candle rows by their opening timestamp."""
    candles_by_time: Dict[int, Candle] = {}
    invalid_count = 0
    duplicate_count = 0

    for raw_candle in candles or []:
        candle = _normalize_candle(raw_candle)
        if candle is None:
            invalid_count += 1
            continue
        if candle.time in candles_by_time:
            duplicate_count += 1
        candles_by_time[candle.time] = candle

    ordered = [candles_by_time[timestamp] for timestamp in sorted(candles_by_time)]
    return ordered, {
        "invalid_candle_count": invalid_count,
        "duplicate_candle_count": duplicate_count,
    }


def candles_available_at(
    candles: Sequence[Candle],
    timeframe: str,
    decision_time: Optional[int],
    *,
    max_count: Optional[int] = DEFAULT_POINT_IN_TIME_CANDLE_LIMIT,
) -> List[Candle]:
    """Return only candles fully closed by the decision timestamp."""
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {normalized_timeframe}.")
    if decision_time is None:
        return []

    duration = int(TIMEFRAME_SECONDS[normalized_timeframe])
    ordered, _ = normalize_candles(candles)
    close_times = [candle.time + duration for candle in ordered]
    last_index = bisect_right(close_times, int(decision_time))
    if last_index <= 0:
        return []

    selected = ordered[:last_index]
    if max_count is not None:
        count = int(max_count)
        if count <= 0:
            return []
        selected = selected[-count:]
    return selected


def _candle_coverage(candles: Sequence[Candle]) -> Dict[str, Any]:
    """Return the first and last opening timestamps in a normalized archive."""
    if not candles:
        return {
            "candle_count": 0,
            "first_candle_time": None,
            "last_candle_time": None,
        }
    return {
        "candle_count": len(candles),
        "first_candle_time": candles[0].time,
        "last_candle_time": candles[-1].time,
    }


# ============================================================================
# LEARNING SETUP NORMALIZATION
# ============================================================================

def _geometry_item(
    trade_geometry: Mapping[str, Any],
    key: str,
) -> Dict[str, Any]:
    """Return one named trade-geometry object when it is a mapping."""
    value = trade_geometry.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _first_price(
    item: Mapping[str, Any],
    geometry: Mapping[str, Any],
    field_name: str,
) -> Optional[float]:
    """Read a trade price from the setup root, then its geometry object."""
    value = to_float(item.get(field_name))
    if value is not None:
        return value
    return to_float(geometry.get("price"))


def _normalize_manual_zone(
    id: int,
    raw_zone: Mapping[str, Any],
    zone_index: int,
) -> ManualZone:
    """Normalize one user-drawn Support or Resistance zone."""
    errors: List[str] = []
    zone_type = str(raw_zone.get("type", raw_zone.get("zone_type", ""))).strip().upper()

    if zone_type not in _VALID_ZONE_TYPES:
        errors.append(
            f"Zone {zone_index} has unsupported type '{zone_type or 'EMPTY'}'."
        )

    geometry_value = raw_zone.get("geometry")
    geometry = dict(geometry_value) if isinstance(geometry_value, Mapping) else dict(raw_zone)

    start_time = _first_timestamp(
        geometry,
        ("start_time", "start", "from_time", "time_start"),
    )
    end_time = _first_timestamp(
        geometry,
        ("end_time", "end", "to_time", "time_end"),
    )
    low = to_float(geometry.get("low", geometry.get("price_low")))
    high = to_float(geometry.get("high", geometry.get("price_high")))

    if low is None or high is None:
        errors.append(f"Zone {zone_index} is missing valid low/high prices.")
    elif low > high:
        errors.append(f"Zone {zone_index} has low greater than high.")

    if start_time is None or end_time is None:
        errors.append(f"Zone {zone_index} is missing valid start/end times.")
    elif start_time > end_time:
        errors.append(f"Zone {zone_index} has start_time after end_time.")

    return ManualZone(
        zone_id=raw_zone.get("id", raw_zone.get("number", zone_index)),
        zone_type=zone_type,
        start_time=start_time,
        end_time=end_time,
        low=low,
        high=high,
        raw=_strip_legacy_focus_fields(dict(raw_zone)),
        validation_errors=errors,
    )


def _extract_end_time(item: Mapping[str, Any]) -> Optional[int]:
    return _first_timestamp(item, (END_TIME_FIELD,))

def normalize_learning_setup(
    item: Dict[str, Any],
    index: int,
) -> UserSetup:
    """Convert one raw JSON setup into a normalized record with isolated labels."""
    if not isinstance(item, Mapping):
        raise TypeError(f"Learning setup at index {index} must be an object.")

    errors: List[str] = []
    label_errors: List[str] = []

    id, valid_id = _parse_id(item.get("id", index), index)
    if not valid_id:
        errors.append(f"Setup at index {index} has an invalid id.")

    direction_value = str(item.get("direction", "")).strip().upper()
    if direction_value in {"BUY", "LONG"}:
        direction = "LONG"
    elif direction_value in {"SELL", "SHORT"}:
        direction = "SHORT"
    else:
        direction = direction_value
        errors.append(
            f"Setup {id} has an unsupported direction '{direction or 'EMPTY'}'."
        )

    setup_timeframe = _normalize_timeframe(
        item.get("setup_timeframe"),
        "setup_timeframe",
        errors,
    )
    entry_timeframe = _normalize_timeframe(
        item.get("entry_timeframe"),
        "entry_timeframe",
        errors,
    )

    anchor_value = item.get("anchor")
    anchor = dict(anchor_value) if isinstance(anchor_value, Mapping) else {}

    geometry_value = item.get("trade_geometry")
    trade_geometry = dict(geometry_value) if isinstance(geometry_value, Mapping) else {}
    breakout_geometry = _geometry_item(trade_geometry, "breakout")
    entry_geometry = _geometry_item(trade_geometry, "entry")
    sl_geometry = _geometry_item(trade_geometry, "sl")
    tp_geometry = _geometry_item(trade_geometry, "tp")

    entry_candle_time = _first_timestamp(
        item,
        ("entry_candle", "entry_candle_time", "entry_time", "opened_at"),
    )
    if entry_candle_time is None:
        entry_candle_time = _first_timestamp(
            entry_geometry,
            ("time", "timestamp", "candle_time"),
        )

    if entry_candle_time is None:
        errors.append(
            f"Setup {id} is missing a valid entry timestamp. "
            "Anchor time is not used as a substitute for entry time."
        )

    anchor_time = _first_timestamp(item, ("anchor_time",))
    if anchor_time is None:
        anchor_time = _first_timestamp(anchor, ("time", "timestamp"))

    breakout_time = _first_timestamp(item, ("breakout_time",))
    if breakout_time is None:
        breakout_time = _first_timestamp(
            breakout_geometry,
            ("time", "timestamp", "candle_time"),
        )

    breakout = _first_price(item, breakout_geometry, "breakout")
    entry = _first_price(item, entry_geometry, "entry")
    sl = _first_price(item, sl_geometry, "sl")
    tp = _first_price(item, tp_geometry, "tp")

    for field_name, field_value in (("entry", entry), ("sl", sl), ("tp", tp)):
        if field_value is None:
            errors.append(f"Setup {id} is missing a valid {field_name} price.")

    zones_value = item.get("zones", [])
    if zones_value is None:
        zones_value = []
    if not isinstance(zones_value, list):
        errors.append(f"Setup {id} has a zones value that is not a list.")
        zones_value = []

    zones: List[ManualZone] = []
    for zone_index, raw_zone in enumerate(zones_value, start=1):
        if not isinstance(raw_zone, Mapping):
            errors.append(f"Setup {id} zone {zone_index} is not an object.")
            continue
        zone = _normalize_manual_zone(id, raw_zone, zone_index)
        zones.append(zone)
        errors.extend(
            f"Setup {id}: {message}"
            for message in zone.validation_errors
        )

    resolution_value = item.get(RESOLUTION_TIMEFRAME_FIELD)
    if resolution_value in (None, ""):
        resolution_value = trade_geometry.get(RESOLUTION_TIMEFRAME_FIELD)

    resolution_timeframe: Optional[str] = None
    if resolution_value not in (None, ""):
        resolution_timeframe = str(resolution_value).strip().upper()
        if resolution_timeframe not in SUPPORTED_TIMEFRAMES:
            errors.append(
                f"Setup {id} has unsupported {RESOLUTION_TIMEFRAME_FIELD} "
                f"'{resolution_timeframe}'."
            )

    prediction_time = _first_timestamp(
        item,
        ("predicted_at", "prediction_time", "prediction_timestamp"),
    )
    probability_value = item.get(PREDICTED_PROBABILITY_FIELD)
    if probability_value is None:
        probability_value = item.get("predicted_probability", item.get("probability_at_entry"))
    probability = _normalize_probability(probability_value)
    if probability_value is not None and probability is None:
        errors.append(
            f"Setup {id} has an invalid entry probability; expected 0-1 or 0-100."
        )
    if prediction_time is not None and entry_candle_time is not None:
        if prediction_time > entry_candle_time:
            errors.append(
                f"Setup {id} has predicted_at later than its entry time."
            )

    end_time = _extract_end_time(item)
    outcome_value = item.get(ACTUAL_OUTCOME_FIELD)
    actual_outcome = _normalize_outcome(outcome_value)

    explicit_status = _normalize_trade_status(
        item.get("outcome_status", item.get("trade_status", item.get("status")))
    )

    null_outcome_values = {
        "",
        "NULL",
        "NONE",
        "AMBIGUOUS",
        "UNRESOLVED",
    }

    is_null_outcome = (
        outcome_value is None
        or (
            isinstance(outcome_value, str)
            and outcome_value.strip().upper() in null_outcome_values
        )
    )

    if not is_null_outcome and actual_outcome is None:
        label_errors.append(
            "The stored outcome must be TP, SL or NULL."
        )

    if actual_outcome is not None and end_time is None:
        label_errors.append(
            "An actual outcome is present without end_time."
        )

    if entry_candle_time is not None and end_time is not None:
        if end_time < entry_candle_time:
            label_errors.append(
                "Trade end_time precedes its entry time."
            )


    if label_errors:
        actual_outcome = None

    if actual_outcome == "AMBIGUOUS":
        outcome_status = "AMBIGUOUS"
    elif actual_outcome in {"TP", "SL"} and not label_errors:
        outcome_status = "CLOSED_LABELLED"
    elif end_time is not None:
        outcome_status = "CLOSED_UNLABELLED"
    elif explicit_status == "OPEN" or entry_candle_time is not None:
        outcome_status = "OPEN"
    else:
        outcome_status = explicit_status

    sanitized_item = _strip_legacy_focus_fields(dict(item))
    sanitized_geometry = sanitized_item.get("trade_geometry", {})
    if not isinstance(sanitized_geometry, dict):
        sanitized_geometry = {}
    

    return UserSetup(
        id=id,
        direction=direction,
        setup_timeframe=setup_timeframe,
        entry_timeframe=entry_timeframe,
        anchor_time=anchor_time,
        anchor_price=to_float(anchor.get("price", item.get("anchor_price"))),
        breakout_time=breakout_time,
        breakout=breakout,
        entry_candle_time=entry_candle_time,
        entry=entry,
        sl=sl,
        tp=tp,
        rr_mode=(
            str(item.get("rr_mode")).strip().upper()
            if item.get("rr_mode") is not None
            else None
        ),
        resolution_timeframe=resolution_timeframe,
        zones=zones,
        trade_geometry=copy.deepcopy(sanitized_geometry),
        raw=sanitized_item,
        validation_errors=errors,
        prediction_time=prediction_time,
        predicted_probability_at_entry=probability,
        end_time=end_time,
        actual_outcome=actual_outcome,
        outcome_status=outcome_status,
        label_validation_errors=label_errors,
    )


def normalize_learning_setups(records: Sequence[Mapping[str, Any]]) -> List[UserSetup]:
    """Normalize a complete setup list and flag duplicate identifiers."""
    normalized = [
        normalize_learning_setup(item, index)
        for index, item in enumerate(records, start=1)
    ]
    seen: Dict[int, int] = {}
    for index, setup in enumerate(normalized, start=1):
        if setup.id != index:
            setup.validation_errors.append(
                f"Setup id {setup.id} does not match archive position {index}."
            )
        if setup.id in seen:
            previous_index = seen[setup.id]
            message = f"Duplicate setup id {setup.id} at indexes {previous_index} and {index}."
            setup.validation_errors.append(message)
            normalized[previous_index - 1].validation_errors.append(message)
        else:
            seen[setup.id] = index

    return normalized


def setup_time(setup: UserSetup) -> Optional[int]:
    """Return the recorded entry timestamp for one setup."""
    return setup.entry_candle_time


# ============================================================================
# LEARNING SETUP FILE ACCESS
# ============================================================================

def load_learning_setup_records(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> List[Dict[str, Any]]:
    """Load raw setup records for one TradingBot without dropping supported fields."""
    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = Path(get_learning_setups_path(resolved)).expanduser().resolve()
    print(f"[SPECTLINE] Training setup archive: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Learning setup file was not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file_object:
            raw_data = json.load(file_object)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Learning setup JSON is malformed at line {error.lineno}, "
            f"column {error.colno}: {path}"
        ) from error
    except OSError as error:
        raise OSError(f"Could not read Learning setups from {path}: {error}") from error

    if not isinstance(raw_data, list):
        raise ValueError(f"Learning setup file must contain a JSON list: {path}")

    malformed_indexes = [
        index
        for index, item in enumerate(raw_data, start=1)
        if not isinstance(item, dict)
    ]
    if malformed_indexes:
        raise ValueError(
            "Learning setup file contains non-object entries at indexes: "
            + ", ".join(str(index) for index in malformed_indexes)
        )

    return _strip_legacy_focus_fields(raw_data)


def load_learning_setups(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> List[UserSetup]:
    """Load and normalize every setup for one exact customer and TradingBot."""
    raw_records = load_learning_setup_records(
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )
    return normalize_learning_setups(raw_records)


# ============================================================================
# SETUP MATCHING AND FEATURE INPUT PREPARATION
# ============================================================================

def closest_setup(
    setup_times: Sequence[int],
    target_time: int,
    tolerance: int,
) -> Optional[int]:
    """Return the index of the closest setup within a non-negative tolerance."""
    resolved_tolerance = int(tolerance)
    if resolved_tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    best_index: Optional[int] = None
    best_distance: Optional[int] = None
    for index, setup_timestamp in enumerate(setup_times):
        distance = abs(int(setup_timestamp) - int(target_time))
        if distance > resolved_tolerance:
            continue
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def collect_required_timeframes(setups: Sequence[UserSetup]) -> List[str]:
    """Collect unique supported timeframes referenced by all supplied setups."""
    required = {
        timeframe
        for setup in setups
        for timeframe in (
            setup.setup_timeframe,
            setup.entry_timeframe,
            setup.resolution_timeframe,
        )
        if timeframe and timeframe in SUPPORTED_TIMEFRAMES
    }
    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in required]


def extract_manual_levels(
    setups: Sequence[UserSetup],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group each manually drawn Support and Resistance zone by setup ID."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for setup in setups:
        levels: List[Dict[str, Any]] = []
        for zone in setup.zones:
            levels.append(
                {                
                    "zone_id": zone.zone_id,
                    "type": zone.zone_type,
                    "start_time": zone.start_time,
                    "end_time": zone.end_time,
                    "low": zone.low,
                    "high": zone.high,
                    "validation_errors": list(zone.validation_errors),
                }
            )
        grouped[setup.id] = levels
    return grouped


def setup_input_record(setup: UserSetup) -> Dict[str, Any]:
    """Build a setup feature record that excludes prediction and future outcome data."""
    zones = [
        {
            "zone_id": zone.zone_id,
            "type": zone.zone_type,
            "start_time": zone.start_time,
            "end_time": zone.end_time,
            "low": zone.low,
            "high": zone.high,
        }
        for zone in setup.zones
        if not zone.validation_errors
    ]
    return {
        "direction": setup.direction,
        "setup_timeframe": setup.setup_timeframe,
        "entry_timeframe": setup.entry_timeframe,
        "resolution_timeframe": setup.resolution_timeframe,
        "decision_time": setup.decision_time,
        "entry_time": setup.entry_candle_time,
        "anchor_time": setup.anchor_time,
        "anchor_price": setup.anchor_price,
        "breakout_time": setup.breakout_time,
        "breakout": setup.breakout,
        "entry": setup.entry,
        "sl": setup.sl,
        "tp": setup.tp,
        "rr_mode": setup.rr_mode,
        "zones": zones,
    }


def setup_prediction_record(setup: UserSetup) -> Optional[Dict[str, Any]]:
    """Build a separate prediction record without mixing it into feature inputs."""
    if (
        setup.predicted_probability_at_entry is None
        or setup.prediction_time is None
        or setup.entry_candle_time is None
        or setup.prediction_time > setup.entry_candle_time
    ):
        return None
    return {
        "id": setup.id,
        "entry_time": setup.entry_candle_time,
        "predicted_at": setup.prediction_time,
        "predicted_probability_at_entry": setup.predicted_probability_at_entry,
    }


def setup_result_record(
    setup: UserSetup,
    market: str,
) -> Optional[Dict[str, Any]]:
    if (
        setup.end_time is None
        or setup.actual_outcome not in _VALID_OUTCOMES
        or setup.entry_candle_time is None
    ):
        return None

    return {
        "id": setup.id,
        "market": market,
        "entry_time": setup.entry_candle_time,
        "end_time": setup.end_time,
        "actual_outcome": setup.actual_outcome,
    }

def select_required_timeframes(setups_or_timeframes: Iterable[Any]) -> List[str]:
    """Compatibility helper for normalized setups, dictionaries or timeframe strings."""
    requested: set[str] = set()
    values = (setups_or_timeframes,) if isinstance(setups_or_timeframes, str) else setups_or_timeframes

    for item in values:
        if isinstance(item, str):
            candidates = (item,)
        else:
            candidates = (
                _read_object_field(item, "setup_timeframe"),
                _read_object_field(item, "entry_timeframe"),
                _read_object_field(item, "resolution_timeframe"),
            )
        for value in candidates:
            if value is None:
                continue
            timeframe = str(value).strip().upper()
            if timeframe in SUPPORTED_TIMEFRAMES:
                requested.add(timeframe)

    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in requested]


# ============================================================================
# OPTIONAL ENGINE MODULE INTEGRATION
# ============================================================================

def _engine_module_name(module_name: str) -> str:
    """Return the package-qualified name of an engine module."""
    package_name = __package__ or "engine"
    return f"{package_name}.{module_name}"


def _load_optional_engine_module(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """Load an engine module while distinguishing absence from internal import errors."""
    qualified_name = _engine_module_name(module_name)
    try:
        return importlib.import_module(qualified_name), None
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        if missing_name in {qualified_name, module_name, qualified_name.rsplit(".", 1)[0]}:
            return None, f"{qualified_name} is not available."
        raise


def _load_optional_engine_function(
    module_name: str,
    function_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Load a callable from a future engine module without hiding dependency errors."""
    module, issue = _load_optional_engine_module(module_name)
    if module is None:
        return None, issue
    function = getattr(module, function_name, None)
    if not callable(function):
        return None, f"{module_name}.{function_name} is not implemented."
    return function, None


def _invoke_with_supported_keywords(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function with only keyword arguments accepted by its signature."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args, **kwargs)

    accepts_arbitrary_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_arbitrary_keywords:
        return function(*args, **kwargs)

    selected = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(*args, **selected)


def _load_historical_data(
    market: str,
    timeframes: Sequence[str],
    identity: ClientIdentity,
) -> Tuple[Dict[str, List[Candle]], Dict[str, Dict[str, Any]], List[str]]:
    """Load shared market archives and return normalized candles plus reports."""
    if not timeframes:
        return {}, {}, []

    module, module_issue = _load_optional_engine_module("history")
    if module is None:
        return {}, {}, [module_issue or "Historical data module is unavailable."]

    loader = getattr(module, "load_candles_with_reports", None)
    loader_kind = "with_reports"
    if not callable(loader):
        loader = getattr(module, "load_candles_for_timeframes", None)
        loader_kind = "plain"

    if not callable(loader):
        return {}, {}, ["history does not expose a supported candle loader."]

    try:
        loaded = _invoke_with_supported_keywords(
            loader,
            market,
            list(timeframes),
            identity=identity,
        )
    except Exception as error:
        return {}, {}, [f"Could not load historical candles: {error}"]

    raw_candles: Any
    raw_reports: Any
    if loader_kind == "with_reports":
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            return {}, {}, ["history.load_candles_with_reports returned an invalid value."]
        raw_candles, raw_reports = loaded
    else:
        raw_candles = loaded
        raw_reports = {}

    if not isinstance(raw_candles, Mapping):
        return {}, {}, ["The historical candle loader did not return a mapping."]
    if not isinstance(raw_reports, Mapping):
        raw_reports = {}

    normalized: Dict[str, List[Candle]] = {}
    reports: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []

    for timeframe in timeframes:
        raw_rows = raw_candles.get(timeframe, raw_candles.get(str(timeframe).lower(), []))
        candles, quality = normalize_candles(raw_rows or [])
        normalized[timeframe] = candles

        source_report = raw_reports.get(timeframe, {})
        report = dict(source_report) if isinstance(source_report, Mapping) else {}
        report.update(_candle_coverage(candles))
        report.update(quality)
        reports[timeframe] = report

        if quality["invalid_candle_count"]:
            issues.append(
                f"{timeframe} archive contains {quality['invalid_candle_count']} invalid candle row(s)."
            )
        if quality["duplicate_candle_count"]:
            issues.append(
                f"{timeframe} archive contains {quality['duplicate_candle_count']} duplicate timestamp(s)."
            )

    return normalized, reports, issues


def _build_automatic_levels(
    candles_by_timeframe: Dict[str, List[Candle]],
    setups: Sequence[UserSetup],
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """Build automatic Support and Resistance levels at each setup decision time."""
    builder, issue = _load_optional_engine_function(
        "features",
        "build_auto_levels_for_setups",
    )
    if builder is None:
        return {}, [issue or "Automatic level builder is unavailable."]

    try:
        result = builder(
            candles_by_timeframe=candles_by_timeframe,
            setups=setups,
        )
    except Exception as error:
        return {}, [f"Could not build automatic market levels: {error}"]

    if not isinstance(result, Mapping):
        return {}, ["features.build_auto_levels_for_setups returned an invalid value."]

    normalized: Dict[int, List[Dict[str, Any]]] = {}
    for setup in setups:
        levels = result.get(setup.id, result.get(str(setup.id), []))
        normalized[setup.id] = [
            copy.deepcopy(dict(level))
            for level in (levels or [])
            if isinstance(level, Mapping)
        ]
    return normalized, []


# ============================================================================
# MARKET MEMORY AND CUSTOMER RESULT ASSEMBLY
# ============================================================================

def _project_record(
    record: Mapping[str, Any],
    fields: Sequence[str],
) -> Dict[str, Any]:
    """Copy only explicitly approved fields from a historical record."""
    projected: Dict[str, Any] = {}
    for field_name in fields:
        if field_name in record:
            projected[field_name] = copy.deepcopy(record[field_name])
    return projected


def _load_market_memory(
    market: str,
    setups: Sequence[UserSetup],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, Any], List[str]]:
    """Load market patterns and events, retaining only records known at each setup."""
    decision_times = [
        int(setup.decision_time)
        for setup in setups
        if setup.decision_time is not None
    ]
    if not decision_times:
        return {}, {}, {"pattern_count": 0, "event_count": 0, "skipped_without_recognition_time": 0}, []

    maximum_decision_time = max(decision_times)
    storage_module, module_issue = _load_optional_engine_module("storage")
    if storage_module is None:
        return {}, {}, {}, [module_issue or "Historical storage module is unavailable."]

    storage_class = getattr(storage_module, "HistoricalStorage", None)
    if callable(storage_class):
        storage = storage_class()
        pattern_iterator = getattr(storage, "iter_pattern_occurrences", None)
        event_iterator = getattr(storage, "iter_market_events", None)
    else:
        storage = storage_module
        pattern_iterator = getattr(storage, "iter_pattern_occurrences", None)
        event_iterator = getattr(storage, "iter_market_events", None)

    if not callable(pattern_iterator) or not callable(event_iterator):
        return {}, {}, {}, ["engine.storage does not expose market history iterators."]

    patterns_by_timeframe: Dict[str, List[Dict[str, Any]]] = {}
    events_by_timeframe: Dict[str, List[Dict[str, Any]]] = {}
    issues: List[str] = []
    skipped_without_recognition_time = 0
    skipped_invalid_pattern_chronology = 0

    for timeframe in SUPPORTED_TIMEFRAMES:
        try:
            raw_patterns = list(
                _invoke_with_supported_keywords(
                    pattern_iterator,
                    market,
                    timeframe=timeframe,
                    end_time=maximum_decision_time,
                )
            )
        except Exception as error:
            issues.append(f"Could not read {timeframe} pattern archive: {error}")
            raw_patterns = []

        normalized_patterns: List[Dict[str, Any]] = []
        for record in raw_patterns:
            if not isinstance(record, Mapping):
                continue
            recognized_at = parse_timestamp(record.get("recognized_at"))
            if recognized_at is None:
                skipped_without_recognition_time += 1
                continue
            occurrence_start = parse_timestamp(record.get("occurrence_start"))
            occurrence_end = parse_timestamp(record.get("occurrence_end"))
            if recognized_at > maximum_decision_time:
                continue
            if (
                occurrence_start is None
                or occurrence_end is None
                or occurrence_end < occurrence_start
                or recognized_at < occurrence_end
            ):
                skipped_invalid_pattern_chronology += 1
                continue
            projected = _project_record(record, _PATTERN_INPUT_FIELDS)
            projected["occurrence_start"] = occurrence_start
            projected["occurrence_end"] = occurrence_end
            projected["recognized_at"] = recognized_at
            projected["available_at"] = recognized_at
            normalized_patterns.append(projected)

        normalized_patterns.sort(
            key=lambda record: (
                int(record.get("recognized_at", 0)),
                int(record.get("occurrence_start", 0)),
                str(record.get("pattern_id", "")),
            )
        )
        patterns_by_timeframe[timeframe] = normalized_patterns

        try:
            raw_events = list(
                _invoke_with_supported_keywords(
                    event_iterator,
                    market,
                    timeframe=timeframe,
                    end_time=maximum_decision_time,
                )
            )
        except Exception as error:
            issues.append(f"Could not read {timeframe} market event archive: {error}")
            raw_events = []

        normalized_events: List[Dict[str, Any]] = []
        for record in raw_events:
            if not isinstance(record, Mapping):
                continue
            recognized_at = parse_timestamp(record.get("recognized_at"))
            if recognized_at is None:
                skipped_without_recognition_time += 1
                continue
            if recognized_at > maximum_decision_time:
                continue
            projected = _project_record(record, _MARKET_EVENT_INPUT_FIELDS)
            projected["recognized_at"] = recognized_at
            projected["available_at"] = recognized_at
            normalized_events.append(projected)

        normalized_events.sort(
            key=lambda record: (
                int(record.get("recognized_at", 0)),
                int(record.get("event_time", 0)),
                str(record.get("event_type", "")),
            )
        )
        events_by_timeframe[timeframe] = normalized_events

    report = {
        "pattern_count": sum(len(records) for records in patterns_by_timeframe.values()),
        "event_count": sum(len(records) for records in events_by_timeframe.values()),
        "pattern_counts_by_timeframe": {
            timeframe: len(records)
            for timeframe, records in patterns_by_timeframe.items()
        },
        "event_counts_by_timeframe": {
            timeframe: len(records)
            for timeframe, records in events_by_timeframe.items()
        },
        "skipped_without_recognition_time": skipped_without_recognition_time,
        "skipped_invalid_pattern_chronology": skipped_invalid_pattern_chronology,
        "maximum_decision_time": maximum_decision_time,
    }
    return patterns_by_timeframe, events_by_timeframe, report, issues


def _market_memory_by_setup(
    setups: Sequence[UserSetup],
    patterns_by_timeframe: Mapping[str, Sequence[Mapping[str, Any]]],
    events_by_timeframe: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Create an as-of snapshot of market patterns and events for every setup."""
    memory_by_setup: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for setup in setups:
        cutoff = setup.decision_time
        available_patterns: List[Dict[str, Any]] = []
        available_events: List[Dict[str, Any]] = []

        if cutoff is not None:
            for records in patterns_by_timeframe.values():
                available_patterns.extend(
                    copy.deepcopy(dict(record))
                    for record in records
                    if parse_timestamp(record.get("recognized_at")) is not None
                    and int(record["recognized_at"]) <= int(cutoff)
                )
            for records in events_by_timeframe.values():
                available_events.extend(
                    copy.deepcopy(dict(record))
                    for record in records
                    if parse_timestamp(record.get("recognized_at")) is not None
                    and int(record["recognized_at"]) <= int(cutoff)
                )

        available_patterns.sort(
            key=lambda record: (
                int(record.get("recognized_at", 0)),
                int(record.get("occurrence_start", 0)),
                str(record.get("pattern_id", "")),
            )
        )
        available_events.sort(
            key=lambda record: (
                int(record.get("recognized_at", 0)),
                int(record.get("event_time", 0)),
                str(record.get("event_type", "")),
            )
        )
        memory_by_setup[str(setup.id)] = {
            "patterns": available_patterns,
            "events": available_events,
        }

    return memory_by_setup

# ============================================================================
# TRAINING DATA VALIDATION
# ============================================================================

def validate_training_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Validate bundle structure, identity scope and point-in-time separation."""
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(bundle, dict):
        return {
            "valid": False,
            "errors": ["Training data bundle must be a dictionary."],
            "warnings": [],
        }

    required_keys = {
        "identity",
        "market",
        "learning_setups",
        "setup_input_records",
        "setup_prediction_records",
        "setup_labels",
        "historical_data",
        "auto_levels_by_setup",
        "manual_levels_by_setup",
        "timeframe_data",
        "market_memory_by_setup",
        "setup_results",
        "report",
    }
    missing_keys = sorted(required_keys - set(bundle))
    if missing_keys:
        errors.append("Missing bundle keys: " + ", ".join(missing_keys))

    identity = bundle.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("Bundle identity must be a dictionary.")
    else:
        missing_identity = [
            field_name
            for field_name in ("user_id", "market", "strategy_id", "tradingbot_id")
            if not identity.get(field_name)
        ]
        if missing_identity:
            errors.append(
                "Bundle identity is incomplete: " + ", ".join(missing_identity)
            )
        if identity.get("market") != bundle.get("market"):
            errors.append("Bundle market does not match its customer identity.")

    learning_setups = bundle.get("learning_setups", {})
    if not isinstance(learning_setups, Mapping):
        errors.append("learning_setups must be a dictionary.")
    else:
        normalized_setups = learning_setups.get("normalized", [])
        if not isinstance(normalized_setups, list):
            errors.append("learning_setups.normalized must be a list.")
        elif not normalized_setups:
            errors.append("The Learning setup file contains no setups.")

    report = bundle.get("report", {})
    if isinstance(report, Mapping):
        warnings.extend(str(item) for item in report.get("warnings", []))
        if report.get("missing_timeframes"):
            errors.append(
                "Missing historical archives for: "
                + ", ".join(report.get("missing_timeframes", []))
            )
        if report.get("invalid_setup_count", 0):
            errors.append(
                f"{report.get('invalid_setup_count')} Learning setup(s) contain invalid input fields."
            )
        if report.get("history_alignment_issues"):
            errors.extend(str(item) for item in report.get("history_alignment_issues", []))
        if report.get("pending_components"):
            errors.append(
                "Engine components are not ready: "
                + ", ".join(report.get("pending_components", []))
            )
        if not report.get("ready_for_next_stage", False) and not errors:
            errors.append("The Training Data Bundle is not complete yet.")
    else:
        errors.append("Bundle report must be a dictionary.")

    setup_inputs = bundle.get("setup_input_records", {})
    if isinstance(setup_inputs, Mapping):
        forbidden_fields = {
            "id",
            "predicted_at",
            "predicted_probability_at_entry",
            "actual_outcome",
            "end_time",
        }
        for id, input_record in setup_inputs.items():
            if not isinstance(input_record, Mapping):
                errors.append(f"Setup input record {id} is not a dictionary.")
                continue
            leaked = forbidden_fields.intersection(input_record)
            if leaked:
                errors.append(
                    f"Setup input record {id} contains outcome or prediction fields: "
                    + ", ".join(sorted(leaked))
                )

    historical_data = bundle.get("historical_data", {})
    point_in_time_data = (
        historical_data.get("point_in_time_candles_by_setup", {})
        if isinstance(historical_data, Mapping)
        else {}
    )
    timeframe_data = bundle.get("timeframe_data", [])
    if isinstance(point_in_time_data, Mapping) and isinstance(timeframe_data, list):
        cutoff_by_setup = {
            str(record.get("id")): record.get("decision_time")
            for record in timeframe_data
            if isinstance(record, Mapping)
        }
        for id, frames in point_in_time_data.items():
            cutoff = cutoff_by_setup.get(str(id))
            if cutoff is None or not isinstance(frames, Mapping):
                continue
            for timeframe, candles in frames.items():
                duration = TIMEFRAME_SECONDS.get(str(timeframe).upper())
                if duration is None:
                    errors.append(f"Unsupported timeframe in point-in-time data: {timeframe}.")
                    continue
                for candle in candles or []:
                    candle_time = parse_timestamp(_read_object_field(candle, "time"))
                    if candle_time is None or candle_time + duration > int(cutoff):
                        errors.append(
                            f"Setup {id} contains a candle not closed by its decision time."
                        )
                        break

    market_memory = bundle.get("market_memory_by_setup", {})
    if isinstance(market_memory, Mapping) and isinstance(timeframe_data, list):
        cutoff_by_setup = {
            str(record.get("id")): record.get("decision_time")
            for record in timeframe_data
            if isinstance(record, Mapping)
        }
        for id, memory in market_memory.items():
            cutoff = cutoff_by_setup.get(str(id))
            if cutoff is None or not isinstance(memory, Mapping):
                continue
            for collection_name in ("patterns", "events"):
                for record in memory.get(collection_name, []) or []:
                    recognized_at = parse_timestamp(record.get("recognized_at"))
                    if recognized_at is None or recognized_at > int(cutoff):
                        errors.append(
                            f"Setup {id} contains {collection_name} unavailable at its decision time."
                        )
                        break
                    if collection_name == "patterns":
                        occurrence_end = parse_timestamp(record.get("occurrence_end"))
                        if occurrence_end is None or occurrence_end > recognized_at:
                            errors.append(
                                f"Setup {id} contains a pattern recognized before its occurrence ended."
                            )
                            break

    setup_results = bundle.get("setup_results", [])

    if not isinstance(setup_results, list):
        errors.append("setup_results must be a list.")
        setup_results = []

    for result in setup_results:
        if not isinstance(result, Mapping):
            errors.append("A setup result is not a dictionary.")
            continue

        if result.get("actual_outcome") not in _VALID_OUTCOMES:
            errors.append("A setup result has an invalid actual_outcome.")
            continue

        entry_time = parse_timestamp(result.get("entry_time"))
        end_time = parse_timestamp(result.get("end_time"))

        if entry_time is None or end_time is None or end_time < entry_time:
            errors.append("A setup result has invalid entry_time or end_time.")

# ============================================================================
# TRAINING DATA BUNDLE ASSEMBLY
# ============================================================================

setup_labels = bundle.get("setup_labels", [])

    if not isinstance(setup_labels, list):
        errors.append("setup_labels must be a list.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }

def prepare_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    point_in_time_candle_limit: int = DEFAULT_POINT_IN_TIME_CANDLE_LIMIT,
) -> Dict[str, Any]:
    """Assemble customer setups, market history and strictly point-in-time inputs."""
    candle_limit = int(point_in_time_candle_limit)
    if candle_limit <= 0:
        raise ValueError("point_in_time_candle_limit must be greater than zero.")

    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    normalized_market = safe_market_folder(resolved.market)
    identity_payload = _identity_to_dict(resolved)

    raw_records = load_learning_setup_records(
        normalized_market,
        identity=resolved,
    )
    setups = normalize_learning_setups(raw_records)

    required_timeframes = collect_required_timeframes(setups)
    candles_by_timeframe, history_reports, history_issues = _load_historical_data(
        normalized_market,
        required_timeframes,
        resolved,
    )

    auto_levels_by_setup, feature_issues = _build_automatic_levels(
        candles_by_timeframe,
        setups,
    )
    manual_levels_by_setup = extract_manual_levels(setups)

    candle_counts = {
        timeframe: len(candles_by_timeframe.get(timeframe, []))
        for timeframe in required_timeframes
    }
    missing_timeframes = [
        timeframe
        for timeframe in required_timeframes
        if not candles_by_timeframe.get(timeframe)
    ]
    coverage_by_timeframe = {
        timeframe: _candle_coverage(candles_by_timeframe.get(timeframe, []))
        for timeframe in required_timeframes
    }

    timeframe_data: List[Dict[str, Any]] = []
    point_in_time_candles_by_setup: Dict[str, Dict[str, List[Candle]]] = {}
    history_alignment_issues: List[str] = []

    for setup in setups:
        setup_required_timeframes = [
            timeframe
            for timeframe in (
                setup.setup_timeframe,
                setup.entry_timeframe,
                setup.resolution_timeframe,
            )
            if timeframe and timeframe in SUPPORTED_TIMEFRAMES
        ]
        feature_timeframes = list(
            dict.fromkeys(
                timeframe
                for timeframe in (setup.setup_timeframe, setup.entry_timeframe)
                if timeframe and timeframe in SUPPORTED_TIMEFRAMES
            )
        )
        available_at_decision: Dict[str, Dict[str, Any]] = {}
        input_candles: Dict[str, List[Candle]] = {}
        decision_time = setup.decision_time

        for timeframe in feature_timeframes:
            full_history = candles_by_timeframe.get(timeframe, [])
            closed_history = candles_available_at(
                full_history,
                timeframe,
                decision_time,
                max_count=candle_limit,
            )
            input_candles[timeframe] = closed_history
            duration = TIMEFRAME_SECONDS[timeframe]
            closed_count = sum(
                1
                for candle in full_history
                if candle.time + duration <= (decision_time if decision_time is not None else -1)
            )
            latest_closed = closed_history[-1] if closed_history else None
            available_at_decision[timeframe] = {
                "closed_candle_count": closed_count,
                "feature_candle_count": len(closed_history),
                "latest_closed_candle_time": (
                    latest_closed.time if latest_closed is not None else None
                ),
                "latest_closed_candle_close_time": (
                    latest_closed.time + duration if latest_closed is not None else None
                ),
            }

            if decision_time is not None and full_history and not closed_count:
                history_alignment_issues.append(
                    f"Setup {setup.id} has no fully closed {timeframe} "
                    "candle available at its prediction time."
                )

        point_in_time_candles_by_setup[str(setup.id)] = input_candles
        timeframe_data.append(
            {
                "id": setup.id,
                "entry_time": setup.entry_candle_time,
                "decision_time": decision_time,
                "setup_timeframe": setup.setup_timeframe,
                "entry_timeframe": setup.entry_timeframe,
                "resolution_timeframe": setup.resolution_timeframe,
                "required_timeframes": setup_required_timeframes,
                "feature_timeframes": feature_timeframes,
                "available_at_decision": available_at_decision,
            }
        )

    setup_input_records = {
        str(setup.id): setup_input_record(setup)
        for setup in setups
    }
    setup_prediction_records = [
        prediction
        for setup in setups
        for prediction in [setup_prediction_record(setup)]
        if prediction is not None
    ]
    setup_results = [
        result
        for setup in setups
        for result in [setup_result_record(setup, normalized_market)]
        if result is not None
    ]

    setup_labels = [
        {
            "id": result["id"],
            **identity_payload,
            "entry_time": result["entry_time"],
            "end_time": result["end_time"],
            "actual_outcome": result["actual_outcome"],
            "label": 1 if result["actual_outcome"] == "TP" else 0,
            "outcome_name": result["actual_outcome"],
            "label_target": "TP_BEFORE_SL",
            "label_source": "SETUPS_JSON",
        }
        for result in setup_results
    ]

    patterns_by_timeframe, events_by_timeframe, market_memory_report, market_memory_issues = (
        _load_market_memory(normalized_market, setups)
    )
    market_memory_by_setup = _market_memory_by_setup(
        setups,
        patterns_by_timeframe,
        events_by_timeframe,
    )    

    invalid_setup_count = sum(1 for setup in setups if setup.validation_errors)
    label_validation_error_count = sum(
        len(setup.label_validation_errors)
        for setup in setups
    )
    open_trade_count = sum(1 for setup in setups if setup.is_open)
    closed_trade_count = sum(1 for setup in setups if setup.is_closed)
    labelled_setup_count = len(setup_labels)
    manual_support_count = sum(
        1 for setup in setups for zone in setup.zones if zone.zone_type == "SUPPORT"
    )
    manual_resistance_count = sum(
        1 for setup in setups for zone in setup.zones if zone.zone_type == "RESISTANCE"
    )
    manual_zone_error_count = sum(
        len(zone.validation_errors)
        for setup in setups
        for zone in setup.zones
    )
    automatic_level_count = sum(
        len(levels)
        for levels in auto_levels_by_setup.values()
    )
    prediction_count = len(setup_prediction_records)

    pending_components: List[str] = []
    if history_issues:
        pending_components.append("engine.history")
    if feature_issues:
        pending_components.append("engine.features")
    if market_memory_issues:
        pending_components.append("engine.storage")

    warnings = (
        history_issues
        + feature_issues
        + market_memory_issues
    )
    
    if market_memory_report and not market_memory_report.get("pattern_count"):
        warnings.append("No recognized market pattern occurrences are available in storage.")
    if market_memory_report and not market_memory_report.get("event_count"):
        warnings.append("No recognized market events are available in storage.")
    if label_validation_error_count:
        warnings.append(
            f"{label_validation_error_count} trade-label issue(s) were excluded from binary labels."
        )

    ready_for_next_stage = (
        bool(setups)
        and invalid_setup_count == 0
        and not missing_timeframes
        and not history_alignment_issues
        and not pending_components
        and all(setup.id in auto_levels_by_setup for setup in setups)
        and all(setup.id in manual_levels_by_setup for setup in setups)
    )

    report = {
        "setup_count": len(setups),
        "valid_setup_count": len(setups) - invalid_setup_count,
        "invalid_setup_count": invalid_setup_count,
        "required_timeframes": list(required_timeframes),
        "loaded_timeframes": sorted(candles_by_timeframe),
        "candle_counts": candle_counts,
        "coverage_by_timeframe": coverage_by_timeframe,
        "history_load_reports": history_reports,
        "missing_timeframes": missing_timeframes,
        "history_alignment_issues": history_alignment_issues,
        "point_in_time_candle_limit": candle_limit,
        "manual_zone_count": manual_support_count + manual_resistance_count,
        "manual_support_count": manual_support_count,
        "manual_resistance_count": manual_resistance_count,
        "manual_zone_validation_errors": manual_zone_error_count,
        "automatic_level_count": automatic_level_count,
        "automatic_level_counts_by_setup": {
            str(id): len(levels)
            for id, levels in auto_levels_by_setup.items()
        },
        "prediction_record_count": prediction_count,
        "setup_label_count": labelled_setup_count,
        "open_trade_count": open_trade_count,
        "closed_trade_count": closed_trade_count,
        "label_validation_error_count": label_validation_error_count,
        "setup_result_count": len(setup_results),
        "market_memory": market_memory_report,
        "market_pattern_counts_by_setup": {
            id: len(records.get("patterns", []))
            for id, records in market_memory_by_setup.items()
        },
        "market_event_counts_by_setup": {
            id: len(records.get("events", []))
            for id, records in market_memory_by_setup.items()
        },
        "pending_components": pending_components,
        "warnings": warnings,
        "issues": warnings + history_alignment_issues,
        "ready_for_next_stage": ready_for_next_stage,
    }

    bundle: Dict[str, Any] = {
        "schema_version": DATA_BUNDLE_SCHEMA_VERSION,
        "status": "READY_FOR_NEXT_STAGE" if ready_for_next_stage else "INCOMPLETE",
        "identity": identity_payload,
        "market": normalized_market,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "learning_setups": {
            "raw": copy.deepcopy(raw_records),
            "normalized": setups,
        },
        "setup_input_records": setup_input_records,
        "setup_prediction_records": setup_prediction_records,
        "setup_labels": setup_labels,
        "setup_results": setup_results,
        "historical_data": {
            "timeframes": list(required_timeframes),
            "candles_by_timeframe": candles_by_timeframe,
            "point_in_time_candles_by_setup": point_in_time_candles_by_setup,
        },
        "auto_levels_by_setup": auto_levels_by_setup,
        "manual_levels_by_setup": manual_levels_by_setup,
        "timeframe_data": timeframe_data,
        "market_memory_by_setup": market_memory_by_setup,
        "report": report,
        "model_training_started": False,
    }

    bundle["validation"] = validate_training_data(bundle)
    return bundle


# ============================================================================
# FUTURE DATASET COMPATIBILITY
# ============================================================================

def make_negative_times(*args: Any, **kwargs: Any) -> Any:
    """Delegate no-trade sample generation to the dataset module."""
    function, issue = _load_optional_engine_function("dataset", "make_negative_times")
    if function is None:
        raise NotImplementedError(issue or "Negative sample generation is not ready.")
    return function(*args, **kwargs)


def build_training_dataset(*args: Any, **kwargs: Any) -> Any:
    """Delegate labelled dataset construction to the dataset module."""
    function, issue = _load_optional_engine_function("dataset", "build_training_dataset")
    if function is None:
        raise NotImplementedError(issue or "Training dataset creation is not ready.")
    return function(*args, **kwargs)


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "DATA_BUNDLE_SCHEMA_VERSION",
    "DEFAULT_POINT_IN_TIME_CANDLE_LIMIT",
    "RESOLUTION_TIMEFRAME_FIELD",
    "PREDICTION_TIME_FIELD",
    "PREDICTED_PROBABILITY_FIELD",
    "ACTUAL_OUTCOME_FIELD",
    "END_TIME_FIELD",
    "DataBundleError",
    "Candle",
    "ManualZone",
    "UserSetup",
    "safe_market_folder",
    "safe_user_folder",
    "_resolve_ai_identity",
    "setup_path",
    "to_float",
    "parse_timestamp",
    "normalize_candles",
    "candles_available_at",
    "normalize_learning_setup",
    "normalize_learning_setups",
    "load_learning_setup_records",
    "load_learning_setups",
    "setup_time",
    "closest_setup",
    "collect_required_timeframes",
    "select_required_timeframes",
    "extract_manual_levels",
    "setup_input_record",
    "setup_prediction_record",
    "setup_result_record",
    "validate_training_data",
    "prepare_training_data",
    "make_negative_times",
    "build_training_dataset",
]
