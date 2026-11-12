"""Build point-in-time-safe numerical datasets for TP-before-SL learning."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .settings import (
        MARKET_EVENT_TYPES,
        PATTERN_CATEGORY_ID_TO_NAME,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )
except ImportError:
    try:
        from engine.settings import (
            MARKET_EVENT_TYPES,
            PATTERN_CATEGORY_ID_TO_NAME,
            SUPPORTED_TIMEFRAMES,
            TIMEFRAME_SECONDS,
        )
    except ImportError:
        SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
        TIMEFRAME_SECONDS = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
            "W1": 604800,
        }
        PATTERN_CATEGORY_ID_TO_NAME = {
            0: "LEVELS",
            1: "TRENDLINES_TREND_CHANNELS",
            2: "PARALLEL_PATTERNS",
            3: "TRIANGLES_WEDGES",
            4: "PENNANTS",
            5: "MULTI_PIVOT_PATTERNS",
            6: "COMPLEX_STRUCTURAL_PATTERNS",
            7: "CANDLESTICK_PATTERNS",
            8: "WAVE_PATTERNS",
            9: "HARMONIC_PATTERNS",
        }
        MARKET_EVENT_TYPES = (
            "BREAKOUT",
            "FALSE_BREAKOUT",
            "RETEST",
            "REJECTION",
            "LIQUIDITY_SWEEP",
        )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

DATASET_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
DATASET_TARGET = "TP_BEFORE_SL"
OUTCOME_SL_FIRST = 0
OUTCOME_TP_FIRST = 1
PATTERN_HASH_BUCKETS = 64
EVENT_HASH_BUCKETS = 32
DEFAULT_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_NEGATIVE_SAMPLE_EVERY = 24
DEFAULT_NEGATIVE_EXCLUSION_SECONDS = 1800
MAX_FEATURE_VALUE = 1_000_000.0
_PATTERN_ID_LENGTH = 4
_IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class DatasetError(RuntimeError):
    """Base exception for dataset construction errors."""


class DatasetValidationError(DatasetError, ValueError):
    """Raised when dataset inputs violate the training data contract."""


class DatasetDependencyError(DatasetError, ImportError):
    """Raised when a required engine dependency is unavailable."""


# ============================================================================
# VALUE NORMALIZATION
# ============================================================================

def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> Optional[int]:
    number = _finite_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (OverflowError, ValueError):
        return None


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert Unix seconds, milliseconds, datetime or ISO text to UTC seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _finite_float(value)
        if number is None:
            return None
        if abs(number) >= 100_000_000_000:
            number /= 1000.0
        try:
            return int(number)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        numeric = _finite_float(text)
        if numeric is not None:
            return parse_timestamp(numeric)
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return int(parsed.astimezone(timezone.utc).timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _get_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        try:
            return asdict(record)
        except (TypeError, ValueError):
            pass
    raw = _get_value(record, "raw")
    result = dict(raw) if isinstance(raw, Mapping) else {}
    for field_name in (
        "setup_id",
        "trade_id",
        "direction",
        "setup_timeframe",
        "entry_timeframe",
        "resolution_timeframe",
        "decision_time",
        "entry_candle_time",
        "entry_time",
        "anchor_time",
        "anchor_price",
        "breakout_time",
        "breakout",
        "entry",
        "sl",
        "tp",
        "rr_mode",
        "exit_time",
        "actual_outcome",
        "outcome_known_at",
        "outcome_status",
        "validation_errors",
    ):
        value = _get_value(record, field_name)
        if value is not None:
            result[field_name] = value
    return result


def _clip(value: Any, lower: float = -MAX_FEATURE_VALUE, upper: float = MAX_FEATURE_VALUE) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return float(min(max(number, lower), upper))


def _normalize_timeframe(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return normalized if normalized in TIMEFRAME_SECONDS else None


def _normalize_direction(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    if normalized in {"BUY", "LONG"}:
        return "LONG"
    if normalized in {"SELL", "SHORT"}:
        return "SHORT"
    return None


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_identity(identity: Optional[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise DatasetValidationError("identity must be a mapping.")
    result: Dict[str, str] = {}
    for field_name in _IDENTITY_FIELDS:
        value = identity.get(field_name)
        if value is None or not str(value).strip():
            raise DatasetValidationError(
                "Customer identity is incomplete; missing " + field_name + "."
            )
        normalized = str(value).strip()
        if field_name == "market":
            normalized = normalized.upper()
        result[field_name] = normalized
    return result


def _derive_identity_from_setups(
    setups: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, str]]:
    discovered: Optional[Dict[str, str]] = None
    records_with_identity = 0
    records_without_identity = 0
    for setup in setups:
        nested = setup.get("identity")
        nested_identity = nested if isinstance(nested, Mapping) else {}
        raw = setup.get("raw")
        raw_identity = raw if isinstance(raw, Mapping) else {}
        supplied = {
            field_name: (
                setup.get(field_name)
                if setup.get(field_name) is not None
                else nested_identity.get(field_name, raw_identity.get(field_name))
            )
            for field_name in _IDENTITY_FIELDS
        }
        present = [value is not None and str(value).strip() for value in supplied.values()]
        if not any(present):
            records_without_identity += 1
            continue
        if not all(present):
            raise DatasetValidationError(
                "A setup contains a partial customer identity; dataset scope cannot be verified."
            )
        current = _normalize_identity(supplied)
        records_with_identity += 1
        if discovered is None:
            discovered = current
        elif discovered != current:
            raise DatasetValidationError(
                "Setups from different customer identities cannot share one dataset."
            )
    if discovered is not None and records_without_identity:
        raise DatasetValidationError(
            "Some setups are missing identity fields while others contain them; refusing a mixed-scope dataset."
        )
    return discovered


def _identity_matches(
    record: Mapping[str, Any],
    identity: Optional[Mapping[str, str]],
) -> bool:
    if identity is None:
        return True
    nested = record.get("identity")
    nested_identity = nested if isinstance(nested, Mapping) else {}
    has_identity = any(
        record.get(field_name) is not None or nested_identity.get(field_name) is not None
        for field_name in _IDENTITY_FIELDS
    )
    if not has_identity:
        return False
    raw = record.get("raw")
    raw_identity = raw if isinstance(raw, Mapping) else {}
    for field_name in _IDENTITY_FIELDS:
        direct_value = record.get(field_name)
        nested_value = nested_identity.get(field_name)
        raw_value = raw_identity.get(field_name)
        values = [value for value in (direct_value, nested_value, raw_value) if value is not None]
        if not values:
            return False
        expected = identity[field_name]
        for value in values:
            normalized = str(value).strip()
            if field_name == "market":
                normalized = normalized.upper()
            if normalized != expected:
                return False
    return True


# ============================================================================
# OPTIONAL ENGINE INTEGRATION
# ============================================================================

def _load_feature_functions() -> Dict[str, Any]:
    failures: List[str] = []
    module_names = ("engine.features", ".features")
    for module_name in module_names:
        try:
            if module_name.startswith("."):
                from importlib import import_module
                module = import_module(module_name, package=__package__)
            else:
                from importlib import import_module
                module = import_module(module_name)
            required = ("market_state_at", "market_state_metadata_at", "atr_at", "current_price_at")
            missing = [name for name in required if not callable(getattr(module, name, None))]
            if missing:
                failures.append(f"{module_name} is missing {', '.join(missing)}")
                continue
            return {name: getattr(module, name) for name in required}
        except Exception as error:
            failures.append(f"{module_name}: {error}")
    raise DatasetDependencyError(
        "The numerical feature engine is unavailable: " + "; ".join(failures)
    )


def _load_label_functions() -> Dict[str, Any]:
    failures: List[str] = []
    module_names = ("engine.labels", ".labels")
    for module_name in module_names:
        try:
            from importlib import import_module
            module = (
                import_module(module_name, package=__package__)
                if module_name.startswith(".")
                else import_module(module_name)
            )
            required = ("build_trade_label", "build_labels_from_bundle", "validate_label_record")
            missing = [name for name in required if not callable(getattr(module, name, None))]
            if missing:
                failures.append(f"{module_name} is missing {', '.join(missing)}")
                continue
            return {name: getattr(module, name) for name in required}
        except Exception as error:
            failures.append(f"{module_name}: {error}")
    raise DatasetDependencyError(
        "The trade-label engine is unavailable: " + "; ".join(failures)
    )


# ============================================================================
# CANDLE AND SETUP NORMALIZATION
# ============================================================================

def _normalize_candle_time(candle: Any) -> Optional[int]:
    return parse_timestamp(_get_value(candle, "time"))


def _normalize_candle_map(
    candles_by_timeframe: Optional[Mapping[str, Sequence[Any]]],
) -> Dict[str, List[Any]]:
    if candles_by_timeframe is None:
        return {timeframe: [] for timeframe in SUPPORTED_TIMEFRAMES}
    if not isinstance(candles_by_timeframe, Mapping):
        raise DatasetValidationError("candles_by_timeframe must be a mapping.")

    normalized: Dict[str, List[Any]] = {}
    for raw_timeframe, candles in candles_by_timeframe.items():
        timeframe = _normalize_timeframe(raw_timeframe)
        if timeframe is None:
            continue
        if candles is None:
            normalized[timeframe] = []
            continue
        if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
            raise DatasetValidationError(
                f"Candles for {timeframe} must be a sequence."
            )
        valid = [candle for candle in candles if _normalize_candle_time(candle) is not None]
        valid.sort(key=lambda candle: int(_normalize_candle_time(candle) or 0))
        normalized[timeframe] = valid

    return {
        timeframe: list(normalized.get(timeframe, []))
        for timeframe in SUPPORTED_TIMEFRAMES
    }


def _closed_candles_at(
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    decision_time: int,
) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for timeframe in SUPPORTED_TIMEFRAMES:
        duration = int(TIMEFRAME_SECONDS[timeframe])
        result[timeframe] = [
            candle
            for candle in candles_by_timeframe.get(timeframe, [])
            if (timestamp := _normalize_candle_time(candle)) is not None
            and timestamp + duration <= int(decision_time)
        ]
    return result


def _setup_identifier(setup: Mapping[str, Any]) -> str:
    """Return the single current dynamic ID of a manual setup."""
    value = setup.get("id")

    if value is None or isinstance(value, bool) or not str(value).strip():
        raise DatasetValidationError(
            "Every setup requires a non-empty id."
        )

    text = str(value).strip()

    if not text.isdigit() or int(text) <= 0:
        raise DatasetValidationError(
            "Every setup id must be a positive integer."
        )

    return str(int(text))
    
def _setup_decision_time(setup: Mapping[str, Any]) -> Optional[int]:
    for field_name in ("decision_time", "predicted_at", "prediction_time", "entry_time", "entry_candle_time"):
        value = parse_timestamp(setup.get(field_name))
        if value is not None:
            return value
    return None


def _setup_entry_time(setup: Mapping[str, Any]) -> Optional[int]:
    for field_name in ("entry_time", "entry_candle_time", "opened_at", "entry_timestamp"):
        value = parse_timestamp(setup.get(field_name))
        if value is not None:
            return value
    return None


def _setup_zones(setup: Mapping[str, Any]) -> List[Dict[str, Any]]:
    zones = setup.get("zones", [])
    if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)):
        return []
    return [dict(zone) for zone in zones if isinstance(zone, Mapping)]


def _timeframe_candles_for_setup(
    setup_id: str,
    setup: Mapping[str, Any],
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    point_in_time_candles_by_setup: Optional[Mapping[str, Any]],
    decision_time: int,
) -> Dict[str, List[Any]]:
    if isinstance(point_in_time_candles_by_setup, Mapping):
        raw_frames = point_in_time_candles_by_setup.get(setup_id)
        if isinstance(raw_frames, Mapping):
            normalized = _normalize_candle_map(raw_frames)
            return _closed_candles_at(normalized, decision_time)
    return _closed_candles_at(candles_by_timeframe, decision_time)


# ============================================================================
# POINT-IN-TIME MARKET CONTEXT
# ============================================================================

def _point_in_time_records(
    records: Any,
    decision_time: int,
    *,
    time_fields: Sequence[str],
    recognition_field: str = "recognized_at",
) -> List[Dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    result: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            continue
        recognized_at = parse_timestamp(item.get(recognition_field))
        if recognized_at is None or recognized_at > decision_time:
            continue
        event_time = None
        for field_name in time_fields:
            event_time = parse_timestamp(item.get(field_name))
            if event_time is not None:
                break
        if event_time is not None and event_time > decision_time:
            continue
        result.append(dict(item))
    return result


def _memory_for_setup(
    bundle: Optional[Mapping[str, Any]],
    setup_id: str,
    decision_time: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(bundle, Mapping):
        return [], []
    memory = bundle.get("market_memory_by_setup", {})
    record = memory.get(setup_id, {}) if isinstance(memory, Mapping) else {}
    if not isinstance(record, Mapping):
        record = {}

    patterns = _point_in_time_records(
        record.get("patterns", []),
        decision_time,
        time_fields=("occurrence_end", "occurrence_start", "event_time"),
    )
    valid_patterns: List[Dict[str, Any]] = []
    for pattern in patterns:
        occurrence_end = parse_timestamp(pattern.get("occurrence_end"))
        recognized_at = parse_timestamp(pattern.get("recognized_at"))
        if occurrence_end is None or recognized_at is None:
            continue
        if recognized_at < occurrence_end or recognized_at > decision_time:
            continue
        pattern_id = str(pattern.get("pattern_id", "")).strip()
        if len(pattern_id) != _PATTERN_ID_LENGTH or not pattern_id.isdigit():
            continue
        valid_patterns.append(pattern)

    events = _point_in_time_records(
        record.get("events", []),
        decision_time,
        time_fields=("event_time", "recognized_at"),
    )
    return valid_patterns, events


def _relevance_for_setup(
    bundle: Optional[Mapping[str, Any]],
    setup_id: str,
    decision_time: int,
) -> Dict[str, Any]:
    if not isinstance(bundle, Mapping):
        return {}
    relevance_map = bundle.get("relevance_by_setup")
    relevance_root = bundle.get("relevance", {})
    if not isinstance(relevance_map, Mapping) and isinstance(relevance_root, Mapping):
        relevance_map = relevance_root.get("by_setup", {})
    if not isinstance(relevance_map, Mapping):
        return {}
    item = relevance_map.get(setup_id, {})
    if not isinstance(item, Mapping):
        return {}
    result = dict(item)
    result["patterns_known_at_entry"] = _point_in_time_records(
        item.get("patterns_known_at_entry", []),
        decision_time,
        time_fields=("occurrence_end", "occurrence_start"),
    )
    result["events_known_at_entry"] = _point_in_time_records(
        item.get("events_known_at_entry", []),
        decision_time,
        time_fields=("event_time",),
    )
    result["patterns_during_trade"] = []
    result["events_during_trade"] = []
    return result


def _merge_known_context(
    memory_patterns: Sequence[Mapping[str, Any]],
    memory_events: Sequence[Mapping[str, Any]],
    relevance: Mapping[str, Any],
    decision_time: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pattern_records: Dict[str, Dict[str, Any]] = {}
    for record in list(memory_patterns) + list(relevance.get("patterns_known_at_entry", [])):
        if not isinstance(record, Mapping):
            continue
        recognized_at = parse_timestamp(record.get("recognized_at"))
        occurrence_end = parse_timestamp(record.get("occurrence_end"))
        if recognized_at is None or occurrence_end is None:
            continue
        occurrence_start = parse_timestamp(record.get("occurrence_start"))
        if recognized_at < occurrence_end or recognized_at > decision_time:
            continue
        if occurrence_start is not None and occurrence_start > occurrence_end:
            continue
        pattern_id = str(record.get("pattern_id", "")).strip()
        if len(pattern_id) != _PATTERN_ID_LENGTH or not pattern_id.isdigit():
            continue
        key = str(record.get("occurrence_id") or record.get("record_id") or f"{pattern_id}:{occurrence_end}")
        pattern_records[key] = dict(record)

    event_records: Dict[str, Dict[str, Any]] = {}
    for record in list(memory_events) + list(relevance.get("events_known_at_entry", [])):
        if not isinstance(record, Mapping):
            continue
        recognized_at = parse_timestamp(record.get("recognized_at"))
        if recognized_at is None or recognized_at > decision_time:
            continue
        key = str(record.get("event_id") or record.get("record_id") or f"{record.get('event_type')}:{recognized_at}")
        event_records[key] = dict(record)

    return list(pattern_records.values()), list(event_records.values())


# ============================================================================
# NUMERICAL FEATURE SCHEMA
# ============================================================================

def _build_feature_names() -> List[str]:
    names: List[str] = []
    for timeframe in SUPPORTED_TIMEFRAMES:
        names.extend(
            f"market.{timeframe}.feature_{index:02d}"
            for index in range(20)
        )
    names.extend(
        (
            "calendar.day_sin",
            "calendar.day_cos",
            "calendar.weekday_sin",
            "calendar.weekday_cos",
        )
    )
    names.extend(("setup.direction_long", "setup.direction_short"))
    for field_name in ("setup_timeframe", "entry_timeframe", "resolution_timeframe"):
        names.extend(f"setup.{field_name}.{timeframe}" for timeframe in SUPPORTED_TIMEFRAMES)
    names.extend(
        (
            "setup.anchor_offset_atr",
            "setup.breakout_offset_atr",
            "setup.entry_offset_atr",
            "setup.stop_offset_atr",
            "setup.target_offset_atr",
            "setup.anchor_price_available",
            "setup.breakout_price_available",
            "setup.entry_price_available",
            "setup.stop_price_available",
            "setup.target_price_available",
            "setup.risk_distance_atr",
            "setup.reward_distance_atr",
            "setup.reward_risk_ratio",
            "setup.anchor_age_units",
            "setup.anchor_time_available",
            "setup.breakout_age_units",
            "setup.breakout_time_available",
            "setup.support_zone_count",
            "setup.resistance_zone_count",
            "setup.nearest_support_distance_atr",
            "setup.nearest_support_available",
            "setup.nearest_resistance_distance_atr",
            "setup.nearest_resistance_available",
        )
    )
    names.extend(
        f"market.pattern_category.{category_id}"
        for category_id in sorted(PATTERN_CATEGORY_ID_TO_NAME)
    )
    names.extend(f"market.pattern_hash_{index:02d}" for index in range(PATTERN_HASH_BUCKETS))
    names.extend(f"market.event_hash_{index:02d}" for index in range(EVENT_HASH_BUCKETS))
    names.extend(
        (
            "market.known_pattern_count",
            "market.known_event_count",
            "market.pattern_relevance_mean",
            "market.pattern_relevance_max",
            "market.latest_pattern_age_days_log",
            "market.latest_event_age_days_log",
        )
    )
    names.extend(f"market.timeframe_available.{timeframe}" for timeframe in SUPPORTED_TIMEFRAMES)
    names.extend(f"market.timeframe_history_fraction.{timeframe}" for timeframe in SUPPORTED_TIMEFRAMES)
    return names


FEATURE_NAMES = tuple(_build_feature_names())
FEATURE_COUNT = len(FEATURE_NAMES)


def _market_state_feature_block(
    candles: Mapping[str, Sequence[Any]],
    decision_time: int,
    feature_functions: Mapping[str, Any],
) -> Tuple[List[float], Dict[str, Any]]:
    state = feature_functions["market_state_at"](dict(candles), int(decision_time))
    array = np.asarray(state, dtype=np.float32).reshape(-1)
    expected = 20 * len(SUPPORTED_TIMEFRAMES) + 4
    if array.size != expected:
        raise DatasetValidationError(
            f"market_state_at returned {array.size} values; expected {expected}."
        )
    if not np.all(np.isfinite(array)):
        raise DatasetValidationError("Market-state features contain NaN or infinity.")
    metadata = feature_functions["market_state_metadata_at"](
        dict(candles), int(decision_time)
    )
    return [float(value) for value in array], dict(metadata)


def _direction_and_timeframe_features(setup: Mapping[str, Any]) -> List[float]:
    values: List[float] = []
    direction = _normalize_direction(setup.get("direction"))
    values.extend([1.0 if direction == "LONG" else 0.0, 1.0 if direction == "SHORT" else 0.0])
    for field_name in ("setup_timeframe", "entry_timeframe", "resolution_timeframe"):
        selected = _normalize_timeframe(setup.get(field_name))
        values.extend([1.0 if selected == timeframe else 0.0 for timeframe in SUPPORTED_TIMEFRAMES])
    return values


def _setup_level_features(
    setup: Mapping[str, Any],
    candles: Mapping[str, Sequence[Any]],
    decision_time: int,
    feature_functions: Mapping[str, Any],
) -> List[float]:
    current_price = _finite_float(
        feature_functions["current_price_at"](dict(candles), int(decision_time))
    )
    market_price_available = current_price is not None
    timeframe = (
        _normalize_timeframe(setup.get("entry_timeframe"))
        or _normalize_timeframe(setup.get("setup_timeframe"))
    )
    atr = _finite_float(feature_functions["atr_at"](dict(candles), int(decision_time)))
    if atr is None or atr <= 0.0:
        atr = None
    if current_price is None or not math.isfinite(current_price):
        current_price = _finite_float(setup.get("entry"))
    if current_price is None:
        current_price = _finite_float(setup.get("anchor_price"))
    if atr is None:
        atr = max(abs(current_price or 0.0) * 1e-8, 1e-8)
    atr_available = 1.0 if market_price_available else 0.0

    direction = _normalize_direction(setup.get("direction"))
    sign = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 1.0
    price_fields = ("anchor_price", "breakout", "entry", "sl", "tp")
    normalized_levels: List[float] = []
    masks: List[float] = []
    for field_name in price_fields:
        price = _finite_float(setup.get(field_name))
        available = price is not None and current_price is not None and atr_available > 0.0
        if available:
            normalized_levels.append(_clip(((price - current_price) / atr) * sign, -100.0, 100.0))
            masks.append(1.0)
        else:
            normalized_levels.append(0.0)
            masks.append(0.0)

    entry = _finite_float(setup.get("entry"))
    stop = _finite_float(setup.get("sl"))
    target = _finite_float(setup.get("tp"))
    risk = (abs(entry - stop) / atr) if atr_available > 0.0 and entry is not None and stop is not None else None
    reward = (abs(target - entry) / atr) if atr_available > 0.0 and entry is not None and target is not None else None
    rr_ratio = reward / risk if reward is not None and risk is not None and risk > 1e-12 else None

    timeframe_seconds = float(TIMEFRAME_SECONDS.get(timeframe or "", 1))
    anchor_time = parse_timestamp(setup.get("anchor_time"))
    breakout_time = parse_timestamp(setup.get("breakout_time"))
    anchor_age = (
        math.log1p(max(decision_time - anchor_time, 0) / timeframe_seconds)
        if anchor_time is not None and anchor_time <= decision_time
        else 0.0
    )
    breakout_age = (
        math.log1p(max(decision_time - breakout_time, 0) / timeframe_seconds)
        if breakout_time is not None and breakout_time <= decision_time
        else 0.0
    )

    zones = _setup_zones(setup)
    known_zones: List[Dict[str, Any]] = []
    support_edges: List[float] = []
    resistance_edges: List[float] = []
    for zone in zones:
        zone_type = str(zone.get("zone_type", zone.get("type", ""))).strip().upper()
        zone_start = parse_timestamp(zone.get("start_time"))
        zone_end = parse_timestamp(zone.get("end_time"))
        if (zone_start is not None and zone_start > decision_time) or (zone_end is not None and zone_end > decision_time):
            continue
        low = _finite_float(zone.get("low"))
        high = _finite_float(zone.get("high"))
        if low is None or high is None or high < low:
            continue
        if zone_type not in {"SUPPORT", "RESISTANCE"}:
            continue
        known_zones.append(zone)
        zone_mid = (low + high) / 2.0
        if current_price is None or atr_available <= 0.0:
            continue
        distance = abs(zone_mid - current_price) / atr
        if zone_type == "SUPPORT":
            support_edges.append(distance)
        elif zone_type == "RESISTANCE":
            resistance_edges.append(distance)

    return [
        *normalized_levels,
        *masks,
        _clip(risk, 0.0, 100.0),
        _clip(reward, 0.0, 100.0),
        _clip(rr_ratio, 0.0, 100.0),
        _clip(anchor_age, 0.0, 100.0),
        1.0 if anchor_time is not None and anchor_time <= decision_time else 0.0,
        _clip(breakout_age, 0.0, 100.0),
        1.0 if breakout_time is not None and breakout_time <= decision_time else 0.0,
        _clip(sum(1 for zone in known_zones if str(zone.get("zone_type", zone.get("type", ""))).upper() == "SUPPORT"), 0.0, 1000.0),
        _clip(sum(1 for zone in known_zones if str(zone.get("zone_type", zone.get("type", ""))).upper() == "RESISTANCE"), 0.0, 1000.0),
        _clip(min(support_edges), 0.0, 100.0) if support_edges else 0.0,
        1.0 if support_edges else 0.0,
        _clip(min(resistance_edges), 0.0, 100.0) if resistance_edges else 0.0,
        1.0 if resistance_edges else 0.0,
    ]


def _pattern_event_features(
    patterns: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    relevance: Mapping[str, Any],
    decision_time: int,
) -> List[float]:
    category_counts = {int(category_id): 0 for category_id in PATTERN_CATEGORY_ID_TO_NAME}
    pattern_hash = [0.0] * PATTERN_HASH_BUCKETS
    event_hash = [0.0] * EVENT_HASH_BUCKETS
    relevance_scores: List[float] = []
    pattern_ages: List[int] = []
    event_ages: List[int] = []

    for pattern in patterns:
        pattern_id = str(pattern.get("pattern_id", "")).strip()
        if len(pattern_id) != _PATTERN_ID_LENGTH or not pattern_id.isdigit():
            continue
        category_id = int(pattern_id[0])
        if category_id in category_counts:
            category_counts[category_id] += 1
        bucket = _stable_hash("PATTERN:" + pattern_id) % PATTERN_HASH_BUCKETS
        pattern_hash[bucket] = 1.0
        score = _finite_float(pattern.get("relevance_score"))
        if score is None:
            score = _finite_float(pattern.get("score"))
        if score is not None:
            relevance_scores.append(_clip(score, 0.0, 1.0))
        recognized_at = parse_timestamp(pattern.get("recognized_at"))
        if recognized_at is not None and recognized_at <= decision_time:
            pattern_ages.append(decision_time - recognized_at)

    for event in events:
        event_type = str(event.get("event_type", "UNKNOWN")).strip().upper() or "UNKNOWN"
        bucket = _stable_hash("EVENT:" + event_type) % EVENT_HASH_BUCKETS
        event_hash[bucket] = 1.0
        recognized_at = parse_timestamp(event.get("recognized_at"))
        if recognized_at is not None and recognized_at <= decision_time:
            event_ages.append(decision_time - recognized_at)

    result: List[float] = []
    result.extend(_clip(math.log1p(category_counts[key]), 0.0, 20.0) for key in sorted(category_counts))
    result.extend(pattern_hash)
    result.extend(event_hash)
    result.extend(
        (
            _clip(math.log1p(len(patterns)), 0.0, 20.0),
            _clip(math.log1p(len(events)), 0.0, 20.0),
            _clip(sum(relevance_scores) / len(relevance_scores), 0.0, 1.0) if relevance_scores else 0.0,
            _clip(max(relevance_scores), 0.0, 1.0) if relevance_scores else 0.0,
            _clip(math.log1p(min(pattern_ages) / 86400.0), 0.0, 30.0) if pattern_ages else 0.0,
            _clip(math.log1p(min(event_ages) / 86400.0), 0.0, 30.0) if event_ages else 0.0,
        )
    )
    return result


def _build_setup_features(
    setup: Mapping[str, Any],
    *,
    bundle: Optional[Mapping[str, Any]],
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    point_in_time_candles_by_setup: Optional[Mapping[str, Any]],
    feature_functions: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    setup_id = _setup_identifier(setup)
    decision_time = _setup_decision_time(setup)
    if decision_time is None:
        raise DatasetValidationError(f"Setup {setup_id} has no valid decision_time.")
    entry_time = _setup_entry_time(setup)
    if entry_time is not None and decision_time > entry_time:
        raise DatasetValidationError(
            f"Setup {setup_id} has a decision_time after its entry_time."
        )

    point_in_time_candles = _timeframe_candles_for_setup(
        setup_id,
        setup,
        candles_by_timeframe,
        point_in_time_candles_by_setup,
        decision_time,
    )
    market_state, market_metadata = _market_state_feature_block(
        point_in_time_candles,
        decision_time,
        feature_functions,
    )
    patterns, events = _memory_for_setup(bundle, setup_id, decision_time)
    relevance = _relevance_for_setup(bundle, setup_id, decision_time)
    patterns, events = _merge_known_context(patterns, events, relevance, decision_time)

    feature_values = [
        *market_state,
        *_direction_and_timeframe_features(setup),
        *_setup_level_features(
            setup,
            point_in_time_candles,
            decision_time,
            feature_functions,
        ),
        *_pattern_event_features(patterns, events, relevance, decision_time),
    ]

    availability = market_metadata.get("timeframes", {})
    for timeframe in SUPPORTED_TIMEFRAMES:
        info = availability.get(timeframe, {}) if isinstance(availability, Mapping) else {}
        feature_values.append(1.0 if info.get("available") else 0.0)
    for timeframe in SUPPORTED_TIMEFRAMES:
        info = availability.get(timeframe, {}) if isinstance(availability, Mapping) else {}
        count = _finite_float(info.get("closed_candle_count")) or 0.0
        lookback = max(_finite_float(info.get("configured_lookback")) or 1.0, 1.0)
        feature_values.append(_clip(count / lookback, 0.0, 1.0))

    array = np.asarray([_clip(value) for value in feature_values], dtype=np.float32)
    if array.shape != (FEATURE_COUNT,):
        raise DatasetValidationError(
            f"Setup {setup_id} produced {array.size} features; expected {FEATURE_COUNT}."
        )
    if not np.all(np.isfinite(array)):
        raise DatasetValidationError(f"Setup {setup_id} contains non-finite features.")

    metadata = {
        "id": setup_id,
        "decision_time": decision_time,
        "entry_time": entry_time,
        "setup_timeframe": _normalize_timeframe(setup.get("setup_timeframe")),
        "entry_timeframe": _normalize_timeframe(setup.get("entry_timeframe")),
        "feature_candle_counts": {
            timeframe: int(
                availability.get(timeframe, {}).get("closed_candle_count", 0)
            )
            for timeframe in SUPPORTED_TIMEFRAMES
        },
        "known_pattern_count": len(patterns),
        "known_event_count": len(events),
    }
    return array, metadata



# ============================================================================
# LABEL ALIGNMENT
# ============================================================================

def _label_outcome_name(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = _finite_float(value)

        if number == OUTCOME_TP_FIRST:
            return "TP"

        if number == OUTCOME_SL_FIRST:
            return "SL"

        return None

    text = str(value or "").strip().upper()
    text = text.replace("-", "_").replace(" ", "_")

    if text in {"TP", "TP_FIRST", "TP_BEFORE_SL", "WIN", "WON", "1"}:
        return "TP"

    if text in {"SL", "SL_FIRST", "SL_BEFORE_TP", "LOSS", "LOSING", "0"}:
        return "SL"

    return None


def _labels_from_inputs(
    setups: Sequence[Mapping[str, Any]],
    *,
    bundle: Optional[Mapping[str, Any]],
    identity: Optional[Mapping[str, str]],
    explicit_labels: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build labels from authoritative setup_results using one ID and end_time."""

    if explicit_labels is not None:
        source_records = explicit_labels
        source_name = "EXPLICIT_LABELS"

    elif isinstance(bundle, Mapping) and "setup_results" in bundle:
        source_records = bundle.get("setup_results") or []
        source_name = "SETUP_RESULTS"

    elif isinstance(bundle, Mapping) and "setup_labels" in bundle:
        source_records = bundle.get("setup_labels") or []
        source_name = "SETUP_LABELS"

    else:
        source_records = setups
        source_name = "SETUP_RECORDS"

    if (
        isinstance(source_records, (str, bytes))
        or not isinstance(source_records, Sequence)
    ):
        raise DatasetValidationError(
            "Setup result records must be a sequence."
        )

    result: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    for index, source in enumerate(source_records):

        if not isinstance(source, Mapping):
            excluded.append({
                "index": index,
                "status": "INVALID",
                "reason": "Setup result must be a mapping.",
            })
            continue

        try:
            setup_id = _setup_identifier(source)
        except DatasetValidationError as error:
            excluded.append({
                "index": index,
                "status": "INVALID",
                "reason": str(error),
            })
            continue

        raw_outcome = source.get("actual_outcome")

        if raw_outcome is None:
            raw_outcome = source.get("result", source.get("outcome_name"))

        outcome_name = _label_outcome_name(raw_outcome)

        if outcome_name is None:
            excluded.append({
                "id": setup_id,
                "status": "UNLABELLED",
                "reason": "Only TP and SL results are eligible for supervised training.",
            })
            continue

        entry_time = _setup_entry_time(source)
        end_time = parse_timestamp(source.get("end_time"))

        if (
            entry_time is None
            or end_time is None
            or end_time < entry_time
        ):
            excluded.append({
                "id": setup_id,
                "status": "INVALID",
                "reason": "A valid entry_time and end_time are required.",
            })
            continue

        if source.get("training_eligible") is False:
            excluded.append({
                "id": setup_id,
                "status": "NOT_ELIGIBLE",
                "reason": "The source record is not marked as training eligible.",
            })
            continue

        label_identity: Dict[str, str] = {}

        if identity is not None:
            nested = source.get("identity")
            nested_identity = nested if isinstance(nested, Mapping) else {}

            for field_name in _IDENTITY_FIELDS:
                supplied = source.get(
                    field_name,
                    nested_identity.get(field_name),
                )

                if supplied is not None:
                    normalized = (
                        str(supplied).strip().upper()
                        if field_name == "market"
                        else str(supplied).strip()
                    )

                    if normalized != identity[field_name]:
                        raise DatasetValidationError(
                            f"Setup {setup_id} has a conflicting {field_name}."
                        )

                label_identity[field_name] = identity[field_name]

        numeric_label = (
            OUTCOME_TP_FIRST
            if outcome_name == "TP"
            else OUTCOME_SL_FIRST
        )

        label = {
            "id": int(setup_id),
            "entry_time": entry_time,
            "end_time": end_time,

            # Temporary compatibility with the current model.py interface.
            # Both timestamps represent the same end_time.
            "outcome_known_at": end_time,

            "actual_outcome": int(outcome),
            "outcome_name": label.get("outcome_name"),
            "label_target": DATASET_TARGET,
            "label_source": label.get("label_source"),
        }

        previous = seen.get(setup_id)

        if previous is not None:
            if (
                previous.get("actual_outcome") != label.get("actual_outcome")
                or previous.get("entry_time") != label.get("entry_time")
                or previous.get("end_time") != label.get("end_time")
            ):
                raise DatasetValidationError(
                    f"Conflicting result records were supplied for id {setup_id}."
                )

            continue

        seen[setup_id] = label
        result.append(label)

    report = {
        "source": source_name,
        "candidate_setup_count": len(source_records),
        "label_count": len(result),
        "excluded_count": len(excluded),
        "excluded": excluded,
    }

    return result, report


def _label_key(label: Mapping[str, Any]) -> str:
    return _setup_identifier(label)


def _align_label(
    setup: Mapping[str, Any],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:

    label = labels_by_id.get(_setup_identifier(setup))

    return dict(label) if label is not None else None

    
# ============================================================================
# CHRONOLOGICAL DATASET SPLITTING
# ============================================================================

def _validate_split_fractions(fractions: Sequence[float]) -> Tuple[float, float, float]:
    if not isinstance(fractions, Sequence) or len(fractions) != 3:
        raise DatasetValidationError("split_fractions must contain train, validation and test fractions.")
    values = tuple(_finite_float(value) for value in fractions)
    if any(value is None or value <= 0.0 for value in values):
        raise DatasetValidationError("All split fractions must be finite and positive.")
    total = sum(float(value) for value in values if value is not None)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise DatasetValidationError("split_fractions must sum to one.")
    normalized = tuple(float(value) for value in values)
    return normalized[0], normalized[1], normalized[2]


def _chronological_purged_split(
    metadata: Sequence[Mapping[str, Any]],
    fractions: Sequence[float],
) -> Dict[str, Any]:
    count = len(metadata)
    train_fraction, validation_fraction, test_fraction = _validate_split_fractions(fractions)
    ordered = sorted(
        range(count),
        key=lambda index: (
            int(metadata[index].get("decision_time", 0)),
            str(metadata[index].get("id", "")),
        ),
    )
    if count < 3:
        return {
            "mode": "CHRONOLOGICAL_PURGED",
            "train_indices": ordered,
            "validation_indices": [],
            "test_indices": [],
            "purged_train_count": 0,
            "purged_validation_count": 0,
            "warnings": ["At least three labelled examples are required for chronological train/validation/test splits."],
        }

    train_end = max(1, min(count - 2, int(math.floor(count * train_fraction))))
    validation_end = max(train_end + 1, min(count - 1, int(math.floor(count * (train_fraction + validation_fraction)))))
    train_candidates = ordered[:train_end]
    validation_candidates = ordered[train_end:validation_end]
    test_indices = ordered[validation_end:]
    validation_start = int(metadata[validation_candidates[0]]["decision_time"])
    test_start = int(metadata[test_indices[0]]["decision_time"])

    train_indices = [
        index
        for index in train_candidates
        parse_timestamp(metadata[index].get("end_time")) is not None
        and int(metadata[index]["end_time"]) < validation_start
    ]
    validation_indices = [
        index
        for index in validation_candidates
        parse_timestamp(metadata[index].get("end_time")) is not None
        and int(metadata[index]["end_time"]) < test_start
    ]

    warnings: List[str] = []
    if not train_indices:
        warnings.append("The chronological purge removed every training example.")
    if not validation_indices:
        warnings.append("The chronological purge removed every validation example.")
    if not test_indices:
        warnings.append("The test split is empty.")

    return {
        "mode": "CHRONOLOGICAL_PURGED",
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "purged_train_count": len(train_candidates) - len(train_indices),
        "purged_validation_count": len(validation_candidates) - len(validation_indices),
        "boundary_times": {
            "validation_start": validation_start,
            "test_start": test_start,
        },
        "warnings": warnings,
    }


# ============================================================================
# DATASET CONSTRUCTION
# ============================================================================

def _extract_bundle_parts(
    data_bundle: Optional[Mapping[str, Any]],
    setups: Optional[Sequence[Any]],
    candles_by_timeframe: Optional[Mapping[str, Sequence[Any]]],
    market: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]], Optional[Dict[str, Any]], Optional[str]]:
    if data_bundle is not None and not isinstance(data_bundle, Mapping):
        raise DatasetValidationError("data_bundle must be a mapping.")

    bundle = dict(data_bundle) if isinstance(data_bundle, Mapping) else None
    if bundle is not None:
        learning = bundle.get("learning_setups", {})
        if setups is None and isinstance(learning, Mapping):
            normalized = learning.get("normalized", [])
            if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes)):
                setups = normalized
        input_records = bundle.get("setup_input_records")
        setup_records: List[Dict[str, Any]] = []
        if isinstance(input_records, Mapping):
            for key, record in input_records.items():
                if isinstance(record, Mapping):
                    normalized_record = dict(record)
                    normalized_record.setdefault("id", str(key))
                    setup_records.append(normalized_record)
        if not setup_records and setups is not None:
            setup_records = [_record_dict(item) for item in setups]

        historical = bundle.get("historical_data", {})
        if candles_by_timeframe is None and isinstance(historical, Mapping):
            candles_by_timeframe = historical.get("candles_by_timeframe", {})
        if market is None and bundle.get("market") is not None:
            market = str(bundle.get("market"))
        return (
            setup_records,
            _normalize_candle_map(candles_by_timeframe),
            bundle,
            market,
        )

    if setups is None:
        raise DatasetValidationError("setups or data_bundle must be supplied.")
    if isinstance(setups, (str, bytes)) or not isinstance(setups, Sequence):
        raise DatasetValidationError("setups must be a sequence.")
    setup_records = [_record_dict(item) for item in setups]
    return (
        setup_records,
        _normalize_candle_map(candles_by_timeframe),
        None,
        market,
    )


def build_training_dataset(
    market: Optional[Any] = None,
    setups: Optional[Sequence[Any]] = None,
    candles_by_timeframe: Optional[Mapping[str, Sequence[Any]]] = None,
    *,
    data_bundle: Optional[Mapping[str, Any]] = None,
    identity: Optional[Mapping[str, Any]] = None,
    labels: Optional[Sequence[Mapping[str, Any]]] = None,
    relevance_bundle: Optional[Mapping[str, Any]] = None,
    split_fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    include_unlabelled_candidates: bool = True,
    require_identity: bool = False,
) -> Dict[str, Any]:
    """Build a chronological supervised dataset for the TP-before-SL target.

    The function accepts either a complete ``engine.data`` bundle or the
    legacy argument shape ``market, setups, candles_by_timeframe``. Only
    finalized, validated TP-first and SL-first trades become supervised
    examples. Open, ambiguous and unlabelled setups may be returned as
    inference candidates, but never enter the target arrays.
    """
    if isinstance(market, Mapping) and data_bundle is None:
        data_bundle = market
        market = None

    setup_records, candle_map, bundle, resolved_market = _extract_bundle_parts(
        data_bundle,
        setups,
        candles_by_timeframe,
        str(market) if market is not None else None,
    )
    identity_source = identity if identity is not None else (bundle.get("identity") if bundle else None)
    if identity_source is not None:
        resolved_identity = _normalize_identity(identity_source)
    else:
        resolved_identity = _derive_identity_from_setups(setup_records)
    if require_identity and resolved_identity is None:
        raise DatasetValidationError("A complete customer identity is required for dataset construction.")
    if resolved_identity is not None and resolved_market is not None:
        if str(resolved_market).strip().upper() != resolved_identity["market"].upper():
            raise DatasetValidationError("Dataset market does not match the supplied customer identity.")

    if bundle is not None:
        validation = bundle.get("validation", {})
        if isinstance(validation, Mapping) and validation.get("valid") is False:
            errors = validation.get("errors", [])
            raise DatasetValidationError(
                "The source training bundle is invalid: " + "; ".join(str(item) for item in errors)
            )

    point_in_time_map: Optional[Mapping[str, Any]] = None
    if bundle is not None:
        historical = bundle.get("historical_data", {})
        if isinstance(historical, Mapping):
            candidate = historical.get("point_in_time_candles_by_setup")
            if isinstance(candidate, Mapping):
                point_in_time_map = candidate

    context_bundle = dict(bundle) if bundle is not None else {}
    if isinstance(relevance_bundle, Mapping):
        relevance_map = relevance_bundle.get("by_setup", relevance_bundle.get("relevance_by_setup", {}))
        if isinstance(relevance_map, Mapping):
            context_bundle["relevance_by_setup"] = relevance_map
    if bundle is not None and isinstance(bundle.get("relevance"), Mapping):
        nested_relevance = bundle.get("relevance", {})
        if "relevance_by_setup" not in context_bundle and isinstance(nested_relevance.get("by_setup"), Mapping):
            context_bundle["relevance_by_setup"] = nested_relevance.get("by_setup")

    feature_functions = _load_feature_functions()
    normalized_setups: List[Dict[str, Any]] = []
    feature_by_setup: Dict[str, np.ndarray] = {}
    metadata_by_setup: Dict[str, Dict[str, Any]] = {}
    excluded_setups: List[Dict[str, Any]] = []

    setup_records_by_id: Dict[str, Dict[str, Any]] = {}
    for raw_setup in setup_records:
        try:
            setup_id = _setup_identifier(raw_setup)
        except DatasetValidationError as error:
            excluded_setups.append({"status": "INVALID", "reason": str(error)})
            continue
        if setup_id in setup_records_by_id:
            excluded_setups.append(
                {"setup_id": setup_id, "status": "DUPLICATE", "reason": "Duplicate setup_id was ignored."}
            )
            continue
        setup_records_by_id[setup_id] = raw_setup
        setup_errors = raw_setup.get("validation_errors", [])
        if isinstance(setup_errors, Sequence) and not isinstance(setup_errors, (str, bytes)) and setup_errors:
            excluded_setups.append(
                {"setup_id": setup_id, "status": "INVALID", "reason": "; ".join(str(item) for item in setup_errors)}
            )
            continue
        try:
            feature_vector, sample_metadata = _build_setup_features(
                raw_setup,
                bundle=context_bundle,
                candles_by_timeframe=candle_map,
                point_in_time_candles_by_setup=point_in_time_map,
                feature_functions=feature_functions,
            )
        except (DatasetError, TypeError, ValueError, KeyError) as error:
            excluded_setups.append(
                {"setup_id": setup_id, "status": "FEATURE_ERROR", "reason": str(error)}
            )
            continue
        feature_by_setup[setup_id] = feature_vector
        metadata_by_setup[setup_id] = sample_metadata
        normalized_setups.append(raw_setup)

    label_records, label_report = _labels_from_inputs(
        normalized_setups,
        bundle=bundle,
        identity=resolved_identity,
        explicit_labels=labels,
    )
    labels_by_id: Dict[str, Dict[str, Any]] = {}

    for label in label_records:

        if resolved_identity is not None and not _identity_matches(
            label,
            resolved_identity,
        ):
            raise DatasetValidationError(
                "A generated label belongs to another customer identity."
            )

        key = _label_key(label)
        existing = labels_by_id.get(key)

        if existing is not None and (
            existing.get("actual_outcome") != label.get("actual_outcome")
            or existing.get("entry_time") != label.get("entry_time")
            or existing.get("end_time") != label.get("end_time")
        ):
            raise DatasetValidationError(
                f"Conflicting labels were supplied for id {key}."
            )

        labels_by_id[key] = dict(label)

    training_features: List[np.ndarray] = []
    training_targets: List[int] = []
    sample_metadata: List[Dict[str, Any]] = []
    label_metadata: List[Dict[str, Any]] = []
    used_labels: set[Tuple[str, str]] = set()
    unlabelled_candidates: List[Dict[str, Any]] = []

    for setup_id, setup in setup_records_by_id.items():
        feature_vector = feature_by_setup.get(setup_id)
        if feature_vector is None:
            continue
        metadata = metadata_by_setup[setup_id]
        label = _align_label(setup, labels_by_id)
        if label is None:
            if include_unlabelled_candidates:
                unlabelled_candidates.append(
                    {
                        **metadata,
                        "status": str(setup.get("outcome_status", "UNLABELLED")).upper(),
                        "training_eligible": False,
                        "target": None,
                    }
                )
            continue

        entry_time = parse_timestamp(label.get("entry_time"))
        end_time = parse_timestamp(label.get("end_time"))
        decision_time = parse_timestamp(metadata.get("decision_time"))
        outcome = label.get("actual_outcome")
        if (
            entry_time is None
            or end_time is None
            or end_time < entry_time
            or decision_time is None
            or decision_time > entry_time
            or outcome not in (OUTCOME_SL_FIRST, OUTCOME_TP_FIRST)
        ):
            excluded_setups.append(
                {
                    "setup_id": setup_id,
                    "trade_id": label.get("trade_id"),
                    "status": "INVALID_LABEL_ALIGNMENT",
                    "reason": "Label timestamps or binary outcome violate the dataset contract.",
                }
            )
            continue

        if label.get("training_eligible") is False:
            continue

        training_features.append(feature_vector)
        training_targets.append(int(outcome))
        sample_metadata.append(
            {
                **metadata,
                "end_time": end_time,
            }
        )
        label_metadata.append(
            {
                "label_id": label.get("label_id"),
                "trade_id": str(label.get("trade_id")),
                "setup_id": str(label.get("setup_id")),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "outcome_known_at": outcome_known_at,
                "actual_outcome": int(outcome),
                "label_target": DATASET_TARGET,
                "label_source": label.get("label_source"),
            }
        )
        used_labels.add(_label_key(label))

    if not training_features:
        feature_matrix = np.empty((0, FEATURE_COUNT), dtype=np.float32)
        target_array = np.empty((0,), dtype=np.int64)
    else:
        feature_matrix = np.vstack(training_features).astype(np.float32, copy=False)
        target_array = np.asarray(training_targets, dtype=np.int64)

    split = _chronological_purged_split(sample_metadata, split_fractions)
    target_counts = {
        "SL_FIRST": int(np.sum(target_array == OUTCOME_SL_FIRST)),
        "TP_FIRST": int(np.sum(target_array == OUTCOME_TP_FIRST)),
    }
    decision_times = [int(item["decision_time"]) for item in sample_metadata]
    known_times = [int(item["outcome_known_at"]) for item in label_metadata]
    feature_schema_hash = hashlib.sha256(
        "\n".join(FEATURE_NAMES).encode("utf-8")
    ).hexdigest()
    dataset_payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash,
        "identity": resolved_identity,
        "market": resolved_market,
        "sample_keys": [
            [item.get("trade_id"), item.get("setup_id"), item.get("decision_time")]
            for item in sample_metadata
        ],
        "targets": target_array.tolist(),
    }
    dataset_id = hashlib.sha256(
        json.dumps(dataset_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:24]

    warnings = list(split.get("warnings", []))
    if resolved_identity is None:
        warnings.append(
            "The input was not accompanied by a complete customer identity. The caller must ensure the setups were already isolated."
        )
    if not len(training_features):
        warnings.append("No finalized TP-first or SL-first trades were eligible for supervised training.")
    if target_counts["SL_FIRST"] == 0 or target_counts["TP_FIRST"] == 0:
        warnings.append("The labelled dataset contains only one outcome class.")
    if len(training_features) < 20:
        warnings.append("The labelled sample count is small; validation metrics will have high uncertainty.")

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "target": DATASET_TARGET,
        "target_mapping": {"SL_FIRST": OUTCOME_SL_FIRST, "TP_FIRST": OUTCOME_TP_FIRST},
        "identity": resolved_identity,
        "market": resolved_market,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": feature_matrix,
        "targets": target_array,
        "labels": target_array.copy(),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "feature_count": FEATURE_COUNT,
            "feature_names": list(FEATURE_NAMES),
            "sha256": feature_schema_hash,
            "categorical_pattern_encoding": "STABLE_HASH_BUCKETS_AND_CATEGORY_ONE_HOT",
        },
        "sample_metadata": sample_metadata,
        "label_metadata": label_metadata,
        "candidate_metadata": unlabelled_candidates,
        "candidate_features": np.vstack(
            [feature_by_setup[str(item["setup_id"])] for item in unlabelled_candidates]
        ).astype(np.float32, copy=False) if unlabelled_candidates else np.empty((0, FEATURE_COUNT), dtype=np.float32),
        "candidate_feature_metadata": list(unlabelled_candidates),
        "all_setup_features": np.vstack(
            [feature_by_setup[key] for key in metadata_by_setup]
        ).astype(np.float32, copy=False) if metadata_by_setup else np.empty((0, FEATURE_COUNT), dtype=np.float32),
        "all_setup_feature_metadata": [metadata_by_setup[key] for key in metadata_by_setup],
        "splits": split,
        "stats": {
            "setup_count": len(setup_records_by_id),
            "feature_ready_setup_count": len(feature_by_setup),
            "labelled_example_count": int(len(training_features)),
            "unlabelled_candidate_count": len(unlabelled_candidates),
            "excluded_setup_count": len(excluded_setups),
            "label_count": len(label_records),
            "used_label_count": len(used_labels),
            "target_counts": target_counts,
            "feature_count": FEATURE_COUNT,
            "decision_time_range": {
                "start": min(decision_times) if decision_times else None,
                "end": max(decision_times) if decision_times else None,
            },
            "outcome_known_time_range": {
                "start": min(known_times) if known_times else None,
                "end": max(known_times) if known_times else None,
            },
            "split_counts": {
                "train": len(split.get("train_indices", [])),
                "validation": len(split.get("validation_indices", [])),
                "test": len(split.get("test_indices", [])),
            },
            "purged_train_count": int(split.get("purged_train_count", 0)),
            "purged_validation_count": int(split.get("purged_validation_count", 0)),
        },
        "label_report": label_report,
        "excluded_setups": excluded_setups,
        "warnings": warnings,
        "model_target_schema": "TP_BEFORE_SL_BINARY",
        "legacy_multitask_model_compatible": False,
        "requires_model_update": True,
    }


# ============================================================================
# UNLABELLED NO-TRADE CANDIDATES
# ============================================================================

def make_negative_times(
    candles_by_timeframe: Optional[Mapping[str, Sequence[Any]]] = None,
    *,
    market: Optional[str] = None,
    setups: Optional[Sequence[Any]] = None,
    timeframe: Optional[str] = None,
    sample_every: int = DEFAULT_NEGATIVE_SAMPLE_EVERY,
    exclusion_seconds: int = DEFAULT_NEGATIVE_EXCLUSION_SECONDS,
    start_time: Any = None,
    end_time: Any = None,
) -> List[Dict[str, Any]]:
    """Create deterministic unlabelled no-trade decision timestamps.

    These records are not labelled as SL-first. They represent market-only
    candidate contexts and must not be mixed into the binary TP-before-SL
    target unless a separate, explicitly defined WAIT/NO-TRADE target is added.
    """
    step = _finite_int(sample_every)
    exclusion = _finite_int(exclusion_seconds)
    if step is None or step <= 0:
        raise DatasetValidationError("sample_every must be a positive integer.")
    if exclusion is None or exclusion < 0:
        raise DatasetValidationError("exclusion_seconds must be a non-negative integer.")

    candle_map = _normalize_candle_map(candles_by_timeframe)
    selected_timeframe = _normalize_timeframe(timeframe)
    if selected_timeframe is None:
        available = [
            item for item in SUPPORTED_TIMEFRAMES if candle_map.get(item)
        ]
        if not available:
            return []
        selected_timeframe = min(available, key=lambda item: TIMEFRAME_SECONDS[item])

    begin = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    positive_times: List[int] = []
    active_intervals: List[Tuple[int, int]] = []
    for item in setups or ():
        record = _record_dict(item)
        entry = _setup_entry_time(record)
        if entry is not None:
            positive_times.append(entry)
        end_time = parse_timestamp(record.get("end_time"))
        if entry is not None and end_time is not None and end_time >= entry:
            active_intervals.append((entry, end_time))

    duration = int(TIMEFRAME_SECONDS[selected_timeframe])
    candles = candle_map.get(selected_timeframe, [])
    output: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for index in range(0, len(candles), step):
        candle_time = _normalize_candle_time(candles[index])
        if candle_time is None:
            continue
        decision_time = candle_time + duration
        if begin is not None and decision_time < begin:
            continue
        if end is not None and decision_time > end:
            continue
        if decision_time in seen:
            continue
        seen.add(decision_time)
        if any(abs(decision_time - positive_time) <= exclusion for positive_time in positive_times):
            continue
        if any(start <= decision_time <= stop for start, stop in active_intervals):
            continue
        output.append(
            {
                "market": str(market).strip().upper() if market else None,
                "timeframe": selected_timeframe,
                "decision_time": decision_time,
                "source_candle_time": candle_time,
                "sample_type": "UNLABELLED_NO_TRADE_CONTEXT",
                "target": None,
                "training_eligible": False,
                "exclusion_window_seconds": exclusion,
            }
        )

    return output


# ============================================================================
# DATASET VALIDATION AND EXPORT
# ============================================================================

def validate_training_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate feature/target dimensions, finite values and target semantics."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(dataset, Mapping):
        return {"valid": False, "errors": ["Dataset must be a mapping."], "warnings": []}

    features = np.asarray(dataset.get("features", []), dtype=np.float32)
    targets = np.asarray(dataset.get("targets", []), dtype=np.int64).reshape(-1)
    if features.ndim != 2:
        errors.append("features must be a two-dimensional matrix.")
    else:
        if features.shape[1] != len(dataset.get("feature_names", [])):
            errors.append("Feature matrix width does not match feature_names.")
        if features.shape[0] != targets.shape[0]:
            errors.append("Feature row count does not match target count.")
        if not np.all(np.isfinite(features)):
            errors.append("Feature matrix contains NaN or infinity.")
    if targets.size and not np.all(np.isin(targets, (OUTCOME_SL_FIRST, OUTCOME_TP_FIRST))):
        errors.append("Targets must contain only SL_FIRST=0 or TP_FIRST=1.")
    if targets.size == 0:
        stats = dataset.get("stats", {})
        stats = stats if isinstance(stats, Mapping) else {}

        errors.append(
            "No supervised TP-before-SL examples are available. "
            f"Setups={stats.get('setup_count', 0)}, "
            f"feature_ready={stats.get('feature_ready_setup_count', 0)}, "
            f"labelled={stats.get('labelled_example_count', 0)}, "
            f"unlabelled={stats.get('unlabelled_candidate_count', 0)}, "
            f"excluded={stats.get('excluded_setup_count', 0)}."
        )
    if dataset.get("target") != DATASET_TARGET:
        errors.append("Dataset target must be TP_BEFORE_SL.")
    metadata = dataset.get("sample_metadata", [])
    label_metadata = dataset.get("label_metadata", [])
    if len(metadata) != targets.shape[0] or len(label_metadata) != targets.shape[0]:
        errors.append("Sample and label metadata must align one-to-one with target rows.")
    for index, item in enumerate(label_metadata):
        if not isinstance(item, Mapping):
            errors.append(f"Label metadata row {index} is not a mapping.")
            continue
        entry = parse_timestamp(item.get("entry_time"))
        end_time = parse_timestamp(item.get("end_time"))
        known_at = parse_timestamp(item.get("outcome_known_at", end_time))
        decision = (
            parse_timestamp(metadata[index].get("decision_time"))
            if index < len(metadata)
            else None
        )

        if entry is None or end_time is None or known_at is None or decision is None:
            errors.append(f"Label metadata row {index} has missing timestamps.")
            continue

        if end_time < entry or known_at != end_time or decision > entry:
            errors.append(f"Label metadata row {index} violates chronology.")
            
        if item.get("actual_outcome") not in (OUTCOME_SL_FIRST, OUTCOME_TP_FIRST):
            errors.append(f"Label metadata row {index} has an invalid binary target.")

    split = dataset.get("splits", {})
    if isinstance(split, Mapping):
        all_indices: List[int] = []
        for name in ("train_indices", "validation_indices", "test_indices"):
            values = split.get(name, [])
            if not isinstance(values, Sequence):
                errors.append(f"splits.{name} must be a sequence.")
                continue
            for value in values:
                index = _finite_int(value)
                if index is None or index < 0 or index >= targets.shape[0]:
                    errors.append(f"splits.{name} contains an invalid index.")
                else:
                    all_indices.append(index)
        if len(all_indices) != len(set(all_indices)):
            errors.append("Chronological split indices overlap.")

    if targets.size == 0:
        warnings.append("No supervised examples are available.")
    unique_targets = set(int(item) for item in targets.tolist())
    if targets.size and len(unique_targets) < 2:
        warnings.append("Only one outcome class is present.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "sample_count": int(targets.shape[0]),
        "feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
        "target_counts": {
            "SL_FIRST": int(np.sum(targets == OUTCOME_SL_FIRST)),
            "TP_FIRST": int(np.sum(targets == OUTCOME_TP_FIRST)),
        },
    }


def dataset_to_serializable(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert NumPy arrays in a dataset to JSON-compatible lists."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping.")
    result: Dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, Mapping):
            result[key] = dict(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "DATASET_TARGET",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "PATTERN_HASH_BUCKETS",
    "EVENT_HASH_BUCKETS",
    "DatasetError",
    "DatasetValidationError",
    "DatasetDependencyError",
    "parse_timestamp",
    "build_training_dataset",
    "make_negative_times",
    "validate_training_dataset",
    "dataset_to_serializable",
]
