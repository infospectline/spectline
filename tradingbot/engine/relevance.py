"""Point-in-time matching of shared market patterns and events to customer setups."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

RELEVANCE_SCHEMA_VERSION = 1
DEFAULT_RECENCY_HALF_LIFE_BARS = 24.0
DEFAULT_MAX_PATTERNS_PER_SETUP: Optional[int] = None
DEFAULT_MAX_EVENTS_PER_SETUP: Optional[int] = None
TIMEFRAME_SECONDS: Dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}
TIMEFRAME_ORDER: Dict[str, int] = {
    timeframe: index for index, timeframe in enumerate(TIMEFRAME_SECONDS)
}
IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")
_PATTERN_ID_PATTERN = re.compile(r"^[0-9]{4}$")
_EXCLUDED_FEATURE_PARTS = (
    "timestamp",
    "volume",
    "identifier",
    "pattern_id",
    "record_id",
    "occurrence_id",
)
_RAW_PRICE_FEATURES = frozenset(
    {
        "price",
        "price_min",
        "price_max",
        "price_range",
        "net_price_change",
        "pivot_prices",
    }
)


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class RelevanceError(RuntimeError):
    """Base exception for setup-to-market relevance processing."""


class RelevanceValidationError(RelevanceError, ValueError):
    """Raised when a relevance input violates the expected data contract."""


@dataclass(frozen=True)
class RelevanceConfig:
    """Configuration for ranking patterns and events around a customer setup."""

    recency_half_life_bars: float = DEFAULT_RECENCY_HALF_LIFE_BARS
    pre_entry_lookback_bars: Optional[int] = None
    max_patterns_per_setup: Optional[int] = DEFAULT_MAX_PATTERNS_PER_SETUP
    max_events_per_setup: Optional[int] = DEFAULT_MAX_EVENTS_PER_SETUP
    minimum_relevance_score: float = 0.0
    include_during_trade: bool = True
    include_patterns: bool = True
    include_market_events: bool = True
    recency_weight: float = 0.30
    timeframe_weight: float = 0.10
    feature_weight: float = 0.50
    direction_weight: float = 0.10
    unmatched_timeframe_score: float = 0.50

    def __post_init__(self) -> None:
        half_life = _finite_float(self.recency_half_life_bars)
        if half_life is None or half_life <= 0.0:
            raise RelevanceValidationError(
                "recency_half_life_bars must be a positive finite number."
            )
        object.__setattr__(self, "recency_half_life_bars", half_life)

        if self.pre_entry_lookback_bars is not None:
            lookback = _non_negative_integer(
                self.pre_entry_lookback_bars,
                "pre_entry_lookback_bars",
            )
            object.__setattr__(self, "pre_entry_lookback_bars", lookback)

        for field_name in ("max_patterns_per_setup", "max_events_per_setup"):
            value = getattr(self, field_name)
            if value is not None:
                maximum = _positive_integer(value, field_name)
                object.__setattr__(self, field_name, maximum)

        minimum = _finite_float(self.minimum_relevance_score)
        if minimum is None or not 0.0 <= minimum <= 1.0:
            raise RelevanceValidationError(
                "minimum_relevance_score must be between zero and one."
            )
        object.__setattr__(self, "minimum_relevance_score", minimum)

        weight_names = (
            "recency_weight",
            "timeframe_weight",
            "feature_weight",
            "direction_weight",
        )
        for field_name in weight_names:
            weight = _finite_float(getattr(self, field_name))
            if weight is None or weight < 0.0:
                raise RelevanceValidationError(
                    f"{field_name} must be a finite non-negative number."
                )
            object.__setattr__(self, field_name, weight)
        if sum(getattr(self, name) for name in weight_names) <= 0.0:
            raise RelevanceValidationError("At least one relevance weight must be positive.")

        timeframe_score = _finite_float(self.unmatched_timeframe_score)
        if timeframe_score is None or not 0.0 <= timeframe_score <= 1.0:
            raise RelevanceValidationError(
                "unmatched_timeframe_score must be between zero and one."
            )
        object.__setattr__(self, "unmatched_timeframe_score", timeframe_score)


@dataclass(frozen=True)
class SetupRelevance:
    """Ranked market context for one setup, split by point-in-time availability."""

    setup_id: str
    trade_id: str
    market: str
    decision_time: Optional[int]
    entry_time: Optional[int]
    exit_time: Optional[int]
    patterns_known_at_entry: Tuple[Dict[str, Any], ...]
    patterns_during_trade: Tuple[Dict[str, Any], ...]
    events_known_at_entry: Tuple[Dict[str, Any], ...]
    events_during_trade: Tuple[Dict[str, Any], ...]
    summary: Dict[str, Any]

    @property
    def prediction_pattern_ids(self) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(item["pattern_id"])
                for item in self.patterns_known_at_entry
                if item.get("pattern_id") is not None
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "trade_id": self.trade_id,
            "market": self.market,
            "decision_time": self.decision_time,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "patterns_known_at_entry": list(self.patterns_known_at_entry),
            "patterns_during_trade": list(self.patterns_during_trade),
            "events_known_at_entry": list(self.events_known_at_entry),
            "events_during_trade": list(self.events_during_trade),
            "prediction_pattern_ids": list(self.prediction_pattern_ids),
            "summary": dict(self.summary),
        }


# ============================================================================
# VALUE AND TIME NORMALIZATION
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
    return int(number)


def _positive_integer(value: Any, field_name: str) -> int:
    number = _finite_int(value)
    if number is None or number <= 0:
        raise RelevanceValidationError(f"{field_name} must be a positive integer.")
    return number


def _non_negative_integer(value: Any, field_name: str) -> int:
    number = _finite_int(value)
    if number is None or number < 0:
        raise RelevanceValidationError(
            f"{field_name} must be a non-negative integer."
        )
    return number


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert Unix seconds, Unix milliseconds, datetime or ISO text to UTC seconds."""
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
        number = _finite_float(value)
        if number is None:
            return None
        if abs(number) >= 100_000_000_000:
            number /= 1000.0
        try:
            return int(number)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
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
        return parse_timestamp(parsed)
    return None


def _record_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        try:
            result = asdict(value)
        except (TypeError, ValueError):
            result = {}
        for property_name in ("decision_time", "entry_time", "is_open", "is_closed"):
            try:
                property_value = getattr(value, property_name)
            except Exception:
                continue
            if not callable(property_value):
                result[property_name] = property_value
        return result
    if hasattr(value, "__dict__"):
        result = dict(vars(value))
        for property_name in ("decision_time", "entry_time", "is_open", "is_closed"):
            try:
                property_value = getattr(value, property_name)
            except Exception:
                continue
            if not callable(property_value):
                result[property_name] = property_value
        return result
    raise TypeError("Each setup, pattern and market event must be a mapping or record object.")


def _first_timestamp(record: Mapping[str, Any], names: Sequence[str]) -> Optional[int]:
    for name in names:
        if name in record:
            parsed = parse_timestamp(record.get(name))
            if parsed is not None:
                return parsed
    return None


def _normalize_market(value: Any) -> str:
    market = str(value or "").strip().upper()
    if not market:
        raise RelevanceValidationError("market is required for relevance matching.")
    return market


def _normalize_timeframe(value: Any) -> Optional[str]:
    if value is None:
        return None
    timeframe = str(value).strip().upper()
    return timeframe if timeframe in TIMEFRAME_SECONDS else None


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    direction = str(value).strip().upper()
    aliases = {
        "BUY": "LONG",
        "BULL": "LONG",
        "BULLISH": "LONG",
        "UP": "LONG",
        "UPWARD": "LONG",
        "1": "LONG",
        "SELL": "SHORT",
        "BEAR": "SHORT",
        "BEARISH": "SHORT",
        "DOWN": "SHORT",
        "DOWNWARD": "SHORT",
        "-1": "SHORT",
    }
    normalized = aliases.get(direction, direction)
    return normalized if normalized in {"LONG", "SHORT"} else None


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return parse_timestamp(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


# ============================================================================
# SETUP AND FEATURE EXTRACTION
# ============================================================================

def _setup_times(setup: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    entry_time = _first_timestamp(
        setup,
        ("entry_time", "entry_candle_time", "entry_candle", "opened_at"),
    )
    decision_time = _first_timestamp(
        setup,
        ("decision_time", "predicted_at", "prediction_time"),
    )
    if decision_time is None:
        decision_time = entry_time
    exit_time = _first_timestamp(
        setup,
        ("exit_time", "exit_candle_time", "closed_at", "close_time"),
    )
    return decision_time, entry_time, exit_time


def _setup_identifier(setup: Mapping[str, Any]) -> Tuple[str, str]:
    setup_id_value = setup.get("setup_id", setup.get("id"))
    trade_id_value = setup.get("trade_id", setup_id_value)
    if setup_id_value is None and trade_id_value is None:
        raise RelevanceValidationError("A setup requires setup_id or trade_id.")
    setup_id = str(setup_id_value if setup_id_value is not None else trade_id_value).strip()
    trade_id = str(trade_id_value if trade_id_value is not None else setup_id).strip()
    if not setup_id or not trade_id:
        raise RelevanceValidationError("setup_id and trade_id cannot be empty.")
    return setup_id, trade_id


def _flatten_numeric_features(
    value: Any,
    *,
    prefix: str = "",
    output: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    result = output if output is not None else {}
    if not isinstance(value, Mapping):
        return result
    for raw_key, child in value.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            _flatten_numeric_features(child, prefix=path, output=result)
            continue
        number = _finite_float(child)
        if number is None:
            continue
        normalized_path = path.replace("-", "_")
        leaf_name = normalized_path.rsplit(".", 1)[-1]
        if any(part in normalized_path for part in _EXCLUDED_FEATURE_PARTS):
            continue
        if leaf_name in _RAW_PRICE_FEATURES or leaf_name.endswith("_price"):
            continue
        if leaf_name == "time" or leaf_name.endswith("_time"):
            continue
        if leaf_name == "index" or leaf_name.endswith("_index"):
            continue
        result[normalized_path] = number
    return result


def _feature_map(record: Mapping[str, Any], *, setup: bool) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if setup:
        source_names = (
            "feature_values",
            "setup_features",
            "geometry_features",
            "features",
            "pattern_geometry",
            "geometry",
            "trade_geometry",
        )
    else:
        source_names = ("features", "geometry")
    for source_name in source_names:
        source = record.get(source_name)
        if isinstance(source, Mapping):
            current = _flatten_numeric_features(source)
            for key, value in current.items():
                result.setdefault(key, value)
    return result


def _derive_setup_features(setup: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    entry = _finite_float(setup.get("entry"))
    stop = _finite_float(setup.get("sl"))
    target = _finite_float(setup.get("tp"))
    anchor = _finite_float(setup.get("anchor_price"))
    breakout = _finite_float(setup.get("breakout"))
    risk = abs(entry - stop) if entry is not None and stop is not None else None
    reward = abs(target - entry) if entry is not None and target is not None else None

    if entry is not None and stop is not None and abs(entry) > 1e-12:
        result["stop_distance_fraction"] = abs(entry - stop) / abs(entry)
    if entry is not None and target is not None and abs(entry) > 1e-12:
        result["target_distance_fraction"] = abs(target - entry) / abs(entry)
    if risk is not None and risk > 1e-12 and reward is not None:
        result["reward_risk_ratio"] = reward / risk
    if risk is not None and risk > 1e-12 and anchor is not None and entry is not None:
        result["anchor_distance_r"] = abs(entry - anchor) / risk
    if risk is not None and risk > 1e-12 and breakout is not None and entry is not None:
        result["breakout_distance_r"] = abs(entry - breakout) / risk

    zones = setup.get("zones")
    if isinstance(zones, Sequence) and not isinstance(zones, (str, bytes)):
        supports = 0
        resistances = 0
        for zone in zones:
            if not isinstance(zone, Mapping):
                continue
            zone_type = str(zone.get("type", zone.get("zone_type", ""))).upper()
            supports += int(zone_type == "SUPPORT")
            resistances += int(zone_type == "RESISTANCE")
        result["support_zone_count"] = float(supports)
        result["resistance_zone_count"] = float(resistances)
        result["total_zone_count"] = float(supports + resistances)
    return result


def _combined_setup_features(setup: Mapping[str, Any]) -> Dict[str, float]:
    features = _feature_map(setup, setup=True)
    for key, value in _derive_setup_features(setup).items():
        features.setdefault(key, value)
    raw = setup.get("raw")
    if isinstance(raw, Mapping):
        for key, value in _derive_setup_features(raw).items():
            features.setdefault(key, value)
    return features


def _pattern_explicit_ids(setup: Mapping[str, Any]) -> set[str]:
    values: List[Any] = []
    for field_name in ("linked_pattern_ids", "pattern_ids", "associated_pattern_ids"):
        value = setup.get(field_name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence):
            values.extend(value)
    trade_geometry = setup.get("trade_geometry")
    if isinstance(trade_geometry, Mapping):
        for field_name in ("pattern_id", "linked_pattern_id"):
            if trade_geometry.get(field_name) is not None:
                values.append(trade_geometry[field_name])
    return {
        str(value).strip()
        for value in values
        if _PATTERN_ID_PATTERN.fullmatch(str(value).strip())
    }


def _setup_preferred_timeframes(setup: Mapping[str, Any]) -> Tuple[str, ...]:
    result: List[str] = []
    for field_name in ("setup_timeframe", "entry_timeframe", "resolution_timeframe"):
        timeframe = _normalize_timeframe(setup.get(field_name))
        if timeframe and timeframe not in result:
            result.append(timeframe)
    return tuple(result)


def _validate_setup_identity(
    setup: Mapping[str, Any],
    identity: Optional[Mapping[str, Any]],
) -> None:
    if not identity:
        return
    for field_name in IDENTITY_FIELDS:
        supplied = setup.get(field_name)
        expected = identity.get(field_name)
        if supplied is not None and expected is not None:
            if field_name == "market":
                matches = _normalize_market(supplied) == _normalize_market(expected)
            else:
                matches = str(supplied).strip() == str(expected).strip()
            if not matches:
                raise RelevanceValidationError(
                    f"Setup {field_name} does not match the bundle identity."
                )


# ============================================================================
# SIMILARITY AND RELEVANCE SCORING
# ============================================================================

def _feature_similarity(
    setup_features: Mapping[str, float],
    market_features: Mapping[str, float],
) -> Tuple[Optional[float], int, List[str]]:
    shared = sorted(set(setup_features).intersection(market_features))
    if not shared:
        return None, 0, []
    similarities: List[float] = []
    for key in shared:
        left = float(setup_features[key])
        right = float(market_features[key])
        denominator = max(abs(left), abs(right), 1.0)
        relative_difference = abs(left - right) / denominator
        similarities.append(1.0 / (1.0 + relative_difference))
    return sum(similarities) / len(similarities), len(shared), shared


def _direction_similarity(
    setup_direction: Optional[str],
    market_direction: Optional[str],
) -> Optional[float]:
    if setup_direction is None or market_direction is None:
        return None
    return 1.0 if setup_direction == market_direction else 0.0


def _timeframe_similarity(
    market_timeframe: Optional[str],
    preferred_timeframes: Sequence[str],
    unmatched_score: float,
) -> float:
    if market_timeframe is None or not preferred_timeframes:
        return unmatched_score
    if market_timeframe in preferred_timeframes:
        return 1.0
    market_index = TIMEFRAME_ORDER.get(market_timeframe)
    preferred_indexes = [TIMEFRAME_ORDER[item] for item in preferred_timeframes]
    if market_index is None or not preferred_indexes:
        return unmatched_score
    distance = min(abs(market_index - item) for item in preferred_indexes)
    return max(0.25, 1.0 / (1.0 + 0.5 * distance))


def _recency_score(
    *,
    anchor_time: int,
    comparison_time: int,
    timeframe: Optional[str],
    half_life_bars: float,
) -> float:
    duration = TIMEFRAME_SECONDS.get(timeframe or "", 3600)
    distance_bars = max(0.0, abs(comparison_time - anchor_time) / float(duration))
    return float(2.0 ** (-distance_bars / half_life_bars))


def _weighted_score(
    recency: float,
    timeframe: float,
    feature: Optional[float],
    direction: Optional[float],
    config: RelevanceConfig,
    explicit_match: bool,
) -> float:
    if explicit_match:
        return 1.0
    components = [
        (recency, config.recency_weight),
        (timeframe, config.timeframe_weight),
    ]
    if feature is not None:
        components.append((feature, config.feature_weight))
    if direction is not None:
        components.append((direction, config.direction_weight))
    weight_total = sum(weight for _, weight in components)
    if weight_total <= 0.0:
        return 0.0
    return min(1.0, max(0.0, sum(value * weight for value, weight in components) / weight_total))


def _market_match(record: Mapping[str, Any], market: str) -> bool:
    record_market = record.get("market")
    if record_market is None:
        return True
    return _normalize_market(record_market) == market


def _pattern_identifier(record: Mapping[str, Any]) -> Optional[str]:
    pattern_id = str(record.get("pattern_id", "")).strip()
    return pattern_id if _PATTERN_ID_PATTERN.fullmatch(pattern_id) else None


def _pattern_occurrence_identifier(record: Mapping[str, Any]) -> str:
    supplied = record.get("occurrence_id", record.get("record_id"))
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    return _stable_hash(
        {
            "market": record.get("market"),
            "timeframe": record.get("timeframe"),
            "pattern_id": record.get("pattern_id"),
            "occurrence_start": record.get("occurrence_start"),
            "occurrence_end": record.get("occurrence_end"),
            "recognized_at": record.get("recognized_at"),
        }
    )


def _event_identifier(record: Mapping[str, Any]) -> str:
    supplied = record.get("event_id", record.get("record_id"))
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    return _stable_hash(
        {
            "market": record.get("market"),
            "timeframe": record.get("timeframe"),
            "event_type": record.get("event_type"),
            "event_time": record.get("event_time"),
            "recognized_at": record.get("recognized_at"),
        }
    )


def _pattern_relevance_record(
    setup: Mapping[str, Any],
    pattern: Mapping[str, Any],
    *,
    market: str,
    decision_time: int,
    exit_time: Optional[int],
    config: RelevanceConfig,
    setup_features: Mapping[str, float],
    setup_features_by_timeframe: Mapping[str, Mapping[str, float]],
    setup_direction: Optional[str],
    preferred_timeframes: Sequence[str],
    explicit_pattern_ids: set[str],
) -> Optional[Dict[str, Any]]:
    if not _market_match(pattern, market):
        return None
    pattern_id = _pattern_identifier(pattern)
    if pattern_id is None:
        return None
    occurrence_start = parse_timestamp(pattern.get("occurrence_start"))
    occurrence_end = parse_timestamp(pattern.get("occurrence_end"))
    recognized_at = parse_timestamp(pattern.get("recognized_at"))
    if (
        occurrence_start is None
        or occurrence_end is None
        or recognized_at is None
        or occurrence_end < occurrence_start
        or recognized_at < occurrence_end
    ):
        return None

    if recognized_at <= decision_time:
        availability = "KNOWN_AT_ENTRY"
        prediction_eligible = True
        relation = (
            "PRE_ENTRY_COMPLETED"
            if occurrence_end <= decision_time
            else "ONGOING_AT_DECISION"
        )
        recency_anchor = min(occurrence_end, decision_time)
        comparison_time = decision_time
        if config.pre_entry_lookback_bars is not None:
            timeframe = _normalize_timeframe(pattern.get("timeframe"))
            duration = TIMEFRAME_SECONDS.get(timeframe or "", 3600)
            if decision_time - occurrence_end > config.pre_entry_lookback_bars * duration:
                return None
    else:
        if not config.include_during_trade or exit_time is None or recognized_at > exit_time:
            return None
        availability = "DURING_TRADE"
        prediction_eligible = False
        relation = (
            "PRE_ENTRY_FORMATION_RECOGNIZED_DURING_TRADE"
            if occurrence_start < decision_time
            else "FORMATION_OCCURRED_DURING_TRADE"
        )
        recency_anchor = decision_time
        comparison_time = recognized_at

    timeframe = _normalize_timeframe(pattern.get("timeframe"))
    if timeframe is None:
        return None
    timeframe_score = _timeframe_similarity(
        timeframe,
        preferred_timeframes,
        config.unmatched_timeframe_score,
    )
    recency = _recency_score(
        anchor_time=recency_anchor,
        comparison_time=comparison_time,
        timeframe=timeframe,
        half_life_bars=config.recency_half_life_bars,
    )
    pattern_features = _feature_map(pattern, setup=False)
    comparison_features = dict(setup_features)
    if timeframe and timeframe in setup_features_by_timeframe:
        comparison_features.update(setup_features_by_timeframe[timeframe])
    feature_score, shared_count, shared_fields = _feature_similarity(
        comparison_features,
        pattern_features,
    )
    pattern_direction = _normalize_direction(
        pattern.get("direction", pattern.get("market_direction"))
    )
    direction_score = _direction_similarity(setup_direction, pattern_direction)
    explicit_match = pattern_id in explicit_pattern_ids
    score = _weighted_score(
        recency,
        timeframe_score,
        feature_score,
        direction_score,
        config,
        explicit_match,
    )
    if score < config.minimum_relevance_score:
        return None

    record = {
        "relevance_id": _stable_hash(
            {
                "setup_id": setup.get("setup_id", setup.get("id")),
                "occurrence_id": _pattern_occurrence_identifier(pattern),
                "availability": availability,
            }
        ),
        "setup_id": str(setup.get("setup_id", setup.get("id", ""))),
        "trade_id": str(setup.get("trade_id", setup.get("setup_id", setup.get("id", "")))),
        "market": market,
        "source_type": "PATTERN",
        "occurrence_id": _pattern_occurrence_identifier(pattern),
        "pattern_id": pattern_id,
        "variant_id": pattern.get("variant_id"),
        "timeframe": timeframe,
        "occurrence_start": occurrence_start,
        "occurrence_end": occurrence_end,
        "recognized_at": recognized_at,
        "available_at_entry": prediction_eligible,
        "prediction_eligible": prediction_eligible,
        "availability": availability,
        "relationship": relation,
        "relevance_score": score,
        "recency_score": recency,
        "timeframe_score": timeframe_score,
        "feature_similarity": feature_score,
        "shared_feature_count": shared_count,
        "shared_feature_fields": shared_fields,
        "direction_similarity": direction_score,
        "explicit_match": explicit_match,
        "catalog_version": pattern.get("catalog_version"),
        "pattern_geometry": _json_safe(pattern.get("geometry", {})),
        "pattern_features": _json_safe(pattern.get("features", {})),
    }
    return record


def _event_relevance_record(
    setup: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    market: str,
    decision_time: int,
    exit_time: Optional[int],
    config: RelevanceConfig,
    setup_features: Mapping[str, float],
    setup_features_by_timeframe: Mapping[str, Mapping[str, float]],
    setup_direction: Optional[str],
    preferred_timeframes: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not _market_match(event, market):
        return None
    event_time = parse_timestamp(event.get("event_time"))
    recognized_at = parse_timestamp(event.get("recognized_at", event.get("event_time")))
    if event_time is None or recognized_at is None or recognized_at < event_time:
        return None

    if recognized_at <= decision_time:
        availability = "KNOWN_AT_ENTRY"
        prediction_eligible = True
        relation = "PRE_ENTRY_EVENT" if event_time <= decision_time else "EVENT_ONGOING_AT_DECISION"
        recency_anchor = event_time
        comparison_time = decision_time
    else:
        if not config.include_during_trade or exit_time is None or recognized_at > exit_time:
            return None
        availability = "DURING_TRADE"
        prediction_eligible = False
        relation = "EVENT_RECOGNIZED_DURING_TRADE"
        recency_anchor = decision_time
        comparison_time = recognized_at

    timeframe = _normalize_timeframe(event.get("timeframe"))
    if timeframe is None:
        return None
    timeframe_score = _timeframe_similarity(
        timeframe,
        preferred_timeframes,
        config.unmatched_timeframe_score,
    )
    recency = _recency_score(
        anchor_time=recency_anchor,
        comparison_time=comparison_time,
        timeframe=timeframe,
        half_life_bars=config.recency_half_life_bars,
    )
    event_features = _feature_map(event, setup=False)
    comparison_features = dict(setup_features)
    if timeframe and timeframe in setup_features_by_timeframe:
        comparison_features.update(setup_features_by_timeframe[timeframe])
    feature_score, shared_count, shared_fields = _feature_similarity(
        comparison_features,
        event_features,
    )
    event_direction = _normalize_direction(
        event.get("direction", event_features.get("direction"))
    )
    direction_score = _direction_similarity(setup_direction, event_direction)
    score = _weighted_score(
        recency,
        timeframe_score,
        feature_score,
        direction_score,
        config,
        False,
    )
    if score < config.minimum_relevance_score:
        return None

    event_type = str(event.get("event_type", "UNKNOWN")).strip().upper()
    return {
        "relevance_id": _stable_hash(
            {
                "setup_id": setup.get("setup_id", setup.get("id")),
                "event_id": _event_identifier(event),
                "availability": availability,
            }
        ),
        "setup_id": str(setup.get("setup_id", setup.get("id", ""))),
        "trade_id": str(setup.get("trade_id", setup.get("setup_id", setup.get("id", "")))),
        "market": market,
        "source_type": "MARKET_EVENT",
        "event_id": _event_identifier(event),
        "event_type": event_type,
        "timeframe": timeframe,
        "event_time": event_time,
        "recognized_at": recognized_at,
        "available_at_entry": prediction_eligible,
        "prediction_eligible": prediction_eligible,
        "availability": availability,
        "relationship": relation,
        "relevance_score": score,
        "recency_score": recency,
        "timeframe_score": timeframe_score,
        "feature_similarity": feature_score,
        "shared_feature_count": shared_count,
        "shared_feature_fields": shared_fields,
        "direction_similarity": direction_score,
        "event_features": _json_safe(event.get("features", {})),
    }


# ============================================================================
# SETUP RELEVANCE ASSEMBLY
# ============================================================================

def build_setup_relevance(
    setup: Any,
    pattern_occurrences: Iterable[Any] = (),
    market_events: Iterable[Any] = (),
    *,
    market: Optional[str] = None,
    config: Optional[RelevanceConfig] = None,
    identity: Optional[Mapping[str, Any]] = None,
) -> SetupRelevance:
    """Rank market history for one setup without leaking post-entry information."""
    normalized_setup = _record_dict(setup)
    setup_id, trade_id = _setup_identifier(normalized_setup)
    decision_time, entry_time, exit_time = _setup_times(normalized_setup)
    if decision_time is None:
        raise RelevanceValidationError(
            f"Setup {setup_id} has no valid decision_time or entry_time."
        )
    if entry_time is not None and decision_time > entry_time:
        raise RelevanceValidationError(
            f"Setup {setup_id} decision_time cannot be later than entry_time."
        )
    if exit_time is not None and entry_time is not None and exit_time < entry_time:
        raise RelevanceValidationError(f"Setup {setup_id} exit_time precedes entry_time.")

    resolved_market = _normalize_market(
        market or normalized_setup.get("market") or (identity or {}).get("market")
    )
    _validate_setup_identity(normalized_setup, identity)
    resolved_config = config or RelevanceConfig()
    if not isinstance(resolved_config, RelevanceConfig):
        raise TypeError("config must be a RelevanceConfig instance.")

    setup_features = _combined_setup_features(normalized_setup)
    raw_context = normalized_setup.get("market_context_features_by_timeframe", {})
    setup_features_by_timeframe = (
        {
            str(key).upper(): dict(value)
            for key, value in raw_context.items()
            if isinstance(value, Mapping)
        }
        if isinstance(raw_context, Mapping)
        else {}
    )
    setup_direction = _normalize_direction(normalized_setup.get("direction"))
    preferred_timeframes = _setup_preferred_timeframes(normalized_setup)
    explicit_pattern_ids = _pattern_explicit_ids(normalized_setup)

    known_patterns: List[Dict[str, Any]] = []
    during_patterns: List[Dict[str, Any]] = []
    known_events: List[Dict[str, Any]] = []
    during_events: List[Dict[str, Any]] = []
    seen_patterns: Dict[str, Dict[str, Any]] = {}
    seen_events: Dict[str, Dict[str, Any]] = {}

    if resolved_config.include_patterns:
        for raw_pattern in pattern_occurrences:
            pattern = _record_dict(raw_pattern)
            if not _market_match(pattern, resolved_market):
                continue
            occurrence_id = _pattern_occurrence_identifier(pattern)
            prior = seen_patterns.get(occurrence_id)
            if prior is not None:
                if prior != pattern:
                    raise RelevanceValidationError(
                        f"Conflicting pattern records share occurrence_id {occurrence_id!r}."
                    )
                continue
            seen_patterns[occurrence_id] = pattern
            relevance = _pattern_relevance_record(
                normalized_setup,
                pattern,
                market=resolved_market,
                decision_time=decision_time,
                exit_time=exit_time,
                config=resolved_config,
                setup_features=setup_features,
                setup_features_by_timeframe=setup_features_by_timeframe,
                setup_direction=setup_direction,
                preferred_timeframes=preferred_timeframes,
                explicit_pattern_ids=explicit_pattern_ids,
            )
            if relevance is None:
                continue
            if relevance["prediction_eligible"]:
                known_patterns.append(relevance)
            else:
                during_patterns.append(relevance)

    if resolved_config.include_market_events:
        for raw_event in market_events:
            event = _record_dict(raw_event)
            if not _market_match(event, resolved_market):
                continue
            event_id = _event_identifier(event)
            prior = seen_events.get(event_id)
            if prior is not None:
                if prior != event:
                    raise RelevanceValidationError(
                        f"Conflicting market events share event_id {event_id!r}."
                    )
                continue
            seen_events[event_id] = event
            relevance = _event_relevance_record(
                normalized_setup,
                event,
                market=resolved_market,
                decision_time=decision_time,
                exit_time=exit_time,
                config=resolved_config,
                setup_features=setup_features,
                setup_features_by_timeframe=setup_features_by_timeframe,
                setup_direction=setup_direction,
                preferred_timeframes=preferred_timeframes,
            )
            if relevance is None:
                continue
            if relevance["prediction_eligible"]:
                known_events.append(relevance)
            else:
                during_events.append(relevance)

    sort_key = lambda item: (
        -float(item["relevance_score"]),
        -int(item["recognized_at"]),
        str(item.get("pattern_id", item.get("event_type", ""))),
    )
    known_patterns.sort(key=sort_key)
    during_patterns.sort(key=sort_key)
    known_events.sort(key=sort_key)
    during_events.sort(key=sort_key)

    if resolved_config.max_patterns_per_setup is not None:
        known_patterns = known_patterns[:resolved_config.max_patterns_per_setup]
        during_patterns = during_patterns[:resolved_config.max_patterns_per_setup]
    if resolved_config.max_events_per_setup is not None:
        known_events = known_events[:resolved_config.max_events_per_setup]
        during_events = during_events[:resolved_config.max_events_per_setup]

    summary = {
        "known_pattern_count": len(known_patterns),
        "during_trade_pattern_count": len(during_patterns),
        "known_event_count": len(known_events),
        "during_trade_event_count": len(during_events),
        "prediction_pattern_ids": list(
            dict.fromkeys(item["pattern_id"] for item in known_patterns)
        ),
        "prediction_eligible_pattern_count": sum(
            1 for item in known_patterns if item.get("prediction_eligible")
        ),
        "geometry_compared_pattern_count": sum(
            1 for item in known_patterns if item.get("shared_feature_count", 0) > 0
        ),
        "setup_feature_count": len(setup_features),
        "relevance_score_is_probability": False,
    }

    return SetupRelevance(
        setup_id=setup_id,
        trade_id=trade_id,
        market=resolved_market,
        decision_time=decision_time,
        entry_time=entry_time,
        exit_time=exit_time,
        patterns_known_at_entry=tuple(known_patterns),
        patterns_during_trade=tuple(during_patterns),
        events_known_at_entry=tuple(known_events),
        events_during_trade=tuple(during_events),
        summary=summary,
    )


def find_relevant_patterns(
    setup: Any,
    pattern_occurrences: Iterable[Any],
    *,
    market: Optional[str] = None,
    config: Optional[RelevanceConfig] = None,
    identity: Optional[Mapping[str, Any]] = None,
    include_during_trade: bool = False,
) -> List[Dict[str, Any]]:
    """Return ranked pattern links for a setup, excluding future patterns by default."""
    result = build_setup_relevance(
        setup,
        pattern_occurrences,
        (),
        market=market,
        config=config,
        identity=identity,
    )
    records = list(result.patterns_known_at_entry)
    if include_during_trade:
        records.extend(result.patterns_during_trade)
    return records


def find_relevant_events(
    setup: Any,
    market_events: Iterable[Any],
    *,
    market: Optional[str] = None,
    config: Optional[RelevanceConfig] = None,
    identity: Optional[Mapping[str, Any]] = None,
    include_during_trade: bool = False,
) -> List[Dict[str, Any]]:
    """Return ranked market-event links for a setup, excluding future events by default."""
    result = build_setup_relevance(
        setup,
        (),
        market_events,
        market=market,
        config=config,
        identity=identity,
    )
    records = list(result.events_known_at_entry)
    if include_during_trade:
        records.extend(result.events_during_trade)
    return records


# ============================================================================
# BUNDLE ADAPTERS
# ============================================================================

def _load_feature_function(function_name: str) -> Tuple[Optional[Any], Optional[str]]:
    failures: List[str] = []
    for module_name in ("engine.features", ".features"):
        try:
            if module_name.startswith("."):
                module = importlib.import_module(module_name, package=__package__)
            else:
                module = importlib.import_module(module_name)
            function = getattr(module, function_name, None)
            if callable(function):
                return function, None
            failures.append(f"{module_name} does not expose {function_name}.")
        except Exception as error:
            failures.append(f"{module_name}: {error}")
    return None, " | ".join(failures)


def _derive_point_in_time_setup_context(
    setup: Mapping[str, Any],
    setup_id: str,
    training_bundle: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    normalized = dict(setup)
    history = training_bundle.get("historical_data", {})
    point_in_time = (
        history.get("point_in_time_candles_by_setup", {})
        if isinstance(history, Mapping)
        else {}
    )
    if not isinstance(point_in_time, Mapping):
        return normalized, [f"Setup {setup_id} has no point-in-time candle mapping."]
    frame_candles = point_in_time.get(setup_id, point_in_time.get(str(setup_id), {}))
    if not isinstance(frame_candles, Mapping) or not frame_candles:
        return normalized, [f"Setup {setup_id} has no point-in-time candles for geometry comparison."]

    decision_time, _, _ = _setup_times(normalized)
    if decision_time is None:
        return normalized, [f"Setup {setup_id} has no decision timestamp for geometry comparison."]

    geometry_function, import_issue = _load_feature_function("pattern_geometry")
    if geometry_function is None:
        return normalized, [
            f"Setup {setup_id} geometry comparison is unavailable: {import_issue}"
        ]

    context_by_timeframe: Dict[str, Dict[str, float]] = {}
    geometry_by_timeframe: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    for raw_timeframe, candles in frame_candles.items():
        timeframe = _normalize_timeframe(raw_timeframe)
        if timeframe is None:
            issues.append(f"Setup {setup_id} has unsupported candle timeframe {raw_timeframe!r}.")
            continue
        if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
            issues.append(f"Setup {setup_id} has invalid {timeframe} point-in-time candles.")
            continue
        if not candles:
            continue
        try:
            geometry = geometry_function(
                candles,
                timeframe=timeframe,
                calculated_at=decision_time,
            )
        except Exception as error:
            issues.append(f"Setup {setup_id} {timeframe} geometry failed: {error}")
            continue
        if not isinstance(geometry, Mapping):
            issues.append(f"Setup {setup_id} {timeframe} geometry returned an invalid result.")
            continue
        geometry_end = parse_timestamp(geometry.get("end_time"))
        geometry_recognized = parse_timestamp(geometry.get("recognized_at"))
        if geometry_end is not None and geometry_end > decision_time:
            issues.append(f"Setup {setup_id} {timeframe} geometry extends beyond its decision time.")
            continue
        if geometry_recognized is not None and geometry_recognized > decision_time:
            issues.append(f"Setup {setup_id} {timeframe} geometry was not available at its decision time.")
            continue

        feature_values: Dict[str, float] = {}
        candle_features = geometry.get("features", {})
        if isinstance(candle_features, Mapping):
            feature_values.update(_flatten_numeric_features(candle_features))
        pivot_geometry = geometry.get("pivot_geometry", {})
        if isinstance(pivot_geometry, Mapping):
            pivot_features = pivot_geometry.get("features", {})
            if isinstance(pivot_features, Mapping):
                for key, value in _flatten_numeric_features(pivot_features).items():
                    feature_values.setdefault(key, value)
        context_by_timeframe[timeframe] = feature_values
        geometry_by_timeframe[timeframe] = _json_safe(geometry)

    normalized["market_context_features_by_timeframe"] = context_by_timeframe
    normalized["market_context_geometry_by_timeframe"] = geometry_by_timeframe
    return normalized, issues


def _extract_setup_records(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    setup_container = bundle.get("learning_setups", {})
    raw_setups: List[Any] = []
    if isinstance(setup_container, Mapping):
        normalized = setup_container.get("normalized", [])
        if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes)):
            raw_setups = list(normalized)
    elif isinstance(setup_container, Sequence) and not isinstance(setup_container, (str, bytes)):
        raw_setups = list(setup_container)

    input_records = bundle.get("setup_input_records", {})
    inputs_by_id = (
        {str(key): dict(value) for key, value in input_records.items() if isinstance(value, Mapping)}
        if isinstance(input_records, Mapping)
        else {}
    )
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_setups:
        setup = _record_dict(raw)
        setup_id, trade_id = _setup_identifier(setup)
        if setup_id in inputs_by_id:
            merged = dict(inputs_by_id[setup_id])
            merged.update(setup)
            setup = merged
        setup.setdefault("setup_id", setup_id)
        setup.setdefault("trade_id", trade_id)
        result.append(setup)
        seen.add(setup_id)

    for setup_id, input_record in inputs_by_id.items():
        if setup_id in seen:
            continue
        record = dict(input_record)
        record.setdefault("setup_id", setup_id)
        result.append(record)

    result.sort(
        key=lambda item: (
            _setup_times(item)[0] is None,
            _setup_times(item)[0] or 0,
            str(item.get("setup_id", "")),
        )
    )
    return result


def _bundle_memory_for_setup(
    bundle: Mapping[str, Any],
    setup_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    memory = bundle.get("market_memory_by_setup", {})
    if isinstance(memory, Mapping):
        item = memory.get(setup_id, memory.get(str(setup_id), {}))
        if isinstance(item, Mapping):
            patterns = item.get("patterns", [])
            events = item.get("events", [])
            normalized_patterns = [
                _record_dict(record)
                for record in patterns
                if isinstance(record, Mapping) or hasattr(record, "__dict__")
            ] if isinstance(patterns, Sequence) else []
            normalized_events = [
                _record_dict(record)
                for record in events
                if isinstance(record, Mapping) or hasattr(record, "__dict__")
            ] if isinstance(events, Sequence) else []
            if normalized_patterns or normalized_events:
                return normalized_patterns, normalized_events

    patterns_by_timeframe = bundle.get("patterns_by_timeframe", {})
    events_by_timeframe = bundle.get("events_by_timeframe", {})
    patterns: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    if isinstance(patterns_by_timeframe, Mapping):
        for records in patterns_by_timeframe.values():
            if isinstance(records, Sequence):
                patterns.extend(
                    _record_dict(record)
                    for record in records
                    if isinstance(record, Mapping) or hasattr(record, "__dict__")
                )
    if isinstance(events_by_timeframe, Mapping):
        for records in events_by_timeframe.values():
            if isinstance(records, Sequence):
                events.extend(
                    _record_dict(record)
                    for record in records
                    if isinstance(record, Mapping) or hasattr(record, "__dict__")
                )
    return patterns, events


def _validate_bundle_identity(bundle: Mapping[str, Any]) -> Tuple[Dict[str, str], str]:
    identity = bundle.get("identity")
    if not isinstance(identity, Mapping):
        raise RelevanceValidationError("Training bundle requires an identity mapping.")
    normalized: Dict[str, str] = {}
    for field_name in IDENTITY_FIELDS:
        value = identity.get(field_name)
        if value is None or not str(value).strip():
            raise RelevanceValidationError(
                f"Training bundle identity is missing {field_name}."
            )
        normalized[field_name] = str(value).strip()
    market = _normalize_market(bundle.get("market", normalized["market"]))
    if _normalize_market(normalized["market"]) != market:
        raise RelevanceValidationError("Bundle market does not match its customer identity.")
    normalized["market"] = market
    return normalized, market


def build_relevance_bundle(
    training_bundle: Mapping[str, Any],
    *,
    config: Optional[RelevanceConfig] = None,
) -> Dict[str, Any]:
    """Build a relevance section for every setup in a prepared training bundle."""
    if not isinstance(training_bundle, Mapping):
        raise TypeError("training_bundle must be a mapping.")
    identity, market = _validate_bundle_identity(training_bundle)
    resolved_config = config or RelevanceConfig()
    setups = _extract_setup_records(training_bundle)
    by_setup: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    skipped_setup_count = 0

    for setup in setups:
        setup_id, _ = _setup_identifier(setup)
        validation_errors = setup.get("validation_errors", [])
        if isinstance(validation_errors, Sequence) and not isinstance(validation_errors, (str, bytes)) and validation_errors:
            issues.append(
                f"Setup {setup_id} has invalid input fields: "
                + "; ".join(str(item) for item in validation_errors)
            )
            skipped_setup_count += 1
            continue
        patterns, events = _bundle_memory_for_setup(training_bundle, setup_id)
        setup_with_context, context_issues = _derive_point_in_time_setup_context(
            setup,
            setup_id,
            training_bundle,
        )
        issues.extend(context_issues)
        try:
            relevance = build_setup_relevance(
                setup_with_context,
                patterns,
                events,
                market=market,
                config=resolved_config,
                identity=identity,
            )
        except RelevanceValidationError as error:
            issues.append(f"Setup {setup_id}: {error}")
            skipped_setup_count += 1
            continue
        relevance_payload = relevance.to_dict()
        relevance_payload["market_context_geometry_by_timeframe"] = _json_safe(
            setup_with_context.get("market_context_geometry_by_timeframe", {})
        )
        relevance_payload["summary"]["market_context_timeframes"] = sorted(
            setup_with_context.get("market_context_features_by_timeframe", {}).keys()
        )
        by_setup[setup_id] = relevance_payload

    return {
        "schema_version": RELEVANCE_SCHEMA_VERSION,
        "identity": identity,
        "market": market,
        "config": asdict(resolved_config),
        "by_setup": by_setup,
        "report": {
            "setup_count": len(setups),
            "processed_setup_count": len(by_setup),
            "skipped_setup_count": skipped_setup_count,
            "known_pattern_count": sum(
                item["summary"]["known_pattern_count"] for item in by_setup.values()
            ),
            "during_trade_pattern_count": sum(
                item["summary"]["during_trade_pattern_count"] for item in by_setup.values()
            ),
            "known_event_count": sum(
                item["summary"]["known_event_count"] for item in by_setup.values()
            ),
            "during_trade_event_count": sum(
                item["summary"]["during_trade_event_count"] for item in by_setup.values()
            ),
            "relevance_score_is_probability": False,
            "issues": issues,
        },
    }


def attach_relevance_to_bundle(
    training_bundle: Mapping[str, Any],
    *,
    config: Optional[RelevanceConfig] = None,
) -> Dict[str, Any]:
    """Return a shallow copy of a training bundle with relevance outputs attached."""
    result = dict(training_bundle)
    relevance = build_relevance_bundle(training_bundle, config=config)
    result["relevance"] = relevance
    result["relevance_by_setup"] = relevance["by_setup"]
    return result


# ============================================================================
# MARKET STORAGE ACCESS
# ============================================================================

def _load_storage(storage: Any = None) -> Any:
    if storage is not None:
        return storage
    failures: List[str] = []
    for module_name in ("engine.storage", ".storage"):
        try:
            if module_name.startswith("."):
                module = importlib.import_module(module_name, package=__package__)
            else:
                module = importlib.import_module(module_name)
            storage_class = getattr(module, "HistoricalStorage", None)
            return storage_class() if callable(storage_class) else module
        except Exception as error:
            failures.append(f"{module_name}: {error}")
    raise RelevanceError(
        "Historical storage is unavailable. " + " | ".join(failures)
    )


def load_market_history(
    market: str,
    *,
    timeframes: Optional[Sequence[str]] = None,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    storage: Any = None,
) -> Dict[str, Any]:
    """Load shared patterns and market events without assigning customer ownership."""
    normalized_market = _normalize_market(market)
    start = parse_timestamp(start_time)
    end = parse_timestamp(end_time)
    if start_time is not None and start is None:
        raise RelevanceValidationError("start_time is invalid.")
    if end_time is not None and end is None:
        raise RelevanceValidationError("end_time is invalid.")
    if start is not None and end is not None and end < start:
        raise RelevanceValidationError("end_time cannot precede start_time.")

    if timeframes is None:
        selected_timeframes = tuple(TIMEFRAME_SECONDS)
    else:
        selected: List[str] = []
        for value in timeframes:
            timeframe = _normalize_timeframe(value)
            if timeframe is None:
                raise RelevanceValidationError(f"Unsupported timeframe: {value!r}.")
            if timeframe not in selected:
                selected.append(timeframe)
        selected_timeframes = tuple(selected)

    service = _load_storage(storage)
    pattern_iterator = getattr(service, "iter_pattern_occurrences", None)
    event_iterator = getattr(service, "iter_market_events", None)
    if not callable(pattern_iterator) or not callable(event_iterator):
        raise RelevanceError("Storage must expose iter_pattern_occurrences and iter_market_events.")

    patterns_by_timeframe: Dict[str, List[Dict[str, Any]]] = {}
    events_by_timeframe: Dict[str, List[Dict[str, Any]]] = {}
    issues: List[str] = []

    for timeframe in selected_timeframes:
        try:
            raw_patterns = pattern_iterator(
                normalized_market,
                timeframe=timeframe,
                start_time=start,
                end_time=end,
            )
            patterns_by_timeframe[timeframe] = [
                _record_dict(record) for record in raw_patterns
            ]
        except Exception as error:
            patterns_by_timeframe[timeframe] = []
            issues.append(f"Could not load {timeframe} patterns: {error}")

        try:
            raw_events = event_iterator(
                normalized_market,
                timeframe=timeframe,
                start_time=start,
                end_time=end,
            )
            events_by_timeframe[timeframe] = [
                _record_dict(record) for record in raw_events
            ]
        except Exception as error:
            events_by_timeframe[timeframe] = []
            issues.append(f"Could not load {timeframe} market events: {error}")

    patterns = [record for records in patterns_by_timeframe.values() for record in records]
    events = [record for records in events_by_timeframe.values() for record in records]
    patterns.sort(
        key=lambda item: (
            parse_timestamp(item.get("occurrence_start")) or 0,
            parse_timestamp(item.get("recognized_at")) or 0,
            str(item.get("pattern_id", "")),
        )
    )
    events.sort(
        key=lambda item: (
            parse_timestamp(item.get("event_time")) or 0,
            parse_timestamp(item.get("recognized_at")) or 0,
            str(item.get("event_type", "")),
        )
    )
    return {
        "market": normalized_market,
        "start_time": start,
        "end_time": end,
        "patterns_by_timeframe": patterns_by_timeframe,
        "events_by_timeframe": events_by_timeframe,
        "patterns": patterns,
        "events": events,
        "report": {
            "timeframes": list(selected_timeframes),
            "pattern_count": len(patterns),
            "event_count": len(events),
            "issues": issues,
        },
    }


def build_relevance_from_storage(
    setups: Sequence[Any],
    market: str,
    *,
    identity: Optional[Mapping[str, Any]] = None,
    config: Optional[RelevanceConfig] = None,
    storage: Any = None,
    timeframes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load market memory and rank it separately for each customer setup."""
    if not isinstance(setups, Sequence) or isinstance(setups, (str, bytes)):
        raise TypeError("setups must be a sequence of setup records.")
    normalized_market = _normalize_market(market)
    resolved_config = config or RelevanceConfig()
    setup_records = [_record_dict(item) for item in setups]
    normalized_identity: Optional[Dict[str, str]] = None
    if identity is not None:
        normalized_identity = {}
        for field_name in IDENTITY_FIELDS:
            value = identity.get(field_name)
            if value is None or not str(value).strip():
                raise RelevanceValidationError(
                    f"Identity is missing {field_name}."
                )
            normalized_identity[field_name] = str(value).strip()
        if _normalize_market(normalized_identity["market"]) != normalized_market:
            raise RelevanceValidationError("Identity market does not match requested market.")
        normalized_identity["market"] = normalized_market
        for setup in setup_records:
            _validate_setup_identity(setup, normalized_identity)
    decision_times = [_setup_times(item)[0] for item in setup_records]
    maximum_times: List[int] = [value for value in decision_times if value is not None]
    if resolved_config.include_during_trade:
        maximum_times.extend(
            value
            for item in setup_records
            for value in [_setup_times(item)[2]]
            if value is not None
        )
    if not maximum_times:
        raise RelevanceValidationError("No valid setup decision or exit timestamps were supplied.")
    history = load_market_history(
        normalized_market,
        timeframes=timeframes,
        end_time=max(maximum_times),
        storage=storage,
    )

    by_setup: Dict[str, Dict[str, Any]] = {}
    for setup in setup_records:
        setup_id, _ = _setup_identifier(setup)
        relevance = build_setup_relevance(
            setup,
            history["patterns"],
            history["events"],
            market=normalized_market,
            config=resolved_config,
            identity=normalized_identity,
        )
        by_setup[setup_id] = relevance.to_dict()

    return {
        "schema_version": RELEVANCE_SCHEMA_VERSION,
        "identity": normalized_identity,
        "market": normalized_market,
        "by_setup": by_setup,
        "market_history_report": history["report"],
        "report": {
            "setup_count": len(setup_records),
            "processed_setup_count": len(by_setup),
            "relevance_score_is_probability": False,
        },
    }


# ============================================================================
# VALIDATION AND PUBLIC API
# ============================================================================

def validate_relevance_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate identity isolation and point-in-time guarantees in a relevance bundle."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(bundle, Mapping):
        return {"valid": False, "errors": ["Relevance bundle must be a mapping."], "warnings": []}
    try:
        identity, market = _validate_bundle_identity(bundle)
    except RelevanceValidationError as error:
        return {"valid": False, "errors": [str(error)], "warnings": []}

    by_setup = bundle.get("by_setup")
    if not isinstance(by_setup, Mapping):
        errors.append("by_setup must be a mapping.")
        by_setup = {}

    for setup_id, relevance in by_setup.items():
        if not isinstance(relevance, Mapping):
            errors.append(f"Setup relevance {setup_id} must be a mapping.")
            continue
        cutoff = parse_timestamp(relevance.get("decision_time"))
        if _normalize_market(relevance.get("market", market)) != market:
            errors.append(f"Setup relevance {setup_id} belongs to a different market.")
        for collection_name in (
            "patterns_known_at_entry",
            "events_known_at_entry",
            "patterns_during_trade",
            "events_during_trade",
        ):
            records = relevance.get(collection_name, [])
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                errors.append(f"{setup_id}.{collection_name} must be a sequence.")
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    errors.append(f"{setup_id}.{collection_name} contains a non-mapping record.")
                    continue
                known_at = parse_timestamp(record.get("recognized_at"))
                prediction_eligible = bool(record.get("prediction_eligible"))
                if collection_name.endswith("known_at_entry"):
                    if cutoff is None or known_at is None or known_at > cutoff:
                        errors.append(
                            f"{setup_id} contains information unavailable at its decision time."
                        )
                    if not prediction_eligible:
                        errors.append(
                            f"{setup_id} contains a non-eligible record in {collection_name}."
                        )
                else:
                    if prediction_eligible:
                        errors.append(
                            f"{setup_id} marks a during-trade record as prediction eligible."
                        )
                    exit_time = parse_timestamp(relevance.get("exit_time"))
                    if exit_time is None or known_at is None or known_at > exit_time:
                        errors.append(
                            f"{setup_id} contains a market record not known before trade exit."
                        )
                if record.get("source_type") == "PATTERN":
                    occurrence_start = parse_timestamp(record.get("occurrence_start"))
                    occurrence_end = parse_timestamp(record.get("occurrence_end"))
                    if (
                        occurrence_start is None
                        or occurrence_end is None
                        or known_at is None
                        or occurrence_end < occurrence_start
                        or known_at < occurrence_end
                    ):
                        errors.append(f"{setup_id} contains a pattern with invalid recognition chronology.")
                    if not _PATTERN_ID_PATTERN.fullmatch(str(record.get("pattern_id", ""))):
                        errors.append(f"{setup_id} contains an invalid pattern ID.")
                elif record.get("source_type") == "MARKET_EVENT":
                    event_time = parse_timestamp(record.get("event_time"))
                    if event_time is None or known_at is None or known_at < event_time:
                        errors.append(f"{setup_id} contains a market event with invalid recognition chronology.")
                score = _finite_float(record.get("relevance_score"))
                if score is None or not 0.0 <= score <= 1.0:
                    errors.append(f"{setup_id} contains an invalid relevance score.")

    report = bundle.get("report", {})
    if isinstance(report, Mapping):
        warnings.extend(str(item) for item in report.get("issues", []))
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "identity": identity,
        "market": market,
        "setup_count": len(by_setup),
    }


def relevance_records_for_setup(
    relevance_bundle: Mapping[str, Any],
    setup_id: Any,
    *,
    include_during_trade: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve prediction-eligible pattern and event links for a setup."""
    by_setup = relevance_bundle.get("by_setup", {})
    if not isinstance(by_setup, Mapping):
        raise RelevanceValidationError("relevance_bundle.by_setup must be a mapping.")
    item = by_setup.get(str(setup_id))
    if not isinstance(item, Mapping):
        raise KeyError(f"No relevance record exists for setup {setup_id!r}.")
    patterns = list(item.get("patterns_known_at_entry", []))
    events = list(item.get("events_known_at_entry", []))
    if include_during_trade:
        patterns.extend(item.get("patterns_during_trade", []))
        events.extend(item.get("events_during_trade", []))
    return {"patterns": patterns, "events": events}


build_relevance = build_relevance_bundle
match_patterns_to_setups = build_relevance_from_storage
