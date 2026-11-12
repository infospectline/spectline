"""Estimate point-in-time probabilities of TP being reached before SL."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

PROBABILITY_SCHEMA_VERSION = 1
PROBABILITY_TARGET = "TP_BEFORE_SL"
OUTCOME_SL_FIRST = 0
OUTCOME_TP_FIRST = 1
IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")
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
_PATTERN_ID_PATTERN = re.compile(r"^[0-9]{4}$")
_FORBIDDEN_FEATURE_PARTS = (
    "timestamp",
    "time",
    "price",
    "entry",
    "exit",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "actual_outcome",
    "outcome",
    "result",
    "label",
    "probability",
    "prediction",
    "target",
    "pattern_id",
    "record_id",
    "trade_id",
    "setup_id",
    "user_id",
    "strategy_id",
    "tradingbot_id",
)
_OUTCOME_ALIASES = {
    "0": OUTCOME_SL_FIRST,
    "SL_FIRST": OUTCOME_SL_FIRST,
    "SL_BEFORE_TP": OUTCOME_SL_FIRST,
    "STOP_LOSS_FIRST": OUTCOME_SL_FIRST,
    "LOSS": OUTCOME_SL_FIRST,
    "LOSING": OUTCOME_SL_FIRST,
    "1": OUTCOME_TP_FIRST,
    "TP_FIRST": OUTCOME_TP_FIRST,
    "TP_BEFORE_SL": OUTCOME_TP_FIRST,
    "TAKE_PROFIT_FIRST": OUTCOME_TP_FIRST,
    "WIN": OUTCOME_TP_FIRST,
    "WON": OUTCOME_TP_FIRST,
}


# ============================================================================
# ERRORS AND CONFIGURATION
# ============================================================================

class ProbabilityError(RuntimeError):
    """Base exception for probability estimation errors."""


class ProbabilityValidationError(ProbabilityError, ValueError):
    """Raised when probability inputs violate the data contract."""


@dataclass(frozen=True)
class ProbabilityConfig:
    """Configuration for the weighted historical Beta-Binomial estimator."""

    prior_probability: float = 0.5
    prior_strength: float = 4.0
    minimum_history_count: int = 3
    minimum_effective_sample_size: float = 2.0
    minimum_similarity: float = 0.02
    global_fallback_weight: float = 0.15
    pattern_weight: float = 0.65
    feature_weight: float = 0.20
    direction_weight: float = 0.10
    timeframe_weight: float = 0.05
    recency_half_life_days: Optional[float] = None
    maximum_history_records: int = 2000
    maximum_neighbors: int = 250
    confidence_interval_z: float = 1.96

    def __post_init__(self) -> None:
        probability = _finite_float(self.prior_probability)
        if probability is None or not 0.0 < probability < 1.0:
            raise ProbabilityValidationError(
                "prior_probability must be a finite number between zero and one."
            )
        object.__setattr__(self, "prior_probability", probability)

        strength = _finite_float(self.prior_strength)
        if strength is None or strength <= 0.0:
            raise ProbabilityValidationError("prior_strength must be positive and finite.")
        object.__setattr__(self, "prior_strength", strength)

        history_count = _positive_integer(
            self.minimum_history_count,
            "minimum_history_count",
            allow_zero=False,
        )
        object.__setattr__(self, "minimum_history_count", history_count)

        effective_size = _finite_float(self.minimum_effective_sample_size)
        if effective_size is None or effective_size < 0.0:
            raise ProbabilityValidationError(
                "minimum_effective_sample_size must be finite and non-negative."
            )
        object.__setattr__(self, "minimum_effective_sample_size", effective_size)

        similarity = _finite_float(self.minimum_similarity)
        if similarity is None or not 0.0 <= similarity <= 1.0:
            raise ProbabilityValidationError(
                "minimum_similarity must be between zero and one."
            )
        object.__setattr__(self, "minimum_similarity", similarity)

        fallback_weight = _finite_float(self.global_fallback_weight)
        if fallback_weight is None or not 0.0 < fallback_weight <= 1.0:
            raise ProbabilityValidationError(
                "global_fallback_weight must be greater than zero and at most one."
            )
        object.__setattr__(self, "global_fallback_weight", fallback_weight)

        component_names = (
            "pattern_weight",
            "feature_weight",
            "direction_weight",
            "timeframe_weight",
        )
        component_total = 0.0
        for field_name in component_names:
            weight = _finite_float(getattr(self, field_name))
            if weight is None or weight < 0.0:
                raise ProbabilityValidationError(
                    f"{field_name} must be finite and non-negative."
                )
            object.__setattr__(self, field_name, weight)
            component_total += weight
        if component_total <= 0.0:
            raise ProbabilityValidationError(
                "At least one similarity component weight must be positive."
            )

        if self.recency_half_life_days is not None:
            half_life = _finite_float(self.recency_half_life_days)
            if half_life is None or half_life <= 0.0:
                raise ProbabilityValidationError(
                    "recency_half_life_days must be positive or None."
                )
            object.__setattr__(self, "recency_half_life_days", half_life)

        for field_name in ("maximum_history_records", "maximum_neighbors"):
            value = _positive_integer(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)

        z_value = _finite_float(self.confidence_interval_z)
        if z_value is None or z_value <= 0.0:
            raise ProbabilityValidationError(
                "confidence_interval_z must be positive and finite."
            )
        object.__setattr__(self, "confidence_interval_z", z_value)


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


def _positive_integer(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        raise ProbabilityValidationError(f"{field_name} must be an integer.")
    resolved = int(number)
    if resolved < 0 or (resolved == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ProbabilityValidationError(f"{field_name} must be {qualifier}.")
    return resolved


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert Unix seconds, Unix milliseconds, datetime or ISO text to UTC seconds."""
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


def normalize_probability(value: Any) -> Optional[float]:
    """Normalize a probability expressed as a fraction or percentage."""
    number = _finite_float(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return float(number)


def normalize_outcome(value: Any) -> Optional[int]:
    """Normalize a decisive TP-first or SL-first trade outcome."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _finite_float(value)
        if number is None or number not in (0.0, 1.0):
            return None
        return int(number)
    return _OUTCOME_ALIASES.get(str(value).strip().upper())


def _normalize_identity(identity: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(identity, Mapping):
        raise ProbabilityValidationError("A complete customer identity mapping is required.")
    normalized: Dict[str, str] = {}
    for field_name in IDENTITY_FIELDS:
        value = identity.get(field_name)
        if value is None or not str(value).strip():
            raise ProbabilityValidationError(
                f"Customer identity is missing {field_name}."
            )
        text = str(value).strip()
        normalized[field_name] = text.upper() if field_name == "market" else text
    return normalized


def _identity_matches(
    record: Mapping[str, Any],
    expected_identity: Mapping[str, str],
) -> bool:
    for field_name in IDENTITY_FIELDS:
        supplied = record.get(field_name)
        if supplied is None or not str(supplied).strip():
            return False
        value = str(supplied).strip()
        if field_name == "market":
            value = value.upper()
        if value != expected_identity[field_name]:
            return False
    return True


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    aliases = {"BUY": "LONG", "SELL": "SHORT", "BULLISH": "LONG", "BEARISH": "SHORT"}
    text = aliases.get(text, text)
    return text if text in {"LONG", "SHORT"} else None


def _normalize_timeframe(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in TIMEFRAME_SECONDS else None


# ============================================================================
# FEATURE AND PATTERN EXTRACTION
# ============================================================================

def _feature_key_allowed(name: str) -> bool:
    normalized = name.strip().lower()
    if not normalized:
        return False
    return not any(part in normalized for part in _FORBIDDEN_FEATURE_PARTS)


def _flatten_numeric_features(
    value: Any,
    *,
    prefix: str = "",
    output: Optional[Dict[str, float]] = None,
    depth: int = 0,
) -> Dict[str, float]:
    result = output if output is not None else {}
    if depth > 5:
        return result
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            if not _feature_key_allowed(full_key):
                continue
            if isinstance(item, Mapping):
                _flatten_numeric_features(
                    item,
                    prefix=full_key,
                    output=result,
                    depth=depth + 1,
                )
            elif isinstance(item, (list, tuple)):
                for index, element in enumerate(item):
                    if isinstance(element, Mapping):
                        _flatten_numeric_features(
                            element,
                            prefix=f"{full_key}.{index}",
                            output=result,
                            depth=depth + 1,
                        )
                    else:
                        number = _finite_float(element)
                        if number is not None:
                            result[f"{full_key}.{index}"] = number
            else:
                number = _finite_float(item)
                if number is not None:
                    result[full_key] = number
    return result


def _extract_feature_map(record: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    feature_sources = (
        "features",
        "feature_vector",
        "setup_features",
        "market_context_features",
        "market_context_geometry",
        "geometry_features",
    )
    for field_name in feature_sources:
        source = record.get(field_name)
        if isinstance(source, Mapping):
            _flatten_numeric_features(source, prefix="", output=result)

    timeframe_sources = (
        "market_context_features_by_timeframe",
        "market_context_geometry_by_timeframe",
        "features_by_timeframe",
    )
    for field_name in timeframe_sources:
        source = record.get(field_name)
        if not isinstance(source, Mapping):
            continue
        for timeframe, values in source.items():
            if isinstance(values, Mapping):
                _flatten_numeric_features(
                    values,
                    prefix=str(timeframe).upper(),
                    output=result,
                )
    return result


def _normalize_pattern_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)
    return text if _PATTERN_ID_PATTERN.fullmatch(text) else None


def _extract_pattern_weights(record: Mapping[str, Any]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for field_name in ("prediction_pattern_ids", "linked_pattern_ids", "pattern_ids"):
        values = record.get(field_name)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                pattern_id = _normalize_pattern_id(value)
                if pattern_id is not None:
                    weights[pattern_id] = max(weights.get(pattern_id, 0.0), 1.0)

    for field_name in ("patterns_known_at_entry", "patterns", "relevant_patterns"):
        values = record.get(field_name)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            if isinstance(item, Mapping):
                if item.get("prediction_eligible") is False:
                    continue
                if str(item.get("availability", "KNOWN_AT_ENTRY")).upper() not in {
                    "KNOWN_AT_ENTRY",
                    "AVAILABLE_AT_ENTRY",
                    "PREDICTION_ELIGIBLE",
                }:
                    continue
                pattern_id = _normalize_pattern_id(item.get("pattern_id"))
                score = _finite_float(item.get("relevance_score"))
                weight = min(max(score if score is not None else 1.0, 0.0), 1.0)
            else:
                pattern_id = _normalize_pattern_id(item)
                weight = 1.0
            if pattern_id is not None:
                weights[pattern_id] = max(weights.get(pattern_id, 0.0), weight)
    return weights


def _extract_setup_context(
    setup: Mapping[str, Any],
    relevance_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context = dict(setup)
    if relevance_record is not None:
        if not isinstance(relevance_record, Mapping):
            raise ProbabilityValidationError("relevance_record must be a mapping.")
        for field_name in (
            "patterns_known_at_entry",
            "prediction_pattern_ids",
            "market_context_features_by_timeframe",
            "market_context_geometry_by_timeframe",
        ):
            if field_name in relevance_record:
                context[field_name] = relevance_record[field_name]
    return context


# ============================================================================
# HISTORICAL RECORD VALIDATION
# ============================================================================

def _trade_identifier(record: Mapping[str, Any]) -> Optional[str]:
    for field_name in ("trade_id", "trade_record_id", "setup_id", "id"):
        value = record.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _normalize_historical_trade(
    record: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, str],
    decision_time: int,
    current_trade_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(record, Mapping):
        return None, "NOT_A_MAPPING"
    if not _identity_matches(record, expected_identity):
        return None, "IDENTITY_MISMATCH_OR_INCOMPLETE"

    trade_id = _trade_identifier(record)
    if trade_id is None:
        return None, "MISSING_TRADE_ID"
    if current_trade_id is not None and trade_id == current_trade_id:
        return None, "CURRENT_TRADE"

    outcome = normalize_outcome(record.get("actual_outcome", record.get("label")))
    if outcome is None:
        return None, "INVALID_OR_MISSING_OUTCOME"
    if record.get("training_eligible") is False:
        return None, "NOT_TRAINING_ELIGIBLE"

    status = str(record.get("status", record.get("trade_status", "CLOSED"))).strip().upper()
    if status in {"OPEN", "ACTIVE", "PENDING", "RUNNING", "AMBIGUOUS", "UNRESOLVED"}:
        return None, "NOT_FINALIZED"

    entry_time = parse_timestamp(
        record.get("entry_time", record.get("opened_at", record.get("entry_timestamp")))
    )
    exit_time = parse_timestamp(
        record.get("exit_time", record.get("closed_at", record.get("exit_timestamp")))
    )
    known_at = parse_timestamp(
        record.get("outcome_known_at", record.get("result_known_at"))
    )
    if entry_time is None or exit_time is None or known_at is None:
        return None, "MISSING_CHRONOLOGY"
    if exit_time < entry_time or known_at < exit_time:
        return None, "INVALID_CHRONOLOGY"
    if known_at >= decision_time:
        return None, "OUTCOME_NOT_KNOWN_BEFORE_DECISION"

    normalized = dict(record)
    normalized.update(
        {
            "trade_id": trade_id,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "outcome_known_at": known_at,
            "actual_outcome": outcome,
            "linked_pattern_ids": sorted(_extract_pattern_weights(record)),
        }
    )
    return normalized, None


def _deduplicate_historical_trades(
    records: Iterable[Any],
    *,
    expected_identity: Mapping[str, str],
    decision_time: int,
    current_trade_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    valid: Dict[str, Dict[str, Any]] = {}
    exclusions: Dict[str, int] = {}
    for record in records:
        normalized, reason = _normalize_historical_trade(
            record,
            expected_identity=expected_identity,
            decision_time=decision_time,
            current_trade_id=current_trade_id,
        )
        if normalized is None:
            key = reason or "INVALID_RECORD"
            exclusions[key] = exclusions.get(key, 0) + 1
            continue
        trade_id = normalized["trade_id"]
        previous = valid.get(trade_id)
        if previous is None:
            valid[trade_id] = normalized
            continue
        if (
            previous["actual_outcome"] != normalized["actual_outcome"]
            or previous["entry_time"] != normalized["entry_time"]
            or previous["exit_time"] != normalized["exit_time"]
            or previous["outcome_known_at"] != normalized["outcome_known_at"]
        ):
            valid.pop(trade_id, None)
            exclusions["CONFLICTING_DUPLICATE_TRADE"] = (
                exclusions.get("CONFLICTING_DUPLICATE_TRADE", 0) + 1
            )

    ordered = sorted(
        valid.values(),
        key=lambda item: (
            item["outcome_known_at"],
            item["entry_time"],
            item["trade_id"],
        ),
    )
    return ordered, exclusions


# ============================================================================
# SIMILARITY AND EVIDENCE WEIGHTING
# ============================================================================

def _weighted_jaccard(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Optional[float]:
    if not left or not right:
        return None
    keys = set(left) | set(right)
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    if denominator <= 0.0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def _numeric_feature_similarity(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Optional[float]:
    shared = sorted(set(left) & set(right))
    if not shared:
        return None
    similarities = []
    for key in shared:
        left_value = left[key]
        right_value = right[key]
        scale = max(abs(left_value), abs(right_value), 1.0)
        relative_difference = abs(left_value - right_value) / scale
        similarities.append(1.0 / (1.0 + relative_difference))
    return sum(similarities) / len(similarities)


def _categorical_similarity(left: Optional[str], right: Optional[str]) -> Optional[float]:
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def _trade_similarity(
    candidate: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    config: ProbabilityConfig,
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    candidate_patterns = _extract_pattern_weights(candidate)
    historical_patterns = _extract_pattern_weights(historical)
    candidate_features = _extract_feature_map(candidate)
    historical_features = _extract_feature_map(historical)

    components: Dict[str, Optional[float]] = {
        "pattern_similarity": _weighted_jaccard(candidate_patterns, historical_patterns),
        "feature_similarity": _numeric_feature_similarity(
            candidate_features,
            historical_features,
        ),
        "direction_similarity": _categorical_similarity(
            _normalize_direction(candidate.get("direction")),
            _normalize_direction(historical.get("direction")),
        ),
        "timeframe_similarity": _categorical_similarity(
            _normalize_timeframe(
                candidate.get("entry_timeframe", candidate.get("setup_timeframe", candidate.get("timeframe")))
            ),
            _normalize_timeframe(
                historical.get("entry_timeframe", historical.get("setup_timeframe", historical.get("timeframe")))
            ),
        ),
    }
    weights = {
        "pattern_similarity": config.pattern_weight,
        "feature_similarity": config.feature_weight,
        "direction_similarity": config.direction_weight,
        "timeframe_similarity": config.timeframe_weight,
    }
    available = [
        (components[name], weights[name])
        for name in components
        if components[name] is not None and weights[name] > 0.0
    ]
    if not available:
        return None, components
    total_weight = sum(weight for _, weight in available)
    score = sum(float(value) * weight for value, weight in available) / total_weight
    return min(max(score, 0.0), 1.0), components


def _recency_weight(
    historical: Mapping[str, Any],
    *,
    decision_time: int,
    config: ProbabilityConfig,
) -> float:
    if config.recency_half_life_days is None:
        return 1.0
    entry_time = parse_timestamp(historical.get("entry_time"))
    if entry_time is None:
        return 1.0
    age_seconds = max(decision_time - entry_time, 0)
    half_life_seconds = config.recency_half_life_days * 86400.0
    return math.pow(0.5, age_seconds / half_life_seconds)


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared_total = sum(weight * weight for weight in weights)
    if squared_total <= 0.0:
        return 0.0
    return (total * total) / squared_total


def _confidence_interval(
    alpha: float,
    beta: float,
    *,
    z_value: float,
) -> Tuple[float, float]:
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    margin = z_value * math.sqrt(max(variance, 0.0))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _stable_estimate_id(
    identity: Mapping[str, str],
    setup_id: str,
    decision_time: int,
    evidence_trade_ids: Sequence[str],
) -> str:
    payload = {
        "identity": dict(identity),
        "setup_id": setup_id,
        "decision_time": decision_time,
        "evidence_trade_ids": list(evidence_trade_ids),
        "target": PROBABILITY_TARGET,
        "schema_version": PROBABILITY_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


# ============================================================================
# PROBABILITY ESTIMATION
# ============================================================================

def estimate_tp_before_sl(
    setup: Mapping[str, Any],
    historical_trades: Iterable[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    decision_time: Any = None,
    relevance_record: Optional[Mapping[str, Any]] = None,
    config: Optional[ProbabilityConfig] = None,
) -> Dict[str, Any]:
    """Estimate TP-first probability from customer outcomes known before this decision."""
    if not isinstance(setup, Mapping):
        raise TypeError("setup must be a mapping.")
    resolved_identity = _normalize_identity(identity)
    resolved_config = config or ProbabilityConfig()
    if not isinstance(resolved_config, ProbabilityConfig):
        raise TypeError("config must be a ProbabilityConfig instance.")

    context = _extract_setup_context(setup, relevance_record)
    cutoff = parse_timestamp(
        decision_time
        if decision_time is not None
        else context.get("decision_time", context.get("predicted_at", context.get("entry_time")))
    )
    setup_id = str(context.get("setup_id", context.get("id", ""))).strip()
    trade_id = _trade_identifier(context)
    if cutoff is None:
        raise ProbabilityValidationError(
            "A valid decision_time, predicted_at or entry_time is required."
        )
    if not setup_id:
        setup_id = trade_id or f"setup_at_{cutoff}"
    if trade_id is None:
        trade_id = setup_id

    normalized_history, exclusion_counts = _deduplicate_historical_trades(
        historical_trades,
        expected_identity=resolved_identity,
        decision_time=cutoff,
        current_trade_id=trade_id,
    )
    if len(normalized_history) > resolved_config.maximum_history_records:
        normalized_history = normalized_history[-resolved_config.maximum_history_records :]

    candidate = dict(context)
    candidate["patterns_known_at_entry"] = context.get("patterns_known_at_entry", [])
    candidate_patterns = _extract_pattern_weights(candidate)
    candidate["prediction_pattern_ids"] = sorted(candidate_patterns)

    weighted_examples: List[Dict[str, Any]] = []
    for historical in normalized_history:
        similarity, components = _trade_similarity(
            candidate,
            historical,
            config=resolved_config,
        )
        if similarity is None:
            sample_weight = resolved_config.global_fallback_weight
            match_type = "GLOBAL_FALLBACK"
        elif similarity < resolved_config.minimum_similarity:
            continue
        else:
            sample_weight = similarity
            match_type = "SIMILARITY_MATCH"

        sample_weight *= _recency_weight(
            historical,
            decision_time=cutoff,
            config=resolved_config,
        )
        if sample_weight <= 0.0:
            continue
        weighted_examples.append(
            {
                "trade_id": historical["trade_id"],
                "entry_time": historical["entry_time"],
                "exit_time": historical["exit_time"],
                "outcome_known_at": historical["outcome_known_at"],
                "actual_outcome": historical["actual_outcome"],
                "similarity": similarity,
                "sample_weight": min(sample_weight, 1.0),
                "match_type": match_type,
                "components": components,
            }
        )

    weighted_examples.sort(
        key=lambda item: (item["sample_weight"], item["outcome_known_at"]),
        reverse=True,
    )
    weighted_examples = weighted_examples[: resolved_config.maximum_neighbors]

    weights = [float(item["sample_weight"]) for item in weighted_examples]
    weighted_successes = sum(
        float(item["sample_weight"])
        for item in weighted_examples
        if item["actual_outcome"] == OUTCOME_TP_FIRST
    )
    weighted_failures = sum(
        float(item["sample_weight"])
        for item in weighted_examples
        if item["actual_outcome"] == OUTCOME_SL_FIRST
    )
    total_evidence_weight = weighted_successes + weighted_failures
    effective_size = _effective_sample_size(weights)
    raw_wins = sum(
        1 for item in normalized_history if item["actual_outcome"] == OUTCOME_TP_FIRST
    )
    global_win_rate = (
        raw_wins / len(normalized_history) if normalized_history else None
    )
    matched_win_rate = (
        weighted_successes / total_evidence_weight
        if total_evidence_weight > 0.0
        else None
    )

    alpha = (
        resolved_config.prior_probability * resolved_config.prior_strength
        + weighted_successes
    )
    beta = (
        (1.0 - resolved_config.prior_probability) * resolved_config.prior_strength
        + weighted_failures
    )
    probability = alpha / (alpha + beta) if total_evidence_weight > 0.0 else None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    if probability is not None:
        lower_bound, upper_bound = _confidence_interval(
            alpha,
            beta,
            z_value=resolved_config.confidence_interval_z,
        )

    sufficient_history = len(normalized_history) >= resolved_config.minimum_history_count
    sufficient_effective_size = (
        effective_size >= resolved_config.minimum_effective_sample_size
        and total_evidence_weight >= resolved_config.minimum_effective_sample_size
    )
    prediction_eligible = bool(
        probability is not None
        and sufficient_history
        and sufficient_effective_size
        and total_evidence_weight > 0.0
    )
    if not normalized_history:
        status = "INSUFFICIENT_HISTORY"
    elif probability is None:
        status = "NO_COMPARABLE_EVIDENCE"
    elif prediction_eligible:
        status = "ESTIMATED"
    else:
        status = "LOW_EVIDENCE"

    evidence = [
        {
            **resolved_identity,
            "trade_id": item["trade_id"],
            "entry_time": item["entry_time"],
            "exit_time": item["exit_time"],
            "outcome_known_at": item["outcome_known_at"],
            "actual_outcome": item["actual_outcome"],
            "similarity": item["similarity"],
            "sample_weight": item["sample_weight"],
            "match_type": item["match_type"],
            "components": item["components"],
        }
        for item in weighted_examples
    ]
    evidence_trade_ids = [item["trade_id"] for item in weighted_examples]

    result: Dict[str, Any] = {
        "schema_version": PROBABILITY_SCHEMA_VERSION,
        "estimate_id": _stable_estimate_id(
            resolved_identity,
            setup_id,
            cutoff,
            evidence_trade_ids,
        ),
        "setup_id": setup_id,
        "trade_id": trade_id,
        **resolved_identity,
        "target": PROBABILITY_TARGET,
        "decision_time": cutoff,
        "predicted_at": cutoff,
        "probability": probability,
        "predicted_probability_at_entry": probability if prediction_eligible else None,
        "credible_interval": {
            "level": 0.95,
            "lower": lower_bound,
            "upper": upper_bound,
            "method": "NORMAL_APPROXIMATION_TO_BETA_POSTERIOR",
        },
        "status": status,
        "prediction_eligible": prediction_eligible,
        "estimator": "WEIGHTED_BETA_BINOMIAL_NEIGHBORS",
        "calibration_status": "NOT_CALIBRATED",
        "prior": {
            "mean": resolved_config.prior_probability,
            "strength": resolved_config.prior_strength,
            "alpha": resolved_config.prior_probability * resolved_config.prior_strength,
            "beta": (1.0 - resolved_config.prior_probability) * resolved_config.prior_strength,
        },
        "evidence_summary": {
            "valid_historical_trade_count": len(normalized_history),
            "selected_neighbor_count": len(weighted_examples),
            "weighted_evidence_count": total_evidence_weight,
            "effective_sample_size": effective_size,
            "weighted_tp_first": weighted_successes,
            "weighted_sl_first": weighted_failures,
            "raw_tp_first_rate": global_win_rate,
            "weighted_neighbor_tp_first_rate": matched_win_rate,
            "candidate_pattern_count": len(candidate_patterns),
        },
        "evidence": evidence,
        "exclusion_counts": exclusion_counts,
        "warnings": [],
    }
    if len(normalized_history) < resolved_config.minimum_history_count:
        result["warnings"].append(
            "There are fewer historical outcomes than the configured minimum."
        )
    if not sufficient_effective_size and normalized_history:
        result["warnings"].append(
            "The effective number of comparable historical outcomes is too small."
        )
    if not candidate_patterns:
        result["warnings"].append(
            "No prediction-eligible Pattern IDs were supplied for this setup."
        )
    result["warnings"].append(
        "This empirical estimate is not calibrated and is not a trained model output."
    )
    return result


# ============================================================================
# TRAINING BUNDLE INTEGRATION
# ============================================================================

def _bundle_identity(bundle: Mapping[str, Any]) -> Tuple[Dict[str, str], str]:
    identity = _normalize_identity(bundle.get("identity", {}))
    market = str(bundle.get("market", identity["market"])).strip().upper()
    if market != identity["market"]:
        raise ProbabilityValidationError(
            "Bundle market does not match the customer identity market."
        )
    return identity, market


def _setup_decision_time(
    setup: Mapping[str, Any],
    timeframe_records: Mapping[str, Mapping[str, Any]],
) -> Optional[int]:
    setup_id = str(setup.get("setup_id", setup.get("id", "")))
    timeframe_record = timeframe_records.get(setup_id, {})
    for value in (
        setup.get("decision_time"),
        timeframe_record.get("decision_time"),
        setup.get("predicted_at"),
        setup.get("entry_time"),
    ):
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _relevance_for_setup(
    bundle: Mapping[str, Any],
    relevance_bundle: Optional[Mapping[str, Any]],
    setup_id: str,
) -> Optional[Mapping[str, Any]]:
    source = relevance_bundle
    if source is None:
        candidate = bundle.get("relevance")
        if isinstance(candidate, Mapping):
            source = candidate
    if isinstance(source, Mapping):
        by_setup = source.get("by_setup", {})
        if isinstance(by_setup, Mapping):
            record = by_setup.get(setup_id, by_setup.get(str(setup_id)))
            if isinstance(record, Mapping):
                return record
    direct = bundle.get("relevance_by_setup", {})
    if isinstance(direct, Mapping):
        record = direct.get(setup_id, direct.get(str(setup_id)))
        if isinstance(record, Mapping):
            return record
    return None


def _prior_history_for_setup(
    bundle: Mapping[str, Any],
    setup_id: str,
    *,
    decision_time: int,
    current_trade_id: Optional[str],
) -> List[Dict[str, Any]]:
    prior_by_setup = bundle.get("prior_customer_results_by_setup")
    if isinstance(prior_by_setup, Mapping):
        records = prior_by_setup.get(setup_id, prior_by_setup.get(str(setup_id), []))
    else:
        records = bundle.get("customer_trade_results", [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []

    filtered: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        known_at = parse_timestamp(
            record.get("outcome_known_at", record.get("result_known_at"))
        )
        if known_at is None or known_at >= decision_time:
            continue
        trade_id = _trade_identifier(record)
        if current_trade_id is not None and trade_id == current_trade_id:
            continue
        filtered.append(dict(record))
    return filtered


def build_probability_bundle(
    data_bundle: Mapping[str, Any],
    *,
    relevance_bundle: Optional[Mapping[str, Any]] = None,
    config: Optional[ProbabilityConfig] = None,
) -> Dict[str, Any]:
    """Estimate a point-in-time TP-first probability for each setup in a data bundle."""
    if not isinstance(data_bundle, Mapping):
        raise TypeError("data_bundle must be a mapping.")
    identity, market = _bundle_identity(data_bundle)
    resolved_config = config or ProbabilityConfig()
    if not isinstance(resolved_config, ProbabilityConfig):
        raise TypeError("config must be a ProbabilityConfig instance.")

    setup_inputs = data_bundle.get("setup_input_records", {})
    if not isinstance(setup_inputs, Mapping):
        raise ProbabilityValidationError("data_bundle.setup_input_records must be a mapping.")

    timeframe_lookup: Dict[str, Mapping[str, Any]] = {}
    timeframe_records = data_bundle.get("timeframe_data", [])
    if isinstance(timeframe_records, Sequence) and not isinstance(timeframe_records, (str, bytes)):
        for record in timeframe_records:
            if isinstance(record, Mapping):
                timeframe_lookup[str(record.get("setup_id", ""))] = record

    estimates: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    history_count = 0
    eligible_count = 0
    low_evidence_count = 0
    insufficient_count = 0

    for raw_setup_id, raw_setup in setup_inputs.items():
        setup_id = str(raw_setup_id)
        if not isinstance(raw_setup, Mapping):
            issues.append(f"Setup {setup_id} input is not a mapping and was skipped.")
            continue
        if raw_setup.get("validation_errors"):
            issues.append(f"Setup {setup_id} contains invalid input fields and was skipped.")
            continue

        decision_time = _setup_decision_time(raw_setup, timeframe_lookup)
        if decision_time is None:
            issues.append(f"Setup {setup_id} has no valid point-in-time decision timestamp.")
            continue
        current_trade_id = _trade_identifier(raw_setup) or setup_id
        prior_history = _prior_history_for_setup(
            data_bundle,
            setup_id,
            decision_time=decision_time,
            current_trade_id=current_trade_id,
        )
        relevance_record = _relevance_for_setup(
            data_bundle,
            relevance_bundle,
            setup_id,
        )
        setup_context = dict(raw_setup)
        setup_context["decision_time"] = decision_time
        setup_context["setup_id"] = setup_id
        try:
            estimate = estimate_tp_before_sl(
                setup_context,
                prior_history,
                identity=identity,
                decision_time=decision_time,
                relevance_record=relevance_record,
                config=resolved_config,
            )
        except (ProbabilityError, TypeError, ValueError) as error:
            issues.append(f"Setup {setup_id}: {error}")
            continue

        estimate["market"] = market
        estimates[setup_id] = estimate
        history_count += estimate["evidence_summary"]["valid_historical_trade_count"]
        if estimate["prediction_eligible"]:
            eligible_count += 1
        elif estimate["status"] == "LOW_EVIDENCE":
            low_evidence_count += 1
        elif estimate["status"] in {"INSUFFICIENT_HISTORY", "NO_COMPARABLE_EVIDENCE"}:
            insufficient_count += 1

    return {
        "schema_version": PROBABILITY_SCHEMA_VERSION,
        "target": PROBABILITY_TARGET,
        "identity": identity,
        "market": market,
        "config": asdict(resolved_config),
        "by_setup": estimates,
        "report": {
            "setup_count": len(setup_inputs),
            "estimated_setup_count": len(estimates),
            "prediction_eligible_count": eligible_count,
            "low_evidence_count": low_evidence_count,
            "insufficient_history_count": insufficient_count,
            "historical_trade_reference_count": history_count,
            "probability_is_calibrated": False,
            "issues": issues,
        },
    }


def attach_probability_to_bundle(
    data_bundle: Mapping[str, Any],
    *,
    relevance_bundle: Optional[Mapping[str, Any]] = None,
    config: Optional[ProbabilityConfig] = None,
) -> Dict[str, Any]:
    """Return a copy of a data bundle with estimates attached as separate metadata."""
    result = dict(data_bundle)
    probability_bundle = build_probability_bundle(
        data_bundle,
        relevance_bundle=relevance_bundle,
        config=config,
    )
    result["probability"] = probability_bundle
    result["probability_by_setup"] = probability_bundle["by_setup"]
    return result


# ============================================================================
# PREDICTION PERSISTENCE
# ============================================================================

def append_probability_timeline_event(
    identity: Any,
    estimate: Mapping[str, Any],
    *,
    storage: Any = None,
) -> Any:
    """Append an eligible prediction to the customer timeline as a separate event."""
    if not isinstance(estimate, Mapping):
        raise TypeError("estimate must be a mapping.")
    if estimate.get("prediction_eligible") is not True:
        raise ProbabilityValidationError(
            "Only an estimate with sufficient historical evidence can be persisted as a prediction."
        )
    probability = normalize_probability(estimate.get("probability"))
    predicted_at = parse_timestamp(estimate.get("predicted_at"))
    entry_time = parse_timestamp(estimate.get("entry_time", estimate.get("decision_time")))
    if probability is None or predicted_at is None or entry_time is None:
        raise ProbabilityValidationError(
            "A valid probability, predicted_at and entry or decision time are required."
        )
    if predicted_at > entry_time:
        raise ProbabilityValidationError("predicted_at cannot be later than entry_time.")

    if storage is None:
        failures = []
        for module_name in ("engine.storage", ".storage"):
            try:
                if module_name.startswith("."):
                    from importlib import import_module
                    module = import_module(module_name, package=__package__)
                else:
                    from importlib import import_module
                    module = import_module(module_name)
                storage_class = getattr(module, "HistoricalStorage", None)
                storage = storage_class() if callable(storage_class) else module
                break
            except Exception as error:
                failures.append(str(error))
        if storage is None:
            raise ProbabilityError(
                "Historical storage is unavailable: " + "; ".join(failures)
            )

    writer = getattr(storage, "append_timeline_event", None)
    if not callable(writer):
        raise ProbabilityError("Storage does not expose append_timeline_event().")

    event = {
        "event_type": "TRADE_PREDICTION",
        "event_time": predicted_at,
        "available_at": predicted_at,
        "setup_id": str(estimate.get("setup_id", "")),
        "trade_id": str(estimate.get("trade_id", estimate.get("setup_id", ""))),
        "target": PROBABILITY_TARGET,
        "predicted_probability_at_entry": probability,
        "predicted_at": predicted_at,
        "probability_estimate_id": estimate.get("estimate_id"),
        "estimator": estimate.get("estimator"),
        "calibration_status": estimate.get("calibration_status"),
        "credible_interval": estimate.get("credible_interval"),
        "evidence_summary": estimate.get("evidence_summary"),
    }
    return writer(identity, event)


# ============================================================================
# VALIDATION AND CONVENIENCE API
# ============================================================================

def validate_probability_estimate(estimate: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate probability range, identity, chronology and evidence provenance."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(estimate, Mapping):
        return {"valid": False, "errors": ["Estimate must be a mapping."], "warnings": []}

    try:
        identity = _normalize_identity(estimate)
    except ProbabilityValidationError as error:
        identity = None
        errors.append(str(error))

    if estimate.get("target") != PROBABILITY_TARGET:
        errors.append("Estimate target is not TP_BEFORE_SL.")
    decision_time = parse_timestamp(estimate.get("decision_time"))
    predicted_at = parse_timestamp(estimate.get("predicted_at"))
    if decision_time is None or predicted_at is None:
        errors.append("Estimate requires valid decision_time and predicted_at.")
    elif predicted_at > decision_time:
        errors.append("predicted_at cannot be later than decision_time.")

    probability = estimate.get("probability")
    if probability is not None and normalize_probability(probability) is None:
        errors.append("probability must be between zero and one.")

    eligible = estimate.get("prediction_eligible") is True
    if eligible and probability is None:
        errors.append("An eligible prediction must contain a probability.")
    if eligible and estimate.get("status") != "ESTIMATED":
        errors.append("An eligible prediction must have ESTIMATED status.")

    evidence = estimate.get("evidence", [])
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        errors.append("evidence must be a sequence.")
        evidence = []
    for record in evidence:
        if not isinstance(record, Mapping):
            errors.append("Evidence contains a non-mapping record.")
            continue
        outcome = normalize_outcome(record.get("actual_outcome"))
        known_at = parse_timestamp(record.get("outcome_known_at"))
        if outcome is None:
            errors.append("Evidence contains an invalid actual_outcome.")
        if decision_time is None or known_at is None or known_at >= decision_time:
            errors.append("Evidence contains an outcome unavailable at prediction time.")
        if identity is not None and not _identity_matches(record, identity):
            errors.append("Evidence contains a record from a different customer identity.")

    if estimate.get("calibration_status") == "NOT_CALIBRATED":
        warnings.append("Probability is an empirical estimate and has not been calibrated.")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "prediction_eligible": eligible,
        "evidence_count": len(evidence),
    }


def validate_probability_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a bundle of customer-scoped, point-in-time probability estimates."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(bundle, Mapping):
        return {"valid": False, "errors": ["Probability bundle must be a mapping."], "warnings": []}
    try:
        identity, market = _bundle_identity(bundle)
    except ProbabilityValidationError as error:
        return {"valid": False, "errors": [str(error)], "warnings": []}

    by_setup = bundle.get("by_setup")
    if not isinstance(by_setup, Mapping):
        errors.append("by_setup must be a mapping.")
        by_setup = {}
    for setup_id, estimate in by_setup.items():
        if not isinstance(estimate, Mapping):
            errors.append(f"Setup {setup_id} estimate must be a mapping.")
            continue
        result = validate_probability_estimate(estimate)
        errors.extend(f"{setup_id}: {item}" for item in result["errors"])
        warnings.extend(f"{setup_id}: {item}" for item in result["warnings"])
        if str(estimate.get("setup_id")) != str(setup_id):
            errors.append(f"Setup key {setup_id} does not match estimate setup_id.")
        if str(estimate.get("market", market)).upper() != market:
            errors.append(f"Setup {setup_id} belongs to a different market.")
        for field_name in IDENTITY_FIELDS:
            supplied = estimate.get(field_name)
            if supplied is None:
                errors.append(f"Setup {setup_id} is missing identity field {field_name}.")
            elif field_name == "market":
                if str(supplied).upper() != market:
                    errors.append(f"Setup {setup_id} has a mismatched market identity.")
            elif str(supplied) != identity[field_name]:
                errors.append(f"Setup {setup_id} has a mismatched {field_name}.")

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


def probability_for_setup(
    data_bundle: Mapping[str, Any],
    setup_id: Any,
    *,
    relevance_bundle: Optional[Mapping[str, Any]] = None,
    config: Optional[ProbabilityConfig] = None,
) -> Dict[str, Any]:
    """Build and return one setup estimate from an engine.data bundle."""
    probability_bundle = build_probability_bundle(
        data_bundle,
        relevance_bundle=relevance_bundle,
        config=config,
    )
    by_setup = probability_bundle["by_setup"]
    key = str(setup_id)
    if key not in by_setup:
        raise KeyError(f"No probability estimate was produced for setup {key!r}.")
    return by_setup[key]


estimate_setup_probability = estimate_tp_before_sl
build_probability = build_probability_bundle
attach_probability = attach_probability_to_bundle


__all__ = [
    "PROBABILITY_SCHEMA_VERSION",
    "PROBABILITY_TARGET",
    "OUTCOME_SL_FIRST",
    "OUTCOME_TP_FIRST",
    "ProbabilityError",
    "ProbabilityValidationError",
    "ProbabilityConfig",
    "parse_timestamp",
    "normalize_probability",
    "normalize_outcome",
    "estimate_tp_before_sl",
    "estimate_setup_probability",
    "build_probability_bundle",
    "build_probability",
    "attach_probability_to_bundle",
    "attach_probability",
    "probability_for_setup",
    "append_probability_timeline_event",
    "validate_probability_estimate",
    "validate_probability_bundle",
]
