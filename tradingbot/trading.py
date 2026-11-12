"""Virtual trade execution and historical TP/SL outcome management for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import copy
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .chart import TIMEFRAME_SECONDS
    from .runtime import RuntimeContext, TradingSessionState
    from .setups import save_learning_setup
    from .strategy import evaluate_strategy
except ImportError:
    from chart import TIMEFRAME_SECONDS
    from runtime import RuntimeContext, TradingSessionState
    from setups import save_learning_setup
    from strategy import evaluate_strategy


# ============================================================================
# CONFIGURATION
# ============================================================================

LOGGER = logging.getLogger(__name__)

DEFAULT_PROTECTION_TRIGGER_FRACTION = 0.33
DEFAULT_PROTECTION_SL_FRACTION = 0.05
DEFAULT_MIN_SETUP_PROBABILITY = 0.50
DEFAULT_MODEL_TESTING_COOLDOWN = 0

LOWER_TIMEFRAME_PRIORITY = ("M5", "M1")

_PATTERN_ID_RE = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class TradingConfig:
    """Execution settings used by one testing or virtual-trading run."""

    protection_trigger_fraction: float = DEFAULT_PROTECTION_TRIGGER_FRACTION
    protection_sl_fraction: float = DEFAULT_PROTECTION_SL_FRACTION
    min_setup_probability: float = DEFAULT_MIN_SETUP_PROBABILITY
    model_testing_cooldown: int = DEFAULT_MODEL_TESTING_COOLDOWN

    def __post_init__(self) -> None:
        numeric_fields = (
            self.protection_trigger_fraction,
            self.protection_sl_fraction,
            self.min_setup_probability,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_fields
        ):
            raise ValueError("TradingConfig probability and fraction values must be finite numbers.")

        if not 0.0 <= self.protection_trigger_fraction <= 1.0:
            raise ValueError("protection_trigger_fraction must be between 0 and 1.")
        if not 0.0 <= self.protection_sl_fraction <= 1.0:
            raise ValueError("protection_sl_fraction must be between 0 and 1.")
        if not 0.0 <= self.min_setup_probability <= 1.0:
            raise ValueError("min_setup_probability must be between 0 and 1.")
        if (
            isinstance(self.model_testing_cooldown, bool)
            or int(self.model_testing_cooldown) != self.model_testing_cooldown
            or self.model_testing_cooldown < 0
        ):
            raise ValueError("model_testing_cooldown must be a non-negative integer.")


# ============================================================================
# VALUE AND TIME HELPERS
# ============================================================================

def safe_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float or return None."""
    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return result if math.isfinite(result) else None


def _safe_timestamp(value: Any) -> Optional[int]:
    """Convert a supported timestamp representation to UTC epoch seconds."""
    if value is None or isinstance(value, bool):
        return None

    numeric = safe_float(value)
    if numeric is not None:
        return int(numeric)

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def _candle_value(candle: Any, key: str) -> Optional[float]:
    """Read one numeric candle field from a mapping or Candle object."""
    if isinstance(candle, Mapping):
        value = candle.get(key)
    else:
        value = getattr(candle, key, None)
    return safe_float(value)


def _candle_time(candle: Any) -> Optional[int]:
    """Read a candle timestamp from a mapping or Candle object."""
    if isinstance(candle, Mapping):
        value = candle.get("time")
    else:
        value = getattr(candle, "time", None)
    return _safe_timestamp(value)


def _validate_candle(candle: Any) -> Optional[Dict[str, Any]]:
    """Normalize and validate the OHLC fields required by trade execution."""
    timestamp = _candle_time(candle)
    open_price = _candle_value(candle, "open")
    high = _candle_value(candle, "high")
    low = _candle_value(candle, "low")
    close = _candle_value(candle, "close")

    if None in (timestamp, open_price, high, low, close):
        return None
    if high < low or high < max(open_price, close) or low > min(open_price, close):
        return None

    return {
        "time": int(timestamp),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
    }


def _normalize_probability(value: Any) -> Optional[float]:
    """Normalize a probability represented as a fraction or percentage."""
    probability = safe_float(value)
    if probability is None:
        return None
    if 1.0 < probability <= 100.0:
        probability /= 100.0
    if not 0.0 <= probability <= 1.0:
        return None
    return float(probability)


def _prediction_probability(setup: Mapping[str, Any]) -> Optional[float]:
    """Read only the dedicated TP-before-SL prediction field."""
    candidates = [setup.get("predicted_probability_at_entry")]
    prediction = setup.get("prediction")
    if isinstance(prediction, Mapping):
        candidates.append(prediction.get("tp_before_sl_probability"))
        candidates.append(prediction.get("predicted_probability_at_entry"))

    candidates.append(setup.get("tp_before_sl_probability"))
    for candidate in candidates:
        normalized = _normalize_probability(candidate)
        if normalized is not None:
            return normalized
    return None


def _pattern_ids(setup: Mapping[str, Any]) -> List[str]:
    """Return unique, valid four-digit pattern identifiers from a setup."""
    supplied: List[Any] = []
    linked = setup.get("linked_pattern_ids", [])
    if isinstance(linked, Sequence) and not isinstance(linked, (str, bytes)):
        supplied.extend(linked)

    for field_name in ("pattern_id", "associated_pattern_id"):
        if setup.get(field_name) is not None:
            supplied.append(setup[field_name])

    result: List[str] = []
    for raw_value in supplied:
        text = str(raw_value).strip()
        if text.isdigit() and len(text) < 4:
            text = text.zfill(4)
        if _PATTERN_ID_RE.fullmatch(text) and text not in result:
            result.append(text)
    return result


def _timeframe_candles(runtime: RuntimeContext, timeframe: str) -> List[Any]:
    """Return one timeframe's candles from the shared AI market archive."""
    archive = runtime.ai_market_candles
    if not isinstance(archive, Mapping):
        return []

    values = archive.get(timeframe, [])
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return []


def _current_bar_times(runtime: RuntimeContext) -> Tuple[Optional[int], Optional[int]]:
    """Return the current chart bar start and close timestamps."""
    candles = runtime.get_chart_data()
    session = runtime.current_session
    if not candles:
        return None, None

    index = min(max(int(session.current_index), 0), len(candles) - 1)
    candle_time = _candle_time(candles[index])
    duration = TIMEFRAME_SECONDS.get(runtime.active_timeframe)
    if candle_time is None or duration is None:
        return candle_time, None
    return candle_time, candle_time + int(duration)

# ============================================================================
# TRADE CREATION
# ============================================================================

def open_trade_position(
    runtime: RuntimeContext,
    direction: str,
    price: float,
    setup: Optional[Dict[str, Any]] = None,
    *,
    opened_at: Optional[int] = None,
    config: Optional[TradingConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Open one virtual position inside the supplied TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    if not isinstance(setup, dict):
        return None
    if config is not None and not isinstance(config, TradingConfig):
        raise TypeError("config must be a TradingConfig instance.")

    session = runtime.current_session
    if session.open_trade is not None or not session.trading_enabled:
        return None

    normalized_direction = str(direction).strip().upper()
    if normalized_direction in {"LONG", "BUY"}:
        normalized_direction = "BUY"
    elif normalized_direction in {"SHORT", "SELL"}:
        normalized_direction = "SELL"
    else:
        return None

    entry_price = safe_float(price)
    sl = safe_float(setup.get("sl"))
    tp = safe_float(setup.get("tp"))
    if entry_price is None or sl is None or tp is None:
        return None

    if normalized_direction == "BUY" and not (sl < entry_price < tp):
        return None
    if normalized_direction == "SELL" and not (tp < entry_price < sl):
        return None

    risk_distance = abs(entry_price - sl)
    tp_distance = abs(tp - entry_price)
    if risk_distance <= 0.0 or tp_distance <= 0.0:
        return None

    candles = runtime.get_chart_data()
    if opened_at is None:
        if not candles:
            return None
        bar_start, bar_close = _current_bar_times(runtime)
        opened_at = bar_close if bar_close is not None else bar_start

    entry_time = _safe_timestamp(opened_at)
    if entry_time is None:
        return None

    low1 = setup.get("low1") if isinstance(setup.get("low1"), dict) else {}
    low2 = setup.get("low2") if isinstance(setup.get("low2"), dict) else {}
    high1 = setup.get("high1") if isinstance(setup.get("high1"), dict) else {}
    high2 = setup.get("high2") if isinstance(setup.get("high2"), dict) else {}

    if normalized_direction == "BUY":
        structural_values = [safe_float(low1.get("price")), safe_float(low2.get("price"))]
        structural_values = [value for value in structural_values if value is not None]
        structural_reference_price = min(structural_values) if structural_values else sl
        structural_reference_name = "model-derived lower structure"
        setup_swings = {
            "low1": copy.deepcopy(low1) if low1 else None,
            "low2": copy.deepcopy(low2) if low2 else None,
        }
    else:
        structural_values = [safe_float(high1.get("price")), safe_float(high2.get("price"))]
        structural_values = [value for value in structural_values if value is not None]
        structural_reference_price = max(structural_values) if structural_values else sl
        structural_reference_name = "model-derived upper structure"
        setup_swings = {
            "high1": copy.deepcopy(high1) if high1 else None,
            "high2": copy.deepcopy(high2) if high2 else None,
        }

    trade_config = config or TradingConfig()
    trigger_price = safe_float(setup.get("protection_trigger"))
    protected_sl = safe_float(setup.get("protection_sl"))

    if trigger_price is None:
        trigger_price = (
            entry_price + tp_distance * trade_config.protection_trigger_fraction
            if normalized_direction == "BUY"
            else entry_price - tp_distance * trade_config.protection_trigger_fraction
        )

    if protected_sl is None:
        protected_sl = (
            entry_price + tp_distance * trade_config.protection_sl_fraction
            if normalized_direction == "BUY"
            else entry_price - tp_distance * trade_config.protection_sl_fraction
        )

    if normalized_direction == "BUY":
        protection_levels_valid = entry_price <= protected_sl < trigger_price < tp
    else:
        protection_levels_valid = tp < trigger_price < protected_sl <= entry_price
    if not protection_levels_valid:
        return None

    predicted_probability = _prediction_probability(setup)
    predicted_at = _safe_timestamp(
        setup.get("predicted_at", setup.get("prediction_time", entry_time))
    )
    if predicted_probability is not None and (
        predicted_at is None or predicted_at > entry_time
    ):
        predicted_probability = None
        predicted_at = None

    raw_setup_quality = setup.get("setup_quality_probability")
    if raw_setup_quality is None:
        raw_setup_quality = setup.get("setup_probability")
    setup_quality_probability = _normalize_probability(raw_setup_quality)

    identity_data = (
        runtime.identity.as_dict()
        if callable(getattr(runtime.identity, "as_dict", None))
        else {}
    )
    checkpoint = runtime.ai_checkpoint if isinstance(runtime.ai_checkpoint, Mapping) else {}
    model_version = checkpoint.get(
        "model_version",
        checkpoint.get("version", checkpoint.get("trained_at")),
    )

    trade = {
        "id": setup.get("id"),
        "direction": normalized_direction,
        "entry": entry_price,
        "entry_price": entry_price,
        "opened_at": entry_time,
        "entry_time": entry_time,
        "sl": sl,
        "initial_sl": sl,
        "tp": tp,
        "risk_distance": risk_distance,
        "tp_distance": tp_distance,
        "reward_r_multiple": tp_distance / risk_distance,
        "protection_trigger": trigger_price,
        "protection_sl": protected_sl,
        "protection_trigger_fraction": trade_config.protection_trigger_fraction,
        "protection_sl_fraction": trade_config.protection_sl_fraction,
        "protected": False,
        "protected_at": None,
        "sl_before_protection": None,
        "sl_after_protection": None,
        "pattern": setup.get("pattern", "MODEL SIGNAL"),
        "pattern_id": setup.get("pattern_id"),
        "linked_pattern_ids": _pattern_ids(setup),
        "entry_reason": setup.get("reason", "Market model signal"),
        "setup_level": safe_float(setup.get("level"))
        if safe_float(setup.get("level")) is not None
        else entry_price,
        "selected_level": copy.deepcopy(setup.get("selected_level")),
        "level_strength": safe_float(setup.get("level_strength")) or 0.0,
        "level_touches": int(safe_float(setup.get("level_touches")) or 0),
        "sl_basis": {
            "type": "MODEL",
            "reference": structural_reference_name,
            "reference_price": structural_reference_price,
            "buffer_percent": 0.0,
        },
        "breakout_time": _safe_timestamp(setup.get("breakout_time")),
        "breakout_index": setup.get("breakout_index"),
        "breakout_candle": copy.deepcopy(setup.get("breakout_candle")),
        "setup_swings": setup_swings,
        "setup_snapshot": copy.deepcopy(setup),
        "model_probabilities": copy.deepcopy(setup.get("model_probabilities", {})),
        "setup_quality_probability": setup_quality_probability,
        "predicted_probability_at_entry": predicted_probability,
        "predicted_at": predicted_at if predicted_probability is not None else None,
        "end_time": None,
        "actual_outcome": None,
        "result": "OPEN",
        "training_eligible": False,
        "model_symbol": runtime.market,
        "model_timeframe": runtime.active_timeframe,
        "model_version": str(model_version) if model_version is not None else None,
        "identity": identity_data,
        "storage_status": "pending",
    }

    if normalized_direction == "BUY":
        if safe_float(low1.get("price")) is not None:
            trade["low1"] = float(low1["price"])
        if safe_float(low2.get("price")) is not None:
            trade["low2"] = float(low2["price"])
    else:
        if safe_float(high1.get("price")) is not None:
            trade["high1"] = float(high1["price"])
        if safe_float(high2.get("price")) is not None:
            trade["high2"] = float(high2["price"])

    saved_record = _persist_trade_to_setup_archive(runtime, trade)
    if saved_record is None:
        return None

    trade["id"] = int(saved_record["id"])
    trade["setup_snapshot"]["id"] = trade["id"]
    trade["storage_status"] = "stored"

    session.open_trade = trade
    return copy.deepcopy(trade)


# ============================================================================
# SETUP ARCHIVE STORAGE
# ============================================================================

def _build_setup_archive_record(trade: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = trade.get("setup_snapshot")
    record = copy.deepcopy(snapshot) if isinstance(snapshot, dict) else {}

    for field_name in (
        "trade_id",
        "setup_id",
        "trade_record_id",
        "runtime_trade_id",
        "exit_time",
        "outcome_known_at",
        "outcome_source",
    ):
        record.pop(field_name, None)

    record_id = safe_float(trade.get("id"))
    if record_id is not None and record_id > 0 and record_id.is_integer():
        record["id"] = int(record_id)
    else:
        record.pop("id", None)

    initial_sl = safe_float(trade.get("initial_sl"))
    if initial_sl is None:
        initial_sl = safe_float(trade.get("sl"))

    actual_outcome = trade.get("actual_outcome")
    if actual_outcome not in {"TP", "SL", "AMBIGUOUS"}:
        actual_outcome = None

    record.update({
        "direction": (
            "LONG"
            if str(trade.get("direction", "")).upper() == "BUY"
            else "SHORT"
        ),
        "setup_timeframe": (
            record.get("setup_timeframe")
            or trade.get("model_timeframe")
        ),
        "entry_timeframe": (
            record.get("entry_timeframe")
            or trade.get("model_timeframe")
        ),
        "zones": copy.deepcopy(record.get("zones", {})),
        "trade_geometry": copy.deepcopy(record.get("trade_geometry", {})),
        "entry": safe_float(trade.get("entry_price", trade.get("entry"))),
        "sl": initial_sl,
        "tp": safe_float(trade.get("tp")),
        "end_time": _safe_timestamp(trade.get("end_time")),
        "actual_outcome": actual_outcome,
    })

    return record


def _persist_trade_to_setup_archive(
    runtime: RuntimeContext,
    trade: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    record = _build_setup_archive_record(trade)

    try:
        saved_record = save_learning_setup(runtime.identity, record)

        if not isinstance(saved_record, dict) or saved_record.get("id") is None:
            raise ValueError("The setup archive did not return a valid trade ID.")

        return saved_record

    except Exception:
        LOGGER.exception(
            "Could not save trade to setups.json."
        )
        return None

# ============================================================================
# TRADE CLOSING AND POSITION MANAGEMENT
# ============================================================================

def close_trade(
    runtime: RuntimeContext,
    result: str,
    exit_price: float,
    exit_reason: str,
    *,
    end_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Finalize a virtual trade while keeping its prediction and outcome separate."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    session = runtime.current_session
    trade = session.open_trade
    if trade is None:
        return None

    normalized_result = str(result).strip().upper()
    if normalized_result not in {"WIN", "LOSS", "AMBIGUOUS"}:
        raise ValueError("result must be WIN, LOSS or AMBIGUOUS.")

    normalized_exit = safe_float(exit_price)
    if normalized_exit is None:
        raise ValueError("exit_price must be a finite number.")

    current_bar_start, current_bar_close = _current_bar_times(runtime)

    resolved_end_time = _safe_timestamp(end_time)
    if resolved_end_time is None:
        resolved_end_time = (
            current_bar_close
            if current_bar_close is not None
            else current_bar_start
        )

    if resolved_end_time is None:
        raise ValueError("The trade closure time could not be resolved.")

    entry_time = _safe_timestamp(trade.get("entry_time", trade.get("opened_at")))
    if entry_time is not None and resolved_end_time < entry_time:
        raise ValueError("end_time cannot precede entry_time.")

    closed_trade = copy.deepcopy(trade)
    closed_trade["exit"] = normalized_exit
    closed_trade["exit_price"] = normalized_exit
    closed_trade["end_time"] = resolved_end_time
    closed_trade["exit_reason"] = str(exit_reason)
    closed_trade["result"] = normalized_result
    closed_trade["actual_outcome"] = (
        "TP"
        if normalized_result == "WIN"
        else "SL"
        if normalized_result == "LOSS"
        else "AMBIGUOUS"
    )
    closed_trade["training_eligible"] = normalized_result in {"WIN", "LOSS"}

    risk_distance = safe_float(closed_trade.get("risk_distance")) or 0.0
    entry = safe_float(closed_trade.get("entry_price", closed_trade.get("entry")))
    direction = closed_trade.get("direction")

    if normalized_result == "AMBIGUOUS" or risk_distance <= 0.0 or entry is None:
        closed_trade["realized_r"] = None
    else:
        signed_move = (
            normalized_exit - entry
            if direction == "BUY"
            else entry - normalized_exit
        )
        closed_trade["realized_r"] = signed_move / risk_distance

    saved_record = _persist_trade_to_setup_archive(runtime, closed_trade)
    closed_trade["storage_status"] = (
        "stored" if saved_record is not None else "error"
    )

    session.trade_history.append(closed_trade)
    session.last_trade_result = normalized_result
    session.open_trade = None

    return copy.deepcopy(closed_trade)


def _trade_hit_flags(
    direction: str,
    candle: Mapping[str, Any],
    sl: float,
    tp: float,
) -> Tuple[bool, bool]:
    """Return whether one candle touches the active stop and target."""
    if direction == "BUY":
        return float(candle["low"]) <= sl, float(candle["high"]) >= tp
    return float(candle["high"]) >= sl, float(candle["low"]) <= tp


def _evaluate_trade_candle(
    trade: Dict[str, Any],
    candle: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one candle against the position's current execution state."""
    direction = str(trade.get("direction", "")).upper()
    sl = safe_float(trade.get("sl"))
    tp = safe_float(trade.get("tp"))
    trigger = safe_float(trade.get("protection_trigger"))
    protected_sl = safe_float(trade.get("protection_sl"))

    if direction not in {"BUY", "SELL"} or sl is None or tp is None:
        return {"status": "invalid"}

    sl_hit, tp_hit = _trade_hit_flags(direction, candle, sl, tp)
    protected = bool(trade.get("protected"))

    if protected:
        if sl_hit and tp_hit:
            return {"status": "conflict", "reason": "SL_AND_TP_SAME_CANDLE"}
        if sl_hit:
            return {
                "status": "closed",
                "result": "LOSS",
                "exit_price": sl,
                "exit_reason": "PROTECTED_SL" if trade.get("protected") else "SL",
            }
        if tp_hit:
            return {
                "status": "closed",
                "result": "WIN",
                "exit_price": tp,
                "exit_reason": "TP",
            }
        return {"status": "no_event"}

    trigger_hit = False
    protected_stop_hit = False
    if trigger is not None and protected_sl is not None:
        trigger_hit = (
            float(candle["high"]) >= trigger
            if direction == "BUY"
            else float(candle["low"]) <= trigger
        )
        protected_stop_hit = (
            float(candle["low"]) <= protected_sl
            if direction == "BUY"
            else float(candle["high"]) >= protected_sl
        )

    if sl_hit and (tp_hit or trigger_hit):
        return {"status": "conflict", "reason": "MULTIPLE_EXIT_EVENTS_SAME_CANDLE"}
    if tp_hit:
        return {
            "status": "closed",
            "result": "WIN",
            "exit_price": tp,
            "exit_reason": "TP",
        }
    if sl_hit:
        return {
            "status": "closed",
            "result": "LOSS",
            "exit_price": sl,
            "exit_reason": "SL",
        }
    if trigger_hit:
        if protected_stop_hit:
            return {
                "status": "conflict",
                "reason": "PROTECTION_TRIGGER_AND_PROTECTED_SL_SAME_CANDLE",
            }
        trade["sl_before_protection"] = sl
        trade["sl"] = protected_sl
        trade["sl_after_protection"] = protected_sl
        trade["protected"] = True
        trade["protected_at"] = int(candle["time"])
        return {"status": "updated", "event": "PROTECTION_ACTIVATED"}

    return {"status": "no_event"}


def _complete_lower_timeframe_candles(
    runtime: RuntimeContext,
    timeframe: str,
    parent_start: int,
    parent_end: int,
) -> Optional[List[Dict[str, Any]]]:
    """Return a complete, gap-free lower-timeframe partition of one parent bar."""
    parent_duration = parent_end - parent_start
    lower_duration = TIMEFRAME_SECONDS.get(timeframe)
    if lower_duration is None or lower_duration <= 0 or parent_duration % lower_duration:
        return None

    expected_times = list(range(parent_start, parent_end, lower_duration))
    by_time: Dict[int, Dict[str, Any]] = {}
    for raw_candle in _timeframe_candles(runtime, timeframe):
        normalized = _validate_candle(raw_candle)
        if normalized is None:
            continue
        timestamp = int(normalized["time"])
        if parent_start <= timestamp and timestamp + lower_duration <= parent_end:
            by_time[timestamp] = normalized

    if not expected_times or any(timestamp not in by_time for timestamp in expected_times):
        return None

    return [by_time[timestamp] for timestamp in expected_times]


def _process_lower_timeframe_path(
    trade: Dict[str, Any],
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
) -> Dict[str, Any]:
    """Replay a complete lower-timeframe path without reading future bars."""
    working_trade = copy.deepcopy(trade)
    duration = int(TIMEFRAME_SECONDS[timeframe])
    events_seen = 0

    for candle in candles:
        result = _evaluate_trade_candle(working_trade, candle)
        if result["status"] == "conflict":
            return {
                "status": "conflict",
                "trade": working_trade,
                "candle": candle,
                "reason": result.get("reason", "LOWER_TIMEFRAME_COLLISION"),
            }
        if result["status"] == "closed":
            return {
                "status": "closed",
                "trade": working_trade,
                "candle": candle,
                "result": result["result"],
                "exit_price": result["exit_price"],
                "exit_reason": f"{result['exit_reason']}_RESOLVED_ON_{timeframe}",
                "end_time": int(candle["time"]) + duration,
            }
        if result["status"] == "updated":
            events_seen += 1

    return {
        "status": "updated" if events_seen else "no_event",
        "trade": working_trade,
    }


def _copy_protection_state(source: Mapping[str, Any], target: Dict[str, Any]) -> None:
    """Copy only mutable protection fields from a resolved lower-timeframe path."""
    for field_name in (
        "sl",
        "protected",
        "protected_at",
        "sl_before_protection",
        "sl_after_protection",
    ):
        target[field_name] = source.get(field_name)


def _lower_timeframe_exit_resolution(
    runtime: RuntimeContext,
    trade: Dict[str, Any],
    parent_candle: Dict[str, Any],
    parent_timeframe: str,
) -> Dict[str, Any]:
    """Resolve an ambiguous parent-bar path with complete M5 and then M1 data."""
    parent_duration = TIMEFRAME_SECONDS.get(parent_timeframe)
    if parent_duration is None:
        return {"status": "ambiguous", "reason": "UNSUPPORTED_PARENT_TIMEFRAME"}

    parent_start = int(parent_candle["time"])
    parent_end = parent_start + int(parent_duration)

    last_conflict: Optional[Dict[str, Any]] = None

    for lower_timeframe in LOWER_TIMEFRAME_PRIORITY:
        lower_duration = TIMEFRAME_SECONDS.get(lower_timeframe)
        if lower_duration is None or lower_duration >= parent_duration:
            continue

        lower_candles = _complete_lower_timeframe_candles(
            runtime,
            lower_timeframe,
            parent_start,
            parent_end,
        )
        if lower_candles is None:
            continue

        resolution = _process_lower_timeframe_path(
            trade,
            lower_candles,
            lower_timeframe,
        )
        if resolution["status"] == "conflict":
            conflict_candle = resolution.get("candle", {})
            conflict_time = _safe_timestamp(conflict_candle.get("time"))
            last_conflict = {
                "status": "ambiguous",
                "reason": resolution.get("reason", "LOWER_TIMEFRAME_COLLISION"),
                "end_time": (
                    conflict_time + int(lower_duration)
                    if conflict_time is not None
                    else parent_end
                ),
            }
            continue
        if resolution["status"] in {"closed", "updated"}:
            resolution["timeframe"] = lower_timeframe
            return resolution

    if last_conflict is not None:
        return last_conflict

    return {
        "status": "ambiguous",
        "reason": "SL_AND_TP_ORDER_UNRESOLVED",
        "end_time": parent_end,
    }


def update_open_trade(
    runtime: RuntimeContext,
    candle: Dict[str, Any],
    *,
    timeframe: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update protection and close a position only when its exit order is known."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    session = runtime.current_session
    trade = session.open_trade
    if trade is None:
        return None

    normalized_candle = _validate_candle(candle)
    if normalized_candle is None:
        raise ValueError("Invalid OHLC candle supplied to update_open_trade().")

    active_timeframe = timeframe or runtime.active_timeframe
    if active_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {active_timeframe}")

    candle_start = int(normalized_candle["time"])
    candle_duration = int(TIMEFRAME_SECONDS[active_timeframe])
    candle_end = candle_start + candle_duration
    entry_time = _safe_timestamp(trade.get("entry_time", trade.get("opened_at")))

    if entry_time is not None and candle_end <= entry_time:
        return None

    if entry_time is not None and candle_start < entry_time:
        return None

    evaluated = _evaluate_trade_candle(trade, normalized_candle)
    if evaluated["status"] == "invalid":
        raise ValueError("The active trade has invalid direction, SL or TP.")

    if evaluated["status"] == "no_event":
        return None

    if evaluated["status"] == "updated":
        return None

    if evaluated["status"] == "closed":
        return close_trade(
            runtime,
            evaluated["result"],
            evaluated["exit_price"],
            evaluated["exit_reason"],
            end_time=candle_end,
        )

    resolution = _lower_timeframe_exit_resolution(
        runtime,
        trade,
        normalized_candle,
        active_timeframe,
    )

    if resolution["status"] == "closed":
        lower_candle = resolution["candle"]
        return close_trade(
            runtime,
            resolution["result"],
            resolution["exit_price"],
            resolution["exit_reason"],
            end_time=int(resolution["end_time"]),
        )

    if resolution["status"] == "updated":
        _copy_protection_state(resolution["trade"], trade)
        return None

    ambiguous_exit = safe_float(trade.get("sl"))
    if ambiguous_exit is None:
        ambiguous_exit = safe_float(trade.get("entry"))
    if ambiguous_exit is None:
        ambiguous_exit = normalized_candle["close"]

    return close_trade(
        runtime,
        "AMBIGUOUS",
        ambiguous_exit,
        resolution.get("reason", "SL_AND_TP_ORDER_UNRESOLVED"),
        end_time=int(resolution.get("end_time", candle_end)),
    )


def update_trade(
    runtime: RuntimeContext,
    candle: Dict[str, Any],
    *,
    timeframe: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compatibility alias for update_open_trade()."""
    return update_open_trade(runtime, candle, timeframe=timeframe)


# ============================================================================
# TRADING CONTROLS AND DETAILS
# ============================================================================

def stop_trading(runtime: RuntimeContext) -> None:
    """Disable new virtual trades for one TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    runtime.current_session.trading_enabled = False


def enable_trading(runtime: RuntimeContext) -> None:
    """Enable new virtual trades for one TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    runtime.current_session.trading_enabled = True


def get_trade_detail(
    runtime: RuntimeContext,
    id: int,
) -> Optional[Dict[str, Any]]:
    """Return the stored details of one open or closed runtime trade."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    try:
        requested_id = int(id)
    except (TypeError, ValueError, OverflowError):
        return None

    session = runtime.current_session
    for trade in session.trade_history:
        if int(trade.get("id", -1)) == requested_id:
            return copy.deepcopy(trade)

    if (
        session.open_trade is not None
        and int(session.open_trade.get("id", -1)) == requested_id
    ):
        return copy.deepcopy(session.open_trade)
    return None


# ============================================================================
# TESTING STEP
# ============================================================================

def step_testing(
    runtime: RuntimeContext,
    *,
    ai_api: Any = None,
    config: Optional[TradingConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Advance one closed candle, manage an open trade, and evaluate a new entry."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    if config is not None and not isinstance(config, TradingConfig):
        raise TypeError("config must be a TradingConfig instance.")

    session: TradingSessionState = runtime.current_session
    candles = runtime.get_chart_data()
    if not candles or session.current_index >= len(candles) - 1:
        return None

    session.current_index += 1
    candle = candles[session.current_index]

    closed_trade = update_open_trade(
        runtime,
        candle,
        timeframe=runtime.active_timeframe,
    )

    if session.open_trade is not None:
        return closed_trade

    if (
        runtime.ai_model is None
        or runtime.ai_scaler is None
        or runtime.ai_market_candles is None
    ):
        return closed_trade

    visible_candles = candles[: session.current_index + 1]
    strategy_result = evaluate_strategy(
        runtime,
        visible_candles,
        ai_api=ai_api,
    )

    prediction = strategy_result.get("prediction")
    if not isinstance(prediction, dict):
        session.blocked_trades += 1
        return closed_trade

    signal = session.last_model_signal or {}
    current_price = safe_float(signal.get("current_price"))
    atr = safe_float(signal.get("atr"))
    trade_config = config or TradingConfig()

    if current_price is None or atr is None or atr <= 0.0:
        session.blocked_trades += 1
        return closed_trade
    if not session.trading_enabled:
        return closed_trade

    action = str(strategy_result.get("action", "WAIT")).strip().upper()
    if action not in {"BUY", "SELL"}:
        session.blocked_trades += 1
        return closed_trade

    setup_quality_probability = _normalize_probability(
        prediction.get("setup_probability")
    )
    if (
        setup_quality_probability is None
        or setup_quality_probability < trade_config.min_setup_probability
    ):
        session.blocked_trades += 1
        return closed_trade

    if (
        session.current_index - session.last_model_entry_index
        < trade_config.model_testing_cooldown
    ):
        session.blocked_trades += 1
        return closed_trade

    offsets = prediction.get("price_offsets_atr", {})
    if not isinstance(offsets, Mapping):
        session.blocked_trades += 1
        return closed_trade

    def offset_price(name: str) -> float:
        offset = safe_float(offsets.get(name))
        return float(current_price + (offset if offset is not None else 0.0) * atr)

    entry_price = offset_price("entry")
    sl_price = offset_price("sl")
    tp_price = offset_price("tp")

    if action == "BUY" and not (sl_price < entry_price < tp_price):
        session.blocked_trades += 1
        return closed_trade
    if action == "SELL" and not (tp_price < entry_price < sl_price):
        session.blocked_trades += 1
        return closed_trade

    timeframe_duration = TIMEFRAME_SECONDS.get(runtime.active_timeframe)
    candle_start = _candle_time(candle)
    if timeframe_duration is None or candle_start is None:
        session.blocked_trades += 1
        return closed_trade

    decision_time = int(candle_start) + int(timeframe_duration)
    signal_time = _safe_timestamp(signal.get("time"))
    predicted_at = signal_time if signal_time is not None else decision_time

    setup = {
        "direction": action,
        "pattern": "AI.PY MODEL SIGNAL",
        "level": offset_price("level"),
        "focus_1": offset_price("focus_1"),
        "focus_2": offset_price("focus_2"),
        "breakout": offset_price("breakout"),
        "entry": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "model_probabilities": copy.deepcopy(
            prediction.get("direction_probabilities", {})
        ),
        "setup_quality_probability": setup_quality_probability,
        "setup_timeframe": prediction.get("setup_timeframe"),
        "entry_timeframe": prediction.get("entry_timeframe"),
        "breakout_time": decision_time,
        "breakout_index": session.current_index,
        "breakout_candle": copy.deepcopy(candle),
        "model_symbol": runtime.market,
        "model_timeframe": runtime.active_timeframe,
        "price_offsets_atr": copy.deepcopy(dict(offsets)),
        "prediction": copy.deepcopy(prediction),
        "predicted_probability_at_entry": _prediction_probability(
            {"prediction": prediction}
        ),
        "predicted_at": predicted_at,
        "linked_pattern_ids": prediction.get("linked_pattern_ids", []),
        "pattern_id": prediction.get("pattern_id"),
    }

    if action == "BUY":
        swing_price = safe_float(candle.get("low"))
        if swing_price is None:
            session.blocked_trades += 1
            return closed_trade
        setup["low1"] = {"time": decision_time, "price": swing_price}
        setup["low2"] = {"time": decision_time, "price": swing_price}
    else:
        swing_price = safe_float(candle.get("high"))
        if swing_price is None:
            session.blocked_trades += 1
            return closed_trade
        setup["high1"] = {"time": decision_time, "price": swing_price}
        setup["high2"] = {"time": decision_time, "price": swing_price}

    opened_trade = open_trade_position(
        runtime,
        action,
        entry_price,
        setup,
        opened_at=decision_time,
        config=trade_config,
    )

    if opened_trade is not None:
        session.last_model_entry_index = session.current_index

    return closed_trade or opened_trade


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "TradingConfig",
    "safe_float",
    "open_trade_position",
    "close_trade",
    "update_open_trade",
    "update_trade",
    "stop_trading",
    "enable_trading",
    "get_trade_detail",
    "step_testing",
]
