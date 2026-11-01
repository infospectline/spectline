"""Virtual trade execution and testing-step management for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chart import TIMEFRAME_SECONDS
from runtime import RuntimeContext, TradingSessionState

try:
    from strategy import evaluate_strategy
except ImportError:
    from .strategy import evaluate_strategy


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROTECTION_TRIGGER_FRACTION = 0.33
DEFAULT_PROTECTION_SL_FRACTION = 0.05
DEFAULT_MIN_SETUP_PROBABILITY = 0.50
DEFAULT_MODEL_TESTING_COOLDOWN = 0

LOWER_TIMEFRAME_PRIORITY = ("M5", "M1")


@dataclass(frozen=True)
class TradingConfig:
    """Execution settings used by one testing or virtual-trading run."""

    protection_trigger_fraction: float = DEFAULT_PROTECTION_TRIGGER_FRACTION
    protection_sl_fraction: float = DEFAULT_PROTECTION_SL_FRACTION
    min_setup_probability: float = DEFAULT_MIN_SETUP_PROBABILITY
    model_testing_cooldown: int = DEFAULT_MODEL_TESTING_COOLDOWN

    def __post_init__(self) -> None:
        if not 0.0 <= self.protection_trigger_fraction <= 1.0:
            raise ValueError("protection_trigger_fraction must be between 0 and 1.")
        if not 0.0 <= self.protection_sl_fraction <= 1.0:
            raise ValueError("protection_sl_fraction must be between 0 and 1.")
        if not 0.0 <= self.min_setup_probability <= 1.0:
            raise ValueError("min_setup_probability must be between 0 and 1.")
        if self.model_testing_cooldown < 0:
            raise ValueError("model_testing_cooldown cannot be negative.")


# ============================================================================
# VALUE AND CANDLE HELPERS
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


def _candle_value(candle: Any, key: str) -> Optional[float]:
    """Read one numeric candle field from a dictionary or Candle object."""
    if isinstance(candle, dict):
        value = candle.get(key)
    else:
        value = getattr(candle, key, None)

    return safe_float(value)


def _candle_time(candle: Any) -> Optional[int]:
    """Read a candle timestamp from a dictionary or Candle object."""
    if isinstance(candle, dict):
        value = candle.get("time")
    else:
        value = getattr(candle, "time", None)

    numeric = safe_float(value)
    return int(numeric) if numeric is not None else None


def _validate_candle(candle: Any) -> Optional[Dict[str, float]]:
    """Normalize the OHLC fields required by trade execution."""
    timestamp = _candle_time(candle)
    open_price = _candle_value(candle, "open")
    high = _candle_value(candle, "high")
    low = _candle_value(candle, "low")
    close = _candle_value(candle, "close")

    if None in (timestamp, open_price, high, low, close):
        return None

    if high < low:
        return None

    return {
        "time": int(timestamp),
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
    }


def _trade_hit_flags(
    direction: str,
    candle: Dict[str, float],
    sl: float,
    tp: float,
) -> Tuple[bool, bool]:
    """Return whether the candle touches SL and TP."""
    if direction == "BUY":
        return candle["low"] <= sl, candle["high"] >= tp

    return candle["high"] >= sl, candle["low"] <= tp


def _timeframe_candles(
    runtime: RuntimeContext,
    timeframe: str,
) -> List[Any]:
    """Return a timeframe's historical candles from the AI market archive."""
    archive = runtime.ai_market_candles
    if not isinstance(archive, dict):
        return []

    values = archive.get(timeframe, [])
    return list(values) if values is not None else []


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
    """Open a virtual position inside the supplied TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    if setup is None or not isinstance(setup, dict):
        return None

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
        index = min(max(int(session.current_index), 0), len(candles) - 1)
        opened_at = _candle_time(candles[index])

    if opened_at is None:
        return None

    trade_id = max(
        (int(item.get("trade_id", 0)) for item in session.trade_history),
        default=0,
    ) + 1

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

    trade = {
        "trade_id": trade_id,
        "direction": normalized_direction,
        "entry": entry_price,
        "opened_at": int(opened_at),
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
        "entry_reason": setup.get("reason", "Market model signal"),
        "setup_level": safe_float(setup.get("level")) or entry_price,
        "selected_level": copy.deepcopy(setup.get("selected_level")),
        "level_strength": safe_float(setup.get("level_strength")) or 0.0,
        "level_touches": int(safe_float(setup.get("level_touches")) or 0),
        "sl_basis": {
            "type": "MODEL",
            "reference": structural_reference_name,
            "reference_price": structural_reference_price,
            "buffer_percent": 0.0,
        },
        "breakout_time": setup.get("breakout_time"),
        "breakout_index": setup.get("breakout_index"),
        "breakout_candle": copy.deepcopy(setup.get("breakout_candle")),
        "setup_swings": setup_swings,
        "setup_snapshot": copy.deepcopy(setup),
        "model_probabilities": copy.deepcopy(setup.get("model_probabilities", {})),
        "model_symbol": runtime.market,
        "model_timeframe": runtime.active_timeframe,
        "identity": runtime.identity.as_dict(),
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

    session.open_trade = trade
    return copy.deepcopy(trade)


# ============================================================================
# TRADE CLOSING AND POSITION MANAGEMENT
# ============================================================================

def close_trade(
    runtime: RuntimeContext,
    result: str,
    exit_price: float,
    exit_reason: str,
    *,
    exit_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Close the active virtual trade and append its immutable result record."""
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

    closed_trade = copy.deepcopy(trade)
    closed_trade["exit"] = normalized_exit
    closed_trade["closed_at"] = int(exit_time) if exit_time is not None else None
    closed_trade["exit_reason"] = str(exit_reason)
    closed_trade["result"] = normalized_result
    closed_trade["training_eligible"] = normalized_result in {"WIN", "LOSS"}

    risk_distance = safe_float(closed_trade.get("risk_distance")) or 0.0
    entry = safe_float(closed_trade.get("entry")) or normalized_exit
    direction = closed_trade.get("direction")

    if risk_distance > 0.0:
        signed_move = (
            normalized_exit - entry
            if direction == "BUY"
            else entry - normalized_exit
        )
        closed_trade["realized_r"] = signed_move / risk_distance
    else:
        closed_trade["realized_r"] = None

    session.trade_history.append(closed_trade)
    session.last_trade_result = normalized_result
    session.open_trade = None

    return copy.deepcopy(closed_trade)


def _lower_timeframe_exit_resolution(
    runtime: RuntimeContext,
    trade: Dict[str, Any],
    parent_candle: Dict[str, float],
    parent_timeframe: str,
    sl: float,
    tp: float,
) -> Optional[Tuple[str, float, int, str]]:
    """Resolve a same-candle SL/TP collision using M5 and then M1 candles.

    Return None when the parent candle did not contain a collision. Return a
    tuple when the order is resolved. Return an AMBIGUOUS tuple when M1 still
    cannot determine which level was reached first.
    """
    parent_duration = TIMEFRAME_SECONDS.get(parent_timeframe)
    if parent_duration is None:
        return None

    parent_start = int(parent_candle["time"])
    parent_end = parent_start + int(parent_duration)
    direction = str(trade.get("direction", "")).upper()

    for lower_timeframe in LOWER_TIMEFRAME_PRIORITY:
        if TIMEFRAME_SECONDS.get(lower_timeframe, 0) >= parent_duration:
            continue

        lower_duration = TIMEFRAME_SECONDS[lower_timeframe]
        lower_candles = _timeframe_candles(runtime, lower_timeframe)
        relevant: List[Dict[str, float]] = []

        for raw_candle in lower_candles:
            normalized = _validate_candle(raw_candle)
            if normalized is None:
                continue
            if parent_start <= normalized["time"] and normalized["time"] + lower_duration <= parent_end:
                relevant.append(normalized)

        relevant.sort(key=lambda item: item["time"])
        if not relevant:
            continue

        for lower_candle in relevant:
            sl_hit, tp_hit = _trade_hit_flags(
                direction,
                lower_candle,
                sl,
                tp,
            )

            if sl_hit and tp_hit:
                if lower_timeframe == "M1":
                    return (
                        "AMBIGUOUS",
                        sl,
                        lower_candle["time"],
                        "SL_AND_TP_SAME_M1_CANDLE",
                    )
                break

            if sl_hit:
                return (
                    "LOSS",
                    sl,
                    lower_candle["time"],
                    f"SL_RESOLVED_ON_{lower_timeframe}",
                )

            if tp_hit:
                return (
                    "WIN",
                    tp,
                    lower_candle["time"],
                    f"TP_RESOLVED_ON_{lower_timeframe}",
                )

        else:
            continue

        # An unresolved M5 collision is examined again using M1 data.
        if lower_timeframe == "M5":
            continue

    return (
        "AMBIGUOUS",
        sl,
        parent_start,
        "SL_AND_TP_ORDER_UNRESOLVED",
    )


def update_open_trade(
    runtime: RuntimeContext,
    candle: Dict[str, Any],
    *,
    timeframe: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update protection and close a position when SL or TP is reached.

    If a candle touches both SL and TP, lower M5 and M1 history is used to
    establish the order. A collision unresolved on M1 is recorded as
    AMBIGUOUS and is not eligible as a WIN/LOSS training result.
    """
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

    direction = str(trade.get("direction", "")).upper()
    sl = safe_float(trade.get("sl"))
    tp = safe_float(trade.get("tp"))
    trigger = safe_float(trade.get("protection_trigger"))
    protected_sl = safe_float(trade.get("protection_sl"))

    if direction not in {"BUY", "SELL"} or sl is None or tp is None:
        raise ValueError("The active trade has invalid direction, SL or TP.")

    sl_hit, tp_hit = _trade_hit_flags(
        direction,
        normalized_candle,
        sl,
        tp,
    )

    if sl_hit and tp_hit:
        resolution = _lower_timeframe_exit_resolution(
            runtime,
            trade,
            normalized_candle,
            active_timeframe,
            sl,
            tp,
        )
        if resolution is None:
            resolution = (
                "AMBIGUOUS",
                sl,
                normalized_candle["time"],
                "SL_AND_TP_ORDER_UNRESOLVED",
            )

        result, exit_price, exit_time, reason = resolution
        return close_trade(
            runtime,
            result,
            exit_price,
            reason,
            exit_time=exit_time,
        )

    if sl_hit:
        return close_trade(
            runtime,
            "LOSS",
            sl,
            "SL",
            exit_time=normalized_candle["time"],
        )

    if tp_hit:
        return close_trade(
            runtime,
            "WIN",
            tp,
            "TP",
            exit_time=normalized_candle["time"],
        )

    if not bool(trade.get("protected")) and trigger is not None and protected_sl is not None:
        trigger_hit = (
            normalized_candle["high"] >= trigger
            if direction == "BUY"
            else normalized_candle["low"] <= trigger
        )

        if trigger_hit:
            protected_stop_hit = (
                normalized_candle["low"] <= protected_sl
                if direction == "BUY"
                else normalized_candle["high"] >= protected_sl
            )

            if protected_stop_hit:
                return close_trade(
                    runtime,
                    "AMBIGUOUS",
                    protected_sl,
                    "PROTECTION_TRIGGER_AND_PROTECTED_SL_SAME_CANDLE",
                    exit_time=normalized_candle["time"],
                )

            trade["sl_before_protection"] = sl
            trade["sl"] = protected_sl
            trade["sl_after_protection"] = protected_sl
            trade["protected"] = True
            trade["protected_at"] = normalized_candle["time"]

    return None


def update_trade(
    runtime: RuntimeContext,
    candle: Dict[str, Any],
    *,
    timeframe: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compatibility alias for update_open_trade()."""
    return update_open_trade(
        runtime,
        candle,
        timeframe=timeframe,
    )


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
    trade_id: int,
) -> Optional[Dict[str, Any]]:
    """Return the stored details of one open or closed trade."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    session = runtime.current_session
    requested_id = int(trade_id)

    for trade in session.trade_history:
        if int(trade.get("trade_id", -1)) == requested_id:
            return copy.deepcopy(trade)

    if (
        session.open_trade is not None
        and int(session.open_trade.get("trade_id", -1)) == requested_id
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
    """Advance one candle and use the runtime's AI model to create a trade."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    session: TradingSessionState = runtime.current_session
    candles = runtime.get_chart_data()

    if not candles or session.current_index >= len(candles) - 1:
        return None

    session.current_index += 1
    candle = candles[session.current_index]

    # Update an existing position before asking the model for another entry.
    closed_trade = update_open_trade(
        runtime,
        candle,
        timeframe=runtime.active_timeframe,
    )

    # No second position can be opened while the first one remains active.
    if session.open_trade is not None:
        return closed_trade

    if (
        runtime.ai_model is None
        or runtime.ai_scaler is None
        or runtime.ai_market_candles is None
    ):
        return closed_trade

    visible_candles = candles[:session.current_index + 1]
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

    if strategy_result.get("action") == "WAIT":
        session.blocked_trades += 1
        return closed_trade

    setup_probability = safe_float(prediction.get("setup_probability")) or 0.0
    if setup_probability < trade_config.min_setup_probability:
        session.blocked_trades += 1
        return closed_trade

    if (
        session.current_index - session.last_model_entry_index
        < trade_config.model_testing_cooldown
    ):
        session.blocked_trades += 1
        return closed_trade

    direction = "BUY" if strategy_result.get("action") == "BUY" else "SELL"
    offsets = prediction.get("price_offsets_atr", {})
    if not isinstance(offsets, dict):
        session.blocked_trades += 1
        return closed_trade

    def offset_price(name: str) -> float:
        offset = safe_float(offsets.get(name)) or 0.0
        return float(current_price + offset * atr)

    entry_price = offset_price("entry")
    sl_price = offset_price("sl")
    tp_price = offset_price("tp")

    if direction == "BUY" and not (sl_price < entry_price < tp_price):
        session.blocked_trades += 1
        return closed_trade
    if direction == "SELL" and not (tp_price < entry_price < sl_price):
        session.blocked_trades += 1
        return closed_trade

    target_time = int(candle["time"]) + int(TIMEFRAME_SECONDS[runtime.active_timeframe])

    setup = {
        "direction": direction,
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
        "setup_probability": setup_probability,
        "setup_timeframe": prediction.get("setup_timeframe"),
        "entry_timeframe": prediction.get("entry_timeframe"),
        "breakout_time": target_time,
        "breakout_index": session.current_index,
        "breakout_candle": copy.deepcopy(candle),
        "model_symbol": runtime.market,
        "model_timeframe": runtime.active_timeframe,
        "price_offsets_atr": copy.deepcopy(offsets),
    }

    if direction == "BUY":
        swing_price = safe_float(candle.get("low"))
        setup["low1"] = {"time": target_time, "price": swing_price}
        setup["low2"] = {"time": target_time, "price": swing_price}
    else:
        swing_price = safe_float(candle.get("high"))
        setup["high1"] = {"time": target_time, "price": swing_price}
        setup["high2"] = {"time": target_time, "price": swing_price}

    opened_trade = open_trade_position(
        runtime,
        direction,
        entry_price,
        setup,
        opened_at=int(candle["time"]),
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
