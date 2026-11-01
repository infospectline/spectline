"""Market structure analysis and AI-driven strategy evaluation for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence

try:
    from .chart import TIMEFRAME_SECONDS
    from .runtime import RuntimeContext
except ImportError:
    from chart import TIMEFRAME_SECONDS
    from runtime import RuntimeContext


# ============================================================================
# CONFIGURATION
# ============================================================================

LEVEL_RECALC_INTERVAL = 20
MAX_LEVEL_CANDLES = 240
MAX_SWING_CANDLES = 80


# ============================================================================
# BASIC ANALYSIS HELPERS
# ============================================================================

def safe_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float, or return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _safe_float(value: Any) -> Optional[float]:
    """Compatibility alias for code migrated from backend.py."""
    return safe_float(value)


def price_distance(a: float, b: float) -> float:
    """Return the relative distance between two prices."""
    return abs(a - b) / a if a else 0.0


def candle_body_ratio(candle: Dict[str, Any]) -> float:
    """Return the candle body size as a fraction of its full range."""
    candle_range = float(candle["high"]) - float(candle["low"])

    if candle_range <= 0:
        return 0.0

    return abs(
        float(candle["close"]) - float(candle["open"])
    ) / candle_range


# ============================================================================
# SUPPORT AND RESISTANCE ANALYSIS
# ============================================================================

def find_major_levels(
    data: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find major support and resistance levels in the supplied candles."""
    if len(data) < 30:
        return []

    prices = [float(candle["close"]) for candle in data]

    lowest = min(prices)
    highest = max(prices)
    price_range = highest - lowest

    if price_range <= 0:
        return []

    candidates: List[Dict[str, Any]] = [
        {
            "price": lowest,
            "touches": 0,
            "strength": 100.0,
        },
        {
            "price": highest,
            "touches": 0,
            "strength": 100.0,
        },
    ]

    swing_window = max(5, min(15, len(prices) // 30))

    for index in range(
        swing_window,
        len(prices) - swing_window,
    ):
        price = prices[index]
        left = prices[index - swing_window:index]
        right = prices[index + 1:index + swing_window + 1]

        if price >= max(left) and price >= max(right):
            candidates.append({
                "price": price,
                "touches": 0,
                "strength": 0.0,
            })
        elif price <= min(left) and price <= min(right):
            candidates.append({
                "price": price,
                "touches": 0,
                "strength": 0.0,
            })

    touch_tolerance = price_range * 0.015

    for candidate in candidates:
        level = float(candidate["price"])
        touches = 0
        reactions = 0
        last_touch_index = -999

        for index, price in enumerate(prices):
            if abs(price - level) <= touch_tolerance:
                if index - last_touch_index >= 8:
                    touches += 1
                    last_touch_index = index

                    future_end = min(len(prices), index + 12)
                    future = prices[index + 1:future_end]

                    if future:
                        movement = max(
                            abs(future_price - level)
                            for future_price in future
                        )
                        if movement >= price_range * 0.04:
                            reactions += 1

        candidate["touches"] = touches
        candidate["strength"] = touches * 2.0 + reactions * 5.0

    candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["price"] == lowest
            or candidate["price"] == highest
            or candidate["touches"] >= 2
        )
    ]

    candidates.sort(
        key=lambda candidate: float(candidate["strength"]),
        reverse=True,
    )

    minimum_spacing = price_range * 0.08

    low_candidate = next(
        candidate
        for candidate in candidates
        if float(candidate["price"]) == lowest
    )
    high_candidate = next(
        candidate
        for candidate in candidates
        if float(candidate["price"]) == highest
    )

    selected: List[Dict[str, Any]] = [
        {
            "price": lowest,
            "touches": int(low_candidate["touches"]),
            "strength": 100.0,
        },
        {
            "price": highest,
            "touches": int(high_candidate["touches"]),
            "strength": 100.0,
        },
    ]

    for candidate in candidates:
        price = float(candidate["price"])

        if (
            abs(price - lowest) < minimum_spacing
            or abs(price - highest) < minimum_spacing
        ):
            continue

        too_close = any(
            abs(price - float(existing["price"])) < minimum_spacing
            for existing in selected
        )

        if not too_close:
            selected.append(candidate)

    levels = [
        {
            "low": float(candidate["price"]),
            "high": float(candidate["price"]),
            "center": float(candidate["price"]),
            "touches": int(candidate["touches"]),
            "breakouts": 0,
            "reaction_strength": float(candidate["strength"]),
            "strength": float(candidate["strength"]),
        }
        for candidate in selected
    ]

    levels.sort(key=lambda level: float(level["center"]))
    return levels


def find_levels_for_view(
    data: Sequence[Dict[str, Any]],
    from_time: int,
    to_time: int,
) -> List[Dict[str, Any]]:
    """Calculate levels using only candles inside the requested chart view."""
    if not data:
        return []

    visible = [
        candle
        for candle in data
        if int(from_time) <= int(candle["time"]) <= int(to_time)
    ]

    if len(visible) < 30:
        return []

    return find_major_levels(visible)


def nearest_above(
    price: float,
    zones: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the nearest level strictly above the supplied price."""
    values = [
        zone
        for zone in zones
        if float(zone["center"]) > price
    ]

    if not values:
        return None

    return min(values, key=lambda zone: float(zone["center"]))


def nearest_below(
    price: float,
    zones: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the nearest level strictly below the supplied price."""
    values = [
        zone
        for zone in zones
        if float(zone["center"]) < price
    ]

    if not values:
        return None

    return max(values, key=lambda zone: float(zone["center"]))


# ============================================================================
# PRICE STRUCTURE ANALYSIS
# ============================================================================

def detect_zigzag_range(
    line_data: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Return the recent local-high and local-low range."""
    if len(line_data) < 5:
        return None

    recent = list(line_data[-50:])
    highs: List[float] = []
    lows: List[float] = []

    for index in range(1, len(recent) - 1):
        previous = float(recent[index - 1]["close"])
        current = float(recent[index]["close"])
        following = float(recent[index + 1]["close"])

        if current > previous and current > following:
            highs.append(current)

        if current < previous and current < following:
            lows.append(current)

    if not highs or not lows:
        return None

    return {
        "high": max(highs),
        "low": min(lows),
    }


def recent_swing_setup(
    data: Sequence[Dict[str, Any]],
    direction: str,
    level_price: float,
) -> Dict[str, Any]:
    """Return two recent structural lows for BUY or highs for SELL."""
    if len(data) < 2:
        return {"level": float(level_price)}

    recent = list(data[-min(len(data), MAX_SWING_CANDLES):])
    normalized_direction = str(direction).strip().upper()

    if normalized_direction in {"BUY", "LONG"}:
        lows = sorted(
            [
                {
                    "index": index,
                    "time": candle["time"],
                    "price": float(candle["low"]),
                }
                for index, candle in enumerate(recent)
            ],
            key=lambda item: item["price"],
        )[:2]

        if len(lows) < 2:
            lows = [
                {
                    "index": max(0, len(recent) - 2),
                    "time": recent[-2]["time"],
                    "price": float(recent[-2]["low"]),
                },
                {
                    "index": len(recent) - 1,
                    "time": recent[-1]["time"],
                    "price": float(recent[-1]["low"]),
                },
            ]

        lows.sort(key=lambda item: item["index"])
        return {
            "low1": copy.deepcopy(lows[0]),
            "low2": copy.deepcopy(lows[1]),
            "level": float(level_price),
        }

    highs = sorted(
        [
            {
                "index": index,
                "time": candle["time"],
                "price": float(candle["high"]),
            }
            for index, candle in enumerate(recent)
        ],
        key=lambda item: item["price"],
        reverse=True,
    )[:2]

    if len(highs) < 2:
        highs = [
            {
                "index": max(0, len(recent) - 2),
                "time": recent[-2]["time"],
                "price": float(recent[-2]["high"]),
            },
            {
                "index": len(recent) - 1,
                "time": recent[-1]["time"],
                "price": float(recent[-1]["high"]),
            },
        ]

    highs.sort(key=lambda item: item["index"])
    return {
        "high1": copy.deepcopy(highs[0]),
        "high2": copy.deepcopy(highs[1]),
        "level": float(level_price),
    }


def _recent_swing_setup(
    data: Sequence[Dict[str, Any]],
    direction: str,
    level_price: float,
) -> Dict[str, Any]:
    """Compatibility alias for the original backend function name."""
    return recent_swing_setup(data, direction, level_price)


def choose_model_level(
    price: float,
    levels: Sequence[Dict[str, Any]],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Choose the nearest structural level on the relevant side of price."""
    if not levels:
        return None

    normalized_direction = str(direction).strip().upper()

    if normalized_direction in {"BUY", "LONG"}:
        below = [
            level
            for level in levels
            if float(level.get("center", price)) <= price
        ]
        if below:
            return max(
                below,
                key=lambda level: float(level.get("center", price)),
            )
    else:
        above = [
            level
            for level in levels
            if float(level.get("center", price)) >= price
        ]
        if above:
            return min(
                above,
                key=lambda level: float(level.get("center", price)),
            )

    return min(
        levels,
        key=lambda level: abs(float(level.get("center", price)) - price),
    )


def _choose_model_level(
    price: float,
    levels: Sequence[Dict[str, Any]],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Compatibility alias for the original backend function name."""
    return choose_model_level(price, levels, direction)


# ============================================================================
# AI STRATEGY EVALUATION
# ============================================================================

def _resolve_ai_api(ai_api: Any = None) -> Any:
    """Return the AI module, importing it lazily when not supplied."""
    if ai_api is not None:
        return ai_api

    try:
        from . import ai as resolved_ai
    except ImportError:
        import ai as resolved_ai

    return resolved_ai


def evaluate_strategy(
    runtime: RuntimeContext,
    visible_candles: Sequence[Dict[str, Any]],
    ai_api: Any = None,
    level_recalc_interval: int = LEVEL_RECALC_INTERVAL,
) -> Dict[str, Any]:
    """Evaluate the current market state using this TradingBot's AI runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    session = runtime.current_session
    visible = list(visible_candles)

    if not visible:
        return {
            "action": "WAIT",
            "reason": "No visible market data.",
            "levels": [],
            "model_available": bool(session.model_available),
        }

    current_index = int(session.current_index)
    cached_levels = session.cached_strategy_levels
    last_calculation_index = int(session.last_level_calculation_index)

    if (
        last_calculation_index < 0
        or current_index - last_calculation_index >= int(level_recalc_interval)
    ):
        cached_levels = find_major_levels(
            visible[-min(MAX_LEVEL_CANDLES, len(visible)):]
        )
        session.cached_strategy_levels = cached_levels
        session.last_level_calculation_index = current_index

    session.major_levels = cached_levels

    if (
        runtime.ai_model is None
        or runtime.ai_scaler is None
        or runtime.ai_market_candles is None
    ):
        session.model_available = False
        return {
            "action": "WAIT",
            "reason": (
                f"No trained ai.py model exists for {runtime.market}. "
                "Run TRAINING first."
            ),
            "levels": session.major_levels,
            "model_available": False,
        }

    timeframe = runtime.active_timeframe
    timeframe_seconds = TIMEFRAME_SECONDS.get(timeframe)

    if timeframe_seconds is None:
        return {
            "action": "WAIT",
            "reason": f"Unsupported timeframe: {timeframe}",
            "levels": session.major_levels,
            "model_available": bool(session.model_available),
        }

    target_time = int(visible[-1]["time"]) + int(timeframe_seconds)
    ai_module = _resolve_ai_api(ai_api)

    try:
        features = ai_module.market_state_at(
            runtime.ai_market_candles,
            target_time,
        )

        prediction = ai_module.predict_from_features(
            model=runtime.ai_model,
            scaler=runtime.ai_scaler,
            features=features,
        )

    except Exception as error:
        session.model_available = True
        return {
            "action": "WAIT",
            "reason": f"ai.py prediction error: {error}",
            "levels": session.major_levels,
            "model_available": True,
        }

    current_price = ai_module.current_price_at(
        runtime.ai_market_candles,
        target_time,
    )
    atr = ai_module.atr_at(
        runtime.ai_market_candles,
        target_time,
    )

    session.last_model_signal = {
        "time": target_time,
        "price": float(visible[-1]["close"]),
        "prediction": copy.deepcopy(prediction),
        "current_price": current_price,
        "atr": atr,
    }
    session.model_available = True

    direction_name = prediction.get("direction", "WAIT")

    if direction_name == "WAIT":
        action = "WAIT"
    elif direction_name == "LONG":
        action = "BUY"
    else:
        action = "SELL"

    setup_probability = float(
        prediction.get("setup_probability", 0.0)
    )
    reason = (
        f"ai.py {direction_name} | "
        f"setup probability: {setup_probability:.2%}"
    )

    return {
        "action": action,
        "reason": reason,
        "levels": session.major_levels,
        "model_available": True,
        "probabilities": prediction.get("direction_probabilities", {}),
        "setup_probability": prediction.get("setup_probability", 0.0),
        "setup_timeframe": prediction.get("setup_timeframe"),
        "entry_timeframe": prediction.get("entry_timeframe"),
        "prediction": prediction,
    }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "LEVEL_RECALC_INTERVAL",
    "MAX_LEVEL_CANDLES",
    "MAX_SWING_CANDLES",
    "safe_float",
    "price_distance",
    "candle_body_ratio",
    "find_major_levels",
    "find_levels_for_view",
    "nearest_above",
    "nearest_below",
    "detect_zigzag_range",
    "recent_swing_setup",
    "choose_model_level",
    "evaluate_strategy",
]
