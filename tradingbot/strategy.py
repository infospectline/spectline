"""Market structure analysis and AI-driven strategy evaluation for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


def _candle_field(candle: Any, field_name: str) -> Any:
    if isinstance(candle, Mapping):
        return candle.get(field_name)
    return getattr(candle, field_name, None)


def _timestamp_seconds(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    if abs(number) >= 100_000_000_000:
        number /= 1000.0
    return int(number)


def _closed_candles_by_timeframe(
    runtime: RuntimeContext,
    visible_candles: Sequence[Any],
    decision_time: int,
) -> Dict[str, List[Any]]:
    """Collect only timeframe candles fully closed at the decision time."""
    sources: Dict[str, List[Any]] = {}

    shared = getattr(runtime, "ai_market_candles", None)
    if isinstance(shared, Mapping):
        for timeframe, candles in shared.items():
            if isinstance(candles, Sequence) and not isinstance(candles, (str, bytes)):
                sources[str(timeframe).strip().upper()] = list(candles)

    chart_sources = getattr(runtime, "chart_data_by_timeframe", None)
    if isinstance(chart_sources, Mapping):
        for timeframe, candles in chart_sources.items():
            key = str(timeframe).strip().upper()
            if isinstance(candles, Sequence) and not isinstance(candles, (str, bytes)):
                existing = sources.setdefault(key, [])
                existing.extend(candles)

    m1_data = getattr(runtime, "m1_data", None)
    if isinstance(m1_data, Sequence) and not isinstance(m1_data, (str, bytes)) and m1_data:
        sources.setdefault("M1", []).extend(m1_data)

    active_timeframe = str(runtime.active_timeframe).strip().upper()
    sources.setdefault(active_timeframe, []).extend(list(visible_candles))

    result: Dict[str, List[Any]] = {}
    for timeframe, candles in sources.items():
        duration = TIMEFRAME_SECONDS.get(timeframe)
        if duration is None:
            continue

        unique: Dict[int, Any] = {}
        for candle in candles:
            candle_time = _timestamp_seconds(_candle_field(candle, "time"))
            if candle_time is None:
                continue
            if candle_time + int(duration) > int(decision_time):
                continue
            unique[candle_time] = candle

        if unique:
            result[timeframe] = [unique[key] for key in sorted(unique)]

    return result


def _calculate_atr(candles: Sequence[Any], period: int = 14) -> Optional[float]:
    if len(candles) < 2:
        return None

    ranges: List[float] = []
    start = max(1, len(candles) - int(period))
    for index in range(start, len(candles)):
        candle = candles[index]
        previous = candles[index - 1]
        high = safe_float(_candle_field(candle, "high"))
        low = safe_float(_candle_field(candle, "low"))
        previous_close = safe_float(_candle_field(previous, "close"))
        if high is None or low is None or previous_close is None or high < low:
            continue
        ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    if not ranges:
        return None
    value = sum(ranges) / len(ranges)
    return value if math.isfinite(value) and value > 0.0 else None


def _normalize_direction(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    if normalized in {"BUY", "LONG"}:
        return "LONG"
    if normalized in {"SELL", "SHORT"}:
        return "SHORT"
    return None


def _candidate_from_runtime(runtime: RuntimeContext) -> Optional[Dict[str, Any]]:
    """Read an explicitly supplied candidate without inventing trade direction."""
    candidates = (
        getattr(runtime, "_spectline_pending_setup", None),
        getattr(runtime, "pending_setup", None),
        getattr(runtime.current_session, "pending_setup", None),
        getattr(runtime.current_session, "strategy_candidate", None),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return copy.deepcopy(dict(candidate))
    return None


def set_pending_setup(runtime: RuntimeContext, setup: Mapping[str, Any]) -> None:
    """Attach an explicit trade candidate to a TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    if not isinstance(setup, Mapping):
        raise TypeError("setup must be a mapping.")
    with runtime.lock:
        setattr(runtime, "_spectline_pending_setup", copy.deepcopy(dict(setup)))


def clear_pending_setup(runtime: RuntimeContext) -> None:
    """Remove the pending trade candidate from a TradingBot runtime."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")
    with runtime.lock:
        if hasattr(runtime, "_spectline_pending_setup"):
            delattr(runtime, "_spectline_pending_setup")


def _normalize_candidate_setup(
    runtime: RuntimeContext,
    candidate: Mapping[str, Any],
    decision_time: int,
    current_price: float,
) -> Dict[str, Any]:
    """Validate and normalize a candidate for point-in-time TP-first inference."""
    direction = _normalize_direction(
        candidate.get("direction", candidate.get("side", candidate.get("action")))
    )
    if direction is None:
        raise ValueError("A candidate setup must explicitly specify BUY/LONG or SELL/SHORT.")

    supplied_decision = candidate.get("decision_time")
    if supplied_decision is not None:
        parsed_decision = _timestamp_seconds(supplied_decision)
        if parsed_decision is None or parsed_decision != int(decision_time):
            raise ValueError("Candidate decision_time must match the latest closed candle boundary.")

    supplied_entry_time = candidate.get("entry_time", candidate.get("entry_candle_time"))
    if supplied_entry_time is not None:
        parsed_entry_time = _timestamp_seconds(supplied_entry_time)
        if parsed_entry_time is None or parsed_entry_time != int(decision_time):
            raise ValueError("A live entry candidate must have entry_time equal to decision_time.")

    entry = safe_float(candidate.get("entry", candidate.get("entry_price")))
    stop = safe_float(candidate.get("sl", candidate.get("stop_loss")))
    target = safe_float(candidate.get("tp", candidate.get("take_profit")))
    if entry is None:
        entry = float(current_price)
    if stop is None or target is None:
        raise ValueError("A candidate setup must explicitly provide both SL and TP prices.")
    if min(entry, stop, target) <= 0.0:
        raise ValueError("Entry, SL and TP prices must be positive.")

    if direction == "LONG" and not stop < entry < target:
        raise ValueError("A LONG candidate must satisfy SL < entry < TP.")
    if direction == "SHORT" and not target < entry < stop:
        raise ValueError("A SHORT candidate must satisfy TP < entry < SL.")

    setup_timeframe = str(
        candidate.get("setup_timeframe") or runtime.active_timeframe
    ).strip().upper()
    entry_timeframe = str(
        candidate.get("entry_timeframe") or runtime.active_timeframe
    ).strip().upper()
    resolution_timeframe = str(
        candidate.get("resolution_timeframe") or entry_timeframe
    ).strip().upper()
    for field_name, timeframe in (
        ("setup_timeframe", setup_timeframe),
        ("entry_timeframe", entry_timeframe),
        ("resolution_timeframe", resolution_timeframe),
    ):
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported {field_name}: {timeframe}.")

    candidate_payload = {
        "market": runtime.market,
        "tradingbot": runtime.tradingbot_id,
        "timeframe": entry_timeframe,
        "direction": direction,
        "decision_time": int(decision_time),
        "entry": entry,
        "sl": stop,
        "tp": target,
        "seed": candidate.get("setup_id", candidate.get("id")),
    }
    candidate_key = json.dumps(
        candidate_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    generated_id = "live_" + hashlib.sha256(candidate_key.encode("utf-8")).hexdigest()[:24]
    setup_id = str(candidate.get("setup_id", candidate.get("id", generated_id))).strip()
    if not setup_id:
        setup_id = generated_id

    zones = candidate.get("zones", [])
    if zones is None:
        zones = []
    if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)):
        raise ValueError("Candidate zones must be a sequence of zone mappings.")
    normalized_zones = [
        copy.deepcopy(dict(zone))
        for zone in zones
        if isinstance(zone, Mapping)
    ]

    normalized = {
        "setup_id": setup_id,
        "trade_id": str(candidate.get("trade_id") or setup_id),
        "direction": direction,
        "setup_timeframe": setup_timeframe,
        "entry_timeframe": entry_timeframe,
        "resolution_timeframe": resolution_timeframe,
        "decision_time": int(decision_time),
        "entry_time": int(decision_time),
        "anchor_time": _timestamp_seconds(candidate.get("anchor_time")) or int(decision_time),
        "anchor_price": safe_float(candidate.get("anchor_price")) or entry,
        "breakout_time": _timestamp_seconds(candidate.get("breakout_time")) or int(decision_time),
        "breakout": safe_float(candidate.get("breakout")) or entry,
        "entry": entry,
        "sl": stop,
        "tp": target,
        "rr_mode": str(candidate.get("rr_mode") or "OFF").strip().upper(),
        "zones": normalized_zones,
    }

    quality = candidate.get(
        "setup_quality_probability",
        candidate.get("setup_probability"),
    )
    if quality is not None:
        quality_value = safe_float(quality)
        if quality_value is None or not 0.0 <= quality_value <= 1.0:
            raise ValueError("setup_quality_probability must be between 0 and 1.")
        normalized["setup_quality_probability"] = quality_value

    for optional_name in ("pattern_id", "linked_pattern_ids"):
        if optional_name in candidate:
            normalized[optional_name] = copy.deepcopy(candidate[optional_name])

    for price_name in ("level", "focus_1", "focus_2"):
        value = safe_float(candidate.get(price_name))
        if value is not None:
            normalized[price_name] = value

    return normalized


def _wait_decision(
    reason: str,
    levels: Sequence[Dict[str, Any]],
    model_available: bool,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "action": "WAIT",
        "reason": str(reason),
        "levels": list(levels),
        "model_available": bool(model_available),
        **extra,
    }


def evaluate_strategy(
    runtime: RuntimeContext,
    visible_candles: Sequence[Dict[str, Any]],
    ai_api: Any = None,
    level_recalc_interval: int = LEVEL_RECALC_INTERVAL,
    *,
    candidate_setup: Optional[Mapping[str, Any]] = None,
    probability_threshold: float = 0.50,
    persist_prediction: bool = True,
    require_calibrated: bool = False,
) -> Dict[str, Any]:
    """Score an explicit BUY/SELL candidate with the approved TP-first model.

    The model estimates success probability conditional on a supplied setup. It
    does not generate trade direction, entry, stop-loss or take-profit levels.
    """
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    threshold = safe_float(probability_threshold)
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1.")

    session = runtime.current_session
    visible = list(visible_candles or [])
    if not visible:
        return _wait_decision(
            "No visible closed market candles are available.",
            session.major_levels,
            session.model_available,
        )

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
        return _wait_decision(
            getattr(runtime, "ai_error", None)
            or f"No approved TP-before-SL model is loaded for {runtime.market}.",
            session.major_levels,
            False,
        )

    timeframe = str(runtime.active_timeframe).strip().upper()
    timeframe_seconds = TIMEFRAME_SECONDS.get(timeframe)
    if timeframe_seconds is None:
        return _wait_decision(
            f"Unsupported timeframe: {timeframe}.",
            session.major_levels,
            session.model_available,
        )

    candle_time = _timestamp_seconds(_candle_field(visible[-1], "time"))
    current_price = safe_float(_candle_field(visible[-1], "close"))
    if candle_time is None or current_price is None or current_price <= 0.0:
        return _wait_decision(
            "The latest candle has an invalid timestamp or closing price.",
            session.major_levels,
            session.model_available,
        )

    decision_time = int(candle_time) + int(timeframe_seconds)
    candles_by_timeframe = _closed_candles_by_timeframe(
        runtime,
        visible,
        decision_time,
    )
    if not candles_by_timeframe.get(timeframe):
        return _wait_decision(
            "No fully closed candles are available for the active timeframe.",
            session.major_levels,
            session.model_available,
        )

    atr = _calculate_atr(candles_by_timeframe[timeframe])
    if atr is None or atr <= 0.0:
        return _wait_decision(
            "Insufficient closed candles to calculate a valid ATR.",
            session.major_levels,
            session.model_available,
        )

    candidate = (
        copy.deepcopy(dict(candidate_setup))
        if isinstance(candidate_setup, Mapping)
        else _candidate_from_runtime(runtime)
    )
    if candidate is None:
        session.model_available = True
        session.last_model_signal = {
            "time": decision_time,
            "price": current_price,
            "current_price": current_price,
            "atr": atr,
            "status": "WAIT_NO_CANDIDATE",
        }
        return _wait_decision(
            "The TP-before-SL model scores a defined trade setup; it does not predict BUY/SELL direction or invent SL/TP levels. Supply a candidate setup with direction, SL and TP.",
            session.major_levels,
            True,
            decision_time=decision_time,
            current_price=current_price,
            atr=atr,
            prediction=None,
        )

    try:
        normalized_candidate = _normalize_candidate_setup(
            runtime,
            candidate,
            decision_time,
            current_price,
        )
        ai_module = _resolve_ai_api(ai_api)
        predictor = getattr(ai_module, "predict_setup_from_context", None)
        if not callable(predictor):
            raise AttributeError("ai.py does not expose predict_setup_from_context().")

        prediction = predictor(
            normalized_candidate,
            candles_by_timeframe,
            market=runtime.market,
            identity=runtime.identity,
            prediction_time=decision_time,
            threshold=threshold,
            persist=bool(persist_prediction),
            allow_post_entry_analysis=False,
        )
        if not isinstance(prediction, Mapping):
            raise TypeError("The inference API returned a non-mapping prediction.")
        prediction = copy.deepcopy(dict(prediction))

        probability = safe_float(
            prediction.get(
                "tp_before_sl_probability",
                prediction.get("probability"),
            )
        )
        if probability is None or not 0.0 <= probability <= 1.0:
            raise ValueError("The inference API returned an invalid TP-before-SL probability.")

        calibrated = bool(prediction.get("is_calibrated", False))
        if require_calibrated and not calibrated:
            raise ValueError("The approved model has not passed probability calibration.")

        side = normalized_candidate["direction"]
        action = "BUY" if side == "LONG" else "SELL"
        if probability < threshold:
            action = "WAIT"

        setup_quality = normalized_candidate.get("setup_quality_probability")
        if setup_quality is not None:
            prediction["setup_quality_probability"] = setup_quality
            prediction["setup_probability"] = setup_quality

        prediction["direction"] = side
        prediction["setup_timeframe"] = normalized_candidate["setup_timeframe"]
        prediction["entry_timeframe"] = normalized_candidate["entry_timeframe"]
        prediction["resolution_timeframe"] = normalized_candidate["resolution_timeframe"]
        prediction["price_offsets_atr"] = {
            "entry": (normalized_candidate["entry"] - current_price) / atr,
            "sl": (normalized_candidate["sl"] - current_price) / atr,
            "tp": (normalized_candidate["tp"] - current_price) / atr,
            "level": (
                (normalized_candidate.get("level", normalized_candidate["anchor_price"]) - current_price)
                / atr
            ),
            "focus_1": (
                (normalized_candidate.get("focus_1", normalized_candidate["entry"]) - current_price)
                / atr
            ),
            "focus_2": (
                (normalized_candidate.get("focus_2", normalized_candidate["entry"]) - current_price)
                / atr
            ),
            "breakout": (normalized_candidate["breakout"] - current_price) / atr,
        }
        prediction["pattern_id"] = normalized_candidate.get("pattern_id")
        prediction["linked_pattern_ids"] = copy.deepcopy(
            normalized_candidate.get("linked_pattern_ids", [])
        )

        session.last_model_signal = {
            "time": decision_time,
            "price": current_price,
            "current_price": current_price,
            "atr": atr,
            "prediction": copy.deepcopy(prediction),
            "candidate_setup": copy.deepcopy(normalized_candidate),
        }
        session.model_available = True

        reason = (
            f"{side} candidate | TP-before-SL probability: {probability:.2%} "
            f"(threshold: {threshold:.2%})"
        )
        if not calibrated:
            reason += " | probability is not calibrated"
        if setup_quality is None:
            reason += " | setup quality score not supplied"

        return {
            "action": action,
            "reason": reason,
            "levels": session.major_levels,
            "model_available": True,
            "probability": probability,
            "tp_before_sl_probability": probability,
            "predicted_probability_at_entry": prediction.get(
                "predicted_probability_at_entry"
            ),
            "probability_calibrated": calibrated,
            "setup_quality_probability": setup_quality,
            "setup_timeframe": normalized_candidate["setup_timeframe"],
            "entry_timeframe": normalized_candidate["entry_timeframe"],
            "decision_time": decision_time,
            "candidate_setup": normalized_candidate,
            "prediction": prediction,
        }

    except Exception as error:
        session.model_available = True
        session.last_model_signal = {
            "time": decision_time,
            "price": current_price,
            "current_price": current_price,
            "atr": atr,
            "status": "PREDICTION_REJECTED",
            "error": str(error),
        }
        return _wait_decision(
            f"TP-before-SL inference rejected the candidate: {error}",
            session.major_levels,
            True,
            decision_time=decision_time,
            current_price=current_price,
            atr=atr,
        )


def evaluate_setup(
    runtime: RuntimeContext,
    setup: Mapping[str, Any],
    visible_candles: Optional[Sequence[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate one explicit trade setup using the active TradingBot runtime."""
    candles = visible_candles
    if candles is None:
        candles = runtime.get_chart_data()
    return evaluate_strategy(
        runtime,
        candles,
        candidate_setup=setup,
        **kwargs,
    )


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
    "set_pending_setup",
    "clear_pending_setup",
    "evaluate_setup",
    "evaluate_strategy",
]
