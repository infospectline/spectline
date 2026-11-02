"""Numerical market features and automatic Support/Resistance for Spectline AI."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from engine.data import Candle, UserSetup
    from engine.history import (
        last_closed_index,
        rolling_mean,
        rolling_slope,
        rolling_std,
        slice_closed_history,
        true_range,
    )
    from engine.settings import (
        DIRECTION_TO_INDEX,
        LOOKBACK_BY_TIMEFRAME,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
        TIMEFRAME_TO_INDEX,
    )
except ImportError:
    from .data import Candle, UserSetup
    from .history import (
        last_closed_index,
        rolling_mean,
        rolling_slope,
        rolling_std,
        slice_closed_history,
        true_range,
    )
    from .settings import (
        DIRECTION_TO_INDEX,
        LOOKBACK_BY_TIMEFRAME,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
        TIMEFRAME_TO_INDEX,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

# Keep the AI level window aligned with strategy.MAX_LEVEL_CANDLES while
# calculating levels independently from the chart viewport.
AI_LEVEL_LOOKBACK_CANDLES = 240

# The numerical candle representation contains this many values per timeframe.
TIMEFRAME_FEATURE_SIZE = 20

# Candle priority used when a single representative price is required.
PRICE_TIMEFRAME_PRIORITY = (
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)

ATR_TIMEFRAME_PRIORITY = (
    "M15",
    "H1",
    "M5",
    "M1",
    "M30",
    "H4",
    "D1",
    "W1",
)

# Stable field order for the setup price targets retained by the AI engine.
SETUP_PRICE_TARGET_FIELDS = (
    "anchor_price",
    "breakout",
    "entry",
    "sl",
    "tp",
)
SETUP_PRICE_TARGET_SIZE = len(SETUP_PRICE_TARGET_FIELDS)


# ============================================================================
# BASIC NUMERICAL HELPERS
# ============================================================================

def _finite_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def sigmoid(value: float) -> float:
    """Return a numerically stable scalar sigmoid."""
    bounded = max(min(float(value), 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def _candle_value(candle: Any, field_name: str) -> Optional[float]:
    """Read one numeric candle field from a Candle or dictionary object."""
    if isinstance(candle, Mapping):
        return _finite_float(candle.get(field_name))
    return _finite_float(getattr(candle, field_name, None))


def _candle_time(candle: Any) -> Optional[int]:
    """Read a candle opening timestamp from a Candle or dictionary object."""
    if isinstance(candle, Mapping):
        value = candle.get("time")
    else:
        value = getattr(candle, "time", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _ordered_timeframes(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
) -> List[str]:
    """Return loaded timeframe keys in the configured stable order."""
    present = {str(key).strip().upper() for key in candles_by_timeframe}
    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in present]


def _normalised_direction(direction: Any) -> str:
    """Convert BUY/SELL aliases into LONG/SHORT labels."""
    value = str(direction or "").strip().upper()
    if value in {"BUY", "LONG"}:
        return "LONG"
    if value in {"SELL", "SHORT"}:
        return "SHORT"
    return value


# ============================================================================
# CANDLE FEATURE EXTRACTION
# ============================================================================

def timeframe_features(
    candles: Sequence[Candle],
    timeframe: str,
) -> List[float]:
    """Convert a closed candle window into a fixed-size numerical vector."""
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {normalized_timeframe}")

    recent = list(candles)
    if not recent:
        return [0.0] * TIMEFRAME_FEATURE_SIZE

    current = recent[-1]
    closes = [float(candle.close) for candle in recent]
    ranges = [max(float(candle.high) - float(candle.low), 0.0) for candle in recent]

    true_ranges: List[float] = []
    for index, candle in enumerate(recent):
        previous = recent[index - 1] if index > 0 else None
        true_ranges.append(true_range(previous, candle))

    atr = max(
        rolling_mean(true_ranges[-14:]),
        abs(float(current.close)) * 1e-8,
        1e-12,
    )

    def normalized_return(lookback: int) -> float:
        if len(recent) <= lookback:
            return 0.0
        reference = float(recent[-1 - lookback].close)
        if reference == 0.0:
            return 0.0
        return (float(current.close) - reference) / abs(reference)

    body = abs(float(current.close) - float(current.open))
    candle_range = max(float(current.high) - float(current.low), 1e-12)
    upper_wick = max(
        float(current.high) - max(float(current.open), float(current.close)),
        0.0,
    )
    lower_wick = max(
        min(float(current.open), float(current.close)) - float(current.low),
        0.0,
    )

    recent_mean_range = max(rolling_mean(ranges[-14:]), 1e-12)
    recent_window = recent[-20:]
    highest = max(float(candle.high) for candle in recent_window)
    lowest = min(float(candle.low) for candle in recent_window)
    recent_span = max(highest - lowest, 1e-12)
    close_position = (float(current.close) - lowest) / recent_span

    returns = [
        (closes[index] - closes[index - 1]) / abs(closes[index - 1])
        for index in range(1, len(closes))
        if closes[index - 1] != 0.0
    ]
    volatility = rolling_std(returns[-14:]) if len(returns) >= 2 else 0.0

    volume_mean = max(
        rolling_mean([float(candle.volume) for candle in recent[-20:]]),
        1e-12,
    )
    volume_ratio = float(current.volume) / volume_mean

    price_scale = max(abs(float(current.close)), 1e-12)
    slope = rolling_slope([value / price_scale for value in closes[-20:]])
    compression = rolling_mean(ranges[-5:]) / max(rolling_mean(ranges[-20:]), 1e-12)

    previous_window = recent[-6:-1]
    breakout_up = (
        1.0
        if previous_window and float(current.close) >= max(float(c.high) for c in previous_window)
        else 0.0
    )
    breakout_down = (
        1.0
        if previous_window and float(current.close) <= min(float(c.low) for c in previous_window)
        else 0.0
    )

    features = [
        normalized_return(1),
        normalized_return(3),
        normalized_return(5),
        normalized_return(10),
        slope,
        volatility,
        candle_range / atr,
        body / candle_range,
        upper_wick / candle_range,
        lower_wick / candle_range,
        close_position,
        float(current.close) / max(abs(float(current.open)), 1e-12) - 1.0,
        recent_mean_range / atr,
        compression,
        volume_ratio,
        breakout_up,
        breakout_down,
        (highest - float(current.close)) / atr,
        (float(current.close) - lowest) / atr,
        len(recent) / max(LOOKBACK_BY_TIMEFRAME[normalized_timeframe], 1),
    ]

    bounded = [
        float(max(min(value, 100.0), -100.0))
        if math.isfinite(float(value))
        else 0.0
        for value in features[:TIMEFRAME_FEATURE_SIZE]
    ]
    if len(bounded) < TIMEFRAME_FEATURE_SIZE:
        bounded.extend([0.0] * (TIMEFRAME_FEATURE_SIZE - len(bounded)))
    return bounded


# ============================================================================
# MULTI-TIMEFRAME MARKET STATE
# ============================================================================

def market_state_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> np.ndarray:
    """Build a point-in-time market vector from the TradingBot's loaded frames.

    Only timeframe archives present in ``candles_by_timeframe`` are represented.
    This keeps the input dimensions tied to the timeframe union selected for
    that TradingBot, rather than loading or inventing unused chart intervals.
    """
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    vector: List[float] = []
    for timeframe in _ordered_timeframes(candles_by_timeframe):
        candles = candles_by_timeframe.get(timeframe, [])
        window = slice_closed_history(
            candles=candles,
            target_time=int(target_time),
            timeframe=timeframe,
            max_count=LOOKBACK_BY_TIMEFRAME[timeframe],
        )
        vector.extend(timeframe_features(window, timeframe))

    timestamp = datetime.fromtimestamp(int(target_time), tz=timezone.utc)
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_fraction = minute_of_day / (24.0 * 60.0)
    weekday_fraction = timestamp.weekday() / 6.0

    vector.extend(
        [
            math.sin(2.0 * math.pi * day_fraction),
            math.cos(2.0 * math.pi * day_fraction),
            math.sin(2.0 * math.pi * weekday_fraction),
            math.cos(2.0 * math.pi * weekday_fraction),
        ]
    )

    return np.asarray(vector, dtype=np.float32)


# ============================================================================
# PRICE AND VOLATILITY CONTEXT
# ============================================================================

def current_price_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> Optional[float]:
    """Return the newest known close using the finest loaded timeframe first."""
    available = set(_ordered_timeframes(candles_by_timeframe))
    priorities = [frame for frame in PRICE_TIMEFRAME_PRIORITY if frame in available]

    for timeframe in priorities:
        candles = candles_by_timeframe.get(timeframe, [])
        index = last_closed_index(candles, int(target_time), timeframe)
        if index >= 0:
            price = _finite_float(candles[index].close)
            if price is not None and price > 0.0:
                return price
    return None


def atr_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> float:
    """Return an ATR estimate using the best available loaded timeframe."""
    available = set(_ordered_timeframes(candles_by_timeframe))

    for timeframe in ATR_TIMEFRAME_PRIORITY:
        if timeframe not in available:
            continue
        candles = candles_by_timeframe.get(timeframe, [])
        window = slice_closed_history(
            candles=candles,
            target_time=int(target_time),
            timeframe=timeframe,
            max_count=30,
        )
        if len(window) < 3:
            continue

        values = [
            true_range(window[index - 1] if index > 0 else None, candle)
            for index, candle in enumerate(window)
        ]
        estimate = rolling_mean(values[-14:])
        if estimate > 0.0 and math.isfinite(estimate):
            return float(estimate)

    price = current_price_at(candles_by_timeframe, target_time)
    return max(abs(price or 1.0) * 1e-4, 1e-8)


# ============================================================================
# LEARNING SETUP TARGETS
# ============================================================================

def _setup_timeframe_index(setup: Optional[UserSetup], field_name: str) -> int:
    """Resolve a setup timeframe index, using stable defaults for WAIT samples."""
    if setup is None:
        default = "H1" if field_name == "setup_timeframe" else "M15"
        return TIMEFRAME_TO_INDEX.get(default, 0)

    value = str(getattr(setup, field_name, "")).strip().upper()
    if value not in TIMEFRAME_TO_INDEX:
        raise ValueError(
            f"Setup {setup.setup_id} has unsupported {field_name}: {value!r}."
        )
    return TIMEFRAME_TO_INDEX[value]


def setup_target_vector(
    setup: Optional[UserSetup],
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build direction, quality, five price targets and timeframe labels.

    Price targets are normalized by ATR and use this fixed order:
    anchor, breakout, entry, stop loss and take profit.
    """
    target_prices = np.zeros(SETUP_PRICE_TARGET_SIZE, dtype=np.float32)
    target_masks = np.zeros(SETUP_PRICE_TARGET_SIZE, dtype=np.float32)

    if setup is None:
        return (
            DIRECTION_TO_INDEX["WAIT"],
            np.asarray([0.0], dtype=np.float32),
            target_prices,
            target_masks,
            _setup_timeframe_index(None, "setup_timeframe"),
            _setup_timeframe_index(None, "entry_timeframe"),
        )

    direction = _normalised_direction(setup.direction)
    if direction not in DIRECTION_TO_INDEX or direction == "WAIT":
        raise ValueError(
            f"Setup {setup.setup_id} has unsupported direction: {setup.direction!r}."
        )
    direction_index = DIRECTION_TO_INDEX[direction]

    quality = np.asarray([1.0], dtype=np.float32)
    current_price = current_price_at(candles_by_timeframe, int(target_time))
    if current_price is None:
        current_price = (
            _finite_float(setup.entry)
            or _finite_float(setup.anchor_price)
            or 1.0
        )

    atr = max(atr_at(candles_by_timeframe, int(target_time)), 1e-12)
    raw_prices = (
        _finite_float(setup.anchor_price),
        _finite_float(setup.breakout),
        _finite_float(setup.entry),
        _finite_float(setup.sl),
        _finite_float(setup.tp),
    )

    for index, price in enumerate(raw_prices):
        if price is None:
            continue
        target_prices[index] = np.float32((price - current_price) / atr)
        target_masks[index] = 1.0

    return (
        direction_index,
        quality,
        target_prices,
        target_masks,
        _setup_timeframe_index(setup, "setup_timeframe"),
        _setup_timeframe_index(setup, "entry_timeframe"),
    )


# ============================================================================
# AUTOMATIC SUPPORT AND RESISTANCE
# ============================================================================

def _level_candidates(
    candles: Sequence[Candle],
) -> List[Dict[str, Any]]:
    """Calculate major automatic levels from one already-truncated history."""
    if len(candles) < 30:
        return []

    valid: List[Tuple[int, float]] = []
    for candle in candles:
        candle_time = _candle_time(candle)
        close = _candle_value(candle, "close")
        if candle_time is None or close is None:
            continue
        valid.append((candle_time, close))

    if len(valid) < 30:
        return []

    times = [item[0] for item in valid]
    prices = [item[1] for item in valid]

    lowest = min(prices)
    highest = max(prices)
    price_range = highest - lowest
    if not math.isfinite(price_range) or price_range <= 0.0:
        return []

    lowest_index = prices.index(lowest)
    highest_index = prices.index(highest)

    candidates: List[Dict[str, Any]] = [
        {
            "price": lowest,
            "touches": 0,
            "strength": 100.0,
            "candidate_index": lowest_index,
            "candidate_kind": "RANGE_LOW",
        },
        {
            "price": highest,
            "touches": 0,
            "strength": 100.0,
            "candidate_index": highest_index,
            "candidate_kind": "RANGE_HIGH",
        },
    ]

    swing_window = max(5, min(15, len(prices) // 30))
    for index in range(swing_window, len(prices) - swing_window):
        price = prices[index]
        left = prices[index - swing_window:index]
        right = prices[index + 1:index + swing_window + 1]

        if price >= max(left) and price >= max(right):
            candidates.append(
                {
                    "price": price,
                    "touches": 0,
                    "strength": 0.0,
                    "candidate_index": index,
                    "candidate_kind": "SWING_HIGH",
                }
            )
        elif price <= min(left) and price <= min(right):
            candidates.append(
                {
                    "price": price,
                    "touches": 0,
                    "strength": 0.0,
                    "candidate_index": index,
                    "candidate_kind": "SWING_LOW",
                }
            )

    touch_tolerance = price_range * 0.015
    for candidate in candidates:
        level_price = float(candidate["price"])
        touches = 0
        reactions = 0
        last_touch_index = -999
        touch_indexes: List[int] = []

        for index, price in enumerate(prices):
            if abs(price - level_price) <= touch_tolerance:
                if index - last_touch_index >= 8:
                    touches += 1
                    last_touch_index = index
                    touch_indexes.append(index)

                    # Reaction candles are restricted to this historical input.
                    future = prices[index + 1:min(len(prices), index + 12)]
                    if future:
                        movement = max(abs(value - level_price) for value in future)
                        if movement >= price_range * 0.04:
                            reactions += 1

        candidate["touches"] = touches
        candidate["reactions"] = reactions
        candidate["touch_indexes"] = touch_indexes
        candidate["strength"] = touches * 2.0 + reactions * 5.0
        candidate["candidate_time"] = times[int(candidate["candidate_index"])]
        candidate["first_touch_time"] = (
            times[touch_indexes[0]] if touch_indexes else None
        )
        candidate["last_touch_time"] = (
            times[touch_indexes[-1]] if touch_indexes else None
        )

    candidates = [
        candidate
        for candidate in candidates
        if (
            float(candidate["price"]) == lowest
            or float(candidate["price"]) == highest
            or int(candidate["touches"]) >= 2
        )
    ]
    candidates.sort(
        key=lambda candidate: float(candidate["strength"]),
        reverse=True,
    )

    low_candidate = next(
        candidate for candidate in candidates
        if float(candidate["price"]) == lowest
    )
    high_candidate = next(
        candidate for candidate in candidates
        if float(candidate["price"]) == highest
    )

    minimum_spacing = price_range * 0.08
    selected: List[Dict[str, Any]] = [
        {
            **low_candidate,
            "strength": 100.0,
        },
        {
            **high_candidate,
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

        if any(
            abs(price - float(existing["price"])) < minimum_spacing
            for existing in selected
        ):
            continue
        selected.append(candidate)

    selected.sort(key=lambda candidate: float(candidate["price"]))

    levels: List[Dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        price = float(candidate["price"])
        levels.append(
            {
                "level_id": index,
                "origin": "AUTO",
                "low": price,
                "high": price,
                "center": price,
                "touches": int(candidate["touches"]),
                "reactions": int(candidate.get("reactions", 0)),
                "breakouts": 0,
                "reaction_strength": float(candidate["strength"]),
                "strength": float(candidate["strength"]),
                "candidate_kind": str(candidate["candidate_kind"]),
                "candidate_time": int(candidate["candidate_time"]),
                "first_touch_time": candidate["first_touch_time"],
                "last_touch_time": candidate["last_touch_time"],
            }
        )
    return levels


def find_ai_major_levels(
    candles: Sequence[Candle],
    *,
    timeframe: str,
    calculated_at: int,
    lookback_candles: int = AI_LEVEL_LOOKBACK_CANDLES,
) -> List[Dict[str, Any]]:
    """Find every selected automatic level in a fixed, point-in-time window.

    Unlike the chart implementation, this function does not accept a viewport
    range. It receives only candles fully closed by ``calculated_at`` and uses
    the latest configured number of those candles.
    """
    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported AI level timeframe: {normalized_timeframe}")
    if int(lookback_candles) < 30:
        raise ValueError("lookback_candles must be at least 30.")

    historical_window = slice_closed_history(
        candles=candles,
        target_time=int(calculated_at),
        timeframe=normalized_timeframe,
        max_count=int(lookback_candles),
    )
    levels = _level_candidates(historical_window)

    times = [_candle_time(candle) for candle in historical_window]
    valid_times = [value for value in times if value is not None]
    window_start = min(valid_times) if valid_times else None
    window_end = max(valid_times) if valid_times else None

    for level in levels:
        level["level_id"] = f"{normalized_timeframe}:{level['level_id']}"
        level.update(
            {
                "timeframe": normalized_timeframe,
                "calculated_at": int(calculated_at),
                "history_candle_count": len(historical_window),
                "history_start_time": window_start,
                "history_end_time": window_end,
                "lookback_candles": int(lookback_candles),
            }
        )
    return levels


def _manual_zone_match(level_price: float, setup: UserSetup) -> List[Dict[str, Any]]:
    """Return manual zones whose price band contains an automatic level."""
    matches: List[Dict[str, Any]] = []
    for zone in setup.zones:
        if zone.low is None or zone.high is None:
            continue
        lower = min(float(zone.low), float(zone.high))
        upper = max(float(zone.low), float(zone.high))
        if lower <= level_price <= upper:
            matches.append(
                {
                    "zone_id": zone.zone_id,
                    "type": zone.zone_type,
                    "start_time": zone.start_time,
                    "end_time": zone.end_time,
                    "price_low": lower,
                    "price_high": upper,
                }
            )
    return matches


def build_auto_levels_for_setups(
    candles_by_timeframe: Dict[str, List[Candle]],
    setups: Sequence[UserSetup],
) -> Dict[int, List[Dict[str, Any]]]:
    """Build automatic levels for each Learning setup without viewport input.

    Levels are calculated separately for the setup and entry timeframe. Every
    timeframe uses at most 240 fully closed candles available before the setup's
    entry timestamp. All selected automatic levels are retained, including
    levels that do not overlap a user-drawn Support or Resistance zone.
    """
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    results: Dict[int, List[Dict[str, Any]]] = {}
    for setup in setups:
        setup_id = int(setup.setup_id)
        decision_time = setup.decision_time
        if decision_time is None:
            results[setup_id] = []
            continue

        levels_for_setup: List[Dict[str, Any]] = []
        timeframe_values = (setup.setup_timeframe, setup.entry_timeframe)
        selected_timeframes = list(dict.fromkeys(
            str(value).strip().upper()
            for value in timeframe_values
            if value
        ))

        for timeframe in selected_timeframes:
            if timeframe not in SUPPORTED_TIMEFRAMES:
                continue
            candles = candles_by_timeframe.get(timeframe, [])
            timeframe_levels = find_ai_major_levels(
                candles,
                timeframe=timeframe,
                calculated_at=int(decision_time),
                lookback_candles=AI_LEVEL_LOOKBACK_CANDLES,
            )

            for level in timeframe_levels:
                matches = _manual_zone_match(float(level["center"]), setup)
                level["setup_id"] = setup_id
                level["matches_manual_zone"] = bool(matches)
                level["matched_manual_zones"] = matches
                level["matched_manual_zone_ids"] = [
                    match["zone_id"] for match in matches
                ]
                levels_for_setup.append(level)

        results[setup_id] = levels_for_setup

    return results


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "AI_LEVEL_LOOKBACK_CANDLES",
    "TIMEFRAME_FEATURE_SIZE",
    "SETUP_PRICE_TARGET_FIELDS",
    "SETUP_PRICE_TARGET_SIZE",
    "sigmoid",
    "timeframe_features",
    "market_state_at",
    "current_price_at",
    "atr_at",
    "setup_target_vector",
    "find_ai_major_levels",
    "build_auto_levels_for_setups",
]
