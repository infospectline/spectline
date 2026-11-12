"""Numerical market features and market-structure analysis for Spectline AI."""

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

AI_LEVEL_LOOKBACK_CANDLES = 240
TIMEFRAME_FEATURE_SIZE = 20

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

SETUP_PRICE_TARGET_FIELDS = (
    "anchor_price",
    "breakout",
    "entry",
    "sl",
    "tp",
)
SETUP_PRICE_TARGET_SIZE = len(SETUP_PRICE_TARGET_FIELDS)

PIVOT_FEATURE_FIELDS = (
    "pivot_count",
    "high_pivot_count",
    "low_pivot_count",
    "duration_bars",
    "price_range_atr",
    "net_change_atr",
    "mean_leg_move_atr",
    "max_leg_move_atr",
    "mean_leg_duration_ratio",
    "leg_duration_std_ratio",
    "mean_normalized_slope",
    "slope_std",
    "higher_high_ratio",
    "lower_high_ratio",
    "higher_low_ratio",
    "lower_low_ratio",
    "direction_change_ratio",
    "time_symmetry",
    "price_symmetry",
    "pivot_density",
)
PATTERN_GEOMETRY_FEATURE_SIZE = len(PIVOT_FEATURE_FIELDS)


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


def _finite_int(value: Any) -> Optional[int]:
    """Convert a value to an integer or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _bounded(value: Any, lower: float = -100.0, upper: float = 100.0) -> float:
    """Return a finite float clipped to a stable numerical range."""
    number = _finite_float(value)
    if number is None:
        return 0.0
    return float(max(min(number, upper), lower))


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
        return _finite_int(candle.get("time"))
    return _finite_int(getattr(candle, "time", None))


def _candle_record(candle: Any) -> Optional[Dict[str, float | int]]:
    """Normalize one candle-like object into a finite OHLCV mapping."""
    timestamp = _candle_time(candle)
    open_price = _candle_value(candle, "open")
    high_price = _candle_value(candle, "high")
    low_price = _candle_value(candle, "low")
    close_price = _candle_value(candle, "close")
    volume = _candle_value(candle, "volume")

    if timestamp is None or any(
        value is None
        for value in (open_price, high_price, low_price, close_price)
    ):
        return None

    open_value = float(open_price)
    high_value = float(high_price)
    low_value = float(low_price)
    close_value = float(close_price)

    if high_value < low_value:
        return None
    if high_value < max(open_value, close_value):
        return None
    if low_value > min(open_value, close_value):
        return None

    return {
        "time": int(timestamp),
        "open": open_value,
        "high": high_value,
        "low": low_value,
        "close": close_value,
        "volume": max(float(volume or 0.0), 0.0),
    }


def _ordered_candles(candles: Sequence[Any]) -> List[Dict[str, float | int]]:
    """Return valid candles sorted by time with the last duplicate retained."""
    by_time: Dict[int, Dict[str, float | int]] = {}
    for candle in candles:
        record = _candle_record(candle)
        if record is not None:
            by_time[int(record["time"])] = record
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _ordered_timeframes(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
) -> List[str]:
    """Return loaded timeframe keys in the configured stable order."""
    present = {str(key).strip().upper() for key in candles_by_timeframe}
    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in present]


def _normalized_frame_map(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
) -> Dict[str, Sequence[Candle]]:
    """Return a timeframe mapping with normalized uppercase keys."""
    return {
        str(key).strip().upper(): value
        for key, value in candles_by_timeframe.items()
    }


def _normalised_direction(direction: Any) -> str:
    """Convert BUY/SELL aliases into LONG/SHORT labels."""
    value = str(direction or "").strip().upper()
    if value in {"BUY", "LONG"}:
        return "LONG"
    if value in {"SELL", "SHORT"}:
        return "SHORT"
    return value


def _validate_timeframe(timeframe: str) -> str:
    """Normalize and validate a timeframe for the current runtime mode."""
    normalized = str(timeframe).strip().upper()
    if normalized not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe: {normalized}")
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Timeframe is not available in the current runtime: {normalized}")
    return normalized


def _atr_series(candles: Sequence[Mapping[str, Any]], period: int = 14) -> List[float]:
    """Calculate a simple rolling ATR series from normalized candle mappings."""
    if not candles:
        return []
    true_ranges: List[float] = []
    previous_close: Optional[float] = None
    for candle in candles:
        high = float(candle["high"])
        low = float(candle["low"])
        if previous_close is None:
            current_range = max(high - low, 0.0)
        else:
            current_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
                0.0,
            )
        true_ranges.append(current_range)
        previous_close = float(candle["close"])

    result: List[float] = []
    for index in range(len(true_ranges)):
        window = true_ranges[max(0, index - period + 1):index + 1]
        result.append(max(rolling_mean(window), 1e-12))
    return result


# ============================================================================
# CANDLE FEATURE EXTRACTION
# ============================================================================

def timeframe_features(
    candles: Sequence[Candle],
    timeframe: str,
) -> List[float]:
    """Convert a closed candle window into the stable 20-value feature vector.

    The feature order is intentionally unchanged so existing model input
    dimensions and scalers remain compatible. The caller must pass only candles
    that were fully closed at the decision timestamp.
    """
    normalized_timeframe = _validate_timeframe(timeframe)
    recent = _ordered_candles(candles)
    if not recent:
        return [0.0] * TIMEFRAME_FEATURE_SIZE

    current = recent[-1]
    closes = [float(candle["close"]) for candle in recent]
    ranges = [max(float(candle["high"]) - float(candle["low"]), 0.0) for candle in recent]
    true_ranges: List[float] = []
    previous_close: Optional[float] = None

    for candle in recent:
        high = float(candle["high"])
        low = float(candle["low"])
        if previous_close is None:
            true_ranges.append(max(high - low, 0.0))
        else:
            true_ranges.append(
                max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                    0.0,
                )
            )
        previous_close = float(candle["close"])

    atr = max(
        rolling_mean(true_ranges[-14:]),
        abs(float(current["close"])) * 1e-8,
        1e-12,
    )

    def normalized_return(lookback: int) -> float:
        if len(recent) <= lookback:
            return 0.0
        reference = float(recent[-1 - lookback]["close"])
        if abs(reference) <= 1e-12:
            return 0.0
        return (float(current["close"]) - reference) / abs(reference)

    body = abs(float(current["close"]) - float(current["open"]))
    candle_range = max(float(current["high"]) - float(current["low"]), 1e-12)
    upper_wick = max(
        float(current["high"]) - max(float(current["open"]), float(current["close"])),
        0.0,
    )
    lower_wick = max(
        min(float(current["open"]), float(current["close"])) - float(current["low"]),
        0.0,
    )

    recent_mean_range = max(rolling_mean(ranges[-14:]), 1e-12)
    recent_window = recent[-20:]
    highest = max(float(candle["high"]) for candle in recent_window)
    lowest = min(float(candle["low"]) for candle in recent_window)
    recent_span = max(highest - lowest, 1e-12)
    close_position = (float(current["close"]) - lowest) / recent_span

    returns = [
        (closes[index] - closes[index - 1]) / abs(closes[index - 1])
        for index in range(1, len(closes))
        if abs(closes[index - 1]) > 1e-12
    ]
    volatility = rolling_std(returns[-14:]) if len(returns) >= 2 else 0.0

    volume_mean = max(
        rolling_mean([float(candle["volume"]) for candle in recent[-20:]]),
        1e-12,
    )
    volume_ratio = float(current["volume"]) / volume_mean

    price_scale = max(abs(float(current["close"])), 1e-12)
    slope = rolling_slope([value / price_scale for value in closes[-20:]])
    compression = rolling_mean(ranges[-5:]) / max(rolling_mean(ranges[-20:]), 1e-12)

    previous_window = recent[-6:-1]
    breakout_up = (
        1.0
        if previous_window
        and float(current["close"]) >= max(float(candle["high"]) for candle in previous_window)
        else 0.0
    )
    breakout_down = (
        1.0
        if previous_window
        and float(current["close"]) <= min(float(candle["low"]) for candle in previous_window)
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
        float(current["close"]) / max(abs(float(current["open"])), 1e-12) - 1.0,
        recent_mean_range / atr,
        compression,
        volume_ratio,
        breakout_up,
        breakout_down,
        (highest - float(current["close"])) / atr,
        (float(current["close"]) - lowest) / atr,
        len(recent) / max(LOOKBACK_BY_TIMEFRAME[normalized_timeframe], 1),
    ]

    return [_bounded(value) for value in features[:TIMEFRAME_FEATURE_SIZE]]


# ============================================================================
# MULTI-TIMEFRAME MARKET STATE
# ============================================================================

def market_state_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> np.ndarray:
    """Build a point-in-time market vector from the supplied timeframe archives.

    Only fully closed candles are used. The vector contains one fixed-size
    candle feature block for every loaded supported timeframe, followed by
    four cyclical UTC time features.
    """
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    target_timestamp = int(target_time)
    normalized_frames = _normalized_frame_map(candles_by_timeframe)
    vector: List[float] = []

    for timeframe in _ordered_timeframes(normalized_frames):
        candles = normalized_frames.get(timeframe, [])
        window = slice_closed_history(
            candles=candles,
            target_time=target_timestamp,
            timeframe=timeframe,
            max_count=LOOKBACK_BY_TIMEFRAME[timeframe],
        )
        vector.extend(timeframe_features(window, timeframe))

    timestamp = datetime.fromtimestamp(target_timestamp, tz=timezone.utc)
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_fraction = minute_of_day / (24.0 * 60.0)
    weekday_fraction = timestamp.weekday() / 7.0

    vector.extend(
        [
            math.sin(2.0 * math.pi * day_fraction),
            math.cos(2.0 * math.pi * day_fraction),
            math.sin(2.0 * math.pi * weekday_fraction),
            math.cos(2.0 * math.pi * weekday_fraction),
        ]
    )

    return np.asarray(vector, dtype=np.float32)


def market_state_metadata_at(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
    target_time: int,
) -> Dict[str, Any]:
    """Return the timeframe availability metadata used by market_state_at."""
    target_timestamp = int(target_time)
    normalized_frames = _normalized_frame_map(candles_by_timeframe)
    metadata: Dict[str, Any] = {
        "target_time": target_timestamp,
        "timeframe_order": [],
        "timeframes": {},
    }

    for timeframe in _ordered_timeframes(normalized_frames):
        candles = normalized_frames.get(timeframe, [])
        index = last_closed_index(candles, target_timestamp, timeframe)
        latest_time = _candle_time(candles[index]) if index >= 0 else None
        count = min(index + 1, LOOKBACK_BY_TIMEFRAME[timeframe]) if index >= 0 else 0
        metadata["timeframe_order"].append(timeframe)
        metadata["timeframes"][timeframe] = {
            "latest_closed_candle_time": latest_time,
            "closed_candle_count": max(count, 0),
            "configured_lookback": LOOKBACK_BY_TIMEFRAME[timeframe],
            "available": index >= 0,
        }

    return metadata


# ============================================================================
# PRICE AND VOLATILITY CONTEXT
# ============================================================================

def current_price_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> Optional[float]:
    """Return the newest known close using the finest loaded timeframe first."""
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    normalized_frames = _normalized_frame_map(candles_by_timeframe)
    available = set(_ordered_timeframes(normalized_frames))
    priorities = [frame for frame in PRICE_TIMEFRAME_PRIORITY if frame in available]

    for timeframe in priorities:
        candles = normalized_frames.get(timeframe, [])
        index = last_closed_index(candles, int(target_time), timeframe)
        if index >= 0:
            price = _candle_value(candles[index], "close")
            if price is not None and price > 0.0:
                return float(price)
    return None


def atr_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> float:
    """Return an ATR estimate using the best available loaded timeframe."""
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    normalized_frames = _normalized_frame_map(candles_by_timeframe)
    available = set(_ordered_timeframes(normalized_frames))
    for timeframe in ATR_TIMEFRAME_PRIORITY:
        if timeframe not in available:
            continue
        candles = normalized_frames.get(timeframe, [])
        window = slice_closed_history(
            candles=candles,
            target_time=int(target_time),
            timeframe=timeframe,
            max_count=30,
        )
        if len(window) < 3:
            continue

        values: List[float] = []
        previous: Optional[Candle] = None
        for candle in window:
            values.append(true_range(previous, candle))
            previous = candle
        estimate = rolling_mean(values[-14:])
        if estimate > 0.0 and math.isfinite(estimate):
            return float(estimate)

    price = current_price_at(normalized_frames, target_time)
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
    """Build direction, quality, price targets and timeframe labels.

    Price targets are normalized by ATR and use this fixed order: anchor,
    breakout, entry, stop loss and take profit. Actual outcome labels do not
    belong in this feature vector.
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
# PIVOT AND SWING DETECTION
# ============================================================================

def detect_pivots(
    candles: Sequence[Candle],
    *,
    timeframe: str,
    left_bars: int = 3,
    right_bars: int = 3,
    calculated_at: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Detect confirmed local high and low pivots with explicit recognition time.

    The pivot timestamp identifies the candle containing the extreme. The
    confirmation timestamp identifies when the required right-hand candles
    have closed. A pivot is never treated as known at its original timestamp.
    When calculated_at is supplied, only candles fully closed by that time are
    considered.
    """
    normalized_timeframe = _validate_timeframe(timeframe)
    left = int(left_bars)
    right = int(right_bars)
    if left < 1 or right < 1:
        raise ValueError("left_bars and right_bars must be at least 1.")

    normalized = _ordered_candles(candles)
    if calculated_at is not None:
        normalized = [
            candle
            for candle in normalized
            if int(candle["time"]) + TIMEFRAME_SECONDS[normalized_timeframe]
            <= int(calculated_at)
        ]

    required = left + right + 1
    if len(normalized) < required:
        return []

    atr_values = _atr_series(normalized)
    duration = TIMEFRAME_SECONDS[normalized_timeframe]
    pivots: List[Dict[str, Any]] = []

    for index in range(left, len(normalized) - right):
        candle = normalized[index]
        left_window = normalized[index - left:index]
        right_window = normalized[index + 1:index + right + 1]
        left_highs = [float(item["high"]) for item in left_window]
        right_highs = [float(item["high"]) for item in right_window]
        left_lows = [float(item["low"]) for item in left_window]
        right_lows = [float(item["low"]) for item in right_window]

        high = float(candle["high"])
        low = float(candle["low"])
        is_high = high >= max(left_highs) and high > max(right_highs)
        is_low = low <= min(left_lows) and low < min(right_lows)

        confirmation_index = index + right
        confirmation_candle = normalized[confirmation_index]
        confirmed_at = int(confirmation_candle["time"]) + duration
        if calculated_at is not None and confirmed_at > int(calculated_at):
            continue

        atr = max(float(atr_values[index]), 1e-12)
        left_reference_high = max(left_highs)
        right_reference_high = max(right_highs)
        left_reference_low = min(left_lows)
        right_reference_low = min(right_lows)

        if is_high:
            prominence = max(high - max(left_reference_high, right_reference_high), 0.0)
            pivots.append(
                {
                    "pivot_type": "HIGH",
                    "price": high,
                    "pivot_time": int(candle["time"]),
                    "pivot_index": index,
                    "confirmation_time": confirmed_at,
                    "recognition_time": confirmed_at,
                    "confirmation_index": confirmation_index,
                    "timeframe": normalized_timeframe,
                    "left_bars": left,
                    "right_bars": right,
                    "atr_at_pivot": atr,
                    "prominence": prominence,
                    "prominence_atr": prominence / atr,
                    "confirmed": True,
                }
            )

        if is_low:
            prominence = max(min(left_reference_low, right_reference_low) - low, 0.0)
            pivots.append(
                {
                    "pivot_type": "LOW",
                    "price": low,
                    "pivot_time": int(candle["time"]),
                    "pivot_index": index,
                    "confirmation_time": confirmed_at,
                    "recognition_time": confirmed_at,
                    "confirmation_index": confirmation_index,
                    "timeframe": normalized_timeframe,
                    "left_bars": left,
                    "right_bars": right,
                    "atr_at_pivot": atr,
                    "prominence": prominence,
                    "prominence_atr": prominence / atr,
                    "confirmed": True,
                }
            )

    pivots.sort(key=lambda item: (item["pivot_time"], item["pivot_type"]))
    return pivots


def detect_swings(
    candles: Sequence[Candle],
    *,
    timeframe: str,
    left_bars: int = 3,
    right_bars: int = 3,
    min_move_atr: float = 0.0,
    calculated_at: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Create an alternating swing sequence from confirmed high/low pivots."""
    minimum_move = _finite_float(min_move_atr)
    if minimum_move is None or minimum_move < 0.0:
        raise ValueError("min_move_atr must be a finite non-negative number.")

    pivots = detect_pivots(
        candles,
        timeframe=timeframe,
        left_bars=left_bars,
        right_bars=right_bars,
        calculated_at=calculated_at,
    )
    swings: List[Dict[str, Any]] = []

    for pivot in pivots:
        if not swings:
            swings.append(dict(pivot))
            continue

        previous = swings[-1]
        if pivot["pivot_type"] == previous["pivot_type"]:
            is_more_extreme = (
                float(pivot["price"]) > float(previous["price"])
                if pivot["pivot_type"] == "HIGH"
                else float(pivot["price"]) < float(previous["price"])
            )
            if is_more_extreme:
                swings[-1] = dict(pivot)
            continue

        reference_atr = max(
            float(pivot.get("atr_at_pivot", 0.0)),
            float(previous.get("atr_at_pivot", 0.0)),
            1e-12,
        )
        move_atr = abs(float(pivot["price"]) - float(previous["price"])) / reference_atr
        if move_atr < minimum_move:
            continue

        swing = dict(pivot)
        swing["move_from_previous"] = float(pivot["price"]) - float(previous["price"])
        swing["move_from_previous_atr"] = move_atr
        swings.append(swing)

    return swings


def build_zigzag(
    candles: Sequence[Candle],
    *,
    timeframe: str,
    reversal_atr: float = 1.0,
    calculated_at: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a confirmed ATR-based ZigZag sequence as an analytical tool.

    ZigZag points are not pattern IDs. Each returned point includes the candle
    timestamp of the extreme and the later timestamp when reversal confirmed it.
    The final unconfirmed extreme is deliberately omitted.
    """
    normalized_timeframe = _validate_timeframe(timeframe)
    threshold = _finite_float(reversal_atr)
    if threshold is None or threshold <= 0.0:
        raise ValueError("reversal_atr must be a positive finite number.")

    normalized = _ordered_candles(candles)
    duration = TIMEFRAME_SECONDS[normalized_timeframe]
    if calculated_at is not None:
        normalized = [
            candle
            for candle in normalized
            if int(candle["time"]) + duration <= int(calculated_at)
        ]
    if len(normalized) < 3:
        return []

    atr_values = _atr_series(normalized)
    direction = 0
    initial_price = float(normalized[0]["close"])
    extreme_index = 0
    extreme_price = initial_price
    output: List[Dict[str, Any]] = []

    for index in range(1, len(normalized)):
        candle = normalized[index]
        high = float(candle["high"])
        low = float(candle["low"])
        atr = max(float(atr_values[index]), 1e-12)

        if direction == 0:
            upward_move = (high - initial_price) / atr
            downward_move = (initial_price - low) / atr
            if max(upward_move, downward_move) < threshold:
                continue

            if upward_move >= downward_move:
                direction = 1
                initial_pivot = _zigzag_point(
                    normalized, 0, "LOW", index, duration, atr_values
                )
                initial_pivot["analytical_tool"] = "ZIGZAG"
                output.append(initial_pivot)
                extreme_index = index
                extreme_price = high
            else:
                direction = -1
                initial_pivot = _zigzag_point(
                    normalized, 0, "HIGH", index, duration, atr_values
                )
                initial_pivot["analytical_tool"] = "ZIGZAG"
                output.append(initial_pivot)
                extreme_index = index
                extreme_price = low
            continue

        if direction > 0:
            if high >= extreme_price:
                extreme_price = high
                extreme_index = index
                continue

            if (extreme_price - low) / atr >= threshold:
                output.append(
                    _zigzag_point(
                        normalized,
                        extreme_index,
                        "HIGH",
                        index,
                        duration,
                        atr_values,
                    )
                )
                direction = -1
                extreme_index = index
                extreme_price = low
            continue

        if low <= extreme_price:
            extreme_price = low
            extreme_index = index
            continue

        if (high - extreme_price) / atr >= threshold:
            output.append(
                _zigzag_point(
                    normalized,
                    extreme_index,
                    "LOW",
                    index,
                    duration,
                    atr_values,
                )
            )
            direction = 1
            extreme_index = index
            extreme_price = high

    for point in output:
        point["timeframe"] = normalized_timeframe
        point["reversal_threshold_atr"] = float(threshold)
    return output


def _zigzag_point(
    candles: Sequence[Mapping[str, Any]],
    pivot_index: int,
    pivot_type: str,
    confirmation_index: int,
    timeframe_seconds: int,
    atr_values: Sequence[float],
) -> Dict[str, Any]:
    """Create one explicit ZigZag pivot record."""
    pivot = candles[pivot_index]
    confirmation = candles[confirmation_index]
    price = float(pivot["high"] if pivot_type == "HIGH" else pivot["low"])
    return {
        "pivot_type": pivot_type,
        "price": price,
        "pivot_time": int(pivot["time"]),
        "pivot_index": int(pivot_index),
        "confirmation_time": int(confirmation["time"]) + timeframe_seconds,
        "recognition_time": int(confirmation["time"]) + timeframe_seconds,
        "confirmation_index": int(confirmation_index),
        "timeframe_seconds": int(timeframe_seconds),
        "atr_at_pivot": float(atr_values[pivot_index]),
        "confirmed": True,
        "analytical_tool": "ZIGZAG",
    }


# ============================================================================
# GEOMETRY AND NUMERICAL PATTERN FEATURES
# ============================================================================

def calculate_pivot_geometry(
    pivots: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
    atr: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate scale-normalized geometry from an ordered pivot sequence."""
    normalized_timeframe = _validate_timeframe(timeframe)
    valid: List[Dict[str, Any]] = []

    for item in pivots:
        pivot_type = str(item.get("pivot_type", "")).strip().upper()
        price = _finite_float(item.get("price"))
        pivot_time = _finite_int(item.get("pivot_time"))
        recognition_time = _finite_int(
            item.get("recognition_time", item.get("confirmation_time"))
        )
        if pivot_type not in {"HIGH", "LOW"} or price is None or pivot_time is None:
            continue
        valid.append(
            {
                "pivot_type": pivot_type,
                "price": price,
                "pivot_time": pivot_time,
                "recognition_time": recognition_time,
                "atr_at_pivot": _finite_float(item.get("atr_at_pivot")),
            }
        )

    valid.sort(key=lambda item: (item["pivot_time"], item["pivot_type"]))
    if not valid:
        return {
            "timeframe": normalized_timeframe,
            "pivot_count": 0,
            "pivot_types": [],
            "pivot_times": [],
            "pivot_prices": [],
            "features": {field: 0.0 for field in PIVOT_FEATURE_FIELDS},
        }

    prices = [float(item["price"]) for item in valid]
    times = [int(item["pivot_time"]) for item in valid]
    pivot_atr_values = [item.get("atr_at_pivot") for item in valid]
    valid_atr = [value for value in pivot_atr_values if value is not None and value > 0]
    resolved_atr = _finite_float(atr) or (rolling_mean(valid_atr) if valid_atr else 0.0)
    scale = max(float(resolved_atr or 0.0), 1e-12)

    price_range = max(prices) - min(prices)
    duration_seconds = max(times) - min(times)
    duration_bars = duration_seconds / TIMEFRAME_SECONDS[normalized_timeframe]
    legs: List[Dict[str, float]] = []
    for index in range(1, len(valid)):
        time_delta = max(times[index] - times[index - 1], 0)
        price_delta = prices[index] - prices[index - 1]
        move_atr = abs(price_delta) / scale
        slope = price_delta / max(time_delta, 1)
        legs.append(
            {
                "duration_seconds": float(time_delta),
                "move": float(price_delta),
                "move_atr": float(move_atr),
                "slope": float(slope),
            }
        )

    leg_durations = [item["duration_seconds"] for item in legs]
    leg_moves = [item["move_atr"] for item in legs]
    normalized_slopes = [
        item["slope"] * TIMEFRAME_SECONDS[normalized_timeframe] / scale
        for item in legs
    ]

    highs = [float(item["price"]) for item in valid if item["pivot_type"] == "HIGH"]
    lows = [float(item["price"]) for item in valid if item["pivot_type"] == "LOW"]

    def comparison_ratios(values: Sequence[float], higher: bool) -> Tuple[float, float]:
        if len(values) < 2:
            return 0.0, 0.0
        comparisons = [
            values[index] > values[index - 1] if higher
            else values[index] < values[index - 1]
            for index in range(1, len(values))
        ]
        ratio = sum(1 for value in comparisons if value) / len(comparisons)
        return float(ratio), float(1.0 - ratio)

    higher_high_ratio, lower_high_ratio = comparison_ratios(highs, True)
    higher_low_ratio, lower_low_ratio = comparison_ratios(lows, True)

    directions = [
        1 if legs[index]["move"] > 0 else -1 if legs[index]["move"] < 0 else 0
        for index in range(len(legs))
    ]
    direction_changes = sum(
        1
        for index in range(1, len(directions))
        if directions[index] != 0
        and directions[index - 1] != 0
        and directions[index] != directions[index - 1]
    )
    direction_change_ratio = (
        direction_changes / max(len(directions) - 1, 1)
        if len(directions) >= 2
        else 0.0
    )

    duration_mean = rolling_mean(leg_durations)
    duration_std = rolling_std(leg_durations)
    time_symmetry = 0.0
    price_symmetry = 0.0
    if len(leg_durations) >= 2:
        time_symmetry = min(leg_durations) / max(max(leg_durations), 1.0)
    if len(leg_moves) >= 2:
        price_symmetry = min(leg_moves) / max(max(leg_moves), 1e-12)

    features = {
        "pivot_count": float(len(valid)),
        "high_pivot_count": float(len(highs)),
        "low_pivot_count": float(len(lows)),
        "duration_bars": duration_bars,
        "price_range_atr": price_range / scale,
        "net_change_atr": (prices[-1] - prices[0]) / scale,
        "mean_leg_move_atr": rolling_mean(leg_moves),
        "max_leg_move_atr": max(leg_moves, default=0.0),
        "mean_leg_duration_ratio": duration_mean / max(duration_seconds, 1.0),
        "leg_duration_std_ratio": duration_std / max(duration_mean, 1.0),
        "mean_normalized_slope": rolling_mean(normalized_slopes),
        "slope_std": rolling_std(normalized_slopes),
        "higher_high_ratio": higher_high_ratio,
        "lower_high_ratio": lower_high_ratio,
        "higher_low_ratio": higher_low_ratio,
        "lower_low_ratio": lower_low_ratio,
        "direction_change_ratio": direction_change_ratio,
        "time_symmetry": time_symmetry,
        "price_symmetry": price_symmetry,
        "pivot_density": len(valid) / max(duration_bars, 1.0),
    }

    return {
        "timeframe": normalized_timeframe,
        "pivot_count": len(valid),
        "start_time": times[0],
        "end_time": times[-1],
        "duration_seconds": duration_seconds,
        "price_min": min(prices),
        "price_max": max(prices),
        "price_range": price_range,
        "net_price_change": prices[-1] - prices[0],
        "atr_reference": scale,
        "pivot_types": [item["pivot_type"] for item in valid],
        "pivot_times": times,
        "pivot_prices": prices,
        "recognition_times": [item["recognition_time"] for item in valid],
        "legs": legs,
        "features": features,
    }


def pattern_geometry(
    candles: Sequence[Candle],
    *,
    timeframe: str,
    calculated_at: Optional[int] = None,
    left_bars: int = 3,
    right_bars: int = 3,
) -> Dict[str, Any]:
    """Build a compact numerical geometry description for one market segment."""
    normalized_timeframe = _validate_timeframe(timeframe)
    normalized = _ordered_candles(candles)
    duration = TIMEFRAME_SECONDS[normalized_timeframe]

    if calculated_at is not None:
        normalized = [
            candle
            for candle in normalized
            if int(candle["time"]) + duration <= int(calculated_at)
        ]

    if not normalized:
        return {
            "timeframe": normalized_timeframe,
            "start_time": None,
            "end_time": None,
            "recognized_at": None,
            "candle_count": 0,
            "features": {},
            "pivots": [],
            "pivot_geometry": calculate_pivot_geometry([], timeframe=normalized_timeframe),
        }

    atr_values = _atr_series(normalized)
    atr = max(float(atr_values[-1]), 1e-12)
    first = normalized[0]
    last = normalized[-1]
    highs = [float(item["high"]) for item in normalized]
    lows = [float(item["low"]) for item in normalized]
    closes = [float(item["close"]) for item in normalized]
    opens = [float(item["open"]) for item in normalized]
    ranges = [highs[index] - lows[index] for index in range(len(normalized))]
    price_range = max(highs) - min(lows)
    duration_seconds = max(int(last["time"]) - int(first["time"]), 0)
    mean_range = rolling_mean(ranges)
    mean_close = max(abs(rolling_mean(closes)), 1e-12)
    slope = rolling_slope([value / mean_close for value in closes])

    pivots = detect_pivots(
        normalized,
        timeframe=normalized_timeframe,
        left_bars=left_bars,
        right_bars=right_bars,
        calculated_at=calculated_at,
    )
    pivot_geometry = calculate_pivot_geometry(pivots, timeframe=normalized_timeframe, atr=atr)

    body_sizes = [abs(closes[index] - opens[index]) for index in range(len(normalized))]
    upper_wicks = [max(highs[index] - max(opens[index], closes[index]), 0.0) for index in range(len(normalized))]
    lower_wicks = [max(min(opens[index], closes[index]) - lows[index], 0.0) for index in range(len(normalized))]
    first_price = float(first["open"])
    last_price = float(last["close"])
    candle_count = len(normalized)

    last_candle_close = int(last["time"]) + duration
    recognized_at = max(
        [last_candle_close]
        + [
            int(item["recognition_time"])
            for item in pivots
            if item.get("recognition_time") is not None
        ]
    )
    if calculated_at is not None and recognized_at > int(calculated_at):
        raise ValueError("Pattern geometry includes information beyond calculated_at.")

    features = {
        "candle_count": float(candle_count),
        "duration_seconds": float(duration_seconds),
        "duration_bars": float(candle_count),
        "price_range": price_range,
        "price_range_atr": price_range / atr,
        "net_change": last_price - first_price,
        "net_change_atr": (last_price - first_price) / atr,
        "return_fraction": (last_price - first_price) / max(abs(first_price), 1e-12),
        "normalized_slope": slope,
        "slope_angle_radians": math.atan(slope),
        "slope_angle_degrees": math.degrees(math.atan(slope)),
        "mean_candle_range_atr": mean_range / atr,
        "range_compression": rolling_mean(ranges[-max(1, candle_count // 3):]) / max(mean_range, 1e-12),
        "mean_body_ratio": rolling_mean(
            [body_sizes[index] / max(ranges[index], 1e-12) for index in range(candle_count)]
        ),
        "mean_upper_wick_ratio": rolling_mean(
            [upper_wicks[index] / max(ranges[index], 1e-12) for index in range(candle_count)]
        ),
        "mean_lower_wick_ratio": rolling_mean(
            [lower_wicks[index] / max(ranges[index], 1e-12) for index in range(candle_count)]
        ),
        "close_location_in_range": (last_price - min(lows)) / max(price_range, 1e-12),
        "volatility_atr": rolling_std(
            [
                (closes[index] - closes[index - 1]) / max(abs(closes[index - 1]), 1e-12)
                for index in range(1, len(closes))
            ]
        ),
        "pivot_count": float(len(pivots)),
    }

    return {
        "timeframe": normalized_timeframe,
        "start_time": int(first["time"]),
        "end_time": last_candle_close,
        "last_candle_open_time": int(last["time"]),
        "recognized_at": int(recognized_at),
        "candle_count": candle_count,
        "atr": atr,
        "features": {key: _bounded(value) for key, value in features.items()},
        "pivots": pivots,
        "pivot_geometry": pivot_geometry,
    }


def pattern_geometry_vector(geometry: Mapping[str, Any]) -> np.ndarray:
    """Convert pivot geometry fields into their stable fixed-order vector."""
    pivot_geometry = geometry.get("pivot_geometry", {})
    feature_map = (
        pivot_geometry.get("features", {})
        if isinstance(pivot_geometry, Mapping)
        else {}
    )
    if not isinstance(feature_map, Mapping):
        feature_map = {}
    return np.asarray(
        [_bounded(feature_map.get(field, 0.0)) for field in PIVOT_FEATURE_FIELDS],
        dtype=np.float32,
    )


def candle_geometry_vector(geometry: Mapping[str, Any]) -> np.ndarray:
    """Convert a pattern geometry record into a compact numeric vector."""
    feature_map = geometry.get("features", {})
    if not isinstance(feature_map, Mapping):
        feature_map = {}
    field_order = (
        "candle_count",
        "duration_seconds",
        "price_range_atr",
        "net_change_atr",
        "return_fraction",
        "normalized_slope",
        "slope_angle_radians",
        "mean_candle_range_atr",
        "range_compression",
        "mean_body_ratio",
        "mean_upper_wick_ratio",
        "mean_lower_wick_ratio",
        "close_location_in_range",
        "volatility_atr",
        "pivot_count",
    )
    return np.asarray(
        [_bounded(feature_map.get(field, 0.0)) for field in field_order],
        dtype=np.float32,
    )


# ============================================================================
# AUTOMATIC SUPPORT AND RESISTANCE
# ============================================================================

def _level_candidates(
    candles: Sequence[Candle],
    timeframe: str,
) -> List[Dict[str, Any]]:
    """Calculate major horizontal levels from one already-truncated history."""
    normalized = _ordered_candles(candles)
    if len(normalized) < 30:
        return []

    times = [int(item["time"]) for item in normalized]
    highs = [float(item["high"]) for item in normalized]
    lows = [float(item["low"]) for item in normalized]
    closes = [float(item["close"]) for item in normalized]

    lowest = min(lows)
    highest = max(highs)
    price_range = highest - lowest
    if not math.isfinite(price_range) or price_range <= 0.0:
        return []

    low_index = lows.index(lowest)
    high_index = highs.index(highest)
    swing_window = max(3, min(12, len(normalized) // 40))
    candidates: List[Dict[str, Any]] = [
        {
            "price": lowest,
            "touches": 0,
            "strength": 100.0,
            "candidate_index": low_index,
            "candidate_kind": "RANGE_LOW",
        },
        {
            "price": highest,
            "touches": 0,
            "strength": 100.0,
            "candidate_index": high_index,
            "candidate_kind": "RANGE_HIGH",
        },
    ]

    for index in range(swing_window, len(normalized) - swing_window):
        high = highs[index]
        low = lows[index]
        left_highs = highs[index - swing_window:index]
        right_highs = highs[index + 1:index + swing_window + 1]
        left_lows = lows[index - swing_window:index]
        right_lows = lows[index + 1:index + swing_window + 1]

        if high >= max(left_highs) and high > max(right_highs):
            candidates.append(
                {
                    "price": high,
                    "touches": 0,
                    "strength": 0.0,
                    "candidate_index": index,
                    "candidate_kind": "SWING_HIGH",
                }
            )
        if low <= min(left_lows) and low < min(right_lows):
            candidates.append(
                {
                    "price": low,
                    "touches": 0,
                    "strength": 0.0,
                    "candidate_index": index,
                    "candidate_kind": "SWING_LOW",
                }
            )

    touch_tolerance = max(price_range * 0.01, rolling_mean(_atr_series(normalized)) * 0.25)
    for candidate in candidates:
        level_price = float(candidate["price"])
        touches = 0
        reactions = 0
        last_touch_index = -999
        touch_indexes: List[int] = []

        for index, candle in enumerate(normalized):
            candle_low = float(candle["low"])
            candle_high = float(candle["high"])
            if candle_low - touch_tolerance <= level_price <= candle_high + touch_tolerance:
                if index - last_touch_index >= 5:
                    touches += 1
                    last_touch_index = index
                    touch_indexes.append(index)
                    later = closes[index + 1:min(len(closes), index + 9)]
                    if later:
                        movement = max(abs(value - level_price) for value in later)
                        if movement >= price_range * 0.025:
                            reactions += 1

        candidate["touches"] = touches
        candidate["reactions"] = reactions
        candidate["touch_indexes"] = touch_indexes
        candidate["strength"] = touches * 2.0 + reactions * 5.0
        candidate["candidate_time"] = times[int(candidate["candidate_index"])]
        candidate["first_touch_time"] = times[touch_indexes[0]] if touch_indexes else None
        candidate["last_touch_time"] = times[touch_indexes[-1]] if touch_indexes else None
        if str(candidate["candidate_kind"]).startswith("RANGE_"):
            confirmation_time = times[-1] + TIMEFRAME_SECONDS[timeframe]
        else:
            confirmation_index = min(
                len(times) - 1,
                int(candidate["candidate_index"]) + swing_window,
            )
            confirmation_time = (
                times[confirmation_index] + TIMEFRAME_SECONDS[timeframe]
            )
        candidate["recognition_time"] = confirmation_time

    candidates = [
        candidate
        for candidate in candidates
        if float(candidate["price"]) in {lowest, highest} or int(candidate["touches"]) >= 2
    ]
    candidates.sort(key=lambda item: float(item["strength"]), reverse=True)

    low_candidate = next(item for item in candidates if float(item["price"]) == lowest)
    high_candidate = next(item for item in candidates if float(item["price"]) == highest)

    minimum_spacing = price_range * 0.06
    selected: List[Dict[str, Any]] = [
        {**low_candidate, "strength": max(float(low_candidate["strength"]), 100.0)},
        {**high_candidate, "strength": max(float(high_candidate["strength"]), 100.0)},
    ]

    for candidate in candidates:
        price = float(candidate["price"])
        if any(
            abs(price - float(existing["price"])) < minimum_spacing
            for existing in selected
        ):
            continue
        selected.append(candidate)

    selected.sort(key=lambda item: float(item["price"]))
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
                "recognition_time": int(candidate["recognition_time"]),
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
    """Find selected automatic levels using only candles closed by calculated_at."""
    normalized_timeframe = _validate_timeframe(timeframe)
    lookback = int(lookback_candles)
    if lookback < 30:
        raise ValueError("lookback_candles must be at least 30.")

    historical_window = slice_closed_history(
        candles=candles,
        target_time=int(calculated_at),
        timeframe=normalized_timeframe,
        max_count=lookback,
    )
    levels = _level_candidates(historical_window, normalized_timeframe)
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
                "strength_calculated_at": int(calculated_at),
                "candidate_recognition_time": int(level["recognition_time"]),
                "history_candle_count": len(historical_window),
                "history_start_time": window_start,
                "history_end_time": window_end,
                "lookback_candles": lookback,
            }
        )
    return levels


def _manual_zone_match(level_price: float, setup: UserSetup) -> List[Dict[str, Any]]:
    """Return manual zones whose price band contains an automatic level."""
    matches: List[Dict[str, Any]] = []
    for zone in getattr(setup, "zones", []):
        low = _finite_float(getattr(zone, "low", None))
        high = _finite_float(getattr(zone, "high", None))
        if low is None or high is None:
            continue
        lower = min(low, high)
        upper = max(low, high)
        if lower <= level_price <= upper:
            matches.append(
                {
                    "zone_id": getattr(zone, "zone_id", None),
                    "type": getattr(zone, "zone_type", None),
                    "start_time": getattr(zone, "start_time", None),
                    "end_time": getattr(zone, "end_time", None),
                    "price_low": lower,
                    "price_high": upper,
                }
            )
    return matches


def build_auto_levels_for_setups(
    candles_by_timeframe: Dict[str, List[Candle]],
    setups: Sequence[UserSetup],
) -> Dict[int, List[Dict[str, Any]]]:
    """Build automatic levels for each setup from its point-in-time history."""
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    normalized_frames = _normalized_frame_map(candles_by_timeframe)
    results: Dict[int, List[Dict[str, Any]]] = {}
    for setup in setups:
        setup_id = int(setup.setup_id)
        decision_time = getattr(setup, "decision_time", None)
        if decision_time is None:
            results[setup_id] = []
            continue

        levels_for_setup: List[Dict[str, Any]] = []
        selected_timeframes = list(dict.fromkeys(
            str(value).strip().upper()
            for value in (
                getattr(setup, "setup_timeframe", None),
                getattr(setup, "entry_timeframe", None),
            )
            if value
        ))

        for timeframe in selected_timeframes:
            if timeframe not in SUPPORTED_TIMEFRAMES:
                continue
            candles = normalized_frames.get(timeframe, [])
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
    "PIVOT_FEATURE_FIELDS",
    "PATTERN_GEOMETRY_FEATURE_SIZE",
    "sigmoid",
    "timeframe_features",
    "market_state_at",
    "market_state_metadata_at",
    "current_price_at",
    "atr_at",
    "setup_target_vector",
    "detect_pivots",
    "detect_swings",
    "build_zigzag",
    "calculate_pivot_geometry",
    "pattern_geometry",
    "pattern_geometry_vector",
    "candle_geometry_vector",
    "find_ai_major_levels",
    "build_auto_levels_for_setups",
]
