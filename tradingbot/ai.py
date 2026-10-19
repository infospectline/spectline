"""
Micro AI training engine for the trading platform.

The module trains one numerical PyTorch model per market.

Data sources:
    TRADING SET UP/<MARKET>/setups.json
    CHARTS/<MARKET>/<TIMEFRAME>/candles.csv

Generated model:
    MODELS/<MARKET>/model.pth

The model is intentionally numerical. It does not use text embeddings or an LLM.
It learns from multi-timeframe price history and user-labelled trading setups.

Training concept:
    historical candles
        +
    user setups
        +
    hard negative / no-trade examples
        ↓
    numerical market state
        ↓
    multi-task neural network
        ↓
    market-specific model
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# PATHS AND GLOBAL CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

TRADING_SETUP_ROOT = BASE_DIR / "TRADING SET UP"
CHARTS_ROOT = BASE_DIR / "CHARTS"
MODELS_ROOT = BASE_DIR / "MODELS"

SETUP_FILENAME = "setups.json"
CANDLE_FILENAME = "candles.csv"
MODEL_FILENAME = "model.pth"

SUPPORTED_TIMEFRAMES = (
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)

TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}

# Number of recent closed candles represented for every timeframe.
LOOKBACK_BY_TIMEFRAME = {
    "M1": 32,
    "M5": 32,
    "M15": 32,
    "M30": 24,
    "H1": 24,
    "H4": 20,
    "D1": 16,
    "W1": 12,
}

SEED = 42

DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5

# Hard negatives are sampled around the user's real trade moments.
HARD_NEGATIVE_OFFSETS = (-12, -8, -5, -3, -2, -1, 1, 2, 3, 5, 8, 12)

# Additional random no-trade examples.
RANDOM_NEGATIVE_MULTIPLIER = 2

# Small amount of time around a labelled trade that is treated as protected.
# This avoids accidentally using another positive trade as a negative.
POSITIVE_EXCLUSION_SECONDS = 30 * 60


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    """Set deterministic random seeds where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# BASIC FILESYSTEM HELPERS
# ============================================================================

def safe_market_folder(market: str) -> str:
    """Return a filesystem-safe market folder."""
    return (
        str(market)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
    )


def setup_path(market: str) -> Path:
    """Return the Learning setup JSON path for one market."""
    return (
        TRADING_SETUP_ROOT
        / safe_market_folder(market)
        / SETUP_FILENAME
    )


def model_dir(market: str) -> Path:
    """Return the model directory for one market."""
    return MODELS_ROOT / safe_market_folder(market)


def model_path(market: str) -> Path:
    """Return the trained model path for one market."""
    return model_dir(market) / MODEL_FILENAME


def chart_path(market: str, timeframe: str) -> Path:
    """Return a local chart archive path."""
    return (
        CHARTS_ROOT
        / safe_market_folder(market)
        / timeframe
        / CANDLE_FILENAME
    )


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Candle:
    """Normalized OHLCV candle."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class UserSetup:
    """Normalized Learning setup used as one positive training example."""

    setup_id: int
    direction: str
    setup_timeframe: str
    entry_timeframe: str

    anchor_time: Optional[int]
    anchor_price: Optional[float]

    focus_1: Optional[float]
    focus_2: Optional[float]
    breakout: Optional[float]

    entry_candle_time: Optional[int]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]


@dataclass
class FeatureScaler:
    """Standardization statistics stored inside the model checkpoint."""

    mean: np.ndarray
    std: np.ndarray


# ============================================================================
# TIME PARSING
# ============================================================================

def parse_timestamp(value: Any) -> Optional[int]:
    """
    Convert a timestamp-like value into Unix seconds.

    Accepted inputs include:
        - integer / float Unix seconds
        - ISO-8601 strings
        - strings ending in UTC / Z
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None

        # Treat very large numeric timestamps as milliseconds.
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
    except ValueError:
        pass

    normalized = text.replace(" UTC", "+00:00").replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


# ============================================================================
# JSON LEARNING DATA
# ============================================================================

def load_learning_setups(market: str) -> List[UserSetup]:
    """Load and normalize all saved Learning trades for a market."""
    path = setup_path(market)

    if not path.is_file():
        raise FileNotFoundError(
            f"Learning setup file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(raw).__name__}."
        )

    setups: List[UserSetup] = []

    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue

        try:
            setup_id = int(item.get("id", index))
        except (TypeError, ValueError):
            setup_id = index

        direction = str(
            item.get("direction", "LONG")
        ).strip().upper()

        if direction not in {"LONG", "SHORT", "BUY", "SELL"}:
            direction = "LONG"

        if direction == "BUY":
            direction = "LONG"
        elif direction == "SELL":
            direction = "SHORT"

        setup_timeframe = str(
            item.get("setup_timeframe", "H1")
        ).upper()

        entry_timeframe = str(
            item.get("entry_timeframe", "M15")
        ).upper()

        if setup_timeframe not in SUPPORTED_TIMEFRAMES:
            setup_timeframe = "H1"

        if entry_timeframe not in SUPPORTED_TIMEFRAMES:
            entry_timeframe = "M15"

        anchor = item.get("anchor")
        if not isinstance(anchor, dict):
            anchor = {}

        anchor_time = parse_timestamp(anchor.get("time"))
        anchor_price = to_float(anchor.get("price"))

        entry_candle_time = parse_timestamp(
            item.get("entry_candle")
        )

        setup = UserSetup(
            setup_id=setup_id,
            direction=direction,
            setup_timeframe=setup_timeframe,
            entry_timeframe=entry_timeframe,
            anchor_time=anchor_time,
            anchor_price=anchor_price,
            focus_1=to_float(item.get("focus_1")),
            focus_2=to_float(item.get("focus_2")),
            breakout=to_float(item.get("breakout")),
            entry_candle_time=entry_candle_time,
            entry=to_float(item.get("entry")),
            sl=to_float(item.get("sl")),
            tp=to_float(item.get("tp")),
        )

        # The decision timestamp is the entry candle when available.
        # Anchor time is used as a fallback.
        decision_time = (
            setup.entry_candle_time
            if setup.entry_candle_time is not None
            else setup.anchor_time
        )

        if decision_time is None:
            continue

        setups.append(setup)

    setups.sort(
        key=lambda item: (
            item.entry_candle_time
            if item.entry_candle_time is not None
            else item.anchor_time
            if item.anchor_time is not None
            else 0
        )
    )

    return setups


# ============================================================================
# CSV CANDLE DATA
# ============================================================================

def to_float(value: Any) -> Optional[float]:
    """Convert a value to finite float or None."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def load_candles(market: str, timeframe: str) -> List[Candle]:
    """Load one local chart archive."""
    path = chart_path(market, timeframe)

    if not path.is_file():
        return []

    candles: List[Candle] = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                timestamp = int(float(row["time"]))
                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                volume = float(
                    row.get(
                        "tick_volume",
                        row.get("volume", 0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

            values = (
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
            )

            if not all(math.isfinite(float(value)) for value in values):
                continue

            candles.append(
                Candle(
                    time=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
            )

    candles.sort(key=lambda candle: candle.time)

    # Remove duplicate timestamps.
    unique: Dict[int, Candle] = {}
    for candle in candles:
        unique[candle.time] = candle

    return [unique[key] for key in sorted(unique)]


def load_all_market_candles(
    market: str,
) -> Dict[str, List[Candle]]:
    """Load M1 through W1 history for one market."""
    result: Dict[str, List[Candle]] = {}

    for timeframe in SUPPORTED_TIMEFRAMES:
        candles = load_candles(market, timeframe)
        if candles:
            result[timeframe] = candles

    return result


# ============================================================================
# CANDLE UTILITIES
# ============================================================================

def candle_return(previous: Candle, current: Candle) -> float:
    """Return price change normalized by previous close."""
    if previous.close == 0:
        return 0.0
    return (current.close - previous.close) / abs(previous.close)


def true_range(
    previous: Optional[Candle],
    current: Candle,
) -> float:
    """Calculate true range."""
    if previous is None:
        return max(
            current.high - current.low,
            0.0,
        )

    return max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )


def rolling_mean(values: Sequence[float]) -> float:
    """Return a safe mean."""
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rolling_std(values: Sequence[float]) -> float:
    """Return a safe standard deviation."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def rolling_slope(values: Sequence[float]) -> float:
    """Return a normalized linear slope."""
    if len(values) < 3:
        return 0.0

    y = np.asarray(values, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)

    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 0:
        return 0.0

    slope = float(
        np.sum((x - x.mean()) * (y - y.mean()))
        / denominator
    )

    scale = max(abs(float(y[-1])), 1e-12)
    return slope / scale


def sigmoid(value: float) -> float:
    """Stable scalar sigmoid."""
    value = max(min(value, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-value))


# ============================================================================
# TIMEFRAME ALIGNMENT
# ============================================================================

def last_closed_index(
    candles: Sequence[Candle],
    target_time: int,
    timeframe: str,
) -> int:
    """
    Return the latest candle that was fully closed before target_time.

    A candle with open time T and timeframe duration D is only available
    after T + D. This prevents future leakage during training/testing.
    """
    if not candles:
        return -1

    duration = TIMEFRAME_SECONDS[timeframe]
    cutoff = int(target_time) - int(duration)

    low = 0
    high = len(candles) - 1
    answer = -1

    while low <= high:
        middle = (low + high) // 2
        if candles[middle].time <= cutoff:
            answer = middle
            low = middle + 1
        else:
            high = middle - 1

    return answer


def slice_closed_history(
    candles: Sequence[Candle],
    target_time: int,
    timeframe: str,
    max_count: int,
) -> List[Candle]:
    """Return only candles that were fully closed at target_time."""
    index = last_closed_index(
        candles=candles,
        target_time=target_time,
        timeframe=timeframe,
    )

    if index < 0:
        return []

    start = max(0, index - max_count + 1)
    return list(candles[start:index + 1])


# ============================================================================
# NUMERICAL MARKET FEATURES
# ============================================================================

def timeframe_features(
    candles: Sequence[Candle],
    timeframe: str,
) -> List[float]:
    """
    Convert one timeframe history window into a fixed numerical vector.

    The representation intentionally contains relationships rather than raw
    absolute prices wherever possible.
    """
    if not candles:
        return [0.0] * 20

    recent = list(candles)
    current = recent[-1]

    closes = [c.close for c in recent]
    ranges = [
        max(c.high - c.low, 0.0)
        for c in recent
    ]

    true_ranges: List[float] = []

    for index, candle in enumerate(recent):
        previous = recent[index - 1] if index > 0 else None
        true_ranges.append(
            true_range(previous, candle)
        )

    atr_window = true_ranges[-14:]
    atr = rolling_mean(atr_window)
    atr = max(atr, abs(current.close) * 1e-8, 1e-12)

    def normalized_return(lookback: int) -> float:
        if len(recent) <= lookback:
            return 0.0

        reference = recent[-1 - lookback].close
        if reference == 0:
            return 0.0

        return (
            recent[-1].close - reference
        ) / abs(reference)

    body = abs(current.close - current.open)
    candle_range = max(
        current.high - current.low,
        1e-12,
    )

    upper_wick = max(
        current.high - max(current.open, current.close),
        0.0,
    )

    lower_wick = max(
        min(current.open, current.close) - current.low,
        0.0,
    )

    recent_mean_range = max(
        rolling_mean(ranges[-14:]),
        1e-12,
    )

    highest = max(
        candle.high for candle in recent[-20:]
    )

    lowest = min(
        candle.low for candle in recent[-20:]
    )

    recent_span = max(
        highest - lowest,
        1e-12,
    )

    close_position = (
        (current.close - lowest)
        / recent_span
    )

    returns_1 = normalized_return(1)
    returns_3 = normalized_return(3)
    returns_5 = normalized_return(5)
    returns_10 = normalized_return(10)

    volatility = (
        rolling_std(
            [
                candle_return(
                    recent[i - 1],
                    recent[i],
                )
                for i in range(1, len(recent))
            ][-14:]
        )
        if len(recent) >= 3
        else 0.0
    )

    volume_mean = max(
        rolling_mean(
            [c.volume for c in recent[-20:]]
        ),
        1e-12,
    )

    volume_ratio = current.volume / volume_mean

    slope = rolling_slope(
        [
            value / max(abs(current.close), 1e-12)
            for value in closes[-20:]
        ]
    )

    compression = (
        rolling_mean(ranges[-5:])
        / max(
            rolling_mean(ranges[-20:]),
            1e-12,
        )
    )

    breakout_up = (
        1.0
        if current.close >= max(
            candle.high for candle in recent[-6:-1]
        )
        else 0.0
    )

    breakout_down = (
        1.0
        if current.close <= min(
            candle.low for candle in recent[-6:-1]
        )
        else 0.0
    )

    features = [
        returns_1,
        returns_3,
        returns_5,
        returns_10,
        slope,
        volatility,
        candle_range / atr,
        body / candle_range,
        upper_wick / candle_range,
        lower_wick / candle_range,
        close_position,
        current.close / max(abs(current.open), 1e-12) - 1.0,
        recent_mean_range / atr,
        compression,
        volume_ratio,
        breakout_up,
        breakout_down,
        (highest - current.close) / atr,
        (current.close - lowest) / atr,
        len(recent) / max(LOOKBACK_BY_TIMEFRAME[timeframe], 1),
    ]

    # Keep the vector size fixed.
    if len(features) < 20:
        features.extend(
            [0.0] * (20 - len(features))
        )

    return [
        float(
            max(
                min(feature, 100.0),
                -100.0,
            )
        )
        for feature in features[:20]
    ]


def market_state_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> np.ndarray:
    """
    Build one multi-timeframe numerical market state.

    Every timeframe is evaluated only from candles fully closed by target_time.
    """
    vector: List[float] = []

    for timeframe in SUPPORTED_TIMEFRAMES:
        candles = candles_by_timeframe.get(timeframe, [])

        window = slice_closed_history(
            candles=candles,
            target_time=target_time,
            timeframe=timeframe,
            max_count=LOOKBACK_BY_TIMEFRAME[timeframe],
        )

        vector.extend(
            timeframe_features(
                window,
                timeframe,
            )
        )

    # Add normalized clock information.
    dt = datetime.fromtimestamp(
        int(target_time),
        tz=timezone.utc,
    )

    minute_of_day = (
        dt.hour * 60
        + dt.minute
    )

    day_fraction = (
        minute_of_day / (24.0 * 60.0)
    )

    weekday_fraction = (
        dt.weekday() / 6.0
        if dt.weekday() > 0
        else 0.0
    )

    vector.extend([
        math.sin(2.0 * math.pi * day_fraction),
        math.cos(2.0 * math.pi * day_fraction),
        math.sin(2.0 * math.pi * weekday_fraction),
        math.cos(2.0 * math.pi * weekday_fraction),
    ])

    return np.asarray(
        vector,
        dtype=np.float32,
    )


# ============================================================================
# USER SETUP TARGETS
# ============================================================================

DIRECTION_TO_INDEX = {
    "WAIT": 0,
    "LONG": 1,
    "SHORT": 2,
}

TIMEFRAME_TO_INDEX = {
    timeframe: index
    for index, timeframe in enumerate(
        SUPPORTED_TIMEFRAMES
    )
}


def current_price_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> Optional[float]:
    """Return the most recent fully known H1/M15/M1 close."""
    preferred = (
        "M1",
        "M5",
        "M15",
        "M30",
        "H1",
    )

    for timeframe in preferred:
        candles = candles_by_timeframe.get(timeframe, [])
        index = last_closed_index(
            candles,
            target_time,
            timeframe,
        )
        if index >= 0:
            price = candles[index].close
            if price > 0:
                return float(price)

    return None


def atr_at(
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> float:
    """Return a robust ATR estimate from the M15 or H1 history."""
    for timeframe in ("M15", "H1", "M5", "M1"):
        candles = candles_by_timeframe.get(timeframe, [])

        window = slice_closed_history(
            candles=candles,
            target_time=target_time,
            timeframe=timeframe,
            max_count=30,
        )

        if len(window) < 3:
            continue

        values = []

        for index, candle in enumerate(window):
            previous = (
                window[index - 1]
                if index > 0
                else None
            )
            values.append(
                true_range(
                    previous,
                    candle,
                )
            )

        value = rolling_mean(values[-14:])

        if value > 0:
            return float(value)

    price = current_price_at(
        candles_by_timeframe,
        target_time,
    )

    return max(
        abs(price or 1.0) * 1e-4,
        1e-8,
    )


def setup_target_vector(
    setup: Optional[UserSetup],
    candles_by_timeframe: Dict[str, List[Candle]],
    target_time: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Build labels for:
        - direction class
        - setup quality
        - target price offsets
        - target masks
        - setup timeframe
        - entry timeframe

    Price targets are represented in ATR units relative to the current price.
    """
    target_prices = np.zeros(7, dtype=np.float32)
    target_masks = np.zeros(7, dtype=np.float32)

    if setup is None:
        return (
            DIRECTION_TO_INDEX["WAIT"],
            np.asarray([0.0], dtype=np.float32),
            target_prices,
            target_masks,
            TIMEFRAME_TO_INDEX["H1"],
            TIMEFRAME_TO_INDEX["M15"],
        )

    direction_index = DIRECTION_TO_INDEX.get(
        setup.direction,
        DIRECTION_TO_INDEX["WAIT"],
    )

    quality = np.asarray(
        [1.0],
        dtype=np.float32,
    )

    current_price = current_price_at(
        candles_by_timeframe,
        target_time,
    )

    if current_price is None:
        current_price = (
            setup.entry
            or setup.anchor_price
            or 1.0
        )

    atr = atr_at(
        candles_by_timeframe,
        target_time,
    )

    raw_prices = [
        setup.anchor_price,
        setup.focus_1,
        setup.focus_2,
        setup.breakout,
        setup.entry,
        setup.sl,
        setup.tp,
    ]

    for index, price in enumerate(raw_prices):
        if price is None:
            continue

        target_prices[index] = float(
            (price - current_price) / atr
        )
        target_masks[index] = 1.0

    setup_tf = TIMEFRAME_TO_INDEX.get(
        setup.setup_timeframe,
        TIMEFRAME_TO_INDEX["H1"],
    )

    entry_tf = TIMEFRAME_TO_INDEX.get(
        setup.entry_timeframe,
        TIMEFRAME_TO_INDEX["M15"],
    )

    return (
        direction_index,
        quality,
        target_prices,
        target_masks,
        setup_tf,
        entry_tf,
    )


# ============================================================================
# TRAINING SAMPLE GENERATION
# ============================================================================

def setup_time(setup: UserSetup) -> Optional[int]:
    """Return the canonical decision timestamp for one setup."""
    return (
        setup.entry_candle_time
        if setup.entry_candle_time is not None
        else setup.anchor_time
    )


def closest_setup(
    setup_times: Sequence[int],
    target_time: int,
    tolerance: int,
) -> Optional[int]:
    """Return the index of the closest positive setup within tolerance."""
    if not setup_times:
        return None

    best_index = None
    best_distance = None

    for index, setup_time_value in enumerate(setup_times):
        distance = abs(
            int(setup_time_value)
            - int(target_time)
        )

        if distance > tolerance:
            continue

        if (
            best_distance is None
            or distance < best_distance
        ):
            best_index = index
            best_distance = distance

    return best_index


def make_negative_times(
    setups: Sequence[UserSetup],
    base_candles: Sequence[Candle],
) -> List[int]:
    """
    Create no-trade examples.

    We deliberately combine:
        - hard negatives near real trades
        - random negatives across the history

    This teaches the model not to treat every market movement as an entry.
    """
    if not setups or not base_candles:
        return []

    setup_times = [
        setup_time(setup)
        for setup in setups
        if setup_time(setup) is not None
    ]

    setup_times = [
        int(value)
        for value in setup_times
        if value is not None
    ]

    candle_times = [
        candle.time
        for candle in base_candles
    ]

    time_set = set(candle_times)
    positives = set(setup_times)

    result: set[int] = set()

    # Hard negatives around positive trade decisions.
    base_seconds = TIMEFRAME_SECONDS["M15"]

    for positive_time in setup_times:
        for offset in HARD_NEGATIVE_OFFSETS:
            candidate = (
                int(positive_time)
                + int(offset) * base_seconds
            )

            if candidate not in time_set:
                continue

            if any(
                abs(candidate - positive) <= POSITIVE_EXCLUSION_SECONDS
                for positive in positives
            ):
                continue

            result.add(candidate)

    # Random negatives across the history.
    target_count = max(
        len(setups) * RANDOM_NEGATIVE_MULTIPLIER,
        len(result),
    )

    attempts = 0
    maximum_attempts = max(
        target_count * 20,
        100,
    )

    while (
        len(result) < target_count
        and attempts < maximum_attempts
    ):
        attempts += 1

        candidate = random.choice(candle_times)

        if any(
            abs(candidate - positive)
            <= POSITIVE_EXCLUSION_SECONDS
            for positive in positives
        ):
            continue

        result.add(candidate)

    return sorted(result)


def build_training_dataset(
    market: str,
    setups: Sequence[UserSetup],
    candles_by_timeframe: Dict[str, List[Candle]],
) -> Dict[str, Any]:
    """Create the complete numerical training dataset."""
    if "M15" in candles_by_timeframe:
        base_candles = candles_by_timeframe["M15"]
    elif "M1" in candles_by_timeframe:
        base_candles = candles_by_timeframe["M1"]
    elif candles_by_timeframe:
        base_candles = next(
            iter(candles_by_timeframe.values())
        )
    else:
        raise RuntimeError(
            f"No chart history was found for {market}."
        )

    samples: List[np.ndarray] = []
    direction_labels: List[int] = []
    quality_labels: List[float] = []
    price_targets: List[np.ndarray] = []
    price_masks: List[np.ndarray] = []
    setup_tf_labels: List[int] = []
    entry_tf_labels: List[int] = []

    positive_count = 0
    negative_count = 0
    skipped_setups = 0

    # ---------------------------------------------------------------------
    # Positive examples: exact user decisions.
    # ---------------------------------------------------------------------
    for setup in setups:
        decision_time = setup_time(setup)

        if decision_time is None:
            skipped_setups += 1
            continue

        features = market_state_at(
            candles_by_timeframe,
            decision_time,
        )

        direction, quality, targets, masks, setup_tf, entry_tf = (
            setup_target_vector(
                setup=setup,
                candles_by_timeframe=candles_by_timeframe,
                target_time=decision_time,
            )
        )

        samples.append(features)
        direction_labels.append(direction)
        quality_labels.append(float(quality[0]))
        price_targets.append(targets)
        price_masks.append(masks)
        setup_tf_labels.append(setup_tf)
        entry_tf_labels.append(entry_tf)

        positive_count += 1

    # ---------------------------------------------------------------------
    # Negative examples: places where the user did not trade.
    # ---------------------------------------------------------------------
    negative_times = make_negative_times(
        setups,
        base_candles,
    )

    for decision_time in negative_times:
        features = market_state_at(
            candles_by_timeframe,
            decision_time,
        )

        direction_labels.append(
            DIRECTION_TO_INDEX["WAIT"]
        )
        quality_labels.append(0.0)
        price_targets.append(
            np.zeros(7, dtype=np.float32)
        )
        price_masks.append(
            np.zeros(7, dtype=np.float32)
        )
        setup_tf_labels.append(
            TIMEFRAME_TO_INDEX["H1"]
        )
        entry_tf_labels.append(
            TIMEFRAME_TO_INDEX["M15"]
        )
        samples.append(features)

        negative_count += 1

    if not samples:
        raise RuntimeError(
            f"No usable training examples were created for {market}."
        )

    feature_matrix = np.vstack(
        samples
    ).astype(np.float32)

    return {
        "features": feature_matrix,
        "direction": np.asarray(
            direction_labels,
            dtype=np.int64,
        ),
        "quality": np.asarray(
            quality_labels,
            dtype=np.float32,
        ).reshape(-1, 1),
        "price_targets": np.vstack(
            price_targets
        ).astype(np.float32),
        "price_masks": np.vstack(
            price_masks
        ).astype(np.float32),
        "setup_timeframe": np.asarray(
            setup_tf_labels,
            dtype=np.int64,
        ),
        "entry_timeframe": np.asarray(
            entry_tf_labels,
            dtype=np.int64,
        ),
        "stats": {
            "market": market,
            "positive_examples": positive_count,
            "negative_examples": negative_count,
            "total_examples": len(samples),
            "skipped_setups": skipped_setups,
        },
    }


# ============================================================================
# FEATURE STANDARDIZATION
# ============================================================================

def fit_scaler(
    features: np.ndarray,
) -> FeatureScaler:
    """Fit standardization parameters."""
    mean = np.mean(
        features,
        axis=0,
        keepdims=False,
    )

    std = np.std(
        features,
        axis=0,
        keepdims=False,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    return FeatureScaler(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def apply_scaler(
    features: np.ndarray,
    scaler: FeatureScaler,
) -> np.ndarray:
    """Standardize a feature matrix."""
    transformed = (
        features - scaler.mean
    ) / scaler.std

    return np.nan_to_num(
        transformed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


# ============================================================================
# MICRO AI MODEL
# ============================================================================

class TradingMicroModel(nn.Module):
    """
    Small multi-task neural network.

    Outputs:
        direction:
            WAIT / LONG / SHORT

        setup_probability:
            Probability that the current market state resembles a
            trade-worthy user setup.

        setup_timeframe:
            Predicted setup timeframe.

        entry_timeframe:
            Predicted entry timeframe.

        price_offsets:
            ATR-normalized offsets for:
                level
                focus_1
                focus_2
                breakout
                entry
                sl
                tp
    """

    def __init__(
        self,
        input_size: int,
        hidden_1: int = 256,
        hidden_2: int = 128,
        hidden_3: int = 64,
    ) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(0.08),

            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(0.06),

            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.GELU(),
        )

        self.direction_head = nn.Linear(
            hidden_3,
            3,
        )

        self.setup_probability_head = nn.Linear(
            hidden_3,
            1,
        )

        self.setup_timeframe_head = nn.Linear(
            hidden_3,
            len(SUPPORTED_TIMEFRAMES),
        )

        self.entry_timeframe_head = nn.Linear(
            hidden_3,
            len(SUPPORTED_TIMEFRAMES),
        )

        self.price_offset_head = nn.Linear(
            hidden_3,
            7,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        latent = self.backbone(x)

        return {
            "direction": self.direction_head(
                latent
            ),
            "setup_probability": self.setup_probability_head(
                latent
            ),
            "setup_timeframe": self.setup_timeframe_head(
                latent
            ),
            "entry_timeframe": self.entry_timeframe_head(
                latent
            ),
            "price_offsets": self.price_offset_head(
                latent
            ),
        }


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Smooth L1 loss using only available price labels."""
    raw = nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )

    weighted = raw * mask
    denominator = mask.sum().clamp_min(1.0)

    return weighted.sum() / denominator


def compute_loss(
    outputs: Dict[str, Tensor],
    direction: Tensor,
    quality: Tensor,
    price_targets: Tensor,
    price_masks: Tensor,
    setup_timeframe: Tensor,
    entry_timeframe: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute the multi-task training loss."""
    direction_loss = nn.functional.cross_entropy(
        outputs["direction"],
        direction,
        label_smoothing=0.04,
    )

    quality_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["setup_probability"],
        quality,
    )

    setup_tf_loss = nn.functional.cross_entropy(
        outputs["setup_timeframe"],
        setup_timeframe,
    )

    entry_tf_loss = nn.functional.cross_entropy(
        outputs["entry_timeframe"],
        entry_timeframe,
    )

    price_loss = masked_smooth_l1(
        outputs["price_offsets"],
        price_targets,
        price_masks,
    )

    total = (
        1.00 * direction_loss
        + 0.75 * quality_loss
        + 0.20 * setup_tf_loss
        + 0.20 * entry_tf_loss
        + 0.70 * price_loss
    )

    metrics = {
        "total": float(total.detach().cpu().item()),
        "direction": float(
            direction_loss.detach().cpu().item()
        ),
        "quality": float(
            quality_loss.detach().cpu().item()
        ),
        "setup_timeframe": float(
            setup_tf_loss.detach().cpu().item()
        ),
        "entry_timeframe": float(
            entry_tf_loss.detach().cpu().item()
        ),
        "price": float(
            price_loss.detach().cpu().item()
        ),
    }

    return total, metrics


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    dataset: Dict[str, Any],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> Tuple[TradingMicroModel, FeatureScaler, Dict[str, Any]]:
    """Train the numerical micro model."""
    features = dataset["features"]

    scaler = fit_scaler(features)
    scaled_features = apply_scaler(
        features,
        scaler,
    )

    tensor_features = torch.from_numpy(
        scaled_features
    )

    tensor_direction = torch.from_numpy(
        dataset["direction"]
    )

    tensor_quality = torch.from_numpy(
        dataset["quality"]
    )

    tensor_targets = torch.from_numpy(
        dataset["price_targets"]
    )

    tensor_masks = torch.from_numpy(
        dataset["price_masks"]
    )

    tensor_setup_tf = torch.from_numpy(
        dataset["setup_timeframe"]
    )

    tensor_entry_tf = torch.from_numpy(
        dataset["entry_timeframe"]
    )

    full_dataset = TensorDataset(
        tensor_features,
        tensor_direction,
        tensor_quality,
        tensor_targets,
        tensor_masks,
        tensor_setup_tf,
        tensor_entry_tf,
    )

    loader = DataLoader(
        full_dataset,
        batch_size=max(1, min(batch_size, len(full_dataset))),
        shuffle=True,
        drop_last=False,
    )

    model = TradingMicroModel(
        input_size=scaled_features.shape[1],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    model.train()

    last_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for (
            batch_x,
            batch_direction,
            batch_quality,
            batch_targets,
            batch_masks,
            batch_setup_tf,
            batch_entry_tf,
        ) in loader:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_x)

            loss, metrics = compute_loss(
                outputs=outputs,
                direction=batch_direction,
                quality=batch_quality,
                price_targets=batch_targets,
                price_masks=batch_masks,
                setup_timeframe=batch_setup_tf,
                entry_timeframe=batch_entry_tf,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            batches += 1
            last_metrics = metrics

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            average_loss = (
                epoch_loss / max(batches, 1)
            )

            print(
                f"[TRAIN] epoch={epoch:04d} "
                f"loss={average_loss:.6f} "
                f"direction={last_metrics.get('direction', 0.0):.6f} "
                f"quality={last_metrics.get('quality', 0.0):.6f} "
                f"price={last_metrics.get('price', 0.0):.6f}"
            )

    model.eval()

    training_summary = {
        **dataset["stats"],
        "input_features": int(
            scaled_features.shape[1]
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "final_metrics": last_metrics,
    }

    return (
        model,
        scaler,
        training_summary,
    )


# ============================================================================
# MODEL STORAGE
# ============================================================================

def save_model(
    market: str,
    model: TradingMicroModel,
    scaler: FeatureScaler,
    training_summary: Dict[str, Any],
) -> Path:
    """Save one complete market-specific model."""
    directory = model_dir(market)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = model_path(market)

    checkpoint = {
        "market": str(market),
        "model_state": model.state_dict(),
        "input_size": int(
            model.backbone[0].in_features
        ),
        "hidden_sizes": [
            int(model.backbone[0].out_features),
            int(model.backbone[4].out_features),
            int(model.backbone[8].out_features),
        ],
        "timeframes": list(SUPPORTED_TIMEFRAMES),
        "lookback_by_timeframe": dict(
            LOOKBACK_BY_TIMEFRAME
        ),
        "feature_mean": scaler.mean,
        "feature_std": scaler.std,
        "training_summary": training_summary,
        "format_version": 1,
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_model(
    market: str,
) -> Tuple[TradingMicroModel, FeatureScaler, Dict[str, Any]]:
    """Load a previously trained market-specific model."""
    path = model_path(market)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model does not exist for {market}: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    input_size = int(
        checkpoint["input_size"]
    )

    hidden_sizes = checkpoint.get(
        "hidden_sizes",
        [256, 128, 64],
    )

    model = TradingMicroModel(
        input_size=input_size,
        hidden_1=int(hidden_sizes[0]),
        hidden_2=int(hidden_sizes[1]),
        hidden_3=int(hidden_sizes[2]),
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    scaler = FeatureScaler(
        mean=np.asarray(
            checkpoint["feature_mean"],
            dtype=np.float32,
        ),
        std=np.asarray(
            checkpoint["feature_std"],
            dtype=np.float32,
        ),
    )

    return (
        model,
        scaler,
        checkpoint,
    )


# ============================================================================
# INFERENCE
# ============================================================================

def predict_from_features(
    model: TradingMicroModel,
    scaler: FeatureScaler,
    features: np.ndarray,
) -> Dict[str, Any]:
    """Predict from one already-built market state."""
    one = np.asarray(
        features,
        dtype=np.float32,
    ).reshape(1, -1)

    scaled = apply_scaler(
        one,
        scaler,
    )

    tensor = torch.from_numpy(
        scaled
    )

    with torch.no_grad():
        outputs = model(tensor)

    direction_probabilities = torch.softmax(
        outputs["direction"],
        dim=1,
    )[0].cpu().numpy()

    setup_probability = torch.sigmoid(
        outputs["setup_probability"]
    )[0, 0].item()

    setup_tf_probabilities = torch.softmax(
        outputs["setup_timeframe"],
        dim=1,
    )[0].cpu().numpy()

    entry_tf_probabilities = torch.softmax(
        outputs["entry_timeframe"],
        dim=1,
    )[0].cpu().numpy()

    price_offsets = (
        outputs["price_offsets"][0]
        .cpu()
        .numpy()
    )

    direction_index = int(
        np.argmax(direction_probabilities)
    )

    direction_name = {
        0: "WAIT",
        1: "LONG",
        2: "SHORT",
    }[direction_index]

    setup_tf_index = int(
        np.argmax(setup_tf_probabilities)
    )

    entry_tf_index = int(
        np.argmax(entry_tf_probabilities)
    )

    return {
        "direction": direction_name,
        "direction_probabilities": {
            "WAIT": float(
                direction_probabilities[0]
            ),
            "LONG": float(
                direction_probabilities[1]
            ),
            "SHORT": float(
                direction_probabilities[2]
            ),
        },
        "setup_probability": float(
            setup_probability
        ),
        "setup_timeframe": SUPPORTED_TIMEFRAMES[
            setup_tf_index
        ],
        "setup_timeframe_probability": float(
            setup_tf_probabilities[
                setup_tf_index
            ]
        ),
        "entry_timeframe": SUPPORTED_TIMEFRAMES[
            entry_tf_index
        ],
        "entry_timeframe_probability": float(
            entry_tf_probabilities[
                entry_tf_index
            ]
        ),
        "price_offsets_atr": {
            "level": float(price_offsets[0]),
            "focus_1": float(price_offsets[1]),
            "focus_2": float(price_offsets[2]),
            "breakout": float(price_offsets[3]),
            "entry": float(price_offsets[4]),
            "sl": float(price_offsets[5]),
            "tp": float(price_offsets[6]),
        },
    }


def predict_market(
    market: str,
    target_time: int,
) -> Dict[str, Any]:
    """
    Load a market model and make one point-in-time prediction.

    This function uses only candles fully closed before target_time.
    """
    model, scaler, _ = load_model(market)

    candles_by_timeframe = load_all_market_candles(
        market
    )

    features = market_state_at(
        candles_by_timeframe,
        int(target_time),
    )

    prediction = predict_from_features(
        model=model,
        scaler=scaler,
        features=features,
    )

    current_price = current_price_at(
        candles_by_timeframe,
        int(target_time),
    )

    atr = atr_at(
        candles_by_timeframe,
        int(target_time),
    )

    if current_price is not None:
        price_offsets = prediction[
            "price_offsets_atr"
        ]

        prediction["price_candidates"] = {
            key: float(
                current_price
                + value * atr
            )
            for key, value in price_offsets.items()
        }

    prediction["market"] = market
    prediction["target_time"] = int(target_time)

    return prediction


# ============================================================================
# MARKET TRAINING ENTRY POINT
# ============================================================================

def train_market(
    market: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> Path:
    """
    Train a complete market-specific model from the user's Learning data.

    Example:
        train_market("EURUSD")
    """
    market = str(market).strip().upper()

    if not market:
        raise ValueError("Market must not be empty.")

    print("")
    print("=" * 72)
    print(f"TRAINING MARKET MODEL: {market}")
    print("=" * 72)

    print(
        f"Learning data: {setup_path(market)}"
    )

    setups = load_learning_setups(
        market
    )

    print(
        f"Loaded Learning setups: {len(setups)}"
    )

    if not setups:
        raise RuntimeError(
            f"No usable Learning setups found for {market}."
        )

    print(
        f"Loading chart history from: {CHARTS_ROOT}"
    )

    candles_by_timeframe = load_all_market_candles(
        market
    )

    available_timeframes = [
        timeframe
        for timeframe in SUPPORTED_TIMEFRAMES
        if candles_by_timeframe.get(timeframe)
    ]

    if not available_timeframes:
        raise RuntimeError(
            f"No local chart history found for {market}."
        )

    print(
        "Available timeframes:",
        ", ".join(available_timeframes),
    )

    dataset = build_training_dataset(
        market=market,
        setups=setups,
        candles_by_timeframe=candles_by_timeframe,
    )

    print(
        f"Positive examples: "
        f"{dataset['stats']['positive_examples']}"
    )

    print(
        f"Negative examples: "
        f"{dataset['stats']['negative_examples']}"
    )

    print(
        f"Total examples: "
        f"{dataset['stats']['total_examples']}"
    )

    model, scaler, training_summary = train_model(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    saved_path = save_model(
        market=market,
        model=model,
        scaler=scaler,
        training_summary=training_summary,
    )

    print("")
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Market: {market}")
    print(f"Model:  {saved_path}")
    print(
        f"Examples: "
        f"{training_summary['total_examples']}"
    )
    print("=" * 72)
    print("")

    return saved_path


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def show_model_info(
    market: str,
) -> None:
    """Print information stored in a trained model."""
    _, _, checkpoint = load_model(
        market
    )

    summary = checkpoint.get(
        "training_summary",
        {},
    )

    print("")
    print("=" * 72)
    print(f"MODEL: {market}")
    print("=" * 72)

    print(
        f"Path: {model_path(market)}"
    )

    print(
        f"Input features: "
        f"{checkpoint.get('input_size')}"
    )

    print(
        f"Training examples: "
        f"{summary.get('total_examples', 0)}"
    )

    print(
        f"Positive examples: "
        f"{summary.get('positive_examples', 0)}"
    )

    print(
        f"Negative examples: "
        f"{summary.get('negative_examples', 0)}"
    )

    print(
        "Timeframes: "
        + ", ".join(
            checkpoint.get(
                "timeframes",
                SUPPORTED_TIMEFRAMES,
            )
        )
    )

    print("=" * 72)
    print("")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and inspect numerical market-specific "
            "AI models."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a market-specific model.",
    )

    train_parser.add_argument(
        "market",
        type=str,
        help="Market symbol, for example EURUSD.",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a trained model.",
    )

    info_parser.add_argument(
        "market",
        type=str,
        help="Market symbol, for example EURUSD.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict one historical point using a trained model.",
    )

    predict_parser.add_argument(
        "market",
        type=str,
        help="Market symbol, for example EURUSD.",
    )

    predict_parser.add_argument(
        "time",
        type=str,
        help="Unix timestamp or ISO-8601 datetime.",
    )

    return parser


def main() -> int:
    """CLI entry point."""
    set_seed()

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            train_market(
                market=args.market,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            return 0

        if args.command == "info":
            show_model_info(
                args.market,
            )
            return 0

        if args.command == "predict":
            target_time = parse_timestamp(
                args.time
            )

            if target_time is None:
                raise ValueError(
                    "Could not parse prediction time."
                )

            prediction = predict_market(
                market=args.market,
                target_time=target_time,
            )

            print(
                json.dumps(
                    prediction,
                    indent=4,
                    ensure_ascii=False,
                )
            )

            return 0

        parser.error(
            f"Unknown command: {args.command}"
        )

    except Exception as error:
        print(
            f"[AI ERROR] {error}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
