"""Central configuration for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import random
import re
from pathlib import Path
from typing import Final

import numpy as np
import torch

try:
    from client import (
        BASE_DIR as CLIENT_BASE_DIR,
        CHARTS_ROOT as CLIENT_CHARTS_ROOT,
        DATA_ROOT as CLIENT_DATA_ROOT,
        DEFAULT_CANDLE_FILENAME,
        DEFAULT_MODEL_FILENAME,
        DEFAULT_SETUP_FILENAME,
        WEB_MODE as CLIENT_WEB_MODE,
        SUPPORTED_TIMEFRAMES as CLIENT_SUPPORTED_TIMEFRAMES,
    )
except ImportError:
    from ..client import (
        BASE_DIR as CLIENT_BASE_DIR,
        CHARTS_ROOT as CLIENT_CHARTS_ROOT,
        DATA_ROOT as CLIENT_DATA_ROOT,
        DEFAULT_CANDLE_FILENAME,
        DEFAULT_MODEL_FILENAME,
        DEFAULT_SETUP_FILENAME,
        WEB_MODE as CLIENT_WEB_MODE,
        SUPPORTED_TIMEFRAMES as CLIENT_SUPPORTED_TIMEFRAMES,
    )


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR: Final[Path] = Path(CLIENT_BASE_DIR).resolve()
ENGINE_ROOT: Final[Path] = BASE_DIR / "engine"

WEB_MODE: Final[bool] = bool(CLIENT_WEB_MODE)

TRADING_SETUP_ROOT: Final[Path] = BASE_DIR / "TRADING SET UP"
CHARTS_ROOT: Final[Path] = Path(CLIENT_CHARTS_ROOT).resolve()
MODELS_ROOT: Final[Path] = BASE_DIR / "MODELS"
USER_DATA_ROOT: Final[Path] = Path(CLIENT_DATA_ROOT).resolve()
CUSTOMERS_ROOT: Final[Path] = BASE_DIR / "CUSTOMERS"

MARKET_MEMORY_ROOT: Final[Path] = BASE_DIR / "MARKET_MEMORY"
MARKET_PATTERN_ROOT: Final[Path] = MARKET_MEMORY_ROOT / "PATTERNS"
MARKET_EVENT_ROOT: Final[Path] = MARKET_MEMORY_ROOT / "EVENTS"
PATTERN_CATALOG_ROOT: Final[Path] = MARKET_MEMORY_ROOT / "CATALOGS"

SETUP_FILENAME: Final[str] = DEFAULT_SETUP_FILENAME
CANDLE_FILENAME: Final[str] = DEFAULT_CANDLE_FILENAME
MODEL_FILENAME: Final[str] = DEFAULT_MODEL_FILENAME

CUSTOMER_MODELS_DIRECTORY_NAME: Final[str] = "MODELS"
CUSTOMER_RESULTS_DIRECTORY_NAME: Final[str] = "RESULTS"
CUSTOMER_RUNTIME_DIRECTORY_NAME: Final[str] = "RUNTIME"
CUSTOMER_TIMELINE_DIRECTORY_NAME: Final[str] = "TIMELINE"


# ============================================================================
# IDENTITY AND DATA SCOPE
# ============================================================================

CUSTOMER_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "user_id",
    "market",
    "strategy_id",
    "tradingbot_id",
)

CUSTOMER_EXPERIENCE_SCOPE_FIELDS: Final[tuple[str, ...]] = (
    "user_id",
    "market",
    "strategy_id",
    "tradingbot_id",
)

MARKET_PATTERN_SCOPE_FIELDS: Final[tuple[str, ...]] = (
    "market",
    "timeframe",
)


# ============================================================================
# TIME AND CHRONOLOGY CONFIGURATION
# ============================================================================

TIMESTAMP_UNIT: Final[str] = "unix_seconds"
TIMESTAMP_TIMEZONE: Final[str] = "UTC"
CANDLE_TIMESTAMP_REFERENCE: Final[str] = "OPEN"
HISTORY_PROCESSING_ORDER: Final[str] = "ASCENDING"
HISTORY_STAGE_YEARS: Final[int] = 5

REQUIRE_CLOSED_CANDLES: Final[bool] = True
REQUIRE_POINT_IN_TIME_FEATURES: Final[bool] = True
REQUIRE_PATTERN_RECOGNITION_TIME: Final[bool] = True
EXCLUDE_FUTURE_EVENTS_FROM_PREDICTION: Final[bool] = True


# ============================================================================
# TIMEFRAME CONFIGURATION
# ============================================================================

ALL_TIMEFRAMES: Final[tuple[str, ...]] = (
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)

WEB_MODE_TIMEFRAMES: Final[tuple[str, ...]] = (
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)

LOCAL_TIMEFRAMES: Final[tuple[str, ...]] = ALL_TIMEFRAMES

SUPPORTED_TIMEFRAMES: Final[tuple[str, ...]] = (
    WEB_MODE_TIMEFRAMES if WEB_MODE else LOCAL_TIMEFRAMES
)

ALL_TIMEFRAME_SET: Final[frozenset[str]] = frozenset(ALL_TIMEFRAMES)
SUPPORTED_TIMEFRAME_SET: Final[frozenset[str]] = frozenset(
    SUPPORTED_TIMEFRAMES
)

ALL_TIMEFRAME_TO_INDEX: Final[dict[str, int]] = {
    timeframe: index
    for index, timeframe in enumerate(ALL_TIMEFRAMES)
}

TIMEFRAME_TO_INDEX: Final[dict[str, int]] = {
    timeframe: index
    for index, timeframe in enumerate(SUPPORTED_TIMEFRAMES)
}

TIMEFRAME_INDEX_TO_NAME: Final[dict[int, str]] = {
    index: timeframe
    for timeframe, index in TIMEFRAME_TO_INDEX.items()
}

TIMEFRAME_SECONDS: Final[dict[str, int]] = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}

LOOKBACK_BY_TIMEFRAME: Final[dict[str, int]] = {
    "M1": 32,
    "M5": 32,
    "M15": 32,
    "M30": 24,
    "H1": 24,
    "H4": 20,
    "D1": 16,
    "W1": 12,
}

RESOLUTION_TIMEFRAME_PRIORITY: Final[tuple[str, ...]] = (
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)


# ============================================================================
# PATTERN CATALOG CONFIGURATION
# ============================================================================

PATTERN_ID_DIGITS: Final[int] = 4
PATTERN_ID_STORAGE_TYPE: Final[str] = "string"
PATTERN_ID_ENCODING: Final[str] = "categorical"
PATTERN_CATEGORY_CODE_SUFFIX: Final[str] = "000"
PATTERN_ID_VALIDATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}$"
)

PATTERN_CATEGORY_ID_TO_NAME: Final[dict[int, str]] = {
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

PATTERN_CATEGORY_NAME_TO_ID: Final[dict[str, int]] = {
    name: category_id
    for category_id, name in PATTERN_CATEGORY_ID_TO_NAME.items()
}

MARKET_EVENT_TYPES: Final[tuple[str, ...]] = (
    "BREAKOUT",
    "FALSE_BREAKOUT",
    "RETEST",
    "REJECTION",
    "LIQUIDITY_SWEEP",
)


# ============================================================================
# TRADE OUTCOME AND PROBABILITY CONFIGURATION
# ============================================================================

OUTCOME_SL_FIRST: Final[int] = 0
OUTCOME_TP_FIRST: Final[int] = 1

TRADE_OUTCOME_TO_LABEL: Final[dict[str, int]] = {
    "SL_FIRST": OUTCOME_SL_FIRST,
    "TP_FIRST": OUTCOME_TP_FIRST,
}

LABEL_TO_TRADE_OUTCOME: Final[dict[int, str]] = {
    label: outcome
    for outcome, label in TRADE_OUTCOME_TO_LABEL.items()
}

PROBABILITY_TARGET_NAME: Final[str] = "TP_BEFORE_SL"
PROBABILITY_MIN_VALUE: Final[float] = 0.0
PROBABILITY_MAX_VALUE: Final[float] = 1.0
PROBABILITY_PERCENT_MULTIPLIER: Final[float] = 100.0

PREDICTION_TIME_FIELD: Final[str] = "predicted_at"
PREDICTED_PROBABILITY_FIELD: Final[str] = "predicted_probability_at_entry"
ACTUAL_OUTCOME_FIELD: Final[str] = "actual_outcome"
OUTCOME_KNOWN_AT_FIELD: Final[str] = "outcome_known_at"

EXCLUDE_OPEN_TRADES_FROM_FINAL_RESULTS: Final[bool] = True
EXCLUDE_AMBIGUOUS_TRADES_FROM_BINARY_LABELS: Final[bool] = True


# ============================================================================
# RANDOMNESS AND TRAINING DEFAULTS
# ============================================================================

SEED: Final[int] = 42

DEFAULT_EPOCHS: Final[int] = 80
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-5

HARD_NEGATIVE_OFFSETS: Final[tuple[int, ...]] = (
    -12,
    -8,
    -5,
    -3,
    -2,
    -1,
    1,
    2,
    3,
    5,
    8,
    12,
)

RANDOM_NEGATIVE_MULTIPLIER: Final[int] = 2
POSITIVE_EXCLUSION_SECONDS: Final[int] = 30 * 60


# ============================================================================
# LABEL INDEXES
# ============================================================================

DIRECTION_TO_INDEX: Final[dict[str, int]] = {
    "WAIT": 0,
    "LONG": 1,
    "SHORT": 2,
}


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy and PyTorch."""
    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)


# ============================================================================
# TIMEFRAME HELPERS
# ============================================================================

def normalize_known_timeframe(timeframe: str) -> str:
    """Normalize a timeframe supported by the complete Spectline catalog."""
    normalized = str(timeframe).strip().upper()

    if normalized not in ALL_TIMEFRAME_SET:
        raise ValueError(f"Unknown timeframe: {normalized}.")

    return normalized


def normalize_timeframe(timeframe: str) -> str:
    """Normalize a timeframe and validate it for the current runtime mode."""
    normalized = normalize_known_timeframe(timeframe)

    if normalized not in SUPPORTED_TIMEFRAME_SET:
        mode = "WebMode" if WEB_MODE else "local mode"
        raise ValueError(
            f"Timeframe '{normalized}' is not supported in {mode}."
        )

    return normalized


def timeframe_seconds(timeframe: str) -> int:
    """Return the duration in seconds for a runtime-supported timeframe."""
    normalized = normalize_timeframe(timeframe)
    return TIMEFRAME_SECONDS[normalized]


def known_timeframe_seconds(timeframe: str) -> int:
    """Return the duration in seconds for any configured timeframe."""
    normalized = normalize_known_timeframe(timeframe)
    return TIMEFRAME_SECONDS[normalized]


def timeframe_lookback(timeframe: str) -> int:
    """Return the configured feature lookback for a runtime-supported timeframe."""
    normalized = normalize_timeframe(timeframe)
    return LOOKBACK_BY_TIMEFRAME[normalized]


def available_timeframes(timeframes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return requested timeframes available in the current runtime mode."""
    requested = {normalize_known_timeframe(item) for item in timeframes}
    return tuple(
        timeframe
        for timeframe in SUPPORTED_TIMEFRAMES
        if timeframe in requested
    )


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration() -> None:
    """Validate shared configuration values and cross-module assumptions."""
    if len(ALL_TIMEFRAMES) != len(ALL_TIMEFRAME_SET):
        raise RuntimeError("The complete timeframe configuration has duplicates.")

    if set(ALL_TIMEFRAMES) != set(CLIENT_SUPPORTED_TIMEFRAMES):
        raise RuntimeError(
            "The AI timeframe configuration does not match client.py. "
            "Update both timeframe definitions together."
        )

    if not set(WEB_MODE_TIMEFRAMES).issubset(ALL_TIMEFRAME_SET):
        raise RuntimeError("WebMode contains an unknown timeframe.")

    if set(SUPPORTED_TIMEFRAMES) - set(TIMEFRAME_SECONDS):
        raise RuntimeError("A supported timeframe has no duration configured.")

    if set(SUPPORTED_TIMEFRAMES) - set(LOOKBACK_BY_TIMEFRAME):
        raise RuntimeError("A supported timeframe has no lookback configured.")

    if any(value <= 0 for value in TIMEFRAME_SECONDS.values()):
        raise RuntimeError("Timeframe durations must be positive.")

    if any(value <= 0 for value in LOOKBACK_BY_TIMEFRAME.values()):
        raise RuntimeError("Timeframe lookbacks must be positive.")

    if HISTORY_STAGE_YEARS <= 0:
        raise RuntimeError("History stage duration must be positive.")

    if PATTERN_ID_DIGITS != 4 or len(PATTERN_CATEGORY_ID_TO_NAME) != 10:
        raise RuntimeError("The pattern catalog configuration is invalid.")

    if set(PATTERN_CATEGORY_ID_TO_NAME) != set(range(10)):
        raise RuntimeError("Pattern category IDs must cover digits zero to nine.")

    if TRADE_OUTCOME_TO_LABEL != {"SL_FIRST": 0, "TP_FIRST": 1}:
        raise RuntimeError("Trade outcome labels must remain SL_FIRST=0 and TP_FIRST=1.")


validate_configuration()