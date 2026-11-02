"""Central configuration for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import random
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

WEB_MODE: Final[bool] = bool(CLIENT_WEB_MODE)

# These aliases are retained for compatibility with older AI integrations.
TRADING_SETUP_ROOT: Final[Path] = BASE_DIR / "TRADING SET UP"
CHARTS_ROOT: Final[Path] = Path(CLIENT_CHARTS_ROOT).resolve()
MODELS_ROOT: Final[Path] = BASE_DIR / "MODELS"
USER_DATA_ROOT: Final[Path] = Path(CLIENT_DATA_ROOT).resolve()

SETUP_FILENAME: Final[str] = DEFAULT_SETUP_FILENAME
CANDLE_FILENAME: Final[str] = DEFAULT_CANDLE_FILENAME
MODEL_FILENAME: Final[str] = DEFAULT_MODEL_FILENAME


# ============================================================================
# TIMEFRAME CONFIGURATION
# ============================================================================

# Keep the complete timeframe order stable for feature and model indexing.
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

# WebMode intentionally supports chart archives from M15 and above.
# Smaller chart archives are not uploaded in WebMode to limit storage usage.
WEB_MODE_TIMEFRAMES: Final[tuple[str, ...]] = (
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
)

# Local PC installations may use the complete timeframe set, including M1/M5.
LOCAL_TIMEFRAMES: Final[tuple[str, ...]] = ALL_TIMEFRAMES

# These are the timeframe options supported by the current runtime environment.
# The history module will further narrow this list to the timeframes actually
# referenced by a TradingBot's Learning setups and available as chart archives.
SUPPORTED_TIMEFRAMES: Final[tuple[str, ...]] = (
    WEB_MODE_TIMEFRAMES if WEB_MODE else LOCAL_TIMEFRAMES
)

SUPPORTED_TIMEFRAME_SET: Final[frozenset[str]] = frozenset(
    SUPPORTED_TIMEFRAMES
)

# Timeframe durations in seconds. The full map is retained so that archived
# timestamps can still be interpreted consistently across runtime modes.
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

# Number of recent closed candles represented for each timeframe.
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


# ============================================================================
# RANDOMNESS AND TRAINING DEFAULTS
# ============================================================================

SEED: Final[int] = 42

DEFAULT_EPOCHS: Final[int] = 80
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-5

# Hard negatives are sampled around real user trade decisions.
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

# Additional random no-trade examples are generated across the history.
RANDOM_NEGATIVE_MULTIPLIER: Final[int] = 2

# Keep this time window protected around positive examples when sampling
# negative examples, preventing a real trade from being labelled as WAIT.
POSITIVE_EXCLUSION_SECONDS: Final[int] = 30 * 60


# ============================================================================
# LABEL INDEXES
# ============================================================================

DIRECTION_TO_INDEX: Final[dict[str, int]] = {
    "WAIT": 0,
    "LONG": 1,
    "SHORT": 2,
}

TIMEFRAME_TO_INDEX: Final[dict[str, int]] = {
    timeframe: index
    for index, timeframe in enumerate(SUPPORTED_TIMEFRAMES)
}


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# TIMEFRAME HELPERS
# ============================================================================

def normalize_timeframe(timeframe: str) -> str:
    """Normalize a timeframe and validate it for the current runtime mode."""
    normalized = str(timeframe).strip().upper()

    if normalized not in SUPPORTED_TIMEFRAME_SET:
        raise ValueError(
            f"Timeframe '{normalized}' is not supported in "
            f"{'WebMode' if WEB_MODE else 'local mode'}."
        )

    return normalized


def timeframe_seconds(timeframe: str) -> int:
    """Return the duration in seconds for a supported timeframe."""
    normalized = normalize_timeframe(timeframe)
    return TIMEFRAME_SECONDS[normalized]


def timeframe_lookback(timeframe: str) -> int:
    """Return the configured feature lookback for a supported timeframe."""
    normalized = normalize_timeframe(timeframe)
    return LOOKBACK_BY_TIMEFRAME[normalized]


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

if set(ALL_TIMEFRAMES) != set(CLIENT_SUPPORTED_TIMEFRAMES):
    raise RuntimeError(
        "The AI timeframe configuration does not match client.py. "
        "Update both timeframe definitions together."
    )

if set(SUPPORTED_TIMEFRAMES) - set(TIMEFRAME_SECONDS):
    raise RuntimeError("A supported timeframe has no duration configured.")

if set(SUPPORTED_TIMEFRAMES) - set(LOOKBACK_BY_TIMEFRAME):
    raise RuntimeError("A supported timeframe has no lookback configured.")
