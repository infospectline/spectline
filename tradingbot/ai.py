"""Public AI facade for the Spectline trading engine.

This module exposes a stable application-facing API while the implementation
lives in the internal ``engine`` package.

The facade intentionally contains no data processing, feature engineering,
model architecture, training, or inference implementation.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from client import ClientIdentity
except ImportError:
    try:
        from .client import ClientIdentity
    except ImportError:
        ClientIdentity = Any  # type: ignore[misc,assignment]


# ============================================================================
# PUBLIC MODULE INFORMATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent


class AIEngineNotReadyError(RuntimeError):
    """Raised when a requested internal AI component is not implemented yet."""


# ============================================================================
# INTERNAL MODULE ROUTING
# ============================================================================

_ENGINE_EXPORTS = {
    # Settings
    "WEB_MODE": "settings",
    "TRADING_SETUP_ROOT": "settings",
    "CHARTS_ROOT": "settings",
    "MODELS_ROOT": "settings",
    "USER_DATA_ROOT": "settings",
    "SETUP_FILENAME": "settings",
    "CANDLE_FILENAME": "settings",
    "MODEL_FILENAME": "settings",
    "SUPPORTED_TIMEFRAMES": "settings",
    "TIMEFRAME_SECONDS": "settings",
    "LOOKBACK_BY_TIMEFRAME": "settings",
    "SEED": "settings",
    "DEFAULT_EPOCHS": "settings",
    "DEFAULT_BATCH_SIZE": "settings",
    "DEFAULT_LEARNING_RATE": "settings",
    "DEFAULT_WEIGHT_DECAY": "settings",
    "HARD_NEGATIVE_OFFSETS": "settings",
    "RANDOM_NEGATIVE_MULTIPLIER": "settings",
    "POSITIVE_EXCLUSION_SECONDS": "settings",
    "DIRECTION_TO_INDEX": "settings",
    "TIMEFRAME_TO_INDEX": "settings",
    "set_seed": "settings",

    # Data and identity
    "Candle": "data",
    "UserSetup": "data",
    "safe_market_folder": "data",
    "safe_user_folder": "data",
    "_resolve_ai_identity": "data",
    "setup_path": "data",
    "parse_timestamp": "data",
    "load_learning_setups": "data",
    "to_float": "data",
    "setup_time": "data",
    "closest_setup": "data",
    "make_negative_times": "data",
    "build_training_dataset": "data",
    "prepare_training_data": "data",
    "validate_training_data": "data",

    # Historical price data and time alignment
    "chart_path": "history",
    "load_candles": "history",
    "load_all_market_candles": "history",
    "select_required_timeframes": "history",
    "candle_return": "history",
    "true_range": "history",
    "rolling_mean": "history",
    "rolling_std": "history",
    "rolling_slope": "history",
    "last_closed_index": "history",
    "slice_closed_history": "history",

    # Numerical market features
    "sigmoid": "features",
    "timeframe_features": "features",
    "market_state_at": "features",
    "current_price_at": "features",
    "atr_at": "features",
    "setup_target_vector": "features",

    # Prepared data and model persistence
    "FeatureScaler": "memory",
    "model_dir": "memory",
    "model_path": "memory",
    "fit_scaler": "memory",
    "apply_scaler": "memory",
    "save_model": "memory",
    "load_model": "memory",

    # Neural network and model operations
    "TradingMicroModel": "model",
    "masked_smooth_l1": "model",
    "compute_loss": "model",
    "train_model": "model",
    "predict_from_features": "model",
    "predict_market": "model",
    "train_market": "model",
    "show_model_info": "model",
    "build_parser": "model",
    "main": "model",
}


def _import_engine_module(module_name: str) -> Any:
    """Import one internal engine module with package and script support."""
    if __package__:
        qualified_name = f"{__package__}.engine.{module_name}"
    else:
        qualified_name = f"engine.{module_name}"

    try:
        return importlib.import_module(qualified_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        is_missing_engine_component = (
            missing_name == qualified_name
            or missing_name == "engine"
            or missing_name.endswith(f".engine.{module_name}")
        )

        if not is_missing_engine_component:
            raise

        raise AIEngineNotReadyError(
            f"The AI engine component '{module_name}.py' is not available. "
            "Complete the corresponding engine module before calling this API."
        ) from error


def _get_engine_export(name: str) -> Any:
    """Resolve one public symbol from its assigned internal module."""
    module_name = _ENGINE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ai' has no attribute {name!r}")

    module = _import_engine_module(module_name)

    try:
        return getattr(module, name)
    except AttributeError as error:
        raise AIEngineNotReadyError(
            f"The symbol '{name}' has not been implemented in "
            f"'engine/{module_name}.py' yet."
        ) from error


def _call_engine(
    module_name: str,
    function_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call an internal engine function without exposing its implementation."""
    module = _import_engine_module(module_name)

    try:
        function = getattr(module, function_name)
    except AttributeError as error:
        raise AIEngineNotReadyError(
            f"The function '{function_name}' has not been implemented in "
            f"'engine/{module_name}.py' yet."
        ) from error

    if not callable(function):
        raise TypeError(
            f"'engine/{module_name}.py:{function_name}' is not callable."
        )

    return function(*args, **kwargs)


# ============================================================================
# IDENTITY AND PATH API
# ============================================================================

def setup_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the Learning setup path for one TradingBot identity."""
    return _call_engine(
        "data",
        "setup_path",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def model_dir(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the model directory for one TradingBot identity."""
    return _call_engine(
        "memory",
        "model_dir",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def model_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the model checkpoint path for one TradingBot identity."""
    return _call_engine(
        "memory",
        "model_path",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


# ============================================================================
# LEARNING DATA API
# ============================================================================

def load_learning_setups(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Any:
    """Load all Learning setups belonging to one TradingBot identity."""
    return _call_engine(
        "data",
        "load_learning_setups",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def prepare_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Build and validate the input data bundle without training model weights."""
    return _call_engine(
        "data",
        "prepare_training_data",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )

def prepare_and_store_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    bundle_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Prepare and store a Training Data Bundle for one TradingBot."""

    return _call_engine(
        "memory",
        "prepare_and_store_training_data",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
        bundle_id=bundle_id,
    )


# ============================================================================
# HISTORICAL DATA API
# ============================================================================

def load_all_market_candles(
    market: str,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Load the historical timeframe data required for a market or TradingBot."""
    return _call_engine(
        "history",
        "load_all_market_candles",
        market,
        identity=identity,
    )


# ============================================================================
# FEATURE AND INFERENCE API
# ============================================================================

def market_state_at(
    candles_by_timeframe: Dict[str, Any],
    target_time: int,
) -> Any:
    """Build the numerical market state available at a specific timestamp."""
    return _call_engine(
        "features",
        "market_state_at",
        candles_by_timeframe,
        target_time,
    )


def current_price_at(
    candles_by_timeframe: Dict[str, Any],
    target_time: int,
) -> Optional[float]:
    """Return the latest fully closed candle price available at a timestamp."""
    return _call_engine(
        "features",
        "current_price_at",
        candles_by_timeframe,
        target_time,
    )


def atr_at(
    candles_by_timeframe: Dict[str, Any],
    target_time: int,
) -> float:
    """Return the historical ATR estimate available at a timestamp."""
    return _call_engine(
        "features",
        "atr_at",
        candles_by_timeframe,
        target_time,
    )


def predict_from_features(
    model: Any,
    scaler: Any,
    features: Any,
) -> Dict[str, Any]:
    """Run inference from an already prepared numerical market state."""
    return _call_engine(
        "model",
        "predict_from_features",
        model=model,
        scaler=scaler,
        features=features,
    )


def predict_market(
    market: str,
    target_time: int,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Load one TradingBot model and predict a point-in-time market state."""
    return _call_engine(
        "model",
        "predict_market",
        market,
        target_time,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


# ============================================================================
# MODEL STORAGE AND TRAINING API
# ============================================================================

def load_model(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a model checkpoint after validating its complete identity."""
    return _call_engine(
        "memory",
        "load_model",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def save_model(
    market: str,
    model: Any,
    scaler: Any,
    training_summary: Dict[str, Any],
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Save a model checkpoint under the complete TradingBot identity."""
    return _call_engine(
        "memory",
        "save_model",
        market,
        model,
        scaler,
        training_summary,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def train_market(
    market: str,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Delegate a model-training request to the internal model engine."""
    return _call_engine(
        "model",
        "train_market",
        market,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def show_model_info(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> None:
    """Delegate model inspection to the internal model engine."""
    _call_engine(
        "model",
        "show_model_info",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


# ============================================================================
# LAZY COMPATIBILITY EXPORTS
# ============================================================================

def __getattr__(name: str) -> Any:
    """Resolve internal symbols only when an application requests them."""
    if name in _ENGINE_EXPORTS:
        return _get_engine_export(name)
    raise AttributeError(f"module 'ai' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose facade and delegated API names to IDEs and interactive tools."""
    return sorted(set(globals()) | set(_ENGINE_EXPORTS))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def build_parser() -> Any:
    """Return the parser provided by the internal model engine."""
    return _call_engine("model", "build_parser")


def main() -> int:
    """Run the internal model engine command-line interface."""
    return int(_call_engine("model", "main"))


__all__ = [
    "AIEngineNotReadyError",
    "ClientIdentity",
    "BASE_DIR",
    "setup_path",
    "model_dir",
    "model_path",
    "load_learning_setups",
    "prepare_training_data",
    "load_all_market_candles",
    "market_state_at",
    "current_price_at",
    "atr_at",
    "predict_from_features",
    "predict_market",
    "prepare_and_store_training_data",
    "load_model",
    "save_model",
    "train_market",
    "show_model_info",
    "build_parser",
    "main",
] + list(_ENGINE_EXPORTS)


if __name__ == "__main__":
    raise SystemExit(main())
