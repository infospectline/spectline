"""Identity and data-path management for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

WEB_MODE = os.getenv("WEB_MODE", "false").strip().lower() == "true"

DEFAULT_LOCAL_USER_ID = os.getenv("SPECTLINE_LOCAL_USER_ID", "LOCAL")
DEFAULT_LOCAL_STRATEGY_ID = os.getenv(
    "SPECTLINE_LOCAL_STRATEGY_ID",
    "DEFAULT_STRATEGY",
)
DEFAULT_LOCAL_TRADINGBOT_ID = os.getenv(
    "SPECTLINE_LOCAL_TRADINGBOT_ID",
    "DEFAULT_TRADINGBOT",
)

DEFAULT_MODEL_FILENAME = "model.pth"
DEFAULT_SETUP_FILENAME = "setups.json"
DEFAULT_CANDLE_FILENAME = "candles.csv"

SUPPORTED_TIMEFRAMES = frozenset({
    "M1",
    "M5",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1",
})

DATA_ROOT = Path(
    os.getenv(
        "SPECTLINE_DATA_ROOT",
        "/data/users" if WEB_MODE else str(BASE_DIR / "CUSTOMERS"),
    )
).expanduser().resolve()

CHARTS_ROOT = Path(
    os.getenv(
        "SPECTLINE_CHARTS_ROOT",
        str(BASE_DIR / "CHARTS"),
    )
).expanduser().resolve()


# ============================================================================
# VALIDATION AND NORMALIZATION
# ============================================================================

_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_path_component(value: Any, field_name: str) -> str:
    """Validate one identifier before it is used as a directory name."""
    if value is None:
        raise ValueError(f"{field_name} is required.")

    component = str(value).strip()

    if (
        not component
        or component in {".", ".."}
        or not _SAFE_COMPONENT_PATTERN.fullmatch(component)
    ):
        raise ValueError(f"Invalid {field_name}.")

    return component


def normalize_market(market: Any) -> str:
    """Normalize a market symbol into a safe, consistent folder name."""
    if market is None:
        raise ValueError("market is required.")

    normalized = (
        str(market)
        .strip()
        .upper()
        .replace("/", "_")
        .replace("\\", "_")
    )

    return _validate_path_component(normalized, "market")


def normalize_timeframe(timeframe: Any) -> str:
    """Validate and normalize a supported chart timeframe."""
    if timeframe is None:
        raise ValueError("timeframe is required.")

    normalized = str(timeframe).strip().upper()

    if normalized not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {normalized}")

    return normalized


# ============================================================================
# IDENTITY
# ============================================================================

@dataclass(frozen=True)
class ClientIdentity:
    """Trusted identity of one customer, market, strategy and TradingBot."""

    user_id: str
    market: str
    strategy_id: str
    tradingbot_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "user_id",
            _validate_path_component(self.user_id, "user_id"),
        )
        object.__setattr__(
            self,
            "market",
            normalize_market(self.market),
        )
        object.__setattr__(
            self,
            "strategy_id",
            _validate_path_component(self.strategy_id, "strategy_id"),
        )
        object.__setattr__(
            self,
            "tradingbot_id",
            _validate_path_component(self.tradingbot_id, "tradingbot_id"),
        )

    @property
    def key(self) -> Tuple[str, str, str, str]:
        """Return a stable, hashable key for runtime/session dictionaries."""
        return (
            self.user_id,
            self.market,
            self.strategy_id,
            self.tradingbot_id,
        )

    def as_dict(self) -> Dict[str, str]:
        """Return the identity fields for logs and model metadata."""
        return {
            "user_id": self.user_id,
            "market": self.market,
            "strategy_id": self.strategy_id,
            "tradingbot_id": self.tradingbot_id,
        }


def resolve_identity(
    market: str,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Build an identity from trusted backend/authentication values.

    In WEB_MODE, all identity fields must be supplied by the authenticated
    backend context. In local mode, stable defaults are used when the optional
    identity fields are not supplied.
    """
    if WEB_MODE:
        resolved_user_id = _validate_path_component(user_id, "user_id")
        resolved_strategy_id = _validate_path_component(
            strategy_id,
            "strategy_id",
        )
        resolved_tradingbot_id = _validate_path_component(
            tradingbot_id,
            "tradingbot_id",
        )
    else:
        resolved_user_id = _validate_path_component(
            user_id or DEFAULT_LOCAL_USER_ID,
            "user_id",
        )
        resolved_strategy_id = _validate_path_component(
            strategy_id or DEFAULT_LOCAL_STRATEGY_ID,
            "strategy_id",
        )
        resolved_tradingbot_id = _validate_path_component(
            tradingbot_id or DEFAULT_LOCAL_TRADINGBOT_ID,
            "tradingbot_id",
        )

    return ClientIdentity(
        user_id=resolved_user_id,
        market=market,
        strategy_id=resolved_strategy_id,
        tradingbot_id=resolved_tradingbot_id,
    )


def identity_from_verified_claims(
    claims: Dict[str, Any],
    market: str,
) -> ClientIdentity:
    """Build an identity from claims already verified by the auth layer.

    This function does not verify a ticket or token. Its caller must pass only
    claims returned by the trusted authentication layer.
    """
    if not isinstance(claims, dict):
        raise TypeError("Verified authentication claims must be a dictionary.")

    return resolve_identity(
        user_id=claims.get("sub"),
        market=market,
        strategy_id=claims.get("strategy_id"),
        tradingbot_id=claims.get("tradingbot_id"),
    )


# ============================================================================
# CUSTOMER PATHS
# ============================================================================

@dataclass(frozen=True)
class ClientPaths:
    """Resolved storage locations belonging to one TradingBot identity."""

    identity: ClientIdentity
    customer_root: Path
    market_root: Path
    strategy_root: Path
    tradingbot_root: Path
    learning_setups_file: Path
    models_root: Path
    results_root: Path
    runtime_root: Path

    def model_dir(self, version: Optional[str] = None) -> Path:
        """Return the model directory, optionally for one model version."""
        if version is None:
            return self.models_root

        version_folder = _validate_path_component(version, "model version")
        return self.models_root / version_folder

    def model_path(
        self,
        version: Optional[str] = None,
        filename: str = DEFAULT_MODEL_FILENAME,
    ) -> Path:
        """Return a model checkpoint path."""
        safe_filename = _validate_path_component(filename, "model filename")
        return self.model_dir(version) / safe_filename


def get_client_paths(identity: ClientIdentity) -> ClientPaths:
    """Resolve all private storage paths for one TradingBot identity."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    customer_root = DATA_ROOT / identity.user_id
    market_root = customer_root / identity.market
    strategy_root = market_root / identity.strategy_id
    tradingbot_root = strategy_root / identity.tradingbot_id

    return ClientPaths(
        identity=identity,
        customer_root=customer_root,
        market_root=market_root,
        strategy_root=strategy_root,
        tradingbot_root=tradingbot_root,
        learning_setups_file=tradingbot_root / DEFAULT_SETUP_FILENAME,
        models_root=tradingbot_root / "MODELS",
        results_root=tradingbot_root / "RESULTS",
        runtime_root=tradingbot_root / "RUNTIME",
    )


def ensure_client_directories(identity: ClientIdentity) -> ClientPaths:
    """Create the private directory tree for one TradingBot, if necessary."""
    paths = get_client_paths(identity)

    for directory in (
        paths.tradingbot_root,
        paths.models_root,
        paths.results_root,
        paths.runtime_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return paths


# ============================================================================
# SHARED MARKET DATA PATHS
# ============================================================================

def get_chart_history_path(
    market: str,
    timeframe: str,
    filename: str = DEFAULT_CANDLE_FILENAME,
) -> Path:
    """Return the shared, read-only historical chart archive path."""
    market_folder = normalize_market(market)
    timeframe_folder = normalize_timeframe(timeframe)
    safe_filename = _validate_path_component(filename, "chart filename")

    return CHARTS_ROOT / market_folder / timeframe_folder / safe_filename


# ============================================================================
# PUBLIC PATH HELPERS
# ============================================================================

def get_learning_setups_path(identity: ClientIdentity) -> Path:
    """Return the Learning setups file for one TradingBot."""
    return get_client_paths(identity).learning_setups_file


def get_model_dir(
    identity: ClientIdentity,
    version: Optional[str] = None,
) -> Path:
    """Return the model directory for one TradingBot."""
    return get_client_paths(identity).model_dir(version)


def get_model_path(
    identity: ClientIdentity,
    version: Optional[str] = None,
    filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """Return the model checkpoint path for one TradingBot."""
    return get_client_paths(identity).model_path(
        version=version,
        filename=filename,
    )


def get_results_dir(identity: ClientIdentity) -> Path:
    """Return the results directory for one TradingBot."""
    return get_client_paths(identity).results_root


def get_runtime_dir(identity: ClientIdentity) -> Path:
    """Return the runtime storage directory for one TradingBot."""
    return get_client_paths(identity).runtime_root


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BASE_DIR",
    "WEB_MODE",
    "DATA_ROOT",
    "CHARTS_ROOT",
    "SUPPORTED_TIMEFRAMES",
    "ClientIdentity",
    "ClientPaths",
    "resolve_identity",
    "identity_from_verified_claims",
    "get_client_paths",
    "ensure_client_directories",
    "get_chart_history_path",
    "get_learning_setups_path",
    "get_model_dir",
    "get_model_path",
    "get_results_dir",
    "get_runtime_dir",
    "normalize_market",
    "normalize_timeframe",
]
