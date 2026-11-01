"""Learning setup storage and helper utilities for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from .client import (
        ClientIdentity,
        get_learning_setups_path,
        resolve_identity,
    )
except ImportError:
    from client import (
        ClientIdentity,
        get_learning_setups_path,
        resolve_identity,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

VALID_RR_MODES = frozenset({"OFF", "2x", "3x"})


# ============================================================================
# ERRORS
# ============================================================================

class SetupStorageError(RuntimeError):
    """Raised when Learning setup data cannot be safely read or written."""


# ============================================================================
# FILE LOCKS
# ============================================================================

_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path: Path) -> threading.RLock:
    """Return a process-local lock for one Learning setup file."""
    key = str(path.resolve())

    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock

    return lock


# ============================================================================
# IDENTITY AND PATH RESOLUTION
# ============================================================================

IdentityInput = Union[ClientIdentity, str]


def _resolve_setup_identity(
    market_or_identity: IdentityInput,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Resolve the trusted owner identity for one Learning setup file."""
    if isinstance(market_or_identity, ClientIdentity):
        if any(value is not None for value in (user_id, strategy_id, tradingbot_id)):
            raise TypeError(
                "Do not pass separate identity fields together with ClientIdentity."
            )
        return market_or_identity

    if not isinstance(market_or_identity, str):
        raise TypeError("market_or_identity must be a market string or ClientIdentity.")

    return resolve_identity(
        market=market_or_identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def setup_file_path(
    identity: ClientIdentity,
    create_parent: bool = False,
) -> Path:
    """Return the Learning setup file path for one TradingBot identity."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    path = get_learning_setups_path(identity)

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def _trading_setup_path(
    symbol: str,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> str:
    """Return the legacy-compatible path for the resolved TradingBot."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return str(setup_file_path(identity, create_parent=True))


# ============================================================================
# JSON STORAGE
# ============================================================================

def _read_setup_file(path: Path) -> List[Dict[str, Any]]:
    """Read one setup file without hiding malformed or corrupted data."""
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise SetupStorageError(
            f"Could not read Learning setups from {path}: {error}"
        ) from error

    if not isinstance(data, list):
        raise SetupStorageError(
            f"Learning setup file must contain a JSON list: {path}"
        )

    invalid_indexes = [
        index
        for index, item in enumerate(data)
        if not isinstance(item, dict)
    ]
    if invalid_indexes:
        raise SetupStorageError(
            "Learning setup file contains non-object entries at indexes: "
            + ", ".join(str(index) for index in invalid_indexes)
        )

    return data


def _write_setup_file(
    path: Path,
    setups: List[Dict[str, Any]],
) -> None:
    """Write setup data atomically so an interrupted save cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None

    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary_path = Path(temporary_name)

        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(
                setups,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

        raise SetupStorageError(
            f"Could not save Learning setups to {path}: {error}"
        ) from error


def _load_trading_setups(
    symbol: IdentityInput,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load Learning setups for one customer, market, strategy and bot."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = setup_file_path(identity)

    with _get_file_lock(path):
        return copy.deepcopy(_read_setup_file(path))


def _save_trading_setups(
    symbol: IdentityInput,
    setups: List[Dict[str, Any]],
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> None:
    """Save the complete Learning setup list for one TradingBot."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )

    if not isinstance(setups, list):
        raise TypeError("setups must be a list.")

    copied_setups = copy.deepcopy(setups)
    if any(not isinstance(item, dict) for item in copied_setups):
        raise TypeError("Every Learning setup must be a dictionary.")

    path = setup_file_path(identity, create_parent=True)

    with _get_file_lock(path):
        _write_setup_file(path, copied_setups)


# ============================================================================
# SETUP ID MANAGEMENT
# ============================================================================

def _next_trading_setup_id(setups: List[Dict[str, Any]]) -> int:
    """Return the next positive numeric setup ID."""
    identifiers: List[int] = []

    for setup in setups:
        try:
            identifiers.append(int(setup.get("id")))
        except (TypeError, ValueError, OverflowError):
            continue

    return max(identifiers, default=0) + 1


def _parse_setup_id(value: Any) -> Optional[int]:
    """Convert a setup ID to an integer without accepting booleans."""
    if isinstance(value, bool) or value is None:
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return parsed if parsed > 0 else None


# ============================================================================
# LEARNING SETUP OPERATIONS
# ============================================================================

def load_learning_setups(
    identity: ClientIdentity,
) -> List[Dict[str, Any]]:
    """Load the Learning trades belonging to one TradingBot identity."""
    return _load_trading_setups(identity)


def get_trading_setups(
    symbol: IdentityInput,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all saved Learning setups for one TradingBot.

    The string-based arguments preserve the previous backend calling style.
    In WEB_MODE, user_id, strategy_id and tradingbot_id must all be supplied
    by the authenticated backend context.
    """
    return _load_trading_setups(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def save_learning_setup(
    identity: ClientIdentity,
    setup: Dict[str, Any],
) -> Dict[str, Any]:
    """Create or update one Learning setup for a TradingBot identity."""
    return save_trading_setup(identity, setup)


def save_trading_setup(
    symbol: IdentityInput,
    setup: Dict[str, Any],
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a Learning setup for one TradingBot.

    Existing setup fields are preserved. The RR mode is normalized using the
    same allowed values as the previous backend implementation.
    """
    if not isinstance(setup, dict):
        raise TypeError("setup must be a dictionary.")

    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = setup_file_path(identity, create_parent=True)
    setup_copy = copy.deepcopy(setup)

    rr_mode = setup_copy.get("rr_mode", "OFF")
    if rr_mode not in VALID_RR_MODES:
        rr_mode = "OFF"
    setup_copy["rr_mode"] = rr_mode

    requested_id = _parse_setup_id(setup_copy.get("id"))

    with _get_file_lock(path):
        setups = _read_setup_file(path)

        if requested_id is None:
            requested_id = _next_trading_setup_id(setups)
            setup_copy["id"] = requested_id
            setups.append(setup_copy)
        else:
            setup_copy["id"] = requested_id
            existing_index = next(
                (
                    index
                    for index, existing in enumerate(setups)
                    if _parse_setup_id(existing.get("id")) == requested_id
                ),
                None,
            )

            if existing_index is None:
                setups.append(setup_copy)
            else:
                setups[existing_index] = setup_copy

        _write_setup_file(path, setups)

    return copy.deepcopy(setup_copy)


def delete_learning_setup(
    identity: ClientIdentity,
    setup_id: int,
) -> bool:
    """Delete one Learning setup from a TradingBot identity."""
    return delete_trading_setup(identity, setup_id)


def delete_trading_setup(
    symbol: IdentityInput,
    setup_id: int,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> bool:
    """Delete one Learning setup without affecting other TradingBots."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = setup_file_path(identity)
    parsed_setup_id = _parse_setup_id(setup_id)

    if parsed_setup_id is None:
        return False

    with _get_file_lock(path):
        setups = _read_setup_file(path)
        remaining = [
            setup
            for setup in setups
            if _parse_setup_id(setup.get("id")) != parsed_setup_id
        ]

        deleted = len(remaining) != len(setups)
        if deleted:
            _write_setup_file(path, remaining)

    return deleted


# ============================================================================
# SETUP TIMESTAMP AND GEOMETRY HELPERS
# ============================================================================

def _manual_parse_timestamp(value: Any) -> Optional[int]:
    """Convert supported setup timestamp formats into Unix seconds."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not (float("-inf") < number < float("inf")):
            return None

        if abs(number) > 10_000_000_000:
            number /= 1000.0

        return int(number)

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text.replace(",", ""))
        if float("-inf") < number < float("inf"):
            if abs(number) > 10_000_000_000:
                number /= 1000.0
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass

    normalized = text.replace(" UTC", "+00:00").replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except (TypeError, ValueError, OverflowError):
        return None


def _manual_setup_value(
    setup: Dict[str, Any],
    key: str,
    geometry_key: str,
) -> Optional[float]:
    """Read a setup price from its direct field or trade geometry."""
    value = setup.get(key)

    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        number = None

    if number is not None and number == number and abs(number) != float("inf"):
        return number

    geometry = setup.get("trade_geometry") or {}
    if not isinstance(geometry, dict):
        return None

    point = geometry.get(geometry_key) or {}
    if not isinstance(point, dict):
        return None

    value = point.get("price")
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None

    if number is None or number != number or abs(number) == float("inf"):
        return None

    return number


def _manual_setup_time(setup: Dict[str, Any]) -> Optional[int]:
    """Resolve a setup entry time from its supported timestamp fields."""
    entry_candle = _manual_parse_timestamp(setup.get("entry_candle"))
    if entry_candle is not None:
        return entry_candle

    geometry = setup.get("trade_geometry") or {}
    if isinstance(geometry, dict):
        entry_point = geometry.get("entry") or {}
        if isinstance(entry_point, dict):
            geometry_time = _manual_parse_timestamp(entry_point.get("time"))
            if geometry_time is not None:
                return geometry_time

    return _manual_parse_timestamp(setup.get("breakout_time"))


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "SetupStorageError",
    "setup_file_path",
    "load_learning_setups",
    "get_trading_setups",
    "save_learning_setup",
    "save_trading_setup",
    "delete_learning_setup",
    "delete_trading_setup",
    "_trading_setup_path",
    "_load_trading_setups",
    "_save_trading_setups",
    "_next_trading_setup_id",
    "_manual_parse_timestamp",
    "_manual_setup_value",
    "_manual_setup_time",
]
