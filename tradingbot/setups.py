"""Identity-isolated Learning setup storage and timeline adapters for Spectline."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import json
import math
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

SETUP_SCHEMA_VERSION = 2
VALID_RR_MODES = frozenset({"OFF", "2x", "3x"})
PROBABILITY_FIELD = "predicted_probability_at_entry"
OUTCOME_FIELD = "actual_outcome"


# ============================================================================
# ERRORS
# ============================================================================

class SetupStorageError(RuntimeError):
    """Raised when Learning setup data cannot be safely read or written."""


class SetupValidationError(SetupStorageError, ValueError):
    """Raised when a Learning setup contains invalid identity or trade data."""


# ============================================================================
# FILE LOCKS
# ============================================================================

_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path: Path) -> threading.RLock:
    """Return a process-local reentrant lock for one setup archive."""
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
    """Resolve the exact customer, market, strategy and TradingBot identity."""
    identity_fields = (user_id, strategy_id, tradingbot_id)

    if isinstance(market_or_identity, ClientIdentity):
        if any(value is not None for value in identity_fields):
            raise TypeError(
                "Do not pass separate identity fields together with ClientIdentity."
            )
        return market_or_identity

    if not isinstance(market_or_identity, str) or not market_or_identity.strip():
        raise TypeError(
            "market_or_identity must be a non-empty market string or ClientIdentity."
        )

    return resolve_identity(
        market=market_or_identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def identity_to_dict(identity: ClientIdentity) -> Dict[str, str]:
    """Return the four identity fields used to isolate customer Learning data."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    values = {
        "user_id": getattr(identity, "user_id", None),
        "market": getattr(identity, "market", None),
        "strategy_id": getattr(identity, "strategy_id", None),
        "tradingbot_id": getattr(identity, "tradingbot_id", None),
    }
    missing = [key for key, value in values.items() if value is None or not str(value).strip()]
    if missing:
        raise SetupValidationError(
            "TradingBot identity is incomplete: " + ", ".join(missing)
        )
    return {key: str(value) for key, value in values.items()}


def setup_file_path(
    identity: ClientIdentity,
    create_parent: bool = False,
) -> Path:
    """Return the Learning setup JSON path for one exact TradingBot identity."""
    identity_to_dict(identity)
    path = Path(get_learning_setups_path(identity)).expanduser().resolve()

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def _trading_setup_path(
    symbol: str,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> str:
    """Return the legacy-compatible path for one resolved TradingBot."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return str(setup_file_path(identity, create_parent=True))


# ============================================================================
# JSON VALIDATION AND STORAGE
# ============================================================================

def _validate_setup_list(setups: Any) -> List[Dict[str, Any]]:
    """Validate and normalize the complete market setup archive."""
    if not isinstance(setups, list):
        raise SetupValidationError("Learning setup archive must contain a JSON list.")

    copied: List[Dict[str, Any]] = []

    obsolete_fields = (
        "setup_id",
        "trade_id",
        "trade_record_id",
        "runtime_trade_id",
        "exit_time",
        "outcome_known_at",
        "outcome_source",
    )

    for index, item in enumerate(setups, start=1):
        if not isinstance(item, dict):
            raise SetupValidationError(
                f"Learning setup at index {index} must be a JSON object."
            )

        record = copy.deepcopy(item)

        for field_name in obsolete_fields:
            record.pop(field_name, None)

        record["id"] = index

        raw_outcome = record.get(OUTCOME_FIELD)

        if raw_outcome is None or str(raw_outcome).strip() == "":
            record[OUTCOME_FIELD] = None
        elif isinstance(raw_outcome, bool):
            raise SetupValidationError(
                f"Invalid actual_outcome in setup {index}."
            )
        else:
            outcome = str(raw_outcome).strip().upper()

            if outcome in {"1", "1.0"}:
                outcome = "TP"
            elif outcome in {"0", "0.0"}:
                outcome = "SL"

            if outcome not in {"TP", "SL", "AMBIGUOUS"}:
                raise SetupValidationError(
                    f"Invalid actual_outcome in setup {index}: {raw_outcome}"
                )

            record[OUTCOME_FIELD] = outcome

        copied.append(record)

    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SetupValidationError(
            f"Learning setup data is not valid JSON: {error}"
        ) from error

    return copied


def _read_setup_file(path: Path) -> List[Dict[str, Any]]:
    """Read one setup archive without hiding malformed or corrupted data."""
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise SetupStorageError(
            f"Learning setup JSON is malformed at line {error.lineno}, "
            f"column {error.colno}: {path}"
        ) from error
    except OSError as error:
        raise SetupStorageError(
            f"Could not read Learning setups from {path}: {error}"
        ) from error

    return _validate_setup_list(data)


def _write_setup_file(
    path: Path,
    setups: List[Dict[str, Any]],
) -> None:
    """Atomically replace one JSON archive after validating all records."""
    normalized_setups = _validate_setup_list(setups)
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
                normalized_setups,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None

        try:
            directory_fd = os.open(str(path.parent), getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (AttributeError, OSError):
            pass

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
    """Replace the complete setup archive for one exact TradingBot."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    normalized_setups = _validate_setup_list(setups)
    path = setup_file_path(identity, create_parent=True)

    with _get_file_lock(path):
        _write_setup_file(path, normalized_setups)


# ============================================================================
# SETUP ID MANAGEMENT
# ============================================================================

def _parse_setup_id(value: Any) -> Optional[int]:
    """Convert a positive integer setup ID without accepting booleans or fractions."""
    if isinstance(value, bool) or value is None:
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        numeric = float(parsed)

    if not math.isfinite(numeric) or numeric != float(parsed) or parsed <= 0:
        return None
    return parsed


def _next_trading_setup_id(setups: List[Dict[str, Any]]) -> int:
    """Return the next positive numeric setup ID for one TradingBot."""
    identifiers = [
        parsed
        for item in setups
        if isinstance(item, dict)
        for parsed in [_parse_setup_id(item.get("id"))]
        if parsed is not None
    ]
    return max(identifiers, default=0) + 1


def _find_setup_index(setups: List[Dict[str, Any]], setup_id: int) -> Optional[int]:
    """Find the archive index of a setup with the requested numeric ID."""
    for index, item in enumerate(setups):
        if _parse_setup_id(item.get("id")) == setup_id:
            return index
    return None


# ============================================================================
# LEARNING SETUP OPERATIONS
# ============================================================================

def load_learning_setups(identity: ClientIdentity) -> List[Dict[str, Any]]:
    """Load the Learning trades belonging to one exact TradingBot identity."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    return _load_trading_setups(identity)


def get_trading_setups(
    symbol: IdentityInput,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all saved Learning setups using the legacy backend call signature."""
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
    """Create or update one Learning setup for an exact TradingBot identity."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    return save_trading_setup(identity, setup)


def save_trading_setup(
    symbol: IdentityInput,
    setup: Dict[str, Any],
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update one setup while preserving fields omitted by partial updates."""
    if not isinstance(setup, dict):
        raise TypeError("setup must be a dictionary.")

    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = setup_file_path(identity, create_parent=True)
    incoming = copy.deepcopy(setup)

    obsolete_fields = (
        "setup_id",
        "trade_id",
        "trade_record_id",
        "runtime_trade_id",
        "exit_time",
        "outcome_known_at",
        "outcome_source",
    )

    for field_name in obsolete_fields:
        incoming.pop(field_name, None)

    requested_id = _parse_setup_id(incoming.get("id"))

    with _get_file_lock(path):
        setups = _read_setup_file(path)

        existing_index = (
            _find_setup_index(setups, requested_id)
            if requested_id is not None
            else None
        )

        if existing_index is None:
            incoming["id"] = _next_trading_setup_id(setups)
            merged = incoming
            setups.append(merged)
        else:
            incoming["id"] = requested_id
            merged = copy.deepcopy(setups[existing_index])
            merged.update(incoming)
            merged["id"] = requested_id
            setups[existing_index] = merged

        rr_mode = merged.get("rr_mode", "OFF")
        if rr_mode not in VALID_RR_MODES:
            merged["rr_mode"] = "OFF"

        _write_setup_file(path, setups)

    return copy.deepcopy(merged)


def delete_learning_setup(identity: ClientIdentity, setup_id: int) -> bool:
    """Delete one Learning setup without affecting other TradingBot archives."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    return delete_trading_setup(identity, setup_id)


def delete_trading_setup(
    symbol: IdentityInput,
    setup_id: int,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> bool:
    """Delete one setup from the archive belonging to the resolved identity."""
    identity = _resolve_setup_identity(
        symbol,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    parsed_setup_id = _parse_setup_id(setup_id)
    if parsed_setup_id is None:
        return False

    path = setup_file_path(identity)
    with _get_file_lock(path):
        setups = _read_setup_file(path)
        existing_index = _find_setup_index(setups, parsed_setup_id)
        if existing_index is None:
            return False
        del setups[existing_index]
        _write_setup_file(path, setups)
    return True


# ============================================================================
# VALUE AND TIMESTAMP NORMALIZATION
# ============================================================================

def _manual_parse_timestamp(value: Any) -> Optional[int]:
    """Convert supported Unix, millisecond and ISO-8601 timestamps to UTC seconds."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
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
    except (TypeError, ValueError, OverflowError):
        pass

    normalized = text.replace(" UTC", "+00:00")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.astimezone(timezone.utc).timestamp())
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _finite_number(value: Any) -> Optional[float]:
    """Return a finite float for a valid numeric setup value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _manual_setup_value(
    setup: Dict[str, Any],
    key: str,
    geometry_key: str,
) -> Optional[float]:
    """Read a numeric trade price from a direct field or normalized geometry."""
    if not isinstance(setup, dict):
        return None

    direct_value = _finite_number(setup.get(key))
    if direct_value is not None:
        return direct_value

    geometry = setup.get("trade_geometry")
    if not isinstance(geometry, dict):
        return None

    point = geometry.get(geometry_key)
    if not isinstance(point, dict):
        return None

    return _finite_number(point.get("price"))


def _first_timestamp(setup: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    """Return the first valid timestamp found in an ordered list of fields."""
    for key in keys:
        timestamp = _manual_parse_timestamp(setup.get(key))
        if timestamp is not None:
            return timestamp
    return None


def _manual_setup_time(setup: Dict[str, Any]) -> Optional[int]:
    """Resolve the actual setup entry time from supported legacy and current fields."""
    if not isinstance(setup, dict):
        return None

    direct_time = _first_timestamp(
        setup,
        ("entry_candle", "entry_candle_time", "entry_time", "opened_at"),
    )
    if direct_time is not None:
        return direct_time

    geometry = setup.get("trade_geometry")
    if isinstance(geometry, dict):
        entry_point = geometry.get("entry")
        if isinstance(entry_point, dict):
            for field_name in ("time", "timestamp", "candle_time"):
                timestamp = _manual_parse_timestamp(entry_point.get(field_name))
                if timestamp is not None:
                    return timestamp

    return _first_timestamp(setup, ("breakout_time",))


def _manual_setup_end_time(setup: Dict[str, Any]) -> Optional[int]:
    """Read the actual trade end time from the main setup record."""
    if not isinstance(setup, dict):
        return None

    return _first_timestamp(setup, ("end_time",))


def _manual_setup_prediction_time(setup: Dict[str, Any]) -> Optional[int]:
    """Return the explicitly recorded prediction timestamp, if available."""
    if not isinstance(setup, dict):
        return None
    return _first_timestamp(setup, ("predicted_at", "prediction_time"))


def _manual_setup_probability(setup: Dict[str, Any]) -> Optional[float]:
    """Read entry probability as a fraction while accepting percentage input."""
    if not isinstance(setup, dict):
        return None
    value = setup.get(PROBABILITY_FIELD)
    if value is None:
        value = setup.get("probability_percent")
        if value is not None:
            number = _finite_number(value)
            if number is None or not 0.0 <= number <= 100.0:
                return None
            return number / 100.0
    number = _finite_number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 1.0 < number <= 100.0:
        return number / 100.0
    return None


def _manual_setup_outcome(setup: Dict[str, Any]) -> Optional[str]:
    """Read the finalized trade outcome from the setup archive."""
    if not isinstance(setup, dict):
        return None

    value = setup.get(OUTCOME_FIELD)

    if value is None or str(value).strip() == "":
        return None

    if isinstance(value, bool):
        raise SetupValidationError("Invalid actual_outcome value.")

    outcome = str(value).strip().upper()

    if outcome in {"1", "1.0"}:
        outcome = "TP"
    elif outcome in {"0", "0.0"}:
        outcome = "SL"

    if outcome not in {"TP", "SL", "AMBIGUOUS"}:
        raise SetupValidationError(
            f"Invalid actual_outcome value: {value}"
        )

    return outcome

# ============================================================================
# TIMELINE ADAPTERS
# ============================================================================

def setup_to_timeline_record(setup: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one saved Learning setup into the timeline setup-record format."""
    if not isinstance(setup, dict):
        raise TypeError("setup must be a dictionary.")

    entry_time = _manual_setup_time(setup)
    if entry_time is None:
        raise SetupValidationError(
            "A setup needs a valid entry timestamp before it can enter the timeline."
        )

    record = copy.deepcopy(setup)

    current_id = _parse_setup_id(record.get("id"))
    if current_id is None:
        raise SetupValidationError(
            "A saved setup must contain a valid dynamic id."
        )

    record["id"] = current_id

    obsolete_fields = (
        "setup_id",
        "trade_id",
        "trade_record_id",
        "runtime_trade_id",
        "exit_time",
        "outcome_known_at",
        "outcome_source",
    )

    for field_name in obsolete_fields:
        record.pop(field_name, None)

    record["entry_time"] = entry_time
    record["entry_candle_time"] = entry_time

    prediction_time = _manual_setup_prediction_time(record)
    if prediction_time is not None:
        if prediction_time > entry_time:
            raise SetupValidationError(
                "predicted_at cannot be later than the setup entry time."
            )
        record["predicted_at"] = prediction_time

    probability = _manual_setup_probability(record)
    if probability is not None:
        record[PROBABILITY_FIELD] = probability

    end_time = _manual_setup_end_time(record)
    if end_time is not None:
        if end_time < entry_time:
            raise SetupValidationError(
                "Trade end time cannot precede entry time."
            )
        record["end_time"] = end_time

    outcome = _manual_setup_outcome(record)
    record[OUTCOME_FIELD] = outcome

    if outcome is not None and end_time is None:
        raise SetupValidationError(
            "A finalized trade outcome requires end_time."
        )

    return record

def setups_to_timeline_records(
    setups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert valid saved setups into timeline records in entry-time order."""
    if not isinstance(setups, list):
        raise TypeError("setups must be a list.")

    records = [setup_to_timeline_record(setup) for setup in setups]
    records.sort(
        key=lambda item: (
            int(item["entry_time"]),
            int(item["id"]),
        )
    )
    return records


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "SETUP_SCHEMA_VERSION",
    "VALID_RR_MODES",
    "PROBABILITY_FIELD",
    "OUTCOME_FIELD",
    "SetupStorageError",
    "SetupValidationError",
    "identity_to_dict",
    "setup_file_path",
    "load_learning_setups",
    "get_trading_setups",
    "save_learning_setup",
    "save_trading_setup",
    "delete_learning_setup",
    "delete_trading_setup",
    "setup_to_timeline_record",
    "setups_to_timeline_records",
    "_trading_setup_path",
    "_load_trading_setups",
    "_save_trading_setups",
    "_next_trading_setup_id",
    "_parse_setup_id",
    "_manual_parse_timestamp",
    "_manual_setup_value",
    "_manual_setup_time",
    "_manual_setup_end_time",
    "_manual_setup_prediction_time",
    "_manual_setup_probability",
    "_manual_setup_outcome",
]
