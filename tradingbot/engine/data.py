"""Learning setup loading, validation and Training Data Bundle assembly."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import importlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from client import (
        ClientIdentity,
        get_learning_setups_path,
        normalize_market,
        resolve_identity,
    )
    from engine.settings import (
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )
except ImportError:
    from ..client import (
        ClientIdentity,
        get_learning_setups_path,
        normalize_market,
        resolve_identity,
    )
    from .settings import (
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

_SCHEMA_VERSION = 1
_SAFE_FOLDER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALID_ZONE_TYPES = {"SUPPORT", "RESISTANCE"}

# This is the optional setup field reserved for lower-timeframe TP/SL resolution.
# It is ignored when absent and validated when supplied.
RESOLUTION_TIMEFRAME_FIELD = "resolution_timeframe"


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass(frozen=True)
class Candle:
    """Normalized OHLCV candle used by the history and feature modules."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ManualZone:
    """One user-drawn Support or Resistance zone from a Learning setup."""

    setup_id: int
    zone_id: Optional[Any]
    zone_type: str
    start_time: Optional[int]
    end_time: Optional[int]
    low: Optional[float]
    high: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class UserSetup:
    """Normalized representation of one complete user Learning setup."""

    setup_id: int
    direction: str
    setup_timeframe: str
    entry_timeframe: str
    anchor_time: Optional[int]
    anchor_price: Optional[float]
    breakout_time: Optional[int]
    breakout: Optional[float]
    entry_candle_time: Optional[int]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    rr_mode: Optional[str]
    resolution_timeframe: Optional[str]
    zones: List[ManualZone]
    trade_geometry: Dict[str, Any]
    raw: Dict[str, Any]
    validation_errors: List[str] = field(default_factory=list)

    @property
    def decision_time(self) -> Optional[int]:
        """Return the recorded entry time used for point-in-time analysis."""
        return self.entry_candle_time

    @property
    def is_valid(self) -> bool:
        """Return whether the setup contains the fields needed for alignment."""
        return not self.validation_errors


# ============================================================================
# IDENTITY AND PATH HELPERS
# ============================================================================

def safe_market_folder(market: str) -> str:
    """Return the normalized market folder name."""
    return normalize_market(market)


def safe_user_folder(user_id: Optional[str]) -> str:
    """Validate a customer identifier before it is used in a path."""
    if user_id is None:
        raise ValueError("user_id is required in WebMode.")

    value = str(user_id).strip()
    if not value or not _SAFE_FOLDER_PATTERN.fullmatch(value):
        raise ValueError("Invalid user_id.")
    return value


def _resolve_ai_identity(
    market: str,
    identity: Optional[ClientIdentity] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Resolve and validate the complete customer and TradingBot identity."""
    normalized_market = normalize_market(market)

    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        if identity.market != normalized_market:
            raise ValueError(
                f"Identity market {identity.market} does not match "
                f"requested market {normalized_market}."
            )
        return identity

    return resolve_identity(
        market=normalized_market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def setup_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the Learning setup JSON path for one exact TradingBot identity."""
    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return get_learning_setups_path(resolved)


# ============================================================================
# VALUE AND TIMESTAMP NORMALIZATION
# ============================================================================

def to_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float, returning None for invalid values."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None
    return number


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert Unix, millisecond or ISO-8601 timestamps into Unix seconds."""
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
        return int(parsed.timestamp())
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _strip_legacy_focus_fields(value: Any) -> Any:
    """Remove retired Focus 1/2 fields from an in-memory setup copy."""
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key.startswith(("focus1", "focus2")):
                continue
            cleaned[key] = _strip_legacy_focus_fields(child)
        return cleaned

    if isinstance(value, list):
        return [_strip_legacy_focus_fields(item) for item in value]

    return copy.deepcopy(value)


def _normalize_timeframe(value: Any, field_name: str, errors: List[str]) -> str:
    """Normalize a setup timeframe and record unsupported values as errors."""
    normalized = str(value or "").strip().upper()
    if not normalized:
        errors.append(f"Missing {field_name}.")
        return ""

    if normalized not in SUPPORTED_TIMEFRAMES:
        errors.append(
            f"Unsupported {field_name} '{normalized}' for the current runtime mode."
        )
    return normalized


# ============================================================================
# LEARNING SETUP NORMALIZATION
# ============================================================================

def _geometry_item(
    trade_geometry: Dict[str, Any],
    key: str,
) -> Dict[str, Any]:
    """Return one named trade-geometry object when it is a dictionary."""
    value = trade_geometry.get(key)
    return value if isinstance(value, dict) else {}


def _first_price(
    item: Dict[str, Any],
    geometry: Dict[str, Any],
    field_name: str,
) -> Optional[float]:
    """Read a price from the setup root, falling back to its geometry object."""
    value = to_float(item.get(field_name))
    if value is not None:
        return value
    return to_float(geometry.get("price"))


def _normalize_manual_zone(
    setup_id: int,
    raw_zone: Dict[str, Any],
    zone_index: int,
) -> ManualZone:
    """Normalize one user-drawn zone without limiting the number of zones."""
    errors: List[str] = []
    zone_type = str(raw_zone.get("type", "")).strip().upper()

    if zone_type not in _VALID_ZONE_TYPES:
        errors.append(
            f"Zone {zone_index} has unsupported type '{zone_type or 'EMPTY'}'."
        )

    geometry = raw_zone.get("geometry")
    if not isinstance(geometry, dict):
        geometry = raw_zone

    start_time = parse_timestamp(geometry.get("start_time"))
    end_time = parse_timestamp(geometry.get("end_time"))
    low = to_float(geometry.get("low"))
    high = to_float(geometry.get("high"))

    if low is None or high is None:
        errors.append(f"Zone {zone_index} is missing valid low/high prices.")
    if start_time is None or end_time is None:
        errors.append(f"Zone {zone_index} is missing valid start/end times.")
    elif low > high:
        errors.append(f"Zone {zone_index} has low greater than high.")

    if start_time is not None and end_time is not None and start_time > end_time:
        errors.append(f"Zone {zone_index} has start_time after end_time.")

    return ManualZone(
        setup_id=setup_id,
        zone_id=raw_zone.get("id", raw_zone.get("number", zone_index)),
        zone_type=zone_type,
        start_time=start_time,
        end_time=end_time,
        low=low,
        high=high,
        raw=_strip_legacy_focus_fields(raw_zone),
        validation_errors=errors,
    )


def normalize_learning_setup(
    item: Dict[str, Any],
    index: int,
) -> UserSetup:
    """Convert one raw JSON setup into a normalized, lossless setup record."""
    if not isinstance(item, dict):
        raise TypeError(f"Learning setup at index {index} must be an object.")

    errors: List[str] = []

    try:
        setup_id = int(item.get("id", index))
    except (TypeError, ValueError, OverflowError):
        setup_id = index
        errors.append(f"Setup at index {index} has an invalid id.")

    direction = str(item.get("direction", "")).strip().upper()
    if direction in {"BUY", "LONG"}:
        direction = "LONG"
    elif direction in {"SELL", "SHORT"}:
        direction = "SHORT"
    else:
        errors.append(
            f"Setup {setup_id} has an unsupported direction '{direction or 'EMPTY'}'."
        )

    setup_timeframe = _normalize_timeframe(
        item.get("setup_timeframe"),
        "setup_timeframe",
        errors,
    )
    entry_timeframe = _normalize_timeframe(
        item.get("entry_timeframe"),
        "entry_timeframe",
        errors,
    )

    anchor = item.get("anchor")
    if not isinstance(anchor, dict):
        anchor = {}

    trade_geometry_value = item.get("trade_geometry")
    trade_geometry = (
        trade_geometry_value if isinstance(trade_geometry_value, dict) else {}
    )

    breakout_geometry = _geometry_item(trade_geometry, "breakout")
    entry_geometry = _geometry_item(trade_geometry, "entry")
    sl_geometry = _geometry_item(trade_geometry, "sl")
    tp_geometry = _geometry_item(trade_geometry, "tp")

    entry_candle_time = parse_timestamp(item.get("entry_candle"))
    if entry_candle_time is None:
        entry_candle_time = parse_timestamp(entry_geometry.get("time"))

    if entry_candle_time is None:
        errors.append(
            f"Setup {setup_id} is missing a valid entry timestamp. "
            "Anchor time is not used as a substitute for entry time."
        )

    breakout_time = parse_timestamp(breakout_geometry.get("time"))
    breakout = _first_price(item, breakout_geometry, "breakout")
    entry = _first_price(item, entry_geometry, "entry")
    sl = _first_price(item, sl_geometry, "sl")
    tp = _first_price(item, tp_geometry, "tp")

    for field_name, field_value in (("entry", entry), ("sl", sl), ("tp", tp)):
        if field_value is None:
            errors.append(f"Setup {setup_id} is missing a valid {field_name} price.")

    zones_value = item.get("zones", [])
    if zones_value is None:
        zones_value = []
    if not isinstance(zones_value, list):
        errors.append(f"Setup {setup_id} has a zones value that is not a list.")
        zones_value = []

    zones: List[ManualZone] = []
    for zone_index, raw_zone in enumerate(zones_value, start=1):
        if not isinstance(raw_zone, dict):
            errors.append(
                f"Setup {setup_id} zone {zone_index} is not a JSON object."
            )
            continue
        zones.append(
            _normalize_manual_zone(setup_id, raw_zone, zone_index)
        )

    resolution_value = item.get(RESOLUTION_TIMEFRAME_FIELD)
    if resolution_value is None:
        resolution_value = trade_geometry.get(RESOLUTION_TIMEFRAME_FIELD)

    resolution_timeframe: Optional[str] = None
    if resolution_value not in (None, ""):
        resolution_timeframe = str(resolution_value).strip().upper()
        if resolution_timeframe not in SUPPORTED_TIMEFRAMES:
            errors.append(
                f"Setup {setup_id} has unsupported {RESOLUTION_TIMEFRAME_FIELD} "
                f"'{resolution_timeframe}'."
            )

    for zone_index, zone in enumerate(zones, start=1):
        errors.extend(
            f"Setup {setup_id}: {message}"
            for message in zone.validation_errors
        )

    sanitized_item = _strip_legacy_focus_fields(item)
    sanitized_trade_geometry = sanitized_item.get("trade_geometry", {})
    if not isinstance(sanitized_trade_geometry, dict):
        sanitized_trade_geometry = {}

    return UserSetup(
        setup_id=setup_id,
        direction=direction,
        setup_timeframe=setup_timeframe,
        entry_timeframe=entry_timeframe,
        anchor_time=parse_timestamp(anchor.get("time")),
        anchor_price=to_float(anchor.get("price")),
        breakout_time=breakout_time,
        breakout=breakout,
        entry_candle_time=entry_candle_time,
        entry=entry,
        sl=sl,
        tp=tp,
        rr_mode=(
            str(item.get("rr_mode")).strip()
            if item.get("rr_mode") is not None
            else None
        ),
        resolution_timeframe=resolution_timeframe,
        zones=zones,
        trade_geometry=copy.deepcopy(sanitized_trade_geometry),
        raw=sanitized_item,
        validation_errors=errors,
    )


def setup_time(setup: UserSetup) -> Optional[int]:
    """Return the recorded entry timestamp for one setup."""
    return setup.entry_candle_time


# ============================================================================
# LEARNING SETUP FILE ACCESS
# ============================================================================

def load_learning_setup_records(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> List[Dict[str, Any]]:
    """Load the complete raw JSON list for one TradingBot without dropping fields."""
    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = get_learning_setups_path(resolved)

    if not path.is_file():
        raise FileNotFoundError(f"Learning setup file was not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Learning setup JSON is malformed at line {error.lineno}, "
            f"column {error.colno}: {path}"
        ) from error
    except OSError as error:
        raise OSError(f"Could not read Learning setups from {path}: {error}") from error

    if not isinstance(raw_data, list):
        raise ValueError(
            f"Learning setup file must contain a JSON list: {path}"
        )

    malformed_indexes = [
        index
        for index, item in enumerate(raw_data, start=1)
        if not isinstance(item, dict)
    ]
    if malformed_indexes:
        raise ValueError(
            "Learning setup file contains non-object entries at indexes: "
            + ", ".join(str(index) for index in malformed_indexes)
        )

    return _strip_legacy_focus_fields(raw_data)


def load_learning_setups(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> List[UserSetup]:
    """Load and normalize all Learning setups for one TradingBot identity."""
    raw_records = load_learning_setup_records(
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )

    setups = [
        normalize_learning_setup(item, index)
        for index, item in enumerate(raw_records, start=1)
    ]

    setups.sort(
        key=lambda setup: (
            setup.entry_candle_time is None,
            setup.entry_candle_time or 0,
            setup.setup_id,
        )
    )
    return setups


# ============================================================================
# SETUP MATCHING AND TIMEFRAME REQUIREMENTS
# ============================================================================

def closest_setup(
    setup_times: Sequence[int],
    target_time: int,
    tolerance: int,
) -> Optional[int]:
    """Return the index of the closest setup within the requested tolerance."""
    best_index: Optional[int] = None
    best_distance: Optional[int] = None

    for index, setup_timestamp in enumerate(setup_times):
        distance = abs(int(setup_timestamp) - int(target_time))
        if distance > int(tolerance):
            continue
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index


def collect_required_timeframes(
    setups: Sequence[UserSetup],
) -> List[str]:
    """Collect unique supported timeframes referenced by the supplied setups."""
    required = set()

    for setup in setups:
        for timeframe in (
            setup.setup_timeframe,
            setup.entry_timeframe,
            setup.resolution_timeframe,
        ):
            if timeframe and timeframe in SUPPORTED_TIMEFRAMES:
                required.add(timeframe)

    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in required]


def extract_manual_levels(
    setups: Sequence[UserSetup],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group every user-drawn Support and Resistance zone by setup ID."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}

    for setup in setups:
        levels: List[Dict[str, Any]] = []
        for zone in setup.zones:
            levels.append(
                {
                    "setup_id": setup.setup_id,
                    "zone_id": zone.zone_id,
                    "type": zone.zone_type,
                    "start_time": zone.start_time,
                    "end_time": zone.end_time,
                    "low": zone.low,
                    "high": zone.high,
                    "raw": copy.deepcopy(zone.raw),
                    "validation_errors": list(zone.validation_errors),
                }
            )
        grouped[setup.setup_id] = levels

    return grouped


# ============================================================================
# OPTIONAL ENGINE MODULE INTEGRATION
# ============================================================================

def _load_optional_engine_function(
    module_name: str,
    function_name: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Load a future engine function without hiding unrelated import errors."""
    qualified_name = f"{__package__}.{module_name}" if __package__ else f"engine.{module_name}"

    try:
        module = importlib.import_module(qualified_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        if missing_name in {qualified_name, module_name, "engine"}:
            return None, f"{module_name}.{function_name} is not available yet."
        raise

    function = getattr(module, function_name, None)
    if not callable(function):
        return None, f"{module_name}.{function_name} is not implemented yet."

    return function, None


def _load_historical_data(
    market: str,
    timeframes: Sequence[str],
    identity: ClientIdentity,
) -> Tuple[Dict[str, List[Candle]], List[str]]:
    """Load only the requested chart archives through the history module."""
    loader, issue = _load_optional_engine_function(
        "history",
        "load_candles_for_timeframes",
    )
    if loader is None:
        return {}, [issue or "Historical data loader is unavailable."]

    loaded = loader(
        market,
        list(timeframes),
        identity=identity,
    )

    if not isinstance(loaded, dict):
        return {}, ["history.load_candles_for_timeframes returned an invalid value."]

    normalized: Dict[str, List[Candle]] = {}
    for timeframe, candles in loaded.items():
        key = str(timeframe).strip().upper()
        if key not in timeframes:
            continue
        normalized[key] = list(candles or [])

    return normalized, []


def _build_automatic_levels(
    candles_by_timeframe: Dict[str, List[Candle]],
    setups: Sequence[UserSetup],
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """Build point-in-time automatic levels using the dedicated feature module."""
    builder, issue = _load_optional_engine_function(
        "features",
        "build_auto_levels_for_setups",
    )
    if builder is None:
        return {}, [issue or "Automatic level builder is unavailable."]

    result = builder(
        candles_by_timeframe=candles_by_timeframe,
        setups=setups,
    )

    if not isinstance(result, dict):
        return {}, ["features.build_auto_levels_for_setups returned an invalid value."]

    normalized: Dict[int, List[Dict[str, Any]]] = {}
    for setup in setups:
        levels = result.get(setup.setup_id)
        if levels is None:
            levels = result.get(str(setup.setup_id), [])
        normalized[setup.setup_id] = list(levels or [])

    return normalized, []


# ============================================================================
# TRAINING DATA VALIDATION
# ============================================================================

def validate_training_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the structure and completeness of a prepared input bundle."""
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(bundle, dict):
        return {
            "valid": False,
            "errors": ["Training data bundle must be a dictionary."],
            "warnings": [],
        }

    required_keys = {
        "identity",
        "market",
        "learning_setups",
        "historical_data",
        "auto_levels_by_setup",
        "manual_levels_by_setup",
        "timeframe_data",
        "report",
    }
    missing_keys = sorted(required_keys - set(bundle))
    if missing_keys:
        errors.append("Missing bundle keys: " + ", ".join(missing_keys))

    report = bundle.get("report", {})
    if isinstance(report, dict):
        warnings.extend(str(item) for item in report.get("issues", []))
        if not report.get("setup_count", 0):
            errors.append("The Learning setup file contains no setups.")
        if report.get("missing_timeframes"):
            errors.append(
                "Missing historical archives for: "
                + ", ".join(report["missing_timeframes"])
            )
        if report.get("invalid_setup_count", 0):
            errors.append(
                f"{report['invalid_setup_count']} Learning setup(s) contain invalid fields."
            )
        if report.get("pending_components"):
            errors.append(
                "Engine components are not ready: "
                + ", ".join(report["pending_components"])
            )
        if not report.get("ready_for_next_stage", False) and not errors:
            errors.append("The Training Data Bundle is not complete yet.")
    else:
        errors.append("Bundle report must be a dictionary.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# TRAINING DATA BUNDLE ASSEMBLY
# ============================================================================

def prepare_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Assemble Learning setups, required history and market levels without training."""
    resolved = _resolve_ai_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    normalized_market = resolved.market

    raw_records = load_learning_setup_records(
        normalized_market,
        identity=resolved,
    )
    setups = [
        normalize_learning_setup(item, index)
        for index, item in enumerate(raw_records, start=1)
    ]

    required_timeframes = collect_required_timeframes(setups)
    candles_by_timeframe, history_issues = _load_historical_data(
        normalized_market,
        required_timeframes,
        resolved,
    )
    auto_levels_by_setup, feature_issues = _build_automatic_levels(
        candles_by_timeframe,
        setups,
    )
    manual_levels_by_setup = extract_manual_levels(setups)

    candle_counts = {
        timeframe: len(candles_by_timeframe.get(timeframe, []))
        for timeframe in required_timeframes
    }
    missing_timeframes = [
        timeframe
        for timeframe in required_timeframes
        if not candles_by_timeframe.get(timeframe)
    ]

    coverage_by_timeframe: Dict[str, Dict[str, Any]] = {}
    for timeframe in required_timeframes:
        candles = candles_by_timeframe.get(timeframe, [])
        candle_times = [
            int(candle.time)
            for candle in candles
            if getattr(candle, "time", None) is not None
        ]
        coverage_by_timeframe[timeframe] = {
            "candle_count": len(candles),
            "first_candle_time": min(candle_times) if candle_times else None,
            "last_candle_time": max(candle_times) if candle_times else None,
        }

    setup_timeframe_data: List[Dict[str, Any]] = []
    history_alignment_issues: List[str] = []
    for setup in setups:
        setup_required_timeframes = [
            timeframe
            for timeframe in (
                setup.setup_timeframe,
                setup.entry_timeframe,
                setup.resolution_timeframe,
            )
            if timeframe and timeframe in SUPPORTED_TIMEFRAMES
        ]
        feature_timeframes = [
            timeframe
            for timeframe in (setup.setup_timeframe, setup.entry_timeframe)
            if timeframe and timeframe in SUPPORTED_TIMEFRAMES
        ]
        available_at_decision: Dict[str, Dict[str, Any]] = {}

        for timeframe in feature_timeframes:
            candles = candles_by_timeframe.get(timeframe, [])
            decision_time = setup.decision_time
            duration = TIMEFRAME_SECONDS[timeframe]
            closed_times = [
                int(candle.time)
                for candle in candles
                if getattr(candle, "time", None) is not None
                and int(candle.time) + duration <= (decision_time or 0)
            ]
            available_at_decision[timeframe] = {
                "closed_candle_count": len(closed_times),
                "latest_closed_candle_time": (
                    max(closed_times) if closed_times else None
                ),
            }

            if decision_time is not None and candles and not closed_times:
                history_alignment_issues.append(
                    f"Setup {setup.setup_id} has no fully closed {timeframe} "
                    "candle available at its entry time."
                )

        setup_timeframe_data.append(
            {
                "setup_id": setup.setup_id,
                "decision_time": setup.decision_time,
                "setup_timeframe": setup.setup_timeframe,
                "entry_timeframe": setup.entry_timeframe,
                "resolution_timeframe": setup.resolution_timeframe,
                "required_timeframes": setup_required_timeframes,
                "feature_timeframes": feature_timeframes,
                "available_at_decision": available_at_decision,
            }
        )

    setup_issues: List[str] = []
    invalid_setup_count = 0
    manual_support_count = 0
    manual_resistance_count = 0
    manual_zone_errors = 0

    for setup in setups:
        if setup.validation_errors:
            invalid_setup_count += 1
            setup_issues.extend(
                f"Setup {setup.setup_id}: {message}"
                for message in setup.validation_errors
            )

        for zone in setup.zones:
            if zone.zone_type == "SUPPORT":
                manual_support_count += 1
            elif zone.zone_type == "RESISTANCE":
                manual_resistance_count += 1
            manual_zone_errors += len(zone.validation_errors)

    automatic_level_count = sum(
        len(levels) for levels in auto_levels_by_setup.values()
    )

    pending_components: List[str] = []
    if history_issues:
        pending_components.append("engine.history")
    if feature_issues:
        pending_components.append("engine.features")

    issues = (
        setup_issues
        + history_alignment_issues
        + history_issues
        + feature_issues
    )
    ready_for_next_stage = (
        bool(setups)
        and invalid_setup_count == 0
        and not missing_timeframes
        and not history_alignment_issues
        and not pending_components
        and all(setup.setup_id in auto_levels_by_setup for setup in setups)
    )

    report = {
        "setup_count": len(setups),
        "valid_setup_count": len(setups) - invalid_setup_count,
        "invalid_setup_count": invalid_setup_count,
        "required_timeframes": list(required_timeframes),
        "loaded_timeframes": sorted(candles_by_timeframe),
        "candle_counts": candle_counts,
        "coverage_by_timeframe": coverage_by_timeframe,
        "missing_timeframes": missing_timeframes,
        "history_alignment_issues": history_alignment_issues,
        "manual_zone_count": manual_support_count + manual_resistance_count,
        "manual_support_count": manual_support_count,
        "manual_resistance_count": manual_resistance_count,
        "manual_zone_validation_errors": manual_zone_errors,
        "automatic_level_count": automatic_level_count,
        "automatic_level_counts_by_setup": {
            str(setup_id): len(levels)
            for setup_id, levels in auto_levels_by_setup.items()
        },
        "pending_components": pending_components,
        "issues": issues,
        "ready_for_next_stage": ready_for_next_stage,
    }

    bundle: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": "READY_FOR_NEXT_STAGE" if ready_for_next_stage else "INCOMPLETE",
        "identity": resolved.as_dict(),
        "market": normalized_market,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "learning_setups": {
            "raw": copy.deepcopy(raw_records),
            "normalized": setups,
        },
        "historical_data": {
            "timeframes": list(required_timeframes),
            "candles_by_timeframe": candles_by_timeframe,
        },
        "auto_levels_by_setup": auto_levels_by_setup,
        "manual_levels_by_setup": manual_levels_by_setup,
        "timeframe_data": setup_timeframe_data,
        "report": report,
        "model_training_started": False,
    }

    bundle["validation"] = validate_training_data(bundle)
    return bundle


# ============================================================================
# FUTURE DATASET COMPATIBILITY
# ============================================================================

def make_negative_times(*args: Any, **kwargs: Any) -> Any:
    """Delegate no-trade sample generation to the future dataset module."""
    function, issue = _load_optional_engine_function(
        "dataset",
        "make_negative_times",
    )
    if function is None:
        raise NotImplementedError(issue or "Negative sample generation is not ready.")
    return function(*args, **kwargs)


def build_training_dataset(*args: Any, **kwargs: Any) -> Any:
    """Delegate labelled dataset creation to the future dataset module."""
    function, issue = _load_optional_engine_function(
        "dataset",
        "build_training_dataset",
    )
    if function is None:
        raise NotImplementedError(issue or "Training dataset creation is not ready.")
    return function(*args, **kwargs)


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "Candle",
    "ManualZone",
    "UserSetup",
    "RESOLUTION_TIMEFRAME_FIELD",
    "safe_market_folder",
    "safe_user_folder",
    "_resolve_ai_identity",
    "setup_path",
    "to_float",
    "parse_timestamp",
    "normalize_learning_setup",
    "load_learning_setup_records",
    "load_learning_setups",
    "setup_time",
    "closest_setup",
    "collect_required_timeframes",
    "extract_manual_levels",
    "validate_training_data",
    "prepare_training_data",
    "make_negative_times",
    "build_training_dataset",
]
