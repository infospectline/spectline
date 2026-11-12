"""Historical candle loading, validation and point-in-time timeframe alignment."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from client import (
        ClientIdentity,
        get_chart_history_path,
        normalize_market,
        resolve_identity,
    )
    from engine.data import Candle, parse_timestamp, to_float
    from engine.settings import (
        CANDLE_FILENAME,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
        WEB_MODE,
    )
except ImportError:
    from ..client import (
        ClientIdentity,
        get_chart_history_path,
        normalize_market,
        resolve_identity,
    )
    from .data import Candle, parse_timestamp, to_float
    from .settings import (
        CANDLE_FILENAME,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
        WEB_MODE,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

_CSV_SAMPLE_SIZE = 8192
_CSV_DELIMITERS = ",;\t|"
_CSV_ENCODING = "utf-8-sig"

_OHLC_COLUMNS = {
    "time": ("time", "timestamp", "datetime", "date", "time_msc"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c"),
    "volume": ("tick_volume", "tickvolume", "volume", "real_volume"),
}


# ============================================================================
# TIMEFRAME SELECTION
# ============================================================================

def _setup_value(setup: Any, field_name: str) -> Any:
    """Read a setup field from either a normalized object or a dictionary."""
    if isinstance(setup, Mapping):
        return setup.get(field_name)
    return getattr(setup, field_name, None)


def select_required_timeframes(setups_or_timeframes: Iterable[Any]) -> List[str]:
    """Return the unique runtime-supported timeframes used by the input.

    The input may contain timeframe strings, setup dictionaries or normalized
    setup objects. Setup, entry and optional resolution timeframes are read.
    The returned order always follows the configured timeframe order.
    """
    requested: set[str] = set()
    supported = set(SUPPORTED_TIMEFRAMES)
    items = (
        (setups_or_timeframes,)
        if isinstance(setups_or_timeframes, str)
        else setups_or_timeframes
    )

    for item in items:
        if isinstance(item, str):
            values = (item,)
        else:
            values = (
                _setup_value(item, "setup_timeframe"),
                _setup_value(item, "entry_timeframe"),
                _setup_value(item, "resolution_timeframe"),
            )

        for value in values:
            if value is None:
                continue
            timeframe = str(value).strip().upper()
            if timeframe in supported:
                requested.add(timeframe)

    return [
        timeframe
        for timeframe in SUPPORTED_TIMEFRAMES
        if timeframe in requested
    ]


def _runtime_mode_name() -> str:
    """Return a short label for the active chart runtime mode."""
    return "WebMode" if WEB_MODE else "local mode"


def _normalize_timeframes(timeframes: Iterable[str]) -> List[str]:
    """Validate and de-duplicate an explicit timeframe list."""
    if timeframes is None:
        raise TypeError("timeframes must be an iterable of timeframe names.")

    values = (timeframes,) if isinstance(timeframes, str) else timeframes
    requested: set[str] = set()

    for value in values:
        if value is None:
            raise ValueError("A timeframe value cannot be None.")

        timeframe = str(value).strip().upper()
        if not timeframe:
            continue

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' is not available in {_runtime_mode_name()}."
            )
        requested.add(timeframe)

    return [
        timeframe
        for timeframe in SUPPORTED_TIMEFRAMES
        if timeframe in requested
    ]


# ============================================================================
# ARCHIVE PATHS AND IDENTITY VALIDATION
# ============================================================================

def chart_path(market: str, timeframe: str) -> Path:
    """Return the shared, read-only candle archive path for a market."""
    normalized_market = normalize_market(market)
    normalized_timeframes = _normalize_timeframes((timeframe,))
    normalized_timeframe = normalized_timeframes[0]

    return Path(
        get_chart_history_path(
            normalized_market,
            normalized_timeframe,
            CANDLE_FILENAME,
        )
    )


def _validate_identity_market(
    market: str,
    identity: Optional[ClientIdentity],
) -> None:
    """Validate that an optional customer identity belongs to the market."""
    if identity is None:
        return

    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    if identity.market != market:
        raise ValueError(
            f"Identity market {identity.market} does not match requested market {market}."
        )


# ============================================================================
# CSV NORMALIZATION
# ============================================================================

def _normalized_row(row: Mapping[Any, Any]) -> Dict[str, Any]:
    """Normalize CSV column names while retaining their original values."""
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[str(key).strip().lower()] = value
    return normalized


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    """Return the first present, non-empty value from candidate column names."""
    for name in names:
        value = row.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value
    return None


def _csv_reader(file_object: Any) -> csv.DictReader:
    """Create a DictReader using the delimiter detected in the archive header."""
    sample = file_object.read(_CSV_SAMPLE_SIZE)
    file_object.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS)
        return csv.DictReader(file_object, dialect=dialect)
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter_counts = {
            delimiter: first_line.count(delimiter)
            for delimiter in _CSV_DELIMITERS
        }
        delimiter = max(delimiter_counts, key=delimiter_counts.get)
        if delimiter_counts[delimiter] == 0:
            delimiter = ","
        return csv.DictReader(file_object, delimiter=delimiter)


def _valid_ohlc(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> bool:
    """Return whether the candle prices form a valid OHLC range."""
    values = (open_price, high_price, low_price, close_price)
    if not all(math.isfinite(value) for value in values):
        return False

    if high_price < low_price:
        return False

    if high_price < max(open_price, close_price):
        return False

    if low_price > min(open_price, close_price):
        return False

    return True


def _candle_from_row(row: Mapping[Any, Any]) -> Optional[Candle]:
    """Convert one CSV row into a validated normalized Candle instance."""
    values = _normalized_row(row)

    timestamp = parse_timestamp(_row_value(values, *_OHLC_COLUMNS["time"]))
    open_price = to_float(_row_value(values, *_OHLC_COLUMNS["open"]))
    high_price = to_float(_row_value(values, *_OHLC_COLUMNS["high"]))
    low_price = to_float(_row_value(values, *_OHLC_COLUMNS["low"]))
    close_price = to_float(_row_value(values, *_OHLC_COLUMNS["close"]))
    volume = to_float(_row_value(values, *_OHLC_COLUMNS["volume"]))

    if timestamp is None:
        return None

    prices = (open_price, high_price, low_price, close_price)
    if any(value is None for value in prices):
        return None

    open_value = float(open_price)
    high_value = float(high_price)
    low_value = float(low_price)
    close_value = float(close_price)

    if not _valid_ohlc(
        open_value,
        high_value,
        low_value,
        close_value,
    ):
        return None

    volume_value = float(volume) if volume is not None else 0.0
    if not math.isfinite(volume_value) or volume_value < 0.0:
        volume_value = 0.0

    return Candle(
        time=int(timestamp),
        open=open_value,
        high=high_value,
        low=low_value,
        close=close_value,
        volume=volume_value,
    )


# ============================================================================
# HISTORICAL CANDLE LOADING
# ============================================================================

def _empty_load_report(path: Path, market: str, timeframe: str) -> Dict[str, Any]:
    """Create a consistent load report for an absent or empty archive."""
    return {
        "market": market,
        "timeframe": timeframe,
        "path": str(path),
        "archive_exists": path.is_file(),
        "rows_read": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "duplicate_timestamps": 0,
        "candle_count": 0,
        "first_candle_time": None,
        "last_candle_time": None,
    }


def _load_candles_internal(
    market: str,
    timeframe: str,
) -> Tuple[List[Candle], Dict[str, Any]]:
    """Load one archive and return normalized candles with quality metadata."""
    normalized_market = normalize_market(market)
    normalized_timeframe = _normalize_timeframes((timeframe,))[0]
    path = chart_path(normalized_market, normalized_timeframe)
    report = _empty_load_report(path, normalized_market, normalized_timeframe)

    if not path.is_file():
        return [], report

    candles_by_time: Dict[int, Candle] = {}

    try:
        with path.open("r", newline="", encoding=_CSV_ENCODING) as file_object:
            reader = _csv_reader(file_object)
            for row in reader:
                if not row:
                    continue

                report["rows_read"] += 1
                candle = _candle_from_row(row)
                if candle is None:
                    report["invalid_rows"] += 1
                    continue

                report["valid_rows"] += 1
                if candle.time in candles_by_time:
                    report["duplicate_timestamps"] += 1

                candles_by_time[candle.time] = candle
    except OSError as error:
        raise OSError(
            f"Could not read chart archive '{path}': {error}"
        ) from error
    except (csv.Error, UnicodeError) as error:
        raise ValueError(
            f"Could not parse chart archive '{path}': {error}"
        ) from error

    candles = [
        candles_by_time[timestamp]
        for timestamp in sorted(candles_by_time)
    ]

    report.update(
        {
            "archive_exists": True,
            "candle_count": len(candles),
            "first_candle_time": candles[0].time if candles else None,
            "last_candle_time": candles[-1].time if candles else None,
        }
    )
    return candles, report


def load_candles(market: str, timeframe: str) -> List[Candle]:
    """Load, normalize, chronologically sort and de-duplicate one archive.

    A missing archive returns an empty list. An archive that exists but cannot
    be read or parsed raises an exception, allowing callers to distinguish a
    missing history from an inaccessible or malformed archive. When duplicate
    timestamps occur, the last valid row in the CSV is retained.
    """
    candles, _ = _load_candles_internal(market, timeframe)
    return candles


def load_candles_with_report(
    market: str,
    timeframe: str,
) -> Tuple[List[Candle], Dict[str, Any]]:
    """Load one archive and return candle-quality and coverage metadata."""
    return _load_candles_internal(market, timeframe)


def load_candles_for_timeframes(
    market: str,
    timeframes: Sequence[str],
    *,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, List[Candle]]:
    """Load each requested shared market archive once for a TradingBot.

    Market archives are shared and read-only. The optional identity is checked
    against the market but does not alter the shared archive location.
    Missing archives are represented by empty lists for their timeframe keys.
    """
    normalized_market = normalize_market(market)
    _validate_identity_market(normalized_market, identity)
    selected = _normalize_timeframes(timeframes)

    return {
        timeframe: load_candles(normalized_market, timeframe)
        for timeframe in selected
    }


def load_candles_with_reports(
    market: str,
    timeframes: Sequence[str],
    *,
    identity: Optional[ClientIdentity] = None,
) -> Tuple[Dict[str, List[Candle]], Dict[str, Dict[str, Any]]]:
    """Load multiple timeframe archives and return a report for each one."""
    normalized_market = normalize_market(market)
    _validate_identity_market(normalized_market, identity)
    selected = _normalize_timeframes(timeframes)

    candles_by_timeframe: Dict[str, List[Candle]] = {}
    reports_by_timeframe: Dict[str, Dict[str, Any]] = {}

    for timeframe in selected:
        candles, report = _load_candles_internal(
            normalized_market,
            timeframe,
        )
        candles_by_timeframe[timeframe] = candles
        reports_by_timeframe[timeframe] = report

    return candles_by_timeframe, reports_by_timeframe


def load_all_market_candles(
    market: str,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, List[Candle]]:
    """Load only the timeframes referenced by one TradingBot's Learning setups."""
    normalized_market = normalize_market(market)

    if identity is None:
        resolved_identity = resolve_identity(market=normalized_market)
    else:
        _validate_identity_market(normalized_market, identity)
        resolved_identity = identity

    try:
        from engine.data import load_learning_setups
    except ImportError:
        from .data import load_learning_setups

    setups = load_learning_setups(
        normalized_market,
        identity=resolved_identity,
    )
    required_timeframes = select_required_timeframes(setups)

    return load_candles_for_timeframes(
        normalized_market,
        required_timeframes,
        identity=resolved_identity,
    )


# ============================================================================
# HISTORY QUALITY AND CHRONOLOGY
# ============================================================================

def is_chronological(candles: Sequence[Candle]) -> bool:
    """Return whether candle opening timestamps are in non-decreasing order."""
    previous_time: Optional[int] = None
    for candle in candles:
        current_time = int(candle.time)
        if previous_time is not None and current_time < previous_time:
            return False
        previous_time = current_time
    return True


def require_chronological(candles: Sequence[Candle]) -> None:
    """Raise when a candle sequence is not ordered from oldest to newest."""
    if not is_chronological(candles):
        raise ValueError("Candle history must be sorted in ascending timestamp order.")


def history_coverage(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
) -> Dict[str, Dict[str, Any]]:
    """Return candle counts and timestamp coverage for each loaded timeframe."""
    coverage: Dict[str, Dict[str, Any]] = {}

    for raw_timeframe, candles in candles_by_timeframe.items():
        timeframe = str(raw_timeframe).strip().upper()
        ordered = list(candles or [])
        require_chronological(ordered)

        coverage[timeframe] = {
            "candle_count": len(ordered),
            "first_candle_time": ordered[0].time if ordered else None,
            "last_candle_time": ordered[-1].time if ordered else None,
        }

    return coverage


# ============================================================================
# CANDLE UTILITIES
# ============================================================================

def candle_return(previous: Candle, current: Candle) -> float:
    """Return the close-to-close price change normalized by prior close."""
    denominator = abs(float(previous.close))
    if denominator <= 0.0:
        return 0.0
    return (float(current.close) - float(previous.close)) / denominator


def true_range(previous: Optional[Candle], current: Candle) -> float:
    """Calculate true range using the previous candle close when available."""
    high = float(current.high)
    low = float(current.low)

    if previous is None:
        return max(high - low, 0.0)

    previous_close = float(previous.close)
    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
        0.0,
    )


def _finite_values(values: Sequence[float]) -> List[float]:
    """Convert a numeric sequence to finite float values."""
    result: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def rolling_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of finite values, or zero for empty input."""
    numbers = _finite_values(values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def rolling_std(values: Sequence[float]) -> float:
    """Return the population standard deviation of finite values."""
    numbers = _finite_values(values)
    if len(numbers) < 2:
        return 0.0

    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return math.sqrt(max(variance, 0.0))


def rolling_slope(values: Sequence[float]) -> float:
    """Return a linear-regression slope normalized by the last value."""
    numbers = _finite_values(values)
    count = len(numbers)
    if count < 3:
        return 0.0

    mean_y = sum(numbers) / count
    mean_x = (count - 1) / 2.0
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator <= 0.0:
        return 0.0

    numerator = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(numbers)
    )
    slope = numerator / denominator
    scale = max(abs(numbers[-1]), 1e-12)
    return slope / scale


# ============================================================================
# POINT-IN-TIME CANDLE ALIGNMENT
# ============================================================================

def last_closed_index(
    candles: Sequence[Candle],
    target_time: int,
    timeframe: str,
) -> int:
    """Return the newest candle fully closed by the requested UTC timestamp.

    Candle timestamps represent candle opening times. A candle opened at T
    with duration D is available only when T + D <= target_time. The input
    sequence must already be sorted in ascending timestamp order.
    """
    if not candles:
        return -1

    normalized_timeframe = _normalize_timeframes((timeframe,))[0]
    duration = TIMEFRAME_SECONDS[normalized_timeframe]
    cutoff = int(target_time) - int(duration)

    low_index = 0
    high_index = len(candles) - 1
    result = -1

    while low_index <= high_index:
        middle = (low_index + high_index) // 2
        candle_time = int(candles[middle].time)

        if candle_time <= cutoff:
            result = middle
            low_index = middle + 1
        else:
            high_index = middle - 1

    return result


def slice_closed_history(
    candles: Sequence[Candle],
    target_time: int,
    timeframe: str,
    max_count: int,
) -> List[Candle]:
    """Return up to max_count candles fully closed by the requested timestamp."""
    count = int(max_count)
    if count <= 0 or not candles:
        return []

    last_index = last_closed_index(
        candles=candles,
        target_time=int(target_time),
        timeframe=timeframe,
    )
    if last_index < 0:
        return []

    first_index = max(0, last_index - count + 1)
    return list(candles[first_index:last_index + 1])


def align_timeframes_at(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
    target_time: int,
) -> Dict[str, Candle]:
    """Return the latest fully closed candle for each available timeframe."""
    aligned: Dict[str, Candle] = {}

    for timeframe in _normalize_timeframes(candles_by_timeframe.keys()):
        candles = candles_by_timeframe.get(timeframe, [])
        index = last_closed_index(candles, target_time, timeframe)
        if index >= 0:
            aligned[timeframe] = candles[index]

    return aligned


def slice_timeframes_at(
    candles_by_timeframe: Mapping[str, Sequence[Candle]],
    target_time: int,
    lookbacks: Mapping[str, int],
) -> Dict[str, List[Candle]]:
    """Return point-in-time candle windows for several timeframe archives."""
    windows: Dict[str, List[Candle]] = {}

    for timeframe in _normalize_timeframes(candles_by_timeframe.keys()):
        if timeframe not in lookbacks:
            continue

        windows[timeframe] = slice_closed_history(
            candles=candles_by_timeframe.get(timeframe, []),
            target_time=target_time,
            timeframe=timeframe,
            max_count=int(lookbacks[timeframe]),
        )

    return windows


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "chart_path",
    "load_candles",
    "load_candles_with_report",
    "load_candles_for_timeframes",
    "load_candles_with_reports",
    "load_all_market_candles",
    "select_required_timeframes",
    "is_chronological",
    "require_chronological",
    "history_coverage",
    "candle_return",
    "true_range",
    "rolling_mean",
    "rolling_std",
    "rolling_slope",
    "last_closed_index",
    "slice_closed_history",
    "align_timeframes_at",
    "slice_timeframes_at",
]
