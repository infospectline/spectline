"""Historical candle loading and point-in-time timeframe alignment."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
    )


# ============================================================================
# TIMEFRAME SELECTION
# ============================================================================


def _setup_value(setup: Any, field_name: str) -> Any:
    """Read a setup field from either a normalized object or a dictionary."""
    if isinstance(setup, Mapping):
        return setup.get(field_name)
    return getattr(setup, field_name, None)


def select_required_timeframes(setups_or_timeframes: Iterable[Any]) -> List[str]:
    """Return unique required timeframes in the configured chronological order.

    The input can contain normalized Learning setup objects, setup dictionaries,
    or timeframe strings. For setup records, setup, entry and optional
    resolution timeframes are included. Timeframes unavailable in the current
    runtime mode are not selected.
    """
    requested: set[str] = set()
    supported = set(SUPPORTED_TIMEFRAMES)
    items = (setups_or_timeframes,) if isinstance(setups_or_timeframes, str) else setups_or_timeframes

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

    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in requested]


def _normalize_timeframes(timeframes: Iterable[str]) -> List[str]:
    """Validate and de-duplicate an explicit timeframe list."""
    requested: set[str] = set()
    values = (timeframes,) if isinstance(timeframes, str) else timeframes

    for value in values:
        timeframe = str(value).strip().upper()
        if not timeframe:
            continue
        if timeframe not in SUPPORTED_TIMEFRAMES:
            mode = "WebMode" if "M1" not in SUPPORTED_TIMEFRAMES else "local mode"
            raise ValueError(
                f"Timeframe '{timeframe}' is not available in {mode}."
            )
        requested.add(timeframe)

    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in requested]


# ============================================================================
# ARCHIVE PATHS
# ============================================================================


def chart_path(market: str, timeframe: str) -> Path:
    """Return the shared, read-only candle archive path for a market."""
    normalized_market = normalize_market(market)
    normalized_timeframe = str(timeframe).strip().upper()

    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        mode = "WebMode" if "M1" not in SUPPORTED_TIMEFRAMES else "local mode"
        raise ValueError(
            f"Timeframe '{normalized_timeframe}' is not available in {mode}."
        )

    return get_chart_history_path(
        normalized_market,
        normalized_timeframe,
        CANDLE_FILENAME,
    )


# ============================================================================
# CSV NORMALIZATION
# ============================================================================


def _normalized_row(row: Mapping[Any, Any]) -> Dict[str, Any]:
    """Normalize CSV column names while retaining their original values."""
    return {
        str(key).strip().lower(): value
        for key, value in row.items()
        if key is not None
    }


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    """Return the first present, non-empty value from a list of column names."""
    for name in names:
        value = row.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value
    return None


def _csv_reader(file_object: Any) -> csv.DictReader:
    """Create a DictReader using the delimiter found in the archive header."""
    sample = file_object.read(8192)
    file_object.seek(0)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return csv.DictReader(file_object, dialect=dialect)
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        return csv.DictReader(file_object, delimiter=delimiter)


def _candle_from_row(row: Mapping[Any, Any]) -> Optional[Candle]:
    """Convert one CSV row into a validated normalized Candle instance."""
    values = _normalized_row(row)

    timestamp = parse_timestamp(
        _row_value(values, "time", "timestamp", "datetime", "date", "time_msc")
    )
    open_price = to_float(_row_value(values, "open", "o"))
    high_price = to_float(_row_value(values, "high", "h"))
    low_price = to_float(_row_value(values, "low", "l"))
    close_price = to_float(_row_value(values, "close", "c"))
    volume = to_float(
        _row_value(
            values,
            "tick_volume",
            "tickvolume",
            "volume",
            "real_volume",
        )
    )

    if timestamp is None:
        return None

    prices = (open_price, high_price, low_price, close_price)
    if any(value is None for value in prices):
        return None

    if not all(math.isfinite(float(value)) for value in prices if value is not None):
        return None

    open_value = float(open_price)
    high_value = float(high_price)
    low_value = float(low_price)
    close_value = float(close_price)

    if high_value < low_value:
        return None

    volume_value = float(volume) if volume is not None else 0.0
    if not math.isfinite(volume_value):
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


def load_candles(market: str, timeframe: str) -> List[Candle]:
    """Load, normalize, sort and de-duplicate one chart archive.

    A missing archive returns an empty list. An archive that exists but cannot
    be read raises an OSError so the caller can distinguish missing history
    from an inaccessible file.
    """
    path = chart_path(market, timeframe)
    if not path.is_file():
        return []

    candles_by_time: Dict[int, Candle] = {}

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file_object:
            reader = _csv_reader(file_object)
            for row in reader:
                if not row:
                    continue
                candle = _candle_from_row(row)
                if candle is None:
                    continue
                # The last valid row wins when an archive contains duplicate times.
                candles_by_time[candle.time] = candle
    except OSError as error:
        raise OSError(f"Could not read chart archive '{path}': {error}") from error

    return [candles_by_time[timestamp] for timestamp in sorted(candles_by_time)]


def load_candles_for_timeframes(
    market: str,
    timeframes: Sequence[str],
    *,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, List[Candle]]:
    """Load each requested archive once for the supplied market.

    Historical chart archives are shared and read-only. The identity argument
    is accepted to keep the data-loading interface TradingBot-aware; it is
    validated against the requested market but does not change the shared
    archive location.
    """
    normalized_market = normalize_market(market)

    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        if identity.market != normalized_market:
            raise ValueError(
                f"Identity market {identity.market} does not match "
                f"requested market {normalized_market}."
            )

    selected = _normalize_timeframes(timeframes)
    return {
        timeframe: load_candles(normalized_market, timeframe)
        for timeframe in selected
    }


def load_all_market_candles(
    market: str,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, List[Candle]]:
    """Load only the timeframes referenced by one TradingBot's Learning setups."""
    normalized_market = normalize_market(market)

    if identity is None:
        resolved_identity = resolve_identity(market=normalized_market)
    else:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        if identity.market != normalized_market:
            raise ValueError(
                f"Identity market {identity.market} does not match "
                f"requested market {normalized_market}."
            )
        resolved_identity = identity

    # Import here to avoid a module-level circular dependency with engine.data.
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
    """Return the newest candle fully closed by ``target_time``.

    Candle timestamps represent candle opening times. A candle opened at T
    with duration D is available only when T + D <= target_time. This prevents
    an unfinished candle or future price data from entering historical input.
    """
    if not candles:
        return -1

    normalized_timeframe = str(timeframe).strip().upper()
    if normalized_timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Timeframe '{normalized_timeframe}' is not supported in this runtime."
        )

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
    """Return up to ``max_count`` candles fully closed by ``target_time``."""
    if max_count <= 0 or not candles:
        return []

    last_index = last_closed_index(
        candles=candles,
        target_time=target_time,
        timeframe=timeframe,
    )
    if last_index < 0:
        return []

    first_index = max(0, last_index - int(max_count) + 1)
    return list(candles[first_index:last_index + 1])


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "chart_path",
    "load_candles",
    "load_candles_for_timeframes",
    "load_all_market_candles",
    "select_required_timeframes",
    "candle_return",
    "true_range",
    "rolling_mean",
    "rolling_std",
    "rolling_slope",
    "last_closed_index",
    "slice_closed_history",
]
