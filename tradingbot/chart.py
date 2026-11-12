"""Shared market-chart history and MetaTrader 5 synchronization for Spectline."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import csv
import importlib
import math
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from client import (
    CHARTS_ROOT,
    SUPPORTED_TIMEFRAMES,
    WEB_MODE,
    normalize_market,
    normalize_timeframe,
)
from runtime import RuntimeContext


# ============================================================================
# CONFIGURATION
# ============================================================================

YEARS_BACK = 5
DOWNLOAD_CHUNK_DAYS = 90
LOCAL_CHART_FILENAME = "candles.csv"

TIMEFRAME_SECONDS: Dict[str, int] = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}

WEB_UNSUPPORTED_TIMEFRAMES = frozenset({"M1", "M5"})
CSV_FIELDNAMES = ("time", "open", "high", "low", "close", "volume")
CSV_SAMPLE_SIZE = 8192
CSV_DELIMITERS = ",;\t|"

_ARCHIVE_LOCKS: Dict[str, threading.RLock] = {}
_ARCHIVE_LOCKS_GUARD = threading.Lock()


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def _archive_lock(path: Path) -> threading.RLock:
    """Return a process-local lock for one archive path."""
    key = str(path.resolve())
    with _ARCHIVE_LOCKS_GUARD:
        lock = _ARCHIVE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ARCHIVE_LOCKS[key] = lock
        return lock


def _finite_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _read_field(record: Any, field_name: str) -> Any:
    """Read a field from a mapping, structured array row or object."""
    if isinstance(record, dict):
        normalized = {str(key).strip().lower(): value for key, value in record.items()}
        return normalized.get(field_name.lower())
    try:
        return record[field_name]
    except (KeyError, IndexError, TypeError, ValueError):
        return getattr(record, field_name, None)


def _read_first_field(record: Any, field_names: Sequence[str]) -> Any:
    """Read the first available non-empty field from a sequence of aliases."""
    for field_name in field_names:
        value = _read_field(record, field_name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _normalize_candle_record(record: Any) -> Optional[Dict[str, Any]]:
    """Return one validated OHLCV candle dictionary."""
    timestamp = _read_field(record, "time")
    try:
        timestamp_value = int(timestamp)
    except (TypeError, ValueError, OverflowError):
        return None

    open_price = _finite_float(_read_field(record, "open"))
    high_price = _finite_float(_read_field(record, "high"))
    low_price = _finite_float(_read_field(record, "low"))
    close_price = _finite_float(_read_field(record, "close"))

    if any(value is None for value in (open_price, high_price, low_price, close_price)):
        return None

    open_value = float(open_price)
    high_value = float(high_price)
    low_value = float(low_price)
    close_value = float(close_price)

    if high_value < low_value:
        return None
    if high_value < max(open_value, close_value):
        return None
    if low_value > min(open_value, close_value):
        return None

    volume_value = _finite_float(
        _read_first_field(
            record,
            ("tick_volume", "tickvolume", "volume", "real_volume"),
        )
    )
    if volume_value is None or volume_value < 0.0:
        volume_value = 0.0

    return {
        "time": timestamp_value,
        "open": open_value,
        "high": high_value,
        "low": low_value,
        "close": close_value,
        "volume": int(volume_value),
    }


def _merge_candles(
    existing: Sequence[Dict[str, Any]],
    incoming: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge candle sequences by opening timestamp and return sorted records."""
    by_timestamp: Dict[int, Dict[str, Any]] = {}

    for record in existing:
        normalized = _normalize_candle_record(record)
        if normalized is not None:
            by_timestamp[normalized["time"]] = normalized

    for record in incoming:
        normalized = _normalize_candle_record(record)
        if normalized is not None:
            by_timestamp[normalized["time"]] = normalized

    return [by_timestamp[timestamp] for timestamp in sorted(by_timestamp)]


def _calendar_years_before(moment: datetime, years: int) -> datetime:
    """Subtract calendar years while handling leap-day boundaries."""
    try:
        return moment.replace(year=moment.year - years)
    except ValueError:
        return moment.replace(year=moment.year - years, day=28)


def _to_utc(moment: datetime) -> datetime:
    """Validate and convert a datetime to UTC."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Datetime values must be timezone-aware.")
    return moment.astimezone(timezone.utc)


def _runtime_mode_name(web_mode: bool) -> str:
    """Return a concise runtime mode label."""
    return "WebMode" if web_mode else "local mode"


# ============================================================================
# CHART SERVICE
# ============================================================================

class ChartService:
    """Load shared chart archives and synchronize local data with MT5.

    Candle archives are shared by market and timeframe. Every TradingBot gets
    its own in-memory list, so one runtime cannot mutate another runtime's
    chart data. The runtime owns its market and active timeframe.
    """

    def __init__(
        self,
        charts_root: Path | str = CHARTS_ROOT,
        web_mode: bool = WEB_MODE,
        years_back: int = YEARS_BACK,
        download_chunk_days: int = DOWNLOAD_CHUNK_DAYS,
        mt5_module: Any = None,
    ) -> None:
        self.charts_root = Path(charts_root).expanduser().resolve()
        self.web_mode = bool(web_mode)
        self.years_back = int(years_back)
        self.download_chunk_days = int(download_chunk_days)
        self.mt5 = mt5_module
        self.mt5_connected = False

        if self.years_back < 1:
            raise ValueError("years_back must be at least 1.")
        if self.download_chunk_days < 1:
            raise ValueError("download_chunk_days must be at least 1.")

    # ========================================================================
    # METATRADER 5
    # ========================================================================

    def _get_mt5(self) -> Any:
        """Return the MT5 module, importing it lazily in local mode."""
        if self.web_mode:
            raise RuntimeError("MetaTrader 5 is disabled in WebMode.")

        if self.mt5 is None:
            try:
                self.mt5 = importlib.import_module("MetaTrader5")
            except ImportError as error:
                raise RuntimeError(
                    "MetaTrader5 is not installed in this environment."
                ) from error

        return self.mt5

    def initialize_mt5(self, default_market: Optional[str] = None) -> bool:
        """Initialize the local MT5 connection."""
        if self.web_mode:
            self.mt5_connected = False
            return False

        mt5 = self._get_mt5()
        self.mt5_connected = False

        if not mt5.initialize():
            print(
                "MT5 is not available. "
                "The application will use local CHARTS data."
            )
            return False

        try:
            if default_market:
                resolved_market = self.resolve_symbol(default_market)
                if not mt5.symbol_select(resolved_market, True):
                    raise RuntimeError(
                        f"MT5 symbol selection failed for {resolved_market}."
                    )

            self.mt5_connected = True
            print("MT5 connected.")
            return True
        except Exception as error:
            print(f"MT5 connection setup failed: {error}")
            try:
                mt5.shutdown()
            finally:
                self.mt5_connected = False
            return False

    def shutdown_mt5(self) -> None:
        """Close the MT5 connection if this service initialized it."""
        if not self.mt5_connected or self.mt5 is None:
            return

        try:
            self.mt5.shutdown()
        finally:
            self.mt5_connected = False

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve a requested market against the names available in MT5."""
        requested = str(symbol).strip()
        if not requested:
            raise ValueError("Market symbol must not be empty.")

        mt5 = self._get_mt5()
        if mt5.symbol_info(requested) is not None:
            return requested

        symbols = mt5.symbols_get()
        if symbols:
            exact_matches = [
                item.name
                for item in symbols
                if str(item.name).upper() == requested.upper()
            ]
            if exact_matches:
                return str(exact_matches[0])

            prefix_matches = [
                item.name
                for item in symbols
                if str(item.name).upper().startswith(requested.upper())
            ]
            if prefix_matches:
                return str(prefix_matches[0])

        raise RuntimeError(
            f"MT5 symbol '{requested}' was not found. "
            "Check the symbol name in Market Watch."
        )

    def _mt5_timeframe(self, timeframe: str) -> Any:
        """Return the MT5 timeframe constant for a normalized timeframe."""
        normalized = normalize_timeframe(timeframe)
        mt5 = self._get_mt5()
        attribute_name = f"TIMEFRAME_{normalized}"
        value = getattr(mt5, attribute_name, None)

        if value is None:
            raise RuntimeError(
                f"MT5 does not expose the {attribute_name} constant."
            )
        return value

    # ========================================================================
    # ARCHIVE PATHS AND SERIALIZATION
    # ========================================================================

    def symbol_folder(self, market: str) -> str:
        """Return the normalized market folder name."""
        return normalize_market(market)

    def history_path(self, market: str, timeframe: str) -> Path:
        """Return the shared CSV archive path for one market and timeframe."""
        normalized_market = normalize_market(market)
        normalized_timeframe = normalize_timeframe(timeframe)
        return (
            self.charts_root
            / normalized_market
            / normalized_timeframe
            / LOCAL_CHART_FILENAME
        )

    @staticmethod
    def candle_to_dict(row: Any) -> Dict[str, Any]:
        """Convert an MT5 rate row into the normalized local candle format."""
        normalized = _normalize_candle_record(row)
        if normalized is None:
            raise ValueError("The MT5 rate row does not contain a valid candle.")
        return normalized

    @staticmethod
    def _csv_reader(file_object: Any) -> csv.DictReader:
        """Create a CSV reader using the delimiter detected in the header."""
        sample = file_object.read(CSV_SAMPLE_SIZE)
        file_object.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
            return csv.DictReader(file_object, dialect=dialect)
        except csv.Error:
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            counts = {delimiter: first_line.count(delimiter) for delimiter in CSV_DELIMITERS}
            delimiter = max(counts, key=counts.get)
            if counts[delimiter] == 0:
                delimiter = ","
            return csv.DictReader(file_object, delimiter=delimiter)

    @staticmethod
    def load_local_history(path: Path | str) -> List[Dict[str, Any]]:
        """Load valid candles, sort them and keep one row per opening timestamp."""
        archive_path = Path(path)
        if not archive_path.is_file():
            return []

        candles_by_time: Dict[int, Dict[str, Any]] = {}
        lock = _archive_lock(archive_path)

        with lock:
            try:
                with archive_path.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig",
                ) as file_object:
                    reader = ChartService._csv_reader(file_object)
                    for row in reader:
                        if not row:
                            continue
                        normalized = _normalize_candle_record(row)
                        if normalized is not None:
                            candles_by_time[normalized["time"]] = normalized
            except OSError as error:
                raise OSError(
                    f"Could not read chart archive '{archive_path}': {error}"
                ) from error

        return [
            candles_by_time[timestamp]
            for timestamp in sorted(candles_by_time)
        ]

    @staticmethod
    def save_local_history(
        path: Path | str,
        data: Sequence[Dict[str, Any]],
    ) -> None:
        """Atomically write a normalized, sorted and de-duplicated candle CSV."""
        archive_path = Path(path).expanduser().resolve()
        normalized_data = _merge_candles((), data)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        lock = _archive_lock(archive_path)
        temporary_path: Optional[Path] = None

        with lock:
            try:
                temporary_name = (
                    f".{archive_path.name}.{os.getpid()}."
                    f"{threading.get_ident()}.tmp"
                )
                temporary_path = archive_path.with_name(temporary_name)

                with temporary_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file_object:
                    writer = csv.DictWriter(
                        file_object,
                        fieldnames=list(CSV_FIELDNAMES),
                    )
                    writer.writeheader()
                    writer.writerows(normalized_data)
                    file_object.flush()
                    os.fsync(file_object.fileno())

                os.replace(temporary_path, archive_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    try:
                        temporary_path.unlink()
                    except OSError:
                        pass

    # ========================================================================
    # METATRADER 5 HISTORY DOWNLOAD
    # ========================================================================

    @staticmethod
    def filter_closed_candles(
        rates: Any,
        timeframe: str,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Exclude any candle whose full timeframe interval has not elapsed."""
        if rates is None:
            return []

        normalized_timeframe = normalize_timeframe(timeframe)
        duration = TIMEFRAME_SECONDS[normalized_timeframe]
        current_time = _to_utc(now or datetime.now(timezone.utc))
        cutoff_timestamp = int(current_time.timestamp())

        result: List[Dict[str, Any]] = []
        for row in rates:
            try:
                candle = ChartService.candle_to_dict(row)
            except (TypeError, ValueError, KeyError, IndexError):
                continue

            if candle["time"] + duration <= cutoff_timestamp:
                result.append(candle)

        return _merge_candles((), result)

    def download_mt5_history(
        self,
        market: str,
        timeframe: str,
        utc_from: datetime,
        utc_to: datetime,
    ) -> List[Dict[str, Any]]:
        """Download bounded MT5 date chunks and return unique closed candles."""
        mt5_symbol = str(market).strip()
        if not mt5_symbol:
            raise ValueError("MT5 market symbol must not be empty.")
        normalized_timeframe = normalize_timeframe(timeframe)

        if not self.mt5_connected:
            raise RuntimeError("MT5 is not connected.")

        start_time = _to_utc(utc_from)
        end_time = _to_utc(utc_to)
        if start_time > end_time:
            return []

        mt5 = self._get_mt5()
        mt5_timeframe = self._mt5_timeframe(normalized_timeframe)
        result_by_time: Dict[int, Dict[str, Any]] = {}

        cursor = start_time
        chunk_delta = timedelta(days=self.download_chunk_days)

        while cursor <= end_time:
            chunk_end = min(cursor + chunk_delta, end_time)
            rates = mt5.copy_rates_range(
                mt5_symbol,
                mt5_timeframe,
                cursor,
                chunk_end,
            )

            if rates is None:
                raise RuntimeError(
                    f"MT5 copy_rates_range failed for {mt5_symbol} "
                    f"{normalized_timeframe}: {mt5.last_error()}"
                )

            for candle in self.filter_closed_candles(
                rates,
                normalized_timeframe,
            ):
                result_by_time[candle["time"]] = candle

            cursor = chunk_end + timedelta(seconds=1)

        return [
            result_by_time[timestamp]
            for timestamp in sorted(result_by_time)
        ]

    # ========================================================================
    # RUNTIME PROGRESS
    # ========================================================================

    @staticmethod
    def _set_runtime_progress(
        runtime: Optional[RuntimeContext],
        stage: str,
        progress: int,
        message: str,
    ) -> None:
        """Update progress metadata only for the supplied TradingBot runtime."""
        if runtime is None:
            return

        with runtime.lock:
            runtime.startup_stage = str(stage)
            runtime.startup_progress = max(0, min(100, int(progress)))
            runtime.startup_message = str(message)

    # ========================================================================
    # ARCHIVE SYNCHRONIZATION
    # ========================================================================

    def _requested_history_start(self, now: datetime) -> datetime:
        """Return the UTC start of the configured historical download window."""
        return _calendar_years_before(_to_utc(now), self.years_back)

    def _synchronize_archive(
        self,
        market: str,
        timeframe: str,
        path: Path,
        local_data: Sequence[Dict[str, Any]],
        now: Optional[datetime] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Backfill an old archive boundary and append missing recent candles."""
        if not self.mt5_connected:
            return list(local_data), 0, 0

        normalized_timeframe = normalize_timeframe(timeframe)
        duration = TIMEFRAME_SECONDS[normalized_timeframe]
        current_time = _to_utc(now or datetime.now(timezone.utc))
        requested_start = self._requested_history_start(current_time)
        mt5_symbol = self.resolve_symbol(market)
        mt5 = self._get_mt5()

        if not mt5.symbol_select(mt5_symbol, True):
            raise RuntimeError(f"Cannot select MT5 symbol: {mt5_symbol}")

        merged = _merge_candles((), local_data)
        backfilled_count = 0
        appended_count = 0

        if not merged:
            downloaded = self.download_mt5_history(
                mt5_symbol,
                normalized_timeframe,
                requested_start,
                current_time,
            )
            merged = _merge_candles((), downloaded)
            appended_count = len(merged)
        else:
            first_timestamp = int(merged[0]["time"])
            requested_timestamp = int(requested_start.timestamp())
            if first_timestamp - requested_timestamp > duration:
                backfill_end = datetime.fromtimestamp(
                    first_timestamp - 1,
                    tz=timezone.utc,
                )
                if requested_start < backfill_end:
                    backfill = self.download_mt5_history(
                        mt5_symbol,
                        normalized_timeframe,
                        requested_start,
                        backfill_end,
                    )
                    before_count = len(merged)
                    merged = _merge_candles(backfill, merged)
                    backfilled_count = max(len(merged) - before_count, 0)

            last_timestamp = int(merged[-1]["time"])
            sync_from = datetime.fromtimestamp(
                last_timestamp + duration,
                tz=timezone.utc,
            )
            if sync_from <= current_time:
                recent = self.download_mt5_history(
                    mt5_symbol,
                    normalized_timeframe,
                    sync_from,
                    current_time,
                )
                before_count = len(merged)
                merged = _merge_candles(merged, recent)
                appended_count = max(len(merged) - before_count, 0)

        if merged != list(local_data):
            self.save_local_history(path, merged)

        return merged, backfilled_count, appended_count

    # ========================================================================
    # HISTORY LOADING AND RUNTIME ATTACHMENT
    # ========================================================================

    def load_history(
        self,
        market: str,
        timeframe: str,
        runtime: Optional[RuntimeContext] = None,
    ) -> List[Dict[str, Any]]:
        """Load local history, synchronize it when possible, and return it."""
        normalized_market = normalize_market(market)
        normalized_timeframe = normalize_timeframe(timeframe)

        if runtime is not None and runtime.market != normalized_market:
            raise ValueError(
                "Runtime market does not match the requested chart market."
            )

        if self.web_mode and normalized_timeframe in WEB_UNSUPPORTED_TIMEFRAMES:
            raise RuntimeError(
                f"{normalized_timeframe} is not available in WebMode."
            )

        path = self.history_path(normalized_market, normalized_timeframe)
        self._set_runtime_progress(
            runtime,
            "Loading market data",
            5,
            f"Loading local {normalized_market} {normalized_timeframe} history...",
        )

        local_data = self.load_local_history(path)
        local_count = len(local_data)

        print()
        print("========================================")
        print("LOCAL CHART HISTORY")
        print("========================================")
        print("MARKET:    ", normalized_market)
        print("TIMEFRAME: ", normalized_timeframe)
        print("FILE:      ", path)
        print("LOCAL:     ", f"{local_count:,} candles")
        print("========================================")
        print()

        if self.web_mode:
            if not local_data:
                raise RuntimeError(
                    f"No local chart history available for "
                    f"{normalized_market} {normalized_timeframe}."
                )
            print(
                f"WebMode loaded {len(local_data):,} "
                f"{normalized_market} {normalized_timeframe} candles from CHARTS."
            )
            self._attach_runtime_data(runtime, normalized_timeframe, local_data)
            return local_data

        if self.mt5_connected:
            current_time = datetime.now(timezone.utc)
            try:
                self._set_runtime_progress(
                    runtime,
                    "Synchronizing market history",
                    30,
                    f"Synchronizing {normalized_market} {normalized_timeframe} history...",
                )
                local_data, backfilled, appended = self._synchronize_archive(
                    normalized_market,
                    normalized_timeframe,
                    path,
                    local_data,
                    now=current_time,
                )
                if backfilled or appended:
                    print(
                        f"Synchronized {normalized_market} {normalized_timeframe}: "
                        f"{backfilled:,} backfilled and {appended:,} new candles."
                    )
                elif local_data:
                    print("Chart archive is already up to date.")
            except Exception as error:
                if local_data:
                    print(
                        f"MT5 sync warning for {normalized_market} "
                        f"{normalized_timeframe}: {error}"
                    )
                    print("Using the existing local chart archive.")
                else:
                    raise
        elif not local_data:
            raise RuntimeError(
                f"No chart history is available for {normalized_market} "
                f"{normalized_timeframe}, and MT5 is not connected."
            )

        if not local_data:
            raise RuntimeError(
                f"No chart history available for "
                f"{normalized_market} {normalized_timeframe}."
            )

        self._set_runtime_progress(
            runtime,
            "Preparing chart data",
            75,
            f"Preparing {len(local_data):,} {normalized_timeframe} candles...",
        )
        self._attach_runtime_data(runtime, normalized_timeframe, local_data)

        self._set_runtime_progress(
            runtime,
            "Chart ready",
            100,
            f"{normalized_market} {normalized_timeframe} ready. "
            f"{len(local_data):,} candles loaded.",
        )
        print(
            f"Loaded {len(local_data):,} {normalized_market} "
            f"{normalized_timeframe} candles from local archive."
        )
        return local_data

    @staticmethod
    def _attach_runtime_data(
        runtime: Optional[RuntimeContext],
        timeframe: str,
        candles: List[Dict[str, Any]],
    ) -> None:
        """Attach an independent candle list to one runtime."""
        if runtime is None:
            return

        runtime_candles = [dict(candle) for candle in candles]
        runtime.set_chart_data(runtime_candles, timeframe)

        if timeframe == "M1":
            with runtime.lock:
                runtime.m1_data = runtime_candles

    def load_runtime_history(
        self,
        runtime: RuntimeContext,
        timeframe: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load the active or requested timeframe directly into a runtime."""
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be a RuntimeContext instance.")

        selected_timeframe = normalize_timeframe(
            timeframe if timeframe is not None else runtime.active_timeframe
        )
        return self.load_history(
            market=runtime.market,
            timeframe=selected_timeframe,
            runtime=runtime,
        )

    def switch_timeframe(
        self,
        runtime: RuntimeContext,
        new_timeframe: str,
    ) -> List[Dict[str, Any]]:
        """Load and activate another timeframe within the same TradingBot."""
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be a RuntimeContext instance.")

        normalized_timeframe = normalize_timeframe(new_timeframe)
        if self.web_mode and normalized_timeframe in WEB_UNSUPPORTED_TIMEFRAMES:
            raise RuntimeError(
                f"{normalized_timeframe} is not available in WebMode."
            )

        candles = self.load_history(
            market=runtime.market,
            timeframe=normalized_timeframe,
            runtime=None,
        )
        runtime.set_active_timeframe(normalized_timeframe)
        self._attach_runtime_data(runtime, normalized_timeframe, candles)

        self._set_runtime_progress(
            runtime,
            "Chart ready",
            100,
            f"{runtime.market} {normalized_timeframe} ready. "
            f"{len(candles):,} candles loaded.",
        )
        return candles

    # ========================================================================
    # ARCHIVE DISCOVERY
    # ========================================================================

    def list_available_markets(self) -> List[str]:
        """List market folders that contain at least one candle archive."""
        if not self.charts_root.is_dir():
            return []

        markets: List[str] = []
        for folder in self.charts_root.iterdir():
            if not folder.is_dir():
                continue
            if any(
                (folder / timeframe / LOCAL_CHART_FILENAME).is_file()
                for timeframe in SUPPORTED_TIMEFRAMES
            ):
                markets.append(folder.name)
        return sorted(markets)

    def list_available_timeframes(self, market: str) -> List[str]:
        """List runtime-supported timeframes with an existing candle archive."""
        normalized_market = normalize_market(market)
        available: List[str] = []

        for timeframe in SUPPORTED_TIMEFRAMES:
            path = self.history_path(normalized_market, timeframe)
            if path.is_file():
                available.append(timeframe)
        return available

    def get_archive_status(self, market: str, timeframe: str) -> Dict[str, Any]:
        """Return archive coverage and basic integrity information."""
        normalized_market = normalize_market(market)
        normalized_timeframe = normalize_timeframe(timeframe)
        path = self.history_path(normalized_market, normalized_timeframe)
        candles = self.load_local_history(path)

        if not candles:
            return {
                "market": normalized_market,
                "timeframe": normalized_timeframe,
                "path": str(path),
                "exists": path.is_file(),
                "candle_count": 0,
                "first_candle_time": None,
                "last_candle_time": None,
                "is_chronological": True,
                "is_deduplicated": True,
            }

        timestamps = [int(candle["time"]) for candle in candles]
        return {
            "market": normalized_market,
            "timeframe": normalized_timeframe,
            "path": str(path),
            "exists": True,
            "candle_count": len(candles),
            "first_candle_time": timestamps[0],
            "last_candle_time": timestamps[-1],
            "is_chronological": timestamps == sorted(timestamps),
            "is_deduplicated": len(timestamps) == len(set(timestamps)),
        }

    # ========================================================================
    # BULK ARCHIVE SYNCHRONIZATION
    # ========================================================================

    def update_all_chart_data(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronize all existing local archives with the connected MT5 feed."""
        if self.web_mode or not self.mt5_connected:
            return {
                "success": False,
                "updated": 0,
                "backfilled": 0,
                "charts_updated": 0,
                "charts_checked": 0,
                "errors": [],
                "message": "MT5 is unavailable. Chart archives were not updated.",
            }

        if not self.charts_root.is_dir():
            return {
                "success": True,
                "updated": 0,
                "backfilled": 0,
                "charts_updated": 0,
                "charts_checked": 0,
                "errors": [],
                "message": "No local chart archives were found.",
            }

        market_folders = sorted(
            folder
            for folder in self.charts_root.iterdir()
            if folder.is_dir()
        )
        total_markets = max(len(market_folders), 1)

        updated_candles = 0
        backfilled_candles = 0
        updated_charts = 0
        checked_charts = 0
        errors: List[Dict[str, str]] = []

        def report(stage: str, progress: int, message: str) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "startup_stage": stage,
                    "startup_progress": max(0, min(100, int(progress))),
                    "startup_message": message,
                }
            )

        report("Updating chart data", 5, "Updating shared market archives...")

        for market_index, market_path in enumerate(market_folders, start=1):
            market_folder = market_path.name
            try:
                normalized_market = normalize_market(market_folder)
                mt5_symbol = self.resolve_symbol(normalized_market)
            except Exception as error:
                errors.append(
                    {"market": market_folder, "timeframe": "", "error": str(error)}
                )
                continue

            mt5 = self._get_mt5()
            if not mt5.symbol_select(mt5_symbol, True):
                errors.append(
                    {
                        "market": normalized_market,
                        "timeframe": "",
                        "error": f"Cannot select MT5 symbol: {mt5_symbol}",
                    }
                )
                continue

            for timeframe in SUPPORTED_TIMEFRAMES:
                archive_path = self.history_path(normalized_market, timeframe)
                if not archive_path.is_file():
                    continue

                checked_charts += 1
                try:
                    local_data = self.load_local_history(archive_path)
                    if not local_data:
                        continue

                    merged, backfilled, appended = self._synchronize_archive(
                        normalized_market,
                        timeframe,
                        archive_path,
                        local_data,
                    )
                    if backfilled or appended:
                        backfilled_candles += backfilled
                        updated_candles += appended
                        updated_charts += 1
                        print(
                            f"[UPDATE] {normalized_market} {timeframe}: "
                            f"{backfilled:,} backfilled, {appended:,} new candles"
                        )
                except Exception as error:
                    errors.append(
                        {
                            "market": normalized_market,
                            "timeframe": timeframe,
                            "error": str(error),
                        }
                    )
                    print(
                        f"[UPDATE] {normalized_market} {timeframe} failed: {error}"
                    )

            progress = 5 + int((market_index / total_markets) * 90)
            report(
                "Updating chart data",
                progress,
                f"Processed market {market_index}/{len(market_folders)}: "
                f"{normalized_market}",
            )

        success = not errors
        message = (
            f"Chart update complete. {updated_candles:,} new candles and "
            f"{backfilled_candles:,} backfilled candles added."
        )
        if errors:
            message += f" {len(errors)} archive synchronization error(s)."

        report("Chart ready", 100, message)
        return {
            "success": success,
            "updated": updated_candles,
            "backfilled": backfilled_candles,
            "charts_updated": updated_charts,
            "charts_checked": checked_charts,
            "errors": errors,
            "message": message,
        }


# ============================================================================
# DEFAULT SERVICE AND PUBLIC HELPERS
# ============================================================================

chart_service = ChartService()


def initialize_mt5(default_market: Optional[str] = None) -> bool:
    """Initialize the default chart service's MT5 connection."""
    return chart_service.initialize_mt5(default_market)


def shutdown_mt5() -> None:
    """Shut down the default chart service's MT5 connection."""
    chart_service.shutdown_mt5()


def resolve_symbol(symbol: str) -> str:
    """Resolve a market using the default chart service's MT5 connection."""
    return chart_service.resolve_symbol(symbol)


def load_history(runtime: RuntimeContext) -> List[Dict[str, Any]]:
    """Load the active market and timeframe into one TradingBot runtime."""
    return chart_service.load_runtime_history(runtime)


def load_market_history(
    market: str,
    timeframe: str,
    runtime: Optional[RuntimeContext] = None,
) -> List[Dict[str, Any]]:
    """Load history for a market, optionally attaching it to its runtime."""
    return chart_service.load_history(
        market=market,
        timeframe=timeframe,
        runtime=runtime,
    )


def switch_timeframe(
    runtime: RuntimeContext,
    new_timeframe: str,
) -> List[Dict[str, Any]]:
    """Switch one TradingBot runtime to another chart timeframe."""
    return chart_service.switch_timeframe(runtime, new_timeframe)


def update_all_chart_data(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Update all known shared market archives with the default service."""
    return chart_service.update_all_chart_data(progress_callback)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ChartService",
    "YEARS_BACK",
    "DOWNLOAD_CHUNK_DAYS",
    "TIMEFRAME_SECONDS",
    "WEB_UNSUPPORTED_TIMEFRAMES",
    "chart_service",
    "initialize_mt5",
    "shutdown_mt5",
    "resolve_symbol",
    "load_history",
    "load_market_history",
    "switch_timeframe",
    "update_all_chart_data",
]
