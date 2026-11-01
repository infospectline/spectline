"""Shared market-chart history and MetaTrader 5 synchronization for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import csv
import importlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


# ============================================================================
# CHART SERVICE
# ============================================================================

class ChartService:
    """Load shared market history and synchronize local archives with MT5.

    Historical candle files are shared by market and timeframe. A runtime gets
    its own in-memory list, so one TradingBot cannot mutate another bot's data.
    The service does not own the active market or timeframe; those belong to
    the selected RuntimeContext.
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

    # ------------------------------------------------------------------------
    # MetaTrader 5
    # ------------------------------------------------------------------------

    def _get_mt5(self) -> Any:
        """Return the MT5 module, importing it lazily in local mode."""
        if self.web_mode:
            raise RuntimeError("MetaTrader 5 is disabled in WEB_MODE.")

        if self.mt5 is None:
            try:
                self.mt5 = importlib.import_module("MetaTrader5")
            except ImportError as error:
                raise RuntimeError(
                    "MetaTrader5 is not installed in this environment."
                ) from error

        return self.mt5

    def initialize_mt5(self, default_market: Optional[str] = None) -> bool:
        """Initialize the local MT5 connection when running outside web mode."""
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
                if item.name.upper() == requested.upper()
            ]
            if exact_matches:
                return exact_matches[0]

            prefix_matches = [
                item.name
                for item in symbols
                if item.name.upper().startswith(requested.upper())
            ]
            if prefix_matches:
                return prefix_matches[0]

        raise RuntimeError(
            f"MT5 symbol '{requested}' was not found. "
            "Check the symbol name in Market Watch."
        )

    def _mt5_timeframe(self, timeframe: str) -> Any:
        """Return the MT5 timeframe constant for a normalized timeframe."""
        mt5 = self._get_mt5()
        attribute_name = f"TIMEFRAME_{timeframe}"
        value = getattr(mt5, attribute_name, None)

        if value is None:
            raise RuntimeError(
                f"MT5 does not expose the {attribute_name} constant."
            )

        return value

    # ------------------------------------------------------------------------
    # Paths and candle serialization
    # ------------------------------------------------------------------------

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
        """Convert an MT5 candle row into the local candle representation."""
        try:
            volume = row["tick_volume"]
        except (KeyError, IndexError, TypeError, ValueError):
            volume = row["volume"]

        return {
            "time": int(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(volume),
        }

    @staticmethod
    def load_local_history(path: Path | str) -> List[Dict[str, Any]]:
        """Load and chronologically sort valid rows from a local candle CSV."""
        archive_path = Path(path)
        if not archive_path.is_file():
            return []

        data: List[Dict[str, Any]] = []

        with archive_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                try:
                    candle = {
                        "time": int(row["time"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(float(row["volume"])),
                    }
                except (KeyError, TypeError, ValueError):
                    continue

                data.append(candle)

        data.sort(key=lambda candle: candle["time"])
        return data

    @staticmethod
    def save_local_history(
        path: Path | str,
        data: Sequence[Dict[str, Any]],
    ) -> None:
        """Atomically write candles to a CSV archive."""
        archive_path = Path(path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = archive_path.with_name(f"{archive_path.name}.tmp")

        fieldnames = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        os.replace(temp_path, archive_path)

    # ------------------------------------------------------------------------
    # MT5 history download
    # ------------------------------------------------------------------------

    @staticmethod
    def filter_closed_candles(
        rates: Any,
        timeframe: str,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Exclude the currently forming candle from downloaded MT5 rates."""
        if rates is None:
            return []

        normalized_timeframe = normalize_timeframe(timeframe)
        timeframe_seconds = TIMEFRAME_SECONDS[normalized_timeframe]
        current_time = now or datetime.now(timezone.utc)
        now_timestamp = int(current_time.timestamp())
        current_candle_start = (
            now_timestamp // timeframe_seconds
        ) * timeframe_seconds

        result: List[Dict[str, Any]] = []
        for row in rates:
            candle = ChartService.candle_to_dict(row)
            if candle["time"] < current_candle_start:
                result.append(candle)

        return result

    def download_mt5_history(
        self,
        market: str,
        timeframe: str,
        utc_from: datetime,
        utc_to: datetime,
    ) -> List[Dict[str, Any]]:
        """Download MT5 history in bounded date chunks and deduplicate by time."""
        mt5_symbol = str(market).strip()
        if not mt5_symbol:
            raise ValueError("MT5 market symbol must not be empty.")
        normalized_timeframe = normalize_timeframe(timeframe)

        if not self.mt5_connected:
            raise RuntimeError("MT5 is not connected.")
        if utc_from.tzinfo is None or utc_to.tzinfo is None:
            raise ValueError("MT5 download datetimes must be timezone-aware.")
        if utc_from > utc_to:
            return []

        mt5 = self._get_mt5()
        mt5_timeframe = self._mt5_timeframe(normalized_timeframe)
        result_by_time: Dict[int, Dict[str, Any]] = {}

        cursor = utc_from.astimezone(timezone.utc)
        final_time = utc_to.astimezone(timezone.utc)
        chunk_delta = timedelta(days=self.download_chunk_days)

        while cursor <= final_time:
            chunk_end = min(cursor + chunk_delta, final_time)
            rates = mt5.copy_rates_range(
                mt5_symbol,
                mt5_timeframe,
                cursor,
                chunk_end,
            )

            if rates is None:
                raise RuntimeError(
                    f"MT5 copy_rates_range failed for "
                    f"{mt5_symbol} {normalized_timeframe}: "
                    f"{mt5.last_error()}"
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

    # ------------------------------------------------------------------------
    # Runtime progress helpers
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # History loading and synchronization
    # ------------------------------------------------------------------------

    def load_history(
        self,
        market: str,
        timeframe: str,
        runtime: Optional[RuntimeContext] = None,
    ) -> List[Dict[str, Any]]:
        """Load local history, synchronize it with MT5 when available, and return it.

        When a runtime is supplied, its market must match the requested market.
        The returned candles are attached only to that runtime and timeframe.
        """
        normalized_market = normalize_market(market)
        normalized_timeframe = normalize_timeframe(timeframe)

        if runtime is not None and runtime.market != normalized_market:
            raise ValueError(
                "Runtime market does not match the requested chart market."
            )

        if self.web_mode and normalized_timeframe in WEB_UNSUPPORTED_TIMEFRAMES:
            raise RuntimeError(
                f"{normalized_timeframe} is not available in WEB_MODE."
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
                f"WEB_MODE: loaded {len(local_data):,} "
                f"{normalized_market} {normalized_timeframe} candles from CHARTS."
            )
            self._attach_runtime_data(
                runtime,
                normalized_timeframe,
                local_data,
            )
            return local_data

        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(days=365 * self.years_back)

        try:
            if local_data:
                last_local_timestamp = local_data[-1]["time"]
                timeframe_seconds = TIMEFRAME_SECONDS[normalized_timeframe]
                sync_from_timestamp = last_local_timestamp + timeframe_seconds
                sync_from = datetime.fromtimestamp(
                    sync_from_timestamp,
                    tz=timezone.utc,
                )

                if sync_from <= utc_to:
                    mt5_symbol = self.resolve_symbol(normalized_market)
                    if not self._get_mt5().symbol_select(mt5_symbol, True):
                        raise RuntimeError(
                            f"Cannot select MT5 symbol: {mt5_symbol}"
                        )

                    self._set_runtime_progress(
                        runtime,
                        "Updating local history",
                        35,
                        f"Updating {normalized_market} {normalized_timeframe} "
                        "with new MT5 candles...",
                    )

                    new_data = self.download_mt5_history(
                        mt5_symbol,
                        normalized_timeframe,
                        sync_from,
                        utc_to,
                    )

                    if new_data:
                        existing_times = {
                            candle["time"] for candle in local_data
                        }
                        added_data = [
                            candle
                            for candle in new_data
                            if candle["time"] not in existing_times
                        ]

                        if added_data:
                            local_data.extend(added_data)
                            local_data.sort(
                                key=lambda candle: candle["time"]
                            )
                            self.save_local_history(path, local_data)

                        print(
                            f"Added {len(added_data):,} new "
                            f"{normalized_market} {normalized_timeframe} candles."
                        )
                    else:
                        print("No new MT5 candles to add.")
            else:
                self._set_runtime_progress(
                    runtime,
                    "Creating local chart history",
                    10,
                    f"Downloading {self.years_back} years of "
                    f"{normalized_market} {normalized_timeframe} history...",
                )

                mt5_symbol = self.resolve_symbol(normalized_market)
                if not self._get_mt5().symbol_select(mt5_symbol, True):
                    raise RuntimeError(
                        f"Cannot select MT5 symbol: {mt5_symbol}"
                    )

                local_data = self.download_mt5_history(
                    mt5_symbol,
                    normalized_timeframe,
                    utc_from,
                    utc_to,
                )

                if not local_data:
                    raise RuntimeError(
                        f"No MT5 data returned for "
                        f"{normalized_market} {normalized_timeframe}."
                    )

                self.save_local_history(path, local_data)
                print(
                    f"Created local archive with {len(local_data):,} candles."
                )

        except Exception as error:
            if local_data:
                print(
                    f"MT5 sync warning for "
                    f"{normalized_market} {normalized_timeframe}: {error}"
                )
                print("Using existing local chart history.")
            else:
                raise

        if not local_data:
            raise RuntimeError(
                f"No chart history available for "
                f"{normalized_market} {normalized_timeframe}."
            )

        self._set_runtime_progress(
            runtime,
            "Preparing chart data",
            70,
            f"Preparing {len(local_data):,} "
            f"{normalized_timeframe} candles from local archive...",
        )

        self._attach_runtime_data(
            runtime,
            normalized_timeframe,
            local_data,
        )

        self._set_runtime_progress(
            runtime,
            "Chart ready",
            100,
            f"{normalized_market} {normalized_timeframe} ready. "
            f"{len(local_data):,} candles loaded from local archive.",
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
                f"{normalized_timeframe} is not available in WEB_MODE."
            )

        candles = self.load_history(
            market=runtime.market,
            timeframe=normalized_timeframe,
            runtime=None,
        )

        runtime.set_active_timeframe(normalized_timeframe)
        self._attach_runtime_data(
            runtime,
            normalized_timeframe,
            candles,
        )

        self._set_runtime_progress(
            runtime,
            "Chart ready",
            100,
            f"{runtime.market} {normalized_timeframe} ready. "
            f"{len(candles):,} candles loaded from local archive.",
        )

        return candles

    def list_available_markets(self) -> List[str]:
        """List market folders that already contain local chart archives."""
        if not self.charts_root.is_dir():
            return []

        markets = [
            folder.name
            for folder in self.charts_root.iterdir()
            if folder.is_dir()
        ]
        return sorted(markets)

    def list_available_timeframes(self, market: str) -> List[str]:
        """List timeframes with a local candle archive for the given market."""
        normalized_market = normalize_market(market)
        market_root = self.charts_root / normalized_market

        if not market_root.is_dir():
            return []

        available = []
        for timeframe in sorted(SUPPORTED_TIMEFRAMES):
            path = self.history_path(normalized_market, timeframe)
            if path.is_file():
                available.append(timeframe)

        return available

    # ------------------------------------------------------------------------
    # Archive synchronization
    # ------------------------------------------------------------------------

    def update_all_chart_data(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronize all existing local market/timeframe archives with MT5."""
        if not self.mt5_connected:
            return {
                "success": False,
                "updated": 0,
                "charts_updated": 0,
                "charts_checked": 0,
                "message": "MT5 is switched off. Data are not updated.",
            }

        if not self.charts_root.is_dir():
            return {
                "success": True,
                "updated": 0,
                "charts_updated": 0,
                "charts_checked": 0,
                "message": "No local charts found.",
            }

        updated_candles = 0
        updated_charts = 0
        checked_charts = 0

        def report(stage: str, progress: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback({
                    "startup_stage": stage,
                    "startup_progress": progress,
                    "startup_message": message,
                })

        report(
            "Updating chart data",
            5,
            "Updating all local chart data...",
        )

        market_folders = sorted(
            folder
            for folder in self.charts_root.iterdir()
            if folder.is_dir()
        )

        total_markets = max(len(market_folders), 1)

        for market_index, market_path in enumerate(market_folders, start=1):
            market_folder = market_path.name

            try:
                resolved_market = self.resolve_symbol(market_folder)
                mt5 = self._get_mt5()

                if not mt5.symbol_select(resolved_market, True):
                    print(
                        f"Skipping {market_folder}: "
                        "MT5 symbol could not be selected."
                    )
                    continue

            except Exception as error:
                print(f"Skipping {market_folder}: {error}")
                continue

            for timeframe in sorted(SUPPORTED_TIMEFRAMES):
                if self.web_mode and timeframe in WEB_UNSUPPORTED_TIMEFRAMES:
                    continue

                history_path = self.history_path(market_folder, timeframe)
                if not history_path.is_file():
                    continue

                checked_charts += 1
                local_data = self.load_local_history(history_path)
                if not local_data:
                    continue

                last_local_timestamp = local_data[-1]["time"]
                sync_from_timestamp = (
                    last_local_timestamp + TIMEFRAME_SECONDS[timeframe]
                )
                utc_to = datetime.now(timezone.utc)

                if sync_from_timestamp > int(utc_to.timestamp()):
                    continue

                sync_from = datetime.fromtimestamp(
                    sync_from_timestamp,
                    tz=timezone.utc,
                )

                try:
                    new_data = self.download_mt5_history(
                        resolved_market,
                        timeframe,
                        sync_from,
                        utc_to,
                    )
                    if not new_data:
                        continue

                    existing_times = {
                        candle["time"] for candle in local_data
                    }
                    added_data = [
                        candle
                        for candle in new_data
                        if candle["time"] not in existing_times
                    ]

                    if not added_data:
                        continue

                    local_data.extend(added_data)
                    local_data.sort(key=lambda candle: candle["time"])
                    self.save_local_history(history_path, local_data)

                    updated_candles += len(added_data)
                    updated_charts += 1

                    print(
                        f"[UPDATE] {market_folder} {timeframe}: "
                        f"+{len(added_data)} candles"
                    )

                except Exception as error:
                    print(
                        f"[UPDATE] {market_folder} {timeframe} failed: {error}"
                    )

            progress = 5 + int((market_index / total_markets) * 90)
            report(
                "Updating chart data",
                progress,
                f"Updated market {market_index}/{len(market_folders)}: "
                f"{market_folder}",
            )

        message = (
            f"Data update complete. {updated_candles:,} new candles added."
        )
        report("Chart ready", 100, message)

        return {
            "success": True,
            "updated": updated_candles,
            "charts_updated": updated_charts,
            "charts_checked": checked_charts,
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
    """Load the active market/timeframe history into one TradingBot runtime."""
    return chart_service.load_runtime_history(runtime)


def load_market_history(
    market: str,
    timeframe: str,
    runtime: Optional[RuntimeContext] = None,
) -> List[Dict[str, Any]]:
    """Load history for a market, optionally attaching it to its runtime.

    A market switch is coordinated by the backend: it resolves the new client
    identity, obtains that identity's runtime, and then calls this function.
    This module never mutates a runtime's immutable market identity.
    """
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
    """Update all known shared market archives using the default service."""
    return chart_service.update_all_chart_data(progress_callback)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ChartService",
    "TIMEFRAME_SECONDS",
    "chart_service",
    "initialize_mt5",
    "shutdown_mt5",
    "resolve_symbol",
    "load_history",
    "load_market_history",
    "switch_timeframe",
    "update_all_chart_data",
]
