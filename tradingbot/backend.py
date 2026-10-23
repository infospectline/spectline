# backend.py

import asyncio
import copy
import csv
import json
import math
import os
import re
import ai
import base64
import hashlib
import hmac
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

WEB_MODE = os.getenv("WEB_MODE", "false").lower() == "true"

def verify_tradingbot_ticket(ticket: str) -> Optional[str]:
    secret = os.getenv("TRADINGBOT_SHARED_SECRET")
    if not secret or not ticket:
        return None

    try:
        payload_part, signature_part = ticket.split(".", 1)

        expected_signature = hmac.new(
            secret.encode("utf-8"),
            payload_part.encode("ascii"),
            hashlib.sha256,
        ).digest()

        def decode_base64url(value: str) -> bytes:
            return base64.urlsafe_b64decode(
                value + "=" * (-len(value) % 4)
            )

        supplied_signature = decode_base64url(signature_part)

        if not hmac.compare_digest(
            expected_signature,
            supplied_signature,
        ):
            return None

        payload = json.loads(
            decode_base64url(payload_part)
        )

        user_id = payload.get("sub")
        expires_at = int(payload.get("exp", 0))

        if payload.get("aud") != "spectline-tradingbot":
            return None

        if not isinstance(user_id, str) or not user_id.strip():
            return None

        if expires_at <= int(time.time()):
            return None

        return user_id

    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return None

if WEB_MODE:
    mt5 = None
else:
    import MetaTrader5 as mt5
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from contextlib import asynccontextmanager

# Fixed strategy configuration
SYMBOL = "EURUSD"
YEARS_BACK = 5
STARTING_BALANCE = 100_000.0

TIMEFRAME = "H1"
mt5_connected = False

if WEB_MODE:
    MT5_TIMEFRAMES = {
        "M1": None,
        "M5": None,
        "M15": None,
        "M30": None,
        "H1": None,
        "H4": None,
        "D1": None,
        "W1": None,
    }
else:
    MT5_TIMEFRAMES = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }

analysis_task = None
blocked_trades = 0
blocked_trade_signal = False

SL_PERCENT = 0.005
TP_PERCENT = 0.010
MAX_DRAWDOWN_PERCENT = 0.04
LEVEL_DISTANCE_THRESHOLD = 0.50


m1_data: List[Dict[str, Any]] = []
h1_data: List[Dict[str, Any]] = []

current_index = 0
running = False
trading_enabled = True

balance = STARTING_BALANCE
peak_equity = STARTING_BALANCE

open_trade: Optional[Dict[str, Any]] = None
trade_history: List[Dict[str, Any]] = []
last_trade_result: Optional[str] = None
major_levels: List[Dict[str, Any]] = []

connected_clients: Dict[Optional[str], set] = {}

startup_stage = "Starting"
startup_progress = 0
startup_message = "Starting backend..."

zigzag_filter: Optional[Dict[str, float]] = None

LEVEL_RECALC_INTERVAL = 20

cached_strategy_levels = []
last_level_calculation_index = -1

blocked_trade_zone = None

# ---------------------------------------------------------------------------
# SESSION MANAGEMENT
# ---------------------------------------------------------------------------
# One market is one logical session. Each timeframe inside that market has
# its own analysis state, levels and trades. The currently active session is
# mirrored by the legacy globals below so the existing strategy code can stay
# unchanged.

SESSIONS: Dict[str, Dict[str, Dict[str, Any]]] = {}

analysis_status = "idle"
analysis_started = False
analysis_completed = False

results_source = "testing"
manual_analysis_results: Dict[str, Any] = {}

manual_analysis_progress = 0

USER_MANUAL_ANALYSIS: Dict[
    Optional[str],
    Dict[str, Any]
] = {}

USER_MANUAL_ANALYSIS_TASKS: Dict[
    Optional[str],
    asyncio.Task
] = {}


def _new_manual_analysis_state() -> Dict[str, Any]:
    return {
        "analysis_status": "idle",
        "analysis_started": False,
        "analysis_completed": False,
        "analysis_phase": "idle",
        "manual_analysis_progress": 0,
        "results_source": "testing",
        "manual_analysis_results": {},
        "startup_stage": "Waiting for analysis",
        "startup_progress": 0,
        "startup_message": "",
    }


def _get_manual_analysis_state(
    user_id: Optional[str]
) -> Dict[str, Any]:

    if user_id not in USER_MANUAL_ANALYSIS:
        USER_MANUAL_ANALYSIS[user_id] = (
            _new_manual_analysis_state()
        )

    return USER_MANUAL_ANALYSIS[user_id]

MANUAL_RISK_PERCENT = 1.0

def _new_session_state() -> Dict[str, Any]:
    return {
        "current_index": 0,
        "running": False,
        "trading_enabled": True,
        "balance": STARTING_BALANCE,
        "peak_equity": STARTING_BALANCE,
        "open_trade": None,
        "trade_history": [],
        "last_trade_result": None,
        "major_levels": [],
        "cached_strategy_levels": [],
        "last_level_calculation_index": -1,
        "blocked_trade_zone": None,
        "zigzag_filter": None,
        "blocked_trades": 0,
        "analysis_status": "idle",
        "analysis_started": False,
        "analysis_completed": False,
        "analysis_phase": "idle",
        "model_available": False,
        "model_info": {},
        "last_training_info": {},
        "last_model_signal": None,
        "last_model_entry_index": -10_000_000,
    }


def _get_session_state(
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    market_sessions = SESSIONS.setdefault(symbol, {})

    if timeframe not in market_sessions:
        market_sessions[timeframe] = _new_session_state()

    return market_sessions[timeframe]


def _capture_active_session() -> None:
    state = _get_session_state(SYMBOL, TIMEFRAME)

    state["current_index"] = current_index
    state["running"] = False
    state["trading_enabled"] = trading_enabled
    state["balance"] = balance
    state["peak_equity"] = peak_equity
    state["open_trade"] = copy.deepcopy(open_trade)
    state["trade_history"] = copy.deepcopy(trade_history)
    state["last_trade_result"] = last_trade_result
    state["major_levels"] = copy.deepcopy(major_levels)
    state["cached_strategy_levels"] = copy.deepcopy(cached_strategy_levels)
    state["last_level_calculation_index"] = last_level_calculation_index
    state["blocked_trade_zone"] = copy.deepcopy(blocked_trade_zone)
    state["zigzag_filter"] = copy.deepcopy(zigzag_filter)
    state["blocked_trades"] = blocked_trades
    state["analysis_status"] = analysis_status
    state["analysis_started"] = analysis_started
    state["analysis_completed"] = analysis_completed
    state["analysis_phase"] = analysis_phase
    state["model_available"] = model_available
    state["model_info"] = copy.deepcopy(model_info)
    state["last_training_info"] = copy.deepcopy(last_training_info)
    state["last_model_signal"] = copy.deepcopy(last_model_signal)
    state["last_model_entry_index"] = last_model_entry_index


def _restore_active_session() -> None:
    global current_index, running, trading_enabled
    global balance, peak_equity, open_trade, trade_history
    global last_trade_result, major_levels
    global cached_strategy_levels, last_level_calculation_index
    global blocked_trade_zone, zigzag_filter, blocked_trades
    global analysis_status, analysis_started, analysis_completed
    global analysis_phase, model_available, model_info
    global last_training_info, last_model_signal, last_model_entry_index

    state = _get_session_state(SYMBOL, TIMEFRAME)

    current_index = min(
        max(0, int(state.get("current_index", 0))),
        max(0, len(h1_data) - 1),
    )
    running = False
    trading_enabled = bool(state.get("trading_enabled", True))
    balance = float(state.get("balance", STARTING_BALANCE))
    peak_equity = float(state.get("peak_equity", STARTING_BALANCE))
    open_trade = copy.deepcopy(state.get("open_trade"))
    trade_history = copy.deepcopy(state.get("trade_history", []))
    last_trade_result = state.get("last_trade_result")
    major_levels = copy.deepcopy(state.get("major_levels", []))
    cached_strategy_levels = copy.deepcopy(
        state.get("cached_strategy_levels", [])
    )
    last_level_calculation_index = int(
        state.get("last_level_calculation_index", -1)
    )
    blocked_trade_zone = copy.deepcopy(
        state.get("blocked_trade_zone")
    )
    zigzag_filter = copy.deepcopy(state.get("zigzag_filter"))
    blocked_trades = int(state.get("blocked_trades", 0))
    analysis_status = state.get("analysis_status", "idle")
    analysis_started = bool(state.get("analysis_started", False))
    analysis_completed = bool(state.get("analysis_completed", False))
    analysis_phase = state.get("analysis_phase", "idle")
    model_available = bool(state.get("model_available", False))
    model_info = copy.deepcopy(state.get("model_info", {}))
    last_training_info = copy.deepcopy(state.get("last_training_info", {}))
    last_model_signal = copy.deepcopy(state.get("last_model_signal"))
    last_model_entry_index = int(state.get("last_model_entry_index", -10_000_000))

    # A cancelled replay is never considered running when a session is
    # re-entered. Its previous point-in-time state is preserved.
    if analysis_status == "running":
        analysis_status = "paused"
        state["analysis_status"] = "paused"


def _mark_session_dirty() -> None:
    _capture_active_session()


def _session_id() -> str:
    return f"{SYMBOL}|{TIMEFRAME}"


# MT5 data

def resolve_symbol(symbol: str) -> str:
    if mt5.symbol_info(symbol) is not None:
        return symbol

    symbols = mt5.symbols_get()

    if symbols:
        exact_matches = [
            s.name
            for s in symbols
            if s.name.upper() == symbol.upper()
        ]

        if exact_matches:
            return exact_matches[0]

        prefix_matches = [
            s.name
            for s in symbols
            if s.name.upper().startswith(symbol.upper())
        ]

        if prefix_matches:
            return prefix_matches[0]

    raise RuntimeError(
        f"MT5 symbol '{symbol}' was not found. "
        "Check the symbol name in Market Watch."
    )

def initialize_mt5() -> bool:
    global SYMBOL
    global mt5_connected

    mt5_connected = False

    if not mt5.initialize():
        print(
            "MT5 is not available. "
            "Application will use local CHARTS data."
        )
        return False

    try:
        global_symbol = resolve_symbol(SYMBOL)
        SYMBOL = global_symbol

        if not mt5.symbol_select(global_symbol, True):
            print(
                f"MT5 symbol selection failed for {global_symbol}. "
                "Using local CHARTS data."
            )
            mt5.shutdown()
            return False

        mt5_connected = True

        print("MT5 connected.")
        return True

    except Exception as error:
        print(
            f"MT5 connection setup failed: {error}"
        )

        mt5.shutdown()
        mt5_connected = False
        return False
# ---------------------------------------------------------------------------
# LOCAL CHART HISTORY
# ---------------------------------------------------------------------------
# Market data is permanently archived next to backend.py:
#
# CHARTS\
#   EURUSD\
#       H1\
#           candles.csv
#       M15\
#           candles.csv
#   XAUUSD\
#       H1\
#           candles.csv
#
# MT5 is used only to create/synchronize the archive. The chart itself loads
# its history from the local archive.

CHARTS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "CHARTS",
)

if WEB_MODE:
    TRADING_SETUP_ROOT = "/data/users"
else:
    TRADING_SETUP_ROOT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "TRADING SET UP"
    )

TRADING_SETUP_FILENAME = "setups.json"

LOCAL_CHART_FILENAME = "candles.csv"

TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}

DOWNLOAD_CHUNK_DAYS = 90


# ---------------------------------------------------------------------------
# MODEL STORAGE
# ---------------------------------------------------------------------------
# Each market has its own isolated model. DEFAULT is always a clean model
# template and is never modified by market-specific training.



analysis_phase = "idle"
model_available = False
model_info: Dict[str, Any] = {}
last_training_info: Dict[str, Any] = {}
last_model_signal: Optional[Dict[str, Any]] = None
last_model_entry_index = -10_000_000

# ---------------------------------------------------------------------------
# AI.PY RUNTIME
# ---------------------------------------------------------------------------

ai_model = None
ai_scaler = None
ai_checkpoint: Dict[str, Any] = {}
ai_market_candles = None



def _chart_symbol_folder(symbol: str) -> str:
    """Return a filesystem-safe symbol folder name."""
    return str(symbol).strip().replace("/", "_").replace("\\", "_")


def _chart_history_path(symbol: str, timeframe: str) -> str:
    folder = os.path.join(
        CHARTS_ROOT,
        _chart_symbol_folder(symbol),
        timeframe,
    )
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, LOCAL_CHART_FILENAME)

# ---------------------------------------------------------------------------
# TRADING SET UP STORAGE
# ---------------------------------------------------------------------------

def _trading_setup_symbol_folder(symbol: str) -> str:
    return (
        str(symbol)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
    )


def _trading_setup_path(symbol: str, user_id: Optional[str] = None) -> str:
    symbol_folder = _trading_setup_symbol_folder(symbol)

    if WEB_MODE:
        if not user_id:
            raise ValueError("user_id is required in WEB_MODE")

        user_root = os.path.join(
            TRADING_SETUP_ROOT,
            str(user_id)
        )
    else:
        user_root = TRADING_SETUP_ROOT

    symbol_path = os.path.join(
        user_root,
        symbol_folder
    )

    os.makedirs(
        symbol_path,
        exist_ok=True
    )

    return os.path.join(
        symbol_path,
        TRADING_SETUP_FILENAME
    )


def _load_trading_setups(
    symbol: str,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:

    path = _trading_setup_path(symbol, user_id)

    if not os.path.isfile(path):
        return []

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        return data

    except (
        OSError,
        json.JSONDecodeError
    ) as error:

        print(
            f"Could not load trading setups from {path}:",
            error
        )

        return []


def _save_trading_setups(
    symbol: str,
    setups: List[Dict[str, Any]],
    user_id: Optional[str] = None
) -> None:

    path = _trading_setup_path(symbol, user_id)
    temp_path = f"{path}.tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            setups,
            file,
            indent=4,
            ensure_ascii=False
        )

    os.replace(
        temp_path,
        path
    )


def _next_trading_setup_id(
    setups: List[Dict[str, Any]]
) -> int:

    ids = []

    for setup in setups:

        try:
            ids.append(
                int(setup.get("id"))
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    return max(ids, default=0) + 1


def save_trading_setup(
    symbol: str,
    setup: Dict[str, Any],
    user_id: Optional[str] = None
) -> Dict[str, Any]:

    setups = _load_trading_setups(symbol, user_id)

    setup = copy.deepcopy(setup)

    rr_mode = setup.get("rr_mode", "OFF")

    if rr_mode not in {"OFF", "2x", "3x"}:
        rr_mode = "OFF"

    setup["rr_mode"] = rr_mode

    try:
        setup_id = int(setup.get("id"))

    except (
        TypeError,
        ValueError
    ):
        setup_id = None

    if setup_id is None:

        setup_id = _next_trading_setup_id(
            setups
        )

        setup["id"] = setup_id

        setups.append(setup)

    else:

        updated = False

        for index, existing in enumerate(setups):

            try:
                existing_id = int(
                    existing.get("id")
                )

            except (
                TypeError,
                ValueError
            ):
                continue

            if existing_id == setup_id:

                setup["id"] = setup_id
                setups[index] = setup
                updated = True
                break

        if not updated:
            setups.append(setup)

    _save_trading_setups(
        symbol,
        setups,
        user_id
    )

    return setup


def get_trading_setups(
    symbol: str,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:

    return _load_trading_setups(symbol, user_id)

def _manual_parse_timestamp(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        if abs(number) > 10_000_000_000:
            number /= 1000.0

        return int(number)

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text.replace(",", ""))

        if math.isfinite(number):
            if abs(number) > 10_000_000_000:
                number /= 1000.0

            return int(number)

    except ValueError:
        pass

    normalized = (
        text
        .replace(" UTC", "+00:00")
        .replace("Z", "+00:00")
    )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return int(parsed.timestamp())

    except ValueError:
        return None


def _manual_setup_value(
    setup: Dict[str, Any],
    key: str,
    geometry_key: str,
) -> Optional[float]:

    value = _safe_float(
        setup.get(key)
    )

    if value is not None:
        return value

    geometry = (
        setup.get("trade_geometry")
        or {}
    )

    point = (
        geometry.get(geometry_key)
        or {}
    )

    return _safe_float(
        point.get("price")
    )


def _manual_setup_time(
    setup: Dict[str, Any],
) -> Optional[int]:

    entry_candle = _manual_parse_timestamp(
        setup.get("entry_candle")
    )

    if entry_candle is not None:
        return entry_candle

    geometry = (
        setup.get("trade_geometry")
        or {}
    )

    entry_point = (
        geometry.get("entry")
        or {}
    )

    geometry_time = _manual_parse_timestamp(
        entry_point.get("time")
    )

    if geometry_time is not None:
        return geometry_time

    breakout_time = _manual_parse_timestamp(
        setup.get("breakout_time")
    )

    return breakout_time


def _analyze_one_manual_setup(
    setup: Dict[str, Any],
    ordinal: int,
    current_balance: float,
) -> Dict[str, Any]:

    direction_raw = str(
        setup.get("direction", "")
    ).strip().upper()

    if direction_raw == "LONG":
        direction = "BUY"
    elif direction_raw == "SHORT":
        direction = "SELL"
    elif direction_raw in {"BUY", "SELL"}:
        direction = direction_raw
    else:
        return {
            "valid": False,
            "error": "Invalid direction.",
        }

    entry = _manual_setup_value(
        setup,
        "entry",
        "entry",
    )

    sl = _manual_setup_value(
        setup,
        "sl",
        "sl",
    )

    tp = _manual_setup_value(
        setup,
        "tp",
        "tp",
    )

    entry_time = _manual_setup_time(
        setup
    )

    if (
        entry is None or
        sl is None or
        tp is None or
        entry_time is None
    ):
        return {
            "valid": False,
            "error": "Missing entry, SL, TP or entry time.",
        }

    if direction == "BUY":
        if not (sl < entry < tp):
            return {
                "valid": False,
                "error": "Invalid BUY geometry.",
            }
    else:
        if not (tp < entry < sl):
            return {
                "valid": False,
                "error": "Invalid SELL geometry.",
            }

    entry_timeframe = str(
        setup.get(
            "entry_timeframe",
            TIMEFRAME,
        )
    ).strip().upper()

    if entry_timeframe not in TIMEFRAME_SECONDS:
        entry_timeframe = TIMEFRAME

    chart_path = _chart_history_path(
        SYMBOL,
        entry_timeframe,
    )

    candles = _load_local_chart_history(
        chart_path
    )

    if not candles:
        return {
            "valid": False,
            "error": (
                f"No chart data for "
                f"{SYMBOL} {entry_timeframe}."
            ),
        }

    candle_duration = TIMEFRAME_SECONDS[
        entry_timeframe
    ]

    start_index = None

    for index, candle in enumerate(candles):
        candle_end = (
            int(candle["time"]) +
            candle_duration
        )

        if candle_end > entry_time:
            start_index = index
            break

    if start_index is None:
        return {
            "valid": False,
            "error": "Entry time is outside chart history.",
        }

    risk_distance = abs(
        entry - sl
    )

    reward_distance = abs(
        tp - entry
    )

    if risk_distance <= 0:
        return {
            "valid": False,
            "error": "Invalid SL distance.",
        }

    reward_r_multiple = (
        reward_distance /
        risk_distance
    )

    risk_amount = (
        current_balance *
        (MANUAL_RISK_PERCENT / 100.0)
    )

    result = None
    exit_price = None
    exit_time = None
    exit_reason = None

    for candle in candles[start_index:]:

        candle_high = float(
            candle["high"]
        )

        candle_low = float(
            candle["low"]
        )

        if direction == "BUY":

            sl_hit = (
                candle_low <= sl
            )

            tp_hit = (
                candle_high >= tp
            )

            if sl_hit and tp_hit:
                result = "LOSS"
                exit_price = sl
                exit_reason = (
                    "SL_AND_TP_SAME_CANDLE"
                )
                exit_time = int(
                    candle["time"]
                )
                break

            if sl_hit:
                result = "LOSS"
                exit_price = sl
                exit_reason = "SL"
                exit_time = int(
                    candle["time"]
                )
                break

            if tp_hit:
                result = "WIN"
                exit_price = tp
                exit_reason = "TP"
                exit_time = int(
                    candle["time"]
                )
                break

        else:

            sl_hit = (
                candle_high >= sl
            )

            tp_hit = (
                candle_low <= tp
            )

            if sl_hit and tp_hit:
                result = "LOSS"
                exit_price = sl
                exit_reason = (
                    "SL_AND_TP_SAME_CANDLE"
                )
                exit_time = int(
                    candle["time"]
                )
                break

            if sl_hit:
                result = "LOSS"
                exit_price = sl
                exit_reason = "SL"
                exit_time = int(
                    candle["time"]
                )
                break

            if tp_hit:
                result = "WIN"
                exit_price = tp
                exit_reason = "TP"
                exit_time = int(
                    candle["time"]
                )
                break

    if result is None:

        last_candle = candles[-1]

        exit_price = float(
            last_candle["close"]
        )

        exit_time = int(
            last_candle["time"]
        )

        if direction == "BUY":
            favorable_move = (
                exit_price - entry
            )
        else:
            favorable_move = (
                entry - exit_price
            )

        realized_r = (
            favorable_move /
            risk_distance
        )

        result = (
            "WIN"
            if realized_r >= 0
            else "LOSS"
        )

        exit_reason = "END_OF_DATA"

    if result == "WIN":

        pnl = (
            risk_amount *
            reward_r_multiple
        )

    elif exit_reason == "END_OF_DATA":

        if direction == "BUY":
            realized_r = (
                exit_price - entry
            ) / risk_distance
        else:
            realized_r = (
                entry - exit_price
            ) / risk_distance

        pnl = (
            risk_amount *
            realized_r
        )

    else:

        pnl = -risk_amount

    return {
        "valid": True,

        "trade_id": (
            int(setup.get("id"))
            if str(setup.get("id", "")).isdigit()
            else ordinal
        ),

        "source": "MANUAL_ANALYSIS",

        "direction": direction,

        "entry": entry,
        "sl": sl,
        "tp": tp,

        "opened_at": entry_time,
        "exit": exit_price,
        "closed_at": exit_time,

        "result": result,
        "exit_reason": exit_reason,

        "risk_amount": risk_amount,
        "reward_r_multiple": reward_r_multiple,
        "pnl": pnl,

        "setup_id": setup.get("id"),
        "setup_snapshot": copy.deepcopy(
            setup
        ),
    }


async def manual_analysis_loop(user_id: Optional[str] = None) -> None:

    manual_state = _get_manual_analysis_state(user_id)

    global running
    
    try:

        setups = await asyncio.to_thread(
            get_trading_setups,
            SYMBOL,
            user_id
        )

        if not setups:
            raise ValueError(
                "No manually saved Learning trades."
            )

        results = []

        balance_value = STARTING_BALANCE

        invalid_trades = 0

        total = max(
            1,
            len(setups)
        )

        running = True

        manual_state["analysis_status"] = "manual_analysis"

        manual_state["manual_analysis_progress"] = 0

        manual_state["analysis_phase"] = "manual_analysis"

        manual_state["analysis_started"] = True
        manual_state["analysis_completed"] = False

        manual_state["results_source"] = "manual"

        manual_state["manual_analysis_results"] = {}

        for index, setup in enumerate(
            setups,
            start=1
        ):

            analysis = await asyncio.to_thread(
                _analyze_one_manual_setup,
                setup,
                index,
                balance_value,
            )

            if not analysis["valid"]:
                invalid_trades += 1

            else:

                balance_value += float(
                    analysis["pnl"]
                )

                results.append(
                    analysis
                )

            manual_state["manual_analysis_progress"] = int(
                (index / total) * 100
            )

            await broadcast_snapshot(user_id)

            await asyncio.sleep(0.05)

        wins = sum(
            1
            for trade in results
            if trade.get("result") == "WIN"
        )

        losses = sum(
            1
            for trade in results
            if trade.get("result") == "LOSS"
        )

        total_resolved = (
            wins + losses
        )

        winning_percentage = (
            (wins / total_resolved) * 100.0
            if total_resolved
            else 0.0
        )

        losing_percentage = (
            (losses / total_resolved) * 100.0
            if total_resolved
            else 0.0
        )

        total_return = (
            (
                (
                    balance_value -
                    STARTING_BALANCE
                )
                /
                STARTING_BALANCE
            )
            * 100.0
            if STARTING_BALANCE
            else 0.0
        )

        manual_state["manual_analysis_results"] = {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,

            "starting_balance":
                STARTING_BALANCE,

            "final_balance":
                balance_value,

            "total_return":
                total_return,

            "trades":
                results,

            "invalid_trades":
                invalid_trades,

            "winning_trades":
                wins,

            "losing_trades":
                losses,

            "winning_percentage":
                winning_percentage,

            "losing_percentage":
                losing_percentage,
        }

        manual_state["startup_progress"] = 100

        manual_state["startup_message"] = (
            "MANUAL ANALYSIS complete."
        )

        manual_state["manual_analysis_progress"] = 100

        running = False

        manual_state["analysis_status"] = "completed"

        manual_state["analysis_phase"] = (
            "manual_analysis_completed"
        )

        manual_state["analysis_completed"] = True

        await broadcast_snapshot(user_id)

    except asyncio.CancelledError:

        manual_state["manual_analysis_progress"] = 0

        running = False

        manual_state["analysis_status"] = "paused"

        manual_state["analysis_phase"] = (
            "manual_analysis_paused"
        )

        manual_state["analysis_completed"] = False

        raise

    except Exception as error:

        manual_state["manual_analysis_progress"] = 0
        manual_state["analysis_status"] = "error"
        manual_state["analysis_phase"] = "manual_analysis_error"
        manual_state["analysis_completed"] = False
        manual_state["startup_progress"] = 100
        manual_state["startup_message"] = f"MANUAL ANALYSIS error: {error}"
        manual_state["results_source"] = "manual"
        manual_state["manual_analysis_results"] = {}

        print(
            "MANUAL ANALYSIS ERROR:",
            error
        )

        await broadcast_snapshot(user_id)

def delete_trading_setup(
    symbol: str,
    setup_id: int,
    user_id: Optional[str] = None
) -> bool:

    setups = _load_trading_setups(symbol, user_id)

    new_setups = []

    deleted = False

    for setup in setups:

        try:
            current_id = int(
                setup.get("id")
            )

        except (
            TypeError,
            ValueError
        ):
            current_id = None

        if current_id == setup_id:
            deleted = True
            continue

        new_setups.append(setup)

    if deleted:

        _save_trading_setups(
            symbol,
            new_setups,
            user_id
        )

    return deleted

def _candle_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "time": int(row["time"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(row["tick_volume"]),
    }


def _load_local_chart_history(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []

    data: List[Dict[str, Any]] = []

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                data.append({
                    "time": int(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(float(row["volume"])),
                })
            except (KeyError, TypeError, ValueError):
                # Ignore malformed rows instead of destroying a valid archive.
                continue

    data.sort(key=lambda candle: candle["time"])
    return data


def _save_local_chart_history(
    path: str,
    data: List[Dict[str, Any]],
) -> None:
    temp_path = f"{path}.tmp"

    with open(temp_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        writer.writeheader()
        writer.writerows(data)

    os.replace(temp_path, path)


def _filter_closed_candles(
    rates: Any,
    timeframe: str,
) -> List[Dict[str, Any]]:
    if rates is None:
        return []

    tf_seconds = TIMEFRAME_SECONDS[timeframe]
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_candle_start = (now_ts // tf_seconds) * tf_seconds

    result = []
    for row in rates:
        candle = _candle_to_dict(row)
        if candle["time"] < current_candle_start:
            result.append(candle)

    return result


def _download_mt5_history(
    symbol: str,
    timeframe_name: str,
    utc_from: datetime,
    utc_to: datetime,
) -> List[Dict[str, Any]]:
    """Download a requested range in smaller date chunks to avoid huge IPC calls."""
    timeframe = MT5_TIMEFRAMES[timeframe_name]
    result_by_time: Dict[int, Dict[str, Any]] = {}

    cursor = utc_from
    chunk_delta = timedelta(days=DOWNLOAD_CHUNK_DAYS)

    while cursor <= utc_to:
        chunk_end = min(cursor + chunk_delta, utc_to)

        rates = mt5.copy_rates_range(
            symbol,
            timeframe,
            cursor,
            chunk_end,
        )

        if rates is None:
            raise RuntimeError(
                f"MT5 copy_rates_range failed for "
                f"{symbol} {timeframe_name}: {mt5.last_error()}"
            )

        for candle in _filter_closed_candles(rates, timeframe_name):
            result_by_time[candle["time"]] = candle

        cursor = chunk_end + timedelta(seconds=1)

    return [
        result_by_time[timestamp]
        for timestamp in sorted(result_by_time)
    ]


def load_history() -> None:
    if WEB_MODE:
        global h1_data
        global startup_stage, startup_progress, startup_message

        path = _chart_history_path(SYMBOL, TIMEFRAME)

        startup_stage = "Loading market data"
        startup_progress = 5
        startup_message = (
            f"Loading local {SYMBOL} {TIMEFRAME} history..."
        )

        local_data = _load_local_chart_history(path)

        if not local_data:
            raise RuntimeError(
                f"No local chart history available for "
                f"{SYMBOL} {TIMEFRAME}."
            )

        h1_data = local_data

        startup_stage = "Chart ready"
        startup_progress = 100
        startup_message = (
            f"{SYMBOL} {TIMEFRAME} ready. "
            f"{len(h1_data):,} candles loaded from local archive."
        )

        print(
            f"WEB_MODE: loaded {len(h1_data):,} "
            f"{SYMBOL} {TIMEFRAME} candles from CHARTS."
        )

        return
    
    if WEB_MODE and TIMEFRAME in {"M1", "M5"}:
        raise RuntimeError(
            "M1 and M5 are not available in web mode."
        )

    path = _chart_history_path(SYMBOL, TIMEFRAME)

    startup_stage = "Loading market data"
    startup_progress = 5
    startup_message = (
        f"Loading local {SYMBOL} {TIMEFRAME} history..."
    )

    local_data = _load_local_chart_history(path)
    local_count = len(local_data)

    print()
    print("========================================")
    print("LOCAL CHART HISTORY")
    print("========================================")
    print("SYMBOL:    ", SYMBOL)
    print("TIMEFRAME: ", TIMEFRAME)
    print("FILE:      ", path)
    print("LOCAL:     ", f"{local_count:,} candles")
    print("========================================")
    print()

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=365 * YEARS_BACK)

    try:
        if local_data:
            last_local_ts = local_data[-1]["time"]
            tf_seconds = TIMEFRAME_SECONDS[TIMEFRAME]

            sync_from_ts = last_local_ts + tf_seconds
            sync_from = datetime.fromtimestamp(
                sync_from_ts,
                tz=timezone.utc,
            )

            if sync_from <= utc_to:
                startup_stage = "Updating local history"
                startup_progress = 35
                startup_message = (
                    f"Updating {SYMBOL} {TIMEFRAME} with new MT5 candles..."
                )

                new_data = _download_mt5_history(
                    SYMBOL,
                    TIMEFRAME,
                    sync_from,
                    utc_to,
                )

                if new_data:
                    existing_times = {
                        candle["time"]
                        for candle in local_data
                    }

                    for candle in new_data:
                        if candle["time"] not in existing_times:
                            local_data.append(candle)

                    local_data.sort(key=lambda candle: candle["time"])
                    _save_local_chart_history(path, local_data)

                    print(
                        f"Added {len(new_data):,} new "
                        f"{SYMBOL} {TIMEFRAME} candles."
                    )
                else:
                    print("No new MT5 candles to add.")
        else:
            startup_stage = "Creating local chart history"
            startup_progress = 10
            startup_message = (
                f"Downloading {YEARS_BACK} years of "
                f"{SYMBOL} {TIMEFRAME} history..."
            )

            local_data = _download_mt5_history(
                SYMBOL,
                TIMEFRAME,
                utc_from,
                utc_to,
            )

            if not local_data:
                raise RuntimeError(
                    f"No MT5 data returned for "
                    f"{SYMBOL} {TIMEFRAME}."
                )

            _save_local_chart_history(path, local_data)

            print(
                f"Created local archive with "
                f"{len(local_data):,} candles."
            )

    except Exception as error:
        if local_data:
            # A local archive is already usable even when MT5 cannot provide
            # the newest candles right now.
            print(
                f"MT5 sync warning for {SYMBOL} {TIMEFRAME}: {error}"
            )
            print("Using existing local chart history.")
        else:
            raise

    if not local_data:
        raise RuntimeError(
            f"No chart history available for {SYMBOL} {TIMEFRAME}."
        )

    startup_progress = 70
    startup_message = (
        f"Preparing {len(local_data):,} "
        f"{TIMEFRAME} candles from local archive..."
    )

    h1_data = local_data

    startup_stage = "Chart ready"
    startup_progress = 100
    startup_message = (
        f"{SYMBOL} {TIMEFRAME} ready. "
        f"{len(h1_data):,} candles loaded from local archive."
    )

    print(
        f"Loaded {len(h1_data):,} "
        f"{SYMBOL} {TIMEFRAME} candles from local archive."
    )

def update_all_chart_data() -> Dict[str, Any]:
    global startup_stage, startup_progress, startup_message

    if not mt5_connected:
        return {
            "success": False,
            "updated": 0,
            "message": "MT5 is switched off. Data are not updated."
        }

    updated_candles = 0
    updated_charts = 0
    checked_charts = 0

    startup_stage = "Updating chart data"
    startup_progress = 5
    startup_message = "Updating all local chart data..."

    if not os.path.isdir(CHARTS_ROOT):
        return {
            "success": True,
            "updated": 0,
            "message": "No local charts found."
        }

    for symbol_folder in sorted(os.listdir(CHARTS_ROOT)):
        symbol_path = os.path.join(
            CHARTS_ROOT,
            symbol_folder
        )

        if not os.path.isdir(symbol_path):
            continue

        try:
            resolved_symbol = resolve_symbol(symbol_folder)

            if not mt5.symbol_select(
                resolved_symbol,
                True
            ):
                print(
                    f"Skipping {symbol_folder}: "
                    "MT5 symbol could not be selected."
                )
                continue

        except Exception as error:
            print(
                f"Skipping {symbol_folder}: {error}"
            )
            continue

        for timeframe_name in MT5_TIMEFRAMES:

            if WEB_MODE and timeframe_name in {"M1", "M5"}:
                continue

            timeframe_path = os.path.join(
                symbol_path,
                timeframe_name
            )

            history_path = os.path.join(
                timeframe_path,
                LOCAL_CHART_FILENAME
            )

            if not os.path.isfile(history_path):
                continue

            checked_charts += 1

            local_data = _load_local_chart_history(
                history_path
            )

            if not local_data:
                continue

            last_local_ts = local_data[-1]["time"]
            tf_seconds = TIMEFRAME_SECONDS[timeframe_name]

            sync_from_ts = last_local_ts + tf_seconds
            utc_to = datetime.now(timezone.utc)

            if sync_from_ts > int(utc_to.timestamp()):
                continue

            sync_from = datetime.fromtimestamp(
                sync_from_ts,
                tz=timezone.utc
            )

            try:
                new_data = _download_mt5_history(
                    resolved_symbol,
                    timeframe_name,
                    sync_from,
                    utc_to,
                )

                if not new_data:
                    continue

                existing_times = {
                    candle["time"]
                    for candle in local_data
                }

                added_here = 0

                for candle in new_data:
                    if candle["time"] not in existing_times:
                        local_data.append(candle)
                        added_here += 1

                if added_here > 0:
                    local_data.sort(
                        key=lambda candle: candle["time"]
                    )

                    _save_local_chart_history(
                        history_path,
                        local_data
                    )

                    updated_candles += added_here
                    updated_charts += 1

                    print(
                        f"[UPDATE] "
                        f"{symbol_folder} {timeframe_name}: "
                        f"+{added_here} candles"
                    )

            except Exception as error:
                print(
                    f"[UPDATE] "
                    f"{symbol_folder} {timeframe_name} failed: "
                    f"{error}"
                )

    startup_stage = "Chart ready"
    startup_progress = 100
    startup_message = (
        f"Data update complete. "
        f"{updated_candles:,} new candles added."
    )

    return {
        "success": True,
        "updated": updated_candles,
        "charts_updated": updated_charts,
        "charts_checked": checked_charts,
        "message": startup_message,
    }

def switch_symbol(new_symbol: str) -> None:
    global SYMBOL
    global h1_data

    new_symbol = str(new_symbol).strip()

    if not new_symbol:
        return

    # ---------------------------------------------------------
    # LOCAL CHARTS FIRST
    # ---------------------------------------------------------
    # If the symbol already exists locally, use it directly.
    # MT5 is NOT required for switching to an existing chart.
    # ---------------------------------------------------------

    local_symbol_folder = os.path.join(
        CHARTS_ROOT,
        _chart_symbol_folder(new_symbol),
    )

    if os.path.isdir(local_symbol_folder):
        SYMBOL = new_symbol

        load_history()
        _restore_active_session()
        return

    # ---------------------------------------------------------
    # NO LOCAL DATA
    # ---------------------------------------------------------
    # Only now do we need MT5 to resolve/download the symbol.
    # ---------------------------------------------------------

    if not mt5_connected:
        raise RuntimeError(
            f"No local chart history found for {new_symbol} "
            "and MT5 is switched off."
        )

    resolved_symbol = resolve_symbol(new_symbol)

    if not mt5.symbol_select(
        resolved_symbol,
        True
    ):
        raise RuntimeError(
            f"Cannot select symbol: {resolved_symbol}"
        )

    SYMBOL = resolved_symbol

    load_history()
    _restore_active_session()

def switch_timeframe(new_timeframe: str) -> None:
    global TIMEFRAME
    global h1_data

    new_timeframe = str(new_timeframe).strip().upper()

    if WEB_MODE and new_timeframe in {"M1", "M5"}:
        raise RuntimeError(
            "M1 and M5 are not available in web mode."
        )

    if new_timeframe not in MT5_TIMEFRAMES:
        raise RuntimeError(
            f"Unsupported timeframe: {new_timeframe}"
        )

    TIMEFRAME = new_timeframe

    # Same market session, different timeframe state.
    load_history()
    _restore_active_session()

# Strategy
# ---------------------------------------------------------------------------
# The chart keeps using the existing level-generation logic above. The
# strategy itself uses an additional filter so that only the strongest/main
# levels are considered for entries.
# ---------------------------------------------------------------------------

# Strategy configuration. These values are intentionally centralized so the
# first backtest can be tuned without changing the trading logic.
STRATEGY_LEVEL_TOP_FRACTION = 0.20
STRATEGY_MAX_LEVELS = 6
STRATEGY_MIN_LEVEL_TOUCHES = 2

SWING_WINDOW = 4
MIN_SWING_SEPARATION = 4
SETUP_MAX_AGE_CANDLES = 80

# Trade audit / explanation
TRADE_DETAIL_ACTION = "trade_detail"

REWARD_R_MULTIPLE = 3.0
PROTECTION_TRIGGER_FRACTION = 0.33
PROTECTION_SL_FRACTION = 0.05
STRUCTURE_SL_BUFFER_PERCENT = 0.0


def price_distance(a: float, b: float) -> float:
    return abs(a - b) / a if a else 0.0


def candle_body_ratio(candle: Dict[str, Any]) -> float:
    candle_range = candle["high"] - candle["low"]

    if candle_range <= 0:
        return 0.0

    return abs(candle["close"] - candle["open"]) / candle_range


def find_major_levels(
    data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    if len(data) < 30:
        return []

    prices = [c["close"] for c in data]

    # ---------------------------------------------------------
    # 1. Determine the complete visible price range.
    # ---------------------------------------------------------

    lowest = min(prices)
    highest = max(prices)

    price_range = highest - lowest

    if price_range <= 0:
        return []

    # ---------------------------------------------------------
    # 2. High and Low are always important levels.
    # ---------------------------------------------------------

    candidates = [
        {
            "price": lowest,
            "touches": 0,
            "strength": 100.0,
        },
        {
            "price": highest,
            "touches": 0,
            "strength": 100.0,
        }
    ]

    # ---------------------------------------------------------
    # 3. Build potential levels from meaningful turning points.
    #
    # We deliberately use a larger swing window.
    # Small micro-swings are NOT major levels.
    # ---------------------------------------------------------

    swing_window = max(
        5,
        min(15, len(prices) // 30)
    )

    for i in range(
        swing_window,
        len(prices) - swing_window
    ):

        price = prices[i]

        left = prices[
            i - swing_window:i
        ]

        right = prices[
            i + 1:i + swing_window + 1
        ]

        if (
            price >= max(left)
            and price >= max(right)
        ):
            candidates.append({
                "price": price,
                "touches": 0,
                "strength": 0.0,
            })

        elif (
            price <= min(left)
            and price <= min(right)
        ):
            candidates.append({
                "price": price,
                "touches": 0,
                "strength": 0.0,
            })

    # ---------------------------------------------------------
    # 4. Calculate how strongly price respected each candidate.
    #
    # The tolerance is based on the COMPLETE visible range.
    # ---------------------------------------------------------

    touch_tolerance = price_range * 0.015

    for candidate in candidates:

        level = candidate["price"]

        touches = 0
        reactions = 0

        last_touch_index = -999

        for i, price in enumerate(prices):

            if abs(price - level) <= touch_tolerance:

                # Do not count every candle inside the same visit.
                if i - last_touch_index >= 8:

                    touches += 1
                    last_touch_index = i

                    # Check whether price moved away strongly
                    # after touching the level.
                    future_end = min(
                        len(prices),
                        i + 12
                    )

                    future = prices[
                        i + 1:future_end
                    ]

                    if future:

                        movement = max(
                            abs(p - level)
                            for p in future
                        )

                        if movement >= price_range * 0.04:
                            reactions += 1

        candidate["touches"] = touches

        candidate["strength"] = (
            touches * 2.0
            + reactions * 5.0
        )

    # ---------------------------------------------------------
    # 5. Remove weak candidates.
    # ---------------------------------------------------------

    candidates = [
        c for c in candidates
        if (
            c["price"] == lowest
            or c["price"] == highest
            or c["touches"] >= 2
        )
    ]

    # ---------------------------------------------------------
    # 6. Sort strongest levels first.
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda c: c["strength"],
        reverse=True
    )

    # ---------------------------------------------------------
    # 7. IMPORTANT:
    # Levels must be spread across the COMPLETE range.
    #
    # We use a minimum spacing based on the visible range.
    # This prevents 5 levels from sitting on top of each other.
    # ---------------------------------------------------------

    minimum_spacing = price_range * 0.08

    selected = []

    # High and Low are protected.
    selected.append({
        "price": lowest,
        "touches": candidates[0]["touches"],
        "strength": 100.0,
    })

    selected.append({
        "price": highest,
        "touches": candidates[1]["touches"],
        "strength": 100.0,
    })

    # ---------------------------------------------------------
    # 8. Select strong levels while enforcing spacing.
    # ---------------------------------------------------------

    for candidate in candidates:

        price = candidate["price"]

        # High / Low already exist.
        if (
            abs(price - lowest) < minimum_spacing
            or
            abs(price - highest) < minimum_spacing
        ):
            continue

        too_close = False

        for existing in selected:

            if abs(
                price - existing["price"]
            ) < minimum_spacing:

                too_close = True
                break

        if too_close:
            continue

        selected.append(candidate)

    # ---------------------------------------------------------
    # 9. Convert candidates into the format used by the strategy.
    # ---------------------------------------------------------

    levels = []

    for candidate in selected:

        price = candidate["price"]

        levels.append({
            "low": price,
            "high": price,
            "center": price,
            "touches": candidate["touches"],
            "breakouts": 0,
            "reaction_strength": candidate["strength"],
            "strength": candidate["strength"],
        })

    # ---------------------------------------------------------
    # 10. Sort from LOW to HIGH.
    # ---------------------------------------------------------

    levels.sort(
        key=lambda level: level["center"]
    )

    return levels


def find_levels_for_view(from_time: int, to_time: int):

    if not h1_data:
        return []

    # Use EXACTLY the candles currently visible
    # in the frontend viewport.
    visible = [
        candle
        for candle in h1_data
        if from_time <= candle["time"] <= to_time
    ]

    if len(visible) < 30:
        return []

    return find_major_levels(visible)


def nearest_above(
    price: float,
    zones: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    values = [
        zone
        for zone in zones
        if zone["center"] > price
    ]

    if not values:
        return None

    return min(
        values,
        key=lambda zone: zone["center"]
    )


def nearest_below(
    price: float,
    zones: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    values = [
        zone
        for zone in zones
        if zone["center"] < price
    ]

    if not values:
        return None

    return max(
        values,
        key=lambda zone: zone["center"]
    )


def detect_zigzag_range(line_data: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if len(line_data) < 5:
        return None

    recent = line_data[-50:]

    highs = []
    lows = []

    for i in range(1, len(recent) - 1):
        prev_value = recent[i - 1]["close"]
        value = recent[i]["close"]
        next_value = recent[i + 1]["close"]

        if value > prev_value and value > next_value:
            highs.append(value)

        if value < prev_value and value < next_value:
            lows.append(value)

    if not highs or not lows:
        return None

    return {
        "high": max(highs),
        "low": min(lows),
    }



# ---------------------------------------------------------------------------
# NUMERIC MICRO-MODEL STRATEGY
# ---------------------------------------------------------------------------
# The chart level engine above stays unchanged because the frontend uses it
# for the visual support/resistance map. The old hard-coded trading strategy
# is removed from decision making. TRAINING creates a market-specific model.
# TESTING loads that model and makes independent decisions from candle data.

FEATURE_NAMES = [
    "return_3",
    "return_8",
    "return_20",
    "trend_slope",
    "range_mean_8",
    "range_mean_20",
    "body_mean_8",
    "bull_ratio_8",
    "close_position_20",
    "distance_to_high_20",
    "distance_to_low_20",
    "nearest_level_distance",
    "nearest_level_strength",
    "nearest_level_touches",
    "volatility_ratio",
]


def _model_symbol_dir(symbol: str) -> str:
    return os.path.join(MODEL_ROOT, _chart_symbol_folder(symbol))


def _market_model_path(symbol: str) -> str:
    folder = _model_symbol_dir(symbol)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, MODEL_FILENAME)


def _market_training_info_path(symbol: str) -> str:
    folder = _model_symbol_dir(symbol)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, TRAINING_INFO_FILENAME)


def _default_model_path() -> str:
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    return os.path.join(DEFAULT_MODEL_DIR, MODEL_FILENAME)


def _default_model_template() -> Dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "numeric_logistic_micro_model",
        "symbol": "DEFAULT",
        "timeframe": None,
        "trained": False,
        "feature_names": FEATURE_NAMES.copy(),
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_std": [1.0] * len(FEATURE_NAMES),
        "long_weights": [0.0] * len(FEATURE_NAMES),
        "long_bias": 0.0,
        "short_weights": [0.0] * len(FEATURE_NAMES),
        "short_bias": 0.0,
        "signal_threshold": MODEL_MIN_SIGNAL_PROBABILITY,
        "probability_edge": MODEL_MIN_PROBABILITY_EDGE,
        "risk": {
            "sl_fraction": SL_PERCENT,
            "tp_fraction": TP_PERCENT,
            "reward_multiple": TP_PERCENT / SL_PERCENT if SL_PERCENT else 2.0,
        },
        "training_stats": {},
    }


def _ensure_default_model() -> Dict[str, Any]:
    path = _default_model_path()
    if not os.path.isfile(path):
        template = _default_model_template()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(template, file, indent=4, ensure_ascii=False)
        return template

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Default model is not a JSON object.")
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        template = _default_model_template()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(template, file, indent=4, ensure_ascii=False)
        return template

def _save_market_model(symbol: str, model: Dict[str, Any]) -> None:
    path = _market_model_path(symbol)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def _save_training_info(symbol: str, info: Dict[str, Any]) -> None:
    path = _market_training_info_path(symbol)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _nearest_level_features(
    price: float,
    levels: List[Dict[str, Any]],
) -> tuple[float, float, float]:
    if not levels or price == 0:
        return 1.0, 0.0, 0.0

    nearest = min(
        levels,
        key=lambda level: abs(float(level.get("center", 0.0)) - price),
    )

    center = float(nearest.get("center", price))
    distance = abs(center - price) / abs(price) if price else 1.0
    strength = float(nearest.get("strength", 0.0)) / 100.0
    touches = min(1.0, float(nearest.get("touches", 0)) / 10.0)
    return min(distance, 10.0), strength, touches


def _calculate_features(
    data: List[Dict[str, Any]],
) -> Optional[List[float]]:
    if len(data) < MODEL_LOOKBACK + 2:
        return None

    closes = [float(c["close"]) for c in data]
    ranges = [
        max(0.0, float(c["high"]) - float(c["low"]))
        for c in data
    ]

    current = data[-1]
    price = float(current["close"])

    def pct_return(period: int) -> float:
        if len(closes) <= period or closes[-period - 1] == 0:
            return 0.0
        return (closes[-1] - closes[-period - 1]) / closes[-period - 1]

    trend_window = min(20, len(closes))
    trend_slice = closes[-trend_window:]
    trend_start = trend_slice[0]
    trend_end = trend_slice[-1]
    trend_slope = (
        (trend_end - trend_start) / trend_start / max(1, trend_window - 1)
        if trend_start
        else 0.0
    )

    range_mean_8 = (
        sum(ranges[-8:]) / max(1, len(ranges[-8:])) / price
        if price else 0.0
    )
    range_mean_20 = (
        sum(ranges[-20:]) / max(1, len(ranges[-20:])) / price
        if price else 0.0
    )

    body_values = []
    bull_values = []
    for candle in data[-8:]:
        candle_range = float(candle["high"]) - float(candle["low"])
        if candle_range > 0:
            body_values.append(
                abs(float(candle["close"]) - float(candle["open"])) / candle_range
            )
        bull_values.append(
            1.0 if float(candle["close"]) >= float(candle["open"]) else 0.0
        )

    body_mean_8 = sum(body_values) / max(1, len(body_values))
    bull_ratio_8 = sum(bull_values) / max(1, len(bull_values))

    lookback_closes = closes[-20:]
    highest = max(lookback_closes)
    lowest = min(lookback_closes)
    span = highest - lowest
    close_position_20 = (price - lowest) / span if span > 0 else 0.5
    distance_to_high_20 = (highest - price) / price if price else 0.0
    distance_to_low_20 = (price - lowest) / price if price else 0.0

    recent_mean = range_mean_8
    long_mean = range_mean_20
    volatility_ratio = recent_mean / long_mean if long_mean > 0 else 1.0

    level_data = find_major_levels(data[-min(240, len(data)):])
    level_distance, level_strength, level_touches = _nearest_level_features(
        price,
        level_data,
    )

    values = [
        pct_return(3),
        pct_return(8),
        pct_return(20),
        trend_slope,
        range_mean_8,
        range_mean_20,
        body_mean_8,
        bull_ratio_8,
        close_position_20,
        distance_to_high_20,
        distance_to_low_20,
        level_distance,
        level_strength,
        level_touches,
        volatility_ratio,
    ]

    return [
        value if math.isfinite(value) else 0.0
        for value in values
    ]


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _standardize_dataset(
    features: List[List[float]],
) -> tuple[List[float], List[float], List[List[float]]]:
    if not features:
        return [], [], []

    count = len(features)
    width = len(features[0])
    means = []
    stds = []

    for column in range(width):
        values = [row[column] for row in features]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-12 else 1.0)

    normalized = [
        [
            (value - means[column]) / stds[column]
            for column, value in enumerate(row)
        ]
        for row in features
    ]
    return means, stds, normalized


def _fit_binary_logistic(
    features: List[List[float]],
    targets: List[float],
) -> tuple[List[float], float]:
    if not features:
        return [0.0] * len(FEATURE_NAMES), 0.0

    width = len(features[0])
    weights = [0.0] * width
    bias = 0.0
    count = len(features)

    for _ in range(MODEL_TRAINING_EPOCHS):
        grad_w = [0.0] * width
        grad_b = 0.0

        for row, target in zip(features, targets):
            score = bias + sum(weight * value for weight, value in zip(weights, row))
            prediction = _sigmoid(score)
            error = prediction - target
            grad_b += error
            for index, value in enumerate(row):
                grad_w[index] += error * value

        step = MODEL_LEARNING_RATE / max(1, count)
        bias -= step * grad_b
        for index in range(width):
            weights[index] -= step * grad_w[index]

    return weights, bias


def _time_to_index(timestamp: Any) -> Optional[int]:
    value = _safe_float(timestamp)
    if value is None or not h1_data:
        return None

    target = int(value)
    low = 0
    high = len(h1_data) - 1

    while low <= high:
        middle = (low + high) // 2
        current = int(h1_data[middle]["time"])
        if current == target:
            return middle
        if current < target:
            low = middle + 1
        else:
            high = middle - 1

    candidates = []
    if 0 <= low < len(h1_data):
        candidates.append(low)
    if 0 <= high < len(h1_data):
        candidates.append(high)
    if not candidates:
        return None
    return min(candidates, key=lambda index: abs(int(h1_data[index]["time"]) - target))


def _setup_anchor_index(setup: Dict[str, Any]) -> Optional[int]:
    anchor = setup.get("anchor")
    if isinstance(anchor, dict):
        index = _time_to_index(anchor.get("time"))
        if index is not None:
            return index

    entry_candle = setup.get("entry_candle")
    if isinstance(entry_candle, str) and entry_candle.strip():
        match = re.search(r"(\d{9,})", entry_candle)
        if match:
            index = _time_to_index(match.group(1))
            if index is not None:
                return index

    entry_price = _safe_float(setup.get("entry"))
    if entry_price is None:
        return None

    # Fallback: use the closest candle price when a timestamp is missing.
    return min(
        range(len(h1_data)),
        key=lambda index: abs(float(h1_data[index]["close"]) - entry_price),
    )


def _setup_risk_profile(setups: List[Dict[str, Any]]) -> Dict[str, float]:
    sl_values = []
    tp_values = []

    for setup in setups:
        entry = _safe_float(setup.get("entry"))
        sl = _safe_float(setup.get("sl"))
        tp = _safe_float(setup.get("tp"))
        if entry is None or entry == 0:
            continue
        if sl is not None:
            sl_values.append(abs(entry - sl) / abs(entry))
        if tp is not None:
            tp_values.append(abs(tp - entry) / abs(entry))

    sl_fraction = (
        sorted(sl_values)[len(sl_values) // 2]
        if sl_values
        else SL_PERCENT
    )
    tp_fraction = (
        sorted(tp_values)[len(tp_values) // 2]
        if tp_values
        else TP_PERCENT
    )

    sl_fraction = max(0.0001, min(sl_fraction, 0.25))
    tp_fraction = max(0.0002, min(tp_fraction, 0.50))

    return {
        "sl_fraction": sl_fraction,
        "tp_fraction": tp_fraction,
        "reward_multiple": tp_fraction / sl_fraction if sl_fraction else 2.0,
    }


def _build_training_dataset(
    setups: List[Dict[str, Any]],
) -> tuple[List[List[float]], List[int], Dict[str, Any]]:
    positive_samples: List[tuple[List[float], int]] = []
    used_indices = set()

    for setup in setups:
        index = _setup_anchor_index(setup)
        if index is None or index < MODEL_LOOKBACK:
            continue

        context = h1_data[:index + 1]
        features = _calculate_features(context)
        if features is None:
            continue

        direction = str(setup.get("direction", "")).upper().strip()
        label = 1 if direction in ("LONG", "BUY") else -1 if direction in ("SHORT", "SELL") else 0
        if label == 0:
            continue

        positive_samples.append((features, label))
        used_indices.add(index)

    background: List[List[float]] = []
    if h1_data:
        step = max(1, len(h1_data) // MODEL_BACKGROUND_SAMPLE_LIMIT)
        for index in range(MODEL_LOOKBACK, len(h1_data), step):
            if any(abs(index - used) <= 5 for used in used_indices):
                continue
            features = _calculate_features(h1_data[:index + 1])
            if features is not None:
                background.append(features)
            if len(background) >= MODEL_BACKGROUND_SAMPLE_LIMIT:
                break

    features = [row for row, _ in positive_samples]
    labels = [label for _, label in positive_samples]

    features.extend(background)
    labels.extend([0] * len(background))

    info = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "input_trade_count": len(setups),
        "usable_positive_samples": len(positive_samples),
        "background_samples": len(background),
        "total_samples": len(features),
    }

    return features, labels, info


def _load_training_model_base() -> Dict[str, Any]:
    # Always start from the clean DEFAULT template. Never mutate DEFAULT.
    return copy.deepcopy(_ensure_default_model())


def train_market_model(
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    global model_available, model_info, last_training_info

    if symbol != SYMBOL or timeframe != TIMEFRAME:
        raise RuntimeError("Training context does not match the active market session.")

    setups = get_trading_setups(symbol)
    if not setups:
        raise RuntimeError(
            f"No Learning trades found for {symbol}."
        )

    features, labels, dataset_info = _build_training_dataset(setups)

    if not features:
        raise RuntimeError(
            f"No usable Learning samples could be extracted for {symbol} {timeframe}."
        )

    means, stds, normalized = _standardize_dataset(features)

    long_targets = [1.0 if label == 1 else 0.0 for label in labels]
    short_targets = [1.0 if label == -1 else 0.0 for label in labels]

    long_weights, long_bias = _fit_binary_logistic(normalized, long_targets)
    short_weights, short_bias = _fit_binary_logistic(normalized, short_targets)

    default_model = _load_training_model_base()
    model = default_model
    model.update({
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "numeric_logistic_micro_model",
        "symbol": symbol,
        "timeframe": timeframe,
        "trained": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES.copy(),
        "feature_mean": means,
        "feature_std": stds,
        "long_weights": long_weights,
        "long_bias": long_bias,
        "short_weights": short_weights,
        "short_bias": short_bias,
        "signal_threshold": MODEL_MIN_SIGNAL_PROBABILITY,
        "probability_edge": MODEL_MIN_PROBABILITY_EDGE,
        "risk": _setup_risk_profile(setups),
        "training_stats": dataset_info,
    })

    _save_market_model(symbol, model)

    training_info = {
        **dataset_info,
        "symbol": symbol,
        "timeframe": timeframe,
        "model_path": _market_model_path(symbol),
        "default_model_path": _default_model_path(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "message": "Market-specific model trained from Learning trades.",
    }
    _save_training_info(symbol, training_info)

    model_available = True
    model_info = {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_path": _market_model_path(symbol),
        "trained_at": model.get("trained_at"),
        "training_stats": dataset_info,
    }
    last_training_info = training_info

    return {
        "success": True,
        "model": model_info,
        "training": training_info,
    }


def load_ai_runtime() -> bool:
    """Load the market-specific PyTorch model produced by ai.py."""
    global ai_model, ai_scaler, ai_checkpoint
    global ai_market_candles
    global model_available, model_info, last_training_info

    try:
        ai_model, ai_scaler, ai_checkpoint = ai.load_model(SYMBOL)
        ai_market_candles = ai.load_all_market_candles(SYMBOL)

        model_available = True
        model_info = {
            "symbol": SYMBOL,
            "model_path": str(ai.model_path(SYMBOL)),
            "trained_at": ai_checkpoint.get("trained_at"),
            "format_version": ai_checkpoint.get("format_version"),
            "training_summary": ai_checkpoint.get("training_summary", {}),
        }
        last_training_info = copy.deepcopy(
            ai_checkpoint.get("training_summary", {})
        )
        return True

    except FileNotFoundError:
        ai_model = None
        ai_scaler = None
        ai_checkpoint = {}
        ai_market_candles = None
        model_available = False
        model_info = {}
        return False

    except Exception as error:
        print(f"AI model load error for {SYMBOL}: {error}")
        ai_model = None
        ai_scaler = None
        ai_checkpoint = {}
        ai_market_candles = None
        model_available = False
        model_info = {}
        return False


def _load_legacy_market_model() -> Optional[Dict[str, Any]]:
    """Legacy JSON model loader retained only for backward compatibility."""
    model = _load_market_model(SYMBOL)
    if model is None:
        return None
    return model


def _recent_swing_setup(
    data: List[Dict[str, Any]],
    direction: str,
    level_price: float,
) -> Dict[str, Any]:
    recent = data[-min(len(data), 80):]
    if direction == "BUY":
        lows = sorted(
            [
                {
                    "index": index,
                    "time": candle["time"],
                    "price": float(candle["low"]),
                }
                for index, candle in enumerate(recent)
            ],
            key=lambda item: item["price"],
        )[:2]
        if len(lows) < 2:
            lows = [
                {
                    "index": max(0, len(recent) - 2),
                    "time": recent[-2]["time"],
                    "price": float(recent[-2]["low"]),
                },
                {
                    "index": len(recent) - 1,
                    "time": recent[-1]["time"],
                    "price": float(recent[-1]["low"]),
                },
            ]
        lows = sorted(lows, key=lambda item: item["index"])
        return {
            "low1": copy.deepcopy(lows[0]),
            "low2": copy.deepcopy(lows[1]),
            "level": level_price,
        }

    highs = sorted(
        [
            {
                "index": index,
                "time": candle["time"],
                "price": float(candle["high"]),
            }
            for index, candle in enumerate(recent)
        ],
        key=lambda item: item["price"],
        reverse=True,
    )[:2]
    if len(highs) < 2:
        highs = [
            {
                "index": max(0, len(recent) - 2),
                "time": recent[-2]["time"],
                "price": float(recent[-2]["high"]),
            },
            {
                "index": len(recent) - 1,
                "time": recent[-1]["time"],
                "price": float(recent[-1]["high"]),
            },
        ]
    highs = sorted(highs, key=lambda item: item["index"])
    return {
        "high1": copy.deepcopy(highs[0]),
        "high2": copy.deepcopy(highs[1]),
        "level": level_price,
    }


def _choose_model_level(
    price: float,
    levels: List[Dict[str, Any]],
    direction: str,
) -> Optional[Dict[str, Any]]:
    if not levels:
        return None

    if direction == "BUY":
        below = [
            level for level in levels
            if float(level.get("center", price)) <= price
        ]
        if below:
            return max(below, key=lambda level: float(level.get("center", price)))

    else:
        above = [
            level for level in levels
            if float(level.get("center", price)) >= price
        ]
        if above:
            return min(above, key=lambda level: float(level.get("center", price)))

    return min(
        levels,
        key=lambda level: abs(float(level.get("center", price)) - price),
    )

def evaluate_strategy(
    visible_h1: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compatibility wrapper that exposes ai.py predictions to the frontend."""
    global major_levels, cached_strategy_levels, last_level_calculation_index
    global last_model_signal, model_available

    if not visible_h1:
        return {
            "action": "WAIT",
            "reason": "No visible market data.",
            "levels": [],
            "model_available": model_available,
        }

    # Keep the existing level engine purely for chart visualization.
    if (
        last_level_calculation_index < 0
        or current_index - last_level_calculation_index >= LEVEL_RECALC_INTERVAL
    ):
        cached_strategy_levels = find_major_levels(
            visible_h1[-min(240, len(visible_h1)):]
        )
        last_level_calculation_index = current_index

    major_levels = cached_strategy_levels

    if ai_model is None or ai_scaler is None or ai_market_candles is None:
        return {
            "action": "WAIT",
            "reason": f"No trained ai.py model exists for {SYMBOL}. Run TRAINING first.",
            "levels": major_levels,
            "model_available": False,
        }

    target_time = int(visible_h1[-1]["time"]) + TIMEFRAME_SECONDS[TIMEFRAME]

    try:
        features = ai.market_state_at(
            ai_market_candles,
            target_time,
        )

        prediction = ai.predict_from_features(
            model=ai_model,
            scaler=ai_scaler,
            features=features,
        )

    except Exception as error:
        model_available = True
        return {
            "action": "WAIT",
            "reason": f"ai.py prediction error: {error}",
            "levels": major_levels,
            "model_available": True,
        }

    current_price = ai.current_price_at(
        ai_market_candles,
        target_time,
    )
    atr = ai.atr_at(
        ai_market_candles,
        target_time,
    )

    last_model_signal = {
        "time": target_time,
        "price": float(visible_h1[-1]["close"]),
        "prediction": copy.deepcopy(prediction),
        "current_price": current_price,
        "atr": atr,
    }

    direction_name = prediction.get("direction", "WAIT")

    if direction_name == "WAIT":
        action = "WAIT"
    elif direction_name == "LONG":
        action = "BUY"
    else:
        action = "SELL"

    reason = (
        f"ai.py {direction_name} | "
        f"setup probability: {float(prediction.get('setup_probability', 0.0)):.2%}"
    )

    result = {
        "action": action,
        "reason": reason,
        "levels": major_levels,
        "model_available": True,
        "probabilities": prediction.get("direction_probabilities", {}),
        "setup_probability": prediction.get("setup_probability", 0.0),
        "setup_timeframe": prediction.get("setup_timeframe"),
        "entry_timeframe": prediction.get("entry_timeframe"),
        "prediction": prediction,
    }

    return result


# Virtual execution



def open_trade_position(
    direction: str,
    price: float,
    setup: Optional[Dict[str, Any]] = None,
) -> None:
    global open_trade

    if open_trade is not None or not trading_enabled or setup is None:
        return

    entry_price = float(price)
    sl = _safe_float(setup.get("sl"))
    tp = _safe_float(setup.get("tp"))
    if sl is None or tp is None:
        return

    if direction == "BUY" and not (sl < entry_price < tp):
        return
    if direction == "SELL" and not (tp < entry_price < sl):
        return

    risk_distance = abs(entry_price - sl)
    tp_distance = abs(tp - entry_price)
    if risk_distance <= 0 or tp_distance <= 0:
        return

    trade_id = len(trade_history) + 1
    opened_at = int(h1_data[current_index]["time"])

    if direction == "BUY":
        structural_reference_price = float(
            min(
                float(setup.get("low1", {}).get("price", sl)),
                float(setup.get("low2", {}).get("price", sl)),
            )
        )
        structural_reference_name = "model-derived lower structure"
        setup_swings = {
            "low1": copy.deepcopy(setup.get("low1")),
            "low2": copy.deepcopy(setup.get("low2")),
        }
    else:
        structural_reference_price = float(
            max(
                float(setup.get("high1", {}).get("price", sl)),
                float(setup.get("high2", {}).get("price", sl)),
            )
        )
        structural_reference_name = "model-derived upper structure"
        setup_swings = {
            "high1": copy.deepcopy(setup.get("high1")),
            "high2": copy.deepcopy(setup.get("high2")),
        }

    trigger_price = _safe_float(setup.get("protection_trigger"))
    protected_sl = _safe_float(setup.get("protection_sl"))

    if trigger_price is None:
        trigger_price = (
            entry_price + tp_distance * PROTECTION_TRIGGER_FRACTION
            if direction == "BUY"
            else entry_price - tp_distance * PROTECTION_TRIGGER_FRACTION
        )

    if protected_sl is None:
        protected_sl = (
            entry_price + tp_distance * PROTECTION_SL_FRACTION
            if direction == "BUY"
            else entry_price - tp_distance * PROTECTION_SL_FRACTION
        )

    open_trade = {
        "trade_id": trade_id,
        "direction": direction,
        "entry": entry_price,
        "opened_at": opened_at,
        "sl": sl,
        "initial_sl": sl,
        "tp": tp,
        "risk_distance": risk_distance,
        "tp_distance": tp_distance,
        "reward_r_multiple": tp_distance / risk_distance if risk_distance else 0.0,
        "protection_trigger": trigger_price,
        "protection_sl": protected_sl,
        "protection_trigger_fraction": PROTECTION_TRIGGER_FRACTION,
        "protection_sl_fraction": PROTECTION_SL_FRACTION,
        "protected": False,
        "protected_at": None,
        "sl_before_protection": None,
        "sl_after_protection": None,
        "pattern": setup.get("pattern", "MODEL SIGNAL"),
        "entry_reason": setup.get("reason", "Market model signal"),
        "setup_level": float(setup.get("level", entry_price)),
        "selected_level": copy.deepcopy(setup.get("selected_level")),
        "level_strength": float(setup.get("level_strength", 0.0)),
        "level_touches": int(setup.get("level_touches", 0)),
        "sl_basis": {
            "type": "MODEL",
            "reference": structural_reference_name,
            "reference_price": structural_reference_price,
            "buffer_percent": 0.0,
        },
        "breakout_time": setup.get("breakout_time"),
        "breakout_index": setup.get("breakout_index"),
        "breakout_candle": copy.deepcopy(setup.get("breakout_candle")),
        "setup_swings": setup_swings,
        "setup_snapshot": copy.deepcopy(setup),
        "model_probabilities": copy.deepcopy(setup.get("model_probabilities", {})),
        "model_symbol": SYMBOL,
        "model_timeframe": TIMEFRAME,
    }

    if direction == "BUY":
        open_trade["low1"] = float(setup_swings["low1"]["price"])
        open_trade["low2"] = float(setup_swings["low2"]["price"])
    else:
        open_trade["high1"] = float(setup_swings["high1"]["price"])
        open_trade["high2"] = float(setup_swings["high2"]["price"])


def stop_trading() -> None:
    global trading_enabled
    trading_enabled = False


# ---------------------------------------------------------------------------
# TRAINING AND TESTING RUNTIME
# ---------------------------------------------------------------------------


def step_testing() -> None:
    """Advance the selected chart timeframe and let ai.py make one decision."""
    global current_index, blocked_trades, last_model_entry_index
    global last_model_signal, major_levels, cached_strategy_levels
    global last_level_calculation_index

    if current_index >= len(h1_data) - 1:
        return

    current_index += 1
    candle = h1_data[current_index]

    update_trade(candle)

    if ai_model is None or ai_scaler is None or ai_market_candles is None:
        return

    target_time = int(candle["time"]) + TIMEFRAME_SECONDS[TIMEFRAME]

    features = ai.market_state_at(
        ai_market_candles,
        target_time,
    )

    prediction = ai.predict_from_features(
        model=ai_model,
        scaler=ai_scaler,
        features=features,
    )

    current_price = ai.current_price_at(
        ai_market_candles,
        target_time,
    )
    atr = ai.atr_at(
        ai_market_candles,
        target_time,
    )

    last_model_signal = {
        "time": target_time,
        "price": float(candle["close"]),
        "prediction": copy.deepcopy(prediction),
        "current_price": current_price,
        "atr": atr,
    }

    if current_price is None or atr <= 0:
        blocked_trades += 1
        return

    if open_trade is not None or not trading_enabled:
        return

    if prediction.get("direction") == "WAIT":
        blocked_trades += 1
        return

    if float(prediction.get("setup_probability", 0.0)) < 0.50:
        blocked_trades += 1
        return

    if current_index - last_model_entry_index < MODEL_TESTING_COOLDOWN:
        blocked_trades += 1
        return

    direction = (
        "BUY"
        if prediction.get("direction") == "LONG"
        else "SELL"
    )

    offsets = prediction.get("price_offsets_atr", {})

    entry_price = (
        current_price
        + float(offsets.get("entry", 0.0)) * atr
    )
    sl_price = (
        current_price
        + float(offsets.get("sl", 0.0)) * atr
    )
    tp_price = (
        current_price
        + float(offsets.get("tp", 0.0)) * atr
    )

    # A model can predict arbitrary numeric offsets. Only accept a geometrically
    # valid LONG or SHORT setup for the existing execution engine.
    if direction == "BUY":
        if not (sl_price < entry_price < tp_price):
            blocked_trades += 1
            return
    else:
        if not (tp_price < entry_price < sl_price):
            blocked_trades += 1
            return

    setup = {
        "direction": direction,
        "pattern": "AI.PY MODEL SIGNAL",
        "level": current_price + float(offsets.get("level", 0.0)) * atr,
        "focus_1": current_price + float(offsets.get("focus_1", 0.0)) * atr,
        "focus_2": current_price + float(offsets.get("focus_2", 0.0)) * atr,
        "breakout": current_price + float(offsets.get("breakout", 0.0)) * atr,
        "entry": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "model_probabilities": copy.deepcopy(
            prediction.get("direction_probabilities", {})
        ),
        "setup_probability": float(
            prediction.get("setup_probability", 0.0)
        ),
        "setup_timeframe": prediction.get("setup_timeframe"),
        "entry_timeframe": prediction.get("entry_timeframe"),
        "breakout_time": target_time,
        "breakout_index": current_index,
        "breakout_candle": copy.deepcopy(candle),
        "model_symbol": SYMBOL,
        "model_timeframe": TIMEFRAME,
        "price_offsets_atr": copy.deepcopy(offsets),
    }

    if direction == "BUY":
        setup["low1"] = {
            "time": target_time,
            "price": float(candle["low"]),
        }
        setup["low2"] = {
            "time": target_time,
            "price": float(candle["low"]),
        }
    else:
        setup["high1"] = {
            "time": target_time,
            "price": float(candle["high"]),
        }
        setup["high2"] = {
            "time": target_time,
            "price": float(candle["high"]),
        }

    open_trade_position(
        direction,
        entry_price,
        setup,
    )

    if open_trade is not None:
        last_model_entry_index = current_index


def reset() -> None:
    global current_index, running, trading_enabled
    global balance, peak_equity, open_trade
    global trade_history, last_trade_result, major_levels
    global blocked_trades
    global cached_strategy_levels
    global blocked_trade_zone
    global last_level_calculation_index
    global zigzag_filter
    global analysis_status, analysis_started, analysis_completed
    global analysis_phase, model_available, model_info
    global last_training_info, last_model_signal, last_model_entry_index
    global manual_analysis_progress

    cached_strategy_levels = []
    last_level_calculation_index = -1

    current_index = 0
    running = False
    trading_enabled = True

    balance = STARTING_BALANCE
    peak_equity = STARTING_BALANCE

    open_trade = None
    trade_history = []
    last_trade_result = None
    major_levels = []
    blocked_trades = 0

    blocked_trade_zone = None
    zigzag_filter = None

    analysis_status = "idle"
    analysis_started = False
    analysis_completed = False
    analysis_phase = "idle"
    last_model_signal = None
    last_model_entry_index = -10_000_000

    # Load the market-specific PyTorch model created by ai.py.
    load_ai_runtime()

    _capture_active_session()


def train_runtime() -> Dict[str, Any]:
    """Synchronous compatibility helper for training the active market with ai.py."""
    global analysis_status, analysis_started, analysis_completed
    global analysis_phase, running, startup_progress, startup_message

    analysis_status = "training"
    analysis_phase = "training"
    analysis_started = True
    analysis_completed = False
    running = True

    startup_stage = "Training model"
    startup_progress = 75
    startup_message = (
        f"Training {SYMBOL} market model with ai.py..."
    )

    path = ai.train_market(SYMBOL)
    load_ai_runtime()

    running = False
    analysis_status = "completed"
    analysis_phase = "training_completed"
    analysis_completed = True

    _capture_active_session()

    return {
        "success": True,
        "model_path": str(path),
        "symbol": SYMBOL,
    }


async def training_loop() -> None:
    global running, startup_progress, startup_message
    global analysis_status, analysis_started, analysis_completed, analysis_phase

    running = True
    analysis_status = "training"
    analysis_phase = "training"
    analysis_started = True
    analysis_completed = False

    try:
        startup_progress = 70
        startup_message = (
            f"Loading Learning trades for {SYMBOL} {TIMEFRAME}..."
        )
        await broadcast_snapshot(user_id)

        await asyncio.sleep(0)

        startup_progress = 82
        startup_message = "Building numeric training dataset..."
        await broadcast_snapshot(user_id)

        await asyncio.sleep(0)

        result = await asyncio.to_thread(
            ai.train_market,
            SYMBOL,
        )
        load_ai_runtime()

        startup_progress = 100
        startup_message = (
            f"TRAINING complete for {SYMBOL} {TIMEFRAME}."
        )
        running = False
        analysis_status = "completed"
        analysis_phase = "training_completed"
        analysis_completed = True

        _capture_active_session()

        print("")
        print("=== TRAINING COMPLETE ===")
        print(f"Market: {SYMBOL} | Timeframe: {TIMEFRAME}")
        print(
            f"Learning trades: "
            f"{result['training']['input_trade_count']}"
        )
        print(
            f"Usable samples: "
            f"{result['training']['usable_positive_samples']}"
        )
        print(
            f"Background samples: "
            f"{result['training']['background_samples']}"
        )
        print(f"Model: {result['model']['model_path']}")

        await broadcast_snapshot(user_id)

    except asyncio.CancelledError:
        running = False
        if analysis_started and not analysis_completed:
            analysis_status = "paused"
            analysis_phase = "training_paused"
        _capture_active_session()
        raise

    except Exception as error:
        running = False
        analysis_status = "error"
        analysis_phase = "training_error"
        analysis_completed = False
        startup_progress = 100
        startup_message = f"TRAINING error: {error}"
        _capture_active_session()
        print("TRAINING ERROR:", error)
        await broadcast_snapshot(user_id)


async def testing_loop() -> None:
    global running, startup_progress, startup_message
    global analysis_status, analysis_started, analysis_completed, analysis_phase

    if not load_ai_runtime():
        analysis_status = "error"
        analysis_phase = "testing_error"
        analysis_started = True
        analysis_completed = False
        running = False
        startup_progress = 100
        startup_message = (
            f"No trained ai.py model exists for {SYMBOL}. "
            "Run TRAINING first."
        )
        _capture_active_session()
        await broadcast_snapshot(user_id)
        return

    reset()
    analysis_status = "testing"
    analysis_phase = "testing"
    analysis_started = True
    analysis_completed = False
    running = True

    total = max(1, len(h1_data) - 1)
    last_reported = -1
    last_trade_count = len(trade_history)

    print("")
    print("=== TESTING START ===")
    print(f"Market: {SYMBOL} | Timeframe: {TIMEFRAME}")
    print(f"Model: {ai.model_path(SYMBOL)}")
    print(f"Total candles: {total:,}")

    try:
        while current_index < len(h1_data) - 1:
            check_for_quit()
            step_testing()

            progress = int((current_index / total) * 100)
            trade_count = len(trade_history)

            if progress != last_reported or trade_count != last_trade_count:
                startup_progress = 70 + int((current_index / total) * 30)
                startup_message = (
                    f"Testing {current_index:,}/{total:,} "
                    f"| {progress}% | Trades: {trade_count}"
                )

                if progress != last_reported:
                    print(
                        f"[TESTING] {progress:3d}% | "
                        f"Candles: {current_index:,}/{total:,} | "
                        f"Trades: {trade_count} | "
                        f"Balance: {balance:,.2f}"
                    )

                if trade_count != last_trade_count:
                    trade = trade_history[-1]
                    print(
                        f"[TRADE] #{trade_count} "
                        f"{trade['direction']} "
                        f"{trade['result']} "
                        f"Entry={trade['entry']:.5f} "
                        f"Exit={trade['exit']:.5f}"
                    )

                last_reported = progress
                last_trade_count = trade_count
                _capture_active_session()
                await broadcast_snapshot(user_id)

            await asyncio.sleep(0)

        # The final candle closes any still-open test position at the last
        # available market price so the test has a deterministic final result.
        if open_trade is not None:
            final_price = float(h1_data[-1]["close"])
            if open_trade["direction"] == "BUY":
                result = "WIN" if final_price >= open_trade["entry"] else "LOSS"
            else:
                result = "WIN" if final_price <= open_trade["entry"] else "LOSS"
            close_trade(result, final_price, "END_OF_TEST_DATA")

        startup_progress = 100
        startup_message = "TESTING complete."
        running = False
        analysis_status = "completed"
        analysis_phase = "testing_completed"
        analysis_started = True
        analysis_completed = True

        _capture_active_session()

        print("")
        print("=== TESTING COMPLETE ===")
        print(f"Market: {SYMBOL} | Timeframe: {TIMEFRAME}")
        print(f"Trades: {len(trade_history)}")
        print(f"Balance: {balance:,.2f}")
        print(
            f"WIN: {sum(1 for trade in trade_history if trade['result'] == 'WIN')}"
        )
        print(
            f"LOSS: {sum(1 for trade in trade_history if trade['result'] == 'LOSS')}"
        )

        await broadcast_snapshot(user_id)

    except asyncio.CancelledError:
        running = False
        if analysis_started and not analysis_completed:
            analysis_status = "paused"
            analysis_phase = "testing_paused"
        _capture_active_session()
        raise

    except Exception as error:
        running = False
        analysis_status = "error"
        analysis_phase = "testing_error"
        analysis_completed = False
        startup_progress = 100
        startup_message = f"TESTING error: {error}"
        _capture_active_session()
        print("TESTING ERROR:", error)
        await broadcast_snapshot(user_id)


def startup_snapshot() -> Dict[str, Any]:
    return {
        "ready": False,
        "loading": True,
        "startup_stage": startup_stage,
        "startup_progress": startup_progress,
        "startup_message": startup_message,
        "mt5_connected": mt5_connected,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "session_id": _session_id(),
        "candles_loaded": len(h1_data),
        "analysis_status": analysis_status,
        "analysis_phase": analysis_phase,
        "model_available": model_available,
        "model_info": model_info,
        "last_training_info": last_training_info,
    }

def get_trade_detail(
    trade_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Return an exact, point-in-time explanation of one trade.

    No strategy is recalculated here.
    We only return the data that was stored when the trade was created.
    """

    # Closed trades.
    for trade in trade_history:
        if int(trade.get("trade_id", -1)) == int(trade_id):
            return copy.deepcopy(trade)

    # Currently open trade.
    if (
        open_trade is not None
        and int(open_trade.get("trade_id", -1)) == int(trade_id)
    ):
        return copy.deepcopy(open_trade)

    return None

def snapshot(user_id: Optional[str] = None) -> Dict[str, Any]:
    visible = h1_data[:current_index + 1]

    manual_state = _get_manual_analysis_state(user_id)
    use_manual_state = manual_state["analysis_phase"] != "idle"

    analysis_status_value = (
        manual_state["analysis_status"]
        if use_manual_state
        else analysis_status
    )

    analysis_started_value = (
        manual_state["analysis_started"]
        if use_manual_state
        else analysis_started
    )

    analysis_completed_value = (
        manual_state["analysis_completed"]
        if use_manual_state
        else analysis_completed
    )

    analysis_phase_value = (
        manual_state["analysis_phase"]
        if use_manual_state
        else analysis_phase
    )

    manual_analysis_progress_value = (
        manual_state["manual_analysis_progress"]
        if use_manual_state
        else manual_analysis_progress
    )

    results_source_value = (
        manual_state["results_source"]
        if use_manual_state
        else results_source
    )

    manual_analysis_results_value = (
        manual_state["manual_analysis_results"]
        if use_manual_state
        else manual_analysis_results
    )

    if not visible:
        return {
            "ready": False,
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "session_id": _session_id(),
            "mt5_connected": mt5_connected,
            "analysis_status": analysis_status_value,
            "analysis_started": analysis_started_value,
            "analysis_completed": analysis_completed_value,
            "analysis_phase": analysis_phase_value,
            "manual_analysis_progress": manual_analysis_progress_value,
            "model_available": model_available,
            "model_info": model_info,
            "last_training_info": last_training_info,
            "learning_trades": get_trading_setups(SYMBOL, user_id),
        }

    decision = evaluate_strategy(visible)

    # Keep the currently active session synchronized whenever a snapshot is
    # produced. This is what makes a browser reload recover the exact active
    # market + timeframe state without starting analysis again.
    _capture_active_session()

    return {
        "ready": True,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "session_id": _session_id(),
        "analysis_status": analysis_status_value,
        "analysis_started": analysis_started_value,
        "analysis_completed": analysis_completed_value,
        "analysis_phase": analysis_phase_value,
        "manual_analysis_progress": manual_analysis_progress_value,
        "current_time": visible[-1]["time"],
        "current_price": visible[-1]["close"],
        "h1": h1_data,
        "line": [
            {
                "time": candle["time"],
                "value": candle["close"],
            }
            for candle in h1_data
        ],
        "levels": decision["levels"],
        "strategy": decision,
        "trade": open_trade,
        "trades": trade_history,
        "balance": balance,
        "equity": balance,

        "starting_balance": STARTING_BALANCE,
        "final_balance": balance,

        "total_return": (
            ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100.0
            if STARTING_BALANCE
            else 0.0
        ),

        "blocked_trades": blocked_trades,

        "trading_enabled": trading_enabled,
        "model_available": model_available,
        "model_info": model_info,
        "last_training_info": last_training_info,
        "last_model_signal": last_model_signal,

        "learning_trades": get_trading_setups(SYMBOL, user_id),

        "results_source": results_source_value,
        "manual_analysis_results": manual_analysis_results_value,
    }


async def broadcast_snapshot(
    user_id: Optional[str] = None
) -> None:

    clients = connected_clients.get(
        user_id,
        set()
    )

    if not clients:
        return

    data = snapshot(user_id)

    manual_state = _get_manual_analysis_state(user_id)

    if manual_state["analysis_phase"] != "idle":
        data["startup_stage"] = manual_state["startup_stage"]
        data["startup_progress"] = manual_state["startup_progress"]
        data["startup_message"] = manual_state["startup_message"]
    else:
        data["startup_stage"] = startup_stage
        data["startup_progress"] = startup_progress
        data["startup_message"] = startup_message

    payload = json.dumps(data)

    dead_clients = []

    for client in list(clients):
        try:
            await client.send_text(payload)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        clients.discard(client)

    if not clients:
        connected_clients.pop(user_id, None)

def check_for_quit():
    if msvcrt.kbhit():
        key = msvcrt.getwch().lower()

        if key == "q":
            print("")
            print("Q pressed -> shutting down immediately.")
            os._exit(0)

@asynccontextmanager
async def lifespan(_app: FastAPI):

    global startup_stage, startup_progress, startup_message
    global analysis_status, analysis_started, analysis_completed

    if WEB_MODE:
        mt5_available = False
        print("WEB_MODE: MT5 disabled. Using CHARTS data.")
    else:
        mt5_available = initialize_mt5()

    startup_stage = "Loading market data"
    startup_progress = 3

    if mt5_available:
        startup_message = "Loading local history and synchronizing MT5..."
    else:
        startup_message = (
            "MT5 is switched off. "
            "Loading local chart history..."
        )

    # First application state is always the backend defaults. Once a browser
    # session is active, page reloads keep using this exact live state.
    _get_session_state(SYMBOL, TIMEFRAME)
    load_history()
    _restore_active_session()
    load_ai_runtime()

    startup_stage = "Chart ready"
    startup_progress = 100
    startup_message = (
        f"{SYMBOL} {TIMEFRAME} ready. Waiting for analysis..."
    )

    global analysis_task

    try:
        yield

    finally:
        if analysis_task is not None:
            analysis_task.cancel()

            try:
                await analysis_task
            except asyncio.CancelledError:
                pass

        analysis_task = None

        if h1_data:
            _capture_active_session()

        startup_stage = "Stopped"
        startup_progress = 100
        startup_message = "Backend stopped."

        if not WEB_MODE:
            mt5.shutdown()


app = FastAPI(lifespan=lifespan)

@app.get("/api/config")
async def get_config():
    return {
        "web_mode": WEB_MODE
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/frontend.html")
async def frontend_page():
    return FileResponse(os.path.join(BASE_DIR, "frontend.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    global WEB_MODE

    user_id = None

    if WEB_MODE:
        ticket = websocket.query_params.get("ticket", "")
        user_id = verify_tradingbot_ticket(ticket)

        if user_id is None:
            await websocket.close(code=1008)
            return

    global analysis_task
    
    global startup_stage, startup_progress, startup_message
    global results_source
    global manual_analysis_results
    global current_index, running, trading_enabled
    global balance, peak_equity, open_trade
    global trade_history, last_trade_result, major_levels
    global cached_strategy_levels, last_level_calculation_index
    global blocked_trade_zone, zigzag_filter, blocked_trades
    global analysis_status, analysis_started, analysis_completed
    global analysis_phase, model_available, model_info
    global last_training_info, last_model_signal, last_model_entry_index

    await websocket.accept()

    connected_clients.setdefault(
        user_id,
        set()
    ).add(websocket)

    # The backend is the single source of truth for market + timeframe.
    await websocket.send_text(
        json.dumps(startup_snapshot())
    )

    if h1_data:
        data = snapshot(user_id)

        data["startup_stage"] = startup_stage
        data["startup_progress"] = startup_progress
        data["startup_message"] = startup_message

        await websocket.send_text(
            json.dumps(data)
        )

    while True:
        try:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            action = message.get("action")

            # =====================================================
            # MARKET SWITCH
            # =====================================================

            if action == "symbol":
                new_symbol = message.get("symbol", "")

                try:
                    print("")
                    print(
                        f"=== SWITCHING MARKET TO {new_symbol} ==="
                    )

                    user_analysis_task = USER_MANUAL_ANALYSIS_TASKS.get(user_id)

                    if user_analysis_task is not None:
                        if not user_analysis_task.done():
                            user_analysis_task.cancel()

                            try:
                                await user_analysis_task
                            except asyncio.CancelledError:
                                pass

                        USER_MANUAL_ANALYSIS_TASKS.pop(
                            user_id,
                            None
                        )

                    if analysis_task is not None:
                        analysis_task.cancel()

                        try:
                            await analysis_task
                        except asyncio.CancelledError:
                            pass

                        analysis_task = None

                    # Save the old market/timeframe session before switching.
                    _capture_active_session()

                    data = snapshot(user_id)
                    data["startup_stage"] = "Loading market data"
                    data["startup_progress"] = 5
                    data["startup_message"] = (
                        f"Loading {new_symbol} {TIMEFRAME}..."
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    await asyncio.to_thread(
                        switch_symbol,
                        new_symbol
                    )

                    load_ai_runtime()

                    data = snapshot(user_id)
                    data["startup_stage"] = "Chart ready"
                    data["startup_progress"] = 100
                    data["startup_message"] = (
                        f"{SYMBOL} {TIMEFRAME} ready. "
                        f"Waiting for analysis..."
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except Exception as error:
                    print(
                        f"Symbol switch error for {new_symbol}:",
                        error
                    )

                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 0,
                            "startup_message": str(error),
                        })
                    )

                    continue

            # =====================================================
            # TIMEFRAME SWITCH
            # =====================================================

            if action == "timeframe":
                new_timeframe = message.get("timeframe", "")

                if WEB_MODE and new_timeframe in {"M1", "M5"}:
                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 0,
                            "startup_message": "M1 and M5 are not available in web mode.",
                        })
                    )
                    continue

                try:
                    print("")
                    print(
                        f"=== SWITCHING TIMEFRAME TO {new_timeframe} ==="
                    )

                    user_analysis_task = USER_MANUAL_ANALYSIS_TASKS.get(user_id)

                    if user_analysis_task is not None:
                        if not user_analysis_task.done():
                            user_analysis_task.cancel()

                            try:
                                await user_analysis_task
                            except asyncio.CancelledError:
                                pass

                        USER_MANUAL_ANALYSIS_TASKS.pop(
                            user_id,
                            None
                        )

                    if analysis_task is not None:
                        analysis_task.cancel()

                        try:
                            await analysis_task
                        except asyncio.CancelledError:
                            pass

                        analysis_task = None

                    # Keep this market as the same logical session. Only the
                    # timeframe child-session changes.
                    _capture_active_session()

                    data = snapshot(user_id)
                    data["startup_stage"] = "Loading market data"
                    data["startup_progress"] = 5
                    data["startup_message"] = (
                        f"Loading {SYMBOL} {new_timeframe}..."
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    await asyncio.to_thread(
                        switch_timeframe,
                        new_timeframe
                    )

                    load_ai_runtime()

                    data = snapshot(user_id)
                    data["startup_stage"] = "Chart ready"
                    data["startup_progress"] = 100
                    data["startup_message"] = (
                        f"{SYMBOL} {TIMEFRAME} ready. "
                        f"Waiting for analysis..."
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except Exception as error:
                    print(
                        f"Timeframe switch error for {new_timeframe}:",
                        error
                    )

                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 0,
                            "startup_message": str(error),
                        })
                    )

                    continue

            # =====================================================
            # UPDATE ALL LOCAL CHART DATA
            # =====================================================

            if action == "update_data":
                try:
                    print("")
                    print("=== UPDATING ALL LOCAL CHART DATA ===")

                    result = await asyncio.to_thread(
                        update_all_chart_data
                    )

                    data = snapshot(user_id)

                    data["startup_stage"] = startup_stage
                    data["startup_progress"] = startup_progress
                    data["startup_message"] = result["message"]
                    data["mt5_connected"] = mt5_connected
                    data["data_update"] = result

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except Exception as error:
                    print(
                        "Update data error:",
                        error
                    )

                    data = snapshot(user_id)

                    data["startup_stage"] = "Error"
                    data["startup_progress"] = 100
                    data["startup_message"] = str(error)
                    data["mt5_connected"] = mt5_connected
                    data["data_update"] = {
                        "success": False,
                        "message": str(error)
                    }

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

            # =====================================================
            # MANUAL ANALYSIS
            # =====================================================

            if action == "manual_analysis":
                try:

                    manual_state = _get_manual_analysis_state(user_id)
                    user_analysis_task = USER_MANUAL_ANALYSIS_TASKS.get(user_id)

                    if (
                        user_analysis_task is not None
                        and not user_analysis_task.done()
                    ):
                        data = snapshot(user_id)

                        data["startup_stage"] = (
                            "Manual analysis running"
                        )

                        data["startup_progress"] = (
                            startup_progress
                        )

                        data["startup_message"] = (
                            f"MANUAL ANALYSIS is already "
                            f"running for "
                            f"{SYMBOL} {TIMEFRAME}..."
                        )

                        await websocket.send_text(
                            json.dumps(data)
                        )

                        continue

                    manual_state["results_source"] = "manual"
                    manual_state["manual_analysis_results"] = {}
                    manual_state["manual_analysis_progress"] = 0

                    manual_state["analysis_status"] = "manual_analysis"
                    manual_state["analysis_phase"] = "manual_analysis"
                    manual_state["analysis_started"] = True
                    manual_state["analysis_completed"] = False

                    manual_state["startup_stage"] = "Manual analysis"
                    manual_state["startup_progress"] = 70
                    manual_state["startup_message"] = (
                        f"Starting MANUAL ANALYSIS "
                        f"for {SYMBOL} {TIMEFRAME}..."
                    )

                    USER_MANUAL_ANALYSIS_TASKS[user_id] = asyncio.create_task(
                        manual_analysis_loop(user_id)
                    )

                    data = snapshot(user_id)

                    data["manual_analysis_progress"] = (
                        manual_state["manual_analysis_progress"]
                    )

                    data["startup_stage"] = (
                        manual_state["startup_stage"]
                    )

                    data["startup_progress"] = (
                        manual_state["startup_progress"]
                    )

                    data["startup_message"] = (
                        manual_state["startup_message"]
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except Exception as error:

                    print(
                        "MANUAL ANALYSIS start error:",
                        error
                    )

                    manual_state = _get_manual_analysis_state(user_id)

                    manual_state["results_source"] = "testing"
                    manual_state["manual_analysis_results"] = {}
                    manual_state["analysis_status"] = "error"
                    manual_state["analysis_phase"] = "manual_analysis_error"
                    manual_state["analysis_started"] = True
                    manual_state["analysis_completed"] = False
                    manual_state["startup_progress"] = 100
                    manual_state["startup_message"] = str(error)

                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 100,
                            "startup_message": str(error),
                            "analysis_status": "error",
                            "analysis_phase":
                                "manual_analysis_error",
                        })
                    )

                    continue
                    
            # =====================================================
            # TRAINING
            # =====================================================

            if action in ("training", "play"):
                try:
                    print("")
                    print(
                        f"=== TRAINING REQUESTED: {SYMBOL} {TIMEFRAME} ==="
                    )

                    if analysis_task is not None and not analysis_task.done():
                        data = snapshot(user_id)
                        data["startup_stage"] = "Training running"
                        data["startup_progress"] = startup_progress
                        data["startup_message"] = (
                            f"TRAINING is already running for "
                            f"{SYMBOL} {TIMEFRAME}..."
                        )
                        await websocket.send_text(json.dumps(data))
                        continue
                       
                    results_source = "testing"
                    manual_analysis_results = {}

                    analysis_task = None
                    reset()

                    analysis_status = "training"
                    analysis_phase = "training"
                    analysis_started = True
                    analysis_completed = False
                    startup_stage = "Training model"
                    startup_progress = 70
                    startup_message = (
                        f"Starting TRAINING for {SYMBOL} {TIMEFRAME}..."
                    )

                    analysis_task = asyncio.create_task(training_loop())

                    data = snapshot(user_id)
                    data["startup_stage"] = startup_stage
                    data["startup_progress"] = startup_progress
                    data["startup_message"] = startup_message
                    await websocket.send_text(json.dumps(data))
                    continue

                except Exception as error:
                    print("TRAINING start error:", error)
                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 0,
                            "startup_message": str(error),
                            "analysis_status": "error",
                            "analysis_phase": "training_error",
                        })
                    )
                    continue


            # =====================================================
            # TESTING
            # =====================================================

            if action == "testing":
                try:
                    print("")
                    print(
                        f"=== TESTING REQUESTED: {SYMBOL} {TIMEFRAME} ==="
                    )

                    if analysis_task is not None and not analysis_task.done():
                        data = snapshot(user_id)
                        data["startup_stage"] = "Testing running"
                        data["startup_progress"] = startup_progress
                        data["startup_message"] = (
                            f"TESTING is already running for "
                            f"{SYMBOL} {TIMEFRAME}..."
                        )
                        await websocket.send_text(json.dumps(data))
                        continue

                    if not load_ai_runtime():
                        data = snapshot(user_id)
                        data["startup_stage"] = "Testing unavailable"
                        data["startup_progress"] = 100
                        data["startup_message"] = (
                            f"No trained ai.py model exists for {SYMBOL}. "
                            "Run TRAINING first."
                        )
                        data["analysis_status"] = "error"
                        data["analysis_phase"] = "testing_error"
                        await websocket.send_text(json.dumps(data))
                        continue

                    results_source = "testing"
                    manual_analysis_results = {}

                    analysis_task = None
                    reset()

                    analysis_status = "testing"
                    analysis_phase = "testing"
                    analysis_started = True
                    analysis_completed = False
                    startup_stage = "Testing model"
                    startup_progress = 70
                    startup_message = (
                        f"Starting TESTING for {SYMBOL} {TIMEFRAME}..."
                    )

                    analysis_task = asyncio.create_task(testing_loop())

                    data = snapshot(user_id)
                    data["startup_stage"] = startup_stage
                    data["startup_progress"] = startup_progress
                    data["startup_message"] = startup_message
                    await websocket.send_text(json.dumps(data))
                    continue

                except Exception as error:
                    print("TESTING start error:", error)
                    await websocket.send_text(
                        json.dumps({
                            "ready": False,
                            "loading": False,
                            "symbol": SYMBOL,
                            "timeframe": TIMEFRAME,
                            "session_id": _session_id(),
                            "startup_stage": "Error",
                            "startup_progress": 0,
                            "startup_message": str(error),
                            "analysis_status": "error",
                            "analysis_phase": "testing_error",
                        })
                    )
                    continue


            # =====================================================
            # LEARNING / TRADING SET UP SAVE
            # =====================================================

            if action == "learning_save":
                try:
                    setup = message.get("setup")

                    if not isinstance(setup, dict):
                        raise ValueError(
                            "Invalid learning setup data."
                        )

                    saved_setup = await asyncio.to_thread(
                        save_trading_setup,
                        SYMBOL,
                        setup,
                        user_id
                    )

                    data = snapshot(user_id)

                    data["learning_action"] = "saved"
                    data["learning_setup"] = saved_setup

                    data["learning_trades"] = await asyncio.to_thread(
                        get_trading_setups,
                        SYMBOL,
                        user_id
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except Exception as error:
                    print(
                        "Learning setup save error:",
                        error
                    )

                    await websocket.send_text(
                        json.dumps({
                            "learning_action": "save_error",
                            "success": False,
                            "error": str(error),
                        })
                    )

                    continue

            # =====================================================
            # LEARNING / TRADING SET UP DELETE
            # =====================================================

            if action == "learning_delete":
                try:
                    setup_id = int(
                        message.get("setup_id")
                    )

                    deleted = await asyncio.to_thread(
                        delete_trading_setup,
                        SYMBOL,
                        setup_id,
                        user_id
                    )

                    data = snapshot(user_id)

                    data["learning_action"] = "deleted"
                    data["learning_setup_id"] = setup_id
                    data["learning_deleted"] = deleted

                    data["learning_trades"] = await asyncio.to_thread(
                        get_trading_setups,
                        SYMBOL,
                        user_id
                    )

                    await websocket.send_text(
                        json.dumps(data)
                    )

                    continue

                except (TypeError, ValueError):
                    await websocket.send_text(
                        json.dumps({
                            "learning_action": "delete_error",
                            "success": False,
                            "error": "Invalid setup_id.",
                        })
                    )

                    continue

                except Exception as error:
                    print(
                        "Learning setup delete error:",
                        error
                    )

                    await websocket.send_text(
                        json.dumps({
                            "learning_action": "delete_error",
                            "success": False,
                            "error": str(error),
                        })
                    )

                    continue

            # =====================================================
            # LEARNING / TRADING SET UP LOAD
            # =====================================================

            if action == "learning_load":
                try:
                    setups = await asyncio.to_thread(
                        get_trading_setups,
                        SYMBOL,
                        user_id
                    )

                    await websocket.send_text(
                        json.dumps({
                            "learning_action": "loaded",
                            "success": True,
                            "symbol": SYMBOL,
                            "learning_trades": setups,
                        })
                    )

                    continue

                except Exception as error:
                    print(
                        "Learning setup load error:",
                        error
                    )

                    await websocket.send_text(
                        json.dumps({
                            "learning_action": "load_error",
                            "success": False,
                            "error": str(error),
                        })
                    )

                    continue

            

            # =====================================================
            # TRADE DETAIL
            # =====================================================

            if action == TRADE_DETAIL_ACTION:
                trade_id_raw = message.get("trade_id")

                try:
                    trade_id = int(trade_id_raw)
                except (TypeError, ValueError):
                    await websocket.send_text(
                        json.dumps({
                            "type": "trade_detail",
                            "success": False,
                            "error": "Invalid trade_id.",
                        })
                    )
                    continue

                trade_detail = get_trade_detail(
                    trade_id
                )

                if trade_detail is None:
                    await websocket.send_text(
                        json.dumps({
                            "type": "trade_detail",
                            "success": False,
                            "trade_id": trade_id,
                            "error": (
                                f"Trade #{trade_id} "
                                "was not found."
                            ),
                        })
                    )
                    continue

                await websocket.send_text(
                    json.dumps({
                        "type": "trade_detail",
                        "success": True,
                        "trade_id": trade_id,
                        "symbol": SYMBOL,
                        "timeframe": TIMEFRAME,
                        "trade": trade_detail,
                    })
                )

                continue

            # =====================================================
            # VIEWPORT CHANGE
            # =====================================================

            if action == "viewport":
                from_time = int(
                    message.get("from", 0)
                )

                to_time = int(
                    message.get("to", 0)
                )

                view_levels = find_levels_for_view(
                    from_time,
                    to_time
                )

                data = snapshot(user_id)

                data["levels"] = view_levels

                data["startup_stage"] = startup_stage
                data["startup_progress"] = startup_progress
                data["startup_message"] = startup_message

                await websocket.send_text(
                    json.dumps(data)
                )

                continue

        except WebSocketDisconnect:
            print("Browser disconnected.")
            break

        except Exception as error:
            print("WebSocket error:", error)
            break

    clients = connected_clients.get(user_id)

    if clients is not None:
        clients.discard(websocket)

        if not clients:
            connected_clients.pop(user_id, None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/",
    StaticFiles(
        directory=BASE_DIR,
        html=True
    ),
    name="static"
)

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
