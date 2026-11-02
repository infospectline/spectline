# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

try:
    import ai
except ImportError:
    ai = None

try:
    from client import (
        BASE_DIR,
        ClientIdentity,
        ensure_client_directories,
        get_model_path,
        resolve_identity,
    )
    from runtime import RuntimeContext, RuntimeRegistry, cancel_runtime_tasks
    from webmode import (
        AuthenticationError,
        VerifiedTradingBotClaims,
        accept_customer_connection,
        authenticate_websocket,
        disconnect_customer_connection,
        reject_websocket,
        ticket_identity_for_market,
        WEB_MODE,
    )
    from chart import ChartService, TIMEFRAME_SECONDS
    import setups as setup_storage
    import strategy as strategy_engine
    import trading as trading_engine
except ImportError:
    from .client import (
        BASE_DIR,
        ClientIdentity,
        ensure_client_directories,
        get_model_path,
        resolve_identity,
    )
    from .runtime import RuntimeContext, RuntimeRegistry, cancel_runtime_tasks
    from .webmode import (
        AuthenticationError,
        VerifiedTradingBotClaims,
        accept_customer_connection,
        authenticate_websocket,
        disconnect_customer_connection,
        reject_websocket,
        ticket_identity_for_market,
        WEB_MODE,
    )
    from .chart import ChartService, TIMEFRAME_SECONDS
    from . import setups as setup_storage
    from . import strategy as strategy_engine
    from . import trading as trading_engine


# ============================================================================
# CONFIGURATION
# ============================================================================

LOGGER = logging.getLogger("spectline.backend")

DEFAULT_MARKET = os.getenv("SPECTLINE_DEFAULT_MARKET", "EURUSD").strip().upper()
DEFAULT_TIMEFRAME = os.getenv("SPECTLINE_DEFAULT_TIMEFRAME", "H1").strip().upper()
STARTING_BALANCE = float(os.getenv("SPECTLINE_STARTING_BALANCE", "100000"))
MANUAL_RISK_PERCENT = float(os.getenv("SPECTLINE_MANUAL_RISK_PERCENT", "1.0"))
LEVEL_RECALC_INTERVAL = int(os.getenv("SPECTLINE_LEVEL_RECALC_INTERVAL", "20"))

RUNTIME_REGISTRY = RuntimeRegistry()
CHART_SERVICE = ChartService()
TRADING_CONFIG = trading_engine.TradingConfig()

# Shared chart files are protected from concurrent load/update operations.
CHART_IO_LOCK = asyncio.Lock()

FRONTEND_HTML_CANDIDATES = (
    "frontend.html",
    "index.html",
    "frontend(1).html",
)
FRONTEND_JS_CANDIDATES = (
    "frontend.js",
    "frontend(1).js",
)
FRONTEND_CSS_CANDIDATES = (
    "frontend.css",
    "frontend(1).css",
)


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def _json_default(value: Any) -> Any:
    """Convert common Python and NumPy values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return list(value)

    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()

    if hasattr(value, "item") and callable(value.item):
        return value.item()

    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, default=_json_default, ensure_ascii=False)


def _find_frontend_file(candidates: Sequence[str]) -> Optional[Path]:
    for filename in candidates:
        candidate = BASE_DIR / filename
        if candidate.is_file():
            return candidate
    return None


# ============================================================================
# IDENTITY AND RUNTIME RESOLUTION
# ============================================================================

def _identity_for_market(
    market: str,
    claims: Optional[VerifiedTradingBotClaims],
    current_identity: Optional[ClientIdentity] = None,
) -> ClientIdentity:
    """Create a market-specific identity from trusted connection claims."""
    if claims is not None:
        return ticket_identity_for_market(claims, market)

    if WEB_MODE:
        raise AuthenticationError("Verified customer claims are required in WEB_MODE.")

    if current_identity is not None:
        return resolve_identity(
            market=market,
            user_id=current_identity.user_id,
            strategy_id=current_identity.strategy_id,
            tradingbot_id=current_identity.tradingbot_id,
        )

    return resolve_identity(market=market)


def _get_or_create_runtime(
    identity: ClientIdentity,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> RuntimeContext:
    runtime = RUNTIME_REGISTRY.get_or_create(identity, timeframe)
    ensure_client_directories(identity)
    return runtime


# ============================================================================
# AI INTEGRATION BOUNDARY
# ============================================================================

class AIIntegrationPending(RuntimeError):
    """Raised until ai.py supports full TradingBot identity routing."""


def _identity_kwargs_for_ai(function: Any, identity: ClientIdentity) -> Dict[str, Any]:
    """Build identity arguments only when the AI API supports full isolation."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as error:
        raise AIIntegrationPending(
            "Cannot inspect the AI function signature."
        ) from error

    if "identity" in parameters:
        return {"identity": identity}

    required_identity_fields = {"user_id", "strategy_id", "tradingbot_id"}
    if required_identity_fields.issubset(parameters):
        return {
            "user_id": identity.user_id,
            "strategy_id": identity.strategy_id,
            "tradingbot_id": identity.tradingbot_id,
        }

    raise AIIntegrationPending(
        "ai.py still accepts only market/user_id and does not yet accept "
        "strategy_id and tradingbot_id. The backend will not load or train "
        "a legacy model because it could belong to another TradingBot."
    )


def _verify_checkpoint_identity(
    checkpoint: Dict[str, Any],
    identity: ClientIdentity,
) -> None:
    """Reject checkpoints that do not identify the exact TradingBot owner."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("AI checkpoint is not a valid dictionary.")
    stored_identity = checkpoint.get("identity")
    if not isinstance(stored_identity, dict):
        stored_identity = checkpoint

    expected = identity.as_dict()
    missing = [
        field_name
        for field_name in ("user_id", "market", "strategy_id", "tradingbot_id")
        if stored_identity.get(field_name) != expected[field_name]
    ]
    if missing:
        raise RuntimeError(
            "AI checkpoint identity is missing or does not match this TradingBot: "
            + ", ".join(missing)
        )


def load_ai_runtime(runtime: RuntimeContext) -> bool:
    """Load the model and shared market features for one TradingBot runtime."""
    session = runtime.current_session

    runtime.ai_model = None
    runtime.ai_scaler = None
    runtime.ai_checkpoint = {}
    runtime.ai_market_candles = None
    runtime.ai_error = None
    session.model_available = False
    session.model_info = {}
    session.last_training_info = {}

    if ai is None:
        runtime.ai_error = "The ai.py module could not be imported."
        return False

    try:
        identity_kwargs = _identity_kwargs_for_ai(ai.load_model, runtime.identity)
        model, scaler, checkpoint = ai.load_model(
            runtime.market,
            **identity_kwargs,
        )
        _verify_checkpoint_identity(checkpoint, runtime.identity)

        market_candles = ai.load_all_market_candles(
            runtime.market,
            identity=runtime.identity,
        )
        if not market_candles:
            raise RuntimeError(
                f"No AI market candle history is available for {runtime.market}."
            )

        runtime.ai_model = model
        runtime.ai_scaler = scaler
        runtime.ai_checkpoint = checkpoint
        runtime.ai_market_candles = market_candles
        session.model_available = True

        model_info = {
            **runtime.identity.as_dict(),
            "model_path": str(get_model_path(runtime.identity)),
            "trained_at": checkpoint.get("trained_at"),
            "format_version": checkpoint.get("format_version"),
            "training_summary": checkpoint.get("training_summary", {}),
        }
        session.model_info = model_info
        session.last_training_info = copy.deepcopy(
            checkpoint.get("training_summary", {})
        )
        return True

    except FileNotFoundError as error:
        runtime.ai_error = str(error)
        session.model_available = False
        return False

    except Exception as error:
        runtime.ai_error = str(error)
        session.model_available = False
        LOGGER.warning(
            "AI model load failed for %s: %s",
            runtime.identity.key,
            error,
        )
        return False


def _train_ai_for_runtime(runtime: RuntimeContext) -> Dict[str, Any]:
    """Call ai.py only when its API accepts the complete TradingBot identity."""
    if ai is None:
        raise AIIntegrationPending("The ai.py module could not be imported.")

    identity_kwargs = _identity_kwargs_for_ai(ai.train_market, runtime.identity)
    return ai.train_market(
        runtime.market,
        **identity_kwargs,
    )

def _prepare_ai_data_for_runtime(
    runtime: RuntimeContext,
) -> Dict[str, Any]:
    """Prepare and store training data for one exact TradingBot identity."""

    if ai is None:
        raise AIIntegrationPending(
            "The ai.py module could not be imported."
        )

    prepare_function = getattr(
        ai,
        "prepare_and_store_training_data",
        None,
    )

    if not callable(prepare_function):
        raise AIIntegrationPending(
            "ai.py does not expose prepare_and_store_training_data()."
        )

    bundle, stored_path = prepare_function(
        runtime.market,
        identity=runtime.identity,
    )

    validation = bundle.get("validation")

    if not isinstance(validation, dict):
        validation = ai.validate_training_data(bundle)

    report = bundle.get("report", {})

    return {
        "identity": runtime.identity.as_dict(),
        "bundle_status": bundle.get("status", "UNKNOWN"),
        "bundle_file": stored_path.name,
        "report": report,
        "validation": validation,
        "stored": stored_path.is_file(),
        "ready_for_next_stage": bool(
            report.get("ready_for_next_stage", False)
        ),
        "model_training_started": False,
    }


# ============================================================================
# CHART AND SNAPSHOT HELPERS
# ============================================================================

def startup_snapshot(runtime: RuntimeContext) -> Dict[str, Any]:
    session = runtime.current_session
    return {
        "ready": False,
        "loading": True,
        "startup_stage": runtime.startup_stage,
        "startup_progress": runtime.startup_progress,
        "startup_message": runtime.startup_message,
        "mt5_connected": CHART_SERVICE.mt5_connected,
        "symbol": runtime.market,
        "timeframe": runtime.active_timeframe,
        "session_id": runtime.session_id,
        "candles_loaded": len(runtime.get_chart_data()),
        "analysis_status": session.analysis_status,
        "analysis_phase": session.analysis_phase,
        "model_available": session.model_available,
        "model_info": session.model_info,
        "last_training_info": session.last_training_info,
    }


def snapshot(runtime: RuntimeContext) -> Dict[str, Any]:
    """Build the legacy-compatible frontend snapshot for one TradingBot."""
    session = runtime.current_session
    candles = runtime.get_chart_data()
    visible = candles[:session.current_index + 1]
    manual = runtime.manual_analysis

    use_manual_state = manual.analysis_phase != "idle"
    analysis_status = manual.analysis_status if use_manual_state else session.analysis_status
    analysis_started = manual.analysis_started if use_manual_state else session.analysis_started
    analysis_completed = manual.analysis_completed if use_manual_state else session.analysis_completed
    analysis_phase = manual.analysis_phase if use_manual_state else session.analysis_phase
    manual_progress = manual.progress if use_manual_state else 0
    results_source = manual.results_source if use_manual_state else runtime.results_source
    manual_results = manual.results if use_manual_state else {}

    learning_trades: List[Dict[str, Any]] = []
    try:
        learning_trades = setup_storage.get_trading_setups(runtime.identity)
    except Exception as error:
        LOGGER.warning("Could not load Learning setups for %s: %s", runtime.key, error)

    decision = {
        "action": "WAIT",
        "reason": "No chart data available.",
        "levels": session.major_levels,
        "model_available": session.model_available,
    }
    if visible:
        try:
            decision = strategy_engine.evaluate_strategy(
                runtime,
                visible,
                ai_api=ai,
                level_recalc_interval=LEVEL_RECALC_INTERVAL,
            )
        except Exception as error:
            decision = {
                "action": "WAIT",
                "reason": f"Strategy evaluation error: {error}",
                "levels": session.major_levels,
                "model_available": session.model_available,
            }

    common = {
        "symbol": runtime.market,
        "timeframe": runtime.active_timeframe,
        "session_id": runtime.session_id,
        "mt5_connected": CHART_SERVICE.mt5_connected,
        "analysis_status": analysis_status,
        "analysis_started": analysis_started,
        "analysis_completed": analysis_completed,
        "analysis_phase": analysis_phase,
        "manual_analysis_progress": manual_progress,
        "model_available": session.model_available,
        "model_info": session.model_info,
        "last_training_info": session.last_training_info,
        "learning_trades": learning_trades,
        "results_source": results_source,
        "manual_analysis_results": manual_results,
        "startup_stage": manual.startup_stage if use_manual_state else runtime.startup_stage,
        "startup_progress": manual.startup_progress if use_manual_state else runtime.startup_progress,
        "startup_message": manual.startup_message if use_manual_state else runtime.startup_message,
        "running": session.running,
        "trading_enabled": session.trading_enabled,
    }

    if not visible:
        return {
            **common,
            "ready": False,
            "loading": runtime.startup_progress < 100,
            "candles_loaded": len(candles),
            "strategy": decision,
            "levels": decision.get("levels", []),
            "trade": session.open_trade,
            "trades": session.trade_history,
            "balance": session.balance,
            "equity": session.balance,
            "starting_balance": STARTING_BALANCE,
            "final_balance": session.balance,
            "blocked_trades": session.blocked_trades,
            "last_model_signal": session.last_model_signal,
        }

    current = visible[-1]
    total_return = (
        ((session.balance - STARTING_BALANCE) / STARTING_BALANCE) * 100.0
        if STARTING_BALANCE
        else 0.0
    )

    return {
        **common,
        "ready": True,
        "loading": False,
        "current_time": current.get("time"),
        "current_price": current.get("close"),
        "h1": candles,
        "line": [
            {"time": candle.get("time"), "value": candle.get("close")}
            for candle in candles
        ],
        "levels": decision.get("levels", []),
        "strategy": decision,
        "trade": session.open_trade,
        "trades": session.trade_history,
        "balance": session.balance,
        "equity": session.balance,
        "starting_balance": STARTING_BALANCE,
        "final_balance": session.balance,
        "total_return": total_return,
        "blocked_trades": session.blocked_trades,
        "last_model_signal": session.last_model_signal,
    }


async def send_snapshot(websocket: WebSocket, runtime: RuntimeContext) -> None:
    await websocket.send_text(_json_dumps(snapshot(runtime)))


async def broadcast_snapshot(runtime: RuntimeContext) -> None:
    """Broadcast only to WebSocket clients attached to the supplied runtime."""
    payload = _json_dumps(snapshot(runtime))
    dead_clients = []

    for websocket in runtime.get_websocket_clients():
        try:
            await websocket.send_text(payload)
        except Exception:
            dead_clients.append(websocket)

    for websocket in dead_clients:
        disconnect_customer_connection(websocket, runtime)


async def _load_runtime_chart(runtime: RuntimeContext) -> None:
    async with CHART_IO_LOCK:
        candles = await asyncio.to_thread(
            CHART_SERVICE.load_runtime_history,
            runtime,
        )

    runtime.startup_stage = "Chart ready"
    runtime.startup_progress = 100
    runtime.startup_message = (
        f"{runtime.market} {runtime.active_timeframe} ready. "
        f"{len(candles):,} candles loaded."
    )


# ============================================================================
# MANUAL LEARNING TRADE ANALYSIS
# ============================================================================

def _manual_hit_flags(
    direction: str,
    candle: Dict[str, Any],
    sl: float,
    tp: float,
) -> Tuple[bool, bool]:
    if direction == "BUY":
        return float(candle["low"]) <= sl, float(candle["high"]) >= tp
    return float(candle["high"]) >= sl, float(candle["low"]) <= tp


def _resolve_manual_collision(
    market: str,
    parent_candle: Dict[str, Any],
    parent_timeframe: str,
    direction: str,
    sl: float,
    tp: float,
    lower_cache: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, Optional[float], Optional[int], str]:
    """Resolve a same-candle TP/SL collision through M5 and then M1."""
    parent_start = int(parent_candle["time"])
    parent_duration = TIMEFRAME_SECONDS[parent_timeframe]
    parent_end = parent_start + parent_duration

    for lower_timeframe in ("M5", "M1"):
        lower_duration = TIMEFRAME_SECONDS[lower_timeframe]
        if lower_duration >= parent_duration:
            continue

        if lower_timeframe not in lower_cache:
            path = CHART_SERVICE.history_path(market, lower_timeframe)
            lower_cache[lower_timeframe] = CHART_SERVICE.load_local_history(path)

        relevant = [
            candle
            for candle in lower_cache[lower_timeframe]
            if (
                parent_start <= int(candle["time"])
                and int(candle["time"]) + lower_duration <= parent_end
            )
        ]
        relevant.sort(key=lambda candle: int(candle["time"]))

        if not relevant:
            continue

        collision_on_this_timeframe = False
        for candle in relevant:
            sl_hit, tp_hit = _manual_hit_flags(direction, candle, sl, tp)

            if sl_hit and tp_hit:
                if lower_timeframe == "M1":
                    return (
                        "AMBIGUOUS",
                        None,
                        int(candle["time"]),
                        "SL_AND_TP_SAME_M1_CANDLE",
                    )
                collision_on_this_timeframe = True
                break

            if sl_hit:
                return "LOSS", sl, int(candle["time"]), f"SL_RESOLVED_ON_{lower_timeframe}"

            if tp_hit:
                return "WIN", tp, int(candle["time"]), f"TP_RESOLVED_ON_{lower_timeframe}"

        if collision_on_this_timeframe:
            continue

    return "AMBIGUOUS", None, None, "SL_TP_ORDER_UNRESOLVED"


def _analyze_manual_setup(
    runtime: RuntimeContext,
    setup: Dict[str, Any],
    ordinal: int,
    current_balance: float,
    lower_cache: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    direction_raw = str(setup.get("direction", "")).strip().upper()
    direction = {"LONG": "BUY", "SHORT": "SELL"}.get(direction_raw, direction_raw)
    if direction not in {"BUY", "SELL"}:
        return {"valid": False, "error": "Invalid direction."}

    entry = setup_storage._manual_setup_value(setup, "entry", "entry")
    sl = setup_storage._manual_setup_value(setup, "sl", "sl")
    tp = setup_storage._manual_setup_value(setup, "tp", "tp")
    entry_time = setup_storage._manual_setup_time(setup)

    if entry is None or sl is None or tp is None or entry_time is None:
        return {"valid": False, "error": "Missing entry, SL, TP or entry time."}

    if direction == "BUY" and not (sl < entry < tp):
        return {"valid": False, "error": "Invalid BUY geometry."}
    if direction == "SELL" and not (tp < entry < sl):
        return {"valid": False, "error": "Invalid SELL geometry."}

    timeframe = str(setup.get("entry_timeframe", runtime.active_timeframe)).strip().upper()
    if timeframe not in TIMEFRAME_SECONDS:
        timeframe = runtime.active_timeframe

    history_path = CHART_SERVICE.history_path(runtime.market, timeframe)
    candles = CHART_SERVICE.load_local_history(history_path)
    if not candles:
        return {"valid": False, "error": f"No chart data for {runtime.market} {timeframe}."}

    duration = TIMEFRAME_SECONDS[timeframe]
    start_index = next(
        (
            index
            for index, candle in enumerate(candles)
            if int(candle["time"]) + duration > int(entry_time)
        ),
        None,
    )
    if start_index is None:
        return {"valid": False, "error": "Entry time is outside chart history."}

    risk_distance = abs(float(entry) - float(sl))
    reward_distance = abs(float(tp) - float(entry))
    if risk_distance <= 0:
        return {"valid": False, "error": "Invalid SL distance."}

    reward_r_multiple = reward_distance / risk_distance
    risk_amount = current_balance * (MANUAL_RISK_PERCENT / 100.0)

    result = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    exit_reason = "END_OF_AVAILABLE_HISTORY"

    for candle in candles[start_index:]:
        sl_hit, tp_hit = _manual_hit_flags(direction, candle, float(sl), float(tp))
        if sl_hit and tp_hit:
            result, exit_price, exit_time, exit_reason = _resolve_manual_collision(
                runtime.market,
                candle,
                timeframe,
                direction,
                float(sl),
                float(tp),
                lower_cache,
            )
            break
        if sl_hit:
            result = "LOSS"
            exit_price = float(sl)
            exit_time = int(candle["time"])
            exit_reason = "SL"
            break
        if tp_hit:
            result = "WIN"
            exit_price = float(tp)
            exit_time = int(candle["time"])
            exit_reason = "TP"
            break

    if result == "WIN":
        pnl = risk_amount * reward_r_multiple
    elif result == "LOSS":
        pnl = -risk_amount
    else:
        pnl = 0.0

    return {
        "valid": True,
        "trade_id": int(setup.get("id")) if str(setup.get("id", "")).isdigit() else ordinal,
        "source": "MANUAL_ANALYSIS",
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "opened_at": int(entry_time),
        "exit": exit_price,
        "closed_at": exit_time,
        "result": result,
        "exit_reason": exit_reason,
        "risk_amount": risk_amount,
        "reward_r_multiple": reward_r_multiple,
        "pnl": pnl,
        "training_eligible": result in {"WIN", "LOSS"},
        "setup_id": setup.get("id"),
        "setup_snapshot": copy.deepcopy(setup),
    }


async def manual_analysis_loop(runtime: RuntimeContext) -> None:
    """Analyze all saved manual Learning trades for one TradingBot."""
    manual = runtime.manual_analysis
    session = runtime.current_session

    try:
        setups = await asyncio.to_thread(
            setup_storage.load_learning_setups,
            runtime.identity,
        )
        if not setups:
            raise ValueError("No manually saved Learning trades.")

        manual.analysis_status = "manual_analysis"
        manual.analysis_phase = "manual_analysis"
        manual.analysis_started = True
        manual.analysis_completed = False
        manual.progress = 0
        manual.results_source = "manual"
        manual.results = {}
        manual.startup_stage = "Manual analysis"
        manual.startup_progress = 70
        manual.startup_message = f"Analyzing {len(setups)} Learning trades..."

        results: List[Dict[str, Any]] = []
        invalid_trades = 0
        ambiguous_trades = 0
        open_trades = 0
        balance = STARTING_BALANCE
        lower_cache: Dict[str, List[Dict[str, Any]]] = {}

        total = max(len(setups), 1)
        for index, setup in enumerate(setups, start=1):
            analysis = await asyncio.to_thread(
                _analyze_manual_setup,
                runtime,
                setup,
                index,
                balance,
                lower_cache,
            )

            if not analysis.get("valid"):
                invalid_trades += 1
            else:
                results.append(analysis)
                if analysis["result"] == "WIN":
                    balance += float(analysis["pnl"])
                elif analysis["result"] == "LOSS":
                    balance += float(analysis["pnl"])
                elif analysis["result"] == "AMBIGUOUS":
                    ambiguous_trades += 1
                else:
                    open_trades += 1

            manual.progress = int(index / total * 100)
            manual.startup_progress = 70 + int(index / total * 30)
            manual.startup_message = f"Analyzed {index}/{total} Learning trades."
            await broadcast_snapshot(runtime)
            await asyncio.sleep(0)

        wins = sum(1 for trade in results if trade.get("result") == "WIN")
        losses = sum(1 for trade in results if trade.get("result") == "LOSS")
        resolved = wins + losses
        total_return = (
            ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100.0
            if STARTING_BALANCE
            else 0.0
        )

        manual.results = {
            "symbol": runtime.market,
            "timeframe": runtime.active_timeframe,
            "starting_balance": STARTING_BALANCE,
            "final_balance": balance,
            "total_return": total_return,
            "trades": results,
            "invalid_trades": invalid_trades,
            "ambiguous_trades": ambiguous_trades,
            "open_trades": open_trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "winning_percentage": (wins / resolved * 100.0) if resolved else 0.0,
            "losing_percentage": (losses / resolved * 100.0) if resolved else 0.0,
        }
        manual.analysis_status = "completed"
        manual.analysis_phase = "manual_analysis_completed"
        manual.analysis_completed = True
        manual.progress = 100
        manual.startup_stage = "Manual analysis complete"
        manual.startup_progress = 100
        manual.startup_message = "MANUAL ANALYSIS complete."

    except asyncio.CancelledError:
        manual.analysis_status = "paused"
        manual.analysis_phase = "manual_analysis_paused"
        manual.analysis_completed = False
        manual.startup_message = "Manual analysis was cancelled."
        raise

    except Exception as error:
        manual.analysis_status = "error"
        manual.analysis_phase = "manual_analysis_error"
        manual.analysis_started = True
        manual.analysis_completed = False
        manual.startup_progress = 100
        manual.startup_message = str(error)
        LOGGER.exception("Manual analysis failed for %s", runtime.key)

    finally:
        await broadcast_snapshot(runtime)


# ============================================================================
# TRAINING AND TESTING TASKS
# ============================================================================

async def training_loop(runtime: RuntimeContext) -> None:
    session = runtime.current_session
    session.running = True
    session.analysis_status = "training"
    session.analysis_phase = "training"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "testing"
    runtime.manual_analysis.reset()
    runtime.startup_stage = "Training model"
    runtime.startup_progress = 70
    runtime.startup_message = f"Starting TRAINING for {runtime.market}..."
    await broadcast_snapshot(runtime)

    try:
        await asyncio.sleep(0)
        runtime.startup_stage = "Training model"
        runtime.startup_progress = 82
        runtime.startup_message = "Building identity-scoped AI training dataset..."
        await broadcast_snapshot(runtime)

        result = await asyncio.to_thread(_train_ai_for_runtime, runtime)
        if not load_ai_runtime(runtime):
            raise RuntimeError(
                runtime.ai_error or "The newly trained model could not be loaded."
            )

        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "training_completed"
        session.analysis_completed = True
        runtime.startup_stage = "Training complete"
        runtime.startup_progress = 100
        runtime.startup_message = f"TRAINING complete for {runtime.market}."
        session.last_training_info = copy.deepcopy(result.get("training", {}))
        session.model_info = copy.deepcopy(result.get("model", {}))

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "training_paused"
        session.analysis_completed = False
        runtime.startup_message = "Training was cancelled."
        raise

    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "training_error"
        session.analysis_completed = False
        runtime.startup_stage = "Training error"
        runtime.startup_progress = 100
        runtime.startup_message = str(error)
        LOGGER.exception("Training failed for %s", runtime.key)

    finally:
        await broadcast_snapshot(runtime)


async def testing_loop(runtime: RuntimeContext) -> None:
    session = runtime.current_session
    session.reset(STARTING_BALANCE)
    session.running = True
    session.analysis_status = "testing"
    session.analysis_phase = "testing"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "testing"
    runtime.manual_analysis.reset()

    if not load_ai_runtime(runtime):
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "testing_error"
        session.analysis_started = True
        session.analysis_completed = False
        runtime.startup_stage = "Testing unavailable"
        runtime.startup_progress = 100
        runtime.startup_message = (
            runtime.ai_error
            or f"No identity-compatible trained model exists for {runtime.market}."
        )
        await broadcast_snapshot(runtime)
        return

    candles = runtime.get_chart_data()
    if len(candles) < 2:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "testing_error"
        runtime.startup_stage = "Testing unavailable"
        runtime.startup_progress = 100
        runtime.startup_message = "At least two chart candles are required for testing."
        await broadcast_snapshot(runtime)
        return

    total = max(1, len(candles) - 1)
    last_reported = -1
    last_trade_count = 0

    runtime.startup_stage = "Testing model"
    runtime.startup_progress = 70
    runtime.startup_message = f"Starting TESTING for {runtime.market} {runtime.active_timeframe}..."
    await broadcast_snapshot(runtime)

    try:
        while session.current_index < len(candles) - 1:
            trading_engine.step_testing(
                runtime,
                ai_api=ai,
                config=TRADING_CONFIG,
            )

            progress = int(session.current_index / total * 100)
            trade_count = len(session.trade_history)
            if progress != last_reported or trade_count != last_trade_count:
                runtime.startup_stage = "Testing model"
                runtime.startup_progress = 70 + int(progress * 0.30)
                runtime.startup_message = (
                    f"Testing {session.current_index:,}/{total:,} "
                    f"| {progress}% | Trades: {trade_count}"
                )
                last_reported = progress
                last_trade_count = trade_count
                await broadcast_snapshot(runtime)

            if session.current_index % 50 == 0:
                await asyncio.sleep(0)

        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "testing_completed"
        session.analysis_completed = True
        runtime.startup_stage = "Testing complete"
        runtime.startup_progress = 100
        runtime.startup_message = f"TESTING complete for {runtime.market}."

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "testing_paused"
        session.analysis_completed = False
        runtime.startup_message = "Testing was cancelled."
        raise

    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "testing_error"
        session.analysis_completed = False
        runtime.startup_stage = "Testing error"
        runtime.startup_progress = 100
        runtime.startup_message = str(error)
        LOGGER.exception("Testing failed for %s", runtime.key)

    finally:
        await broadcast_snapshot(runtime)


# ============================================================================
# TASK MANAGEMENT
# ============================================================================

def _start_runtime_task(
    runtime: RuntimeContext,
    task_name: str,
    coroutine: Any,
) -> Tuple[asyncio.Task, bool]:
    existing = runtime.get_task(task_name)
    if existing is not None and not existing.done():
        return existing, False

    task = asyncio.create_task(coroutine)
    runtime.set_task(task_name, task)
    task.add_done_callback(
        lambda completed: runtime.clear_task(task_name, completed)
    )
    return task, True


async def _cancel_runtime_task(runtime: RuntimeContext, task_name: str) -> None:
    task = runtime.get_task(task_name)
    if task is None or task.done():
        runtime.clear_task(task_name)
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    runtime.clear_task(task_name, task)


# ============================================================================
# WEBSOCKET ACTION HANDLERS
# ============================================================================

async def _handle_market_switch(
    websocket: WebSocket,
    current_runtime: RuntimeContext,
    claims: Optional[VerifiedTradingBotClaims],
    new_market: str,
) -> RuntimeContext:
    market = str(new_market).strip().upper()
    if not market:
        raise ValueError("Market symbol must not be empty.")

    identity = _identity_for_market(
        market,
        claims,
        current_identity=current_runtime.identity,
    )
    new_runtime = _get_or_create_runtime(identity, current_runtime.active_timeframe)

    if new_runtime is current_runtime:
        return current_runtime

    disconnect_customer_connection(websocket, current_runtime)
    new_runtime.add_websocket_client(websocket)

    if not new_runtime.get_chart_data():
        try:
            await _load_runtime_chart(new_runtime)
        except Exception as error:
            new_runtime.startup_stage = "Chart error"
            new_runtime.startup_progress = 100
            new_runtime.startup_message = str(error)

    load_ai_runtime(new_runtime)
    return new_runtime


async def _handle_action(
    websocket: WebSocket,
    runtime: RuntimeContext,
    claims: Optional[VerifiedTradingBotClaims],
    message: Dict[str, Any],
) -> RuntimeContext:
    action = message.get("action")

    if action == "symbol":
        new_runtime = await _handle_market_switch(
            websocket,
            runtime,
            claims,
            message.get("symbol", ""),
        )
        await send_snapshot(websocket, new_runtime)
        return new_runtime

    if action == "timeframe":
        timeframe = str(message.get("timeframe", "")).strip().upper()
        async with CHART_IO_LOCK:
            await asyncio.to_thread(
                CHART_SERVICE.switch_timeframe,
                runtime,
                timeframe,
            )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "update_data":
        runtime.startup_stage = "Updating chart data"
        runtime.startup_progress = 5
        runtime.startup_message = "Synchronizing local chart archives with MT5..."
        await send_snapshot(websocket, runtime)

        async with CHART_IO_LOCK:
            result = await asyncio.to_thread(CHART_SERVICE.update_all_chart_data)
            if result.get("success"):
                await asyncio.to_thread(
                    CHART_SERVICE.load_runtime_history,
                    runtime,
                    runtime.active_timeframe,
                )

        runtime.startup_stage = "Chart ready" if result.get("success") else "Update unavailable"
        runtime.startup_progress = 100
        runtime.startup_message = result.get("message", "Chart update finished.")
        response = snapshot(runtime)
        response["update_result"] = result
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action == "manual_analysis":
        manual = runtime.manual_analysis
        existing = runtime.get_task("manual_analysis")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime

        manual.reset()
        manual.results_source = "manual"
        manual.analysis_status = "manual_analysis"
        manual.analysis_phase = "manual_analysis"
        manual.analysis_started = True
        manual.startup_stage = "Manual analysis"
        manual.startup_progress = 70
        manual.startup_message = f"Starting MANUAL ANALYSIS for {runtime.market}..."
        runtime.results_source = "manual"
        _start_runtime_task(
            runtime,
            "manual_analysis",
            manual_analysis_loop(runtime),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "prepare_ai_data":

        result = await asyncio.to_thread(
            _prepare_ai_data_for_runtime,
            runtime,
        )

        response = snapshot(runtime)

        response["ai_data_preparation"] = result

        await websocket.send_text(
            _json_dumps(response)
        )

        return runtime

    if action in {"training", "play"}:
        existing = runtime.get_task("training")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime

        await _cancel_runtime_task(runtime, "testing")
        session = runtime.current_session
        session.reset(STARTING_BALANCE)
        runtime.results_source = "testing"
        runtime.manual_analysis.reset()
        task, started = _start_runtime_task(
            runtime,
            "training",
            training_loop(runtime),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "testing":
        existing = runtime.get_task("testing")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime

        await _cancel_runtime_task(runtime, "training")
        runtime.results_source = "testing"
        runtime.manual_analysis.reset()
        _start_runtime_task(
            runtime,
            "testing",
            testing_loop(runtime),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "stop":
        await _cancel_runtime_task(runtime, "testing")
        await _cancel_runtime_task(runtime, "training")
        await _cancel_runtime_task(runtime, "manual_analysis")
        runtime.current_session.running = False
        runtime.current_session.trading_enabled = False
        runtime.startup_stage = "Stopped"
        runtime.startup_progress = 100
        runtime.startup_message = "TradingBot stopped."
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "learning_save":
        setup = message.get("setup")
        if not isinstance(setup, dict):
            raise ValueError("Invalid learning setup data.")

        saved_setup = await asyncio.to_thread(
            setup_storage.save_learning_setup,
            runtime.identity,
            setup,
        )
        response = snapshot(runtime)
        response["learning_action"] = "saved"
        response["learning_setup"] = saved_setup
        response["learning_trades"] = await asyncio.to_thread(
            setup_storage.load_learning_setups,
            runtime.identity,
        )
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action == "learning_delete":
        try:
            setup_id = int(message.get("setup_id"))
        except (TypeError, ValueError):
            raise ValueError("Invalid setup_id.")

        deleted = await asyncio.to_thread(
            setup_storage.delete_learning_setup,
            runtime.identity,
            setup_id,
        )
        response = snapshot(runtime)
        response["learning_action"] = "deleted"
        response["learning_setup_id"] = setup_id
        response["learning_deleted"] = deleted
        response["learning_trades"] = await asyncio.to_thread(
            setup_storage.load_learning_setups,
            runtime.identity,
        )
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action == "learning_load":
        learning_trades = await asyncio.to_thread(
            setup_storage.load_learning_setups,
            runtime.identity,
        )
        await websocket.send_text(_json_dumps({
            "learning_action": "loaded",
            "success": True,
            "symbol": runtime.market,
            "learning_trades": learning_trades,
        }))
        return runtime

    if action == "trade_detail":
        try:
            trade_id = int(message.get("trade_id"))
        except (TypeError, ValueError):
            await websocket.send_text(_json_dumps({
                "type": "trade_detail",
                "success": False,
                "error": "Invalid trade_id.",
            }))
            return runtime

        detail = trading_engine.get_trade_detail(runtime, trade_id)
        await websocket.send_text(_json_dumps({
            "type": "trade_detail",
            "success": detail is not None,
            "trade_id": trade_id,
            "symbol": runtime.market,
            "timeframe": runtime.active_timeframe,
            "trade": detail,
            "error": None if detail is not None else f"Trade #{trade_id} was not found.",
        }))
        return runtime

    if action == "viewport":
        try:
            from_time = int(message.get("from", 0))
            to_time = int(message.get("to", 0))
        except (TypeError, ValueError):
            raise ValueError("Invalid viewport range.")

        view_levels = strategy_engine.find_levels_for_view(
            runtime.get_chart_data(),
            from_time,
            to_time,
        )
        response = snapshot(runtime)
        response["levels"] = view_levels
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action == "stop_trading":
        trading_engine.stop_trading(runtime)
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "enable_trading":
        trading_engine.enable_trading(runtime)
        await send_snapshot(websocket, runtime)
        return runtime

    await websocket.send_text(_json_dumps({
        "success": False,
        "error": f"Unknown action: {action}",
    }))
    return runtime


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Starting Spectline backend. WEB_MODE=%s", WEB_MODE)

    if not WEB_MODE:
        try:
            connected = await asyncio.to_thread(
                CHART_SERVICE.initialize_mt5,
                DEFAULT_MARKET,
            )
            LOGGER.info("MT5 connected: %s", connected)
        except Exception as error:
            LOGGER.warning("MT5 initialization failed: %s", error)

    yield

    for runtime in RUNTIME_REGISTRY.list_runtimes():
        try:
            await cancel_runtime_tasks(runtime)
        except Exception:
            LOGGER.exception("Could not stop runtime tasks for %s", runtime.key)

    try:
        CHART_SERVICE.shutdown_mt5()
    except Exception:
        LOGGER.exception("Could not shut down MT5 cleanly.")

    LOGGER.info("Spectline backend stopped.")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(lifespan=lifespan)


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    return {
        "web_mode": WEB_MODE,
        "default_market": DEFAULT_MARKET,
        "supported_timeframes": sorted(TIMEFRAME_SECONDS),
    }


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "web_mode": WEB_MODE,
        "active_runtimes": len(RUNTIME_REGISTRY),
    }


@app.get("/")
@app.get("/frontend.html")
async def frontend_page():
    frontend_file = _find_frontend_file(FRONTEND_HTML_CANDIDATES)
    if frontend_file is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Frontend file was not found.",
                "expected": list(FRONTEND_HTML_CANDIDATES),
            },
        )
    return FileResponse(frontend_file)


@app.get("/frontend.js")
async def frontend_script():
    frontend_file = _find_frontend_file(FRONTEND_JS_CANDIDATES)
    if frontend_file is None:
        return JSONResponse(status_code=404, content={"error": "Frontend JavaScript file was not found."})
    return FileResponse(frontend_file, media_type="application/javascript")


@app.get("/frontend.css")
async def frontend_styles():
    frontend_file = _find_frontend_file(FRONTEND_CSS_CANDIDATES)
    if frontend_file is None:
        return JSONResponse(status_code=404, content={"error": "Frontend CSS file was not found."})
    return FileResponse(frontend_file, media_type="text/css")


@app.get("/images/{filename:path}")
async def frontend_image(filename: str):
    images_root = (BASE_DIR / "images").resolve()
    requested = (images_root / filename).resolve()
    if images_root not in requested.parents or not requested.is_file():
        return JSONResponse(status_code=404, content={"error": "Image was not found."})
    return FileResponse(requested)


# ============================================================================
# WEBSOCKET ENTRY POINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    claims: Optional[VerifiedTradingBotClaims] = None
    runtime: Optional[RuntimeContext] = None

    try:
        claims = await authenticate_websocket(websocket)
        identity = _identity_for_market(DEFAULT_MARKET, claims)
        runtime = _get_or_create_runtime(identity, DEFAULT_TIMEFRAME)

        await accept_customer_connection(websocket, runtime)

        if not runtime.get_chart_data():
            try:
                await _load_runtime_chart(runtime)
            except Exception as error:
                runtime.startup_stage = "Chart unavailable"
                runtime.startup_progress = 100
                runtime.startup_message = str(error)
                LOGGER.warning("Initial chart load failed for %s: %s", runtime.key, error)

        load_ai_runtime(runtime)
        await websocket.send_text(_json_dumps(startup_snapshot(runtime)))

        if runtime.get_chart_data():
            await send_snapshot(websocket, runtime)

        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_text(_json_dumps({
                    "success": False,
                    "error": "Invalid JSON message.",
                }))
                continue

            if not isinstance(message, dict):
                await websocket.send_text(_json_dumps({
                    "success": False,
                    "error": "WebSocket message must be a JSON object.",
                }))
                continue

            try:
                runtime = await _handle_action(
                    websocket,
                    runtime,
                    claims,
                    message,
                )
            except Exception as error:
                LOGGER.exception("WebSocket action failed: %s", message.get("action"))
                await websocket.send_text(_json_dumps({
                    "ready": False,
                    "loading": False,
                    "success": False,
                    "symbol": runtime.market,
                    "timeframe": runtime.active_timeframe,
                    "session_id": runtime.session_id,
                    "startup_stage": "Error",
                    "startup_progress": 100,
                    "startup_message": str(error),
                    "analysis_status": "error",
                }))

    except AuthenticationError as error:
        await reject_websocket(websocket, code=1008, reason=str(error))
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected.")
    except Exception as error:
        LOGGER.exception("WebSocket connection failed: %s", error)
        try:
            await reject_websocket(websocket, code=1011, reason="Backend connection failed.")
        except Exception:
            pass
    finally:
        if runtime is not None:
            disconnect_customer_connection(websocket, runtime)


# ============================================================================
# COMMAND LINE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host=os.getenv("SPECTLINE_HOST", "127.0.0.1"),
        port=int(os.getenv("SPECTLINE_PORT", "8000")),
        reload=False,
    )
