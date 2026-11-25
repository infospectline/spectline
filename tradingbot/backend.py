# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

try:
    import ai
except ImportError:
    try:
        from . import ai
    except ImportError:
        ai = None

try:
    from client import (
        BASE_DIR,
        ClientIdentity,
        ensure_client_directories,
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
DEFAULT_PROBABILITY_THRESHOLD = float(os.getenv("SPECTLINE_PROBABILITY_THRESHOLD", "0.50"))
STARTING_BALANCE = float(os.getenv("SPECTLINE_STARTING_BALANCE", "100000"))
MANUAL_RISK_PERCENT = float(os.getenv("SPECTLINE_MANUAL_RISK_PERCENT", "1.0"))
LEVEL_RECALC_INTERVAL = int(os.getenv("SPECTLINE_LEVEL_RECALC_INTERVAL", "20"))

RUNTIME_REGISTRY = RuntimeRegistry()
CHART_SERVICE = ChartService()
TRADING_CONFIG = trading_engine.TradingConfig()
MODEL_OPERATION_TASKS = ("training", "evaluation", "historical_test")

CHART_IO_LOCK = asyncio.Lock()
LEARNING_ARCHIVE_LOCKS: Dict[str, asyncio.Lock] = {}

FRONTEND_HTML_CANDIDATES = (
    "frontend.html",
)
FRONTEND_JS_CANDIDATES = (
    "frontend.js",
)
FRONTEND_MODULE_FILES = frozenset({
    "state.js",
    "chart.js",
    "results.js",
    "learning.js",
    "geometry.js",
    "gestures.js",
    "socket.js",
})
FRONTEND_CSS_CANDIDATES = (
    "frontend.css",
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

def _cleanup_runtime_directory(
    runtime: RuntimeContext,
    *,
    expected_model_version_id: str,
) -> Dict[str, Any]:
    """Remove RUNTIME only after a signed .pth reload succeeds with RUNTIME absent.

    PC and WEB runtime roots are market-scoped, not TradingBot-scoped. If another
    registered TradingBot shares the same root, cleanup is deliberately deferred
    so its in-progress or pending training/evaluation data is not destroyed.
    """
    if runtime.ai_model is None or runtime.ai_scaler is None:
        return {"success": False, "skipped": True, "reason": "No active model and scaler are loaded."}

    paths = ensure_client_directories(runtime.identity)
    raw_models_root = Path(paths.models_root).expanduser()
    raw_runtime_root = Path(paths.runtime_root).expanduser()
    if raw_runtime_root.is_symlink():
        return {"success": False, "skipped": True, "reason": "RUNTIME is a symbolic link."}

    models_root = raw_models_root.resolve()
    runtime_dir = raw_runtime_root.resolve()
    if runtime_dir.name != "RUNTIME" or runtime_dir.parent != models_root:
        return {"success": False, "skipped": True, "reason": "Resolved RUNTIME path is not the expected MODELS/<MARKET>/RUNTIME directory."}

    try:
        loaded_model = _verify_approved_model_metadata(
            runtime.ai_checkpoint,
            runtime.identity,
            expected_model_version_id=expected_model_version_id,
            expected_models_root=models_root,
        )
    except Exception as error:
        return {"success": False, "skipped": True, "reason": f"The active .pth approval could not be re-verified: {error}"}

    if str(runtime.ai_model_version or "") != str(expected_model_version_id):
        return {"success": False, "skipped": True, "reason": "The active in-memory model does not match the approved model version."}

    # One RUNTIME directory is shared by every identity that maps to the same
    # customer/market (WEB) or market (PC). Never erase it while another bot
    # context is registered against that same path.
    for other_runtime in RUNTIME_REGISTRY.list_runtimes():
        if other_runtime.identity.key == runtime.identity.key:
            continue
        try:
            other_paths = ensure_client_directories(other_runtime.identity)
            other_runtime_root = Path(other_paths.runtime_root).expanduser().resolve()
        except Exception:
            return {"success": False, "skipped": True, "reason": "Could not establish that other TradingBot runtime storage is separate."}
        if other_runtime_root == runtime_dir:
            return {
                "success": False,
                "skipped": True,
                "reason": "RUNTIME is shared with another registered TradingBot. Cleanup was deferred to protect its temporary data.",
            }

    for task_name in ("training", "evaluation", "historical_test", "testing", "manual_analysis"):
        task = runtime.get_task(task_name)
        if task is not None and not task.done():
            return {"success": False, "skipped": True, "reason": f"Task {task_name!r} is still active."}

    if not runtime_dir.exists():
        return {"success": True, "already_clean": True, "model_version_id": expected_model_version_id}

    identity_digest = hashlib.sha256(
        "|".join(runtime.identity.key).encode("utf-8")
    ).hexdigest()[:16]
    backup_dir = runtime_dir.with_name(f".RUNTIME.cleanup-{identity_digest}")
    if backup_dir.exists():
        return {"success": False, "skipped": True, "reason": "A previous RUNTIME cleanup recovery directory already exists."}

    try:
        # Hide the runtime directory first. A subsequent model load must succeed
        # while the conventional RUNTIME path does not exist.
        os.replace(runtime_dir, backup_dir)

        reloaded_model, reloaded_scaler, reloaded_metadata = _load_approved_model_without_runtime(
            runtime,
            expected_model_version_id,
            models_root,
        )

        if not isinstance(runtime.ai_market_candles, dict) or not runtime.ai_market_candles:
            raise RuntimeError("Market candles are unavailable to restore the active runtime after the .pth reload.")

        runtime.set_ai_artifacts(
            reloaded_model,
            reloaded_scaler,
            reloaded_metadata,
            runtime.ai_market_candles,
            model_version=reloaded_metadata.get("model_version_id"),
            scaler_version=reloaded_metadata.get("scaler_version_id"),
            dataset_version=reloaded_metadata.get("dataset_version_id"),
        )

        # Only now, after the .pth has loaded with RUNTIME absent, permanently
        # remove the hidden temporary directory.
        shutil.rmtree(backup_dir)

        return {
            "success": True,
            "already_clean": False,
            "model_version_id": expected_model_version_id,
            "model_path": str(loaded_model["model_path"]),
            "runtime_independent_reload_verified": True,
            "removed": True,
        }

    except Exception as error:
        # Restore all temporary files if the runtime-independent reload or
        # deletion failed. Never leave the application half-cleaned.
        try:
            if backup_dir.exists() and not runtime_dir.exists():
                os.replace(backup_dir, runtime_dir)
        except OSError as restore_error:
            LOGGER.exception(
                "[RUNTIME] Failed to restore %s after cleanup error: %s",
                backup_dir,
                restore_error,
            )
        LOGGER.exception(
            "[RUNTIME] Cleanup aborted for %s: %s",
            runtime.identity.key,
            error,
        )
        return {
            "success": False,
            "skipped": False,
            "reason": str(error),
            "runtime_restored": runtime_dir.exists(),
            "recovery_directory": str(backup_dir) if backup_dir.exists() else None,
        }


def _learning_archive_lock(
    runtime: RuntimeContext,
) -> asyncio.Lock:
    key = runtime.identity.key
    lock = LEARNING_ARCHIVE_LOCKS.get(key)

    if lock is None:
        lock = asyncio.Lock()
        LEARNING_ARCHIVE_LOCKS[key] = lock

    return lock


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


def _verify_approved_model_metadata(
    model_metadata: Dict[str, Any],
    identity: ClientIdentity,
    *,
    expected_model_version_id: Optional[str] = None,
    expected_models_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify that metadata describes one signed, self-contained approved .pth."""
    if not isinstance(model_metadata, dict):
        raise RuntimeError("The inference loader returned invalid model metadata.")

    _verify_checkpoint_identity(model_metadata, identity)

    if str(model_metadata.get("model_status", "")).upper() != "APPROVED":
        raise RuntimeError("Only a signed APPROVED model can be activated.")

    if str(model_metadata.get("market", "")).upper() != identity.market:
        raise RuntimeError("The approved model market does not match this TradingBot.")

    model_version_id = str(model_metadata.get("model_version_id", "")).strip()
    if not model_version_id:
        raise RuntimeError("The approved .pth does not contain a model_version_id.")
    if expected_model_version_id is not None and model_version_id != str(expected_model_version_id):
        raise RuntimeError("The loaded .pth is not the model version selected for approval.")

    approval = model_metadata.get("approval")
    if not isinstance(approval, dict):
        raise RuntimeError("The .pth does not expose its embedded approval certificate.")
    if approval.get("approval_signature_verified") is not True:
        raise RuntimeError("The embedded approval signature has not been verified.")
    if str(approval.get("evaluated_split", "")).lower() != "test":
        raise RuntimeError("The embedded approval is not based on the independent test split.")
    evaluation_hash = str(approval.get("evaluation_sha256", "")).lower()
    if len(evaluation_hash) != 64 or any(ch not in "0123456789abcdef" for ch in evaluation_hash):
        raise RuntimeError("The embedded evaluation report hash is missing or invalid.")
    if not str(approval.get("evaluation_version_id", "")).strip():
        raise RuntimeError("The embedded approval certificate has no evaluation identifier.")

    if expected_models_root is not None:
        models_root = Path(expected_models_root).expanduser().resolve()
    else:
        paths = ensure_client_directories(identity)
        models_root = Path(paths.models_root).expanduser().resolve()

    raw_model_path = model_metadata.get("model_path") or model_metadata.get("loaded_from")
    if not raw_model_path:
        raise RuntimeError("The approved model metadata has no .pth path.")

    raw_path = Path(str(raw_model_path)).expanduser()
    if raw_path.is_symlink():
        raise RuntimeError("Symbolic-link model paths are not accepted for production inference.")
    model_path = raw_path.resolve()

    if model_path.parent != models_root or model_path.suffix.lower() != ".pth":
        raise RuntimeError("The approved model must be a .pth file directly inside MODELS/<MARKET>.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Approved model file is missing: {model_path}")

    expected_file_hash = str(model_metadata.get("model_file_sha256", "")).lower()
    if len(expected_file_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_file_hash):
        raise RuntimeError("The inference loader did not provide a valid .pth SHA-256 digest.")

    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_file_hash = digest.hexdigest()
    if actual_file_hash != expected_file_hash:
        raise RuntimeError("The approved .pth file changed after inference verification.")

    feature_names = model_metadata.get("feature_names")
    if not isinstance(feature_names, (list, tuple)) or not feature_names:
        raise RuntimeError("The approved .pth has no explicit feature order.")
    normalized_features = [str(item).strip() for item in feature_names]
    if any(not item for item in normalized_features) or len(set(normalized_features)) != len(normalized_features):
        raise RuntimeError("The approved .pth feature order is invalid.")
    expected_schema_hash = hashlib.sha256("\n".join(normalized_features).encode("utf-8")).hexdigest()
    supplied_schema_hash = str(model_metadata.get("feature_schema_hash", "")).lower()
    if supplied_schema_hash != expected_schema_hash:
        raise RuntimeError("The approved .pth feature schema hash does not match its feature order.")

    return {
        "model_version_id": model_version_id,
        "model_path": model_path,
        "model_file_sha256": actual_file_hash,
        "feature_names": normalized_features,
        "feature_schema_hash": expected_schema_hash,
        "approval": copy.deepcopy(approval),
    }


def _load_approved_model_without_runtime(
    runtime: RuntimeContext,
    expected_model_version_id: str,
    models_root: Path,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Reload the exact approved .pth while its RUNTIME directory is absent."""
    if ai is None or not callable(getattr(ai, "load_model", None)):
        raise AIIntegrationPending("ai.py does not expose load_model().")

    identity_kwargs = _identity_kwargs_for_ai(ai.load_model, runtime.identity)
    try:
        parameters = inspect.signature(ai.load_model).parameters
    except (TypeError, ValueError) as error:
        raise AIIntegrationPending("Cannot inspect ai.load_model() signature.") from error
    if "model_version_id" not in parameters:
        raise AIIntegrationPending("ai.load_model() must support explicit model_version_id selection.")
    identity_kwargs["model_version_id"] = str(expected_model_version_id)

    model, scaler, metadata = ai.load_model(runtime.market, **identity_kwargs)
    if model is None or scaler is None:
        raise RuntimeError("The approved .pth did not restore both model and embedded scaler.")
    _verify_approved_model_metadata(
        metadata,
        runtime.identity,
        expected_model_version_id=expected_model_version_id,
        expected_models_root=models_root,
    )
    return model, scaler, metadata


def load_ai_runtime(
    runtime: RuntimeContext,
    model_version_id: Optional[str] = None,
) -> bool:
    """Load a verified approved model without discarding a working prior model."""
    if ai is None:
        runtime.ai_error = "The ai.py module could not be imported."
        if runtime.ai_model is None:
            runtime.clear_ai_artifacts(runtime.ai_error)
        return False

    previous_model = runtime.ai_model
    session = runtime.current_session

    try:
        identity_kwargs = _identity_kwargs_for_ai(ai.load_model, runtime.identity)
        if model_version_id is not None:
            try:
                load_parameters = inspect.signature(ai.load_model).parameters
            except (TypeError, ValueError) as error:
                raise AIIntegrationPending("Cannot inspect ai.load_model() signature.") from error
            if "model_version_id" not in load_parameters:
                raise AIIntegrationPending(
                    "ai.load_model() must support explicit model_version_id selection."
                )
            identity_kwargs["model_version_id"] = str(model_version_id)

        model, scaler, model_metadata = ai.load_model(
            runtime.market,
            **identity_kwargs,
        )
        verified_model = _verify_approved_model_metadata(
            model_metadata,
            runtime.identity,
            expected_model_version_id=model_version_id,
        )

        if str(model_metadata.get("model_target_schema", "")).upper() not in {
            "TP_BEFORE_SL_BINARY",
            "TP_BEFORE_SL",
        }:
            raise RuntimeError("The approved model does not use the TP-before-SL binary target.")

        market_candles = ai.load_all_market_candles(
            runtime.market,
            identity=runtime.identity,
        )
        if not isinstance(market_candles, dict) or not any(market_candles.values()):
            timeframe_list = list(
                getattr(ai, "SUPPORTED_TIMEFRAMES", (runtime.active_timeframe,))
            )
            market_candles = ai.load_candles_for_timeframes(
                runtime.market,
                timeframe_list,
                identity=runtime.identity,
            )
        if not isinstance(market_candles, dict) or not any(market_candles.values()):
            raise RuntimeError(
                f"No closed market candle history is available for {runtime.market}."
            )

        runtime.set_ai_artifacts(
            model,
            scaler,
            model_metadata,
            market_candles,
            model_version=model_metadata.get("model_version_id"),
            scaler_version=model_metadata.get("scaler_version_id"),
            dataset_version=model_metadata.get("dataset_version_id"),
        )

        model_info = {
            **runtime.identity.as_dict(),
            "model_status": model_metadata.get("model_status"),
            "model_version_id": model_metadata.get("model_version_id"),
            "dataset_version_id": model_metadata.get("dataset_version_id"),
            "scaler_version_id": model_metadata.get("scaler_version_id"),
            "model_target_schema": model_metadata.get("model_target_schema"),
            "feature_count": len(model_metadata.get("feature_names", ())),
            "feature_schema_hash": model_metadata.get("feature_schema_hash"),
            "trained_at": model_metadata.get("trained_at"),
            "approval": copy.deepcopy(model_metadata.get("approval", {})),
            "calibration_status": model_metadata.get("calibration_status", "NOT_CALIBRATED"),
            "model_path": str(verified_model["model_path"]),
            "model_file_sha256": verified_model["model_file_sha256"],
            "feature_names": list(verified_model["feature_names"]),
            "feature_schema_hash": verified_model["feature_schema_hash"],
        }
        for timeframe_session in runtime.timeframe_sessions.values():
            timeframe_session.model_available = True
            timeframe_session.model_info = copy.deepcopy(model_info)
        runtime.ai_error = None
        _invalidate_strategy_cache(runtime)
        return True

    except FileNotFoundError as error:
        runtime.ai_error = str(error)
        if previous_model is None:
            runtime.clear_ai_artifacts(str(error))
        else:
            LOGGER.warning(
                "Could not replace the active approved model for %s; keeping version %s: %s",
                runtime.identity.key,
                runtime.ai_model_version,
                error,
            )
        return False
    except Exception as error:
        runtime.ai_error = str(error)
        if previous_model is None:
            runtime.clear_ai_artifacts(str(error))
        else:
            LOGGER.warning(
                "Could not replace the active approved model for %s; keeping version %s: %s",
                runtime.identity.key,
                runtime.ai_model_version,
                error,
            )
        return False

def _train_ai_for_runtime(
    runtime: RuntimeContext,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Call ai.py with identity and optional training progress reporting."""
    if ai is None:
        raise AIIntegrationPending("The ai.py module could not be imported.")

    identity_kwargs = _identity_kwargs_for_ai(ai.train_market, runtime.identity)

    if progress_callback is not None:
        try:
            parameters = inspect.signature(ai.train_market).parameters
        except (TypeError, ValueError):
            parameters = {}

        if "progress_callback" in parameters:
            identity_kwargs["progress_callback"] = progress_callback

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
# STRATEGY DECISION CACHE
# ============================================================================

def _invalidate_strategy_cache(runtime: RuntimeContext) -> None:
    """Invalidate the cached decision after any input or model change."""
    for session in runtime.timeframe_sessions.values():
        session.__dict__.pop("_spectline_strategy_cache_key", None)
        session.__dict__.pop("_spectline_strategy_cache_value", None)


def _strategy_cache_key(runtime: RuntimeContext, visible: Sequence[Dict[str, Any]]) -> str:
    """Build a stable key from the latest closed candle, candidate and model."""
    latest = visible[-1] if visible else {}
    candidate = runtime.get_pending_setup()
    payload = {
        "identity": runtime.identity.as_dict(),
        "timeframe": runtime.active_timeframe,
        "latest_candle": latest,
        "candidate": candidate,
        "model_version_id": runtime.ai_model_version,
        "scaler_version_id": runtime.ai_scaler_version,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluate_strategy_cached(
    runtime: RuntimeContext,
    visible: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a strategy once per unchanged candle, candidate and model version."""
    session = runtime.current_session
    cache_key = _strategy_cache_key(runtime, visible)
    previous_key = session.__dict__.get("_spectline_strategy_cache_key")
    previous_value = session.__dict__.get("_spectline_strategy_cache_value")
    if previous_key == cache_key and isinstance(previous_value, dict):
        return copy.deepcopy(previous_value)

    decision = strategy_engine.evaluate_strategy(
        runtime,
        visible,
        ai_api=ai,
        level_recalc_interval=LEVEL_RECALC_INTERVAL,
        probability_threshold=DEFAULT_PROBABILITY_THRESHOLD,
        persist_prediction=True,
    )
    if not isinstance(decision, dict):
        raise TypeError("The strategy engine returned an invalid decision.")

    prediction = decision.get("prediction")
    if isinstance(prediction, dict):
        try:
            prediction_time = prediction.get("predicted_at")
            if prediction_time is not None and (
                session.last_prediction_time is None
                or int(prediction_time) > int(session.last_prediction_time)
            ):
                session.record_prediction(prediction)
        except (TypeError, ValueError):
            LOGGER.warning("A strategy prediction could not be attached to the runtime history.")

    session.__dict__["_spectline_strategy_cache_key"] = cache_key
    session.__dict__["_spectline_strategy_cache_value"] = copy.deepcopy(decision)
    return decision


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
        "training_progress": runtime.training_progress,
        "training_stage": runtime.training_stage,
        "training_message": runtime.training_message,
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
        "last_evaluation_info": getattr(session, "last_evaluation_info", {}),
        "last_historical_test_info": getattr(session, "last_historical_test_info", {}),
        "active_model_version": runtime.ai_model_version,
        "model_status": runtime.ai_status,
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
            decision = _evaluate_strategy_cached(runtime, visible)
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
        "training_progress": runtime.training_progress,
        "training_stage": runtime.training_stage,
        "training_message": runtime.training_message,
        "model_available": session.model_available,
        "model_info": session.model_info,
        "last_training_info": session.last_training_info,
        "last_evaluation_info": getattr(session, "last_evaluation_info", {}),
        "last_historical_test_info": getattr(session, "last_historical_test_info", {}),
        "active_model_version": runtime.ai_model_version,
        "model_status": runtime.ai_status,
        "pending_setup": runtime.get_pending_setup(),
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

    _invalidate_strategy_cache(runtime)
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
    ambiguous_end_time = parent_end

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
        collision_end_time: Optional[int] = None

        for candle in relevant:
            sl_hit, tp_hit = _manual_hit_flags(direction, candle, sl, tp)

            if sl_hit and tp_hit:
                candle_end_time = int(candle["time"]) + lower_duration

                if lower_timeframe == "M1":
                    return (
                        "AMBIGUOUS",
                        None,
                        candle_end_time,
                        "SL_AND_TP_SAME_M1_CANDLE",
                    )

                collision_on_this_timeframe = True
                collision_end_time = candle_end_time
                break

            if sl_hit:
                return (
                    "LOSS",
                    sl,
                    int(candle["time"]) + lower_duration,
                    f"SL_RESOLVED_ON_{lower_timeframe}",
                )

            if tp_hit:
                return (
                    "WIN",
                    tp,
                    int(candle["time"]) + lower_duration,
                    f"TP_RESOLVED_ON_{lower_timeframe}",
                )

        if collision_on_this_timeframe:
            ambiguous_end_time = collision_end_time or parent_end
            continue
        else:
            ambiguous_end_time = parent_end

    return (
        "AMBIGUOUS",
        None,
        ambiguous_end_time,
        "SL_TP_ORDER_UNRESOLVED",
    )


def _analyze_manual_setup(
    runtime: RuntimeContext,
    setup: Dict[str, Any],
    current_balance: float,
    lower_cache: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    raw_id = setup.get("id")

    if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id <= 0:
        return {
            "valid": False,
            "error": "Invalid or missing trade id.",
        }

    current_id = raw_id
    
    direction_raw = str(setup.get("direction", "")).strip().upper()
    direction = {"LONG": "BUY", "SHORT": "SELL"}.get(direction_raw, direction_raw)
    if direction not in {"BUY", "SELL"}:
        return {"valid": False, "error": "Invalid direction."}

    
    entry = setup_storage._manual_setup_value(setup, "entry", "entry")
    sl = setup_storage._manual_setup_value(setup, "sl", "sl")
    tp = setup_storage._manual_setup_value(setup, "tp", "tp")

    geometry = setup.get("trade_geometry")
    entry_point = geometry.get("entry") if isinstance(geometry, dict) else None

    entry_time = None

    if isinstance(entry_point, dict):
        for field_name in ("time", "timestamp", "candle_time"):
            entry_time = setup_storage._manual_parse_timestamp(
                entry_point.get(field_name)
            )
            if entry_time is not None:
                break

    if entry_time is None:
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
            if int(candle["time"]) >= int(entry_time)
        ),
        None,
    )

    if start_index is None:
        return {"valid": False, "error": "Entry time is outside chart history."}

    entry_candle_time = int(candles[start_index]["time"])

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
            exit_time = int(candle["time"]) + duration
            exit_reason = "SL"
            break
        if tp_hit:
            result = "WIN"
            exit_price = float(tp)
            exit_time = int(candle["time"]) + duration
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
        "id": current_id,
        "source": "MANUAL_ANALYSIS",
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),       
        "opened_at": entry_candle_time,
        "entry_candle_time": entry_candle_time,
        "exit": exit_price,
        "end_time": exit_time,
        "actual_outcome": (
            "TP" if result == "WIN"
            else "SL" if result == "LOSS"
            else "AMBIGUOUS" if result == "AMBIGUOUS"
            else None
        ),
        "result": result,
        "exit_reason": exit_reason,
        "risk_amount": risk_amount,
        "reward_r_multiple": reward_r_multiple,
        "pnl": pnl,
        "training_eligible": result in {"WIN", "LOSS"},
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
                balance,
                lower_cache,
            )

            if not analysis.get("valid"):
                invalid_trades += 1
            
            else:
                analysis_id = int(analysis["id"])
                update_record = {
                    "id": analysis_id,
                    "entry_candle_time": int(analysis["entry_candle_time"]),
                    "end_time": analysis["end_time"],
                    "actual_outcome": analysis["actual_outcome"],
                }

                async with _learning_archive_lock(runtime):
                    archived_setups = await asyncio.to_thread(
                        setup_storage.load_learning_setups,
                        runtime.identity,
                    )

                    setup_exists = any(
                        isinstance(item, dict)
                        and setup_storage._parse_setup_id(item.get("id")) == analysis_id
                        for item in archived_setups
                    )

                    if not setup_exists:
                        raise RuntimeError(
                            f"Learning setup {analysis_id} no longer exists; "
                            "refusing to create a replacement record."
                        )

                    await asyncio.to_thread(
                        setup_storage.save_learning_setup,
                        runtime.identity,
                        update_record,
                    )

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

def _compact_metrics_report(report: Any) -> Dict[str, Any]:
    """Keep frontend reports concise while preserving the complete report internally."""
    if not isinstance(report, dict):
        return {}
    allowed = (
        "schema_version",
        "evaluation_id",
        "testing_run_id",
        "run_id",
        "model_version_id",
        "dataset_version_id",
        "scaler_version_id",
        "split",
        "sample_count",
        "trade_count",
        "metrics",
        "approval_gate",
        "calibration_status",
        "model_approval",
        "approval_blocked",
        "approval_blocking_reasons",
        "warnings",
        "persistence",
    )
    result = {key: copy.deepcopy(report[key]) for key in allowed if key in report}
    result["status"] = "COMPLETED"
    return result


def _resolve_model_version_for_runtime(
    runtime: RuntimeContext,
    requested_version_id: Optional[Any] = None,
) -> str:
    """Resolve an explicit, newly trained or active approved model version."""
    if requested_version_id is not None and str(requested_version_id).strip():
        return str(requested_version_id).strip()

    session = runtime.current_session
    training_info = session.last_training_info if isinstance(session.last_training_info, dict) else {}
    model_section = training_info.get("model", {})
    if isinstance(model_section, dict) and model_section.get("model_version_id"):
        return str(model_section["model_version_id"])
    if training_info.get("model_version_id"):
        return str(training_info["model_version_id"])

    evaluation_info = getattr(session, "last_evaluation_info", {})
    if isinstance(evaluation_info, dict) and evaluation_info.get("model_version_id"):
        return str(evaluation_info["model_version_id"])

    if runtime.ai_model_version:
        return str(runtime.ai_model_version)

    model_info = session.model_info if isinstance(session.model_info, dict) else {}
    if model_info.get("model_version_id"):
        return str(model_info["model_version_id"])

    raise RuntimeError(
        "No model version is available. Train a model or supply model_version_id."
    )


async def training_loop(runtime: RuntimeContext) -> None:
    """Create a new DRAFT candidate without replacing the active approved model."""
    session = runtime.current_session
    session.running = True
    session.analysis_status = "training"
    session.analysis_phase = "training"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "model_training"
    runtime.manual_analysis.reset()
    runtime.set_training_progress(
        0,
        stage="Preparing dataset",
        message="Preparing training data...",
    )
    runtime.set_startup_progress(
        "Training model",
        70,
        f"Starting TP-before-SL training for {runtime.market}...",
    )
    session.last_evaluation_info = {}
    session.last_historical_test_info = {}
    session.__dict__.pop("_spectline_last_evaluation_report", None)
    session.__dict__.pop("_spectline_last_testing_report", None)
    await broadcast_snapshot(runtime)

    try:
        await asyncio.sleep(0)
        runtime.set_startup_progress(
            "Training model",
            82,
            "Building point-in-time dataset and fitting the train-only scaler...",
        )
        await broadcast_snapshot(runtime)
       
        loop = asyncio.get_running_loop()

        async def publish_training_progress(
            progress: int,
            stage: str,
            message: str,
        ) -> None:
            runtime.set_training_progress(
                progress,
                stage=stage,
                message=message,
            )
            await broadcast_snapshot(runtime)

        def report_training_progress(
            progress: int,
            stage: str = "Training",
            message: str = "",
        ) -> None:
            future = asyncio.run_coroutine_threadsafe(
                publish_training_progress(progress, stage, message),
                loop,
            )

            try:
                future.result()
            except Exception:
                LOGGER.exception(
                    "[TRAIN] Failed to publish progress update."
                )
           
        result = await asyncio.to_thread(
            _train_ai_for_runtime,
            runtime,
            report_training_progress,
        )

        if not isinstance(result, dict):
            raise RuntimeError("The training pipeline returned an invalid result.")

        runtime.set_training_progress(
            99,
            stage="Finalizing",
            message="Validating and finalizing the trained model...",
        )
        await broadcast_snapshot(runtime)

        print(
            f"[TRAIN][DEBUG] Pipeline result type: {type(result).__name__}",
            flush=True,
        )

        model_section = result.get("model", {})
        training_section = result.get("training", {})

        if not isinstance(model_section, dict) or not model_section.get("model_version_id"):
            raise RuntimeError("Training completed without registering a model version.")

        if str(model_section.get("status", "")).upper() != "DRAFT":
            raise RuntimeError("A newly trained model must remain in DRAFT state.")

        # Final candidates must be standalone .pth files directly in the market folder.
        candidate_identity = model_section.get("identity")
        if candidate_identity != runtime.identity.as_dict():
            raise RuntimeError("The trained candidate belongs to a different TradingBot identity.")
        model_paths = ensure_client_directories(runtime.identity)
        candidate_path_value = model_section.get("model_path")
        if not candidate_path_value:
            raise RuntimeError("Training completed without returning the final .pth path.")
        candidate_path = Path(str(candidate_path_value)).expanduser().resolve()
        expected_models_root = Path(model_paths.models_root).expanduser().resolve()
        if candidate_path.parent != expected_models_root or candidate_path.suffix.lower() != ".pth":
            raise RuntimeError("The final candidate must be a .pth file directly inside MODELS/<MARKET>.")
        if not candidate_path.is_file():
            raise FileNotFoundError(f"The final candidate .pth does not exist: {candidate_path}")

        if not isinstance(training_section, dict) or str(
            training_section.get("model_target_schema", "")
        ).upper() != "TP_BEFORE_SL_BINARY":
            raise RuntimeError("The training result does not use the TP-before-SL binary objective.")

        session.last_training_info = copy.deepcopy(result)
        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "training_completed_draft"
        session.analysis_completed = True
        runtime.set_training_progress(
            100,
            stage="Completed",
            message="Training completed successfully.",
        )
        runtime.set_startup_progress(
            "Draft model created",
            100,
            "Training completed. The new model is DRAFT and must pass evaluation and historical testing before approval.",
        )
        _invalidate_strategy_cache(runtime)

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "training_paused"
        session.analysis_completed = False
        runtime.set_startup_progress("Training paused", 100, "Training was cancelled.")

        runtime.set_training_progress(
            runtime.training_progress,
            stage="Paused",
            message="Training was cancelled.",
        )
        raise

    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "training_error"
        session.analysis_completed = False
        runtime.set_startup_progress("Training error", 100, str(error))

        runtime.set_training_progress(
            runtime.training_progress,
            stage="Failed",
            message=str(error),
        )
        LOGGER.exception("Training failed for %s", runtime.key)

    finally:
        await broadcast_snapshot(runtime)


async def evaluation_loop(
    runtime: RuntimeContext,
    model_version_id: Optional[str] = None,
) -> None:
    """Evaluate one immutable DRAFT model on its linked held-out test split."""
    session = runtime.current_session
    session.running = True
    session.analysis_status = "evaluation"
    session.analysis_phase = "evaluation"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "model_evaluation"
    runtime.set_startup_progress("Evaluating model", 70, "Preparing independent test evaluation...")
    await broadcast_snapshot(runtime)

    try:
        selected_version = _resolve_model_version_for_runtime(runtime, model_version_id)
        if ai is None or not callable(getattr(ai, "evaluate_model_version", None)):
            raise AIIntegrationPending("ai.py does not expose evaluate_model_version().")
        report = await asyncio.to_thread(
            ai.evaluate_model_version,
            runtime.market,
            selected_version,
            identity=runtime.identity,
            split="test",
            approve_if_passed=False,
        )
        if not isinstance(report, dict):
            raise RuntimeError("Evaluation returned an invalid report.")
        report_model_id = str(report.get("model_version_id", ""))
        if report_model_id != selected_version:
            raise RuntimeError("Evaluation report references a different model version.")
        if report.get("identity") != runtime.identity.as_dict():
            raise RuntimeError("Evaluation report belongs to a different TradingBot identity.")
        evaluation_artifact = report.get("evaluation_artifact")
        if (
            not isinstance(evaluation_artifact, dict)
            or evaluation_artifact.get("storage") != "RUNTIME"
            or evaluation_artifact.get("verified") is not True
        ):
            raise RuntimeError("Evaluation report was not persisted and verified in RUNTIME.")
        evaluation_hash = str(evaluation_artifact.get("sha256", "")).lower()
        if len(evaluation_hash) != 64 or any(ch not in "0123456789abcdef" for ch in evaluation_hash):
            raise RuntimeError("Evaluation report has an invalid SHA-256 digest.")
        report_model_hash = str(report.get("model_artifact_sha256", "")).lower()
        if len(report_model_hash) != 64 or any(ch not in "0123456789abcdef" for ch in report_model_hash):
            raise RuntimeError("Evaluation report has no valid model artifact SHA-256 digest.")
        session.__dict__["_spectline_last_evaluation_report"] = copy.deepcopy(report)
        session.last_evaluation_info = _compact_metrics_report(report)
        session.last_evaluation_info["model_version_id"] = selected_version
        gate = report.get("approval_gate", {})
        passed = isinstance(gate, dict) and gate.get("passed") is True and gate.get("approval_eligible") is True
        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "evaluation_passed" if passed else "evaluation_blocked"
        session.analysis_completed = True
        runtime.set_startup_progress(
            "Evaluation complete",
            100,
            "Held-out evaluation passed. Run historical testing, then explicitly approve the model."
            if passed
            else "Held-out evaluation did not pass the approval gate. The model remains DRAFT.",
        )

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "evaluation_paused"
        session.analysis_completed = False
        runtime.set_startup_progress("Evaluation paused", 100, "Evaluation was cancelled.")
        raise
    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "evaluation_error"
        session.analysis_completed = False
        runtime.set_startup_progress("Evaluation error", 100, str(error))
        LOGGER.exception("Evaluation failed for %s", runtime.key)
    finally:
        await broadcast_snapshot(runtime)


async def testing_loop(runtime: RuntimeContext) -> None:
    """Replay the active chart candle by candle using the existing TradingBot flow."""
    session = runtime.current_session
    session.reset(STARTING_BALANCE)
    session.running = True
    session.analysis_status = "testing"
    session.analysis_phase = "paper_replay"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "testing"
    runtime.manual_analysis.reset()

    loaded = await asyncio.to_thread(load_ai_runtime, runtime)
    if not loaded:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "paper_replay_error"
        session.analysis_completed = False
        runtime.set_startup_progress(
            "Testing unavailable",
            100,
            runtime.ai_error or f"No approved TP-before-SL model exists for {runtime.market}.",
        )
        await broadcast_snapshot(runtime)
        return

    candles = runtime.get_chart_data()
    if len(candles) < 2:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "paper_replay_error"
        session.analysis_completed = False
        runtime.set_startup_progress(
            "Testing unavailable",
            100,
            "At least two chart candles are required for paper replay.",
        )
        await broadcast_snapshot(runtime)
        return

    total = max(1, len(candles) - 1)
    last_reported = -1
    last_trade_count = 0
    runtime.set_startup_progress(
        "Paper replay",
        70,
        f"Starting candle-by-candle replay for {runtime.market} {runtime.active_timeframe}...",
    )
    await broadcast_snapshot(runtime)

    try:
        while session.current_index < len(candles) - 1:
            await asyncio.to_thread(
                trading_engine.step_testing,
                runtime,
                ai_api=ai,
                config=TRADING_CONFIG,
            )
            progress = int(session.current_index / total * 100)
            trade_count = len(session.trade_history)
            if progress != last_reported or trade_count != last_trade_count:
                runtime.set_startup_progress(
                    "Paper replay",
                    70 + int(progress * 0.30),
                    f"Replayed {session.current_index:,}/{total:,} candles | {progress}% | Trades: {trade_count}",
                )
                last_reported = progress
                last_trade_count = trade_count
                await broadcast_snapshot(runtime)

            if session.current_index % 50 == 0:
                await asyncio.sleep(0)

        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "paper_replay_completed"
        session.analysis_completed = True
        runtime.set_startup_progress(
            "Paper replay complete",
            100,
            f"Candle-by-candle replay completed for {runtime.market} {runtime.active_timeframe}.",
        )

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "paper_replay_paused"
        session.analysis_completed = False
        runtime.set_startup_progress("Paper replay paused", 100, "Paper replay was cancelled.")
        raise
    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "paper_replay_error"
        session.analysis_completed = False
        runtime.set_startup_progress("Paper replay error", 100, str(error))
        LOGGER.exception("Paper replay failed for %s", runtime.key)
    finally:
        await broadcast_snapshot(runtime)


async def historical_testing_loop(
    runtime: RuntimeContext,
    model_version_id: Optional[str] = None,
) -> None:
    """Run the immutable historical test for one explicit model version."""
    session = runtime.current_session
    session.running = True
    session.analysis_status = "testing"
    session.analysis_phase = "historical_testing"
    session.analysis_started = True
    session.analysis_completed = False
    runtime.results_source = "historical_model_test"
    runtime.set_startup_progress("Historical model test", 70, "Loading the selected immutable model and test split...")
    await broadcast_snapshot(runtime)

    try:
        selected_version = _resolve_model_version_for_runtime(runtime, model_version_id)
        if ai is None or not callable(getattr(ai, "run_model_version_test", None)):
            raise AIIntegrationPending("ai.py does not expose run_model_version_test().")
        report = await asyncio.to_thread(
            ai.run_model_version_test,
            runtime.market,
            selected_version,
            identity=runtime.identity,
            persist=True,
        )
        if not isinstance(report, dict):
            raise RuntimeError("Historical testing returned an invalid report.")
        if str(report.get("model_version_id", "")) != selected_version:
            raise RuntimeError("Historical test report references a different model version.")
        if report.get("identity") != runtime.identity.as_dict():
            raise RuntimeError("Historical test report belongs to a different TradingBot identity.")
        if report.get("split") != "test" or report.get("testing_mode") != "FROZEN_MODEL_CHRONOLOGICAL_FORWARD_TEST":
            raise RuntimeError("Historical test report is not an independent chronological test.")
        persistence = report.get("persistence", {})
        if not isinstance(persistence, dict) or persistence.get("persisted") is not True:
            raise RuntimeError("Historical test report was not persisted.")
        persisted_report_path = Path(str(persistence.get("report_path", ""))).expanduser().resolve()
        runtime_root = Path(ensure_client_directories(runtime.identity).runtime_root).expanduser().resolve()
        expected_test_root = (runtime_root / "HISTORICAL_TESTS").resolve()
        try:
            persisted_report_path.relative_to(expected_test_root)
        except ValueError as error:
            raise RuntimeError("Historical test report is outside RUNTIME/HISTORICAL_TESTS.") from error
        if not persisted_report_path.is_file():
            raise FileNotFoundError("Persisted historical test report is missing.")
        session.__dict__["_spectline_last_testing_report"] = copy.deepcopy(report)
        session.last_historical_test_info = _compact_metrics_report(report)
        session.last_historical_test_info["model_version_id"] = selected_version
        session.running = False
        session.analysis_status = "completed"
        session.analysis_phase = "historical_testing_completed"
        session.analysis_completed = True
        runtime.set_startup_progress(
            "Historical test complete",
            100,
            "Historical test report was stored. The model remains DRAFT until the separate approval action.",
        )

    except asyncio.CancelledError:
        session.running = False
        session.analysis_status = "paused"
        session.analysis_phase = "historical_testing_paused"
        session.analysis_completed = False
        runtime.set_startup_progress("Historical test paused", 100, "Historical testing was cancelled.")
        raise
    except Exception as error:
        session.running = False
        session.analysis_status = "error"
        session.analysis_phase = "historical_testing_error"
        session.analysis_completed = False
        runtime.set_startup_progress("Historical test error", 100, str(error))
        LOGGER.exception("Historical testing failed for %s", runtime.key)
    finally:
        await broadcast_snapshot(runtime)


def approve_model_for_runtime(
    runtime: RuntimeContext,
    model_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve one tested DRAFT and activate that exact signed .pth file."""
    if ai is None or not callable(getattr(ai, "approve_evaluated_model", None)):
        raise AIIntegrationPending("ai.py does not expose approve_evaluated_model().")

    session = runtime.current_session
    selected_version = _resolve_model_version_for_runtime(runtime, model_version_id)
    evaluation_report = session.__dict__.get("_spectline_last_evaluation_report")
    testing_report = session.__dict__.get("_spectline_last_testing_report")
    if not isinstance(evaluation_report, dict):
        raise RuntimeError("Run held-out model evaluation before approval.")
    if not isinstance(testing_report, dict):
        raise RuntimeError("Run historical model testing before approval.")
    if str(evaluation_report.get("model_version_id", "")) != selected_version:
        raise RuntimeError("The saved evaluation belongs to a different model version.")
    if str(testing_report.get("model_version_id", "")) != selected_version:
        raise RuntimeError("The saved historical test belongs to a different model version.")
    if evaluation_report.get("identity") != runtime.identity.as_dict():
        raise RuntimeError("The evaluation report belongs to a different TradingBot identity.")
    if testing_report.get("identity") != runtime.identity.as_dict():
        raise RuntimeError("The historical test report belongs to a different TradingBot identity.")
    if str(testing_report.get("model_target_schema", "")).upper() not in {"TP_BEFORE_SL_BINARY", "TP_BEFORE_SL"}:
        raise RuntimeError("The historical test report does not use the TP-before-SL target schema.")
    if testing_report.get("split") != "test":
        raise RuntimeError("Historical model testing must use the independent test split.")
    if testing_report.get("testing_mode") != "FROZEN_MODEL_CHRONOLOGICAL_FORWARD_TEST":
        raise RuntimeError("The saved report is not a frozen chronological model test.")

    evaluation_artifact = evaluation_report.get("evaluation_artifact", {})
    if (
        not isinstance(evaluation_artifact, dict)
        or evaluation_artifact.get("storage") != "RUNTIME"
        or evaluation_artifact.get("verified") is not True
    ):
        raise RuntimeError("A verified RUNTIME evaluation report is required for approval.")

    evaluation_gate = evaluation_report.get("approval_gate", {})
    if (
        not isinstance(evaluation_gate, dict)
        or evaluation_gate.get("evaluated_split") != "test"
        or evaluation_gate.get("passed") is not True
        or evaluation_gate.get("approval_eligible") is not True
    ):
        raise RuntimeError("The independent held-out evaluation gate did not pass.")

    testing_integrity = testing_report.get("integrity", {})
    if not isinstance(testing_integrity, dict) or any(
        testing_integrity.get(flag) is not True
        for flag in (
            "chronological_purge_verified",
            "test_rows_excluded_from_scaler_fit",
            "test_rows_excluded_from_model_training",
            "model_weights_frozen_during_test",
        )
    ):
        raise RuntimeError("The historical test integrity checks are incomplete.")
    version_integrity = testing_report.get("version_integrity", {})
    if not isinstance(version_integrity, dict) or any(
        not isinstance(version_integrity.get(name), dict)
        or version_integrity[name].get("valid") is not True
        for name in ("model", "dataset", "scaler")
    ):
        raise RuntimeError("The historical test did not verify every required test dependency.")

    evaluation_model_hash = str(evaluation_report.get("model_artifact_sha256", "")).lower()
    testing_model_hash = str(testing_report.get("model_artifact_sha256", "")).lower()
    if not evaluation_model_hash or evaluation_model_hash != testing_model_hash:
        raise RuntimeError("Evaluation and historical testing did not use the same immutable .pth file.")

    testing_run_id = str(testing_report.get("testing_run_id") or "").strip()
    report_loader = getattr(ai, "load_test_report", None)
    if not testing_run_id or not callable(report_loader):
        raise RuntimeError("A persisted, verifiable historical test report is required for approval.")
    verified_testing_report = report_loader(
        testing_run_id,
        identity=runtime.identity,
        verify=True,
    )
    verification = verified_testing_report.get("report_verification", {})
    if not isinstance(verification, dict) or verification.get("valid") is not True:
        raise RuntimeError("The persisted historical test report failed its integrity check.")
    if str(verified_testing_report.get("model_version_id", "")) != selected_version:
        raise RuntimeError("The persisted historical test report references another model version.")
    if verified_testing_report.get("identity") != runtime.identity.as_dict():
        raise RuntimeError("The persisted historical test report identity does not match this TradingBot.")

    approval_result = ai.approve_evaluated_model(
        selected_version,
        evaluation_report,
        identity=runtime.identity,
    )

    # Pin activation to the exact candidate just approved, not merely the
    # latest approved model for this customer and market.
    if not load_ai_runtime(runtime, model_version_id=selected_version):
        raise RuntimeError(
            "The model was approved but could not be loaded for production inference: "
            + str(runtime.ai_error or "unknown model loading error")
        )
    if str(runtime.ai_model_version or "") != selected_version:
        raise RuntimeError("A different model version was activated after approval.")

    loaded_metadata = runtime.ai_checkpoint
    verified_model = _verify_approved_model_metadata(
        loaded_metadata,
        runtime.identity,
        expected_model_version_id=selected_version,
    )

    _invalidate_strategy_cache(runtime)
    return {
        "model_version_id": selected_version,
        "approval": copy.deepcopy(loaded_metadata.get("approval", {})),
        "approval_operation": (
            approval_result.to_dict()
            if callable(getattr(approval_result, "to_dict", None))
            else approval_result
        ),
        "active_model_version": runtime.ai_model_version,
        # ai_status is an in-memory loading state (usually "loaded"), not the
        # lifecycle status stored and signed in the .pth file.
        "model_status": str(loaded_metadata.get("model_status", "")).upper(),
        "runtime_status": runtime.ai_status,
        "model_path": str(verified_model["model_path"]),
        "model_file_sha256": verified_model["model_file_sha256"],
        "runtime_load_verified": True,
    }


# ============================================================================
# TASK MANAGEMENT
# ============================================================================

def _start_runtime_task(
    runtime: RuntimeContext,
    task_name: str,
    coroutine: Any,
) -> Tuple[asyncio.Task, bool]:
    """Start one runtime task and avoid leaking an unawaited coroutine."""
    existing = runtime.get_task(task_name)
    if existing is not None and not existing.done():
        close_method = getattr(coroutine, "close", None)
        if callable(close_method):
            close_method()
        return existing, False

    task = asyncio.create_task(coroutine)
    if hasattr(runtime, "register_task"):
        if not runtime.register_task(task_name, task):
            task.cancel()
            return runtime.get_task(task_name), False
    else:
        runtime.set_task(task_name, task)
    task.add_done_callback(
        lambda completed: runtime.clear_task(task_name, completed)
    )
    return task, True


def _active_model_operation(runtime: RuntimeContext) -> Optional[str]:
    for task_name in MODEL_OPERATION_TASKS:
        task = runtime.get_task(task_name)
        if task is not None and not task.done():
            return task_name
    return None


async def _send_operation_conflict(
    websocket: WebSocket,
    runtime: RuntimeContext,
    requested_operation: str,
) -> None:
    active_operation = _active_model_operation(runtime)
    response = snapshot(runtime)
    response["success"] = False
    response["operation_conflict"] = {
        "requested": requested_operation,
        "active": active_operation,
        "message": (
            f"Cannot start {requested_operation} while {active_operation} is running. "
            "The active model operation is not interrupted because its worker thread "
            "cannot be safely cancelled."
        ),
    }
    await websocket.send_text(_json_dumps(response))


async def _cancel_runtime_task(runtime: RuntimeContext, task_name: str) -> None:
    task = runtime.get_task(task_name)
    if task is None or task.done():
        runtime.clear_task(task_name)
        return
    if task_name in MODEL_OPERATION_TASKS:
        LOGGER.warning(
            "Leaving non-interruptible model operation %s running for %s",
            task_name,
            runtime.key,
        )
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
    _invalidate_strategy_cache(new_runtime)
    return new_runtime


async def _handle_action(
    websocket: WebSocket,
    runtime: RuntimeContext,
    claims: Optional[VerifiedTradingBotClaims],
    message: Dict[str, Any],
) -> RuntimeContext:
    action = str(message.get("action", "")).strip().lower()

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
        _invalidate_strategy_cache(runtime)
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "update_data":
        runtime.set_startup_progress(
            "Updating chart data",
            5,
            "Synchronizing local chart archives with MT5...",
        )
        await send_snapshot(websocket, runtime)

        async with CHART_IO_LOCK:
            result = await asyncio.to_thread(CHART_SERVICE.update_all_chart_data)
            if result.get("success"):
                await asyncio.to_thread(
                    CHART_SERVICE.load_runtime_history,
                    runtime,
                    runtime.active_timeframe,
                )

        if result.get("success"):
            load_ai_runtime(runtime)
        _invalidate_strategy_cache(runtime)
        runtime.set_startup_progress(
            "Chart ready" if result.get("success") else "Update unavailable",
            100,
            result.get("message", "Chart update finished."),
        )
        response = snapshot(runtime)
        response["update_result"] = result
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action in {"set_setup", "setup_candidate", "candidate_setup", "predict_setup"}:
        setup = message.get("setup", message.get("candidate_setup"))
        if not isinstance(setup, dict):
            raise ValueError("A setup candidate mapping is required.")
        candidate = copy.deepcopy(setup)
        candidate.setdefault("market", runtime.market)
        candidate.setdefault("user_id", runtime.user_id)
        candidate.setdefault("strategy_id", runtime.strategy_id)
        candidate.setdefault("tradingbot_id", runtime.tradingbot_id)
        candidate_timeframe = str(
            candidate.get("entry_timeframe") or candidate.get("timeframe") or runtime.active_timeframe
        ).strip().upper()
        candidate["entry_timeframe"] = candidate_timeframe
        candidate.setdefault("setup_timeframe", candidate_timeframe)
        runtime.set_pending_setup(
            candidate,
            timeframe=candidate_timeframe,
        )
        _invalidate_strategy_cache(runtime)
        await send_snapshot(websocket, runtime)
        return runtime

    if action in {"clear_setup", "clear_candidate", "clear_pending_setup"}:
        timeframe = message.get("timeframe")
        runtime.clear_pending_setup(timeframe=timeframe)
        _invalidate_strategy_cache(runtime)
        await send_snapshot(websocket, runtime)
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
        active_operation = _active_model_operation(runtime)
        if active_operation is not None:
            await _send_operation_conflict(websocket, runtime, "prepare_ai_data")
            return runtime
        result = await asyncio.to_thread(
            _prepare_ai_data_for_runtime,
            runtime,
        )
        response = snapshot(runtime)
        response["ai_data_preparation"] = result
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action in {"training", "play"}:
        async with _learning_archive_lock(runtime):
            existing = runtime.get_task("training")

            if existing is not None and not existing.done():
                await send_snapshot(websocket, runtime)
                return runtime

            active_operation = _active_model_operation(runtime)

            if active_operation is not None:
                await _send_operation_conflict(
                    websocket,
                    runtime,
                    "training",
                )
                return runtime

            await _cancel_runtime_task(runtime, "testing")
            await _cancel_runtime_task(runtime, "manual_analysis")

            _start_runtime_task(
                runtime,
                "training",
                training_loop(runtime),
            )

        await send_snapshot(websocket, runtime)
        return runtime

    if action in {"evaluation", "evaluate_model", "evaluate"}:
        existing = runtime.get_task("evaluation")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime
        active_operation = _active_model_operation(runtime)
        if active_operation is not None:
            await _send_operation_conflict(websocket, runtime, "evaluation")
            return runtime
        await _cancel_runtime_task(runtime, "testing")
        version_id = message.get("model_version_id")
        _start_runtime_task(
            runtime,
            "evaluation",
            evaluation_loop(runtime, version_id),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "testing":
        existing = runtime.get_task("testing")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime
        active_operation = _active_model_operation(runtime)
        if active_operation is not None:
            await _send_operation_conflict(websocket, runtime, "paper_replay")
            return runtime
        _start_runtime_task(
            runtime,
            "testing",
            testing_loop(runtime),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action in {"historical_test", "model_test"}:
        existing = runtime.get_task("historical_test")
        if existing is not None and not existing.done():
            await send_snapshot(websocket, runtime)
            return runtime
        active_operation = _active_model_operation(runtime)
        if active_operation is not None:
            await _send_operation_conflict(websocket, runtime, "historical_test")
            return runtime
        await _cancel_runtime_task(runtime, "testing")
        version_id = message.get("model_version_id")
        _start_runtime_task(
            runtime,
            "historical_test",
            historical_testing_loop(runtime, version_id),
        )
        await send_snapshot(websocket, runtime)
        return runtime

    if action in {"approve_model", "approve_model_version"}:
        active_operation = _active_model_operation(runtime)
        if active_operation is not None:
            await _send_operation_conflict(websocket, runtime, "approve_model")
            return runtime

        result = await asyncio.to_thread(
            approve_model_for_runtime,
            runtime,
            message.get("model_version_id"),
        )

        if (
            isinstance(result, dict)
            and str(result.get("model_status", "")).upper() == "APPROVED"
        ):
            cleanup_result = await asyncio.to_thread(
                _cleanup_runtime_directory,
                runtime,
                expected_model_version_id=str(result.get("model_version_id", "")),
            )
            result["runtime_cleanup"] = cleanup_result

        response = snapshot(runtime)

        response["model_approval"] = result
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action in {"reload_model", "load_model"}:
        loaded = await asyncio.to_thread(load_ai_runtime, runtime)
        _invalidate_strategy_cache(runtime)
        response = snapshot(runtime)
        response["model_reload"] = {
            "success": loaded,
            "error": runtime.ai_error,
            "model_version_id": runtime.ai_model_version,
        }
        await websocket.send_text(_json_dumps(response))
        return runtime

    if action == "timeline":
        if ai is None or not callable(getattr(ai, "build_timeline_from_storage", None)):
            raise AIIntegrationPending("ai.py does not expose build_timeline_from_storage().")
        timeline = await asyncio.to_thread(
            ai.build_timeline_from_storage,
            runtime.market,
            identity=runtime.identity,
            timeframe=message.get("timeframe"),
            start_time=message.get("start_time"),
            end_time_exclusive=message.get("end_time_exclusive"),
            as_of_time=message.get("as_of_time"),
        )
        await websocket.send_text(_json_dumps({
            "type": "timeline",
            "success": True,
            "symbol": runtime.market,
            "identity": runtime.identity.as_dict(),
            "timeline": timeline,
        }))
        return runtime

    if action == "engine_capabilities":
        if ai is None or not callable(getattr(ai, "get_engine_capabilities", None)):
            raise AIIntegrationPending("ai.py does not expose get_engine_capabilities().")
        capabilities = ai.get_engine_capabilities(import_modules=False)
        await websocket.send_text(_json_dumps({
            "type": "engine_capabilities",
            "success": True,
            "capabilities": capabilities,
        }))
        return runtime

    if action == "stop":
        for task_name in ("testing", "manual_analysis"):
            await _cancel_runtime_task(runtime, task_name)
        active_operation = _active_model_operation(runtime)
        runtime.current_session.trading_enabled = False
        if active_operation is None:
            runtime.current_session.running = False
            runtime.set_startup_progress("Stopped", 100, "TradingBot stopped.")
        else:
            runtime.set_startup_progress(
                "Trading stopped",
                100,
                f"Trading is disabled. The active {active_operation} operation will finish safely.",
            )
        await send_snapshot(websocket, runtime)
        return runtime

    if action == "learning_save":
        setup = message.get("setup")
        if not isinstance(setup, dict):
            raise ValueError("Invalid learning setup data.")

        async with _learning_archive_lock(runtime):
            if _active_model_operation(runtime) == "training":
                await _send_operation_conflict(
                    websocket,
                    runtime,
                    "learning_save",
                )
                return runtime

            await _cancel_runtime_task(runtime, "manual_analysis")

            saved_setup = await asyncio.to_thread(
                setup_storage.save_learning_setup,
                runtime.identity,
                setup,
            )

            runtime.manual_analysis.reset()
            _invalidate_strategy_cache(runtime)

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
        raw_id = message.get("id")

        try:
            if isinstance(raw_id, bool):
                raise ValueError
            id_value = int(raw_id)
            if id_value <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Invalid id.")

        async with _learning_archive_lock(runtime):
            if _active_model_operation(runtime) == "training":
                await _send_operation_conflict(
                    websocket,
                    runtime,
                    "learning_delete",
                )
                return runtime

            await _cancel_runtime_task(runtime, "manual_analysis")

            deleted = await asyncio.to_thread(
                setup_storage.delete_learning_setup,
                runtime.identity,
                id_value,
            )

            runtime.manual_analysis.reset()
            _invalidate_strategy_cache(runtime)

        response = snapshot(runtime)
        response["learning_action"] = "deleted"
        response["id"] = id_value
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
        raw_id = message.get("id")

        try:
            if isinstance(raw_id, bool):
                raise ValueError
            id_value = int(raw_id)
            if id_value <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            await websocket.send_text(_json_dumps({
                "type": "trade_detail",
                "success": False,
                "error": "Invalid id.",
            }))
            return runtime

        detail = trading_engine.get_trade_detail(runtime, id_value)

        await websocket.send_text(_json_dumps({
            "type": "trade_detail",
            "success": detail is not None,
            "id": id_value,
            "symbol": runtime.market,
            "timeframe": runtime.active_timeframe,
            "trade": detail,
            "error": None if detail is not None else f"Trade #{id_value} was not found.",
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
        "model_target": "TP_BEFORE_SL",
        "probability_threshold": DEFAULT_PROBABILITY_THRESHOLD,
        "supported_ai_actions": ["prepare_ai_data", "training", "evaluation", "testing", "historical_test", "approve_model", "reload_model", "timeline"],
    }


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "web_mode": WEB_MODE,
        "active_runtimes": len(RUNTIME_REGISTRY),
        "ai_facade_available": ai is not None,
        "ai_target": "TP_BEFORE_SL",
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

@app.get("/{filename}")
async def frontend_module(filename: str):
    if filename not in FRONTEND_MODULE_FILES:
        return JSONResponse(
            status_code=404,
            content={"error": "Frontend module was not found."},
        )

    frontend_root = BASE_DIR.resolve()
    requested = (frontend_root / filename).resolve()

    if requested.parent != frontend_root or not requested.is_file():
        return JSONResponse(
            status_code=404,
            content={"error": f"Frontend module {filename} was not found."},
        )

    return FileResponse(
        requested,
        media_type="application/javascript",
    )


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
        LOGGER.warning(
            "WebSocket authentication rejected: %s",
            str(error),
        )
        await reject_websocket(
            websocket,
            code=1008,
            reason=str(error),
        )

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
