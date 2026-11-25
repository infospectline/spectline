"""Training orchestration and run tracking for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import importlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# MODULE DEPENDENCIES
# ============================================================================

try:
    from . import settings as settings_engine
except ImportError:
    try:
        from engine import settings as settings_engine
    except ImportError:
        settings_engine = None

try:
    from client import ClientIdentity, normalize_market, resolve_identity
except ImportError:
    try:
        from ..client import ClientIdentity, normalize_market, resolve_identity
    except ImportError:
        ClientIdentity = Any

        def normalize_market(value: Any) -> str:
            normalized = str(value).strip().upper().replace("/", "_").replace("\\", "_")
            if not normalized:
                raise ValueError("market cannot be empty.")
            return normalized

        def resolve_identity(
            market: str,
            user_id: Optional[str] = None,
            strategy_id: Optional[str] = None,
            tradingbot_id: Optional[str] = None,
        ) -> Any:
            values = {
                "user_id": user_id,
                "market": normalize_market(market),
                "strategy_id": strategy_id,
                "tradingbot_id": tradingbot_id,
            }
            missing = [key for key, value in values.items() if value is None or not str(value).strip()]
            if missing:
                raise ValueError("A complete TradingBot identity is required: " + ", ".join(missing))
            return values


# ============================================================================
# CONFIGURATION
# ============================================================================

MODULE_SCHEMA_VERSION = 1
TRAINING_RUN_ARTIFACT_TYPE = "training_run"
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET_NAME = "TP_BEFORE_SL"
RUN_EVENT_SCHEMA_VERSION = 1

DEFAULT_EPOCHS = int(getattr(settings_engine, "DEFAULT_EPOCHS", 80)) if settings_engine else 80
DEFAULT_BATCH_SIZE = int(getattr(settings_engine, "DEFAULT_BATCH_SIZE", 64)) if settings_engine else 64
DEFAULT_LEARNING_RATE = float(getattr(settings_engine, "DEFAULT_LEARNING_RATE", 1e-3)) if settings_engine else 1e-3
DEFAULT_WEIGHT_DECAY = float(getattr(settings_engine, "DEFAULT_WEIGHT_DECAY", 1e-5)) if settings_engine else 1e-5
DEFAULT_SEED = int(getattr(settings_engine, "SEED", 42)) if settings_engine else 42

_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class TrainingError(RuntimeError):
    """Base exception for training orchestration failures."""


class TrainingValidationError(TrainingError, ValueError):
    """Raised when a training request violates the training contract."""


class TrainingDependencyError(TrainingError, ImportError):
    """Raised when a required engine module is unavailable."""


class TrainingInProgressError(TrainingError):
    """Raised when another training run is active for the same TradingBot."""


@dataclass(frozen=True)
class TrainingConfig:
    """Validated settings passed to the model training entry point."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not _is_finite(self.epochs):
            raise TrainingValidationError("epochs must be a positive integer.")
        if isinstance(self.batch_size, bool) or not _is_finite(self.batch_size):
            raise TrainingValidationError("batch_size must be a positive integer.")
        epochs_value = float(self.epochs)
        batch_value = float(self.batch_size)
        if not epochs_value.is_integer() or epochs_value <= 0:
            raise TrainingValidationError("epochs must be a positive integer.")
        if not batch_value.is_integer() or batch_value <= 0:
            raise TrainingValidationError("batch_size must be a positive integer.")
        if isinstance(self.learning_rate, bool) or not _is_finite(self.learning_rate):
            raise TrainingValidationError("learning_rate must be a finite number greater than zero.")
        if isinstance(self.weight_decay, bool) or not _is_finite(self.weight_decay):
            raise TrainingValidationError("weight_decay must be a finite non-negative number.")
        if float(self.learning_rate) <= 0.0:
            raise TrainingValidationError("learning_rate must be greater than zero.")
        if float(self.weight_decay) < 0.0:
            raise TrainingValidationError("weight_decay cannot be negative.")
        object.__setattr__(self, "epochs", int(epochs_value))
        object.__setattr__(self, "batch_size", int(batch_value))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# VALIDATION AND NORMALIZATION HELPERS
# ============================================================================

def _is_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return numeric == numeric and abs(numeric) != float("inf")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_id(value: Optional[str] = None) -> str:
    raw = str(value or "").strip()
    if not raw:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw = f"training_{timestamp}_{uuid.uuid4().hex[:12]}"
    if len(raw) > 128 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-" for character in raw):
        raise TrainingValidationError("run_id contains unsupported characters or is too long.")
    if raw in {".", ".."}:
        raise TrainingValidationError("run_id is not a valid identifier.")
    return raw


def _identity_values(identity: Any) -> Dict[str, str]:
    if isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        method = getattr(identity, "as_dict", None)
        if not callable(method):
            raise TrainingValidationError("identity must be a mapping or expose as_dict().")
        raw = method()
    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [key for key in required if raw.get(key) is None or not str(raw[key]).strip()]
    if missing:
        raise TrainingValidationError("A complete TradingBot identity is required: " + ", ".join(missing))
    normalized = {
        "user_id": str(raw["user_id"]).strip(),
        "market": normalize_market(raw["market"]),
        "strategy_id": str(raw["strategy_id"]).strip(),
        "tradingbot_id": str(raw["tradingbot_id"]).strip(),
    }
    for field, value in normalized.items():
        if len(value) > 128 or value in {".", ".."} or any(character in value for character in ("/", "\\", "\x00")):
            raise TrainingValidationError(f"Identity field {field} is not a safe identifier.")
    return normalized


def _identity_lock_key(identity: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_identity_lock(identity: Mapping[str, str]) -> threading.Lock:
    key = _identity_lock_key(identity)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not _is_finite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)

# ============================================================================
# DEPENDENCY LOADERS
# ============================================================================

def _load_model_engine() -> Any:
    candidates = []
    if __package__:
        candidates.append(f"{__package__}.model")
    candidates.append("engine.model")
    last_error: Optional[BaseException] = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if callable(getattr(module, "train_market", None)):
                return module
        except ImportError as error:
            last_error = error
    raise TrainingDependencyError("engine.model.train_market is unavailable.") from last_error


def _load_memory_engine() -> Any:
    candidates = []
    if __package__:
        candidates.append(f"{__package__}.memory")
    candidates.append("engine.memory")
    last_error: Optional[BaseException] = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if callable(getattr(module, "model_dir", None)):
                return module
        except ImportError as error:
            last_error = error
    raise TrainingDependencyError("engine.memory is unavailable.") from last_error


def _resolve_identity(
    market: str,
    *,
    identity: Any = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Any:
    model_engine = _load_model_engine()
    resolver = getattr(model_engine, "_resolve_model_identity", None)
    if callable(resolver):
        resolved = resolver(
            normalize_market(market),
            identity=identity,
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
        values = _identity_values(resolved)
        if values["market"] != normalize_market(market):
            raise TrainingValidationError("Identity market does not match the requested market.")
        return resolved

    if identity is not None:
        values = _identity_values(identity)
        if values["market"] != normalize_market(market):
            raise TrainingValidationError("Identity market does not match the requested market.")
        return identity
    try:
        return resolve_identity(
            market=normalize_market(market),
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
    except Exception as error:
        raise TrainingValidationError("A complete TradingBot identity is required.") from error


# ============================================================================
# APPEND-ONLY RUN JOURNAL
# ============================================================================

def _journal_path(identity: Any, market: str) -> Path:
    """Return the temporary training journal path inside the market RUNTIME."""

    memory_engine = _load_memory_engine()

    models_root = Path(
        memory_engine.model_dir(
            market,
            identity=identity,
        )
    ).expanduser().resolve()

    runtime_root = models_root / "RUNTIME"

    return runtime_root / "TRAINING_RUNS" / "EVENTS.jsonl"

def _append_run_event(
    identity: Any,
    market: str,
    *,
    run_id: str,
    event_type: str,
    status: str,
    timestamp: str,
    config: Mapping[str, Any],
    details: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = _journal_path(identity, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity_data = _identity_values(identity)
    event = {
        "schema_version": RUN_EVENT_SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": uuid.uuid4().hex,
        "event_type": str(event_type).upper(),
        "status": str(status).upper(),
        "timestamp": timestamp,
        "identity": identity_data,
        "target_schema": MODEL_TARGET_SCHEMA,
        "config": _json_safe(config),
        "details": _json_safe(dict(details or {})),
    }
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("Training journal write was incomplete.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path

def _build_progress_callback(
    identity: Any,
    market: str,
    *,
    run_id: str,
    config: Mapping[str, Any],
    downstream_callback: Optional[Callable[[int, str, str], None]] = None,
) -> Callable[[int, str, str], None]:
    """
    Persist training progress inside RUNTIME and forward progress
    to the original application callback.
    """

    def _callback(progress: int, stage: str, message: str) -> None:

        try:
            normalized_progress = max(0, min(99, int(progress)))
        except (TypeError, ValueError, OverflowError):
            normalized_progress = 0

        normalized_stage = str(stage)
        normalized_message = str(message)

        try:
            _append_run_event(
                identity,
                market,
                run_id=run_id,
                event_type="PROGRESS",
                status="RUNNING",
                timestamp=_utc_now(),
                config=config,
                details={
                    "progress": normalized_progress,
                    "stage": normalized_stage,
                    "message": normalized_message,
                },
            )

        except Exception as error:

            # Logging failure must not interrupt neural network training.
            print(
                f"[TRAIN][WARNING] Could not persist progress event: {error}",
                flush=True,
            )

        if downstream_callback is not None:

            downstream_callback(
                normalized_progress,
                normalized_stage,
                normalized_message,
            )

    return _callback


def read_training_events(
    *,
    market: str,
    identity: Any,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read append-only lifecycle events for one TradingBot."""
    normalized_market = normalize_market(market)
    identity_data = _identity_values(identity)
    if identity_data["market"] != normalized_market:
        raise TrainingValidationError("Identity market does not match the requested market.")
    path = _journal_path(identity, normalized_market)
    if not path.is_file():
        return []
    output: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingError(f"Training journal contains invalid JSON on line {line_number}.") from error
            if not isinstance(record, Mapping):
                raise TrainingError(f"Training journal line {line_number} is not an object.")
                
            record_identity = record.get("identity")

            if not isinstance(record_identity, Mapping):
                raise TrainingError(
                    f"Training journal line {line_number} has no valid identity."
                )

            # Other TradingBots can share the same market RUNTIME.
            # Return only events belonging to the requested TradingBot.

            if dict(record_identity) != identity_data:
                continue

            if run_id is None or str(record.get("run_id")) == str(run_id):
                output.append(dict(record))
    return output


# ============================================================================
# RUNTIME-LOCAL TRAINING RUN RECORDS
# ============================================================================


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _training_runs_directory(
    identity: Any,
    market: Optional[str] = None,
) -> Path:
    """
    Return the private RUNS directory for one TradingBot
    inside its market RUNTIME directory.
    """

    identity_data = _identity_values(identity)

    normalized_market = normalize_market(
        market or identity_data["market"]
    )

    if identity_data["market"] != normalized_market:
        raise TrainingValidationError(
            "Identity market does not match the requested training-run directory."
        )

    # Separate TradingBots can safely reuse the same run_id.
    identity_hash = _identity_lock_key(identity_data)

    return (
        _journal_path(identity, normalized_market).parent
        / "RUNS"
        / identity_hash
    )


def _training_run_path(
    identity: Any,
    run_id: str,
) -> Path:

    safe_run_id = _safe_run_id(run_id)

    return (
        _training_runs_directory(identity)
        / f"{safe_run_id}.json"
    )


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    """
    Atomically write one JSON record without leaving partial files.
    """

    path = Path(path).expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = _canonical_json_bytes(value)

    temporary_path: Optional[Path] = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:

            temporary_path = Path(handle.name)

            handle.write(encoded)
            handle.flush()

            os.fsync(handle.fileno())

        if path.exists():
            raise TrainingError(
                f"Training-run record already exists and is immutable: {path.name}"
            )

        os.replace(
            temporary_path,
            path,
        )

        temporary_path = None

    finally:

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_training_run_record(
    identity: Any,
    run_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Read a runtime-local training record and verify its identity
    and SHA-256 integrity.
    """

    safe_run_id = _safe_run_id(run_id)

    identity_data = _identity_values(identity)

    path = _training_run_path(
        identity_data,
        safe_run_id,
    )

    if not path.is_file():
        raise TrainingError(
            f"Training run {safe_run_id!r} was not found in RUNTIME."
        )

    try:

        stored = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError) as error:

        raise TrainingError(
            f"Training-run record {safe_run_id!r} cannot be read."
        ) from error

    if not isinstance(stored, Mapping):
        raise TrainingError("Training-run record is malformed.")

    record = dict(stored)

    if record.get("schema_version") != MODULE_SCHEMA_VERSION:
        raise TrainingError(
            "Training-run record schema version is unsupported."
        )

    if record.get("record_type") != TRAINING_RUN_ARTIFACT_TYPE:
        raise TrainingError(
            "Stored file is not a training-run record."
        )

    if str(record.get("run_id")) != safe_run_id:
        raise TrainingError(
            "Training-run record ID does not match its filename."
        )

    stored_identity = record.get("identity")

    if (
        not isinstance(stored_identity, Mapping)
        or dict(stored_identity) != identity_data
    ):
        raise TrainingValidationError(
            "Training-run identity does not match the requested TradingBot."
        )

    payload = record.get("payload")

    if not isinstance(payload, Mapping):
        raise TrainingError(
            "Training-run payload is malformed."
        )

    actual_hash = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()

    expected_hash = str(
        record.get("payload_sha256") or ""
    )

    if not expected_hash or actual_hash != expected_hash:
        raise TrainingError(
            "Training-run payload failed its SHA-256 integrity check."
        )

    verification = {
        "valid": True,
        "run_id": safe_run_id,
        "identity": identity_data,
        "sha256": actual_hash,
        "path": str(path),
    }

    return record, dict(payload), verification


def _training_run_summary(
    record: Mapping[str, Any],
    path: Path,
) -> Dict[str, Any]:
    """Create a public summary compatible with the previous run-record API."""

    payload = record.get("payload", {})

    if not isinstance(payload, Mapping):
        payload = {}

    dependencies = record.get("dependencies", [])

    if not isinstance(dependencies, list):
        dependencies = []

    status = str(
        record.get(
            "run_status",
            payload.get("run_status", "UNKNOWN"),
        )
    ).upper()

    identity_data = dict(record.get("identity", {}))

    return {
        "version_id": str(record.get("run_id", path.stem)),
        "run_id": str(record.get("run_id", path.stem)),
        "artifact_type": TRAINING_RUN_ARTIFACT_TYPE,
        "scope_type": "CUSTOMER",
        "scope_key": _identity_lock_key(identity_data),
        "market": identity_data.get("market"),
        "identity": identity_data,
        "artifact_uri": str(path),
        "artifact_sha256": str(record.get("payload_sha256", "")),
        "artifact_size_bytes": len(
            _canonical_json_bytes(payload)
        ),
        "metadata": {
            "run_status": status,
            "model_target_schema": record.get(
                "target_schema",
                MODEL_TARGET_SCHEMA,
            ),
            "model_version_id": record.get("model_version_id"),
        },
        "dependencies": [
            dict(item)
            for item in dependencies
            if isinstance(item, Mapping)
        ],
        "parent_version_id": None,
        "created_at": str(record.get("created_at", "")),
        "status": status,
        "manifest_path": str(path),
        "run_status": status,
    }


def _register_training_run(
    *,
    identity: Any,
    run_id: str,
    payload: Mapping[str, Any],
    training_result: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save a training-run record inside RUNTIME.
    No external VERSIONS registry is used.
    """

    safe_run_id = _safe_run_id(run_id)

    identity_data = _identity_values(identity)

    payload_data = _json_safe(dict(payload))

    if not isinstance(payload_data, Mapping):
        raise TrainingValidationError(
            "Training-run payload must be a JSON object."
        )

    result = (
        training_result
        if isinstance(training_result, Mapping)
        else {}
    )

    dependencies: List[Dict[str, str]] = []

    for artifact_type, section, version_field in (
        ("dataset", result.get("dataset"), "version_id"),
        ("scaler", result.get("scaler"), "version_id"),
        ("model", result.get("model"), "model_version_id"),
    ):

        if not isinstance(section, Mapping):
            continue

        version_id = section.get(version_field)

        if version_id:
            dependencies.append(
                {
                    "artifact_type": artifact_type,
                    "version_id": str(version_id),
                }
            )

    payload_hash = hashlib.sha256(
        _canonical_json_bytes(payload_data)
    ).hexdigest()

    model_section = result.get("model", {})

    model_version_id = (
        model_section.get("model_version_id")
        if isinstance(model_section, Mapping)
        else None
    )

    record = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "record_type": TRAINING_RUN_ARTIFACT_TYPE,
        "run_id": safe_run_id,
        "identity": identity_data,
        "created_at": _utc_now(),
        "run_status": str(
            payload_data.get("run_status", "UNKNOWN")
        ).upper(),
        "target_schema": MODEL_TARGET_SCHEMA,
        "model_version_id": (
            str(model_version_id)
            if model_version_id
            else None
        ),
        "dependencies": dependencies,
        "payload_sha256": payload_hash,
        "payload": dict(payload_data),
    }

    path = _training_run_path(
        identity_data,
        safe_run_id,
    )

    _atomic_write_json(
        path,
        record,
    )

    return _training_run_summary(
        record,
        path,
    )
    
# ============================================================================
# TRAINING ORCHESTRATION
# ============================================================================

def train_market(
    market: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[Any] = None,
    run_id: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> Dict[str, Any]:
    """Run the complete TP-before-SL training pipeline for one TradingBot.

    The model engine builds and validates the point-in-time training dataset,
    fits the scaler on training rows only, trains against the validation split,
    and stores the resulting model as DRAFT. This orchestration layer adds
    per-TradingBot concurrency control and append-only run tracking.
    """
    normalized_market = normalize_market(market)
    resolved_identity = _resolve_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    identity_data = _identity_values(resolved_identity)
    selected_config = config or TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    if not isinstance(selected_config, TrainingConfig):
        if isinstance(selected_config, Mapping):
            selected_config = TrainingConfig(**dict(selected_config))
        else:
            raise TrainingValidationError("config must be a TrainingConfig or mapping.")
    resolved_run_id = _safe_run_id(run_id)
    run_lock = _get_identity_lock(identity_data)
    if not run_lock.acquire(blocking=False):
        raise TrainingInProgressError("A training run is already active for this TradingBot.")

    started_at = _utc_now()
    journal_path: Optional[Path] = None
    started_recorded = False
    try:
        existing_events = read_training_events(
            market=normalized_market,
            identity=resolved_identity,
            run_id=resolved_run_id,
        )
        if existing_events:
            raise TrainingValidationError("run_id has already been used for this TradingBot.")

        journal_path = _append_run_event(
            resolved_identity,
            normalized_market,
            run_id=resolved_run_id,
            event_type="STARTED",
            status="RUNNING",
            timestamp=started_at,
            config=selected_config.to_dict(),
            details={"module_schema_version": MODULE_SCHEMA_VERSION},
        )
        started_recorded = True

        model_engine = _load_model_engine()
        
        model_train_kwargs: Dict[str, Any] = {
            "epochs": selected_config.epochs,
            "batch_size": selected_config.batch_size,
            "learning_rate": selected_config.learning_rate,
            "weight_decay": selected_config.weight_decay,
            "identity": resolved_identity,
        }

        model_train_kwargs["progress_callback"] = _build_progress_callback(
            resolved_identity,
            normalized_market,
            run_id=resolved_run_id,
            config=selected_config.to_dict(),
            downstream_callback=progress_callback,
        )

        result = model_engine.train_market(
            normalized_market,
            **model_train_kwargs,
        )

        if not isinstance(result, Mapping):
            raise TrainingValidationError("engine.model.train_market returned an invalid result.")
        result_data = dict(result)
        training_summary = result_data.get("training")
        model_section = result_data.get("model")
        dataset_section = result_data.get("dataset")
        scaler_section = result_data.get("scaler")
        if not isinstance(training_summary, Mapping):
            raise TrainingValidationError("Training result does not contain a training summary.")
        if str(training_summary.get("model_target_schema", "")).upper() != MODEL_TARGET_SCHEMA:
            raise TrainingValidationError("Training result does not use the TP-before-SL binary target schema.")
        if not isinstance(model_section, Mapping) or not model_section.get("model_version_id"):
            raise TrainingValidationError("Training result does not contain a registered model version.")
        if str(model_section.get("status", "")).upper() != "DRAFT":
            raise TrainingValidationError("A newly trained model must remain in DRAFT state.")
        if not isinstance(dataset_section, Mapping) or not dataset_section.get("version_id"):
            raise TrainingValidationError("Training result does not contain a dataset version.")
        if not isinstance(scaler_section, Mapping) or not scaler_section.get("version_id"):
            raise TrainingValidationError("Training result does not contain a scaler version.")

        validation_available = training_summary.get("validation_available") is True
        run_status = "COMPLETED" if validation_available else "COMPLETED_WITH_WARNINGS"
        warnings = list(training_summary.get("training_warnings", []))
        if not validation_available:
            warnings.append(
                "No chronological validation rows were available. The checkpoint remains DRAFT and must not be approved without a separate evaluation."
            )
        if training_summary.get("selection_metric") == "TRAIN_LOG_LOSS_FALLBACK":
            warnings.append("Early stopping used training loss because validation data was unavailable.")
        training_summary_data = dict(training_summary)
        training_summary_data["training_warnings"] = list(dict.fromkeys(str(item) for item in warnings))
        result_data["training"] = training_summary_data
        result_data["orchestration"] = {
            "module_schema_version": MODULE_SCHEMA_VERSION,
            "run_id": resolved_run_id,
            "run_status": run_status,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "identity": identity_data,
            "target_schema": MODEL_TARGET_SCHEMA,
            "approval_status": "NOT_APPROVED",
            "evaluation_required": True,
            "calibration_required": True,
            "journal_path": str(journal_path) if journal_path else None,
            "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        }
        completed_at = result_data["orchestration"]["completed_at"]
        event_details = {
            "model_version_id": model_section.get("model_version_id"),
            "dataset_version_id": dataset_section.get("version_id"),
            "scaler_version_id": scaler_section.get("version_id"),
            "validation_available": validation_available,
            "best_epoch": training_summary_data.get("best_epoch"),
            "validation_metrics": training_summary_data.get("validation_metrics"),
        }
        _append_run_event(
            resolved_identity,
            normalized_market,
            run_id=resolved_run_id,
            event_type="COMPLETED",
            status=run_status,
            timestamp=completed_at,
            config=selected_config.to_dict(),
            details=event_details,
        )
        run_payload = {
            "schema_version": MODULE_SCHEMA_VERSION,
            "run_id": resolved_run_id,
            "run_status": run_status,
            "started_at": started_at,
            "completed_at": completed_at,
            "identity": identity_data,
            "market": normalized_market,
            "target_schema": MODEL_TARGET_SCHEMA,
            "config": selected_config.to_dict(),
            "training_summary": training_summary_data,
            "artifacts": {
                "dataset_version_id": dataset_section.get("version_id"),
                "scaler_version_id": scaler_section.get("version_id"),
                "model_version_id": model_section.get("model_version_id"),
                "model_status": model_section.get("status"),
            },
            "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        }
        try:
            result_data["training_run_version"] = _register_training_run(
                identity=resolved_identity,
                run_id=resolved_run_id,
                payload=run_payload,
                training_result=result_data,
            )
        except Exception as error:
            result_data["orchestration"]["tracking_warning"] = (
                "The training result was produced, but its RUNTIME training-run record could not be saved: "
                + str(error)
            )
        return result_data

    except Exception as error:
        if started_recorded:
            failed_at = _utc_now()
            failure_details = {
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
            }
            try:
                _append_run_event(
                    resolved_identity,
                    normalized_market,
                    run_id=resolved_run_id,
                    event_type="FAILED",
                    status="FAILED",
                    timestamp=failed_at,
                    config=selected_config.to_dict(),
                    details=failure_details,
                )
            except Exception:
                pass
            failed_payload = {
                "schema_version": MODULE_SCHEMA_VERSION,
                "run_id": resolved_run_id,
                "run_status": "FAILED",
                "started_at": started_at,
                "completed_at": failed_at,
                "identity": identity_data,
                "market": normalized_market,
                "target_schema": MODEL_TARGET_SCHEMA,
                "config": selected_config.to_dict(),
                "failure": failure_details,
            }
            try:
                _register_training_run(
                    identity=resolved_identity,
                    run_id=resolved_run_id,
                    payload=failed_payload,
                )
            except Exception:
                pass
        raise
    finally:
        run_lock.release()


# ============================================================================
# TRAINING ENTRY POINT ALIASES
# ============================================================================

def run_market_training(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Explicit alias for the market-level training orchestration function."""
    return train_market(*args, **kwargs)


def run_training(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility alias for callers that use a generic training entry point."""
    return train_market(*args, **kwargs)


# ============================================================================
# TRAINING RUN QUERIES
# ============================================================================


def list_training_runs(
    *,
    identity: Any,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List verified training-run records stored in RUNTIME.
    """

    identity_data = _identity_values(identity)

    directory = _training_runs_directory(identity_data)

    if not directory.is_dir():
        return []

    requested_status = (
        str(status).strip().upper()
        if status is not None
        else None
    )

    output: List[Dict[str, Any]] = []

    for path in sorted(
        directory.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):

        record, _payload, _verification = _read_training_run_record(
            identity_data,
            path.stem,
        )

        summary = _training_run_summary(
            record,
            path,
        )

        if (
            requested_status is None
            or summary["run_status"] == requested_status
        ):
            output.append(summary)

    return output


def get_training_run(
    run_id: str,
    *,
    identity: Any,
) -> Dict[str, Any]:
    """
    Retrieve a RUNTIME training-run record
    and verify its payload integrity.
    """

    record, payload, verification = _read_training_run_record(
        identity,
        run_id,
    )

    path = _training_run_path(
        identity,
        run_id,
    )

    summary = _training_run_summary(
        record,
        path,
    )

    return {
        **summary,
        "payload": payload,
        "verification": verification,
    }


def verify_training_run(
    run_id: str,
    *,
    identity: Any,
) -> Dict[str, Any]:
    """
    Verify SHA-256 integrity and identity of a RUNTIME training run.
    """

    _record, _payload, verification = _read_training_run_record(
        identity,
        run_id,
    )

    return verification

# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "MODULE_SCHEMA_VERSION",
    "TRAINING_RUN_ARTIFACT_TYPE",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET_NAME",
    "TrainingError",
    "TrainingValidationError",
    "TrainingDependencyError",
    "TrainingInProgressError",
    "TrainingConfig",
    "train_market",
    "run_market_training",
    "run_training",
    "read_training_events",
    "list_training_runs",
    "get_training_run",
    "verify_training_run",
]
