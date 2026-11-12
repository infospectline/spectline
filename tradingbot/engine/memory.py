"""Identity-isolated, version-aware artifact storage for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

try:
    from client import (
        ClientIdentity,
        get_model_dir,
        get_model_path,
        get_results_dir,
        resolve_identity,
    )
except ImportError:
    @dataclass(frozen=True)
    class ClientIdentity:
        user_id: str
        market: str
        strategy_id: str
        tradingbot_id: str

        def as_dict(self) -> Dict[str, str]:
            return {
                "user_id": self.user_id,
                "market": self.market,
                "strategy_id": self.strategy_id,
                "tradingbot_id": self.tradingbot_id,
            }

    def resolve_identity(
        market: str,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        tradingbot_id: Optional[str] = None,
    ) -> ClientIdentity:
        values = {
            "user_id": user_id,
            "market": market,
            "strategy_id": strategy_id,
            "tradingbot_id": tradingbot_id,
        }
        missing = [key for key, value in values.items() if value is None or not str(value).strip()]
        if missing:
            raise RuntimeError(
                "The project client.py identity resolver is required for incomplete identity: "
                + ", ".join(missing)
            )
        return ClientIdentity(
            user_id=str(user_id).strip(),
            market=str(market).strip().upper().replace("/", "_").replace("\\", "_"),
            strategy_id=str(strategy_id).strip(),
            tradingbot_id=str(tradingbot_id).strip(),
        )

    def _fallback_identity_root(identity: ClientIdentity) -> Path:
        root = Path(os.environ.get("SPECTLINE_DATA_ROOT", Path.cwd() / "DATA"))
        safe = [identity.user_id, identity.market, identity.strategy_id, identity.tradingbot_id]
        return root.joinpath(*[_safe_path_component(item) for item in safe])

    def get_model_dir(identity: ClientIdentity) -> Path:
        return _fallback_identity_root(identity) / "MODELS"

    def get_model_path(identity: ClientIdentity) -> Path:
        return get_model_dir(identity) / "model.pt"

    def get_results_dir(identity: ClientIdentity) -> Path:
        return _fallback_identity_root(identity) / "RESULTS"

try:
    from engine.settings import LOOKBACK_BY_TIMEFRAME, MODEL_FILENAME, SUPPORTED_TIMEFRAMES
except ImportError:
    try:
        from .settings import LOOKBACK_BY_TIMEFRAME, MODEL_FILENAME, SUPPORTED_TIMEFRAMES
    except ImportError:
        LOOKBACK_BY_TIMEFRAME = {
            "M1": 32,
            "M5": 32,
            "M15": 32,
            "M30": 24,
            "H1": 24,
            "H4": 20,
            "D1": 16,
            "W1": 12,
        }
        SUPPORTED_TIMEFRAMES = tuple(LOOKBACK_BY_TIMEFRAME)
        MODEL_FILENAME = "model.pt"


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

STORAGE_SCHEMA_VERSION = 2
MODEL_FORMAT_VERSION = 4
BUNDLE_STORAGE_DIRECTORY = Path("AI_PREPARATION") / "TRAINING_BUNDLES"
DATASET_STORAGE_DIRECTORY = Path("AI_PREPARATION") / "DATASETS"
SCALER_STORAGE_DIRECTORY = Path("AI_PREPARATION") / "SCALERS"
DEFAULT_BUNDLE_SUFFIX = ".json.gz"
DEFAULT_DATASET_SUFFIX = ".dataset.json.gz"
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class MemoryErrorBase(RuntimeError):
    """Base exception for Spectline memory operations."""


class MemoryValidationError(MemoryErrorBase, ValueError):
    """Raised when stored data or an artifact violates its storage contract."""


class MemoryIntegrityError(MemoryErrorBase):
    """Raised when an artifact no longer matches its stored checksum."""


class MemoryVersionError(MemoryErrorBase):
    """Raised when an immutable artifact version cannot be registered or loaded."""


@dataclass
class FeatureScaler:
    """Backward-compatible standard scaler used by the original model interface."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float32).reshape(-1)
        if self.mean.size == 0 or self.std.size == 0:
            raise ValueError("Scaler mean and standard deviation cannot be empty.")
        if self.mean.shape != self.std.shape:
            raise ValueError("Scaler mean and standard deviation must have equal shapes.")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ValueError("Scaler values must be finite.")
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "method": "standard_legacy",
        }


# ============================================================================
# IDENTITY AND PATH HELPERS
# ============================================================================

def _safe_path_component(value: Any) -> str:
    normalized = str(value).strip()
    if not _SAFE_COMPONENT_PATTERN.fullmatch(normalized) or normalized in {".", ".."}:
        raise MemoryValidationError("Identity contains an invalid path component.")
    return normalized


def _identity_dict(identity: Any) -> Dict[str, str]:
    if isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        method = getattr(identity, "as_dict", None)
        if not callable(method):
            raise TypeError("identity must provide as_dict() or be a mapping.")
        raw = method()
    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [key for key in required if raw.get(key) is None or not str(raw[key]).strip()]
    if missing:
        raise MemoryValidationError("Identity is incomplete: " + ", ".join(missing))
    return {
        "user_id": _safe_path_component(raw["user_id"]),
        "market": _safe_path_component(str(raw["market"]).upper().replace("/", "_").replace("\\", "_")),
        "strategy_id": _safe_path_component(raw["strategy_id"]),
        "tradingbot_id": _safe_path_component(raw["tradingbot_id"]),
    }


def _resolve_memory_identity(
    market: str,
    identity: Optional[ClientIdentity] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    normalized_market = str(market).strip().upper().replace("/", "_").replace("\\", "_")
    if not normalized_market:
        raise MemoryValidationError("market cannot be empty.")
    if identity is not None:
        identity_values = _identity_dict(identity)
        if identity_values["market"] != normalized_market:
            raise MemoryValidationError(
                f"Identity market {identity_values['market']} does not match {normalized_market}."
            )
        if isinstance(identity, ClientIdentity):
            return identity
        return resolve_identity(
            market=identity_values["market"],
            user_id=identity_values["user_id"],
            strategy_id=identity_values["strategy_id"],
            tradingbot_id=identity_values["tradingbot_id"],
        )
    return resolve_identity(
        market=normalized_market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def _resolve_identity_from_bundle(
    bundle: Mapping[str, Any],
    identity: Optional[ClientIdentity] = None,
) -> ClientIdentity:
    stored_identity = bundle.get("identity")
    if identity is not None:
        if _identity_dict(stored_identity) != _identity_dict(identity):
            raise PermissionError("Training data identity does not match the requested TradingBot.")
        return identity
    values = _identity_dict(stored_identity)
    return resolve_identity(
        market=values["market"],
        user_id=values["user_id"],
        strategy_id=values["strategy_id"],
        tradingbot_id=values["tradingbot_id"],
    )


def model_dir(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return Path(get_model_dir(resolved)).expanduser().resolve()


def _legacy_model_path(identity: ClientIdentity) -> Path:
    return Path(get_model_path(identity)).expanduser().resolve()


def _version_registry() -> Any:
    module_candidates = []
    if __package__:
        module_candidates.append(f"{__package__}.versions")
    module_candidates.append("engine.versions")
    last_error: Optional[BaseException] = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            getter = getattr(module, "get_default_registry", None)
            if callable(getter):
                return getter()
        except (ImportError, ModuleNotFoundError) as error:
            last_error = error
    raise MemoryVersionError("engine.versions.VersionRegistry is unavailable.") from last_error


def model_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    version_id: Optional[str] = None,
    approved_only: bool = True,
) -> Path:
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    registry = _version_registry()
    if version_id is not None:
        record = registry.get_version("model", version_id, identity=resolved)
        if approved_only and record.status != "APPROVED":
            raise MemoryVersionError("Requested model version is not approved.")
        registry.verify_artifact("model", version_id, identity=resolved)
        return Path(record.artifact_uri).resolve()
    approved = registry.latest_approved("model", identity=resolved)
    if approved is not None:
        registry.verify_artifact("model", approved.version_id, identity=resolved)
        return Path(approved.artifact_uri).resolve()
    return _legacy_model_path(resolved)


def _bundle_directory(identity: ClientIdentity) -> Path:
    return Path(get_results_dir(identity)).expanduser().resolve() / BUNDLE_STORAGE_DIRECTORY


def _dataset_directory(identity: ClientIdentity) -> Path:
    return Path(get_results_dir(identity)).expanduser().resolve() / DATASET_STORAGE_DIRECTORY


def _validate_bundle_id(bundle_id: str) -> str:
    value = str(bundle_id).strip()
    if value.endswith(DEFAULT_BUNDLE_SUFFIX):
        value = value[: -len(DEFAULT_BUNDLE_SUFFIX)]
    if not _SAFE_COMPONENT_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise MemoryValidationError("Invalid artifact identifier.")
    return value


def training_bundle_path(identity: ClientIdentity, bundle_id: str) -> Path:
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    safe_id = _validate_bundle_id(bundle_id)
    return _bundle_directory(identity) / f"{safe_id}{DEFAULT_BUNDLE_SUFFIX}"


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def _atomic_write_bytes(path: Path, payload: bytes, *, immutable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    with _thread_lock(path):
        if immutable and path.exists():
            raise FileExistsError(f"Immutable artifact already exists: {path}")
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            if immutable:
                try:
                    os.link(temporary_path, path)
                except FileExistsError:
                    raise
                except OSError:
                    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    try:
                        with os.fdopen(descriptor, "wb") as destination:
                            destination.write(payload)
                            destination.flush()
                            os.fsync(destination.fileno())
                    except Exception:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        raise
            else:
                os.replace(temporary_path, path)
                temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _atomic_write_gzip_json(path: Path, payload: Mapping[str, Any], *, immutable: bool) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    buffer_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(buffer_path, "wb", compresslevel=6) as compressed_file:
            compressed_file.write(encoded)
        with buffer_path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())
        payload_bytes = buffer_path.read_bytes()
        _atomic_write_bytes(path, payload_bytes, immutable=immutable)
    finally:
        try:
            buffer_path.unlink(missing_ok=True)
        except OSError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================================
# FEATURE SCALING COMPATIBILITY
# ============================================================================

def fit_scaler(features: np.ndarray) -> FeatureScaler:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise MemoryValidationError("features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise MemoryValidationError("features contain NaN or infinite values.")
    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureScaler(mean=mean, std=std)


def apply_scaler(features: np.ndarray, scaler: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise MemoryValidationError("features must be a one- or two-dimensional matrix.")
    if not hasattr(scaler, "mean") or not hasattr(scaler, "std"):
        raise TypeError("scaler must expose mean and std arrays.")
    mean = np.asarray(scaler.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(scaler.std, dtype=np.float32).reshape(-1)
    if matrix.shape[1] != mean.size or std.size != mean.size:
        raise MemoryValidationError("Feature width does not match scaler width.")
    if not np.all(np.isfinite(matrix)):
        raise MemoryValidationError("Inference features contain non-finite values.")
    transformed = (matrix - mean) / np.where(std < 1e-6, 1.0, std)
    return transformed.astype(np.float32)


def fit_training_dataset_scaler(dataset: Mapping[str, Any], **kwargs: Any) -> Any:
    module = importlib.import_module(f"{__package__}.scaling" if __package__ else "engine.scaling")
    return module.fit_dataset_scaler(dataset, **kwargs)


def apply_training_dataset_scaler(features: Any, scaler: Any, **kwargs: Any) -> np.ndarray:
    module = importlib.import_module(f"{__package__}.scaling" if __package__ else "engine.scaling")
    return module.apply_scaler(features, scaler, **kwargs)


# ============================================================================
# SAFE TRAINING DATA SERIALIZATION
# ============================================================================

def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MemoryValidationError("Training data contains a non-finite float.")
        return value
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return {
            "__spectline_type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _to_json_compatible(value.tolist()),
        }
    if isinstance(value, Path):
        return {"__spectline_type__": "path", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__spectline_type__": "dataclass",
            "class_name": type(value).__name__,
            "fields": {
                item.name: _to_json_compatible(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        items = list(value.items())
        if all(isinstance(key, str) for key, _ in items) and "__spectline_type__" not in value:
            return {key: _to_json_compatible(child) for key, child in items}
        return {
            "__spectline_type__": "mapping",
            "items": [[_to_json_compatible(key), _to_json_compatible(child)] for key, child in items],
        }
    if isinstance(value, tuple):
        return {"__spectline_type__": "tuple", "items": [_to_json_compatible(item) for item in value]}
    if isinstance(value, (list, set, frozenset)):
        converted = [_to_json_compatible(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return {"__spectline_type__": "set", "items": converted}
        return converted
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {
            "__spectline_type__": "object_mapping",
            "class_name": type(value).__name__,
            "value": _to_json_compatible(value.to_dict()),
        }
    raise TypeError(f"Unsupported value in Training Data Bundle: {type(value).__name__}.")


def _restore_json_compatible(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_compatible(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__spectline_type__")
    if marker is None:
        return {key: _restore_json_compatible(child) for key, child in value.items()}
    if marker == "ndarray":
        array = np.asarray(value.get("data", []), dtype=value.get("dtype", "float32"))
        shape = tuple(int(item) for item in value.get("shape", array.shape))
        try:
            return array.reshape(shape)
        except ValueError as error:
            raise MemoryIntegrityError("Stored NumPy array has an invalid shape.") from error
    if marker == "path":
        return Path(str(value.get("value", "")))
    if marker == "tuple":
        return tuple(_restore_json_compatible(item) for item in value.get("items", []))
    if marker == "set":
        return set(_restore_json_compatible(item) for item in value.get("items", []))
    if marker == "mapping":
        restored: Dict[Any, Any] = {}
        for pair in value.get("items", []):
            if not isinstance(pair, list) or len(pair) != 2:
                raise MemoryIntegrityError("Stored mapping contains an invalid item.")
            restored[_restore_json_compatible(pair[0])] = _restore_json_compatible(pair[1])
        return restored
    if marker in {"dataclass", "object_mapping"}:
        class_name = str(value.get("class_name", ""))
        raw_fields = value.get("fields") if marker == "dataclass" else value.get("value")
        restored_fields = _restore_json_compatible(raw_fields)
        if not isinstance(restored_fields, Mapping):
            raise MemoryIntegrityError("Stored object fields are malformed.")
        if class_name == "FeatureScaler":
            if "feature_names" in restored_fields or "method" in restored_fields:
                try:
                    scaling_module = importlib.import_module(
                        f"{__package__}.scaling" if __package__ else "engine.scaling"
                    )
                    return scaling_module.scaler_from_serializable(restored_fields)
                except (ImportError, AttributeError, TypeError, ValueError):
                    return FeatureScaler(restored_fields["mean"], restored_fields["std"])
            return FeatureScaler(restored_fields["mean"], restored_fields["std"])
        allowed_classes = {"Candle", "ManualZone", "UserSetup"}
        if class_name not in allowed_classes:
            raise MemoryIntegrityError(f"Unsupported stored dataclass type: {class_name}.")
        module_name = f"{__package__}.data" if __package__ else "engine.data"
        try:
            data_module = importlib.import_module(module_name)
            data_class = getattr(data_module, class_name)
            return data_class(**dict(restored_fields))
        except (ImportError, AttributeError, TypeError) as error:
            raise MemoryIntegrityError(f"Could not restore stored {class_name} data.") from error
    raise MemoryIntegrityError(f"Unknown Training Data Bundle marker: {marker}.")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_gzip_json(path: Path) -> Dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as compressed_file:
            payload = json.load(compressed_file)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise MemoryIntegrityError(f"Stored artifact is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise MemoryIntegrityError("Stored artifact envelope must be a dictionary.")
    return payload


# ============================================================================
# TRAINING DATA BUNDLE STORAGE
# ============================================================================

def save_training_data_bundle(
    bundle: Mapping[str, Any],
    *,
    identity: Optional[ClientIdentity] = None,
    bundle_id: Optional[str] = None,
) -> Path:
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping.")
    resolved = _resolve_identity_from_bundle(bundle, identity)
    payload_text = _canonical_json(dict(bundle))
    payload_encoded = json.loads(payload_text)
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if bundle_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_id = f"bundle_{timestamp}_{uuid.uuid4().hex[:8]}"
    else:
        safe_id = _validate_bundle_id(bundle_id)
    target_path = training_bundle_path(resolved, safe_id)
    envelope = {
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "bundle_id": safe_id,
        "identity": _identity_dict(resolved),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": payload_hash,
        "bundle": payload_encoded,
    }
    _atomic_write_gzip_json(target_path, envelope, immutable=True)
    return target_path


def load_training_data_bundle(
    bundle_id: str,
    *,
    identity: ClientIdentity,
) -> Dict[str, Any]:
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    path = training_bundle_path(identity, bundle_id)
    if not path.is_file():
        raise FileNotFoundError(f"Training Data Bundle was not found: {path}")
    envelope = _read_gzip_json(path)
    if envelope.get("storage_schema_version") not in {1, STORAGE_SCHEMA_VERSION}:
        raise MemoryIntegrityError("Unsupported Training Data Bundle schema version.")
    if envelope.get("identity") != _identity_dict(identity):
        raise PermissionError("Training Data Bundle identity does not match the requested TradingBot.")
    encoded_bundle = envelope.get("bundle")
    if not isinstance(encoded_bundle, dict):
        raise MemoryIntegrityError("Training Data Bundle payload is invalid.")
    canonical_payload = json.dumps(
        encoded_bundle,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    if actual_hash != envelope.get("payload_sha256"):
        raise MemoryIntegrityError("Training Data Bundle failed its integrity check.")
    restored = _restore_json_compatible(encoded_bundle)
    if not isinstance(restored, dict):
        raise MemoryIntegrityError("Restored Training Data Bundle is not a dictionary.")
    if restored.get("identity") is not None and _identity_dict(restored["identity"]) != _identity_dict(identity):
        raise PermissionError("Restored Training Data Bundle identity does not match its storage scope.")
    return restored


def list_training_data_bundles(*, identity: ClientIdentity) -> List[Dict[str, Any]]:
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    directory = _bundle_directory(identity)
    if not directory.is_dir():
        return []
    results: List[Dict[str, Any]] = []
    for path in sorted(directory.glob(f"*{DEFAULT_BUNDLE_SUFFIX}")):
        try:
            stat = path.stat()
            checksum = _sha256_file(path)
        except OSError:
            continue
        results.append(
            {
                "bundle_id": path.name[: -len(DEFAULT_BUNDLE_SUFFIX)],
                "filename": path.name,
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "file_sha256": checksum,
            }
        )
    return results


def delete_training_data_bundle(bundle_id: str, *, identity: ClientIdentity) -> bool:
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    path = training_bundle_path(identity, bundle_id)
    with _thread_lock(path):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False


def prepare_and_store_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    bundle_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    module_name = f"{__package__}.data" if __package__ else "engine.data"
    data_module = importlib.import_module(module_name)
    prepare_function = getattr(data_module, "prepare_training_data", None)
    if not callable(prepare_function):
        raise RuntimeError("engine.data.prepare_training_data is not available.")
    bundle = prepare_function(resolved.market, identity=resolved)
    stored_path = save_training_data_bundle(bundle, identity=resolved, bundle_id=bundle_id)
    return bundle, stored_path


# ============================================================================
# VERSIONED DATASET STORAGE
# ============================================================================

def save_training_dataset(
    dataset: Mapping[str, Any],
    *,
    identity: Optional[ClientIdentity] = None,
    version_id: Optional[str] = None,
    dependencies: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping.")
    resolved = _resolve_identity_from_bundle(dataset, identity)
    dataset_id = str(version_id or dataset.get("dataset_id") or "").strip()
    if not dataset_id:
        dataset_id = f"dataset_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    safe_id = _validate_bundle_id(dataset_id)
    directory = _dataset_directory(resolved)
    path = directory / f"{safe_id}{DEFAULT_DATASET_SUFFIX}"
    encoded_text = _canonical_json(dict(dataset))
    encoded_dataset = json.loads(encoded_text)
    envelope = {
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "dataset_id": safe_id,
        "identity": _identity_dict(resolved),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": hashlib.sha256(encoded_text.encode("utf-8")).hexdigest(),
        "dataset": encoded_dataset,
    }
    _atomic_write_gzip_json(path, envelope, immutable=True)
    try:
        registry = _version_registry()
        return registry.register_dataset(
            dataset,
            identity=resolved,
            version_id=safe_id,
            artifact_path=path,
            dependencies=dependencies,
            metadata=metadata,
        )
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_training_dataset(
    version_id: str,
    *,
    identity: ClientIdentity,
    require_approved: bool = False,
) -> Dict[str, Any]:
    registry = _version_registry()
    record = registry.get_version("dataset", version_id, identity=identity)
    if require_approved and record.status != "APPROVED":
        raise MemoryVersionError("Requested dataset version is not approved.")
    registry.verify_artifact("dataset", version_id, identity=identity)
    path = Path(record.artifact_uri).resolve()
    envelope = _read_gzip_json(path)
    if envelope.get("storage_schema_version") != STORAGE_SCHEMA_VERSION:
        raise MemoryIntegrityError("Unsupported versioned dataset storage schema.")
    if envelope.get("identity") != _identity_dict(identity):
        raise PermissionError("Dataset identity does not match the requested TradingBot.")
    encoded_dataset = envelope.get("dataset")
    if not isinstance(encoded_dataset, dict):
        raise MemoryIntegrityError("Stored dataset payload is invalid.")
    canonical_payload = json.dumps(
        encoded_dataset,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_hash = envelope.get("payload_sha256")
    if hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest() != expected_hash:
        raise MemoryIntegrityError("Stored dataset payload failed its checksum.")
    restored = _restore_json_compatible(encoded_dataset)
    if not isinstance(restored, dict):
        raise MemoryIntegrityError("Restored dataset payload is not a dictionary.")
    if _identity_dict(restored.get("identity")) != _identity_dict(identity):
        raise PermissionError("Restored dataset identity does not match the requested TradingBot.")
    return restored


def list_dataset_versions(*, identity: ClientIdentity, status: Optional[str] = None) -> List[Dict[str, Any]]:
    records = _version_registry().list_versions("dataset", identity=identity, status=status)
    return [record.to_dict() for record in records]


# ============================================================================
# VERSIONED SCALER STORAGE
# ============================================================================

def _scaler_payload(scaler: Any) -> Dict[str, Any]:
    if isinstance(scaler, Mapping):
        payload = dict(scaler)
    else:
        method = getattr(scaler, "to_dict", None)
        if callable(method):
            payload = dict(method())
        elif hasattr(scaler, "mean") and hasattr(scaler, "std"):
            payload = {
                "mean": np.asarray(scaler.mean).reshape(-1).tolist(),
                "std": np.asarray(scaler.std).reshape(-1).tolist(),
                "method": "standard_legacy",
            }
        else:
            raise TypeError("scaler must be a mapping or expose to_dict().")
    mean = np.asarray(payload.get("mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(payload.get("std", []), dtype=np.float64).reshape(-1)
    if mean.size == 0 or mean.shape != std.shape or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise MemoryValidationError("Scaler payload contains invalid mean or std arrays.")
    return _to_json_compatible(payload)


def save_scaler_artifact(
    scaler: Any,
    *,
    identity: ClientIdentity,
    version_id: Optional[str] = None,
    dataset_version_id: Optional[str] = None,
    dependencies: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    payload = _scaler_payload(scaler)
    dependency_list = list(dependencies or ())
    if dataset_version_id:
        dependency_list.append(
            {
                "artifact_type": "dataset",
                "version_id": dataset_version_id,
                "identity": _identity_dict(identity),
            }
        )
    registry = _version_registry()
    scaler_module = None
    try:
        scaler_module = importlib.import_module(
            f"{__package__}.scaling" if __package__ else "engine.scaling"
        )
    except ImportError:
        pass
    scaler_object = scaler
    if isinstance(scaler, Mapping) and scaler_module is not None:
        scaler_object = scaler_module.scaler_from_serializable(scaler)
    elif scaler_module is not None and not callable(getattr(scaler, "to_dict", None)):
        if "feature_names" in payload:
            scaler_object = scaler_module.scaler_from_serializable(payload)
    return registry.register_scaler(
        scaler_object,
        identity=identity,
        version_id=version_id,
        dependencies=dependency_list,
        metadata=metadata,
    )


def load_scaler_artifact(
    version_id: str,
    *,
    identity: ClientIdentity,
    require_approved: bool = False,
) -> Any:
    registry = _version_registry()
    record = registry.get_version("scaler", version_id, identity=identity)
    if require_approved and record.status != "APPROVED":
        raise MemoryVersionError("Requested scaler version is not approved.")
    registry.verify_artifact("scaler", version_id, identity=identity)
    payload_path = Path(record.artifact_uri).resolve()
    try:
        wrapper = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MemoryIntegrityError("Stored scaler artifact is unreadable.") from error
    payload = wrapper.get("artifact_payload") if isinstance(wrapper, dict) else None
    if not isinstance(payload, Mapping):
        raise MemoryIntegrityError("Stored scaler artifact payload is malformed.")
    try:
        module = importlib.import_module(f"{__package__}.scaling" if __package__ else "engine.scaling")
        return module.scaler_from_serializable(payload)
    except (ImportError, AttributeError, TypeError, ValueError):
        return FeatureScaler(payload["mean"], payload["std"])


# ============================================================================
# MODEL CHECKPOINT STORAGE
# ============================================================================

def _model_architecture_metadata(model: nn.Module) -> Tuple[int, List[int]]:
    declared_input = getattr(model, "input_size", None)
    backbone = getattr(model, "backbone", None)
    modules = list(backbone.modules()) if isinstance(backbone, nn.Module) else list(model.modules())
    linear_layers = [module for module in modules if isinstance(module, nn.Linear)]
    input_size = int(declared_input) if declared_input is not None else (
        int(linear_layers[0].in_features) if linear_layers else 0
    )
    if input_size <= 0:
        raise MemoryValidationError("The model must expose input_size or contain a Linear input layer.")
    declared_hidden = getattr(model, "hidden_sizes", None)
    if isinstance(declared_hidden, Sequence) and not isinstance(declared_hidden, (str, bytes)):
        hidden_sizes = [int(item) for item in declared_hidden[:3]]
    else:
        hidden_sizes = [int(layer.out_features) for layer in linear_layers[1:4]]
    return input_size, hidden_sizes


def _model_class_metadata(model: nn.Module) -> Dict[str, Any]:
    model_type = type(model)
    return {
        "model_class_module": str(model_type.__module__),
        "model_class_name": str(model_type.__name__),
    }


def _scaler_for_checkpoint(scaler: Any) -> Tuple[Dict[str, Any], str]:
    payload = _scaler_payload(scaler)
    if isinstance(scaler, FeatureScaler) or payload.get("method") == "standard_legacy":
        return payload, "legacy_standard"
    if "feature_names" in payload or "passthrough_mask" in payload:
        return payload, "engine_scaling"
    return payload, "legacy_standard"


def _checkpoint_dependencies(training_summary: Mapping[str, Any], identity: ClientIdentity) -> List[Dict[str, Any]]:
    dependencies: List[Dict[str, Any]] = []
    aliases = (
        ("dataset", ("dataset_version_id", "dataset_id")),
        ("scaler", ("scaler_version_id",)),
        ("pattern_catalog", ("pattern_catalog_version", "catalog_version")),
    )
    for artifact_type, keys in aliases:
        version_id = next((training_summary.get(key) for key in keys if training_summary.get(key)), None)
        if version_id:
            item: Dict[str, Any] = {
                "artifact_type": artifact_type,
                "version_id": str(version_id),
            }
            if artifact_type == "pattern_catalog":
                item["market"] = identity.market
            else:
                item["identity"] = _identity_dict(identity)
            dependencies.append(item)
    return dependencies


def _safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise MemoryIntegrityError(f"Model checkpoint could not be read: {path}") from error
    if not isinstance(checkpoint, dict):
        raise MemoryIntegrityError("Model checkpoint is not a valid dictionary.")
    return checkpoint


def save_model(
    market: str,
    model: nn.Module,
    scaler: Any,
    training_summary: Dict[str, Any],
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    version_id: Optional[str] = None,
    dependencies: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    register_version: bool = True,
    scaler_version_id: Optional[str] = None,
    dataset_version_id: Optional[str] = None,
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module instance.")
    if not isinstance(training_summary, dict):
        raise TypeError("training_summary must be a dictionary.")
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    input_size, hidden_sizes = _model_architecture_metadata(model)
    scaler_payload, scaler_format = _scaler_for_checkpoint(scaler)
    model_version_id = _validate_bundle_id(
        version_id or f"model_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:10]}"
    )
    summary_payload = _to_json_compatible(training_summary)
    if not isinstance(summary_payload, dict):
        raise MemoryValidationError("training_summary must serialize to a dictionary.")

    resolved_dataset_version = dataset_version_id or next(
        (
            summary_payload.get(key)
            for key in ("dataset_version_id", "dataset_id")
            if summary_payload.get(key)
        ),
        None,
    )
    resolved_scaler_version = scaler_version_id or summary_payload.get("scaler_version_id")
    if register_version and not resolved_scaler_version:
        resolved_scaler_version = (
            f"scaler_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:10]}"
        )
        save_scaler_artifact(
            scaler,
            identity=resolved,
            version_id=resolved_scaler_version,
            dataset_version_id=str(resolved_dataset_version) if resolved_dataset_version else None,
            metadata={"created_for_model_version": model_version_id},
        )
    elif register_version and resolved_scaler_version:
        try:
            _version_registry().get_version(
                "scaler",
                str(resolved_scaler_version),
                identity=resolved,
            )
        except Exception as error:
            raise MemoryVersionError(
                "The referenced scaler version is not registered for this TradingBot."
            ) from error

    if resolved_scaler_version:
        summary_payload["scaler_version_id"] = str(resolved_scaler_version)
    if resolved_dataset_version:
        summary_payload["dataset_version_id"] = str(resolved_dataset_version)

    checkpoint = {
        "identity": _identity_dict(resolved),
        "market": resolved.market,
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        **_model_class_metadata(model),
        "timeframes": list(SUPPORTED_TIMEFRAMES),
        "lookback_by_timeframe": dict(LOOKBACK_BY_TIMEFRAME),
        "scaler": scaler_payload,
        "scaler_format": scaler_format,
        "scaler_version_id": str(resolved_scaler_version) if resolved_scaler_version else None,
        "dataset_version_id": str(resolved_dataset_version) if resolved_dataset_version else None,
        "feature_mean": scaler_payload.get("mean"),
        "feature_std": scaler_payload.get("std"),
        "training_summary": summary_payload,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "format_version": MODEL_FORMAT_VERSION,
        "model_version_id": model_version_id,
    }
    directory = model_dir(market, identity=resolved) / "STAGING"
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = directory / f"{model_version_id}.pt"
    if temporary_path.exists():
        raise FileExistsError(f"Model version checkpoint already exists: {temporary_path}")
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{model_version_id}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as temporary_file:
            staged_temp = Path(temporary_file.name)
        torch.save(checkpoint, staged_temp)
        with staged_temp.open("rb") as file_handle:
            os.fsync(file_handle.fileno())
        try:
            os.chmod(staged_temp, 0o600)
        except OSError:
            pass
        os.link(staged_temp, temporary_path)
    except Exception:
        try:
            staged_temp.unlink(missing_ok=True)
        except (UnboundLocalError, OSError):
            pass
        raise
    finally:
        try:
            staged_temp.unlink(missing_ok=True)
        except (UnboundLocalError, OSError):
            pass

    if not register_version:
        legacy_path = _legacy_model_path(resolved)
        _atomic_write_bytes(legacy_path, temporary_path.read_bytes(), immutable=False)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return legacy_path

    dependency_list = list(dependencies or ())
    dependency_list.extend(_checkpoint_dependencies(summary_payload, resolved))
    model_metadata = dict(metadata or {})
    model_metadata.update(
        {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "model_class": _model_class_metadata(model),
            "scaler_format": scaler_format,
            "scaler_version_id": resolved_scaler_version,
            "dataset_version_id": resolved_dataset_version,
            "model_target_schema": training_summary.get("model_target_schema"),
            "training_summary_hash": hashlib.sha256(
                json.dumps(summary_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }
    )
    try:
        registry = _version_registry()
        record = registry.register_model(
            temporary_path,
            identity=resolved,
            version_id=model_version_id,
            dependencies=dependency_list,
            metadata=model_metadata,
        )
    except Exception as error:
        raise MemoryVersionError(
            "The checkpoint was written to staging but could not be registered as an immutable model version."
        ) from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return Path(record.artifact_uri).resolve()


def _load_model_class(checkpoint: Mapping[str, Any]) -> Any:
    module_name = str(checkpoint.get("model_class_module", "engine.model"))
    class_name = str(checkpoint.get("model_class_name", "TradingMicroModel"))
    allowed_modules = {"engine.model", f"{__package__}.model" if __package__ else "engine.model"}
    if module_name not in allowed_modules:
        raise MemoryValidationError("Checkpoint model class module is not on the approved allowlist.")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise RuntimeError("The checkpoint model module is not available.") from error
    model_class = getattr(module, class_name, None)
    if model_class is None or not inspect.isclass(model_class) or not issubclass(model_class, nn.Module):
        raise MemoryValidationError("Checkpoint model class is unavailable or invalid.")
    return model_class


def _instantiate_model(checkpoint: Mapping[str, Any]) -> nn.Module:
    model_class = _load_model_class(checkpoint)
    input_size = int(checkpoint.get("input_size", 0))
    if input_size <= 0:
        raise MemoryIntegrityError("Checkpoint input_size is invalid.")
    hidden_sizes = checkpoint.get("hidden_sizes", [])
    if not isinstance(hidden_sizes, Sequence) or isinstance(hidden_sizes, (str, bytes)):
        raise MemoryIntegrityError("Checkpoint hidden_sizes metadata is invalid.")
    signature = inspect.signature(model_class)
    parameters = signature.parameters
    kwargs: Dict[str, Any] = {}
    if "input_size" in parameters:
        kwargs["input_size"] = input_size
    for argument, value in zip(("hidden_1", "hidden_2", "hidden_3"), hidden_sizes[:3]):
        if argument in parameters:
            kwargs[argument] = int(value)
    try:
        model = model_class(**kwargs)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception as error:
        raise RuntimeError(
            "The saved checkpoint architecture is incompatible with the installed model implementation."
        ) from error
    model.eval()
    return model


def _restore_checkpoint_scaler(checkpoint: Mapping[str, Any]) -> Any:
    payload = checkpoint.get("scaler")
    if isinstance(payload, Mapping) and checkpoint.get("scaler_format") == "engine_scaling":
        module_name = f"{__package__}.scaling" if __package__ else "engine.scaling"
        module = importlib.import_module(module_name)
        return module.scaler_from_serializable(payload)
    if isinstance(payload, Mapping):
        return FeatureScaler(payload.get("mean", []), payload.get("std", []))
    return FeatureScaler(
        np.asarray(checkpoint.get("feature_mean", []), dtype=np.float32),
        np.asarray(checkpoint.get("feature_std", []), dtype=np.float32),
    )


def _load_versioned_checkpoint(
    market: str,
    identity: ClientIdentity,
    version_id: Optional[str],
    require_approved: bool,
) -> Tuple[Path, Dict[str, Any], Optional[Any]]:
    registry = _version_registry()
    if version_id is None:
        record = registry.latest_approved("model", identity=identity)
        if record is None:
            raise FileNotFoundError("No approved model version is registered for this TradingBot.")
    else:
        record = registry.get_version("model", version_id, identity=identity)
        if require_approved and record.status != "APPROVED":
            raise MemoryVersionError("Requested model version is not approved.")
    verification = registry.verify_artifact("model", record.version_id, identity=identity)
    if not verification.get("valid", False):
        raise MemoryIntegrityError("Registered model artifact failed its integrity check.")
    path = Path(record.artifact_uri).resolve()
    checkpoint = _safe_torch_load(path)
    return path, checkpoint, record


def load_model(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    version_id: Optional[str] = None,
    require_approved: Optional[bool] = None,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    if require_approved is None:
        require_approved = True
    checkpoint: Dict[str, Any]
    selected_path: Path
    version_record = None
    try:
        selected_path, checkpoint, version_record = _load_versioned_checkpoint(
            resolved.market,
            resolved,
            version_id,
            bool(require_approved),
        )
    except FileNotFoundError:
        if version_id is not None:
            raise
        selected_path = _legacy_model_path(resolved)
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"No approved or legacy model exists for {resolved.as_dict()}."
            )
        checkpoint = _safe_torch_load(selected_path)
    if checkpoint.get("identity") != _identity_dict(resolved):
        raise PermissionError(
            "Model checkpoint identity does not match the requested TradingBot."
        )
    if str(checkpoint.get("market", "")).strip().upper() != resolved.market:
        raise MemoryIntegrityError("Model checkpoint market does not match its identity.")
    model = _instantiate_model(checkpoint)
    scaler = _restore_checkpoint_scaler(checkpoint)
    scaler_version = checkpoint.get("scaler_version_id")
    if scaler_version:
        registered_scaler = load_scaler_artifact(
            str(scaler_version),
            identity=resolved,
            require_approved=bool(require_approved),
        )
        embedded_mean = np.asarray(scaler.mean, dtype=np.float64).reshape(-1)
        embedded_std = np.asarray(scaler.std, dtype=np.float64).reshape(-1)
        registered_mean = np.asarray(registered_scaler.mean, dtype=np.float64).reshape(-1)
        registered_std = np.asarray(registered_scaler.std, dtype=np.float64).reshape(-1)
        if not np.array_equal(embedded_mean, registered_mean) or not np.array_equal(embedded_std, registered_std):
            raise MemoryIntegrityError("Embedded checkpoint scaler does not match its registered scaler version.")
        scaler = registered_scaler
    if np.asarray(scaler.mean).size != int(checkpoint["input_size"]):
        raise MemoryIntegrityError("Stored scaler width does not match the model input size.")
    checkpoint = dict(checkpoint)
    checkpoint["loaded_from"] = str(selected_path)
    if version_record is not None:
        checkpoint["version_record"] = version_record.to_dict()
        checkpoint["model_version_id"] = version_record.version_id
        checkpoint["model_version_status"] = version_record.status
    return model, scaler, checkpoint


# ============================================================================
# VERSION LIFECYCLE HELPERS
# ============================================================================

def validate_artifact_version(artifact_type: str, version_id: str, **kwargs: Any) -> Any:
    return _version_registry().validate_version(artifact_type, version_id, **kwargs)


def approve_artifact_version(artifact_type: str, version_id: str, **kwargs: Any) -> Any:
    normalized_type = str(artifact_type).strip().lower()
    if normalized_type == "model":
        identity = kwargs.pop("identity", None)
        if identity is None:
            raise MemoryValidationError("Model approval requires the complete TradingBot identity.")
        validation_metadata = kwargs.pop("validation_metadata", None)
        reason = kwargs.pop("reason", "Model passed its explicit evaluation gate.")
        if kwargs:
            raise TypeError("Unsupported model approval arguments: " + ", ".join(sorted(kwargs)))
        return approve_model_version(
            version_id,
            identity=identity,
            validation_metadata=validation_metadata,
            reason=reason,
        )
    return _version_registry().approve_version(artifact_type, version_id, **kwargs)


def deprecate_artifact_version(artifact_type: str, version_id: str, **kwargs: Any) -> Any:
    return _version_registry().deprecate_version(artifact_type, version_id, **kwargs)


def approve_model_version(
    version_id: str,
    *,
    identity: ClientIdentity,
    validation_metadata: Optional[Mapping[str, Any]] = None,
    reason: str = "Model passed its explicit evaluation gate.",
) -> Any:
    if not isinstance(validation_metadata, Mapping):
        raise MemoryValidationError(
            "Model approval requires explicit evaluation metadata with a positive pass flag."
        )
    passed = validation_metadata.get("passed") is True or validation_metadata.get("evaluation_passed") is True
    if not passed:
        raise MemoryValidationError("Model approval is blocked because evaluation did not explicitly pass.")

    registry = _version_registry()
    record = registry.get_version("model", version_id, identity=identity)
    target_schema = record.metadata.get("model_target_schema")
    if str(target_schema or "").strip().upper() not in {"TP_BEFORE_SL", "TP_BEFORE_SL_BINARY"}:
        raise MemoryValidationError(
            "Only a model trained for the TP_BEFORE_SL binary target schema can be approved."
        )
    if record.status == "DRAFT":
        registry.validate_version(
            "model",
            version_id,
            identity=identity,
            validation_metadata=dict(validation_metadata),
        )
    return registry.approve_version("model", version_id, identity=identity, reason=reason)


def list_model_versions(*, identity: ClientIdentity, status: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        record.to_dict()
        for record in _version_registry().list_versions("model", identity=identity, status=status)
    ]


def latest_approved_model_version(*, identity: ClientIdentity) -> Optional[Dict[str, Any]]:
    record = _version_registry().latest_approved("model", identity=identity)
    return record.to_dict() if record is not None else None


def verify_model_version(version_id: str, *, identity: ClientIdentity) -> Dict[str, Any]:
    return _version_registry().verify_artifact("model", version_id, identity=identity)


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "STORAGE_SCHEMA_VERSION",
    "MODEL_FORMAT_VERSION",
    "BUNDLE_STORAGE_DIRECTORY",
    "DATASET_STORAGE_DIRECTORY",
    "SCALER_STORAGE_DIRECTORY",
    "MemoryErrorBase",
    "MemoryValidationError",
    "MemoryIntegrityError",
    "MemoryVersionError",
    "FeatureScaler",
    "model_dir",
    "model_path",
    "training_bundle_path",
    "fit_scaler",
    "apply_scaler",
    "fit_training_dataset_scaler",
    "apply_training_dataset_scaler",
    "save_training_data_bundle",
    "load_training_data_bundle",
    "list_training_data_bundles",
    "delete_training_data_bundle",
    "prepare_and_store_training_data",
    "save_training_dataset",
    "load_training_dataset",
    "list_dataset_versions",
    "save_scaler_artifact",
    "load_scaler_artifact",
    "save_model",
    "load_model",
    "validate_artifact_version",
    "approve_artifact_version",
    "deprecate_artifact_version",
    "approve_model_version",
    "list_model_versions",
    "latest_approved_model_version",
    "verify_model_version",
]
