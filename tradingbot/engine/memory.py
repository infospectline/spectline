"""Identity-isolated storage and memory utilities for the Spectline AI engine."""

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
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    from engine.settings import (
        LOOKBACK_BY_TIMEFRAME,
        MODEL_FILENAME,
        SUPPORTED_TIMEFRAMES,
    )
except ImportError:
    from ..client import (
        ClientIdentity,
        get_model_dir,
        get_model_path,
        get_results_dir,
        resolve_identity,
    )
    from .settings import (
        LOOKBACK_BY_TIMEFRAME,
        MODEL_FILENAME,
        SUPPORTED_TIMEFRAMES,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

STORAGE_SCHEMA_VERSION = 1
MODEL_FORMAT_VERSION = 3
BUNDLE_STORAGE_DIRECTORY = Path("AI_PREPARATION") / "TRAINING_BUNDLES"
DEFAULT_BUNDLE_SUFFIX = ".json.gz"
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class FeatureScaler:
    """Feature standardization statistics persisted with a model checkpoint."""

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


# ============================================================================
# IDENTITY AND PATH HELPERS
# ============================================================================

def _resolve_memory_identity(
    market: str,
    identity: Optional[ClientIdentity] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Resolve the complete customer, market, strategy and TradingBot identity."""
    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")

        normalized_market = str(market).strip().upper().replace("/", "_").replace("\\", "_")
        if identity.market != normalized_market:
            raise ValueError(
                f"Identity market {identity.market} does not match {normalized_market}."
            )
        return identity

    return resolve_identity(
        market=market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def _resolve_identity_from_bundle(
    bundle: Mapping[str, Any],
    identity: Optional[ClientIdentity] = None,
) -> ClientIdentity:
    """Resolve an identity from a bundle or validate an explicitly supplied one."""
    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        stored_identity = bundle.get("identity")
        if stored_identity != identity.as_dict():
            raise ValueError("Training Data Bundle identity does not match the requested TradingBot.")
        return identity

    stored_identity = bundle.get("identity")
    if not isinstance(stored_identity, Mapping):
        raise ValueError("Training Data Bundle is missing its complete identity.")

    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [key for key in required if not stored_identity.get(key)]
    if missing:
        raise ValueError(
            "Training Data Bundle identity is incomplete: " + ", ".join(missing)
        )

    return resolve_identity(
        market=str(stored_identity["market"]),
        user_id=str(stored_identity["user_id"]),
        strategy_id=str(stored_identity["strategy_id"]),
        tradingbot_id=str(stored_identity["tradingbot_id"]),
    )


def model_dir(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the model directory for one exact TradingBot identity."""
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return get_model_dir(resolved)


def model_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Return the model checkpoint path for one exact TradingBot identity."""
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    return get_model_path(resolved)


def _bundle_directory(identity: ClientIdentity) -> Path:
    """Return the private prepared-data directory for one TradingBot."""
    return get_results_dir(identity) / BUNDLE_STORAGE_DIRECTORY


def _validate_bundle_id(bundle_id: str) -> str:
    """Validate a bundle identifier before using it in a file path."""
    value = str(bundle_id).strip()
    if not _SAFE_COMPONENT_PATTERN.fullmatch(value):
        raise ValueError("Invalid training bundle ID.")
    if value.endswith(DEFAULT_BUNDLE_SUFFIX):
        value = value[: -len(DEFAULT_BUNDLE_SUFFIX)]
    if not value or value in {".", ".."}:
        raise ValueError("Invalid training bundle ID.")
    return value


def training_bundle_path(
    identity: ClientIdentity,
    bundle_id: str,
) -> Path:
    """Return the path for one stored Training Data Bundle."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")
    safe_id = _validate_bundle_id(bundle_id)
    return _bundle_directory(identity) / f"{safe_id}{DEFAULT_BUNDLE_SUFFIX}"


# ============================================================================
# FEATURE STANDARDIZATION
# ============================================================================

def fit_scaler(features: np.ndarray) -> FeatureScaler:
    """Fit a standard scaler over a two-dimensional feature matrix."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features cannot be empty.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contain NaN or infinite values.")

    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return FeatureScaler(mean=mean, std=std)


def apply_scaler(features: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    """Apply stored standardization statistics to one or more feature rows."""
    if not isinstance(scaler, FeatureScaler):
        raise TypeError("scaler must be a FeatureScaler instance.")

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("features must be a one- or two-dimensional array.")
    if matrix.shape[1] != scaler.mean.size:
        raise ValueError(
            f"Feature width {matrix.shape[1]} does not match scaler width {scaler.mean.size}."
        )

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    transformed = (matrix - scaler.mean) / scaler.std
    return np.nan_to_num(
        transformed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


# ============================================================================
# SAFE TRAINING DATA SERIALIZATION
# ============================================================================

def _to_json_compatible(value: Any) -> Any:
    """Convert supported engine objects into a lossless JSON-compatible form."""
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Training data contains a non-finite float.")
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
        return {
            "__spectline_type__": "path",
            "value": str(value),
        }

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
            return {
                key: _to_json_compatible(child)
                for key, child in items
            }
        return {
            "__spectline_type__": "mapping",
            "items": [
                [_to_json_compatible(key), _to_json_compatible(child)]
                for key, child in items
            ],
        }

    if isinstance(value, tuple):
        return {
            "__spectline_type__": "tuple",
            "items": [_to_json_compatible(item) for item in value],
        }

    if isinstance(value, (list, set, frozenset)):
        converted = [_to_json_compatible(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return {"__spectline_type__": "set", "items": converted}
        return converted

    raise TypeError(
        f"Unsupported value in Training Data Bundle: {type(value).__name__}."
    )


def _restore_json_compatible(value: Any) -> Any:
    """Restore supported engine dataclasses and typed containers from JSON."""
    if isinstance(value, list):
        return [_restore_json_compatible(item) for item in value]

    if not isinstance(value, dict):
        return value

    marker = value.get("__spectline_type__")
    if marker is None:
        return {
            key: _restore_json_compatible(child)
            for key, child in value.items()
        }

    if marker == "ndarray":
        array = np.asarray(value.get("data", []), dtype=value.get("dtype", "float32"))
        expected_shape = tuple(int(item) for item in value.get("shape", array.shape))
        try:
            return array.reshape(expected_shape)
        except ValueError as error:
            raise ValueError("Stored NumPy array has an invalid shape.") from error

    if marker == "path":
        return Path(str(value.get("value", "")))

    if marker == "tuple":
        return tuple(_restore_json_compatible(item) for item in value.get("items", []))

    if marker == "set":
        return set(_restore_json_compatible(item) for item in value.get("items", []))

    if marker == "mapping":
        result: Dict[Any, Any] = {}
        for pair in value.get("items", []):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Stored mapping contains an invalid item.")
            key = _restore_json_compatible(pair[0])
            result[key] = _restore_json_compatible(pair[1])
        return result

    if marker == "dataclass":
        class_name = str(value.get("class_name", ""))
        raw_fields = value.get("fields", {})
        if not isinstance(raw_fields, dict):
            raise ValueError("Stored dataclass fields are malformed.")

        allowed_classes = {"Candle", "ManualZone", "UserSetup", "FeatureScaler"}
        if class_name not in allowed_classes:
            raise ValueError(f"Unsupported stored dataclass type: {class_name}.")

        try:
            if class_name == "FeatureScaler":
                restored_fields = {
                    key: _restore_json_compatible(child)
                    for key, child in raw_fields.items()
                }
                return FeatureScaler(**restored_fields)

            module_name = f"{__package__}.data" if __package__ else "engine.data"
            data_module = importlib.import_module(module_name)
            data_class = getattr(data_module, class_name)
            restored_fields = {
                key: _restore_json_compatible(child)
                for key, child in raw_fields.items()
            }
            return data_class(**restored_fields)
        except (ImportError, AttributeError, TypeError) as error:
            raise RuntimeError(
                f"Could not restore stored {class_name} data."
            ) from error

    raise ValueError(f"Unknown Training Data Bundle serialization marker: {marker}.")


def _canonical_json(value: Any) -> str:
    """Return deterministic JSON text for hashing and storage."""
    return json.dumps(
        _to_json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# ============================================================================
# TRAINING DATA BUNDLE STORAGE
# ============================================================================

def save_training_data_bundle(
    bundle: Mapping[str, Any],
    *,
    identity: Optional[ClientIdentity] = None,
    bundle_id: Optional[str] = None,
) -> Path:
    """Persist a prepared Training Data Bundle as compressed, identity-scoped JSON."""
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping.")

    resolved = _resolve_identity_from_bundle(bundle, identity)
    payload = _canonical_json(dict(bundle))
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if bundle_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_id = f"bundle_{timestamp}_{uuid.uuid4().hex[:8]}"
    else:
        safe_id = _validate_bundle_id(bundle_id)

    target_path = training_bundle_path(resolved, safe_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "bundle_id": safe_id,
        "identity": resolved.as_dict(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": payload_hash,
        "bundle": json.loads(payload),
    }

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{safe_id}.",
            suffix=".tmp",
            dir=target_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as compressed_file:
            json.dump(
                envelope,
                compressed_file,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass

        with temporary_path.open("rb") as completed_file:
            os.fsync(completed_file.fileno())

        os.replace(temporary_path, target_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return target_path


def load_training_data_bundle(
    bundle_id: str,
    *,
    identity: ClientIdentity,
) -> Dict[str, Any]:
    """Load and identity-check one previously stored Training Data Bundle."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    path = training_bundle_path(identity, bundle_id)
    if not path.is_file():
        raise FileNotFoundError(f"Training Data Bundle was not found: {path}")

    try:
        with gzip.open(path, "rt", encoding="utf-8") as compressed_file:
            envelope = json.load(compressed_file)
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise ValueError(f"Stored Training Data Bundle is unreadable: {path}") from error

    if not isinstance(envelope, dict):
        raise ValueError("Stored Training Data Bundle envelope is invalid.")

    if envelope.get("storage_schema_version") != STORAGE_SCHEMA_VERSION:
        raise ValueError(
            "Stored Training Data Bundle uses an unsupported storage schema version."
        )

    if envelope.get("identity") != identity.as_dict():
        raise PermissionError(
            "Stored Training Data Bundle identity does not match the requested TradingBot."
        )

    encoded_bundle = envelope.get("bundle")
    if not isinstance(encoded_bundle, dict):
        raise ValueError("Stored Training Data Bundle payload is invalid.")

    canonical_payload = json.dumps(
        encoded_bundle,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    if actual_hash != envelope.get("payload_sha256"):
        raise ValueError("Stored Training Data Bundle failed its integrity check.")

    restored = _restore_json_compatible(encoded_bundle)
    if not isinstance(restored, dict):
        raise ValueError("Restored Training Data Bundle payload is not a dictionary.")
    return restored


def list_training_data_bundles(
    *,
    identity: ClientIdentity,
) -> List[Dict[str, Any]]:
    """List saved Training Data Bundles belonging to one TradingBot."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    directory = _bundle_directory(identity)
    if not directory.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    for path in sorted(directory.glob(f"*{DEFAULT_BUNDLE_SUFFIX}")):
        try:
            stat = path.stat()
        except OSError:
            continue

        results.append(
            {
                "bundle_id": path.name[: -len(DEFAULT_BUNDLE_SUFFIX)],
                "filename": path.name,
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )

    return results


def delete_training_data_bundle(
    bundle_id: str,
    *,
    identity: ClientIdentity,
) -> bool:
    """Delete one stored Training Data Bundle without touching other customer data."""
    if not isinstance(identity, ClientIdentity):
        raise TypeError("identity must be a ClientIdentity instance.")

    path = training_bundle_path(identity, bundle_id)
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
    """Prepare and persist a bundle without starting model training."""
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
    stored_path = save_training_data_bundle(
        bundle,
        identity=resolved,
        bundle_id=bundle_id,
    )
    return bundle, stored_path


# ============================================================================
# MODEL CHECKPOINT STORAGE
# ============================================================================

def _model_architecture_metadata(model: nn.Module) -> Tuple[int, List[int]]:
    """Extract the model dimensions required by the current model factory."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("The model must expose a 'backbone' module for checkpointing.")

    linear_layers = [
        module
        for module in backbone.modules()
        if isinstance(module, nn.Linear)
    ]
    if len(linear_layers) < 3:
        raise ValueError(
            "The model backbone must contain at least three Linear layers."
        )

    input_size = int(linear_layers[0].in_features)
    hidden_sizes = [int(layer.out_features) for layer in linear_layers[:3]]
    return input_size, hidden_sizes


def save_model(
    market: str,
    model: nn.Module,
    scaler: FeatureScaler,
    training_summary: Dict[str, Any],
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    """Save a model checkpoint under the exact TradingBot identity."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module instance.")
    if not isinstance(scaler, FeatureScaler):
        raise TypeError("scaler must be a FeatureScaler instance.")
    if not isinstance(training_summary, dict):
        raise TypeError("training_summary must be a dictionary.")

    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )

    directory = model_dir(market, identity=resolved)
    directory.mkdir(parents=True, exist_ok=True)
    path = model_path(market, identity=resolved)
    input_size, hidden_sizes = _model_architecture_metadata(model)

    checkpoint = {
        "identity": resolved.as_dict(),
        "market": resolved.market,
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "timeframes": list(SUPPORTED_TIMEFRAMES),
        "lookback_by_timeframe": dict(LOOKBACK_BY_TIMEFRAME),
        "feature_mean": scaler.mean.tolist(),
        "feature_std": scaler.std.tolist(),
        "training_summary": training_summary,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "format_version": MODEL_FORMAT_VERSION,
    }

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        torch.save(checkpoint, temporary_path)
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return path


def _load_model_class() -> Any:
    """Import the model class lazily to avoid a module-level dependency cycle."""
    module_name = f"{__package__}.model" if __package__ else "engine.model"
    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or error.name == "engine.model":
            raise RuntimeError(
                "engine.model is not implemented yet; model loading is unavailable."
            ) from error
        raise

    model_class = getattr(model_module, "TradingMicroModel", None)
    if model_class is None:
        raise RuntimeError("engine.model.TradingMicroModel is not implemented yet.")
    return model_class


def load_model(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Tuple[nn.Module, FeatureScaler, Dict[str, Any]]:
    """Load and identity-check the model checkpoint for one TradingBot."""
    resolved = _resolve_memory_identity(
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    path = model_path(market, identity=resolved)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model does not exist for {resolved.as_dict()}: {path}"
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Model checkpoint could not be read: {path}") from error

    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint is not a valid dictionary.")

    if checkpoint.get("identity") != resolved.as_dict():
        raise ValueError(
            "Model checkpoint identity does not match the requested TradingBot. "
            "Legacy checkpoints without a full identity must be migrated or retrained."
        )

    if str(checkpoint.get("market", "")).strip().upper() != resolved.market:
        raise ValueError("Model checkpoint market does not match its identity.")

    input_size = int(checkpoint["input_size"])
    hidden_sizes = checkpoint.get("hidden_sizes", [256, 128, 64])
    if not isinstance(hidden_sizes, Sequence) or len(hidden_sizes) < 3:
        raise ValueError("Model checkpoint hidden_sizes metadata is invalid.")

    model_class = _load_model_class()
    constructor_parameters = inspect.signature(model_class).parameters
    constructor_kwargs: Dict[str, Any] = {"input_size": input_size}
    for argument, value in zip(
        ("hidden_1", "hidden_2", "hidden_3"),
        hidden_sizes[:3],
    ):
        if argument in constructor_parameters:
            constructor_kwargs[argument] = int(value)

    try:
        model = model_class(**constructor_kwargs)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except Exception as error:
        raise RuntimeError(
            "The saved checkpoint architecture is incompatible with the current "
            "engine.model implementation. Keep the old checkpoint as a backup "
            "and train or migrate a compatible model version."
        ) from error

    model.eval()
    scaler = FeatureScaler(
        mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
    )
    if scaler.mean.size != input_size:
        raise ValueError("Stored scaler width does not match the model input size.")

    return model, scaler, checkpoint


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "STORAGE_SCHEMA_VERSION",
    "MODEL_FORMAT_VERSION",
    "BUNDLE_STORAGE_DIRECTORY",
    "FeatureScaler",
    "model_dir",
    "model_path",
    "training_bundle_path",
    "fit_scaler",
    "apply_scaler",
    "save_training_data_bundle",
    "load_training_data_bundle",
    "list_training_data_bundles",
    "delete_training_data_bundle",
    "prepare_and_store_training_data",
    "save_model",
    "load_model",
]
