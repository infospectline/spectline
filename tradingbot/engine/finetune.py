"""Controlled, immutable fine-tuning for approved Spectline TP-before-SL models."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import hashlib
import importlib
import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# MODULE DEPENDENCIES
# ============================================================================

try:
    from . import memory as memory_engine
    from . import model as model_engine
    from . import scaling as scaling_engine
    from . import versions as versions_engine
except ImportError:
    try:
        from engine import memory as memory_engine
        from engine import model as model_engine
        from engine import scaling as scaling_engine
        from engine import versions as versions_engine
    except ImportError:
        memory_engine = None
        model_engine = None
        scaling_engine = None
        versions_engine = None

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
            if memory_engine is None:
                raise ImportError("The project client.py identity resolver is unavailable.")
            values = {
                "user_id": user_id,
                "market": normalize_market(market),
                "strategy_id": strategy_id,
                "tradingbot_id": tradingbot_id,
            }
            missing = [key for key, value in values.items() if value is None or not str(value).strip()]
            if missing:
                raise ValueError("A complete TradingBot identity is required: " + ", ".join(missing))
            return memory_engine.ClientIdentity(**{key: str(value) for key, value in values.items()})


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

MODULE_SCHEMA_VERSION = 1
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET_NAME = "TP_BEFORE_SL"
FINETUNE_RUN_ARTIFACT_TYPE = "training_run"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BACKBONE_LR_MULTIPLIER = 0.25
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_REPLAY_SAMPLES_PER_CLASS = 256
DEFAULT_HEAD_ONLY_EPOCHS = 2
DEFAULT_VALIDATION_LOSS_TOLERANCE = 0.01
DEFAULT_MAX_GRADIENT_NORM = 0.5
DEFAULT_SEED = 42


# ============================================================================
# ERRORS AND CONFIGURATION
# ============================================================================

class FinetuneError(RuntimeError):
    """Base exception for controlled fine-tuning operations."""


class FinetuneValidationError(FinetuneError, ValueError):
    """Raised when a fine-tuning request violates the model lifecycle contract."""


class FinetuneDependencyError(FinetuneError, ImportError):
    """Raised when a required Spectline engine module is unavailable."""


class FinetuneIntegrityError(FinetuneError):
    """Raised when an immutable dependency fails an integrity check."""


class FinetuneInProgressError(FinetuneError):
    """Raised when another fine-tuning run is active for the same TradingBot."""


@dataclass(frozen=True)
class FinetuneConfig:
    """Validated settings for one incremental model update."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    backbone_lr_multiplier: float = DEFAULT_BACKBONE_LR_MULTIPLIER
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA
    replay_samples_per_class: int = DEFAULT_REPLAY_SAMPLES_PER_CLASS
    head_only_epochs: int = DEFAULT_HEAD_ONLY_EPOCHS
    validation_loss_tolerance: float = DEFAULT_VALIDATION_LOSS_TOLERANCE
    max_gradient_norm: float = DEFAULT_MAX_GRADIENT_NORM
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        integer_fields = ("epochs", "batch_size", "patience", "replay_samples_per_class", "head_only_epochs", "seed")
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise FinetuneValidationError(f"{field_name} must be an integer.")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise FinetuneValidationError(f"{field_name} must be an integer.") from error
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise FinetuneValidationError(f"{field_name} must be an integer.")
            if field_name in {"epochs", "batch_size"} and numeric <= 0:
                raise FinetuneValidationError(f"{field_name} must be positive.")
            if field_name in {"patience", "replay_samples_per_class", "head_only_epochs", "seed"} and numeric < 0:
                raise FinetuneValidationError(f"{field_name} cannot be negative.")
            object.__setattr__(self, field_name, int(numeric))
        if self.head_only_epochs > self.epochs:
            raise FinetuneValidationError("head_only_epochs cannot exceed epochs.")

        numeric_fields = (
            "learning_rate",
            "backbone_lr_multiplier",
            "weight_decay",
            "min_delta",
            "validation_loss_tolerance",
            "max_gradient_norm",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise FinetuneValidationError(f"{field_name} must be numeric.")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise FinetuneValidationError(f"{field_name} must be numeric.") from error
            if not math.isfinite(numeric):
                raise FinetuneValidationError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, numeric)
        if self.learning_rate <= 0.0:
            raise FinetuneValidationError("learning_rate must be positive.")
        if self.backbone_lr_multiplier < 0.0:
            raise FinetuneValidationError("backbone_lr_multiplier cannot be negative.")
        if self.weight_decay < 0.0 or self.min_delta < 0.0:
            raise FinetuneValidationError("weight_decay and min_delta cannot be negative.")
        if self.validation_loss_tolerance < 0.0:
            raise FinetuneValidationError("validation_loss_tolerance cannot be negative.")
        if self.max_gradient_norm <= 0.0:
            raise FinetuneValidationError("max_gradient_norm must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# IDENTITY, TIME AND SERIALIZATION HELPERS
# ============================================================================

_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_identifier(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 128 or normalized in {".", ".."}:
        raise FinetuneValidationError(f"{name} must be a non-empty safe identifier.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-" for character in normalized):
        raise FinetuneValidationError(f"{name} contains unsupported characters.")
    return normalized


def _generated_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:12]}"


def _identity_dict(identity: Any) -> Dict[str, str]:
    if isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        method = getattr(identity, "as_dict", None)
        if not callable(method):
            raise FinetuneValidationError("identity must be a mapping or expose as_dict().")
        raw = method()
    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [field for field in required if raw.get(field) is None or not str(raw[field]).strip()]
    if missing:
        raise FinetuneValidationError("A complete TradingBot identity is required: " + ", ".join(missing))
    return {
        "user_id": str(raw["user_id"]).strip(),
        "market": normalize_market(raw["market"]),
        "strategy_id": str(raw["strategy_id"]).strip(),
        "tradingbot_id": str(raw["tradingbot_id"]).strip(),
    }


def _resolve_identity(
    market: str,
    identity: Any = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Any:
    _require_dependencies()
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if callable(resolver):
        return resolver(
            normalize_market(market),
            identity=identity,
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
    if identity is not None:
        values = _identity_dict(identity)
        if values["market"] != normalize_market(market):
            raise FinetuneValidationError("Identity market does not match the requested market.")
        return resolve_identity(
            market=values["market"],
            user_id=values["user_id"],
            strategy_id=values["strategy_id"],
            tradingbot_id=values["tradingbot_id"],
        )
    return resolve_identity(
        market=normalize_market(market),
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def _require_dependencies() -> None:
    missing = [
        name
        for module, name in (
            (memory_engine, "engine.memory"),
            (model_engine, "engine.model"),
            (scaling_engine, "engine.scaling"),
            (versions_engine, "engine.versions"),
        )
        if module is None
    ]
    if missing:
        raise FinetuneDependencyError("Required engine modules are unavailable: " + ", ".join(missing))


def _identity_lock_key(identity: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_lock(identity: Mapping[str, str]) -> threading.Lock:
    key = _identity_lock_key(identity)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def _record_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    method = getattr(record, "to_dict", None)
    if callable(method):
        result = method()
        if isinstance(result, Mapping):
            return dict(result)
    raise FinetuneValidationError("Version registry returned an invalid record.")


def _version_status(record: Any) -> str:
    if isinstance(record, Mapping):
        value = record.get("status")
    else:
        value = getattr(record, "status", None)
    return str(value or "").strip().upper()


def _parse_time(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise FinetuneValidationError(f"{name} must be a valid Unix timestamp.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FinetuneValidationError(f"{name} must be a valid Unix timestamp.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise FinetuneValidationError(f"{name} must be a finite integer timestamp.")
    return int(numeric)


def _sha256_indices(indices: Sequence[int]) -> str:
    payload = json.dumps([int(item) for item in indices], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ============================================================================
# DATASET AND SCALER VALIDATION
# ============================================================================

def _extract_feature_hash(dataset: Mapping[str, Any]) -> Optional[str]:
    schema = dataset.get("feature_schema", {})
    value = schema.get("sha256") if isinstance(schema, Mapping) else None
    value = value or dataset.get("feature_schema_hash")
    return str(value) if value else None


def _dataset_identity(dataset: Mapping[str, Any]) -> Dict[str, str]:
    return _identity_dict(dataset.get("identity"))


def _normalise_dataset(dataset: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    if not isinstance(dataset, Mapping):
        raise FinetuneValidationError(f"{label} dataset must be a mapping.")
    normalizer = getattr(model_engine, "_normalize_training_dataset", None)
    if not callable(normalizer):
        raise FinetuneDependencyError("engine.model._normalize_training_dataset is unavailable.")
    normalized = normalizer(dataset)
    if not isinstance(normalized, Mapping):
        raise FinetuneValidationError(f"{label} dataset normalization returned an invalid object.")
    if dataset.get("target") != DATASET_TARGET_NAME:
        raise FinetuneValidationError(f"{label} dataset does not use the TP_BEFORE_SL target.")
    declared_schema = str(dataset.get("model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if declared_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise FinetuneValidationError(f"{label} dataset target schema is incompatible.")
    validation_indices = np.asarray(normalized.get("validation_indices", []), dtype=np.int64)
    if validation_indices.size == 0:
        raise FinetuneValidationError(f"{label} dataset requires a non-empty validation split.")
    features = np.asarray(normalized["features"], dtype=np.float32)
    targets = np.asarray(normalized["targets"], dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise FinetuneValidationError(f"{label} dataset feature and target shapes are inconsistent.")
    if not np.all(np.isfinite(features)) or not np.all(np.isin(targets, (0, 1))):
        raise FinetuneValidationError(f"{label} dataset contains invalid numeric data or targets.")
    return dict(normalized)


def _sample_times(dataset: Mapping[str, Any], indices: Sequence[int]) -> List[int]:
    rows = dataset.get("sample_metadata", [])
    result: List[int] = []
    for index in indices:
        try:
            result.append(_parse_time(rows[int(index)]["decision_time"], "decision_time"))
        except (IndexError, KeyError, TypeError) as error:
            raise FinetuneValidationError("Dataset sample metadata is missing decision_time values.") from error
    return result


def _outcome_known_times(dataset: Mapping[str, Any], indices: Sequence[int]) -> List[int]:
    rows = dataset.get("label_metadata", [])
    result: List[int] = []
    for index in indices:
        try:
            result.append(_parse_time(rows[int(index)]["outcome_known_at"], "outcome_known_at"))
        except (IndexError, KeyError, TypeError) as error:
            raise FinetuneValidationError("Dataset label metadata is missing outcome_known_at values.") from error
    return result


def _trade_keys(dataset: Mapping[str, Any], indices: Sequence[int]) -> set[str]:
    rows = dataset.get("sample_metadata", [])
    keys: set[str] = set()
    for index in indices:
        row = rows[int(index)]
        if not isinstance(row, Mapping):
            raise FinetuneValidationError("Dataset sample metadata contains an invalid row.")
        value = row.get("trade_id") or row.get("setup_id")
        if value is not None and str(value).strip():
            keys.add(str(value).strip())
    return keys


def _validate_dataset_identity(dataset: Mapping[str, Any], identity: Mapping[str, str], label: str) -> None:
    if _dataset_identity(dataset) != dict(identity):
        raise FinetuneValidationError(f"{label} dataset belongs to another customer or TradingBot.")


def _validate_feature_contract(
    parent_dataset: Mapping[str, Any],
    new_dataset: Mapping[str, Any],
    scaler: Any,
    model: Any,
) -> Dict[str, Any]:
    parent_names = tuple(str(item) for item in parent_dataset.get("feature_names", ()))
    new_names = tuple(str(item) for item in new_dataset.get("feature_names", ()))
    if not parent_names or parent_names != new_names:
        raise FinetuneValidationError("Fine-tuning requires exactly the same ordered feature names as the parent model.")
    parent_hash = _extract_feature_hash(parent_dataset)
    new_hash = _extract_feature_hash(new_dataset)
    scaler_hash = getattr(scaler, "feature_schema_hash", None)
    if not parent_hash or not new_hash or parent_hash != new_hash:
        raise FinetuneValidationError("Parent and incremental dataset feature schema hashes must match.")
    if scaler_hash and str(scaler_hash) != new_hash:
        raise FinetuneValidationError("Parent scaler feature schema does not match the incremental dataset.")
    scaler_names = tuple(str(item) for item in getattr(scaler, "feature_names", ()) or ())
    if scaler_names and scaler_names != new_names:
        raise FinetuneValidationError("Parent scaler feature order does not match the incremental dataset.")
    feature_count = int(np.asarray(new_dataset["features"]).shape[1])
    if int(getattr(model, "input_size", -1)) != feature_count:
        raise FinetuneValidationError("Incremental dataset feature count does not match the parent model input size.")
    scaler_count = int(getattr(scaler, "feature_count", np.asarray(getattr(scaler, "mean", [])).size))
    if scaler_count != feature_count:
        raise FinetuneValidationError("Parent scaler width does not match the incremental dataset.")
    if str(getattr(scaler, "fit_scope", "")).upper() != "TRAIN_SPLIT_ONLY":
        raise FinetuneValidationError("Parent scaler was not fitted exclusively on its original training split.")
    return {
        "feature_count": feature_count,
        "feature_schema_hash": new_hash,
        "feature_names": list(new_names),
    }


def _validate_incremental_chronology(
    parent_dataset: Mapping[str, Any],
    new_dataset: Mapping[str, Any],
    parent_normalized: Mapping[str, Any],
    new_normalized: Mapping[str, Any],
) -> Dict[str, Any]:
    parent_train = np.asarray(parent_normalized["train_indices"], dtype=np.int64)
    new_train = np.asarray(new_normalized["train_indices"], dtype=np.int64)
    parent_decisions = _sample_times(parent_dataset, parent_train)
    new_decisions = _sample_times(new_dataset, new_train)
    parent_known = _outcome_known_times(parent_dataset, parent_train)
    new_known = _outcome_known_times(new_dataset, new_train)
    parent_latest_decision = max(parent_decisions)
    new_earliest_decision = min(new_decisions)
    parent_latest_known = max(parent_known)
    if new_earliest_decision <= parent_latest_decision:
        raise FinetuneValidationError(
            "Incremental training decisions must start strictly after the parent model's training decisions."
        )
    if parent_latest_known >= new_earliest_decision:
        raise FinetuneValidationError(
            "All parent training outcomes must be known before the first incremental decision."
        )
    overlap = _trade_keys(parent_dataset, parent_train).intersection(_trade_keys(new_dataset, new_train))
    if overlap:
        raise FinetuneValidationError("Incremental training data contains trades already used by the parent model.")
    if max(new_known) >= min(_sample_times(new_dataset, new_normalized["validation_indices"])):
        raise FinetuneValidationError("Incremental training outcomes must be known before its validation period starts.")
    return {
        "parent_training_decision_range": {
            "start": min(parent_decisions),
            "end": parent_latest_decision,
        },
        "parent_training_outcome_known_at_max": parent_latest_known,
        "incremental_training_decision_range": {
            "start": new_earliest_decision,
            "end": max(new_decisions),
        },
        "incremental_training_outcome_known_at_max": max(new_known),
        "new_training_sample_count": int(len(new_train)),
        "new_validation_sample_count": int(len(new_normalized["validation_indices"])),
    }


# ============================================================================
# REPLAY SELECTION AND MODEL HELPERS
# ============================================================================

def _select_replay_indices(
    targets: np.ndarray,
    train_indices: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    if samples_per_class <= 0:
        return np.empty(0, dtype=np.int64)
    selected: List[int] = []
    for class_value in (0, 1):
        candidates = train_indices[targets[train_indices] == class_value]
        if candidates.size == 0:
            continue
        if candidates.size <= samples_per_class:
            chosen = candidates
        else:
            positions = np.linspace(0, candidates.size - 1, num=samples_per_class, dtype=np.int64)
            chosen = candidates[np.unique(positions)]
        selected.extend(int(item) for item in chosen)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _predict_logits(model: Any, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        outputs = model(tensor)
        if isinstance(outputs, Mapping):
            logits = outputs.get("tp_before_sl_logit", outputs.get("logits"))
        else:
            logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise FinetuneValidationError("Model forward pass did not return TP-before-SL logits.")
        result = logits.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise FinetuneValidationError("Model produced non-finite logits during fine-tuning.")
    return result


def _log_loss_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    selected = np.asarray(logits, dtype=np.float64).reshape(-1)
    actual = np.asarray(targets, dtype=np.float64).reshape(-1)
    if selected.size == 0 or selected.size != actual.size:
        raise FinetuneValidationError("Log-loss inputs must be non-empty and have equal lengths.")
    if not np.all(np.isfinite(selected)) or not np.all(np.isin(actual, (0.0, 1.0))):
        raise FinetuneValidationError("Log-loss inputs contain invalid values.")
    clipped_logits = np.clip(selected, -60.0, 60.0)
    return float(np.mean(np.logaddexp(0.0, clipped_logits) - actual * clipped_logits))


def _set_backbone_trainable(model: Any, trainable: bool) -> None:
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise FinetuneValidationError("Parent model does not expose the expected backbone module.")
    for parameter in backbone.parameters():
        parameter.requires_grad_(bool(trainable))


def _train_candidate(
    parent_model: Any,
    scaler: Any,
    new_dataset: Mapping[str, Any],
    new_normalized: Mapping[str, Any],
    parent_dataset: Mapping[str, Any],
    parent_normalized: Mapping[str, Any],
    config: FinetuneConfig,
) -> Tuple[Any, Dict[str, Any]]:
    new_features = np.asarray(new_normalized["features"], dtype=np.float32)
    new_targets = np.asarray(new_normalized["targets"], dtype=np.int64)
    new_train_indices = np.asarray(new_normalized["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(new_normalized["validation_indices"], dtype=np.int64)

    parent_features = np.asarray(parent_normalized["features"], dtype=np.float32)
    parent_targets = np.asarray(parent_normalized["targets"], dtype=np.int64)
    replay_indices = _select_replay_indices(
        parent_targets,
        np.asarray(parent_normalized["train_indices"], dtype=np.int64),
        config.replay_samples_per_class,
        config.seed,
    )

    feature_names = tuple(str(item) for item in new_dataset.get("feature_names", ()))
    scaled_new_train = scaling_engine.apply_scaler(
        new_features[new_train_indices],
        scaler,
        feature_names=feature_names,
    )
    if replay_indices.size:
        scaled_replay = scaling_engine.apply_scaler(
            parent_features[replay_indices],
            scaler,
            feature_names=feature_names,
        )
        train_x = np.concatenate((scaled_new_train, scaled_replay), axis=0)
        train_y = np.concatenate((new_targets[new_train_indices], parent_targets[replay_indices]), axis=0)
    else:
        train_x = scaled_new_train
        train_y = new_targets[new_train_indices].copy()

    scaled_validation = scaling_engine.apply_scaler(
        new_features[validation_indices],
        scaler,
        feature_names=feature_names,
    )
    validation_y = new_targets[validation_indices].astype(np.int64, copy=False)
    parent_model.eval()
    baseline_logits = _predict_logits(parent_model, scaled_validation)
    baseline_validation_loss = _log_loss_from_logits(baseline_logits, validation_y)

    candidate = copy.deepcopy(parent_model).cpu()
    candidate.train()
    _set_backbone_trainable(candidate, config.head_only_epochs == 0)
    head = getattr(candidate, "tp_before_sl_head", None)
    backbone = getattr(candidate, "backbone", None)
    if not isinstance(head, nn.Module) or not isinstance(backbone, nn.Module):
        raise FinetuneValidationError("Parent model architecture is incompatible with controlled fine-tuning.")

    optimizer = torch.optim.AdamW(
        [
            {"params": list(backbone.parameters()), "lr": config.learning_rate * config.backbone_lr_multiplier},
            {"params": list(head.parameters()), "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    tensor_dataset = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32, copy=False)),
        torch.from_numpy(train_y.astype(np.float32, copy=False).reshape(-1, 1)),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(config.batch_size, len(tensor_dataset)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    best_state = copy.deepcopy(candidate.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        if epoch == config.head_only_epochs + 1:
            _set_backbone_trainable(candidate, True)
        candidate.train()
        running_loss = 0.0
        seen_rows = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            output = candidate(batch_x)
            logits = output.get("tp_before_sl_logit") if isinstance(output, Mapping) else output
            if not isinstance(logits, torch.Tensor):
                raise FinetuneValidationError("Candidate model did not return binary logits.")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite fine-tuning loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), config.max_gradient_norm)
            optimizer.step()
            row_count = int(batch_x.shape[0])
            running_loss += float(loss.detach().cpu().item()) * row_count
            seen_rows += row_count
        scheduler.step()

        candidate.eval()
        validation_logits = _predict_logits(candidate, scaled_validation)
        validation_loss = _log_loss_from_logits(validation_logits, validation_y)
        train_loss = running_loss / max(1, seen_rows)
        improved = validation_loss < best_validation_loss - config.min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(candidate.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_log_loss": float(validation_loss),
                "learning_rate_head": float(optimizer.param_groups[1]["lr"]),
                "learning_rate_backbone": float(optimizer.param_groups[0]["lr"]),
                "backbone_trainable": bool(any(parameter.requires_grad for parameter in backbone.parameters())),
            }
        )
        if config.patience > 0 and stale_epochs >= config.patience:
            break

    candidate.load_state_dict(best_state, strict=True)
    candidate.eval()
    final_logits = _predict_logits(candidate, scaled_validation)
    final_validation_loss = _log_loss_from_logits(final_logits, validation_y)
    tolerance_limit = baseline_validation_loss + config.validation_loss_tolerance
    passed_guard = final_validation_loss <= tolerance_limit
    summary = {
        "baseline_validation_log_loss": float(baseline_validation_loss),
        "candidate_validation_log_loss": float(final_validation_loss),
        "validation_loss_tolerance": float(config.validation_loss_tolerance),
        "validation_regression_limit": float(tolerance_limit),
        "validation_guard_passed": bool(passed_guard),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "training_row_count": int(len(train_y)),
        "new_training_row_count": int(len(new_train_indices)),
        "replay_row_count": int(len(replay_indices)),
        "replay_class_counts": {
            "sl_first": int(np.sum(parent_targets[replay_indices] == 0)) if replay_indices.size else 0,
            "tp_first": int(np.sum(parent_targets[replay_indices] == 1)) if replay_indices.size else 0,
        },
        "validation_row_count": int(len(validation_indices)),
        "epoch_history": history,
    }
    return candidate, summary


# ============================================================================
# IMMUTABLE RUN JOURNAL
# ============================================================================

def _journal_path(identity: Any, market: str) -> Path:
    root = Path(memory_engine.model_dir(market, identity=identity)).expanduser().resolve()
    return root / "FINETUNE_RUNS" / "EVENTS.jsonl"


def _append_event(
    identity: Any,
    market: str,
    *,
    run_id: str,
    event_type: str,
    status: str,
    config: Mapping[str, Any],
    details: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = _journal_path(identity, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": uuid.uuid4().hex,
        "event_type": str(event_type).upper(),
        "status": str(status).upper(),
        "timestamp": _utc_now(),
        "identity": _identity_dict(identity),
        "market": normalize_market(market),
        "target_schema": MODEL_TARGET_SCHEMA,
        "config": _json_safe(dict(config)),
        "details": _json_safe(dict(details or {})),
    }
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("Fine-tuning event write was incomplete.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _read_events(identity: Any, market: str, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _journal_path(identity, market)
    if not path.is_file():
        return []
    expected_identity = _identity_dict(identity)
    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FinetuneIntegrityError(f"Fine-tuning journal has invalid JSON at line {line_number}.") from error
            if not isinstance(record, Mapping) or record.get("identity") != expected_identity:
                raise FinetuneIntegrityError("Fine-tuning journal identity mismatch detected.")
            if run_id is None or str(record.get("run_id")) == str(run_id):
                results.append(dict(record))
    return results


def _register_run_manifest(
    *,
    identity: Any,
    run_id: str,
    payload: Mapping[str, Any],
    dependencies: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    registry = versions_engine.get_default_registry()
    record = registry.register_version(
        FINETUNE_RUN_ARTIFACT_TYPE,
        run_id,
        payload=_json_safe(dict(payload)),
        identity=identity,
        dependencies=list(dependencies),
        metadata={
            "run_status": str(payload.get("run_status", "UNKNOWN")),
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "parent_model_version_id": payload.get("parent_model_version_id"),
            "candidate_model_version_id": payload.get("candidate_model_version_id"),
        },
    )
    return _record_dict(record)


# ============================================================================
# FINE-TUNING ORCHESTRATION
# ============================================================================

def finetune_model_version(
    market: str,
    parent_model_version_id: str,
    dataset_version_id: str,
    *,
    identity: Any,
    config: Optional[FinetuneConfig] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fine-tune an approved parent model on a strictly later approved dataset.

    The parent checkpoint and scaler are loaded read-only. The child checkpoint
    is always stored as a new DRAFT version and is never approved by this module.
    """
    _require_dependencies()
    normalized_market = normalize_market(market)
    resolved_identity = _resolve_identity(normalized_market, identity=identity)
    identity_values = _identity_dict(resolved_identity)
    if identity_values["market"] != normalized_market:
        raise FinetuneValidationError("Requested market does not match the TradingBot identity.")

    parent_version = _safe_identifier(parent_model_version_id, "parent_model_version_id")
    new_dataset_version = _safe_identifier(dataset_version_id, "dataset_version_id")
    selected_config = config or FinetuneConfig()
    if isinstance(selected_config, Mapping):
        selected_config = FinetuneConfig(**dict(selected_config))
    if not isinstance(selected_config, FinetuneConfig):
        raise FinetuneValidationError("config must be a FinetuneConfig or mapping.")
    resolved_run_id = _safe_identifier(run_id or _generated_id("finetune"), "run_id")

    lock = _identity_lock(identity_values)
    if not lock.acquire(blocking=False):
        raise FinetuneInProgressError("Another fine-tuning run is active for this TradingBot.")

    started_at = _utc_now()
    started_recorded = False
    try:
        if _read_events(resolved_identity, normalized_market, resolved_run_id):
            raise FinetuneValidationError("run_id has already been used for this TradingBot.")
        _append_event(
            resolved_identity,
            normalized_market,
            run_id=resolved_run_id,
            event_type="STARTED",
            status="RUNNING",
            config=selected_config.to_dict(),
            details={
                "parent_model_version_id": parent_version,
                "dataset_version_id": new_dataset_version,
            },
        )
        started_recorded = True

        registry = versions_engine.get_default_registry()
        parent_record = registry.get_version("model", parent_version, identity=resolved_identity)
        if _version_status(parent_record) != "APPROVED":
            raise FinetuneValidationError("Only an APPROVED parent model can be fine-tuned.")
        parent_metadata = getattr(parent_record, "metadata", {})
        if not isinstance(parent_metadata, Mapping) or str(parent_metadata.get("model_target_schema", "")).upper() not in {
            MODEL_TARGET_SCHEMA,
            "TP_BEFORE_SL",
        }:
            raise FinetuneValidationError("Parent model is not a TP-before-SL binary model.")
        parent_verification = registry.verify_artifact("model", parent_version, identity=resolved_identity)
        if not parent_verification.get("valid"):
            raise FinetuneIntegrityError("Parent model checkpoint failed its integrity check.")

        new_dataset_record = registry.get_version("dataset", new_dataset_version, identity=resolved_identity)
        if _version_status(new_dataset_record) != "APPROVED":
            raise FinetuneValidationError("Incremental dataset must be APPROVED before fine-tuning.")
        new_dataset_verification = registry.verify_artifact("dataset", new_dataset_version, identity=resolved_identity)
        if not new_dataset_verification.get("valid"):
            raise FinetuneIntegrityError("Incremental dataset failed its integrity check.")

        parent_model, parent_scaler, parent_checkpoint = memory_engine.load_model(
            normalized_market,
            identity=resolved_identity,
            version_id=parent_version,
            require_approved=True,
        )
        if parent_checkpoint.get("identity") != identity_values:
            raise FinetuneValidationError("Parent checkpoint identity does not match the requested TradingBot.")
        parent_summary = parent_checkpoint.get("training_summary", {})
        if not isinstance(parent_summary, Mapping) or str(parent_summary.get("model_target_schema", "")).upper() != MODEL_TARGET_SCHEMA:
            raise FinetuneValidationError("Parent checkpoint training summary has an incompatible target schema.")
        if not parent_checkpoint.get("dataset_version_id") or not parent_checkpoint.get("scaler_version_id"):
            raise FinetuneValidationError("Parent checkpoint must reference immutable dataset and scaler versions.")
        parent_dataset_version = str(parent_checkpoint["dataset_version_id"])
        parent_scaler_version = str(parent_checkpoint["scaler_version_id"])
        if parent_dataset_version == new_dataset_version:
            raise FinetuneValidationError("Incremental dataset must be a new version, not the parent's training dataset.")

        parent_dataset_record = registry.get_version("dataset", parent_dataset_version, identity=resolved_identity)
        parent_scaler_record = registry.get_version("scaler", parent_scaler_version, identity=resolved_identity)
        if _version_status(parent_dataset_record) != "APPROVED" or _version_status(parent_scaler_record) != "APPROVED":
            raise FinetuneValidationError("Parent model dataset and scaler dependencies must both be APPROVED.")
        parent_dataset_verification = registry.verify_artifact("dataset", parent_dataset_version, identity=resolved_identity)
        parent_scaler_verification = registry.verify_artifact("scaler", parent_scaler_version, identity=resolved_identity)
        if not parent_dataset_verification.get("valid") or not parent_scaler_verification.get("valid"):
            raise FinetuneIntegrityError("Parent dataset or scaler failed its integrity check.")

        parent_dataset = memory_engine.load_training_dataset(
            parent_dataset_version,
            identity=resolved_identity,
            require_approved=True,
        )
        new_dataset = memory_engine.load_training_dataset(
            new_dataset_version,
            identity=resolved_identity,
            require_approved=True,
        )
        _validate_dataset_identity(parent_dataset, identity_values, "Parent")
        _validate_dataset_identity(new_dataset, identity_values, "Incremental")
        parent_normalized = _normalise_dataset(parent_dataset, label="Parent")
        new_normalized = _normalise_dataset(new_dataset, label="Incremental")
        feature_contract = _validate_feature_contract(parent_dataset, new_dataset, parent_scaler, parent_model)
        chronology = _validate_incremental_chronology(
            parent_dataset,
            new_dataset,
            parent_normalized,
            new_normalized,
        )
        scaler_metadata = getattr(parent_scaler, "to_dict", lambda: {})()
        if not isinstance(scaler_metadata, Mapping):
            raise FinetuneIntegrityError("Parent scaler does not expose its immutable fit metadata.")
        scaler_dataset_id = scaler_metadata.get("fit_dataset_id")
        scaler_fit_indices_hash = scaler_metadata.get("fit_indices_hash")
        scaler_fit_count = int(scaler_metadata.get("fit_sample_count", 0))
        expected_fit_indices_hash = _sha256_indices(
            np.asarray(parent_normalized["train_indices"], dtype=np.int64).tolist()
        )
        if not scaler_dataset_id or str(scaler_dataset_id) != str(parent_dataset.get("dataset_id")):
            raise FinetuneIntegrityError("Parent scaler was fitted on a different dataset than the parent model references.")
        if not scaler_fit_indices_hash or str(scaler_fit_indices_hash) != expected_fit_indices_hash:
            raise FinetuneIntegrityError("Parent scaler fit-index hash does not match the parent dataset training split.")
        if scaler_fit_count != int(len(parent_normalized["train_indices"])):
            raise FinetuneIntegrityError("Parent scaler fit sample count does not match the parent training split.")

        torch.manual_seed(selected_config.seed)
        np.random.seed(selected_config.seed)
        candidate_model, fine_tune_summary = _train_candidate(
            parent_model,
            parent_scaler,
            new_dataset,
            new_normalized,
            parent_dataset,
            parent_normalized,
            selected_config,
        )

        base_result = {
            "schema_version": MODULE_SCHEMA_VERSION,
            "run_id": resolved_run_id,
            "identity": identity_values,
            "market": normalized_market,
            "target_schema": MODEL_TARGET_SCHEMA,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "parent_model_version_id": parent_version,
            "parent_dataset_version_id": parent_dataset_version,
            "parent_scaler_version_id": parent_scaler_version,
            "incremental_dataset_version_id": new_dataset_version,
            "config": selected_config.to_dict(),
            "feature_contract": feature_contract,
            "chronology": chronology,
            "scaler_reuse": {
                "scaler_version_id": parent_scaler_version,
                "fit_scope": getattr(parent_scaler, "fit_scope", None),
                "fit_dataset_id": getattr(parent_scaler, "fit_dataset_id", None),
                "fit_indices_hash": scaler_fit_indices_hash,
                "scaler_refitted": False,
            },
            "dependency_integrity": {
                "parent_model": parent_verification,
                "parent_dataset": parent_dataset_verification,
                "parent_scaler": parent_scaler_verification,
                "incremental_dataset": new_dataset_verification,
            },
            "fine_tuning": fine_tune_summary,
        }

        if not fine_tune_summary["validation_guard_passed"]:
            base_result.update(
                {
                    "run_status": "REJECTED_VALIDATION_REGRESSION",
                    "candidate_model_version_id": None,
                    "candidate_model_status": None,
                    "approval_status": "NOT_APPROVED",
                    "rejection_reason": "Fine-tuned validation log-loss exceeded the parent-model regression tolerance.",
                }
            )
            completed = _utc_now()
            base_result["completed_at"] = completed
            _append_event(
                resolved_identity,
                normalized_market,
                run_id=resolved_run_id,
                event_type="REJECTED",
                status="REJECTED_VALIDATION_REGRESSION",
                config=selected_config.to_dict(),
                details={
                    "parent_model_version_id": parent_version,
                    "dataset_version_id": new_dataset_version,
                    "fine_tuning": fine_tune_summary,
                },
            )
            manifest = _register_run_manifest(
                identity=resolved_identity,
                run_id=resolved_run_id,
                payload=base_result,
                dependencies=[
                    {"artifact_type": "model", "version_id": parent_version},
                    {"artifact_type": "dataset", "version_id": new_dataset_version},
                    {"artifact_type": "scaler", "version_id": parent_scaler_version},
                ],
            )
            base_result["finetune_run_version"] = manifest
            return base_result

        candidate_version_id = _generated_id("model_finetune")
        training_summary = {
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "dataset_target": DATASET_TARGET_NAME,
            "dataset_version_id": new_dataset_version,
            "scaler_version_id": parent_scaler_version,
            "feature_schema_hash": feature_contract["feature_schema_hash"],
            "feature_names": feature_contract["feature_names"],
            "input_features": feature_contract["feature_count"],
            "parent_model_version_id": parent_version,
            "fine_tune_run_id": resolved_run_id,
            "fine_tune_config": selected_config.to_dict(),
            "fine_tune_summary": fine_tune_summary,
            "chronology": chronology,
            "selection_metric": "INCREMENTAL_VALIDATION_LOG_LOSS",
            "validation_available": True,
            "test_metrics": None,
            "calibration_status": "NOT_CALIBRATED",
            "trained_at": _utc_now(),
        }
        candidate_path = memory_engine.save_model(
            normalized_market,
            candidate_model,
            parent_scaler,
            training_summary,
            identity=resolved_identity,
            version_id=candidate_version_id,
            dependencies=[{"artifact_type": "model", "version_id": parent_version}],
            register_version=True,
            scaler_version_id=parent_scaler_version,
            dataset_version_id=new_dataset_version,
            metadata={
                "model_target_schema": MODEL_TARGET_SCHEMA,
                "feature_schema_hash": feature_contract["feature_schema_hash"],
                "parent_model_version_id": parent_version,
                "fine_tune_run_id": resolved_run_id,
                "fine_tuned": True,
            },
        )
        candidate_record = registry.get_version("model", candidate_version_id, identity=resolved_identity)
        if _version_status(candidate_record) != "DRAFT":
            raise FinetuneIntegrityError("New fine-tuned checkpoint did not remain in DRAFT state.")
        candidate_verification = registry.verify_artifact("model", candidate_version_id, identity=resolved_identity)
        if not candidate_verification.get("valid"):
            raise FinetuneIntegrityError("New fine-tuned checkpoint failed its registered integrity check.")

        base_result.update(
            {
                "run_status": "COMPLETED",
                "candidate_model_version_id": candidate_version_id,
                "candidate_model_status": "DRAFT",
                "candidate_model_path": str(candidate_path),
                "candidate_model_integrity": candidate_verification,
                "approval_status": "NOT_APPROVED",
                "evaluation_required": True,
                "historical_test_required": True,
                "parent_model_unchanged": True,
                "candidate_dataset_version_id": new_dataset_version,
                "candidate_scaler_version_id": parent_scaler_version,
            }
        )
        completed = _utc_now()
        base_result["completed_at"] = completed
        _append_event(
            resolved_identity,
            normalized_market,
            run_id=resolved_run_id,
            event_type="COMPLETED",
            status="COMPLETED",
            config=selected_config.to_dict(),
            details={
                "parent_model_version_id": parent_version,
                "candidate_model_version_id": candidate_version_id,
                "dataset_version_id": new_dataset_version,
                "scaler_version_id": parent_scaler_version,
                "fine_tuning": fine_tune_summary,
            },
        )
        dependencies = [
            {"artifact_type": "model", "version_id": parent_version},
            {"artifact_type": "dataset", "version_id": new_dataset_version},
            {"artifact_type": "scaler", "version_id": parent_scaler_version},
            {"artifact_type": "model", "version_id": candidate_version_id},
        ]
        base_result["finetune_run_version"] = _register_run_manifest(
            identity=resolved_identity,
            run_id=resolved_run_id,
            payload=base_result,
            dependencies=dependencies,
        )
        return base_result

    except Exception as error:
        if started_recorded:
            failure_payload = {
                "schema_version": MODULE_SCHEMA_VERSION,
                "run_id": resolved_run_id,
                "run_status": "FAILED",
                "identity": identity_values,
                "market": normalized_market,
                "target_schema": MODEL_TARGET_SCHEMA,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "parent_model_version_id": parent_version,
                "incremental_dataset_version_id": new_dataset_version,
                "config": selected_config.to_dict(),
                "failure": {
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                },
            }
            try:
                _append_event(
                    resolved_identity,
                    normalized_market,
                    run_id=resolved_run_id,
                    event_type="FAILED",
                    status="FAILED",
                    config=selected_config.to_dict(),
                    details=failure_payload["failure"],
                )
                _register_run_manifest(
                    identity=resolved_identity,
                    run_id=resolved_run_id,
                    payload=failure_payload,
                    dependencies=[{"artifact_type": "model", "version_id": parent_version}],
                )
            except Exception:
                pass
        raise
    finally:
        lock.release()


# ============================================================================
# RUN DISCOVERY
# ============================================================================

def list_finetuning_runs(
    *,
    identity: Any,
    market: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """List append-only fine-tuning events for one TradingBot."""
    _require_dependencies()
    identity_values = _identity_dict(identity)
    normalized_market = normalize_market(market or identity_values["market"])
    if identity_values["market"] != normalized_market:
        raise FinetuneValidationError("Requested market does not match the supplied identity.")
    events = _read_events(identity, normalized_market)
    latest_by_run: Dict[str, Dict[str, Any]] = {}
    for event in events:
        latest_by_run[str(event["run_id"])] = event
    output = list(latest_by_run.values())
    output.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    if limit is not None:
        if isinstance(limit, bool) or int(limit) < 0:
            raise FinetuneValidationError("limit must be a non-negative integer or None.")
        output = output[: int(limit)]
    return output


def get_finetuning_run(
    run_id: str,
    *,
    identity: Any,
    market: Optional[str] = None,
) -> Dict[str, Any]:
    """Return all lifecycle events for one fine-tuning run."""
    _require_dependencies()
    safe_run_id = _safe_identifier(run_id, "run_id")
    identity_values = _identity_dict(identity)
    normalized_market = normalize_market(market or identity_values["market"])
    if identity_values["market"] != normalized_market:
        raise FinetuneValidationError("Requested market does not match the supplied identity.")
    events = _read_events(identity, normalized_market, safe_run_id)
    if not events:
        raise FileNotFoundError(f"Fine-tuning run {safe_run_id!r} was not found.")
    return {
        "run_id": safe_run_id,
        "identity": identity_values,
        "market": normalized_market,
        "events": events,
        "latest_status": events[-1].get("status"),
    }


def run_finetuning(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility alias for the versioned fine-tuning entry point."""
    return finetune_model_version(*args, **kwargs)


def run_finetune(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility alias for the versioned fine-tuning entry point."""
    return finetune_model_version(*args, **kwargs)


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "MODULE_SCHEMA_VERSION",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET_NAME",
    "FinetuneError",
    "FinetuneValidationError",
    "FinetuneDependencyError",
    "FinetuneIntegrityError",
    "FinetuneInProgressError",
    "FinetuneConfig",
    "finetune_model_version",
    "run_finetuning",
    "run_finetune",
    "list_finetuning_runs",
    "get_finetuning_run",
]
