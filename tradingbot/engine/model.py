"""Binary TP-before-SL model training and inference for Spectline."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import copy
import importlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# MODULE DEPENDENCIES
# ============================================================================

try:
    from . import dataset as dataset_engine
    from . import memory as memory_engine
    from . import scaling as scaling_engine
    from . import settings as settings_engine
except ImportError:
    try:
        from engine import dataset as dataset_engine
        from engine import memory as memory_engine
        from engine import scaling as scaling_engine
        from engine import settings as settings_engine
    except ImportError:
        dataset_engine = None
        memory_engine = None
        scaling_engine = None
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
# MODEL CONFIGURATION
# ============================================================================

MODEL_FORMAT_VERSION = 4
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET_NAME = "TP_BEFORE_SL"
DEFAULT_HIDDEN_SIZES = (256, 128, 64)
DEFAULT_DROPOUT_1 = 0.10
DEFAULT_DROPOUT_2 = 0.08
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
DEFAULT_EARLY_STOPPING_PATIENCE = 12
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_SCALING_METHOD = "standard"
MAX_GRADIENT_NORM = 1.0

if settings_engine is not None:
    DEFAULT_EPOCHS = int(getattr(settings_engine, "DEFAULT_EPOCHS", 80))
    DEFAULT_BATCH_SIZE = int(getattr(settings_engine, "DEFAULT_BATCH_SIZE", 64))
    DEFAULT_LEARNING_RATE = float(getattr(settings_engine, "DEFAULT_LEARNING_RATE", 1e-3))
    DEFAULT_WEIGHT_DECAY = float(getattr(settings_engine, "DEFAULT_WEIGHT_DECAY", 1e-5))
    SEED = int(getattr(settings_engine, "SEED", 42))
    SUPPORTED_TIMEFRAMES = tuple(getattr(settings_engine, "SUPPORTED_TIMEFRAMES", ()))
else:
    DEFAULT_EPOCHS = 80
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_WEIGHT_DECAY = 1e-5
    SEED = 42
    SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")


# ============================================================================
# ERRORS
# ============================================================================

class ModelError(RuntimeError):
    """Base exception for binary model operations."""


class ModelValidationError(ModelError, ValueError):
    """Raised when model inputs or training data violate the model contract."""


class ModelDependencyError(ModelError, ImportError):
    """Raised when a required engine dependency is unavailable."""


class ModelSchemaError(ModelError):
    """Raised when a model, dataset or scaler schema is incompatible."""


# ============================================================================
# REPRODUCIBILITY AND NUMERICAL HELPERS
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible model training."""
    normalized = int(seed)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_float(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelValidationError(f"{name} must be a finite number.") from error
    if not math.isfinite(result):
        raise ModelValidationError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ModelValidationError(f"{name} must be at least {minimum}.")
    return result


def _finite_matrix(values: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelValidationError(f"{name} must be a numeric matrix.") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ModelValidationError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ModelValidationError(f"{name} contains NaN or infinite values.")
    return matrix


def _finite_targets(values: Any, sample_count: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelValidationError("Dataset targets must be a one-dimensional binary array.") from error
    if raw.shape[0] != sample_count:
        raise ModelValidationError("Feature and target row counts do not match.")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        raise ModelValidationError("Targets must contain finite integer labels only.")
    if not np.all(np.isin(raw, (0.0, 1.0))):
        raise ModelValidationError("Targets must use 0=SL_FIRST and 1=TP_FIRST.")
    return raw.astype(np.int64, copy=False)


def _safe_index_list(values: Any, name: str, sample_count: int) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ModelValidationError(f"Dataset split '{name}' must be a sequence of indices.")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ModelValidationError(f"Dataset split '{name}' contains a boolean index.")
        try:
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("Index is not an integer.")
            index = int(numeric)
        except (TypeError, ValueError, OverflowError) as error:
            raise ModelValidationError(f"Dataset split '{name}' contains an invalid index.") from error
        if index < 0 or index >= sample_count:
            raise ModelValidationError(f"Dataset split '{name}' contains an out-of-range index.")
        result.append(index)
    if len(result) != len(set(result)):
        raise ModelValidationError(f"Dataset split '{name}' contains duplicate indices.")
    return result


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TradingMicroModel(nn.Module):
    """Numerical neural network with one binary TP-before-SL output."""

    def __init__(
        self,
        input_size: int,
        hidden_1: int = DEFAULT_HIDDEN_SIZES[0],
        hidden_2: int = DEFAULT_HIDDEN_SIZES[1],
        hidden_3: int = DEFAULT_HIDDEN_SIZES[2],
    ) -> None:
        super().__init__()
        dimensions = (input_size, hidden_1, hidden_2, hidden_3)
        if any(isinstance(value, bool) or int(value) <= 0 for value in dimensions):
            raise ModelValidationError("Model input and hidden dimensions must be positive integers.")

        self.input_size = int(input_size)
        self.hidden_sizes = (int(hidden_1), int(hidden_2), int(hidden_3))
        self.backbone = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.LayerNorm(self.hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(DEFAULT_DROPOUT_1),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.LayerNorm(self.hidden_sizes[1]),
            nn.GELU(),
            nn.Dropout(DEFAULT_DROPOUT_2),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            nn.LayerNorm(self.hidden_sizes[2]),
            nn.GELU(),
        )
        self.tp_before_sl_head = nn.Linear(self.hidden_sizes[2], 1)
        self.model_target_schema = MODEL_TARGET_SCHEMA

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """Return the raw TP-before-SL logit and its sigmoid probability."""
        if features.ndim != 2:
            raise ModelValidationError("Model input must be a two-dimensional tensor.")
        if features.shape[1] != self.input_size:
            raise ModelValidationError(
                f"Model expects {self.input_size} input features, received {features.shape[1]}."
            )
        latent = self.backbone(features)
        logits = self.tp_before_sl_head(latent)
        return {
            "tp_before_sl_logit": logits,
            "tp_before_sl_probability": torch.sigmoid(logits),
        }


# ============================================================================
# LOSS AND METRICS
# ============================================================================

def _extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        logits = outputs
    elif isinstance(outputs, Mapping):
        logits = outputs.get("tp_before_sl_logit", outputs.get("logits"))
    else:
        logits = None
    if not isinstance(logits, Tensor):
        raise ModelValidationError("Model outputs do not contain TP-before-SL logits.")
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ModelValidationError("Binary model logits must have shape (samples, 1).")
    return logits


def compute_loss(
    outputs: Any,
    targets: Any = None,
    *legacy_args: Any,
    **legacy_kwargs: Any,
) -> Tuple[Tensor, Dict[str, float]]:
    """Calculate binary cross-entropy loss for TP-before-SL labels only."""
    if legacy_args or legacy_kwargs:
        raise ModelSchemaError(
            "The former multi-task loss is no longer supported. Supply only binary TP-before-SL targets."
        )
    if targets is None:
        raise ModelValidationError("Binary targets are required.")
    logits = _extract_logits(outputs)
    target_tensor = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.reshape(-1, 1)
    if target_tensor.shape != logits.shape:
        raise ModelValidationError("Binary target shape must match model logits.")
    if not torch.isfinite(target_tensor).all():
        raise ModelValidationError("Binary targets contain non-finite values.")
    if not torch.all((target_tensor == 0) | (target_tensor == 1)):
        raise ModelValidationError("Binary targets must contain only 0 or 1.")
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)
    probability = torch.sigmoid(logits)
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "brier_score": float(torch.mean((probability - target_tensor) ** 2).detach().cpu().item()),
        "positive_rate": float(target_tensor.mean().detach().cpu().item()),
    }
    return loss, metrics


def masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Retain the former generic masked regression utility for API compatibility."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Prediction, target and mask shapes must match.")
    safe_mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    raw_loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return (raw_loss * safe_mask).sum() / safe_mask.sum().clamp_min(1.0)


def _binary_auc(targets: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_values = probabilities[order]
    ranks = np.empty(len(targets), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[targets == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ModelValidationError("Metric target and probability lengths do not match.")
    if y.size == 0:
        return {
            "sample_count": 0,
            "log_loss": None,
            "brier_score": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "positive_rate": None,
            "mean_probability": None,
        }
    if not np.all(np.isfinite(p)):
        raise ModelValidationError("Predicted probabilities contain non-finite values.")
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    predicted = (p >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(np.int64)
    accuracy = float(np.mean(predicted == y))
    class_recalls = []
    for label in (0, 1):
        selected = y == label
        if np.any(selected):
            class_recalls.append(float(np.mean(predicted[selected] == label)))
    balanced_accuracy = float(np.mean(class_recalls)) if len(class_recalls) == 2 else None
    return {
        "sample_count": int(y.size),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p))),
        "brier_score": float(np.mean((p - y) ** 2)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": _binary_auc(y, p),
        "positive_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
    }


def _predict_probabilities(model: TradingMicroModel, features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(np.asarray(features, dtype=np.float32)))
        probabilities = output["tp_before_sl_probability"].reshape(-1).cpu().numpy()
    return probabilities.astype(np.float32, copy=False)


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def _normalize_training_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the TP-before-SL dataset and its chronological split indices."""
    if not isinstance(dataset, Mapping):
        raise ModelValidationError("Training dataset must be a mapping.")
    if dataset.get("target") != DATASET_TARGET_NAME:
        raise ModelSchemaError("Training dataset target must be TP_BEFORE_SL.")
    declared_schema = str(dataset.get("model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if declared_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise ModelSchemaError("Dataset model_target_schema is incompatible with TP-before-SL training.")
    if dataset_engine is None:
        raise ModelDependencyError("engine.dataset is unavailable.")

    raw_features = np.asarray(
        dataset.get("features", []),
        dtype=np.float32,
    )

    if raw_features.ndim != 2 or raw_features.shape[0] == 0:
        stats = dataset.get("stats", {})
        stats = stats if isinstance(stats, Mapping) else {}

        raise ModelValidationError(
            "No labelled training rows are available. "
            f"Setups={stats.get('setup_count', 0)}, "
            f"feature_ready={stats.get('feature_ready_setup_count', 0)}, "
            f"labelled={stats.get('labelled_example_count', 0)}, "
            f"unlabelled={stats.get('unlabelled_candidate_count', 0)}, "
            f"excluded={stats.get('excluded_setup_count', 0)}. "
            "Check the dataset label_report and excluded_setups."
        )

    report = dataset_engine.validate_training_dataset(dataset)
    if not isinstance(report, Mapping) or report.get("valid") is not True:
        errors = report.get("errors", []) if isinstance(report, Mapping) else []
        raise ModelValidationError("Training dataset is invalid: " + "; ".join(str(item) for item in errors))

    features = _finite_matrix(dataset.get("features"), "features")
    targets = _finite_targets(dataset.get("targets", dataset.get("labels")), features.shape[0])
    feature_names = tuple(str(item) for item in dataset.get("feature_names", ()))
    if len(feature_names) != features.shape[1] or any(not name for name in feature_names):
        raise ModelSchemaError("feature_names do not match the feature matrix width.")
    if len(set(feature_names)) != len(feature_names):
        raise ModelSchemaError("feature_names must be unique.")

    label_metadata = dataset.get("label_metadata", [])
    sample_metadata = dataset.get("sample_metadata", [])
    if len(label_metadata) != len(targets) or len(sample_metadata) != len(targets):
        raise ModelValidationError("Sample and label metadata must align with target rows.")

    splits = dataset.get("splits")
    if not isinstance(splits, Mapping):
        raise ModelValidationError("Dataset must contain chronological train/validation/test splits.")
    train_indices = _safe_index_list(splits.get("train_indices", []), "train_indices", len(targets))
    validation_indices = _safe_index_list(splits.get("validation_indices", []), "validation_indices", len(targets))
    test_indices = _safe_index_list(splits.get("test_indices", []), "test_indices", len(targets))
    all_indices = train_indices + validation_indices + test_indices
    if len(all_indices) != len(set(all_indices)):
        raise ModelValidationError("Chronological dataset splits overlap.")
    if not train_indices:
        raise ModelValidationError("Chronological train split is empty after purging.")

    def _decision_time(index: int) -> int:
        try:
            return int(sample_metadata[index]["decision_time"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ModelValidationError(f"Sample metadata row {index} has no valid decision_time.") from error

    def _known_time(index: int) -> int:
        try:
            return int(label_metadata[index]["outcome_known_at"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ModelValidationError(f"Label metadata row {index} has no valid outcome_known_at.") from error

    if validation_indices:
        validation_start = min(_decision_time(index) for index in validation_indices)
        if max(_decision_time(index) for index in train_indices) >= validation_start:
            raise ModelValidationError("Train and validation decisions are not chronologically separated.")
        if any(_known_time(index) >= validation_start for index in train_indices):
            raise ModelValidationError("A training outcome became known after validation had started.")
    if validation_indices and test_indices:
        test_start = min(_decision_time(index) for index in test_indices)
        if max(_decision_time(index) for index in validation_indices) >= test_start:
            raise ModelValidationError("Validation and test decisions are not chronologically separated.")
        if any(_known_time(index) >= test_start for index in validation_indices):
            raise ModelValidationError("A validation outcome became known after the test period had started.")

    train_classes = set(int(value) for value in targets[train_indices].tolist())
    if train_classes != {0, 1}:
        raise ModelValidationError(
            "Training split must contain both SL_FIRST=0 and TP_FIRST=1 examples."
        )

    for index, label in enumerate(label_metadata):
        if not isinstance(label, Mapping):
            raise ModelValidationError(f"Label metadata row {index} is not a mapping.")
        if int(label.get("actual_outcome", -1)) != int(targets[index]):
            raise ModelValidationError(f"Label metadata row {index} does not match its binary target.")

    feature_schema = dataset.get("feature_schema", {})
    schema_hash = feature_schema.get("sha256") if isinstance(feature_schema, Mapping) else None
    if not schema_hash:
        schema_hash = dataset.get("feature_schema_hash")
    return {
        "features": features,
        "targets": targets,
        "feature_names": feature_names,
        "feature_schema_hash": str(schema_hash) if schema_hash else None,
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "validation_indices": np.asarray(validation_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "stats": dict(dataset.get("stats", {})) if isinstance(dataset.get("stats"), Mapping) else {},
        "dataset_id": str(dataset.get("dataset_id", "")),
        "identity": dataset.get("identity"),
        "warnings": list(dataset.get("warnings", [])) + list(report.get("warnings", [])),
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    dataset: Mapping[str, Any],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    *,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    scaling_method: str = DEFAULT_SCALING_METHOD,
) -> Tuple[TradingMicroModel, Any, Dict[str, Any]]:
    """Train the binary model with a train-only scaler and chronological validation."""
    normalized = _normalize_training_dataset(dataset)
    if isinstance(epochs, bool) or int(epochs) <= 0:
        raise ModelValidationError("epochs must be a positive integer.")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ModelValidationError("batch_size must be a positive integer.")
    if isinstance(patience, bool) or int(patience) < 0:
        raise ModelValidationError("patience must be a non-negative integer.")
    learning_rate = _finite_float(learning_rate, "learning_rate", minimum=1e-12)
    weight_decay = _finite_float(weight_decay, "weight_decay", minimum=0.0)
    min_delta = _finite_float(min_delta, "min_delta", minimum=0.0)
    hidden = tuple(int(value) for value in hidden_sizes)
    if len(hidden) != 3 or any(value <= 0 for value in hidden):
        raise ModelValidationError("hidden_sizes must contain three positive integers.")
    if scaling_engine is None:
        raise ModelDependencyError("engine.scaling is unavailable.")

    set_seed(SEED)
    scaler = scaling_engine.fit_dataset_scaler(dataset, method=str(scaling_method))
    scaler_report = scaling_engine.validate_scaler(scaler)
    if not scaler_report.get("valid") or scaler.fit_scope != "TRAIN_SPLIT_ONLY":
        raise ModelSchemaError("Scaler is invalid or was not fitted exclusively on training rows.")

    scaled = scaling_engine.apply_scaler(
        normalized["features"],
        scaler,
        feature_names=normalized["feature_names"],
    )
    train_indices = normalized["train_indices"]
    validation_indices = normalized["validation_indices"]
    train_x = scaled[train_indices]
    train_y = normalized["targets"][train_indices].astype(np.float32).reshape(-1, 1)
    validation_x = scaled[validation_indices]
    validation_y = normalized["targets"][validation_indices].astype(np.float32).reshape(-1, 1)

    tensor_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    effective_batch_size = min(int(batch_size), len(tensor_dataset))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)
    loader = DataLoader(
        tensor_dataset,
        batch_size=max(1, effective_batch_size),
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )

    model = TradingMicroModel(
        input_size=scaled.shape[1],
        hidden_1=hidden[0],
        hidden_2=hidden[1],
        hidden_3=hidden[2],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_monitor = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[Dict[str, Any]] = []
    has_validation = len(validation_indices) > 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        seen_rows = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss, _ = compute_loss(outputs, batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite binary loss detected at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
            row_count = int(batch_features.shape[0])
            running_loss += float(loss.detach().cpu().item()) * row_count
            seen_rows += row_count

        scheduler.step()
        train_loss = running_loss / max(1, seen_rows)
        if has_validation:
            validation_probabilities = _predict_probabilities(model, validation_x)
            clipped = np.clip(validation_probabilities, 1e-7, 1.0 - 1e-7)
            monitor_loss = float(
                -np.mean(validation_y.reshape(-1) * np.log(clipped) + (1.0 - validation_y.reshape(-1)) * np.log(1.0 - clipped))
            )
        else:
            monitor_loss = train_loss

        improved = monitor_loss < (best_monitor - min_delta)
        if improved:
            best_monitor = monitor_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(monitor_loss) if has_validation else None,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            validation_text = f" validation_loss={monitor_loss:.6f}" if has_validation else " validation_loss=unavailable"
            print(f"[TRAIN] epoch={epoch:04d} train_loss={train_loss:.6f}{validation_text}")
        if has_validation and int(patience) > 0 and stale_epochs >= int(patience):
            break

    model.load_state_dict(best_state, strict=True)
    model.eval()
    train_probabilities = _predict_probabilities(model, train_x)
    train_metrics = _binary_metrics(normalized["targets"][train_indices], train_probabilities)
    validation_probabilities = _predict_probabilities(model, validation_x)
    validation_metrics = _binary_metrics(normalized["targets"][validation_indices], validation_probabilities)

    summary: Dict[str, Any] = {
        **normalized["stats"],
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "dataset_target": DATASET_TARGET_NAME,
        "dataset_id": normalized["dataset_id"],
        "input_features": int(scaled.shape[1]),
        "feature_names": list(normalized["feature_names"]),
        "feature_schema_hash": normalized["feature_schema_hash"],
        "hidden_sizes": list(hidden),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "batch_size": int(effective_batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "early_stopping_patience": int(patience),
        "selection_metric": "VALIDATION_LOG_LOSS" if has_validation else "TRAIN_LOG_LOSS_FALLBACK",
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_available": bool(has_validation),
        "test_sample_count": int(len(normalized["test_indices"])),
        "test_metrics": None,
        "scaling_method": str(scaling_method).strip().lower(),
        "scaler_fit_scope": str(scaler.fit_scope),
        "scaler_fit_sample_count": int(scaler.fit_sample_count),
        "calibration_status": "NOT_CALIBRATED",
        "classification_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "training_warnings": list(normalized["warnings"]),
        "epoch_history": history,
        "trained_at": _utc_now(),
    }
    if not has_validation:
        summary["training_warnings"].append(
            "No validation rows survived chronological purging; early stopping used training loss. The model must not be approved without a separate evaluation."
        )
    return model, scaler, summary


# ============================================================================
# IDENTITY AND VERSION HELPERS
# ============================================================================

def _resolve_model_identity(
    market: str,
    identity: Optional[Any] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Any:
    """Resolve the exact TradingBot identity through the memory contract."""
    if memory_engine is None:
        raise ModelDependencyError("engine.memory is unavailable.")
    normalized_market = normalize_market(market)
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if callable(resolver):
        return resolver(
            normalized_market,
            identity=identity,
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
    if identity is not None:
        values = identity.as_dict() if callable(getattr(identity, "as_dict", None)) else dict(identity)
        if str(values.get("market", "")).upper() != normalized_market:
            raise ModelValidationError("Identity market does not match the requested market.")
        return identity
    return resolve_identity(
        market=normalized_market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def _record_version_id(record: Any) -> str:
    if isinstance(record, Mapping):
        value = record.get("version_id") or record.get("dataset_id")
    else:
        value = getattr(record, "version_id", None)
    if value is None or not str(value).strip():
        raise ModelSchemaError("Artifact storage did not return a version identifier.")
    return str(value)


def _generated_version_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:10]}"


def _dataset_timeframes(bundle: Mapping[str, Any]) -> list[str]:
    historical = bundle.get("historical_data", {})
    raw = historical.get("timeframes", []) if isinstance(historical, Mapping) else []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    selected = {str(value).strip().upper() for value in raw}
    return [value for value in SUPPORTED_TIMEFRAMES if value in selected]


def _load_data_bundle(market: str, identity: Any) -> Dict[str, Any]:
    try:
        module_name = f"{__package__}.data" if __package__ else "engine.data"
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ModelDependencyError("engine.data is unavailable.") from error
    function = getattr(module, "prepare_training_data", None)
    if not callable(function):
        raise ModelDependencyError("engine.data.prepare_training_data is unavailable.")
    bundle = function(market, identity=identity)
    if not isinstance(bundle, Mapping):
        raise ModelValidationError("engine.data returned an invalid training bundle.")
    return dict(bundle)


def _build_binary_dataset(bundle: Mapping[str, Any], identity: Any) -> Dict[str, Any]:
    if dataset_engine is None:
        raise ModelDependencyError("engine.dataset is unavailable.")
    function = getattr(dataset_engine, "build_training_dataset", None)
    if not callable(function):
        raise ModelDependencyError("engine.dataset.build_training_dataset is unavailable.")
    identity_values = identity.as_dict() if callable(getattr(identity, "as_dict", None)) else dict(identity)
    result = function(
        data_bundle=bundle,
        identity=identity_values,
        require_identity=True,
    )
    if not isinstance(result, Mapping):
        raise ModelValidationError("engine.dataset returned an invalid labelled dataset.")
    return dict(result)


def _register_dataset_and_approve(dataset: Mapping[str, Any], identity: Any) -> str:
    if memory_engine is None:
        raise ModelDependencyError("engine.memory is unavailable.")
    content_dataset_id = str(dataset.get("dataset_id", "")).strip()
    if not content_dataset_id:
        raise ModelValidationError("Dataset has no immutable dataset_id.")
    dataset_version_id = _generated_version_id("dataset")
    record = memory_engine.save_training_dataset(
        dataset,
        identity=identity,
        version_id=dataset_version_id,
        metadata={
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "content_dataset_id": content_dataset_id,
        },
    )
    version_id = _record_version_id(record)
    memory_engine.validate_artifact_version("dataset", version_id, identity=identity)
    memory_engine.approve_artifact_version(
        "dataset",
        version_id,
        identity=identity,
        reason="Validated TP-before-SL training dataset.",
    )
    return version_id


def _register_scaler_and_approve(
    scaler: Any,
    *,
    identity: Any,
    dataset_version_id: str,
) -> str:
    if memory_engine is None:
        raise ModelDependencyError("engine.memory is unavailable.")
    version_id = _generated_version_id("scaler")
    record = memory_engine.save_scaler_artifact(
        scaler,
        identity=identity,
        version_id=version_id,
        dataset_version_id=dataset_version_id,
        metadata={
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "fit_scope": getattr(scaler, "fit_scope", None),
        },
    )
    resolved_id = _record_version_id(record)
    memory_engine.validate_artifact_version("scaler", resolved_id, identity=identity)
    memory_engine.approve_artifact_version(
        "scaler",
        resolved_id,
        identity=identity,
        reason="Validated train-only feature scaler for TP-before-SL dataset.",
    )
    return resolved_id


# ============================================================================
# MARKET TRAINING ENTRY POINT
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
) -> Dict[str, Any]:
    """Build, train and immutably register one customer TradingBot model."""
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    bundle = _load_data_bundle(normalized_market, resolved)
    bundle_validation = bundle.get("validation", {})
    bundle_report = bundle.get("report", {})
    if not isinstance(bundle_validation, Mapping) or bundle_validation.get("valid") is not True:
        errors = bundle_validation.get("errors", []) if isinstance(bundle_validation, Mapping) else []
        raise ModelValidationError("Training Data Bundle is invalid: " + "; ".join(str(item) for item in errors))
    if not isinstance(bundle_report, Mapping) or bundle_report.get("ready_for_next_stage") is not True:
        issues = bundle_report.get("issues", []) if isinstance(bundle_report, Mapping) else []
        raise ModelValidationError(
            "Training Data Bundle is not ready for labelled dataset generation: "
            + "; ".join(str(item) for item in issues)
        )

    dataset = _build_binary_dataset(bundle, resolved)
    dataset_validation = dataset_engine.validate_training_dataset(dataset)
    if dataset_validation.get("valid") is not True:
        errors = dataset_validation.get("errors", [])
        raise ModelValidationError("Labelled dataset is invalid: " + "; ".join(str(item) for item in errors))
    normalized_dataset = _normalize_training_dataset(dataset)
    dataset_version_id = _register_dataset_and_approve(dataset, resolved)

    model, scaler, training_summary = train_model(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scaler_version_id = _register_scaler_and_approve(
        scaler,
        identity=resolved,
        dataset_version_id=dataset_version_id,
    )
    model_version_id = _generated_version_id("model")

    training_summary.update(
        {
            "identity": resolved.as_dict() if callable(getattr(resolved, "as_dict", None)) else dict(resolved),
            "market": normalized_market,
            "input_timeframes": _dataset_timeframes(bundle),
            "dataset_version_id": dataset_version_id,
            "scaler_version_id": scaler_version_id,
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "model_version_id": model_version_id,
            "source_bundle_schema_version": bundle.get("schema_version"),
            "source_bundle_created_at": bundle.get("created_at"),
        }
    )
    saved_path = memory_engine.save_model(
        normalized_market,
        model,
        scaler,
        training_summary,
        identity=resolved,
        version_id=model_version_id,
        scaler_version_id=scaler_version_id,
        dataset_version_id=dataset_version_id,
        register_version=True,
        metadata={
            "model_target_schema": MODEL_TARGET_SCHEMA,
            "feature_schema_hash": normalized_dataset["feature_schema_hash"],
        },
    )

    return {
        "training": training_summary,
        "dataset": {
            "version_id": dataset_version_id,
            "content_dataset_id": dataset.get("dataset_id"),
            "sample_count": int(len(dataset.get("targets", []))),
            "feature_count": int(len(dataset.get("feature_names", []))),
            "target_counts": dataset_validation.get("target_counts", {}),
        },
        "scaler": {
            "version_id": scaler_version_id,
            "fit_scope": scaler.fit_scope,
            "fit_sample_count": scaler.fit_sample_count,
            "method": scaler.method,
        },
        "model": {
            "identity": resolved.as_dict() if callable(getattr(resolved, "as_dict", None)) else dict(resolved),
            "market": normalized_market,
            "model_version_id": model_version_id,
            "model_path": str(saved_path),
            "status": "DRAFT",
            "approval_required": True,
            "approval_reason": "A separate evaluation and calibration gate must approve this model before production inference.",
        },
    }


# ============================================================================
# INFERENCE
# ============================================================================

def predict_from_features(
    model: TradingMicroModel,
    scaler: Any,
    features: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Predict the TP-before-SL probability from one dataset-schema feature row."""
    if not isinstance(model, TradingMicroModel):
        raise TypeError("model must be a TradingMicroModel instance.")
    threshold_value = _finite_float(threshold, "threshold", minimum=0.0)
    if threshold_value > 1.0:
        raise ModelValidationError("threshold cannot exceed 1.0.")
    try:
        matrix = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelValidationError("Inference features must be numeric.") from error
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != 1:
        raise ModelValidationError("Inference requires exactly one feature row.")
    if matrix.shape[1] != model.input_size:
        raise ModelSchemaError(
            f"Model expects {model.input_size} features, received {matrix.shape[1]}. "
            "The caller must use the feature builder and schema from engine.dataset."
        )
    if not np.all(np.isfinite(matrix)):
        raise ModelValidationError("Inference features contain NaN or infinite values.")
    if scaling_engine is None:
        raise ModelDependencyError("engine.scaling is unavailable.")
    expected_feature_names = tuple(getattr(scaler, "feature_names", ()) or ())
    if expected_feature_names and feature_names is None:
        raise ModelSchemaError("feature_names are required to verify the trained feature order.")
    scaled = scaling_engine.apply_scaler(matrix, scaler, feature_names=feature_names)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(scaled))
        probability = float(output["tp_before_sl_probability"][0, 0].cpu().item())

    return {
        "probability": probability,
        "tp_before_sl_probability": probability,
        "predicted_probability_at_entry": probability,
        "predicted_outcome": "TP_FIRST" if probability >= threshold_value else "SL_FIRST",
        "classification_threshold": threshold_value,
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "calibration_status": "NOT_CALIBRATED",
    }


def predict_market(
    market: str,
    target_time: int,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[Any] = None,
    features: Optional[Any] = None,
    feature_names: Optional[Sequence[str]] = None,
    model_version_id: Optional[str] = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Load an approved model and predict from an explicit setup feature row."""
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    if features is None:
        raise ModelValidationError(
            "predict_market requires a feature row built with engine.dataset. "
            "A market timestamp or raw market_state_at vector is not a substitute for the TP-before-SL setup feature schema."
        )
    if isinstance(features, Mapping):
        if feature_names is None and isinstance(features.get("feature_names"), Sequence):
            feature_names = features.get("feature_names")
        features = features.get("features", features.get("feature_vector"))
        if features is None:
            raise ModelValidationError("Feature mapping must contain features or feature_vector.")

    model, scaler, checkpoint = memory_engine.load_model(
        normalized_market,
        identity=resolved,
        version_id=model_version_id,
        require_approved=True,
    )
    summary = checkpoint.get("training_summary", {})
    target_schema = summary.get("model_target_schema") if isinstance(summary, Mapping) else None
    if str(target_schema or "").upper() != MODEL_TARGET_SCHEMA:
        raise ModelSchemaError("Loaded checkpoint does not use the TP-before-SL binary target schema.")
    prediction = predict_from_features(
        model,
        scaler,
        features,
        feature_names=feature_names,
        threshold=threshold,
    )
    prediction.update(
        {
            "market": normalized_market,
            "target_time": int(target_time),
            "predicted_at": int(target_time),
            "identity": resolved.as_dict() if callable(getattr(resolved, "as_dict", None)) else dict(resolved),
            "model_version_id": checkpoint.get("model_version_id"),
            "dataset_version_id": checkpoint.get("dataset_version_id"),
            "scaler_version_id": checkpoint.get("scaler_version_id"),
        }
    )
    return prediction


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def show_model_info(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return metadata for the approved model assigned to one TradingBot."""
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    model, scaler, checkpoint = memory_engine.load_model(normalized_market, identity=resolved)
    summary = checkpoint.get("training_summary", {})
    result = {
        "identity": resolved.as_dict() if callable(getattr(resolved, "as_dict", None)) else dict(resolved),
        "market": normalized_market,
        "path": str(checkpoint.get("loaded_from", memory_engine.model_path(normalized_market, identity=resolved))),
        "input_features": int(model.input_size),
        "hidden_sizes": list(model.hidden_sizes),
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "dataset_version_id": checkpoint.get("dataset_version_id"),
        "scaler_version_id": checkpoint.get("scaler_version_id"),
        "model_version_id": checkpoint.get("model_version_id"),
        "scaler_feature_count": int(scaler.feature_count) if hasattr(scaler, "feature_count") else int(np.asarray(scaler.mean).size),
        "trained_at": checkpoint.get("trained_at"),
        "format_version": checkpoint.get("format_version"),
        "training_summary": dict(summary) if isinstance(summary, Mapping) else {},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--strategy-id", type=str, default=None)
    parser.add_argument("--tradingbot-id", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, inspect and query Spectline TP-before-SL binary models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a TradingBot TP-before-SL model.")
    train_parser.add_argument("market", type=str)
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    _add_identity_arguments(train_parser)

    info_parser = subparsers.add_parser("info", help="Show the approved model checkpoint.")
    info_parser.add_argument("market", type=str)
    _add_identity_arguments(info_parser)

    predict_parser = subparsers.add_parser("predict", help="Predict from a saved NumPy feature row.")
    predict_parser.add_argument("market", type=str)
    predict_parser.add_argument("time", type=str)
    predict_parser.add_argument("--features-file", type=str, required=True)
    predict_parser.add_argument("--feature-names-file", type=str, default=None)
    _add_identity_arguments(predict_parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_seed(SEED)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        identity_args = {
            "user_id": args.user_id,
            "strategy_id": args.strategy_id,
            "tradingbot_id": args.tradingbot_id,
        }
        if args.command == "train":
            result = train_market(
                market=args.market,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                **identity_args,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0
        if args.command == "info":
            show_model_info(args.market, **identity_args)
            return 0
        if args.command == "predict":
            try:
                target_time = int(float(args.time))
            except (TypeError, ValueError, OverflowError):
                parsed = datetime.fromisoformat(str(args.time).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                target_time = int(parsed.timestamp())
            feature_row = np.load(Path(args.features_file).expanduser(), allow_pickle=False)
            names = None
            if args.feature_names_file:
                loaded_names = np.load(Path(args.feature_names_file).expanduser(), allow_pickle=False)
                names = [str(item) for item in np.asarray(loaded_names).reshape(-1).tolist()]
            result = predict_market(
                market=args.market,
                target_time=target_time,
                features=feature_row,
                feature_names=names,
                **identity_args,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0
        parser.error(f"Unknown command: {args.command}")
    except Exception as error:
        print(f"[AI ERROR] {error}")
        return 1
    return 0


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "MODEL_FORMAT_VERSION",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET_NAME",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "TradingMicroModel",
    "ModelError",
    "ModelValidationError",
    "ModelSchemaError",
    "compute_loss",
    "masked_smooth_l1",
    "train_model",
    "train_market",
    "predict_from_features",
    "predict_market",
    "show_model_info",
    "build_parser",
    "main",
]
