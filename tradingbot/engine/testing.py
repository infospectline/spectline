"""Chronological out-of-sample historical testing for Spectline TP-before-SL models."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import importlib
import json
import math
import os
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# MODULE DEPENDENCIES
# ============================================================================

try:
    from . import dataset as dataset_engine
    from . import evaluation as evaluation_engine
    from . import memory as memory_engine
    from . import scaling as scaling_engine
    from . import versions as versions_engine
except ImportError:
    try:
        from engine import dataset as dataset_engine
        from engine import evaluation as evaluation_engine
        from engine import memory as memory_engine
        from engine import scaling as scaling_engine
        from engine import versions as versions_engine
    except ImportError:
        dataset_engine = None
        evaluation_engine = None
        memory_engine = None
        scaling_engine = None
        versions_engine = None


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

MODULE_SCHEMA_VERSION = 1
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET_NAME = "TP_BEFORE_SL"
TEST_RUN_DIRECTORY = "HISTORICAL_TESTS"
TEST_RUN_LOG_FILENAME = "TEST_RUNS.jsonl"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_MINIMUM_TEST_SAMPLES = 1
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
PROBABILITY_BUCKETS = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.000000000001),
)

_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class TestingError(RuntimeError):
    """Base exception for historical model testing failures."""


class TestingValidationError(TestingError, ValueError):
    """Raised when test data or a testing request violates the contract."""


class TestingDependencyError(TestingError, ImportError):
    """Raised when a required engine dependency is unavailable."""


class TestingIntegrityError(TestingError):
    """Raised when a model, dataset or report fails an integrity check."""


class TestingSchemaError(TestingError):
    """Raised when model and dataset schemas are incompatible."""


# ============================================================================
# VALIDATION AND SERIALIZATION HELPERS
# ============================================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generated_id(prefix: str = "historical_test") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:12]}"


def _safe_identifier(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise TestingValidationError(f"{name} is empty or contains unsupported characters.")
    return normalized


def _finite_float(value: Any, name: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise TestingValidationError(f"{name} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TestingValidationError(f"{name} must be a finite number.") from error
    if not math.isfinite(normalized):
        raise TestingValidationError(f"{name} must be a finite number.")
    if minimum is not None and normalized < minimum:
        raise TestingValidationError(f"{name} must be at least {minimum}.")
    if maximum is not None and normalized > maximum:
        raise TestingValidationError(f"{name} must not exceed {maximum}.")
    return normalized


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TestingValidationError("Testing report is not JSON serializable.") from error
    return text.encode("utf-8")


def _parse_timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise TestingSchemaError(f"{name} is missing or invalid.")
    if isinstance(value, (int, np.integer)):
        timestamp = int(value)
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise TestingSchemaError(f"{name} must be an integer Unix timestamp.")
        timestamp = int(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise TestingSchemaError(f"{name} is empty.")
        try:
            numeric = float(raw)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError
            timestamp = int(numeric)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as error:
                raise TestingSchemaError(f"{name} is not a supported timestamp.") from error
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = int(parsed.timestamp())
    else:
        raise TestingSchemaError(f"{name} has an unsupported timestamp type.")
    if timestamp < 0:
        raise TestingSchemaError(f"{name} cannot be negative.")
    return timestamp


def _identity_dict(identity: Any) -> Dict[str, str]:
    if identity is None:
        raise TestingValidationError("A complete TradingBot identity is required.")
    if isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        method = getattr(identity, "as_dict", None)
        if not callable(method):
            raise TestingValidationError("identity must be a mapping or expose as_dict().")
        raw = dict(method())
    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [key for key in required if raw.get(key) is None or not str(raw[key]).strip()]
    if missing:
        raise TestingValidationError("A complete TradingBot identity is required: " + ", ".join(missing))
    normalized = {key: str(raw[key]).strip() for key in required}
    normalized["market"] = normalized["market"].upper().replace("/", "_").replace("\\", "_")
    for key, value in normalized.items():
        if len(value) > 128 or value in {".", ".."} or any(char in value for char in ("/", "\\", "\x00")):
            raise TestingValidationError(f"Identity field {key} contains an unsafe value.")
    return normalized


def _record_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    method = getattr(record, "to_dict", None)
    if callable(method):
        result = method()
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _module_dependency(module: Any, name: str) -> Any:
    if module is None:
        raise TestingDependencyError(f"{name} is unavailable.")
    return module


# ============================================================================
# DATASET AND CHRONOLOGY VALIDATION
# ============================================================================

def _normalize_dataset_arrays(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(dataset, Mapping):
        raise TestingValidationError("dataset must be a mapping.")
    if dataset.get("target") != DATASET_TARGET_NAME:
        raise TestingSchemaError("Dataset target must be TP_BEFORE_SL.")
    schema = str(dataset.get("model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise TestingSchemaError("Dataset target schema is not TP-before-SL binary classification.")

    try:
        features = np.asarray(dataset.get("features"), dtype=np.float32)
        targets = np.asarray(dataset.get("targets"), dtype=np.int64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise TestingValidationError("Dataset features and targets must be numeric arrays.") from error
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise TestingValidationError("Dataset features must be a non-empty two-dimensional matrix.")
    if features.shape[0] != targets.shape[0]:
        raise TestingSchemaError("Dataset feature and target row counts do not match.")
    if not np.all(np.isfinite(features)):
        raise TestingValidationError("Dataset features contain NaN or infinite values.")
    if not np.all(np.isin(targets, (0, 1))):
        raise TestingSchemaError("Dataset targets must contain only 0=SL_FIRST and 1=TP_FIRST.")

    feature_names = tuple(str(item) for item in dataset.get("feature_names", ()))
    if len(feature_names) != features.shape[1] or len(set(feature_names)) != len(feature_names):
        raise TestingSchemaError("Dataset feature_names do not match the feature matrix width.")
    lowered_names = tuple(name.lower() for name in feature_names)
    forbidden_tokens = (
        "actual_outcome",
        "outcome_known_at",
        "exit_time",
        "exit_price",
        "label_target",
        "label_id",
        "tp_hit",
        "sl_hit",
    )
    leaked_names = [
        name for name, lowered in zip(feature_names, lowered_names)
        if any(token in lowered for token in forbidden_tokens)
    ]
    if leaked_names:
        raise TestingSchemaError("Dataset feature schema contains outcome or label fields: " + ", ".join(leaked_names[:10]))

    sample_metadata = dataset.get("sample_metadata")
    label_metadata = dataset.get("label_metadata")
    if not isinstance(sample_metadata, Sequence) or isinstance(sample_metadata, (str, bytes)):
        raise TestingSchemaError("Dataset sample_metadata must be a sequence.")
    if not isinstance(label_metadata, Sequence) or isinstance(label_metadata, (str, bytes)):
        raise TestingSchemaError("Dataset label_metadata must be a sequence.")
    if len(sample_metadata) != targets.size or len(label_metadata) != targets.size:
        raise TestingSchemaError("Dataset metadata rows must align one-to-one with targets.")
    for index, row in enumerate(sample_metadata):
        if not isinstance(row, Mapping):
            raise TestingSchemaError(f"sample_metadata[{index}] must be a mapping.")
    for index, row in enumerate(label_metadata):
        if not isinstance(row, Mapping):
            raise TestingSchemaError(f"label_metadata[{index}] must be a mapping.")
        raw_outcome = row.get("actual_outcome")
        if isinstance(raw_outcome, bool):
            raise TestingSchemaError(f"label_metadata[{index}] outcome must be an integer 0 or 1.")
        try:
            numeric_outcome = float(raw_outcome)
        except (TypeError, ValueError, OverflowError) as error:
            raise TestingSchemaError(f"label_metadata[{index}] outcome must be an integer 0 or 1.") from error
        if not math.isfinite(numeric_outcome) or not numeric_outcome.is_integer() or int(numeric_outcome) not in (0, 1):
            raise TestingSchemaError(f"label_metadata[{index}] outcome must be an integer 0 or 1.")
        if int(numeric_outcome) != int(targets[index]):
            raise TestingSchemaError(f"label_metadata[{index}] outcome does not match the target array.")

    splits = dataset.get("splits")
    if not isinstance(splits, Mapping):
        raise TestingSchemaError("Dataset is missing chronological splits.")
    normalized_splits: Dict[str, List[int]] = {}
    all_seen: set[int] = set()
    for name in ("train_indices", "validation_indices", "test_indices"):
        values = splits.get(name, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TestingSchemaError(f"splits.{name} must be a sequence.")
        selected: List[int] = []
        for raw_index in values:
            if isinstance(raw_index, bool):
                raise TestingSchemaError(f"splits.{name} contains an invalid index.")
            try:
                index = int(raw_index)
            except (TypeError, ValueError, OverflowError) as error:
                raise TestingSchemaError(f"splits.{name} contains an invalid index.") from error
            if index < 0 or index >= targets.size or index != raw_index:
                raise TestingSchemaError(f"splits.{name} contains an out-of-range or non-integer index.")
            if index in all_seen:
                raise TestingSchemaError("Chronological dataset splits overlap.")
            all_seen.add(index)
            selected.append(index)
        normalized_splits[name] = selected

    if not normalized_splits["train_indices"]:
        raise TestingSchemaError("A historical test requires a non-empty training split.")
    if not normalized_splits["validation_indices"]:
        raise TestingSchemaError("A historical test requires a non-empty validation split.")
    if not normalized_splits["test_indices"]:
        raise TestingSchemaError("A historical test requires a non-empty held-out test split.")

    feature_schema = dataset.get("feature_schema", {})
    schema_hash = None
    if isinstance(feature_schema, Mapping):
        schema_hash = feature_schema.get("sha256")
    schema_hash = schema_hash or dataset.get("feature_schema_hash")
    if not schema_hash:
        raise TestingSchemaError("Dataset does not contain a feature schema hash.")

    return {
        "features": features,
        "targets": targets,
        "feature_names": feature_names,
        "sample_metadata": list(sample_metadata),
        "label_metadata": list(label_metadata),
        "splits": normalized_splits,
        "feature_schema_hash": str(schema_hash),
    }


def _validate_point_in_time_rows(data: Mapping[str, Any], indices: Sequence[int]) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for index in indices:
        sample = data["sample_metadata"][index]
        label = data["label_metadata"][index]
        decision_time = _parse_timestamp(sample.get("decision_time"), f"sample_metadata[{index}].decision_time")
        entry_time = _parse_timestamp(label.get("entry_time"), f"label_metadata[{index}].entry_time")
        exit_time = _parse_timestamp(label.get("exit_time"), f"label_metadata[{index}].exit_time")
        outcome_known_at = _parse_timestamp(
            label.get("outcome_known_at"),
            f"label_metadata[{index}].outcome_known_at",
        )
        if decision_time > entry_time:
            raise TestingSchemaError(f"Test sample {index} uses a decision time after trade entry.")
        if exit_time < entry_time:
            raise TestingSchemaError(f"Test sample {index} has exit_time before entry_time.")
        if outcome_known_at < exit_time:
            raise TestingSchemaError(f"Test sample {index} has outcome_known_at before exit_time.")
        rows.append(
            {
                "index": int(index),
                "decision_time": decision_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "outcome_known_at": outcome_known_at,
            }
        )
    return rows


def _validate_chronological_purge(data: Mapping[str, Any]) -> Dict[str, Any]:
    split_rows: Dict[str, List[Dict[str, int]]] = {}
    for split_name in ("train", "validation", "test"):
        indices = data["splits"][f"{split_name}_indices"]
        split_rows[split_name] = _validate_point_in_time_rows(data, indices)
        ordered = sorted(split_rows[split_name], key=lambda item: (item["decision_time"], item["index"]))
        if ordered != split_rows[split_name]:
            raise TestingSchemaError(f"The {split_name} split is not stored in chronological decision-time order.")

    train_rows = split_rows["train"]
    validation_rows = split_rows["validation"]
    test_rows = split_rows["test"]
    validation_start = min(row["decision_time"] for row in validation_rows)
    test_start = min(row["decision_time"] for row in test_rows)
    train_max_decision = max(row["decision_time"] for row in train_rows)
    validation_max_decision = max(row["decision_time"] for row in validation_rows)
    train_max_known = max(row["outcome_known_at"] for row in train_rows)
    validation_max_known = max(row["outcome_known_at"] for row in validation_rows)

    if train_max_decision >= validation_start:
        raise TestingSchemaError("Training decisions do not precede validation decisions.")
    if validation_max_decision >= test_start:
        raise TestingSchemaError("Validation decisions do not precede test decisions.")
    if train_max_known >= validation_start:
        raise TestingSchemaError("Training outcomes were not fully known before validation began.")
    if validation_max_known >= test_start:
        raise TestingSchemaError("Validation outcomes were not fully known before testing began.")

    test_rows_sorted = sorted(test_rows, key=lambda item: (item["decision_time"], item["index"]))
    return {
        "mode": str((data.get("raw_splits") or {}).get("mode", "CHRONOLOGICAL_PURGED")),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "train_max_decision_time": train_max_decision,
        "train_max_outcome_known_at": train_max_known,
        "validation_start_time": validation_start,
        "validation_max_decision_time": validation_max_decision,
        "validation_max_outcome_known_at": validation_max_known,
        "test_start_time": test_start,
        "test_end_time": max(row["decision_time"] for row in test_rows_sorted),
        "test_rows": test_rows_sorted,
        "purge_verified": True,
    }


def _extract_dataset_identity(dataset: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    raw = dataset.get("identity")
    if raw is None:
        return None
    try:
        return _identity_dict(raw)
    except TestingValidationError:
        return None


# ============================================================================
# MODEL AND SCALER INFERENCE
# ============================================================================

def _predict_probabilities(model: Any, scaler: Any, data: Mapping[str, Any], indices: Sequence[int]) -> np.ndarray:
    if torch is None:
        raise TestingDependencyError("PyTorch is required for historical model testing.")
    _module_dependency(scaling_engine, "engine.scaling")
    if not callable(model):
        raise TestingValidationError("model must be callable.")
    if getattr(scaler, "fit_scope", None) != "TRAIN_SPLIT_ONLY":
        raise TestingSchemaError("Scaler must be fitted exclusively on the training split.")
    if int(getattr(scaler, "fit_sample_count", 0)) != len(data["splits"]["train_indices"]):
        raise TestingSchemaError("Scaler fit sample count does not match the dataset training split.")
    scaler_names = tuple(getattr(scaler, "feature_names", ()) or ())
    if scaler_names and scaler_names != data["feature_names"]:
        raise TestingSchemaError("Dataset feature order does not match the trained scaler.")
    dataset_id = getattr(scaler, "fit_dataset_id", None)
    if dataset_id and str(dataset_id) != str(data.get("dataset_id", "")):
        raise TestingSchemaError("Scaler was fitted on a different dataset version or content dataset.")
    expected_fit_indices_hash = hashlib.sha256(
        json.dumps(data["splits"]["train_indices"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    scaler_fit_indices_hash = getattr(scaler, "fit_indices_hash", None)
    if not scaler_fit_indices_hash or str(scaler_fit_indices_hash) != expected_fit_indices_hash:
        raise TestingSchemaError("Scaler fit indices do not match the exact chronological training split.")
    scaler_hash = getattr(scaler, "feature_schema_hash", None)
    if scaler_hash and str(scaler_hash) != data["feature_schema_hash"]:
        raise TestingSchemaError("Dataset feature schema hash does not match the trained scaler.")

    input_size = int(getattr(model, "input_size", 0))
    if input_size != data["features"].shape[1]:
        raise TestingSchemaError("Model input width does not match the dataset feature width.")

    selected_features = data["features"][np.asarray(indices, dtype=np.int64)]
    scaled = scaling_engine.apply_scaler(
        selected_features,
        scaler,
        feature_names=data["feature_names"],
    )
    try:
        parameters = list(model.parameters())
        device = parameters[0].device if parameters else torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    was_training = bool(getattr(model, "training", False))
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        with torch.no_grad():
            tensor = torch.as_tensor(scaled, dtype=torch.float32, device=device)
            outputs = model(tensor)
            probability_tensor = None
            if isinstance(outputs, Mapping):
                probability_tensor = outputs.get("tp_before_sl_probability")
                if probability_tensor is None and outputs.get("tp_before_sl_logit") is not None:
                    probability_tensor = torch.sigmoid(outputs["tp_before_sl_logit"])
            elif isinstance(outputs, torch.Tensor):
                if outputs.ndim == 2 and outputs.shape[1] == 1:
                    probability_tensor = torch.sigmoid(outputs)
                else:
                    probability_tensor = outputs
            if not isinstance(probability_tensor, torch.Tensor):
                raise TestingSchemaError("Model output does not expose TP-before-SL probabilities.")
            probabilities = probability_tensor.detach().to("cpu").numpy().reshape(-1).astype(np.float64)
    finally:
        if was_training and callable(train_method):
            train_method()

    if probabilities.size != len(indices):
        raise TestingSchemaError("Model output row count does not match the selected test rows.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise TestingSchemaError("Model returned invalid probabilities outside [0, 1].")
    return probabilities


# ============================================================================
# HISTORICAL TEST REPORTS
# ============================================================================

def _overlap_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    events: List[Tuple[int, int]] = []
    for row in rows:
        entry_time = int(row["entry_time"])
        exit_time = int(row["exit_time"])
        events.append((entry_time, 1))
        events.append((exit_time, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    max_concurrent = 0
    for _, delta in events:
        active += delta
        max_concurrent = max(max_concurrent, active)
    return {"maximum_concurrent_trades": int(max_concurrent)}


def _monthly_summary(rows: Sequence[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        month = datetime.fromtimestamp(int(row["decision_time"]), tz=timezone.utc).strftime("%Y-%m")
        buckets[month].append(row)
    report: List[Dict[str, Any]] = []
    for month in sorted(buckets):
        values = buckets[month]
        targets = np.asarray([int(item["actual_outcome"]) for item in values], dtype=np.int64)
        probabilities = np.asarray([float(item["probability"]) for item in values], dtype=np.float64)
        predicted = (probabilities >= threshold).astype(np.int64)
        report.append(
            {
                "month": month,
                "sample_count": int(len(values)),
                "tp_first_count": int(np.sum(targets == 1)),
                "sl_first_count": int(np.sum(targets == 0)),
                "actual_tp_first_rate": float(np.mean(targets)),
                "mean_predicted_probability": float(np.mean(probabilities)),
                "classification_threshold": float(threshold),
                "accuracy_at_threshold": float(np.mean(predicted == targets)),
                "brier_score": float(np.mean(np.square(probabilities - targets))),
            }
        )
    return report


def _probability_bucket_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for lower, upper in PROBABILITY_BUCKETS:
        selected = [
            row for row in rows
            if lower <= float(row["probability"]) < upper
        ]
        if not selected:
            result.append(
                {
                    "lower_bound": float(lower),
                    "upper_bound": float(min(upper, 1.0)),
                    "sample_count": 0,
                    "mean_predicted_probability": None,
                    "actual_tp_first_rate": None,
                }
            )
            continue
        result.append(
            {
                "lower_bound": float(lower),
                "upper_bound": float(min(upper, 1.0)),
                "sample_count": int(len(selected)),
                "mean_predicted_probability": float(np.mean([float(item["probability"]) for item in selected])),
                "actual_tp_first_rate": float(np.mean([int(item["actual_outcome"]) for item in selected])),
            }
        )
    return result


def _build_trade_results(
    data: Mapping[str, Any],
    test_rows: Sequence[Mapping[str, int]],
    probabilities: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    if probabilities.size != len(test_rows):
        raise TestingSchemaError("Probability and test metadata counts do not match.")
    results: List[Dict[str, Any]] = []
    ordered = sorted(test_rows, key=lambda item: (item["decision_time"], item["index"]))
    probability_by_index = {int(row["index"]): float(probability) for row, probability in zip(test_rows, probabilities)}
    for sequence_index, chronology in enumerate(ordered):
        index = int(chronology["index"])
        sample = data["sample_metadata"][index]
        label = data["label_metadata"][index]
        actual = int(data["targets"][index])
        probability = probability_by_index[index]
        predicted = int(probability >= threshold)
        results.append(
            {
                "sequence_index": sequence_index,
                "dataset_row_index": index,
                "trade_id": str(label.get("trade_id", sample.get("trade_id", ""))),
                "setup_id": str(label.get("setup_id", sample.get("setup_id", ""))),
                "decision_time": int(chronology["decision_time"]),
                "entry_time": int(chronology["entry_time"]),
                "exit_time": int(chronology["exit_time"]),
                "outcome_known_at": int(chronology["outcome_known_at"]),
                "probability": probability,
                "classification_threshold": float(threshold),
                "predicted_outcome": "TP_FIRST" if predicted == 1 else "SL_FIRST",
                "actual_outcome": actual,
                "actual_outcome_name": "TP_FIRST" if actual == 1 else "SL_FIRST",
                "correct": bool(predicted == actual),
                "brier_contribution": float((probability - actual) ** 2),
            }
        )
    return results


def _baseline_metrics(training_targets: np.ndarray, test_targets: np.ndarray, threshold: float) -> Dict[str, Any]:
    if training_targets.size == 0 or test_targets.size == 0:
        return {"status": "UNAVAILABLE", "reason": "Training or test targets are empty."}
    prior = float(np.mean(training_targets))
    clipped = min(max(prior, 1e-7), 1.0 - 1e-7)
    probabilities = np.full(test_targets.shape[0], clipped, dtype=np.float64)
    log_loss = float(-np.mean(test_targets * np.log(probabilities) + (1 - test_targets) * np.log(1.0 - probabilities)))
    brier = float(np.mean(np.square(probabilities - test_targets)))
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "status": "AVAILABLE",
        "baseline": "TRAINING_SET_TP_FIRST_PREVALENCE",
        "training_tp_first_rate": prior,
        "classification_threshold": float(threshold),
        "test_accuracy_at_threshold": float(np.mean(predictions == test_targets)),
        "test_log_loss": log_loss,
        "test_brier_score": brier,
    }


def run_historical_test(
    model: Any,
    scaler: Any,
    dataset: Mapping[str, Any],
    *,
    model_version_id: Optional[str] = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    minimum_test_samples: int = DEFAULT_MINIMUM_TEST_SAMPLES,
    identity: Optional[Any] = None,
    run_id: Optional[str] = None,
    persist: bool = False,
) -> Dict[str, Any]:
    """Run a frozen-model forward test on the dataset's untouched chronological test split."""
    if evaluation_engine is None or scaling_engine is None:
        raise TestingDependencyError("engine.evaluation and engine.scaling are required.")
    if torch is None:
        raise TestingDependencyError("PyTorch is required for historical testing.")
    threshold_value = _finite_float(threshold, "threshold", minimum=0.0, maximum=1.0)
    if threshold_value in (0.0, 1.0):
        raise TestingValidationError("threshold must be strictly between 0 and 1.")
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, (int, np.integer)) or int(calibration_bins) < 2 or int(calibration_bins) > 100:
        raise TestingValidationError("calibration_bins must be an integer between 2 and 100.")
    if isinstance(minimum_test_samples, bool) or not isinstance(minimum_test_samples, (int, np.integer)) or int(minimum_test_samples) < 1:
        raise TestingValidationError("minimum_test_samples must be a positive integer.")

    normalized = _normalize_dataset_arrays(dataset)
    validator = getattr(dataset_engine, "validate_training_dataset", None) if dataset_engine is not None else None
    if callable(validator):
        dataset_validation = validator(dataset)
        if not isinstance(dataset_validation, Mapping) or dataset_validation.get("valid") is not True:
            errors = dataset_validation.get("errors", []) if isinstance(dataset_validation, Mapping) else []
            raise TestingSchemaError("Dataset validation failed: " + "; ".join(str(item) for item in errors))
    normalized["dataset_id"] = str(dataset.get("dataset_id", ""))
    if not normalized["dataset_id"]:
        raise TestingSchemaError("Dataset is missing its immutable dataset_id.")
    normalized["raw_splits"] = dataset.get("splits", {})
    chronology = _validate_chronological_purge(normalized)
    test_rows = chronology["test_rows"]
    if len(test_rows) < int(minimum_test_samples):
        raise TestingValidationError(
            f"The held-out test split contains {len(test_rows)} rows; {minimum_test_samples} are required."
        )
    test_indices = [int(row["index"]) for row in test_rows]
    test_targets = normalized["targets"][np.asarray(test_indices, dtype=np.int64)]
    if not np.all(np.isin(test_targets, (0, 1))):
        raise TestingSchemaError("The held-out test targets are not binary.")

    model_schema = str(getattr(model, "model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if model_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise TestingSchemaError("Loaded model does not use the TP-before-SL target schema.")
    probabilities = _predict_probabilities(model, scaler, normalized, test_indices)
    metrics_report = evaluation_engine.evaluate_predictions(
        test_targets,
        probabilities,
        threshold=threshold_value,
        calibration_bins=int(calibration_bins),
        split="test",
    )
    trade_results = _build_trade_results(normalized, test_rows, probabilities, threshold_value)

    train_targets = normalized["targets"][np.asarray(normalized["splits"]["train_indices"], dtype=np.int64)]
    validation_targets = normalized["targets"][np.asarray(normalized["splits"]["validation_indices"], dtype=np.int64)]
    test_time_range = {
        "start": min(row["decision_time"] for row in test_rows),
        "end": max(row["decision_time"] for row in test_rows),
    }
    known_time_range = {
        "start": min(row["outcome_known_at"] for row in test_rows),
        "end": max(row["outcome_known_at"] for row in test_rows),
    }
    dataset_identity = _extract_dataset_identity(dataset)
    supplied_identity = _identity_dict(identity) if identity is not None else dataset_identity
    if supplied_identity is not None and dataset_identity is not None and supplied_identity != dataset_identity:
        raise TestingValidationError("Dataset identity does not match the requested TradingBot identity.")

    target_counts = {
        "SL_FIRST": int(np.sum(test_targets == 0)),
        "TP_FIRST": int(np.sum(test_targets == 1)),
    }
    report: Dict[str, Any] = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "testing_run_id": _safe_identifier(run_id or _generated_id(), "run_id"),
        "created_at": _utc_now(),
        "testing_mode": "FROZEN_MODEL_CHRONOLOGICAL_FORWARD_TEST",
        "target": DATASET_TARGET_NAME,
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "model_version_id": str(model_version_id) if model_version_id else None,
        "dataset_id": str(dataset.get("dataset_id", "")),
        "identity": supplied_identity,
        "split": "test",
        "threshold": threshold_value,
        "sample_count": int(len(test_targets)),
        "feature_count": int(normalized["features"].shape[1]),
        "feature_schema_hash": normalized["feature_schema_hash"],
        "test_target_counts": target_counts,
        "chronology": {
            key: value for key, value in chronology.items() if key != "test_rows"
        },
        "test_decision_time_range": test_time_range,
        "test_outcome_known_time_range": known_time_range,
        "metrics": metrics_report["metrics"],
        "calibration_status": metrics_report.get("calibration_status", "ASSESSED_NOT_CALIBRATED"),
        "training_prevalence_baseline": _baseline_metrics(train_targets, test_targets, threshold_value),
        "monthly_summary": _monthly_summary(trade_results, threshold_value),
        "probability_bucket_summary": _probability_bucket_summary(trade_results),
        "overlap_summary": _overlap_summary(trade_results),
        "trade_results": trade_results,
        "integrity": {
            "chronological_purge_verified": True,
            "features_checked_for_label_field_names": True,
            "scaler_fit_scope": str(getattr(scaler, "fit_scope", "")),
            "scaler_fit_sample_count": int(getattr(scaler, "fit_sample_count", 0)),
            "train_sample_count": int(len(train_targets)),
            "validation_sample_count": int(len(validation_targets)),
            "test_sample_count": int(len(test_targets)),
            "test_rows_excluded_from_scaler_fit": True,
            "test_rows_excluded_from_model_training": True,
            "model_weights_frozen_during_test": True,
        },
        "limitations": [
            "This is a frozen-model test on precomputed point-in-time setup features, not a tick-level or candle-by-candle execution simulator.",
            "The dataset labels encode TP-first or SL-first outcomes; they do not provide enough information to calculate realized portfolio PnL, spread, slippage, fills or position sizing.",
            "The model is not retrained, recalibrated or threshold-tuned on the held-out test split.",
            "Historical probabilities remain NOT_CALIBRATED unless a separate calibration procedure is completed.",
        ],
        "approval_performed": False,
        "model_state_changed": False,
    }
    if persist:
        if supplied_identity is None:
            raise TestingValidationError("Persisting a test report requires a complete TradingBot identity.")
        report["persistence"] = _persist_test_report(report, supplied_identity)
    else:
        report["persistence"] = {"persisted": False}
    return report


# ============================================================================
# VERSIONED MODEL TESTING
# ============================================================================

def _load_identity(market: str, identity: Any) -> Any:
    _module_dependency(memory_engine, "engine.memory")
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if not callable(resolver):
        raise TestingDependencyError("engine.memory identity resolver is unavailable.")
    resolved = resolver(market, identity=identity)
    if _identity_dict(resolved) != _identity_dict(identity):
        raise TestingValidationError("Resolved TradingBot identity does not match the supplied identity.")
    return resolved


def run_model_version_test(
    market: str,
    model_version_id: str,
    *,
    identity: Any,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    minimum_test_samples: int = DEFAULT_MINIMUM_TEST_SAMPLES,
    persist: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one immutable model version and test it on its linked held-out dataset."""
    for module, name in (
        (memory_engine, "engine.memory"),
        (versions_engine, "engine.versions"),
    ):
        _module_dependency(module, name)
    normalized_market = str(market).strip().upper().replace("/", "_").replace("\\", "_")
    if not normalized_market:
        raise TestingValidationError("market cannot be empty.")
    selected_model_version = _safe_identifier(model_version_id, "model_version_id")
    resolved_identity = _load_identity(normalized_market, identity)
    identity_values = _identity_dict(resolved_identity)
    if identity_values["market"] != normalized_market:
        raise TestingValidationError("Requested market does not match the supplied TradingBot identity.")

    registry = versions_engine.get_default_registry()
    model_record = registry.get_version("model", selected_model_version, identity=resolved_identity)
    record_status = str(getattr(model_record, "status", "")).upper()
    if record_status in {"REJECTED", "RETIRED"}:
        raise TestingValidationError(f"Model version in {record_status} state cannot be tested.")
    metadata = getattr(model_record, "metadata", {})
    if not isinstance(metadata, Mapping) or str(metadata.get("model_target_schema", "")).upper() not in {
        MODEL_TARGET_SCHEMA,
        "TP_BEFORE_SL",
    }:
        raise TestingSchemaError("Model registry record does not identify a TP-before-SL binary model.")

    verification = registry.verify_artifact("model", selected_model_version, identity=resolved_identity)
    if not verification.get("valid"):
        raise TestingIntegrityError("Model artifact failed its registered integrity check.")

    model, scaler, checkpoint = memory_engine.load_model(
        normalized_market,
        identity=resolved_identity,
        version_id=selected_model_version,
        require_approved=False,
    )
    training_summary = checkpoint.get("training_summary", {})
    checkpoint_schema = training_summary.get("model_target_schema") if isinstance(training_summary, Mapping) else None
    if str(checkpoint_schema or "").upper() != MODEL_TARGET_SCHEMA:
        raise TestingSchemaError("Model checkpoint target schema is incompatible.")
    if checkpoint.get("identity") != identity_values:
        raise TestingValidationError("Model checkpoint identity does not match the requested TradingBot.")
    dataset_version_id = checkpoint.get("dataset_version_id")
    if not dataset_version_id:
        raise TestingSchemaError("Model checkpoint does not reference an immutable dataset version.")
    scaler_version_id = checkpoint.get("scaler_version_id")
    if not scaler_version_id:
        raise TestingSchemaError("Model checkpoint does not reference an immutable scaler version.")

    dataset_verification = registry.verify_artifact(
        "dataset",
        str(dataset_version_id),
        identity=resolved_identity,
    )
    scaler_verification = registry.verify_artifact(
        "scaler",
        str(scaler_version_id),
        identity=resolved_identity,
    )
    if not dataset_verification.get("valid") or not scaler_verification.get("valid"):
        raise TestingIntegrityError("Dataset or scaler dependency failed its registered integrity check.")

    dataset = memory_engine.load_training_dataset(
        str(dataset_version_id),
        identity=resolved_identity,
        require_approved=False,
    )
    if _extract_dataset_identity(dataset) != identity_values:
        raise TestingValidationError("Stored dataset belongs to another TradingBot identity.")
    dataset_feature_schema = dataset.get("feature_schema", {})
    dataset_feature_hash = (
        dataset_feature_schema.get("sha256") if isinstance(dataset_feature_schema, Mapping) else None
    ) or dataset.get("feature_schema_hash")
    registered_model_feature_hash = metadata.get("feature_schema_hash")
    summary_dataset_version = training_summary.get("dataset_version_id") if isinstance(training_summary, Mapping) else None
    summary_scaler_version = training_summary.get("scaler_version_id") if isinstance(training_summary, Mapping) else None
    if summary_dataset_version and str(summary_dataset_version) != str(dataset_version_id):
        raise TestingIntegrityError("Model training summary references a different dataset version.")
    if summary_scaler_version and str(summary_scaler_version) != str(scaler_version_id):
        raise TestingIntegrityError("Model training summary references a different scaler version.")
    if registered_model_feature_hash and str(registered_model_feature_hash) != str(dataset_feature_hash):
        raise TestingSchemaError("Registered model feature schema hash does not match its linked dataset.")
    summary_feature_hash = training_summary.get("feature_schema_hash") if isinstance(training_summary, Mapping) else None
    if summary_feature_hash and str(summary_feature_hash) != str(dataset_feature_hash):
        raise TestingSchemaError("Model training summary feature schema hash does not match its linked dataset.")
    dataset_record = registry.get_version("dataset", str(dataset_version_id), identity=resolved_identity)
    if str(getattr(dataset_record, "status", "")).upper() != "APPROVED":
        raise TestingValidationError("The model's linked training dataset is not approved.")
    scaler_record = registry.get_version("scaler", str(scaler_version_id), identity=resolved_identity)
    if str(getattr(scaler_record, "status", "")).upper() != "APPROVED":
        raise TestingValidationError("The model's linked scaler is not approved.")

    report = run_historical_test(
        model,
        scaler,
        dataset,
        model_version_id=selected_model_version,
        threshold=threshold,
        calibration_bins=calibration_bins,
        minimum_test_samples=minimum_test_samples,
        identity=resolved_identity,
        run_id=run_id,
        persist=False,
    )
    report.update(
        {
            "market": normalized_market,
            "model_version_status": record_status,
            "model_artifact_sha256": getattr(model_record, "artifact_sha256", None),
            "dataset_version_id": str(dataset_version_id),
            "dataset_artifact_sha256": getattr(dataset_record, "artifact_sha256", None),
            "scaler_version_id": str(scaler_version_id),
            "scaler_artifact_sha256": getattr(scaler_record, "artifact_sha256", None),
            "training_selection_metric": training_summary.get("selection_metric") if isinstance(training_summary, Mapping) else None,
            "training_validation_available": training_summary.get("validation_available") is True if isinstance(training_summary, Mapping) else False,
            "version_integrity": {
                "model": verification,
                "dataset": dataset_verification,
                "scaler": scaler_verification,
            },
        }
    )
    if report["training_validation_available"] is not True:
        raise TestingValidationError("Model training did not have a validation split; formal historical testing is blocked.")
    if persist:
        report["persistence"] = _persist_test_report(report, identity_values)
    return report


def run_testing(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility alias for versioned historical model testing."""
    return run_model_version_test(*args, **kwargs)


# ============================================================================
# IMMUTABLE TEST REPORT STORAGE
# ============================================================================

def _identity_lock_key(identity: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_identity_lock(identity: Mapping[str, str]) -> threading.RLock:
    key = _identity_lock_key(identity)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def _results_root(identity: Mapping[str, str]) -> Path:
    _module_dependency(memory_engine, "engine.memory")
    getter = getattr(memory_engine, "get_results_dir", None)
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if not callable(getter) or not callable(resolver):
        raise TestingDependencyError("engine.memory does not expose the customer RESULTS identity resolver.")
    try:
        resolved_identity = resolver(str(identity["market"]), identity=identity)
        if _identity_dict(resolved_identity) != _identity_dict(identity):
            raise TestingValidationError("Resolved RESULTS identity does not match the requested TradingBot.")
        root = Path(getter(resolved_identity)).resolve()
    except TestingError:
        raise
    except Exception as error:
        raise TestingDependencyError("Customer RESULTS directory could not be resolved.") from error
    return root


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.link(temporary, path)
    except FileExistsError as error:
        raise TestingIntegrityError("An immutable test report with this ID already exists.") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = _canonical_json(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _persist_test_report(report: Mapping[str, Any], identity: Mapping[str, str]) -> Dict[str, Any]:
    normalized_identity = _identity_dict(identity)
    testing_run_id = _safe_identifier(report.get("testing_run_id"), "testing_run_id")
    root = _results_root(normalized_identity) / TEST_RUN_DIRECTORY
    report_path = root / f"{testing_run_id}.json"
    encoded = _canonical_json(report)
    digest = hashlib.sha256(encoded).hexdigest()
    lock = _get_identity_lock(normalized_identity)
    with lock:
        _atomic_create(report_path, encoded)
        event = {
            "schema_version": MODULE_SCHEMA_VERSION,
            "event_id": _generated_id("testing_event"),
            "event_type": "HISTORICAL_TEST_COMPLETED",
            "testing_run_id": testing_run_id,
            "timestamp": _utc_now(),
            "identity": normalized_identity,
            "model_version_id": report.get("model_version_id"),
            "dataset_version_id": report.get("dataset_version_id"),
            "scaler_version_id": report.get("scaler_version_id"),
            "sample_count": report.get("sample_count"),
            "report_path": str(report_path),
            "report_sha256": digest,
        }
        try:
            _append_jsonl(root / TEST_RUN_LOG_FILENAME, event)
        except Exception:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    return {
        "persisted": True,
        "report_path": str(report_path),
        "report_sha256": digest,
        "event_log_path": str(root / TEST_RUN_LOG_FILENAME),
    }


def list_testing_runs(*, identity: Any, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """List append-only historical test records for one TradingBot."""
    normalized_identity = _identity_dict(identity)
    if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
        raise TestingValidationError("limit must be a positive integer when supplied.")
    log_path = _results_root(normalized_identity) / TEST_RUN_DIRECTORY / TEST_RUN_LOG_FILENAME
    if not log_path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TestingIntegrityError(f"Testing event log is malformed at line {line_number}.") from error
            if not isinstance(record, Mapping):
                raise TestingIntegrityError(f"Testing event log contains an invalid record at line {line_number}.")
            if record.get("identity") != normalized_identity:
                raise TestingIntegrityError("Testing event log contains a record for another identity.")
            records.append(dict(record))
    records.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return records[: int(limit)] if limit is not None else records


def load_test_report(testing_run_id: str, *, identity: Any, verify: bool = True) -> Dict[str, Any]:
    """Load one immutable historical test report and verify its recorded checksum."""
    normalized_identity = _identity_dict(identity)
    selected_id = _safe_identifier(testing_run_id, "testing_run_id")
    log_records = list_testing_runs(identity=normalized_identity)
    matching = [item for item in log_records if item.get("testing_run_id") == selected_id]
    if len(matching) != 1:
        raise FileNotFoundError("Historical test report was not found for this TradingBot.")
    event = matching[0]
    path = Path(str(event.get("report_path", ""))).resolve()
    expected_root = (_results_root(normalized_identity) / TEST_RUN_DIRECTORY).resolve()
    try:
        path.relative_to(expected_root)
    except ValueError as error:
        raise TestingIntegrityError("Stored test report path escapes the TradingBot RESULTS directory.") from error
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TestingIntegrityError("Stored historical test report cannot be read.") from error
    digest = hashlib.sha256(payload).hexdigest()
    if verify and digest != event.get("report_sha256"):
        raise TestingIntegrityError("Historical test report failed its SHA-256 integrity check.")
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TestingIntegrityError("Historical test report is not valid JSON.") from error
    if not isinstance(report, Mapping) or report.get("identity") != normalized_identity:
        raise TestingIntegrityError("Historical test report identity does not match the requested TradingBot.")
    return {**dict(report), "report_verification": {"valid": digest == event.get("report_sha256"), "sha256": digest}}


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "MODULE_SCHEMA_VERSION",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET_NAME",
    "TEST_RUN_DIRECTORY",
    "TEST_RUN_LOG_FILENAME",
    "TestingError",
    "TestingValidationError",
    "TestingDependencyError",
    "TestingIntegrityError",
    "TestingSchemaError",
    "run_historical_test",
    "run_model_version_test",
    "run_testing",
    "list_testing_runs",
    "load_test_report",
]
