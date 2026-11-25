"""Independent evaluation and approval gates for Spectline TP-before-SL models."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
import os
import re
import tempfile
import uuid

from dataclasses import asdict, dataclass
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
    from . import memory as memory_engine
    from . import scaling as scaling_engine

except ImportError:

    try:
        from engine import dataset as dataset_engine
        from engine import memory as memory_engine
        from engine import scaling as scaling_engine

    except ImportError:

        dataset_engine = None
        memory_engine = None
        scaling_engine = None


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

MODULE_SCHEMA_VERSION = 1
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET_NAME = "TP_BEFORE_SL"
EVALUATION_ARTIFACT_TYPE = "evaluation"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_BOOTSTRAP_SAMPLES = 0
EVALUATION_STORAGE_DIRECTORY = "EVALUATIONS"
EVALUATION_RECORD_TYPE = "MODEL_EVALUATION"


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class EvaluationError(RuntimeError):
    """Base exception for model evaluation failures."""


class EvaluationValidationError(EvaluationError, ValueError):
    """Raised when evaluation inputs violate the evaluation contract."""


class EvaluationDependencyError(EvaluationError, ImportError):
    """Raised when a required evaluation dependency is unavailable."""


class EvaluationSchemaError(EvaluationError):
    """Raised when a model, scaler or dataset schema is incompatible."""


@dataclass(frozen=True)
class EvaluationPolicy:
    """Configurable quality gates used to decide evaluation eligibility."""

    minimum_samples: int = 50
    minimum_samples_per_class: int = 10
    minimum_roc_auc: float = 0.55
    minimum_balanced_accuracy: float = 0.50
    maximum_log_loss: float = 0.70
    maximum_brier_score: float = 0.24
    maximum_expected_calibration_error: float = 0.10
    maximum_calibration_gap: float = 0.20
    minimum_calibration_samples: int = 30
    require_both_classes: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "minimum_samples",
            "minimum_samples_per_class",
            "minimum_calibration_samples",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 1:
                raise EvaluationValidationError(f"{field_name} must be a positive integer.")
        bounded_fields = (
            ("minimum_roc_auc", self.minimum_roc_auc, 0.0, 1.0),
            ("minimum_balanced_accuracy", self.minimum_balanced_accuracy, 0.0, 1.0),
            ("maximum_log_loss", self.maximum_log_loss, 0.0, None),
            ("maximum_brier_score", self.maximum_brier_score, 0.0, 1.0),
            (
                "maximum_expected_calibration_error",
                self.maximum_expected_calibration_error,
                0.0,
                1.0,
            ),
            ("maximum_calibration_gap", self.maximum_calibration_gap, 0.0, 1.0),
        )
        for field_name, raw_value, minimum, maximum in bounded_fields:
            value = _finite_float(raw_value, field_name)
            if value < minimum or (maximum is not None and value > maximum):
                raise EvaluationValidationError(f"{field_name} is outside its permitted range.")
        if not isinstance(self.require_both_classes, bool):
            raise EvaluationValidationError("require_both_classes must be a boolean.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generated_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:10]}"

# ============================================================================
# RUNTIME EVALUATION STORAGE
# ============================================================================

def _safe_evaluation_id(value: Any) -> str:
    """Validate an evaluation identifier before using it in a path."""

    selected = str(value).strip()

    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", selected)
        or selected in {".", ".."}
    ):
        raise EvaluationValidationError(
            "Invalid evaluation identifier."
        )

    return selected


def _evaluation_directory(identity: Any) -> Path:
    """
    Resolve the exact customer/market RUNTIME evaluation directory.
    """

    if memory_engine is None:
        raise EvaluationDependencyError(
            "engine.memory is unavailable."
        )

    runtime_resolver = getattr(
        memory_engine,
        "get_runtime_dir",
        None,
    )

    if not callable(runtime_resolver):
        raise EvaluationDependencyError(
            "The central RUNTIME path resolver is unavailable."
        )

    runtime_root = Path(
        runtime_resolver(identity)
    ).expanduser().resolve()

    return runtime_root / EVALUATION_STORAGE_DIRECTORY


def _evaluation_record_path(
    identity: Any,
    evaluation_id: str,
) -> Path:

    safe_id = _safe_evaluation_id(evaluation_id)

    return (
        _evaluation_directory(identity)
        / f"{safe_id}.json"
    )


def _evaluation_json_safe(value: Any) -> Any:
    """
    Convert evaluation data into deterministic JSON-compatible values.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return _evaluation_json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _evaluation_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _evaluation_json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationValidationError(
                "Evaluation report contains a non-finite number."
            )

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise EvaluationValidationError(
        f"Unsupported evaluation report value: {type(value).__name__}"
    )


def _canonical_evaluation_json(value: Any) -> bytes:
    return json.dumps(
        _evaluation_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _persist_evaluation_report(
    report: Mapping[str, Any],
    *,
    identity: Any,
) -> Dict[str, Any]:
    """
    Persist one immutable evaluation report inside RUNTIME.

    The report is stored with its TradingBot identity and SHA-256 digest.
    """

    if not isinstance(report, Mapping):
        raise EvaluationValidationError(
            "Evaluation report must be a mapping."
        )

    evaluation_id = _safe_evaluation_id(
        report.get("evaluation_id")
    )

    identity_data = _identity_dict(identity)

    report_payload = _evaluation_json_safe(
        dict(report)
    )

    report_bytes = _canonical_evaluation_json(
        report_payload
    )

    report_hash = hashlib.sha256(
        report_bytes
    ).hexdigest()

    record = {
        "schema_version": MODULE_SCHEMA_VERSION,
        "record_type": EVALUATION_RECORD_TYPE,
        "evaluation_id": evaluation_id,
        "identity": identity_data,
        "created_at": _utc_now(),
        "model_version_id": str(
            report.get("model_version_id") or ""
        ),
        "payload_sha256": report_hash,
        "report": report_payload,
    }

    encoded_record = _canonical_evaluation_json(
        record
    )

    path = _evaluation_record_path(
        identity,
        evaluation_id,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise EvaluationValidationError(
            f"Evaluation record already exists: {evaluation_id}"
        )

    temporary_path: Optional[Path] = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{evaluation_id}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(encoded_record)
            temporary_file.flush()

            os.fsync(temporary_file.fileno())

        if path.exists():
            raise EvaluationValidationError(
                f"Evaluation record already exists: {evaluation_id}"
            )

        os.replace(
            temporary_path,
            path,
        )

        temporary_path = None

    finally:

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "evaluation_id": evaluation_id,
        "storage": "RUNTIME",
        "path": str(path),
        "sha256": report_hash,
        "verified": True,
    }


def _load_evaluation_report(
    evaluation_id: str,
    *,
    identity: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and verify one persisted RUNTIME evaluation report.
    """

    safe_id = _safe_evaluation_id(evaluation_id)

    path = _evaluation_record_path(
        identity,
        safe_id,
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation report {safe_id!r} does not exist in RUNTIME."
        )

    try:

        record = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError) as error:

        raise EvaluationSchemaError(
            "Persisted evaluation report is unreadable."
        ) from error

    if not isinstance(record, Mapping):
        raise EvaluationSchemaError(
            "Persisted evaluation record is malformed."
        )

    if record.get("schema_version") != MODULE_SCHEMA_VERSION:
        raise EvaluationSchemaError(
            "Unsupported evaluation record schema."
        )

    if record.get("record_type") != EVALUATION_RECORD_TYPE:
        raise EvaluationSchemaError(
            "Invalid evaluation record type."
        )

    if str(record.get("evaluation_id")) != safe_id:
        raise EvaluationSchemaError(
            "Evaluation identifier does not match its filename."
        )

    expected_identity = _identity_dict(identity)

    if record.get("identity") != expected_identity:
        raise PermissionError(
            "Evaluation report belongs to a different TradingBot."
        )

    report = record.get("report")

    if not isinstance(report, Mapping):
        raise EvaluationSchemaError(
            "Persisted evaluation payload is malformed."
        )

    actual_hash = hashlib.sha256(
        _canonical_evaluation_json(report)
    ).hexdigest()

    expected_hash = str(
        record.get("payload_sha256") or ""
    )

    if not expected_hash or actual_hash != expected_hash:
        raise EvaluationSchemaError(
            "Evaluation report failed its SHA-256 integrity check."
        )

    if str(record.get("model_version_id")) != str(
        report.get("model_version_id")
    ):
        raise EvaluationSchemaError(
            "Evaluation record references an inconsistent model version."
        )

    artifact_info = {
        "evaluation_id": safe_id,
        "storage": "RUNTIME",
        "path": str(path),
        "sha256": actual_hash,
        "verified": True,
    }

    return dict(report), artifact_info


def _sha256_file(path: Any) -> str:
    """Calculate the SHA-256 digest of an existing model file."""

    selected_path = Path(path).expanduser().resolve()

    if not selected_path.is_file():
        raise EvaluationSchemaError(
            "The evaluated model checkpoint is unavailable."
        )

    digest = hashlib.sha256()

    try:

        with selected_path.open("rb") as file_handle:

            for chunk in iter(
                lambda: file_handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

    except OSError as error:

        raise EvaluationSchemaError(
            "The model checkpoint could not be verified."
        ) from error

    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise EvaluationValidationError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvaluationValidationError(f"{name} must be a finite number.") from error
    if not math.isfinite(result):
        raise EvaluationValidationError(f"{name} must be a finite number.")
    return result


def _identity_dict(identity: Any) -> Dict[str, str]:
    if identity is None:
        raise EvaluationValidationError("A complete TradingBot identity is required.")
    if isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        method = getattr(identity, "as_dict", None)
        if not callable(method):
            raise EvaluationValidationError("identity must be a mapping or provide as_dict().")
        raw = dict(method())
    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [key for key in required if raw.get(key) is None or not str(raw[key]).strip()]
    if missing:
        raise EvaluationValidationError("Identity is incomplete: " + ", ".join(missing))
    normalized = {key: str(raw[key]).strip() for key in required}
    normalized["market"] = normalized["market"].upper().replace("/", "_").replace("\\", "_")
    return normalized


def _normalise_split_name(value: Any) -> str:
    selected = str(value).strip().lower()
    if selected not in {"validation", "test"}:
        raise EvaluationValidationError("split must be 'validation' or 'test'.")
    return selected


def _dataset_arrays(dataset: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Dict[str, np.ndarray]]:
    if dataset_engine is None:
        raise EvaluationDependencyError("engine.dataset is unavailable.")
    validation = dataset_engine.validate_training_dataset(dataset)
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        errors = validation.get("errors", []) if isinstance(validation, Mapping) else []
        raise EvaluationValidationError("Dataset validation failed: " + "; ".join(str(item) for item in errors))

    try:
        features = np.asarray(dataset.get("features"), dtype=np.float32)
        targets = np.asarray(dataset.get("targets", dataset.get("labels")), dtype=np.int64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvaluationValidationError("Dataset features and targets must be numeric.") from error
    if features.ndim != 2 or features.shape[0] != targets.size:
        raise EvaluationValidationError("Dataset feature and target dimensions do not align.")
    if not np.all(np.isfinite(features)):
        raise EvaluationValidationError("Dataset feature matrix contains NaN or infinity.")
    if targets.size == 0 or not np.all(np.isin(targets, (0, 1))):
        raise EvaluationValidationError("Dataset targets must contain at least one binary 0/1 row.")

    names = tuple(str(item) for item in dataset.get("feature_names", ()))
    if len(names) != features.shape[1] or any(not item for item in names):
        raise EvaluationSchemaError("feature_names do not match the feature matrix width.")
    if len(set(names)) != len(names):
        raise EvaluationSchemaError("feature_names contain duplicate entries.")

    raw_splits = dataset.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise EvaluationValidationError("Dataset has no chronological split metadata.")
    splits: Dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        raw_indices = raw_splits.get(f"{name}_indices", [])
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raise EvaluationValidationError(f"splits.{name}_indices must be a sequence.")
        try:
            indices = np.asarray([int(item) for item in raw_indices], dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise EvaluationValidationError(f"splits.{name}_indices contains invalid values.") from error
        if indices.size and (np.any(indices < 0) or np.any(indices >= targets.size)):
            raise EvaluationValidationError(f"splits.{name}_indices contains out-of-range indices.")
        splits[name] = indices

    combined = np.concatenate([splits[name] for name in ("train", "validation", "test")])
    if combined.size != np.unique(combined).size:
        raise EvaluationValidationError("Chronological dataset splits overlap.")

    sample_metadata = dataset.get("sample_metadata", [])
    label_metadata = dataset.get("label_metadata", [])
    if len(sample_metadata) != targets.size or len(label_metadata) != targets.size:
        raise EvaluationValidationError("Sample and label metadata must align with targets.")

    def _timestamp(record: Mapping[str, Any], name: str, row: int) -> int:
        try:
            value = int(record[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise EvaluationValidationError(f"Metadata row {row} is missing a valid {name}.") from error
        if value < 0:
            raise EvaluationValidationError(f"Metadata row {row} has a negative {name}.")
        return value

    decisions = np.asarray(
        [_timestamp(sample_metadata[index], "decision_time", index) for index in range(targets.size)],
        dtype=np.int64,
    )
    known_times = np.asarray(
        [_timestamp(label_metadata[index], "outcome_known_at", index) for index in range(targets.size)],
        dtype=np.int64,
    )
    exits = np.asarray(
        [_timestamp(label_metadata[index], "exit_time", index) for index in range(targets.size)],
        dtype=np.int64,
    )
    entries = np.asarray(
        [_timestamp(label_metadata[index], "entry_time", index) for index in range(targets.size)],
        dtype=np.int64,
    )

    if np.any(decisions > entries) or np.any(exits < entries) or np.any(known_times < exits):
        raise EvaluationValidationError("Dataset contains a label that violates trade chronology.")

    train_indices = splits["train"]
    validation_indices = splits["validation"]
    test_indices = splits["test"]
    if train_indices.size == 0:
        raise EvaluationValidationError("Dataset training split is empty.")
    if train_indices.size:
        if validation_indices.size:
            validation_start = int(np.min(decisions[validation_indices]))
            if int(np.max(decisions[train_indices])) >= validation_start:
                raise EvaluationValidationError("Train decisions are not before validation decisions.")
            if np.any(known_times[train_indices] >= validation_start):
                raise EvaluationValidationError("A training label became known after validation started.")
        if validation_indices.size and test_indices.size:
            test_start = int(np.min(decisions[test_indices]))
            if int(np.max(decisions[validation_indices])) >= test_start:
                raise EvaluationValidationError("Validation decisions are not before test decisions.")
            if np.any(known_times[validation_indices] >= test_start):
                raise EvaluationValidationError("A validation label became known after test started.")
        elif test_indices.size and not validation_indices.size:
            test_start = int(np.min(decisions[test_indices]))
            if int(np.max(decisions[train_indices])) >= test_start:
                raise EvaluationValidationError("Train decisions are not before test decisions.")
            if np.any(known_times[train_indices] >= test_start):
                raise EvaluationValidationError("A training label became known after test started.")

    return features, targets, names, splits


def _validate_identity_match(dataset: Mapping[str, Any], identity: Optional[Any]) -> Optional[Dict[str, str]]:
    if identity is None:
        return None
    expected = _identity_dict(identity)
    dataset_identity = dataset.get("identity")
    if dataset_identity is None:
        raise EvaluationValidationError("Dataset does not declare its TradingBot identity.")
    actual = _identity_dict(dataset_identity)
    if actual != expected:
        raise PermissionError("Dataset identity does not match the requested TradingBot.")
    return expected


def _scaler_is_train_only(scaler: Any) -> None:
    if scaling_engine is None:
        raise EvaluationDependencyError("engine.scaling is unavailable.")
    report = scaling_engine.validate_scaler(scaler)
    if not isinstance(report, Mapping) or report.get("valid") is not True:
        errors = report.get("errors", []) if isinstance(report, Mapping) else []
        raise EvaluationValidationError("Scaler validation failed: " + "; ".join(str(item) for item in errors))
    if str(getattr(scaler, "fit_scope", "")).upper() != "TRAIN_SPLIT_ONLY":
        raise EvaluationSchemaError("Evaluation requires a scaler fitted exclusively on the training split.")


def _extract_probabilities(model: Any, scaled_features: np.ndarray) -> np.ndarray:
    if torch is None:
        raise EvaluationDependencyError("PyTorch is required for model evaluation.")
    if not callable(model):
        raise EvaluationValidationError("model must be callable.")
    if scaled_features.ndim != 2:
        raise EvaluationValidationError("Scaled evaluation features must be a matrix.")
    if scaled_features.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    try:
        parameters = list(model.parameters())
        device = parameters[0].device if parameters else torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "eval", None)):
        model.eval()
    try:
        with torch.no_grad():
            tensor = torch.as_tensor(scaled_features, dtype=torch.float32, device=device)
            outputs = model(tensor)
            if isinstance(outputs, Mapping):
                probability_tensor = outputs.get("tp_before_sl_probability")
                if probability_tensor is None and outputs.get("tp_before_sl_logit") is not None:
                    probability_tensor = torch.sigmoid(outputs["tp_before_sl_logit"])
            elif isinstance(outputs, torch.Tensor):
                probability_tensor = torch.sigmoid(outputs) if outputs.ndim == 2 and outputs.shape[1] == 1 else outputs
            else:
                probability_tensor = None
            if not isinstance(probability_tensor, torch.Tensor):
                raise EvaluationSchemaError("Model output does not expose TP-before-SL probabilities.")
            probabilities = probability_tensor.detach().to("cpu").numpy().reshape(-1).astype(np.float64)
    finally:
        if was_training and callable(getattr(model, "train", None)):
            model.train()

    if probabilities.size != scaled_features.shape[0]:
        raise EvaluationSchemaError("Model probability count does not match the evaluation sample count.")
    if not np.all(np.isfinite(probabilities)):
        raise EvaluationValidationError("Model returned non-finite probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise EvaluationValidationError("Model probabilities must be within [0, 1].")
    return probabilities


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def _roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(targets.size, dtype=np.float64)
    position = 0
    while position < targets.size:
        end = position + 1
        while end < targets.size and sorted_probabilities[end] == sorted_probabilities[position]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    positive_rank_sum = float(np.sum(ranks[targets == 1]))
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(targets: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    positives = int(np.sum(targets == 1))
    if positives == 0:
        return None
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_targets = targets[order]
    cumulative_true_positives = 0
    cumulative_false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    position = 0
    while position < targets.size:
        end = position + 1
        while end < targets.size and sorted_probabilities[end] == sorted_probabilities[position]:
            end += 1
        group_targets = sorted_targets[position:end]
        cumulative_true_positives += int(np.sum(group_targets == 1))
        cumulative_false_positives += int(np.sum(group_targets == 0))
        recall = cumulative_true_positives / positives
        precision = cumulative_true_positives / max(1, cumulative_true_positives + cumulative_false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        position = end
    return float(average_precision)


def _wilson_interval(successes: int, total: int, z_score: float = 1.959963984540054) -> Optional[Dict[str, float]]:
    if total <= 0:
        return None
    proportion = successes / total
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z_score
        * math.sqrt((proportion * (1.0 - proportion) / total) + z_squared / (4.0 * total * total))
        / denominator
    )
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin), "confidence_level": 0.95}


def _calibration_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> Dict[str, Any]:
    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    bins: List[Dict[str, Any]] = []
    total = int(targets.size)
    ece = 0.0
    maximum_gap = 0.0
    occupied = 0
    for index in range(int(bin_count)):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == int(bin_count) - 1:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        count = int(np.sum(selected))
        if count:
            occupied += 1
            mean_probability = float(np.mean(probabilities[selected]))
            observed_rate = float(np.mean(targets[selected]))
            gap = abs(mean_probability - observed_rate)
            ece += (count / max(1, total)) * gap
            maximum_gap = max(maximum_gap, gap)
        else:
            mean_probability = None
            observed_rate = None
            gap = None
        bins.append(
            {
                "bin_index": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "sample_count": count,
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_gap": gap,
            }
        )
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(maximum_gap),
        "occupied_bin_count": occupied,
        "bin_count": int(bin_count),
        "bins": bins,
        "assessment": "ASSESSED_NOT_CALIBRATED",
    }


def _calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration_bins: int,
) -> Dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise EvaluationValidationError("Target and probability lengths do not match.")
    if y.size == 0:
        raise EvaluationValidationError("The selected evaluation split is empty.")
    if not np.all(np.isin(y, (0, 1))):
        raise EvaluationValidationError("Evaluation targets must contain only 0 and 1.")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise EvaluationValidationError("Probabilities must be finite and within [0, 1].")

    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    predicted = (p >= threshold).astype(np.int64)
    true_positive = int(np.sum((y == 1) & (predicted == 1)))
    true_negative = int(np.sum((y == 0) & (predicted == 0)))
    false_positive = int(np.sum((y == 0) & (predicted == 1)))
    false_negative = int(np.sum((y == 1) & (predicted == 0)))
    count = int(y.size)
    positives = int(np.sum(y == 1))
    negatives = count - positives

    accuracy = (true_positive + true_negative) / count
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positives if positives else None
    specificity = true_negative / negatives if negatives else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0.0
        else (0.0 if recall is not None else None)
    )
    recalls = []
    if positives:
        recalls.append(float(recall))
    if negatives:
        recalls.append(float(specificity))
    balanced_accuracy = float(np.mean(recalls)) if len(recalls) == 2 else None

    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    matthews_correlation = (
        (true_positive * true_negative - false_positive * false_negative) / denominator
        if denominator > 0.0
        else None
    )
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)))
    brier_score = float(np.mean((p - y) ** 2))
    calibration = _calibration_metrics(y, p, calibration_bins)

    return {
        "sample_count": count,
        "class_counts": {"SL_FIRST": negatives, "TP_FIRST": positives},
        "positive_rate": float(positives / count),
        "mean_predicted_probability": float(np.mean(p)),
        "classification_threshold": float(threshold),
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "accuracy": float(accuracy),
        "accuracy_confidence_interval_95": _wilson_interval(true_positive + true_negative, count),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision),
        "recall": None if recall is None else float(recall),
        "specificity": None if specificity is None else float(specificity),
        "f1_score": None if f1 is None else float(f1),
        "matthews_correlation_coefficient": None if matthews_correlation is None else float(matthews_correlation),
        "roc_auc": _roc_auc(y, p),
        "average_precision": _average_precision(y, p),
        "log_loss": log_loss,
        "brier_score": brier_score,
        "calibration": calibration,
    }


def _approval_gate(
    metrics: Mapping[str, Any],
    policy: EvaluationPolicy,
    *,
    split: str,
    versioned_model: bool = False,
) -> Dict[str, Any]:
    reasons: List[str] = []
    sample_count = int(metrics.get("sample_count", 0))
    class_counts = metrics.get("class_counts", {})
    negative_count = int(class_counts.get("SL_FIRST", 0))
    positive_count = int(class_counts.get("TP_FIRST", 0))
    calibration = metrics.get("calibration", {})

    if split != "test":
        reasons.append("APPROVAL_REQUIRES_HELD_OUT_TEST_SPLIT")
    if not versioned_model:
        reasons.append("VERSIONED_MODEL_EVALUATION_REQUIRED")
    if sample_count < policy.minimum_samples:
        reasons.append("INSUFFICIENT_TEST_SAMPLE_COUNT")
    if policy.require_both_classes and (negative_count == 0 or positive_count == 0):
        reasons.append("BOTH_OUTCOME_CLASSES_REQUIRED")
    if negative_count < policy.minimum_samples_per_class or positive_count < policy.minimum_samples_per_class:
        reasons.append("INSUFFICIENT_SAMPLES_PER_CLASS")

    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or float(roc_auc) < policy.minimum_roc_auc:
        reasons.append("ROC_AUC_BELOW_MINIMUM")
    balanced_accuracy = metrics.get("balanced_accuracy")
    if balanced_accuracy is None or float(balanced_accuracy) < policy.minimum_balanced_accuracy:
        reasons.append("BALANCED_ACCURACY_BELOW_MINIMUM")

    log_loss = metrics.get("log_loss")
    if log_loss is None or float(log_loss) > policy.maximum_log_loss:
        reasons.append("LOG_LOSS_ABOVE_MAXIMUM")
    brier_score = metrics.get("brier_score")
    if brier_score is None or float(brier_score) > policy.maximum_brier_score:
        reasons.append("BRIER_SCORE_ABOVE_MAXIMUM")

    if sample_count < policy.minimum_calibration_samples:
        reasons.append("INSUFFICIENT_CALIBRATION_SAMPLE_COUNT")
    ece = calibration.get("expected_calibration_error") if isinstance(calibration, Mapping) else None
    mce = calibration.get("maximum_calibration_error") if isinstance(calibration, Mapping) else None
    if ece is None or float(ece) > policy.maximum_expected_calibration_error:
        reasons.append("EXPECTED_CALIBRATION_ERROR_ABOVE_MAXIMUM")
    if mce is None or float(mce) > policy.maximum_calibration_gap:
        reasons.append("MAXIMUM_CALIBRATION_GAP_ABOVE_MAXIMUM")

    return {
        "passed": not reasons,
        "approval_eligible": split == "test" and not reasons,
        "blocking_reasons": reasons,
        "policy": policy.to_dict(),
        "evaluated_split": split,
        "test_split_required": True,
        "versioned_model_required": True,
        "calibration_was_fitted": False,
    }


# ============================================================================
# DATASET AND MODEL EVALUATION
# ============================================================================

def evaluate_predictions(
    targets: Any,
    probabilities: Any,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    policy: Optional[EvaluationPolicy] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Calculate classification, probability and calibration metrics without fitting."""
    threshold_value = _finite_float(threshold, "threshold")
    if not 0.0 < threshold_value < 1.0:
        raise EvaluationValidationError("threshold must be strictly between 0 and 1.")
    if isinstance(calibration_bins, bool) or int(calibration_bins) < 2 or int(calibration_bins) > 100:
        raise EvaluationValidationError("calibration_bins must be an integer between 2 and 100.")
    try:
        target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
        probability_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvaluationValidationError("targets and probabilities must be numeric sequences.") from error
    selected_policy = policy or EvaluationPolicy()
    selected_split = _normalise_split_name(split)
    metrics = _calculate_metrics(
        target_array,
        probability_array,
        threshold=threshold_value,
        calibration_bins=int(calibration_bins),
    )
    gate = _approval_gate(metrics, selected_policy, split=selected_split)
    return {
        "schema_version": MODULE_SCHEMA_VERSION,
        "target": DATASET_TARGET_NAME,
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "split": selected_split,
        "evaluated_at": _utc_now(),
        "metrics": metrics,
        "approval_gate": gate,
        "calibration_status": "ASSESSED_NOT_CALIBRATED",
    }


def evaluate_model(
    model: Any,
    scaler: Any,
    dataset: Mapping[str, Any],
    *,
    split: str = "test",
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    policy: Optional[EvaluationPolicy] = None,
    identity: Optional[Any] = None,
    model_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model on a chronological validation or held-out test split."""
    if not isinstance(dataset, Mapping):
        raise EvaluationValidationError("dataset must be a mapping.")
    selected_split = _normalise_split_name(split)
    expected_schema = str(dataset.get("model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if dataset.get("target") != DATASET_TARGET_NAME or expected_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise EvaluationSchemaError("Dataset does not use the TP-before-SL binary target schema.")
    model_schema = str(getattr(model, "model_target_schema", MODEL_TARGET_SCHEMA)).strip().upper()
    if model_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise EvaluationSchemaError("Model does not use the TP-before-SL binary target schema.")

    resolved_identity = _validate_identity_match(dataset, identity)
    features, targets, feature_names, splits = _dataset_arrays(dataset)
    selected_indices = splits[selected_split]
    if selected_indices.size == 0:
        raise EvaluationValidationError(f"The dataset has no {selected_split} rows to evaluate.")

    _scaler_is_train_only(scaler)
    if int(getattr(model, "input_size", features.shape[1])) != features.shape[1]:
        raise EvaluationSchemaError("Model input width does not match the dataset feature width.")
    scaler_names = tuple(getattr(scaler, "feature_names", ()) or ())
    if scaler_names and scaler_names != feature_names:
        raise EvaluationSchemaError("Dataset feature order does not match the trained scaler.")
    scaler_hash = getattr(scaler, "feature_schema_hash", None)
    feature_schema = dataset.get("feature_schema", {})
    dataset_hash = feature_schema.get("sha256") if isinstance(feature_schema, Mapping) else None
    dataset_hash = dataset_hash or dataset.get("feature_schema_hash")
    if scaler_hash and dataset_hash and str(scaler_hash) != str(dataset_hash):
        raise EvaluationSchemaError("Dataset feature schema hash does not match the trained scaler.")

    scaled = scaling_engine.apply_scaler(
        features[selected_indices],
        scaler,
        feature_names=feature_names,
    )
    probabilities = _extract_probabilities(model, scaled)
    selected_targets = targets[selected_indices]
    selected_policy = policy or EvaluationPolicy()
    report = evaluate_predictions(
        selected_targets,
        probabilities,
        threshold=threshold,
        calibration_bins=calibration_bins,
        policy=selected_policy,
        split=selected_split,
    )

    sample_metadata = dataset.get("sample_metadata", [])
    label_metadata = dataset.get("label_metadata", [])
    selected_decisions = [int(sample_metadata[int(index)]["decision_time"]) for index in selected_indices]
    selected_known_times = [int(label_metadata[int(index)]["outcome_known_at"]) for index in selected_indices]
    report.update(
        {
            "model_version_id": str(model_version_id) if model_version_id else None,
            "dataset_id": str(dataset.get("dataset_id", "")),
            "identity": resolved_identity,
            "feature_count": int(features.shape[1]),
            "feature_schema_hash": str(dataset_hash) if dataset_hash else None,
            "scaler_fit_scope": str(getattr(scaler, "fit_scope", "")),
            "scaler_fit_sample_count": int(getattr(scaler, "fit_sample_count", 0)),
            "evaluated_decision_time_range": {
                "start": min(selected_decisions),
                "end": max(selected_decisions),
            },
            "label_outcome_known_time_range": {
                "start": min(selected_known_times),
                "end": max(selected_known_times),
            },
            "sample_indices_sha256": hashlib.sha256(
                json.dumps(selected_indices.astype(int).tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    )
    return report


# ============================================================================
# VERSIONED EVALUATION AND APPROVAL
# ============================================================================

def _load_identity(market: str, identity: Any) -> Any:
    if memory_engine is None:
        raise EvaluationDependencyError("engine.memory is unavailable.")
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if callable(resolver):
        return resolver(market, identity=identity)
    return identity


def evaluate_model_version(
    market: str,
    model_version_id: str,
    *,
    identity: Any,
    policy: Optional[EvaluationPolicy] = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    split: str = "test",
    approve_if_passed: bool = False,
    evaluation_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a customer-isolated model and persist the evaluation
    report directly inside its RUNTIME directory.
    """

    if memory_engine is None:
        raise EvaluationDependencyError(
            "engine.memory is required for model evaluation."
        )

    selected_model_id = str(model_version_id).strip()

    if not selected_model_id:
        raise EvaluationValidationError(
            "model_version_id cannot be empty."
        )

    selected_split = _normalise_split_name(split)

    if approve_if_passed and selected_split != "test":
        raise EvaluationValidationError(
            "A model can only be approved from the held-out test split."
        )

    resolved_identity = _load_identity(
        str(market),
        identity,
    )

    expected_identity = _identity_dict(
        resolved_identity
    )

    # ========================================================================
    # LOAD AND VERIFY THE MODEL CHECKPOINT
    # ========================================================================

    model, scaler, checkpoint = memory_engine.load_model(
        str(market),
        identity=resolved_identity,
        version_id=selected_model_id,
        require_approved=False,
    )

    if not isinstance(checkpoint, Mapping):
        raise EvaluationSchemaError(
            "Loaded model checkpoint is malformed."
        )

    checkpoint_identity = checkpoint.get("identity")

    if checkpoint_identity != expected_identity:
        raise PermissionError(
            "Model checkpoint belongs to a different TradingBot."
        )

    if str(checkpoint.get("market", "")).upper() != expected_identity["market"]:
        raise EvaluationSchemaError(
            "Model checkpoint market does not match its identity."
        )

    checkpoint_model_id = str(
        checkpoint.get("model_version_id") or ""
    )

    if checkpoint_model_id != selected_model_id:
        raise EvaluationSchemaError(
            "Loaded checkpoint does not match the requested model version."
        )

    loaded_path = checkpoint.get("loaded_from")

    if not loaded_path:
        raise EvaluationSchemaError(
            "Loaded checkpoint has no verified storage path."
        )

    model_artifact_hash = _sha256_file(
        loaded_path
    )

    # ========================================================================
    # VERIFY TRAINING METADATA
    # ========================================================================

    training_summary = checkpoint.get(
        "training_summary",
        {},
    )

    if not isinstance(training_summary, Mapping):
        raise EvaluationSchemaError(
            "Model training summary is malformed."
        )

    checkpoint_schema = (
        training_summary.get("model_target_schema")
        or checkpoint.get("model_target_schema")
    )

    if str(checkpoint_schema or "").upper() not in {
        MODEL_TARGET_SCHEMA,
        "TP_BEFORE_SL",
    }:
        raise EvaluationSchemaError(
            "Requested model is not a TP-before-SL binary model."
        )

    dataset_version_id = checkpoint.get(
        "dataset_version_id"
    )

    if not dataset_version_id:
        raise EvaluationSchemaError(
            "Model checkpoint does not reference its immutable training dataset."
        )

    validation_available = (
        training_summary.get("validation_available") is True
    )

    # ========================================================================
    # LOAD THE TRAINING DATASET
    # ========================================================================

    dataset = memory_engine.load_training_dataset(
        str(dataset_version_id),
        identity=resolved_identity,
        require_approved=False,
    )

    if not isinstance(dataset, Mapping):
        raise EvaluationSchemaError(
            "Stored training dataset is malformed."
        )

    dataset_identity = dataset.get("identity")

    if _identity_dict(dataset_identity) != expected_identity:
        raise PermissionError(
            "Training dataset belongs to a different TradingBot."
        )

    # ========================================================================
    # RUN INDEPENDENT EVALUATION
    # ========================================================================

    report = evaluate_model(
        model,
        scaler,
        dataset,
        split=selected_split,
        threshold=threshold,
        calibration_bins=calibration_bins,
        policy=policy,
        identity=resolved_identity,
        model_version_id=selected_model_id,
    )

    selected_policy = policy or EvaluationPolicy()

    versioned_gate = _approval_gate(
        report["metrics"],
        selected_policy,
        split=selected_split,
        versioned_model=True,
    )

    if not validation_available:
        versioned_gate["blocking_reasons"].append(
            "MODEL_TRAINING_DID_NOT_USE_VALIDATION_SPLIT"
        )

        versioned_gate["passed"] = False
        versioned_gate["approval_eligible"] = False

    report["approval_gate"] = versioned_gate

    version_record = checkpoint.get(
        "version_record"
    )

    if isinstance(version_record, Mapping):
        model_status = str(
            version_record.get("status", "DRAFT")
        )
    else:
        model_status = str(
            checkpoint.get("model_version_status", "DRAFT")
        )

    # ========================================================================
    # COMPLETE EVALUATION METADATA
    # ========================================================================

    selected_evaluation_id = _safe_evaluation_id(
        evaluation_version_id or _generated_id("evaluation")
    )

    report.update(
        {
            "evaluation_id": selected_evaluation_id,

            "created_at": _utc_now(),

            "market": expected_identity["market"],

            "identity": expected_identity,

            "model_version_id": selected_model_id,

            "model_version_status_before_evaluation": model_status,

            "model_artifact_sha256": model_artifact_hash,

            "dataset_version_id": str(dataset_version_id),

            "scaler_version_id": checkpoint.get(
                "scaler_version_id"
            ),

            "training_validation_available": validation_available,

            "evaluation_storage": "RUNTIME",
        }
    )

    # ========================================================================
    # SAVE EVALUATION REPORT INTO RUNTIME
    # ========================================================================

    evaluation_artifact = _persist_evaluation_report(
        report,
        identity=resolved_identity,
    )

    report["evaluation_artifact"] = evaluation_artifact

    # ========================================================================
    # APPROVAL GATE
    # ========================================================================

    approval_result = None

    if approve_if_passed:

        gate = report["approval_gate"]

        if (
            gate.get("approval_eligible") is True
            and gate.get("passed") is True
        ):

            approval_result = memory_engine.approve_model_version(
                selected_model_id,
                identity=resolved_identity,
                validation_metadata={
                    "passed": True,
                    "evaluation_passed": True,

                    "evaluation_version_id": selected_evaluation_id,

                    "evaluation_artifact_sha256": (
                        evaluation_artifact["sha256"]
                    ),

                    "model_artifact_sha256": model_artifact_hash,

                    "evaluated_split": selected_split,

                    "model_target_schema": MODEL_TARGET_SCHEMA,

                    "metrics": report["metrics"],

                    "policy": gate["policy"],
                },
                reason=(
                    "Model passed the independent held-out test evaluation gate."
                ),
            )

        else:

            report["approval_blocked"] = True

            report["approval_blocking_reasons"] = list(
                gate.get("blocking_reasons", [])
            )

    # ========================================================================
    # FINAL EVALUATION RESULT
    # ========================================================================

    if approval_result is not None:

        report["model_approval"] = (
            approval_result.to_dict()
            if callable(getattr(approval_result, "to_dict", None))
            else approval_result
        )

    else:

        report["model_approval"] = {
            "approved": False,
            "status": model_status,
            "reason": (
                "Approval was not requested or the evaluation gate did not pass."
            ),
        }

    return report

def approve_evaluated_model(
    model_version_id: str,
    evaluation_report: Mapping[str, Any],
    *,
    identity: Any,
    reason: str = "Model passed the independent held-out test evaluation gate.",
) -> Any:
    """
    Approve a model only after verifying its persisted RUNTIME evaluation,
    exact TradingBot identity and unchanged model checkpoint.
    """

    if memory_engine is None:
        raise EvaluationDependencyError(
            "engine.memory is unavailable."
        )

    if not isinstance(evaluation_report, Mapping):
        raise EvaluationValidationError(
            "evaluation_report must be a mapping."
        )

    selected_model_id = str(
        model_version_id
    ).strip()

    if not selected_model_id:
        raise EvaluationValidationError(
            "model_version_id cannot be empty."
        )

    resolved_identity = _load_identity(
        str(_identity_dict(identity)["market"]),
        identity,
    )

    expected_identity = _identity_dict(
        resolved_identity
    )

    evaluation_id = _safe_evaluation_id(
        evaluation_report.get("evaluation_id")
    )

    # ========================================================================
    # LOAD AND VERIFY THE PERSISTED EVALUATION
    # ========================================================================

    stored_report, artifact_info = _load_evaluation_report(
        evaluation_id,
        identity=resolved_identity,
    )

    if str(stored_report.get("model_version_id", "")) != selected_model_id:
        raise EvaluationSchemaError(
            "Persisted evaluation references a different model version."
        )

    if str(stored_report.get("model_target_schema", "")).upper() not in {
        MODEL_TARGET_SCHEMA,
        "TP_BEFORE_SL",
    }:
        raise EvaluationSchemaError(
            "Persisted evaluation target schema is incompatible."
        )

    if stored_report.get("identity") != expected_identity:
        raise PermissionError(
            "Persisted evaluation belongs to a different TradingBot."
        )

    # ========================================================================
    # PREVENT APPROVAL USING A MODIFIED IN-MEMORY REPORT
    # ========================================================================

    supplied_report = dict(evaluation_report)

    # These fields are added after the immutable evaluation report is saved.
    supplied_report.pop("evaluation_artifact", None)
    supplied_report.pop("model_approval", None)
    supplied_report.pop("approval_blocked", None)
    supplied_report.pop("approval_blocking_reasons", None)

    if (
        _canonical_evaluation_json(supplied_report)
        != _canonical_evaluation_json(stored_report)
    ):
        raise EvaluationSchemaError(
            "The supplied evaluation report differs from its persisted RUNTIME record."
        )

    # ========================================================================
    # VERIFY THE HELD-OUT TEST GATE
    # ========================================================================

    gate = stored_report.get(
        "approval_gate"
    )

    if not isinstance(gate, Mapping):
        raise EvaluationValidationError(
            "Persisted evaluation has no valid approval gate."
        )

    if (
        gate.get("evaluated_split") != "test"
        or gate.get("passed") is not True
        or gate.get("approval_eligible") is not True
    ):
        raise EvaluationValidationError(
            "Model approval is blocked because the held-out test gate did not pass."
        )

    if str(stored_report.get("split")) != "test":
        raise EvaluationValidationError(
            "Model approval requires a held-out test evaluation."
        )

    # ========================================================================
    # VERIFY THE CURRENT MODEL CHECKPOINT
    # ========================================================================

    _model, _scaler, checkpoint = memory_engine.load_model(
        expected_identity["market"],
        identity=resolved_identity,
        version_id=selected_model_id,
        require_approved=False,
    )

    if checkpoint.get("identity") != expected_identity:
        raise PermissionError(
            "Model checkpoint belongs to a different TradingBot."
        )

    if str(checkpoint.get("model_version_id", "")) != selected_model_id:
        raise EvaluationSchemaError(
            "Loaded checkpoint does not match the evaluation model version."
        )

    current_model_hash = _sha256_file(
        checkpoint.get("loaded_from")
    )

    expected_model_hash = str(
        stored_report.get("model_artifact_sha256") or ""
    )

    if (
        not expected_model_hash
        or current_model_hash != expected_model_hash
    ):
        raise EvaluationSchemaError(
            "Model checkpoint changed after its evaluation."
        )

    # ========================================================================
    # APPROVE MODEL THROUGH THE CURRENT MEMORY API
    # ========================================================================

    return memory_engine.approve_model_version(
        selected_model_id,
        identity=resolved_identity,
        validation_metadata={
            "passed": True,
            "evaluation_passed": True,

            "evaluation_version_id": evaluation_id,

            "evaluation_artifact_sha256": artifact_info["sha256"],

            "model_artifact_sha256": current_model_hash,

            "evaluated_split": "test",

            "model_target_schema": MODEL_TARGET_SCHEMA,

            "metrics": stored_report.get("metrics"),

            "policy": gate.get("policy"),

        },
        reason=reason,
    )

    
# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "MODULE_SCHEMA_VERSION",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET_NAME",
    "EVALUATION_ARTIFACT_TYPE",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "DEFAULT_CALIBRATION_BINS",
    "EvaluationError",
    "EvaluationValidationError",
    "EvaluationDependencyError",
    "EvaluationSchemaError",
    "EvaluationPolicy",
    "evaluate_predictions",
    "evaluate_model",
    "evaluate_model_version",
    "approve_evaluated_model",
]
