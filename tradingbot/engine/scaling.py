"""Train-only feature scaling for the Spectline numerical dataset pipeline."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

SCALING_SCHEMA_VERSION = 1
DEFAULT_SCALING_METHOD = "standard"
SUPPORTED_SCALING_METHODS = ("standard", "robust")
DEFAULT_CLIP_VALUE = 10.0
MIN_SCALE = 1e-8


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class ScalingError(RuntimeError):
    """Base exception for feature scaling errors."""


class ScalingValidationError(ScalingError, ValueError):
    """Raised when a feature matrix or scaler violates the scaling contract."""


@dataclass
class FeatureScaler:
    """Persistable feature scaling statistics fitted on training rows only."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: Tuple[str, ...] = ()
    passthrough_mask: Optional[np.ndarray] = None
    constant_mask: Optional[np.ndarray] = None
    method: str = DEFAULT_SCALING_METHOD
    clip_value: Optional[float] = DEFAULT_CLIP_VALUE
    feature_schema_hash: Optional[str] = None
    fit_dataset_id: Optional[str] = None
    fit_sample_count: int = 0
    fit_indices_hash: Optional[str] = None
    fit_scope: str = "TRAIN_SPLIT_ONLY"
    schema_version: int = SCALING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if self.mean.size == 0 or self.std.size == 0:
            raise ScalingValidationError("Scaler statistics cannot be empty.")
        if self.mean.shape != self.std.shape:
            raise ScalingValidationError("Scaler mean and scale arrays must have equal shapes.")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ScalingValidationError("Scaler statistics must be finite.")
        if np.any(self.std <= 0.0):
            raise ScalingValidationError("Every scaler scale value must be positive.")

        self.feature_names = tuple(str(item) for item in self.feature_names)
        if self.feature_names and len(self.feature_names) != self.mean.size:
            raise ScalingValidationError("Feature name count does not match scaler width.")

        if self.passthrough_mask is None:
            self.passthrough_mask = np.zeros(self.mean.size, dtype=bool)
        else:
            self.passthrough_mask = np.asarray(self.passthrough_mask, dtype=bool).reshape(-1)
        if self.passthrough_mask.size != self.mean.size:
            raise ScalingValidationError("Passthrough mask width does not match scaler width.")

        if self.constant_mask is None:
            self.constant_mask = np.zeros(self.mean.size, dtype=bool)
        else:
            self.constant_mask = np.asarray(self.constant_mask, dtype=bool).reshape(-1)
        if self.constant_mask.size != self.mean.size:
            raise ScalingValidationError("Constant mask width does not match scaler width.")

        self.method = str(self.method).strip().lower()
        if self.method not in SUPPORTED_SCALING_METHODS:
            raise ScalingValidationError("Unsupported scaling method: " + self.method)
        if self.clip_value is not None:
            self.clip_value = float(self.clip_value)
            if not math.isfinite(self.clip_value) or self.clip_value <= 0.0:
                raise ScalingValidationError("clip_value must be a finite positive number or None.")
        self.fit_sample_count = int(self.fit_sample_count)
        if self.fit_sample_count < 0:
            raise ScalingValidationError("fit_sample_count cannot be negative.")
        self.schema_version = int(self.schema_version)

    @property
    def scale(self) -> np.ndarray:
        """Return the scale vector under the conventional scaler attribute name."""
        return self.std

    @property
    def feature_count(self) -> int:
        """Return the number of features represented by this scaler."""
        return int(self.mean.size)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation of the fitted scaler."""
        return {
            "schema_version": self.schema_version,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.feature_names),
            "passthrough_mask": self.passthrough_mask.astype(bool).tolist(),
            "constant_mask": self.constant_mask.astype(bool).tolist(),
            "method": self.method,
            "clip_value": self.clip_value,
            "feature_schema_hash": self.feature_schema_hash,
            "fit_dataset_id": self.fit_dataset_id,
            "fit_sample_count": self.fit_sample_count,
            "fit_indices_hash": self.fit_indices_hash,
            "fit_scope": self.fit_scope,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureScaler":
        """Restore a scaler from its serialized mapping."""
        if not isinstance(value, Mapping):
            raise ScalingValidationError("Serialized scaler must be a mapping.")
        required = ("mean", "std")
        missing = [name for name in required if name not in value]
        if missing:
            raise ScalingValidationError("Serialized scaler is missing: " + ", ".join(missing))
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
            feature_names=tuple(value.get("feature_names", ())),
            passthrough_mask=value.get("passthrough_mask"),
            constant_mask=value.get("constant_mask"),
            method=str(value.get("method", DEFAULT_SCALING_METHOD)),
            clip_value=value.get("clip_value", DEFAULT_CLIP_VALUE),
            feature_schema_hash=value.get("feature_schema_hash"),
            fit_dataset_id=value.get("fit_dataset_id"),
            fit_sample_count=int(value.get("fit_sample_count", 0)),
            fit_indices_hash=value.get("fit_indices_hash"),
            fit_scope=str(value.get("fit_scope", "TRAIN_SPLIT_ONLY")),
            schema_version=int(value.get("schema_version", SCALING_SCHEMA_VERSION)),
        )


# ============================================================================
# FEATURE SCHEMA HELPERS
# ============================================================================

def _finite_matrix(features: Any, *, allow_empty_rows: bool = False) -> np.ndarray:
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalingValidationError("Features must be convertible to a numeric matrix.") from error
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ScalingValidationError("Features must be a one- or two-dimensional array.")
    if matrix.shape[1] == 0:
        raise ScalingValidationError("Feature matrix must contain at least one column.")
    if matrix.shape[0] == 0 and not allow_empty_rows:
        raise ScalingValidationError("Feature matrix must contain at least one row.")
    if not np.all(np.isfinite(matrix)):
        raise ScalingValidationError("Feature matrix contains NaN or infinity.")
    return matrix


def _feature_schema_hash(feature_names: Sequence[str]) -> Optional[str]:
    if not feature_names:
        return None
    payload = "\n".join(str(item) for item in feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_passthrough_feature(name: str) -> bool:
    normalized = str(name).strip().lower()
    passthrough_prefixes = (
        "setup.direction_",
        "setup.setup_timeframe.",
        "setup.entry_timeframe.",
        "setup.resolution_timeframe.",
        "market.pattern_hash_",
        "market.event_hash_",
        "market.timeframe_available.",
    )
    if normalized.startswith(passthrough_prefixes):
        return True
    if normalized.endswith("_available") or normalized.endswith("_time_available"):
        return True
    if normalized.endswith(".available"):
        return True
    return False


def build_passthrough_mask(feature_names: Sequence[str]) -> np.ndarray:
    """Identify one-hot, hashed-category and availability features to preserve."""
    if isinstance(feature_names, (str, bytes)):
        raise ScalingValidationError("feature_names must be a sequence of names.")
    return np.asarray([_is_passthrough_feature(str(name)) for name in feature_names], dtype=bool)


def _normalize_feature_names(
    feature_names: Optional[Sequence[str]],
    feature_count: int,
) -> Tuple[str, ...]:
    if feature_names is None:
        return ()
    if isinstance(feature_names, (str, bytes)):
        raise ScalingValidationError("feature_names must be a sequence of strings.")
    names = tuple(str(item) for item in feature_names)
    if len(names) != feature_count:
        raise ScalingValidationError(
            f"Feature name count {len(names)} does not match matrix width {feature_count}."
        )
    if any(not name for name in names):
        raise ScalingValidationError("Feature names cannot be empty.")
    if len(set(names)) != len(names):
        raise ScalingValidationError("Feature names must be unique.")
    return names


# ============================================================================
# FITTING
# ============================================================================

def _fit_statistics(
    matrix: np.ndarray,
    *,
    method: str,
    passthrough_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_method = str(method).strip().lower()
    if normalized_method not in SUPPORTED_SCALING_METHODS:
        raise ScalingValidationError(
            "Unsupported scaling method. Expected one of: " + ", ".join(SUPPORTED_SCALING_METHODS)
        )

    if normalized_method == "standard":
        center = np.mean(matrix, axis=0, dtype=np.float64)
        raw_scale = np.std(matrix, axis=0, dtype=np.float64, ddof=0)
    else:
        center = np.median(matrix, axis=0)
        q25 = np.percentile(matrix, 25.0, axis=0)
        q75 = np.percentile(matrix, 75.0, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        median_absolute_deviation = np.median(np.abs(matrix - center), axis=0) * 1.4826
        raw_scale = np.where(iqr_scale > MIN_SCALE, iqr_scale, median_absolute_deviation)

    constant_mask = raw_scale <= MIN_SCALE
    scale = np.where(constant_mask, 1.0, raw_scale)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    center[passthrough_mask] = 0.0
    scale[passthrough_mask] = 1.0
    constant_mask[passthrough_mask] = False

    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ScalingValidationError("Fitted scaler statistics contain non-finite values.")
    return center, scale, constant_mask


def fit_scaler(
    features: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    method: str = DEFAULT_SCALING_METHOD,
    passthrough_mask: Optional[Sequence[bool]] = None,
    clip_value: Optional[float] = DEFAULT_CLIP_VALUE,
    dataset_id: Optional[str] = None,
    fit_indices: Optional[Sequence[int]] = None,
) -> FeatureScaler:
    """Fit a scaler from a feature matrix that must contain training rows only.

    For a complete dataset mapping, use ``fit_dataset_scaler`` instead. This
    matrix-level function cannot infer whether the caller supplied validation
    or test rows, so the caller is responsible for passing training rows only.
    """
    if isinstance(features, Mapping):
        if "features" in features and "splits" in features:
            return fit_dataset_scaler(
                features,
                method=method,
                clip_value=clip_value,
            )
        raise ScalingValidationError("Mapping input must be a complete dataset with features and splits.")

    matrix = _finite_matrix(features)
    names = _normalize_feature_names(feature_names, matrix.shape[1])
    if passthrough_mask is None:
        preserve = build_passthrough_mask(names) if names else np.zeros(matrix.shape[1], dtype=bool)
    else:
        preserve = np.asarray(passthrough_mask, dtype=bool).reshape(-1)
        if preserve.size != matrix.shape[1]:
            raise ScalingValidationError("Explicit passthrough mask does not match feature width.")

    center, scale, constant_mask = _fit_statistics(
        matrix,
        method=method,
        passthrough_mask=preserve,
    )
    indices_hash = None
    if fit_indices is not None:
        index_values = [int(item) for item in fit_indices]
        indices_hash = hashlib.sha256(
            json.dumps(index_values, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return FeatureScaler(
        mean=center,
        std=scale,
        feature_names=names,
        passthrough_mask=preserve,
        constant_mask=constant_mask,
        method=method,
        clip_value=clip_value,
        feature_schema_hash=_feature_schema_hash(names),
        fit_dataset_id=dataset_id,
        fit_sample_count=int(matrix.shape[0]),
        fit_indices_hash=indices_hash,
        fit_scope="CALLER_SUPPLIED_ROWS",
    )


def _training_indices(dataset: Mapping[str, Any], sample_count: int) -> np.ndarray:
    splits = dataset.get("splits")
    if not isinstance(splits, Mapping):
        raise ScalingValidationError("Dataset must contain chronological split metadata.")
    raw_train = splits.get("train_indices")
    if not isinstance(raw_train, Sequence) or isinstance(raw_train, (str, bytes)):
        raise ScalingValidationError("Dataset splits must contain train_indices.")

    train: list[int] = []
    for raw_index in raw_train:
        if isinstance(raw_index, bool):
            raise ScalingValidationError("Training split contains a boolean index.")
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError) as error:
            raise ScalingValidationError("Training split contains a non-integer index.") from error
        if index < 0 or index >= sample_count:
            raise ScalingValidationError("Training split contains an out-of-range index.")
        train.append(index)

    if not train:
        raise ScalingValidationError("Training split is empty; refusing to fit a scaler.")
    if len(train) != len(set(train)):
        raise ScalingValidationError("Training split contains duplicate indices.")

    other_indices: list[int] = []
    for split_name in ("validation_indices", "test_indices"):
        values = splits.get(split_name, [])
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ScalingValidationError(f"Dataset splits.{split_name} must be a sequence.")
        for raw_index in values:
            try:
                index = int(raw_index)
            except (TypeError, ValueError, OverflowError) as error:
                raise ScalingValidationError(f"Dataset splits.{split_name} contains an invalid index.") from error
            if index < 0 or index >= sample_count:
                raise ScalingValidationError(f"Dataset splits.{split_name} contains an out-of-range index.")
            other_indices.append(index)

    if set(train).intersection(other_indices):
        raise ScalingValidationError("Training rows overlap validation or test rows.")
    return np.asarray(train, dtype=np.int64)


def fit_dataset_scaler(
    dataset: Mapping[str, Any],
    *,
    method: str = DEFAULT_SCALING_METHOD,
    clip_value: Optional[float] = DEFAULT_CLIP_VALUE,
) -> FeatureScaler:
    """Fit scaling statistics from the dataset's train split and nothing else."""
    if not isinstance(dataset, Mapping):
        raise ScalingValidationError("dataset must be a mapping.")
    if "features" not in dataset:
        raise ScalingValidationError("Dataset is missing its features matrix.")

    matrix = _finite_matrix(dataset["features"])
    indices = _training_indices(dataset, matrix.shape[0])
    train_matrix = matrix[indices]
    names = _normalize_feature_names(dataset.get("feature_names"), matrix.shape[1])

    feature_schema = dataset.get("feature_schema", {})
    schema_hash = None
    if isinstance(feature_schema, Mapping):
        schema_hash = feature_schema.get("sha256")
    schema_hash = schema_hash or dataset.get("feature_schema_hash") or _feature_schema_hash(names)

    scaler = fit_scaler(
        train_matrix,
        feature_names=names or None,
        method=method,
        clip_value=clip_value,
        dataset_id=str(dataset.get("dataset_id")) if dataset.get("dataset_id") is not None else None,
        fit_indices=indices.tolist(),
    )
    scaler.feature_schema_hash = str(schema_hash) if schema_hash else None
    scaler.fit_scope = "TRAIN_SPLIT_ONLY"
    return scaler


# ============================================================================
# APPLYING AND REVERSING SCALING
# ============================================================================

def _coerce_scaler(scaler: Any) -> FeatureScaler:
    if isinstance(scaler, FeatureScaler):
        return scaler
    if isinstance(scaler, Mapping):
        return FeatureScaler.from_dict(scaler)

    mean = getattr(scaler, "mean", None)
    std = getattr(scaler, "std", getattr(scaler, "scale", None))
    if mean is None or std is None:
        raise ScalingValidationError("scaler must expose mean and std/scale arrays.")
    return FeatureScaler(mean=mean, std=std)


def _check_feature_schema(
    scaler: FeatureScaler,
    feature_names: Optional[Sequence[str]],
    feature_count: int,
) -> None:
    if scaler.feature_count != feature_count:
        raise ScalingValidationError(
            f"Feature width {feature_count} does not match scaler width {scaler.feature_count}."
        )
    if feature_names is None:
        return
    names = _normalize_feature_names(feature_names, feature_count)
    if scaler.feature_names and names != scaler.feature_names:
        raise ScalingValidationError("Feature names do not match the fitted scaler schema.")
    current_hash = _feature_schema_hash(names)
    if scaler.feature_schema_hash and current_hash and scaler.feature_schema_hash != current_hash:
        raise ScalingValidationError("Feature schema hash does not match the fitted scaler.")


def apply_scaler(
    features: Any,
    scaler: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Apply a previously fitted scaler without changing its stored statistics."""
    fitted = _coerce_scaler(scaler)
    matrix = _finite_matrix(features, allow_empty_rows=True)
    _check_feature_schema(fitted, feature_names, matrix.shape[1])

    transformed = (matrix - fitted.mean) / fitted.std
    preserve = fitted.passthrough_mask
    if fitted.clip_value is not None:
        continuous = ~preserve
        transformed[:, continuous] = np.clip(
            transformed[:, continuous],
            -float(fitted.clip_value),
            float(fitted.clip_value),
        )
    transformed[:, preserve] = matrix[:, preserve]
    if not np.all(np.isfinite(transformed)):
        raise ScalingValidationError("Scaled feature matrix contains NaN or infinity.")
    return transformed.astype(np.float32, copy=False)


def inverse_scaler(
    scaled_features: Any,
    scaler: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Reverse standard or robust scaling for continuous feature columns."""
    fitted = _coerce_scaler(scaler)
    matrix = _finite_matrix(scaled_features, allow_empty_rows=True)
    _check_feature_schema(fitted, feature_names, matrix.shape[1])
    restored = matrix * fitted.std + fitted.mean
    preserve = fitted.passthrough_mask
    restored[:, preserve] = matrix[:, preserve]
    if not np.all(np.isfinite(restored)):
        raise ScalingValidationError("Inverse-scaled matrix contains NaN or infinity.")
    return restored.astype(np.float32, copy=False)


# ============================================================================
# DATASET TRANSFORMATION
# ============================================================================

def _dataset_feature_names(dataset: Mapping[str, Any]) -> Optional[Sequence[str]]:
    names = dataset.get("feature_names")
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return names
    return None


def _transform_optional_matrix(
    dataset: Mapping[str, Any],
    source_key: str,
    target_key: str,
    scaler: FeatureScaler,
    feature_names: Optional[Sequence[str]],
) -> Optional[np.ndarray]:
    value = dataset.get(source_key)
    if value is None:
        return None
    matrix = _finite_matrix(value, allow_empty_rows=True)
    if matrix.shape[0] == 0:
        return np.empty((0, scaler.feature_count), dtype=np.float32)
    return apply_scaler(matrix, scaler, feature_names=feature_names)


def transform_dataset(
    dataset: Mapping[str, Any],
    scaler: Any,
    *,
    replace_features: bool = False,
    transform_candidates: bool = True,
    transform_all_setup_features: bool = True,
) -> Dict[str, Any]:
    """Return a dataset copy with scaled matrices and the raw data preserved.

    By default the original ``features`` matrix is retained and its transformed
    counterpart is stored in ``scaled_features``. Set ``replace_features`` to
    make ``features`` point to the scaled matrix while preserving the original
    matrix in ``raw_features``.
    """
    if not isinstance(dataset, Mapping):
        raise ScalingValidationError("dataset must be a mapping.")
    fitted = _coerce_scaler(scaler)
    result = dict(dataset)
    feature_names = _dataset_feature_names(dataset)

    raw_features = _finite_matrix(dataset.get("features"), allow_empty_rows=True)
    scaled_features = apply_scaler(raw_features, fitted, feature_names=feature_names)
    result["scaled_features"] = scaled_features
    if replace_features:
        result["raw_features"] = raw_features.copy()
        result["features"] = scaled_features

    if transform_candidates:
        scaled_candidates = _transform_optional_matrix(
            dataset,
            "candidate_features",
            "scaled_candidate_features",
            fitted,
            feature_names,
        )
        if scaled_candidates is not None:
            result["scaled_candidate_features"] = scaled_candidates
            if replace_features:
                result["raw_candidate_features"] = np.asarray(
                    dataset.get("candidate_features"), dtype=np.float32
                ).copy()
                result["candidate_features"] = scaled_candidates

    if transform_all_setup_features:
        scaled_all = _transform_optional_matrix(
            dataset,
            "all_setup_features",
            "scaled_all_setup_features",
            fitted,
            feature_names,
        )
        if scaled_all is not None:
            result["scaled_all_setup_features"] = scaled_all
            if replace_features:
                result["raw_all_setup_features"] = np.asarray(
                    dataset.get("all_setup_features"), dtype=np.float32
                ).copy()
                result["all_setup_features"] = scaled_all

    result["scaling"] = {
        "schema_version": SCALING_SCHEMA_VERSION,
        "status": "FITTED",
        "method": fitted.method,
        "fit_scope": fitted.fit_scope,
        "fit_sample_count": fitted.fit_sample_count,
        "feature_count": fitted.feature_count,
        "feature_schema_hash": fitted.feature_schema_hash,
        "clip_value": fitted.clip_value,
    }
    result["scaler"] = fitted.to_dict()
    return result


def fit_transform_dataset(
    dataset: Mapping[str, Any],
    *,
    method: str = DEFAULT_SCALING_METHOD,
    clip_value: Optional[float] = DEFAULT_CLIP_VALUE,
    replace_features: bool = False,
) -> Tuple[Dict[str, Any], FeatureScaler]:
    """Fit from train indices and transform all dataset feature matrices."""
    scaler = fit_dataset_scaler(dataset, method=method, clip_value=clip_value)
    transformed = transform_dataset(
        dataset,
        scaler,
        replace_features=replace_features,
    )
    return transformed, scaler


def transform_splits(
    dataset: Mapping[str, Any],
    scaler: Any,
) -> Dict[str, np.ndarray]:
    """Return independently indexed train, validation and test feature matrices."""
    if not isinstance(dataset, Mapping):
        raise ScalingValidationError("dataset must be a mapping.")
    fitted = _coerce_scaler(scaler)
    matrix = _finite_matrix(dataset.get("features"), allow_empty_rows=True)
    names = _dataset_feature_names(dataset)
    scaled = apply_scaler(matrix, fitted, feature_names=names)
    splits = dataset.get("splits", {})
    if not isinstance(splits, Mapping):
        raise ScalingValidationError("dataset.splits must be a mapping.")

    result: Dict[str, np.ndarray] = {}
    for split_name in ("train", "validation", "test"):
        raw_indices = splits.get(split_name + "_indices", [])
        if not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raise ScalingValidationError(f"splits.{split_name}_indices must be a sequence.")
        indices = np.asarray([int(item) for item in raw_indices], dtype=np.int64)
        if indices.size and (np.any(indices < 0) or np.any(indices >= matrix.shape[0])):
            raise ScalingValidationError(f"splits.{split_name}_indices contains an invalid index.")
        result[split_name] = scaled[indices]
    return result


# ============================================================================
# VALIDATION AND SERIALIZATION
# ============================================================================

def validate_scaler(scaler: Any) -> Dict[str, Any]:
    """Validate a scaler object or serialized mapping."""
    errors = []
    warnings = []
    try:
        fitted = _coerce_scaler(scaler)
    except (ScalingError, TypeError, ValueError) as error:
        return {"valid": False, "errors": [str(error)], "warnings": []}

    if fitted.fit_sample_count <= 0:
        warnings.append("Scaler does not record a positive fitting sample count.")
    if fitted.fit_scope != "TRAIN_SPLIT_ONLY":
        warnings.append("Scaler was not proven to be fitted exclusively on the training split.")
    if not fitted.feature_names:
        warnings.append("Scaler has no feature names; only feature width can be checked.")
    if np.any(fitted.constant_mask):
        warnings.append(
            f"{int(np.sum(fitted.constant_mask))} constant feature(s) were assigned unit scale."
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "feature_count": fitted.feature_count,
        "fit_sample_count": fitted.fit_sample_count,
        "fit_scope": fitted.fit_scope,
        "method": fitted.method,
        "feature_schema_hash": fitted.feature_schema_hash,
    }


def scaler_to_serializable(scaler: Any) -> Dict[str, Any]:
    """Convert a scaler to a JSON-compatible dictionary."""
    return _coerce_scaler(scaler).to_dict()


def scaler_from_serializable(value: Mapping[str, Any]) -> FeatureScaler:
    """Restore a scaler from a JSON-compatible dictionary."""
    return FeatureScaler.from_dict(value)


def dataset_scaling_report(dataset: Mapping[str, Any], scaler: Any) -> Dict[str, Any]:
    """Summarize the dataset/scaler compatibility and train-only fitting scope."""
    fitted = _coerce_scaler(scaler)
    matrix = _finite_matrix(dataset.get("features"), allow_empty_rows=True)
    names = _dataset_feature_names(dataset)
    _check_feature_schema(fitted, names, matrix.shape[1])
    splits = dataset.get("splits", {})
    train_count = len(splits.get("train_indices", [])) if isinstance(splits, Mapping) else 0
    validation_count = len(splits.get("validation_indices", [])) if isinstance(splits, Mapping) else 0
    test_count = len(splits.get("test_indices", [])) if isinstance(splits, Mapping) else 0
    return {
        "valid": True,
        "dataset_id": dataset.get("dataset_id"),
        "fit_dataset_id": fitted.fit_dataset_id,
        "feature_schema_hash": fitted.feature_schema_hash,
        "feature_count": fitted.feature_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "fit_sample_count": fitted.fit_sample_count,
        "fit_scope": fitted.fit_scope,
        "method": fitted.method,
    }


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "SCALING_SCHEMA_VERSION",
    "DEFAULT_SCALING_METHOD",
    "SUPPORTED_SCALING_METHODS",
    "DEFAULT_CLIP_VALUE",
    "ScalingError",
    "ScalingValidationError",
    "FeatureScaler",
    "build_passthrough_mask",
    "fit_scaler",
    "fit_dataset_scaler",
    "apply_scaler",
    "inverse_scaler",
    "transform_dataset",
    "fit_transform_dataset",
    "transform_splits",
    "validate_scaler",
    "scaler_to_serializable",
    "scaler_from_serializable",
    "dataset_scaling_report",
]
