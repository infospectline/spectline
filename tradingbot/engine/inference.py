"""Point-in-time-safe inference for approved Spectline TP-before-SL models."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# ENGINE DEPENDENCIES
# ============================================================================

try:
    from . import data as data_engine
    from . import dataset as dataset_engine
    from . import memory as memory_engine
    from . import model as model_engine
    from . import storage as storage_engine
    from . import versions as versions_engine
except ImportError:
    try:
        from engine import data as data_engine
        from engine import dataset as dataset_engine
        from engine import memory as memory_engine
        from engine import model as model_engine
        from engine import storage as storage_engine
        from engine import versions as versions_engine
    except ImportError:
        data_engine = None
        dataset_engine = None
        memory_engine = None
        model_engine = None
        storage_engine = None
        versions_engine = None


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

INFERENCE_SCHEMA_VERSION = 1
MODEL_TARGET_SCHEMA = "TP_BEFORE_SL_BINARY"
DATASET_TARGET = "TP_BEFORE_SL"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
DEFAULT_PERSIST_PREDICTIONS = True
MAX_FUTURE_PREDICTION_SKEW_SECONDS = 5
_IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")
_FEATURE_SETUP_FIELDS = (
    "setup_id",
    "trade_id",
    "direction",
    "setup_timeframe",
    "entry_timeframe",
    "resolution_timeframe",
    "decision_time",
    "entry_time",
    "entry_candle_time",
    "anchor_time",
    "anchor_price",
    "breakout_time",
    "breakout",
    "entry",
    "sl",
    "tp",
    "rr_mode",
    "zones",
)


# ============================================================================
# ERRORS AND CONFIGURATION
# ============================================================================

class InferenceError(RuntimeError):
    """Base exception for production inference operations."""


class InferenceValidationError(InferenceError, ValueError):
    """Raised when an inference request violates the input contract."""


class InferenceDependencyError(InferenceError, ImportError):
    """Raised when a required engine module is unavailable."""


class InferenceModelError(InferenceError):
    """Raised when an approved model or one of its dependencies is invalid."""


class InferenceChronologyError(InferenceError):
    """Raised when model or feature timestamps violate point-in-time rules."""


class InferencePersistenceError(InferenceError):
    """Raised when a requested prediction cannot be immutably recorded."""



# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def _finite_float(value: Any, field_name: str) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_timestamp(value: Any) -> Optional[int]:
    if data_engine is not None:
        parser = getattr(data_engine, "parse_timestamp", None)
        if callable(parser):
            return parser(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _finite_float(value, "timestamp")
        if number is None:
            return None
        if abs(number) >= 100_000_000_000:
            number /= 1000.0
        return int(number)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return int(parsed.astimezone(timezone.utc).timestamp())
    except (OverflowError, OSError, ValueError):
        return None


def _identity_payload(identity: Any) -> Dict[str, str]:
    if identity is None:
        raise InferenceValidationError("A complete TradingBot identity is required.")
    if callable(getattr(identity, "as_dict", None)):
        raw = identity.as_dict()
    elif isinstance(identity, Mapping):
        raw = dict(identity)
    else:
        raw = {field_name: getattr(identity, field_name, None) for field_name in _IDENTITY_FIELDS}
    result: Dict[str, str] = {}
    for field_name in _IDENTITY_FIELDS:
        value = raw.get(field_name)
        if value is None or not str(value).strip():
            raise InferenceValidationError(f"TradingBot identity is missing {field_name}.")
        normalized = str(value).strip()
        if field_name == "market":
            normalized = normalized.upper().replace("/", "_").replace("\\", "_")
        result[field_name] = normalized
    return result


def _resolve_identity(
    market: str,
    identity: Any = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> Any:
    if memory_engine is None:
        raise InferenceDependencyError("engine.memory is unavailable.")
    if model_engine is not None and callable(getattr(model_engine, "normalize_market", None)):
        normalized_market = model_engine.normalize_market(market)
    else:
        normalized_market = str(market).strip().upper().replace("/", "_").replace("\\", "_")
    if not normalized_market:
        raise InferenceValidationError("market cannot be empty.")
    resolver = getattr(memory_engine, "_resolve_memory_identity", None)
    if callable(resolver):
        return resolver(
            normalized_market,
            identity=identity,
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
    resolver = getattr(memory_engine, "resolve_identity", None)
    if callable(resolver):
        if identity is not None:
            values = _identity_payload(identity)
            return resolver(
                market=normalized_market,
                user_id=values["user_id"],
                strategy_id=values["strategy_id"],
                tradingbot_id=values["tradingbot_id"],
            )
        return resolver(
            market=normalized_market,
            user_id=user_id,
            strategy_id=strategy_id,
            tradingbot_id=tradingbot_id,
        )
    raise InferenceDependencyError("engine.memory does not expose an identity resolver.")


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    payload = "\n".join(str(item) for item in feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



# ============================================================================
# DEPENDENCY AND MODEL VERSION VALIDATION
# ============================================================================

def _required_modules() -> None:
    missing = [
        name
        for name, module in (
            ("engine.data", data_engine),
            ("engine.dataset", dataset_engine),
            ("engine.memory", memory_engine),
            ("engine.model", model_engine),
            ("engine.storage", storage_engine),
            ("engine.versions", versions_engine),
        )
        if module is None
    ]
    if missing:
        raise InferenceDependencyError("Required inference modules are unavailable: " + ", ".join(missing))


def _registry() -> Any:
    _required_modules()
    getter = getattr(versions_engine, "get_default_registry", None)
    if not callable(getter):
        raise InferenceDependencyError("engine.versions does not expose get_default_registry().")
    return getter()


def _approval_evidence(
    registry: Any,
    version_record: Any,
    identity: Any,
) -> Dict[str, Any]:
    scope_key = getattr(version_record, "scope_key", None)
    version_id = str(getattr(version_record, "version_id", ""))
    iterator = getattr(registry, "_status_events", None)
    if not scope_key or not callable(iterator):
        raise InferenceModelError("Model approval history is unavailable; refusing production inference.")

    matching_events = [
        dict(event)
        for event in iterator(str(scope_key), "model")
        if str(event.get("version_id", "")) == version_id
    ]
    validation_event = next(
        (
            event
            for event in matching_events
            if str(event.get("to_status", "")).upper() == "VALIDATED"
        ),
        None,
    )
    approval_event = next(
        (
            event
            for event in matching_events
            if str(event.get("to_status", "")).upper() == "APPROVED"
        ),
        None,
    )
    if validation_event is None or approval_event is None:
        raise InferenceModelError("Model has no complete validation and approval event history.")

    validation_metadata = validation_event.get("validation_metadata", {})
    if not isinstance(validation_metadata, Mapping):
        raise InferenceModelError("Model approval event has malformed evaluation metadata.")
    evaluation_id = str(validation_metadata.get("evaluation_version_id", "")).strip()
    passed = validation_metadata.get("passed") is True or validation_metadata.get("evaluation_passed") is True
    evaluated_split = str(validation_metadata.get("evaluated_split", "")).strip().lower()
    target_schema = str(validation_metadata.get("model_target_schema", "")).strip().upper()
    if not passed or evaluated_split != "test" or target_schema not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"} or not evaluation_id:
        raise InferenceModelError(
            "Model approval is missing a successful independent held-out test evaluation record."
        )

    try:
        evaluation_record = registry.get_version("evaluation", evaluation_id, identity=identity)
        registry.verify_artifact("evaluation", evaluation_id, identity=identity)
    except Exception as error:
        raise InferenceModelError("The model's held-out evaluation artifact is missing or invalid.") from error
    if str(evaluation_record.metadata.get("model_version_id", "")) != version_id:
        raise InferenceModelError("The held-out evaluation artifact references another model version.")
    if evaluation_record.metadata.get("passed") is not True:
        raise InferenceModelError("The persisted evaluation artifact does not contain a passing result.")
    if str(evaluation_record.metadata.get("split", "")).lower() != "test":
        raise InferenceModelError("The persisted evaluation artifact is not based on the test split.")

    approval_time = _parse_timestamp(approval_event.get("event_timestamp"))
    validation_time = _parse_timestamp(validation_event.get("event_timestamp"))
    if approval_time is None or validation_time is None or approval_time < validation_time:
        raise InferenceModelError("Model approval timestamps are missing or inconsistent.")
    return {
        "evaluation_version_id": evaluation_id,
        "evaluated_split": "test",
        "validation_time": validation_time,
        "approval_time": approval_time,
        "validation_event_id": validation_event.get("event_id"),
        "approval_event_id": approval_event.get("event_id"),
    }


def load_approved_model(
    market: str,
    *,
    identity: Any,
    model_version_id: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load one approved version and verify all model, dataset and scaler dependencies."""
    _required_modules()
    resolved = _resolve_identity(market, identity=identity)
    identity_data = _identity_payload(resolved)
    normalized_market = identity_data["market"]

    try:
        model, scaler, checkpoint = memory_engine.load_model(
            normalized_market,
            identity=resolved,
            version_id=model_version_id,
            require_approved=True,
        )
    except Exception as error:
        raise InferenceModelError("Could not load an approved model for this TradingBot.") from error

    version_payload = checkpoint.get("version_record")
    if not isinstance(version_payload, Mapping):
        raise InferenceModelError(
            "The loaded checkpoint is not backed by an immutable approved model version."
        )
    if str(version_payload.get("status", "")).upper() != "APPROVED":
        raise InferenceModelError("The loaded model version is not approved.")
    if str(version_payload.get("artifact_type", "")).lower() != "model":
        raise InferenceModelError("The loaded version record is not a model artifact.")
    if _identity_payload(version_payload.get("identity")) != identity_data:
        raise PermissionError("The approved model belongs to another TradingBot identity.")

    loaded_version_id = str(checkpoint.get("model_version_id", "")).strip()
    if not loaded_version_id:
        raise InferenceModelError("The loaded checkpoint has no immutable model version ID.")
    if model_version_id is not None and loaded_version_id != str(model_version_id):
        raise InferenceModelError("The loaded model version does not match the requested version.")

    summary = checkpoint.get("training_summary", {})
    if not isinstance(summary, Mapping):
        raise InferenceModelError("Model training summary is missing or malformed.")
    target_schema = str(summary.get("model_target_schema", "")).strip().upper()
    if target_schema != MODEL_TARGET_SCHEMA:
        raise InferenceModelError("The loaded model does not use the TP-before-SL binary target schema.")
    if summary.get("validation_available") is not True:
        raise InferenceModelError("The model was not trained with a usable chronological validation split.")
    if str(summary.get("selection_metric", "")).upper() != "VALIDATION_LOG_LOSS":
        raise InferenceModelError("The model was not selected using validation log loss.")

    dataset_version_id = str(checkpoint.get("dataset_version_id") or summary.get("dataset_version_id") or "").strip()
    scaler_version_id = str(checkpoint.get("scaler_version_id") or summary.get("scaler_version_id") or "").strip()
    if not dataset_version_id or not scaler_version_id:
        raise InferenceModelError("The model checkpoint does not reference an immutable dataset and scaler.")

    registry = _registry()
    try:
        model_record = registry.get_version("model", loaded_version_id, identity=resolved)
        dataset_record = registry.get_version("dataset", dataset_version_id, identity=resolved)
        scaler_record = registry.get_version("scaler", scaler_version_id, identity=resolved)
        registry.verify_artifact("model", loaded_version_id, identity=resolved)
        registry.verify_artifact("dataset", dataset_version_id, identity=resolved)
        registry.verify_artifact("scaler", scaler_version_id, identity=resolved)
    except Exception as error:
        raise InferenceModelError("One or more immutable model dependencies failed verification.") from error

    if model_record.status != "APPROVED":
        raise InferenceModelError("Model version is not approved in the version registry.")
    if dataset_record.status != "APPROVED" or scaler_record.status != "APPROVED":
        raise InferenceModelError("Model dataset and scaler dependencies must both be approved.")
    if str(model_record.metadata.get("model_target_schema", "")).upper() != MODEL_TARGET_SCHEMA:
        raise InferenceModelError("Model registry metadata contains an incompatible target schema.")
    if int(getattr(model, "input_size", -1)) <= 0:
        raise InferenceModelError("Loaded model has an invalid input width.")

    feature_names = summary.get("feature_names")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)) or not feature_names:
        raise InferenceModelError("Model checkpoint does not contain its ordered feature schema.")
    normalized_feature_names = tuple(str(item) for item in feature_names)
    calculated_schema_hash = _feature_schema_hash(normalized_feature_names)
    expected_schema_hash = str(summary.get("feature_schema_hash", "")).strip()
    if not expected_schema_hash or calculated_schema_hash != expected_schema_hash:
        raise InferenceModelError("Model feature names do not match the stored feature schema hash.")
    if len(normalized_feature_names) != int(model.input_size):
        raise InferenceModelError("Model input width does not match its stored feature names.")
    if scaler is None or int(getattr(scaler, "feature_count", -1)) != len(normalized_feature_names):
        raise InferenceModelError("Scaler width does not match the model feature schema.")
    scaler_names = tuple(str(item) for item in (getattr(scaler, "feature_names", ()) or ()))
    if scaler_names != normalized_feature_names:
        raise InferenceModelError("Scaler feature order does not match the model feature order.")
    if str(getattr(scaler, "fit_scope", "")).upper() != "TRAIN_SPLIT_ONLY":
        raise InferenceModelError("Scaler was not fitted exclusively on the training split.")
    if str(getattr(scaler, "feature_schema_hash", "")) != expected_schema_hash:
        raise InferenceModelError("Scaler feature schema hash does not match the model.")
    if int(getattr(scaler, "fit_sample_count", 0)) <= 0:
        raise InferenceModelError("Scaler has no valid training fit sample count.")

    metadata_schema_hash = str(model_record.metadata.get("feature_schema_hash", "")).strip()
    if metadata_schema_hash and metadata_schema_hash != expected_schema_hash:
        raise InferenceModelError("Model version metadata has a conflicting feature schema hash.")

    dependencies = [dict(item) for item in model_record.dependencies]
    dependency_pairs = {
        (str(item.get("artifact_type", "")).lower(), str(item.get("version_id", "")))
        for item in dependencies
    }
    if ("dataset", dataset_version_id) not in dependency_pairs:
        raise InferenceModelError("Model version manifest does not declare its dataset dependency.")
    if ("scaler", scaler_version_id) not in dependency_pairs:
        raise InferenceModelError("Model version manifest does not declare its scaler dependency.")

    approval = _approval_evidence(registry, model_record, resolved)
    trained_at = _parse_timestamp(checkpoint.get("trained_at") or summary.get("trained_at"))
    if trained_at is None:
        raise InferenceModelError("Model checkpoint does not contain a valid training timestamp.")

    metadata = {
        "identity": identity_data,
        "market": normalized_market,
        "model_version_id": loaded_version_id,
        "dataset_version_id": dataset_version_id,
        "scaler_version_id": scaler_version_id,
        "model_target_schema": target_schema,
        "feature_names": normalized_feature_names,
        "feature_schema_hash": expected_schema_hash,
        "trained_at": trained_at,
        "model_status": model_record.status,
        "model_created_at": _parse_timestamp(model_record.created_at),
        "approval": approval,
        "calibration_status": str(summary.get("calibration_status", "NOT_CALIBRATED")).upper(),
        "classification_threshold": float(summary.get("classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)),
    }
    return model, scaler, metadata


# ============================================================================
# POINT-IN-TIME FEATURE PREPARATION
# ============================================================================

def _bundle_identity(bundle: Mapping[str, Any]) -> Dict[str, str]:
    identity = bundle.get("identity")
    return _identity_payload(identity)


def _setup_record_from_bundle(bundle: Mapping[str, Any], setup_id: str) -> Dict[str, Any]:
    records = bundle.get("setup_input_records", {})
    if isinstance(records, Mapping):
        candidate = records.get(setup_id)
        if candidate is None:
            for key, value in records.items():
                if str(key) == setup_id:
                    candidate = value
                    break
        if isinstance(candidate, Mapping):
            return dict(candidate)
    learning = bundle.get("learning_setups", {})
    normalized = learning.get("normalized", []) if isinstance(learning, Mapping) else []
    for item in normalized if isinstance(normalized, Sequence) else ():
        if str(getattr(item, "setup_id", "")) == setup_id:
            record_builder = getattr(data_engine, "setup_input_record", None)
            if callable(record_builder):
                return dict(record_builder(item))
    raise InferenceValidationError(f"Setup {setup_id!r} was not found in the supplied data bundle.")


def _validate_data_bundle(bundle: Mapping[str, Any], identity: Any, market: str) -> None:
    if not isinstance(bundle, Mapping):
        raise InferenceValidationError("data_bundle must be a mapping.")
    identity_data = _identity_payload(identity)
    if _bundle_identity(bundle) != identity_data:
        raise PermissionError("Data bundle identity does not match the requested TradingBot.")
    bundle_market = str(bundle.get("market", "")).strip().upper().replace("/", "_").replace("\\", "_")
    if bundle_market != market:
        raise InferenceValidationError("Data bundle market does not match the requested market.")
    validation = bundle.get("validation", {})
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        errors = validation.get("errors", []) if isinstance(validation, Mapping) else []
        raise InferenceValidationError("Inference data bundle failed validation: " + "; ".join(str(item) for item in errors))


def _normalize_candle_sequence(candles: Any) -> list[Any]:
    if data_engine is None or not callable(getattr(data_engine, "normalize_candles", None)):
        raise InferenceDependencyError("engine.data.normalize_candles() is unavailable.")
    if candles is None:
        return []
    if isinstance(candles, (str, bytes)):
        raise InferenceValidationError("Candle collections must be sequences, not text.")
    try:
        normalized, report = data_engine.normalize_candles(candles)
    except Exception as error:
        raise InferenceValidationError("Could not normalize a supplied candle collection.") from error
    if report.get("invalid_candle_count", 0):
        raise InferenceValidationError(
            f"Candle collection contains {int(report['invalid_candle_count'])} invalid record(s)."
        )
    return list(normalized)


def _normalize_bundle_candles(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    normalized_bundle = dict(bundle)
    historical = bundle.get("historical_data", {})
    if not isinstance(historical, Mapping):
        raise InferenceValidationError("data_bundle.historical_data must be a mapping.")
    normalized_historical = dict(historical)
    candles_by_timeframe = historical.get("candles_by_timeframe", {})
    if not isinstance(candles_by_timeframe, Mapping):
        raise InferenceValidationError("historical_data.candles_by_timeframe must be a mapping.")
    normalized_historical["candles_by_timeframe"] = {
        str(timeframe).strip().upper(): _normalize_candle_sequence(candles)
        for timeframe, candles in candles_by_timeframe.items()
    }
    point_in_time = historical.get("point_in_time_candles_by_setup", {})
    if point_in_time is not None:
        if not isinstance(point_in_time, Mapping):
            raise InferenceValidationError("point_in_time_candles_by_setup must be a mapping.")
        normalized_point_in_time: Dict[str, Dict[str, list[Any]]] = {}
        for setup_key, timeframe_map in point_in_time.items():
            if not isinstance(timeframe_map, Mapping):
                raise InferenceValidationError("Point-in-time candle entries must be timeframe mappings.")
            normalized_point_in_time[str(setup_key)] = {
                str(timeframe).strip().upper(): _normalize_candle_sequence(candles)
                for timeframe, candles in timeframe_map.items()
            }
        normalized_historical["point_in_time_candles_by_setup"] = normalized_point_in_time
    normalized_bundle["historical_data"] = normalized_historical
    return normalized_bundle


def build_setup_feature_row(
    setup_id: Any,
    data_bundle: Mapping[str, Any],
    *,
    identity: Any,
) -> Dict[str, Any]:
    """Build the exact training-schema feature row for one point-in-time setup."""
    _required_modules()
    normalized_setup_id = str(setup_id).strip()
    if not normalized_setup_id:
        raise InferenceValidationError("setup_id cannot be empty.")
    identity_data = _identity_payload(identity)
    _validate_data_bundle(data_bundle, identity, identity_data["market"])
    data_bundle = _normalize_bundle_candles(data_bundle)

    setup_record = _setup_record_from_bundle(data_bundle, normalized_setup_id)
    decision_time = _parse_timestamp(setup_record.get("decision_time"))
    if decision_time is None:
        decision_time = _parse_timestamp(setup_record.get("predicted_at"))
    if decision_time is None:
        decision_time = _parse_timestamp(setup_record.get("entry_time", setup_record.get("entry_candle_time")))
    if decision_time is None:
        raise InferenceChronologyError("Setup has no valid decision_time for point-in-time feature construction.")

    try:
        dataset = dataset_engine.build_training_dataset(
            data_bundle=data_bundle,
            include_unlabelled_candidates=True,
            require_identity=True,
        )
    except Exception as error:
        raise InferenceValidationError("Could not build point-in-time features from the supplied data bundle.") from error

    matrix = np.asarray(dataset.get("all_setup_features", []), dtype=np.float32)
    metadata_rows = dataset.get("all_setup_feature_metadata", [])
    feature_names = dataset.get("feature_names", [])
    feature_schema = dataset.get("feature_schema", {})
    if matrix.ndim != 2 or not isinstance(metadata_rows, Sequence) or matrix.shape[0] != len(metadata_rows):
        raise InferenceValidationError("Dataset feature builder returned inconsistent setup feature rows.")
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)) or not feature_names:
        raise InferenceValidationError("Dataset feature builder returned no ordered feature names.")
    schema_hash = str(feature_schema.get("sha256", "")).strip() if isinstance(feature_schema, Mapping) else ""
    if not schema_hash or schema_hash != _feature_schema_hash(feature_names):
        raise InferenceValidationError("Dataset feature schema hash is missing or invalid.")

    selected_index = None
    selected_metadata = None
    for index, item in enumerate(metadata_rows):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("setup_id", "")) == normalized_setup_id:
            selected_index = index
            selected_metadata = dict(item)
            break
    if selected_index is None or selected_metadata is None:
        exclusions = [
            dict(item)
            for item in dataset.get("excluded_setups", [])
            if isinstance(item, Mapping) and str(item.get("setup_id", "")) == normalized_setup_id
        ]
        reason = exclusions[0].get("reason") if exclusions else "No feature row was produced for the requested setup."
        raise InferenceValidationError(f"Setup {normalized_setup_id!r} is not inference-ready: {reason}")

    features = np.asarray(matrix[selected_index], dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(features)):
        raise InferenceValidationError("Setup feature row contains NaN or infinite values.")
    feature_decision_time = _parse_timestamp(selected_metadata.get("decision_time"))
    if feature_decision_time is None or feature_decision_time != decision_time:
        raise InferenceChronologyError("Feature builder decision time does not match the setup decision time.")
    if features.size != len(feature_names):
        raise InferenceValidationError("Feature row width does not match the feature schema.")

    return {
        "setup_id": normalized_setup_id,
        "trade_id": str(setup_record.get("trade_id") or normalized_setup_id),
        "market": identity_data["market"],
        "identity": identity_data,
        "decision_time": decision_time,
        "entry_time": _parse_timestamp(
            setup_record.get("entry_time", setup_record.get("entry_candle_time"))
        ),
        "features": features,
        "feature_names": tuple(str(item) for item in feature_names),
        "feature_schema_hash": schema_hash,
        "feature_metadata": selected_metadata,
        "source_bundle_schema_version": data_bundle.get("schema_version"),
        "source_bundle_created_at": data_bundle.get("created_at"),
        "known_pattern_count": int(selected_metadata.get("known_pattern_count", 0)),
        "known_event_count": int(selected_metadata.get("known_event_count", 0)),
    }


def build_setup_feature_row_from_context(
    setup_record: Mapping[str, Any],
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    *,
    market: str,
    identity: Any,
    point_in_time_candles_by_setup: Optional[Mapping[str, Any]] = None,
    market_memory_by_setup: Optional[Mapping[str, Any]] = None,
    relevance_by_setup: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a point-in-time feature row from explicitly supplied market context."""
    _required_modules()
    if not isinstance(setup_record, Mapping):
        raise InferenceValidationError("setup_record must be a mapping.")
    identity_data = _identity_payload(identity)
    normalized_market = str(market).strip().upper().replace("/", "_").replace("\\", "_")
    if normalized_market != identity_data["market"]:
        raise InferenceValidationError("market does not match the supplied TradingBot identity.")
    setup_id_value = setup_record.get("setup_id", setup_record.get("id"))
    if setup_id_value is None or not str(setup_id_value).strip():
        raise InferenceValidationError("setup_record requires setup_id or id.")
    setup_id = str(setup_id_value).strip()
    sanitized = {
        field_name: setup_record[field_name]
        for field_name in _FEATURE_SETUP_FIELDS
        if field_name in setup_record
    }
    sanitized["setup_id"] = setup_id
    sanitized.setdefault("trade_id", str(setup_record.get("trade_id") or setup_id))
    bundle = {
        "schema_version": 1,
        "identity": identity_data,
        "market": normalized_market,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "setup_input_records": {setup_id: sanitized},
        "historical_data": {
            "candles_by_timeframe": dict(candles_by_timeframe),
            "point_in_time_candles_by_setup": dict(point_in_time_candles_by_setup or {}),
        },
        "market_memory_by_setup": dict(market_memory_by_setup or {}),
        "relevance_by_setup": dict(relevance_by_setup or {}),
        "validation": {"valid": True, "errors": [], "warnings": []},
    }
    return build_setup_feature_row(setup_id, bundle, identity=identity)


# ============================================================================
# PREDICTION EXECUTION
# ============================================================================

def _check_prediction_chronology(
    decision_time: int,
    prediction_time: int,
    entry_time: Optional[int],
    model_metadata: Mapping[str, Any],
    *,
    allow_post_entry_analysis: bool,
) -> str:
    if prediction_time < decision_time:
        raise InferenceChronologyError("Prediction time cannot precede the feature decision time.")
    trained_at = _parse_timestamp(model_metadata.get("trained_at"))
    if trained_at is None or decision_time < trained_at:
        raise InferenceChronologyError("The model was trained after the requested decision time.")
    approval = model_metadata.get("approval", {})
    approval_time = _parse_timestamp(approval.get("approval_time")) if isinstance(approval, Mapping) else None
    if approval_time is None or prediction_time < approval_time:
        raise InferenceChronologyError("The model was not approved by the actual prediction time.")
    if prediction_time > int(time.time()) + MAX_FUTURE_PREDICTION_SKEW_SECONDS:
        raise InferenceChronologyError("prediction_time cannot be materially later than the current UTC time.")
    if entry_time is not None and prediction_time <= entry_time:
        return "ENTRY_PREDICTION"
    if entry_time is not None and not allow_post_entry_analysis:
        raise InferenceChronologyError("Prediction was requested after trade entry.")
    if entry_time is not None:
        return "POST_ENTRY_ANALYSIS"
    return "SETUP_CANDIDATE_PREDICTION"


def predict_feature_row(
    market: str,
    features: Any,
    feature_names: Sequence[str],
    *,
    identity: Any,
    decision_time: Any,
    setup_id: Optional[Any] = None,
    trade_id: Optional[Any] = None,
    entry_time: Any = None,
    feature_schema_hash: Optional[str] = None,
    model_version_id: Optional[str] = None,
    prediction_time: Any = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    persist: bool = DEFAULT_PERSIST_PREDICTIONS,
    allow_post_entry_analysis: bool = True,
) -> Dict[str, Any]:
    """Predict from one verified feature row and optionally append a timeline event."""
    _required_modules()
    resolved = _resolve_identity(market, identity=identity)
    identity_data = _identity_payload(resolved)
    normalized_market = identity_data["market"]
    decision_timestamp = _parse_timestamp(decision_time)
    entry_timestamp = _parse_timestamp(entry_time)
    prediction_timestamp = _parse_timestamp(prediction_time) if prediction_time is not None else int(time.time())
    if decision_timestamp is None:
        raise InferenceValidationError("decision_time must be a valid UTC timestamp.")
    if prediction_timestamp is None:
        raise InferenceValidationError("prediction_time must be a valid UTC timestamp.")
    if entry_time is not None and entry_timestamp is None:
        raise InferenceValidationError("entry_time was supplied but could not be parsed.")
    if entry_timestamp is not None and decision_timestamp > entry_timestamp:
        raise InferenceChronologyError("Feature decision time cannot be later than trade entry.")

    feature_name_tuple = tuple(str(item) for item in feature_names)
    if not feature_name_tuple or any(not item for item in feature_name_tuple):
        raise InferenceValidationError("feature_names must contain non-empty ordered names.")
    calculated_hash = _feature_schema_hash(feature_name_tuple)
    if feature_schema_hash is not None and str(feature_schema_hash) != calculated_hash:
        raise InferenceValidationError("Supplied feature_schema_hash does not match feature_names.")

    model, scaler, model_metadata = load_approved_model(
        normalized_market,
        identity=resolved,
        model_version_id=model_version_id,
    )
    if feature_name_tuple != tuple(model_metadata["feature_names"]):
        raise InferenceModelError("Inference feature names or ordering do not match the approved model.")
    if calculated_hash != model_metadata["feature_schema_hash"]:
        raise InferenceModelError("Inference feature schema hash does not match the approved model.")
    if _parse_timestamp(model_metadata.get("approval", {}).get("approval_time")) is None:
        raise InferenceModelError("Approved model has no verifiable approval timestamp.")

    role = _check_prediction_chronology(
        decision_timestamp,
        prediction_timestamp,
        entry_timestamp,
        model_metadata,
        allow_post_entry_analysis=bool(allow_post_entry_analysis),
    )
    try:
        prediction = model_engine.predict_from_features(
            model,
            scaler,
            features,
            feature_names=feature_name_tuple,
            threshold=threshold,
        )
    except Exception as error:
        raise InferenceValidationError("The approved model rejected the supplied feature row.") from error

    probability = _finite_float(prediction.get("tp_before_sl_probability"), "probability")
    if probability is None or not 0.0 <= probability <= 1.0:
        raise InferenceModelError("The model returned an invalid TP-before-SL probability.")
    entry_probability = probability if role == "ENTRY_PREDICTION" else None
    setup_identifier = str(setup_id).strip() if setup_id is not None else None
    trade_identifier = str(trade_id).strip() if trade_id is not None else setup_identifier
    if persist and not setup_identifier:
        raise InferencePersistenceError("Persisted inference requires a setup_id.")

    result: Dict[str, Any] = {
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "status": "PREDICTED",
        "market": normalized_market,
        "identity": identity_data,
        "setup_id": setup_identifier,
        "trade_id": trade_identifier,
        "decision_time": decision_timestamp,
        "feature_as_of_time": decision_timestamp,
        "entry_time": entry_timestamp,
        "predicted_at": prediction_timestamp,
        "prediction_role": role,
        "probability": probability,
        "tp_before_sl_probability": probability,
        "predicted_probability_at_entry": entry_probability,
        "predicted_outcome": prediction.get("predicted_outcome"),
        "classification_threshold": float(prediction.get("classification_threshold", threshold)),
        "target": DATASET_TARGET,
        "model_target_schema": MODEL_TARGET_SCHEMA,
        "calibration_status": model_metadata.get("calibration_status", "NOT_CALIBRATED"),
        "is_calibrated": str(model_metadata.get("calibration_status", "NOT_CALIBRATED")).upper() == "CALIBRATED",
        "model_version_id": model_metadata["model_version_id"],
        "dataset_version_id": model_metadata["dataset_version_id"],
        "scaler_version_id": model_metadata["scaler_version_id"],
        "feature_schema_hash": calculated_hash,
        "feature_count": len(feature_name_tuple),
        "known_pattern_count": None,
        "known_event_count": None,
        "persistence": {"requested": bool(persist), "persisted": False},
    }

    if persist:
        event_type = (
            "TRADE_PREDICTION"
            if role == "ENTRY_PREDICTION"
            else "TRADE_POST_ENTRY_MODEL_ESTIMATE"
            if role == "POST_ENTRY_ANALYSIS"
            else "SETUP_MODEL_PREDICTION"
        )
        event_identity = {
            **identity_data,
            "setup_id": setup_identifier,
            "trade_id": trade_identifier,
            "event_type": event_type,
            "event_time": prediction_timestamp,
            "model_version_id": model_metadata["model_version_id"],
            "probability": probability,
        }
        event_id = "prediction_" + _canonical_hash(event_identity)[:32]
        event: Dict[str, Any] = {
            "event_id": event_id,
            "event_type": event_type,
            "event_time": prediction_timestamp,
            "available_at": prediction_timestamp,
            "setup_id": setup_identifier,
            "trade_id": trade_identifier,
            "target": DATASET_TARGET,
            "probability": probability,
            "predicted_at": prediction_timestamp,
            "decision_time": decision_timestamp,
            "feature_as_of_time": decision_timestamp,
            "prediction_role": role,
            "model_version_id": model_metadata["model_version_id"],
            "dataset_version_id": model_metadata["dataset_version_id"],
            "scaler_version_id": model_metadata["scaler_version_id"],
            "feature_schema_hash": calculated_hash,
            "calibration_status": model_metadata.get("calibration_status", "NOT_CALIBRATED"),
        }
        if entry_probability is not None:
            event["predicted_probability_at_entry"] = entry_probability
        try:
            writer = getattr(storage_engine, "append_timeline_event", None)
            if not callable(writer):
                storage_class = getattr(storage_engine, "HistoricalStorage", None)
                storage = storage_class() if callable(storage_class) else storage_engine
                writer = getattr(storage, "append_timeline_event", None)
            if not callable(writer):
                raise AttributeError("append_timeline_event() is unavailable.")
            write_result = writer(resolved, event)
            result["persistence"] = {
                "requested": True,
                "persisted": True,
                "inserted": bool(getattr(write_result, "inserted", True)),
                "event_id": event_id,
                "record_id": getattr(write_result, "record_id", event_id),
                "checksum": getattr(write_result, "checksum", None),
                "path": str(getattr(write_result, "path", "")) or None,
            }
        except Exception as error:
            raise InferencePersistenceError("Prediction was calculated but could not be appended to the customer timeline.") from error

    return result


def predict_setup(
    market: str,
    setup_id: Any,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Any = None,
    data_bundle: Optional[Mapping[str, Any]] = None,
    model_version_id: Optional[str] = None,
    prediction_time: Any = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    persist: bool = DEFAULT_PERSIST_PREDICTIONS,
    allow_post_entry_analysis: bool = True,
    point_in_time_candle_limit: int = 240,
) -> Dict[str, Any]:
    """Prepare one saved Learning setup and run approved-model inference."""
    _required_modules()
    normalized_market = model_engine.normalize_market(market)
    resolved = _resolve_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    identity_data = _identity_payload(resolved)
    if data_bundle is None:
        try:
            data_bundle = data_engine.prepare_training_data(
                normalized_market,
                identity=resolved,
                point_in_time_candle_limit=point_in_time_candle_limit,
            )
        except Exception as error:
            raise InferenceValidationError("Could not prepare point-in-time market data for inference.") from error
    feature_record = build_setup_feature_row(setup_id, data_bundle, identity=resolved)
    result = predict_feature_row(
        normalized_market,
        feature_record["features"],
        feature_record["feature_names"],
        identity=resolved,
        decision_time=feature_record["decision_time"],
        entry_time=feature_record.get("entry_time"),
        setup_id=feature_record["setup_id"],
        trade_id=feature_record.get("trade_id"),
        feature_schema_hash=feature_record["feature_schema_hash"],
        model_version_id=model_version_id,
        prediction_time=prediction_time,
        threshold=threshold,
        persist=persist,
        allow_post_entry_analysis=allow_post_entry_analysis,
    )
    result["known_pattern_count"] = feature_record["known_pattern_count"]
    result["known_event_count"] = feature_record["known_event_count"]
    result["feature_metadata"] = feature_record["feature_metadata"]
    result["source_bundle_schema_version"] = feature_record["source_bundle_schema_version"]
    return result


def predict_setup_from_context(
    setup_record: Mapping[str, Any],
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    *,
    market: str,
    identity: Any,
    point_in_time_candles_by_setup: Optional[Mapping[str, Any]] = None,
    market_memory_by_setup: Optional[Mapping[str, Any]] = None,
    relevance_by_setup: Optional[Mapping[str, Any]] = None,
    model_version_id: Optional[str] = None,
    prediction_time: Any = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    persist: bool = DEFAULT_PERSIST_PREDICTIONS,
    allow_post_entry_analysis: bool = True,
) -> Dict[str, Any]:
    """Build features from explicit candles and market memory, then predict one setup."""
    feature_record = build_setup_feature_row_from_context(
        setup_record,
        candles_by_timeframe,
        market=market,
        identity=identity,
        point_in_time_candles_by_setup=point_in_time_candles_by_setup,
        market_memory_by_setup=market_memory_by_setup,
        relevance_by_setup=relevance_by_setup,
    )
    return predict_feature_row(
        market,
        feature_record["features"],
        feature_record["feature_names"],
        identity=identity,
        decision_time=feature_record["decision_time"],
        entry_time=feature_record.get("entry_time"),
        setup_id=feature_record["setup_id"],
        trade_id=feature_record.get("trade_id"),
        feature_schema_hash=feature_record["feature_schema_hash"],
        model_version_id=model_version_id,
        prediction_time=prediction_time,
        threshold=threshold,
        persist=persist,
        allow_post_entry_analysis=allow_post_entry_analysis,
    )


def predict_from_features(
    market: str,
    features: Any,
    feature_names: Sequence[str],
    *,
    identity: Any,
    decision_time: Any,
    model_version_id: Optional[str] = None,
    prediction_time: Any = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> Dict[str, Any]:
    """Low-level non-persisting inference for a verified point-in-time feature row."""
    return predict_feature_row(
        market,
        features,
        feature_names,
        identity=identity,
        decision_time=decision_time,
        model_version_id=model_version_id,
        prediction_time=prediction_time,
        threshold=threshold,
        persist=False,
        allow_post_entry_analysis=True,
    )


# ============================================================================
# MODEL INFORMATION AND PUBLIC EXPORTS
# ============================================================================

def get_inference_model_info(
    market: str,
    *,
    identity: Any,
    model_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return verified metadata for the approved model used by inference."""
    resolved = _resolve_identity(market, identity=identity)
    _, scaler, metadata = load_approved_model(
        market,
        identity=resolved,
        model_version_id=model_version_id,
    )
    return {
        **metadata,
        "scaler_fit_scope": getattr(scaler, "fit_scope", None),
        "scaler_fit_sample_count": int(getattr(scaler, "fit_sample_count", 0)),
        "scaler_method": getattr(scaler, "method", None),
    }


run_inference = predict_setup
infer_setup = predict_setup


__all__ = [
    "INFERENCE_SCHEMA_VERSION",
    "MODEL_TARGET_SCHEMA",
    "DATASET_TARGET",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "InferenceError",
    "InferenceValidationError",
    "InferenceDependencyError",
    "InferenceModelError",
    "InferenceChronologyError",
    "InferencePersistenceError",
    "load_approved_model",
    "build_setup_feature_row",
    "build_setup_feature_row_from_context",
    "predict_feature_row",
    "predict_setup",
    "predict_setup_from_context",
    "predict_from_features",
    "run_inference",
    "infer_setup",
    "get_inference_model_info",
]
