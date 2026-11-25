"""Point-in-time-safe inference for approved Spectline TP-before-SL models."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import hmac
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# ============================================================================
# ENGINE DEPENDENCIES
# ============================================================================

try:
    from . import data as data_engine
    from . import dataset as dataset_engine
    from . import memory as memory_engine
    from . import model as model_engine
    from . import scaling as scaling_engine
    from . import storage as storage_engine
except ImportError:
    try:
        from engine import data as data_engine
        from engine import dataset as dataset_engine
        from engine import memory as memory_engine
        from engine import model as model_engine
        from engine import scaling as scaling_engine
        from engine import storage as storage_engine
    except ImportError:
        data_engine = None
        dataset_engine = None
        memory_engine = None
        model_engine = None
        scaling_engine = None
        storage_engine = None


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
# DEPENDENCY AND SELF-CONTAINED MODEL VALIDATION
# ============================================================================

def _required_modules() -> None:
    missing = [
        name for name, module in (
            ("engine.data", data_engine),
            ("engine.dataset", dataset_engine),
            ("engine.memory", memory_engine),
            ("engine.model", model_engine),
            ("engine.scaling", scaling_engine),
            ("engine.storage", storage_engine),
        ) if module is None
    ]
    if missing:
        raise InferenceDependencyError(
            "Required inference modules are unavailable: " + ", ".join(missing)
        )


def _model_state_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Hash neural-network tensor values in a stable, ordered format."""
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise InferenceModelError("Model checkpoint has no valid model_state.")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise InferenceModelError(f"Model state entry {name!r} is not a tensor.")
        tensor = tensor.detach().cpu().contiguous()
        try:
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        except Exception as error:
            raise InferenceModelError(f"Could not hash model state tensor {name!r}.") from error
        header = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _checkpoint_contract(
    checkpoint: Mapping[str, Any],
    model: Any,
    scaler: Any,
) -> Dict[str, Any]:
    """Validate the model architecture, ordered feature contract and embedded scaler."""
    summary = checkpoint.get("training_summary", {})
    inference_meta = checkpoint.get("inference_metadata", {})
    architecture = checkpoint.get("model_architecture", {})
    if not isinstance(summary, Mapping) or not isinstance(inference_meta, Mapping) or not isinstance(architecture, Mapping):
        raise InferenceModelError("Model training or inference metadata is malformed.")

    target = str(
        inference_meta.get("model_target_schema")
        or checkpoint.get("model_target_schema")
        or summary.get("model_target_schema")
        or ""
    ).strip().upper()
    if target not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise InferenceModelError("The model does not use the TP-before-SL binary target schema.")
    if summary.get("validation_available") is not True:
        raise InferenceModelError("The model was not trained with a chronological validation split.")
    if str(summary.get("selection_metric", "")).upper() != "VALIDATION_LOG_LOSS":
        raise InferenceModelError("The model was not selected using validation log loss.")

    raw_names = inference_meta.get("feature_names") or checkpoint.get("feature_names") or summary.get("feature_names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)) or not raw_names:
        raise InferenceModelError("The .pth checkpoint has no ordered feature schema.")
    feature_names = tuple(str(item).strip() for item in raw_names)
    if any(not name for name in feature_names) or len(set(feature_names)) != len(feature_names):
        raise InferenceModelError("Model feature names are empty or duplicated.")

    schema_hash = str(
        inference_meta.get("feature_schema_hash")
        or checkpoint.get("feature_schema_hash")
        or summary.get("feature_schema_hash")
        or ""
    ).strip()
    if not schema_hash or _feature_schema_hash(feature_names) != schema_hash:
        raise InferenceModelError("Model feature names do not match the stored feature schema hash.")

    input_size = int(inference_meta.get("input_size") or architecture.get("input_size") or checkpoint.get("input_size", 0))
    if input_size <= 0 or int(getattr(model, "input_size", -1)) != input_size or input_size != len(feature_names):
        raise InferenceModelError("Model input width does not match its feature schema.")
    if architecture.get("input_size") is not None and int(architecture["input_size"]) != input_size:
        raise InferenceModelError("Nested model architecture has a conflicting input width.")
    if architecture.get("hidden_sizes") is not None and list(architecture["hidden_sizes"]) != list(checkpoint.get("hidden_sizes", ())):
        raise InferenceModelError("Nested model architecture has conflicting hidden sizes.")

    if checkpoint.get("model_class_module") and checkpoint["model_class_module"] != type(model).__module__:
        raise InferenceModelError("Model class module does not match the checkpoint.")
    if checkpoint.get("model_class_name") and checkpoint["model_class_name"] != type(model).__name__:
        raise InferenceModelError("Model class name does not match the checkpoint.")

    scaler_names = tuple(str(item) for item in (getattr(scaler, "feature_names", ()) or ()))
    if scaler_names != feature_names:
        raise InferenceModelError("Embedded scaler feature order does not match the model.")
    if int(getattr(scaler, "feature_count", -1)) != input_size:
        raise InferenceModelError("Embedded scaler width does not match the model.")
    if str(getattr(scaler, "fit_scope", "")).upper() != "TRAIN_SPLIT_ONLY":
        raise InferenceModelError("Embedded scaler was not fitted exclusively on training rows.")
    if int(getattr(scaler, "fit_sample_count", 0)) <= 0:
        raise InferenceModelError("Embedded scaler has no valid training sample count.")
    if str(getattr(scaler, "feature_schema_hash", "")) != schema_hash:
        raise InferenceModelError("Embedded scaler feature schema hash does not match the model.")

    mean = np.asarray(getattr(scaler, "mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(getattr(scaler, "std", []), dtype=np.float64).reshape(-1)
    if mean.size != input_size or std.size != input_size or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise InferenceModelError("Embedded scaler statistics are invalid.")

    trained_at = checkpoint.get("trained_at") or summary.get("trained_at")
    trained_timestamp = _parse_timestamp(trained_at)
    if trained_timestamp is None:
        raise InferenceModelError("Model checkpoint has no valid training timestamp.")

    threshold_value = inference_meta.get("classification_threshold")
    if threshold_value is None:
        threshold_value = checkpoint.get("classification_threshold", summary.get("classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD))
    threshold = _finite_float(threshold_value, "classification_threshold")
    if threshold is None or not 0.0 <= threshold <= 1.0:
        raise InferenceModelError("Model classification threshold is invalid.")

    return {
        "summary": summary,
        "inference_metadata": inference_meta,
        "target_schema": target,
        "feature_names": feature_names,
        "feature_schema_hash": schema_hash,
        "input_size": input_size,
        "trained_at": trained_at,
        "trained_timestamp": trained_timestamp,
        "classification_threshold": threshold,
    }


def _approval_signature_payload(
    checkpoint: Mapping[str, Any],
    approval: Mapping[str, Any],
    identity_data: Mapping[str, str],
    contract: Mapping[str, Any],
    state_hash: str,
) -> Dict[str, Any]:
    unsigned_approval = {str(k): v for k, v in approval.items() if str(k) != "signature"}
    return {
        "approval": unsigned_approval,
        "model": {
            "identity": dict(identity_data),
            "market": str(checkpoint.get("market", "")).upper(),
            "model_version_id": str(checkpoint.get("model_version_id", "")),
            "model_state_sha256": state_hash,
            "feature_names": list(contract["feature_names"]),
            "feature_schema_hash": str(contract["feature_schema_hash"]),
            "model_target_schema": str(contract["target_schema"]),
            "input_size": int(contract["input_size"]),
            "model_class_module": str(checkpoint.get("model_class_module", "")),
            "model_class_name": str(checkpoint.get("model_class_name", "")),
            "hidden_sizes": list(checkpoint.get("hidden_sizes", ())),
            "trained_at": str(contract["trained_at"]),
            "classification_threshold": float(contract["classification_threshold"]),
            "scaler_sha256": _canonical_hash(checkpoint.get("scaler", {})),
            "inference_metadata": dict(checkpoint.get("inference_metadata", {})) if isinstance(checkpoint.get("inference_metadata"), Mapping) else {},
            "model_architecture": dict(checkpoint.get("model_architecture", {})) if isinstance(checkpoint.get("model_architecture"), Mapping) else {},
            "timeframes": list(checkpoint.get("timeframes", ())),
            "lookback_by_timeframe": dict(checkpoint.get("lookback_by_timeframe", {})) if isinstance(checkpoint.get("lookback_by_timeframe"), Mapping) else {},
            "format_version": checkpoint.get("format_version"),
        },
    }


def _verify_embedded_approval(
    checkpoint: Mapping[str, Any],
    identity_data: Mapping[str, str],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify a signed held-out-test approval certificate stored inside .pth."""
    approval = checkpoint.get("approval")
    if not isinstance(approval, Mapping):
        raise InferenceModelError(
            "The .pth has no embedded approval certificate. The model must be re-approved and packaged before production inference."
        )
    if int(approval.get("schema_version", 0)) != 1 or str(approval.get("status", "")).upper() != "APPROVED":
        raise InferenceModelError("The embedded model approval certificate is invalid or not approved.")

    model_id = str(checkpoint.get("model_version_id", "")).strip()
    if not model_id or str(approval.get("model_version_id", "")) != model_id:
        raise InferenceModelError("Approval certificate references another model version.")

    evaluation = approval.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise InferenceModelError("Approval certificate has no evaluation evidence.")
    if evaluation.get("passed") is not True or str(evaluation.get("evaluated_split", "")).lower() != "test":
        raise InferenceModelError("Model approval requires a passing independent held-out test evaluation.")
    if str(evaluation.get("model_target_schema", "")).upper() not in {MODEL_TARGET_SCHEMA, "TP_BEFORE_SL"}:
        raise InferenceModelError("Approval evaluation has an incompatible target schema.")
    if not str(evaluation.get("evaluation_id", "")).strip():
        raise InferenceModelError("Approval certificate has no evaluation ID.")
    report_hash = str(evaluation.get("report_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
        raise InferenceModelError("Approval evaluation SHA-256 is missing or invalid.")
    if not isinstance(evaluation.get("metrics"), Mapping) or not isinstance(evaluation.get("policy"), Mapping):
        raise InferenceModelError("Approval certificate is missing evaluation metrics or policy.")

    approved_at = approval.get("approved_at") or approval.get("approval_time")
    approval_timestamp = _parse_timestamp(approved_at)
    if approval_timestamp is None or approval_timestamp < int(contract["trained_timestamp"]):
        raise InferenceModelError("Approval timestamp is invalid or predates model training.")

    actual_state_hash = _model_state_sha256(checkpoint)
    if str(approval.get("model_state_sha256", "")).lower() != actual_state_hash:
        raise InferenceModelError("Model weights do not match the embedded approval certificate.")

    key = os.environ.get("SPECTLINE_MODEL_APPROVAL_KEY", "")
    if len(key.encode("utf-8")) < 32:
        raise InferenceDependencyError("SPECTLINE_MODEL_APPROVAL_KEY must contain at least 32 bytes.")
    signature = str(approval.get("signature", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise InferenceModelError("Embedded approval signature is missing or malformed.")

    signed = _approval_signature_payload(checkpoint, approval, identity_data, contract, actual_state_hash)
    message = _canonical_hash(signed).encode("ascii")
    expected = hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise InferenceModelError("Embedded approval signature verification failed.")

    return {
        "approval_time": approved_at,
        "evaluation_version_id": str(evaluation["evaluation_id"]),
        "evaluation_sha256": report_hash,
        "evaluated_split": "test",
        "approval_signature_verified": True,
        "model_state_sha256": actual_state_hash,
    }


def _model_file_candidates(model_root: Path, market: str) -> list[Path]:
    """List only final market.pth / market_N.pth files, excluding RUNTIME."""
    stem = str(market).strip().lower()
    result: list[Path] = []
    for path in model_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pth":
            continue
        candidate_stem = path.stem.lower()
        if candidate_stem == stem:
            result.append(path)
        elif candidate_stem.startswith(stem + "_"):
            suffix = candidate_stem[len(stem) + 1:]
            if suffix.isdigit() and int(suffix) >= 1:
                result.append(path)
    return sorted(result, key=lambda item: item.name.lower())


def load_approved_model(
    market: str,
    *,
    identity: Any,
    model_version_id: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the selected approved .pth directly, without VERSIONS or training RUNTIME."""
    _required_modules()
    resolved = _resolve_identity(market, identity=identity)
    identity_data = _identity_payload(resolved)
    normalized_market = identity_data["market"]

    directory_getter = getattr(memory_engine, "model_dir", None)
    checkpoint_loader = getattr(memory_engine, "_safe_torch_load", None)
    model_factory = getattr(memory_engine, "_instantiate_model", None)
    if not all(callable(fn) for fn in (directory_getter, checkpoint_loader, model_factory)):
        raise InferenceDependencyError("engine.memory is missing its direct .pth loading helpers.")
    try:
        model_root = Path(directory_getter(normalized_market, identity=resolved)).expanduser().resolve()
    except Exception as error:
        raise InferenceDependencyError("Could not resolve MODELS/<MARKET>.") from error
    if not model_root.is_dir():
        raise InferenceModelError(f"Model directory does not exist for {normalized_market}.")

    candidates = _model_file_candidates(model_root, normalized_market)
    if not candidates:
        raise InferenceModelError(f"No final .pth models were found for {normalized_market}.")

    requested_id = str(model_version_id).strip() if model_version_id is not None else None
    approved: list[Tuple[int, Path, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    matched_id = False

    for path in candidates:
        resolved_path = path.resolve()
        if resolved_path.parent != model_root:
            raise InferenceModelError("Model file resolves outside its market directory.")
        try:
            checkpoint = checkpoint_loader(resolved_path)
        except Exception as error:
            raise InferenceModelError(f"Could not safely read {path.name}.") from error
        if not isinstance(checkpoint, Mapping):
            raise InferenceModelError(f"Model checkpoint {path.name} is malformed.")
        checkpoint = dict(checkpoint)
        try:
            candidate_identity = _identity_payload(checkpoint.get("identity"))
        except Exception as error:
            raise InferenceModelError(f"Model checkpoint {path.name} has an invalid identity.") from error
        if candidate_identity != identity_data:
            continue
        if str(checkpoint.get("market", "")).upper() != normalized_market:
            raise InferenceModelError(f"Model checkpoint {path.name} has a conflicting market.")
        candidate_id = str(checkpoint.get("model_version_id", "")).strip()
        if not candidate_id:
            raise InferenceModelError(f"Model checkpoint {path.name} has no model_version_id.")
        if requested_id is not None and candidate_id != requested_id:
            continue
        if requested_id is not None:
            matched_id = True

        certificate = checkpoint.get("approval")
        status = str(certificate.get("status", "")).upper() if isinstance(certificate, Mapping) else ""
        if status != "APPROVED":
            if requested_id is not None:
                raise InferenceModelError(f"Model {requested_id} is not approved inside its .pth file.")
            continue

        try:
            model = model_factory(checkpoint)
            scaler = scaling_engine.scaler_from_checkpoint(checkpoint)
        except Exception as error:
            raise InferenceModelError(f"Could not restore model or embedded scaler from {path.name}.") from error
        contract = _checkpoint_contract(checkpoint, model, scaler)
        approval_info = _verify_embedded_approval(checkpoint, identity_data, contract)
        approved_at = _parse_timestamp(approval_info["approval_time"])
        if approved_at is None:
            raise InferenceModelError("Approved model has no valid approval timestamp.")
        approved.append((approved_at, resolved_path, checkpoint, contract, approval_info))

    if requested_id is not None and not matched_id:
        raise InferenceModelError(f"Model version {requested_id!r} was not found for this TradingBot.")
    if not approved:
        raise InferenceModelError("No approved, signed .pth model is available for this TradingBot.")
    if requested_id is not None and len(approved) != 1:
        raise InferenceModelError("Requested model_version_id is not unique in MODELS/<MARKET>.")

    approved.sort(key=lambda item: (item[0], item[1].name.lower()), reverse=True)
    _, selected_path, checkpoint, contract, approval_info = approved[0]
    try:
        model = model_factory(checkpoint)
        scaler = scaling_engine.scaler_from_checkpoint(checkpoint)
        model.eval()
    except Exception as error:
        raise InferenceModelError("Approved .pth model could not be restored for inference.") from error

    summary = contract["summary"]
    inference_meta = contract["inference_metadata"]
    trained_at = contract["trained_at"]
    metadata = {
        "identity": identity_data,
        "market": normalized_market,
        "model_version_id": str(checkpoint["model_version_id"]),
        # IDs below are provenance labels only; no external artifacts are loaded.
        "dataset_version_id": checkpoint.get("dataset_version_id") or summary.get("dataset_version_id"),
        "scaler_version_id": checkpoint.get("scaler_version_id") or summary.get("scaler_version_id"),
        "model_target_schema": contract["target_schema"],
        "feature_names": contract["feature_names"],
        "feature_schema_hash": contract["feature_schema_hash"],
        "trained_at": trained_at,
        "model_status": "APPROVED",
        "model_created_at": contract["trained_timestamp"],
        "model_path": str(selected_path),
        "model_file_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "approval": approval_info,
        "calibration_status": str(inference_meta.get("calibration_status") or summary.get("calibration_status", "NOT_CALIBRATED")).upper(),
        "classification_threshold": contract["classification_threshold"],
    }
    return model, scaler, metadata


# ============================================================================
# POINT-IN-TIME FEATURE PREPARATION
# ============================================================================


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
