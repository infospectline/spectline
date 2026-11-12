"""Create point-in-time-safe TP-before-SL labels for Spectline trade records."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .settings import OUTCOME_SL_FIRST, OUTCOME_TP_FIRST
except ImportError:
    try:
        from engine.settings import OUTCOME_SL_FIRST, OUTCOME_TP_FIRST
    except ImportError:
        OUTCOME_SL_FIRST = 0
        OUTCOME_TP_FIRST = 1


# ============================================================================
# CONFIGURATION
# ============================================================================

LABEL_SCHEMA_VERSION = 2
LABEL_TARGET_NAME = "TP_BEFORE_SL"
LABEL_VALUE_FIELD = "actual_outcome"
LABEL_AVAILABLE_TIME_FIELD = "end_time"
ENTRY_TIME_FIELDS = ("entry_time",)
END_TIME_FIELD = "end_time"
PREDICTION_TIME_FIELDS = ("predicted_at", "prediction_time", "prediction_timestamp")
PREDICTION_PROBABILITY_FIELDS = (
    "predicted_probability_at_entry",
    "tp_before_sl_probability",
)

IDENTITY_FIELDS = ("user_id", "market", "strategy_id", "tradingbot_id")
_PATTERN_ID_PATTERN = re.compile(r"^\d{4}$")

OUTCOME_NAMES = {
    OUTCOME_SL_FIRST: "SL",
    OUTCOME_TP_FIRST: "TP",
}

OUTCOME_ALIASES = {
    "0": "SL",
    "SL": "SL",
    "SL_FIRST": "SL",
    "SL_BEFORE_TP": "SL",
    "STOP_LOSS_FIRST": "SL",
    "LOSS": "SL",
    "LOSING": "SL",
    "1": "TP",
    "TP": "TP",
    "TP_FIRST": "TP",
    "TP_BEFORE_SL": "TP",
    "TAKE_PROFIT_FIRST": "TP",
    "WIN": "TP",
    "WON": "TP",
    "AMBIGUOUS": "AMBIGUOUS",
    "UNRESOLVED": "AMBIGUOUS",
    "UNKNOWN_ORDER": "AMBIGUOUS",
    "SL_TP_ORDER_UNRESOLVED": "AMBIGUOUS",
}

# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class LabelError(RuntimeError):
    """Base exception for trade-label processing errors."""


class LabelValidationError(LabelError, ValueError):
    """Raised when a record violates the supervised-label contract."""


@dataclass(frozen=True)
class LabelBuildResult:
    id: Optional[int]
    status: str
    label: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def eligible(self) -> bool:
        return self.label is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "label": dict(self.label) if self.label is not None else None,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }

# ============================================================================
# VALUE AND TIME NORMALIZATION
# ============================================================================

def _first_value(record: Mapping[str, Any], field_names: Sequence[str]) -> Any:
    for field_name in field_names:
        value = record.get(field_name)
        if value is not None and value != "":
            return value
    return None


def parse_timestamp(value: Any) -> Optional[int]:
    """Convert a supported timestamp representation to UTC epoch seconds."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(numeric)

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError):
            numeric = None
        if numeric is not None:
            if not math.isfinite(numeric):
                return None
            return int(numeric)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def normalize_probability(value: Any) -> Optional[float]:
    """Normalize a probability expressed as a fraction or percentage."""
    if value is None or isinstance(value, bool):
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(probability):
        return None
    if 1.0 < probability <= 100.0:
        probability /= 100.0
    if not 0.0 <= probability <= 1.0:
        return None
    return float(probability)


def normalize_outcome(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if numeric == OUTCOME_SL_FIRST:
            return "SL"
        if numeric == OUTCOME_TP_FIRST:
            return "TP"
        return None

    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    return OUTCOME_ALIASES.get(text)


def normalize_status(record: Mapping[str, Any]) -> str:
    raw_outcome = record.get("actual_outcome")
    actual_outcome = normalize_outcome(raw_outcome)
    end_time = parse_timestamp(record.get(END_TIME_FIELD))

    if actual_outcome == "AMBIGUOUS":
        return "AMBIGUOUS"

    if actual_outcome in ("TP", "SL"):
        return "CLOSED_LABELLED" if end_time is not None else "CLOSED_UNLABELLED"

    if raw_outcome is not None:
        return "INVALID"

    return "OPEN" if end_time is None else "CLOSED_UNLABELLED"

def _identity_values(
    record: Mapping[str, Any],
    inherited_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    nested = record.get("identity")
    nested_identity = nested if isinstance(nested, Mapping) else {}
    inherited = inherited_identity if isinstance(inherited_identity, Mapping) else {}
    result: Dict[str, Optional[str]] = {}

    for field_name in IDENTITY_FIELDS:
        direct_value = record.get(field_name)
        nested_value = nested_identity.get(field_name)
        inherited_value = inherited.get(field_name)

        supplied_values = [
            str(value).strip().upper() if field_name == "market" else str(value).strip()
            for value in (direct_value, nested_value, inherited_value)
            if value is not None and str(value).strip() != ""
        ]
        if supplied_values and any(value != supplied_values[0] for value in supplied_values[1:]):
            raise LabelValidationError(
                f"Trade {field_name} does not match the supplied customer identity."
            )

        value = direct_value
        if value is None:
            value = nested_value
        if value is None:
            value = inherited_value
        if value is None or str(value).strip() == "":
            result[field_name] = None
        elif field_name == "market":
            result[field_name] = str(value).strip().upper()
        else:
            result[field_name] = str(value).strip()

    present = [value is not None for value in result.values()]
    if any(present) and not all(present):
        missing = [name for name, value in result.items() if value is None]
        raise LabelValidationError(
            "Trade identity is incomplete; missing " + ", ".join(missing) + "."
        )
    return result


def _setup_identifier(record: Mapping[str, Any]) -> Optional[int]:
    value = record.get("id")

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        identifier = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        identifier = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        identifier = int(value.strip())
    else:
        return None

    return identifier if identifier > 0 else None

def _pattern_ids(record: Mapping[str, Any]) -> List[str]:
    supplied: List[Any] = []
    raw_ids = record.get("linked_pattern_ids", [])
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
        supplied.extend(raw_ids)
    for field_name in ("pattern_id", "associated_pattern_id"):
        if record.get(field_name) is not None:
            supplied.append(record[field_name])

    normalized: List[str] = []
    for raw_value in supplied:
        value = str(raw_value).strip()
        if value.isdigit() and len(value) < 4:
            value = value.zfill(4)
        if _PATTERN_ID_PATTERN.fullmatch(value) and value not in normalized:
            normalized.append(value)
    return normalized

# ============================================================================
# TRADE CLASSIFICATION
# ============================================================================

def build_trade_label(
    record: Mapping[str, Any],
    *,
    identity: Optional[Mapping[str, Any]] = None,
    label_source: Optional[str] = None,
) -> LabelBuildResult:
    classification = classify_trade_record(record)

    if classification.status != "CLOSED_LABELLED":
        return classification

    setup_id = classification.id
    actual_outcome = normalize_outcome(record.get("actual_outcome"))

    if setup_id is None or actual_outcome not in ("TP", "SL"):
        return LabelBuildResult(
            id=setup_id,
            status="CLOSED_UNLABELLED",
            reason="No valid setup ID or binary outcome is available.",
        )

    entry_time = parse_timestamp(record.get("entry_time"))
    end_time = parse_timestamp(record.get(END_TIME_FIELD))

    if entry_time is None or end_time is None:
        return LabelBuildResult(
            id=setup_id,
            status="CLOSED_UNLABELLED",
            reason="A valid entry_time and end_time are required.",
        )

    if end_time < entry_time:
        return LabelBuildResult(
            id=setup_id,
            status="INVALID",
            reason="end_time cannot precede entry_time.",
        )

    try:
        identity_values = _identity_values(record, identity)
    except LabelValidationError as error:
        return LabelBuildResult(
            id=setup_id,
            status="INVALID",
            reason=str(error),
        )

    source_value = label_source or record.get("label_source") or "SETUPS_JSON"
    source_text = str(source_value).strip().upper() or "SETUPS_JSON"

    numeric_label = (
        OUTCOME_TP_FIRST if actual_outcome == "TP" else OUTCOME_SL_FIRST
    )

    label: Dict[str, Any] = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "id": setup_id,
        "label_target": LABEL_TARGET_NAME,
        **identity_values,
        "entry_time": entry_time,
        "end_time": end_time,
        "actual_outcome": actual_outcome,
        "label": numeric_label,
        "outcome_name": actual_outcome,
        "label_source": source_text,
        "linked_pattern_ids": _pattern_ids(record),
    }

    for field_name in ("timeframe", "entry_timeframe", "model_timeframe", "strategy_name"):
        value = record.get(field_name)
        if value is not None:
            label[field_name] = (
                str(value).strip().upper()
                if "timeframe" in field_name
                else str(value).strip()
            )

    for field_name in ("model_version", "catalog_version", "dataset_version"):
        value = record.get(field_name)
        if value is not None:
            label[field_name] = str(value).strip()

    prediction_probability = normalize_probability(
        _first_value(record, PREDICTION_PROBABILITY_FIELDS)
    )
    prediction_time = parse_timestamp(
        _first_value(record, PREDICTION_TIME_FIELDS)
    )

    prediction_warnings: List[str] = []

    if prediction_probability is not None and prediction_time is not None:
        if prediction_time <= entry_time:
            label["prediction_context"] = {
                "predicted_probability_at_entry": prediction_probability,
                "predicted_at": prediction_time,
                "target": LABEL_TARGET_NAME,
            }
        else:
            prediction_warnings.append(
                "Prediction timestamp is later than entry_time and was excluded."
            )
    elif any(record.get(name) is not None for name in PREDICTION_PROBABILITY_FIELDS):
        prediction_warnings.append(
            "Prediction metadata was incomplete or invalid and was excluded."
        )

    for field_name in ("setup_quality_probability", "model_probabilities"):
        if record.get(field_name) is not None:
            label.setdefault("non_target_model_metadata", {})[field_name] = (
                record[field_name]
            )

    validation = validate_label_record(label)

    if validation.get("errors"):
        return LabelBuildResult(
            id=setup_id,
            status="INVALID",
            reason="; ".join(validation["errors"]),
            warnings=tuple(prediction_warnings),
        )

    return LabelBuildResult(
        id=setup_id,
        status="LABELLED",
        label=label,
        warnings=tuple(prediction_warnings),
    )

# ============================================================================
# LABEL CONSTRUCTION AND VALIDATION
# ============================================================================

def build_trade_label(
    record: Mapping[str, Any],
    *,
    identity: Optional[Mapping[str, Any]] = None,
    label_source: Optional[str] = None,
) -> LabelBuildResult:
    """Build one binary label without assigning labels to open or ambiguous trades."""
    classification = classify_trade_record(record)
    if not classification.eligible and classification.status != "CLOSED_LABELLED":
        return classification

    trade_id = classification.trade_id
    if trade_id is None:
        return classification

    actual_outcome = normalize_outcome(record.get("actual_outcome"))
    if actual_outcome is None:
        actual_outcome = normalize_outcome(record.get("result"))
    if actual_outcome not in (OUTCOME_SL_FIRST, OUTCOME_TP_FIRST):
        return LabelBuildResult(
            trade_id=trade_id,
            status="CLOSED_UNLABELLED",
            reason="No valid binary outcome is available.",
        )

    entry_time = parse_timestamp(_first_value(record, ENTRY_TIME_FIELDS))
    exit_time = parse_timestamp(_first_value(record, EXIT_TIME_FIELDS))
    known_at = parse_timestamp(_first_value(record, OUTCOME_KNOWN_TIME_FIELDS))
    if entry_time is None or exit_time is None or known_at is None:
        return LabelBuildResult(
            trade_id=trade_id,
            status="CLOSED_UNLABELLED",
            reason="A valid entry, exit and outcome-known timestamp is required.",
        )

    try:
        identity_values = _identity_values(record, identity)
    except LabelValidationError as error:
        return LabelBuildResult(
            trade_id=trade_id,
            status="INVALID",
            reason=str(error),
        )

    source_value = label_source or record.get("label_source") or "TRADE_OUTCOME"
    source_text = str(source_value).strip().upper()
    if not source_text:
        source_text = "TRADE_OUTCOME"

    label: Dict[str, Any] = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_id": _stable_label_id(identity_values, trade_id, entry_time, exit_time),
        "label_target": LABEL_TARGET_NAME,
        "trade_id": trade_id,
        "setup_id": str(record.get("setup_id") or trade_id),
        **identity_values,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "outcome_known_at": known_at,
        "actual_outcome": int(actual_outcome),
        "label": int(actual_outcome),
        "outcome_name": OUTCOME_NAMES[int(actual_outcome)],
        "label_source": source_text,
        "linked_pattern_ids": _pattern_ids(record),
    }

    for field_name in ("timeframe", "entry_timeframe", "model_timeframe", "strategy_name"):
        value = record.get(field_name)
        if value is not None:
            label[field_name] = str(value).strip().upper() if "timeframe" in field_name else str(value).strip()

    for field_name in ("model_version", "catalog_version", "dataset_version"):
        value = record.get(field_name)
        if value is not None:
            label[field_name] = str(value).strip()

    prediction_probability = normalize_probability(
        _first_value(record, PREDICTION_PROBABILITY_FIELDS)
    )
    prediction_time = parse_timestamp(_first_value(record, PREDICTION_TIME_FIELDS))
    prediction_warnings: List[str] = []
    if prediction_probability is not None and prediction_time is not None:
        if prediction_time <= entry_time:
            label["prediction_context"] = {
                "predicted_probability_at_entry": prediction_probability,
                "predicted_at": prediction_time,
                "target": LABEL_TARGET_NAME,
            }
        else:
            prediction_warnings.append(
                "Prediction timestamp is later than entry_time and was excluded."
            )
    elif any(record.get(name) is not None for name in PREDICTION_PROBABILITY_FIELDS):
        prediction_warnings.append(
            "Prediction metadata was incomplete or invalid and was excluded."
        )

    for field_name in ("setup_quality_probability", "model_probabilities"):
        if record.get(field_name) is not None:
            label.setdefault("non_target_model_metadata", {})[field_name] = record[field_name]

    validation = validate_label_record(label)
    if validation.get("errors"):
        return LabelBuildResult(
            trade_id=trade_id,
            status="INVALID",
            reason="; ".join(validation["errors"]),
            warnings=tuple(prediction_warnings),
        )

    return LabelBuildResult(
        trade_id=trade_id,
        status="LABELLED",
        label=label,
        warnings=tuple(prediction_warnings),
    )


def validate_label_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(record, Mapping):
        return {
            "valid": False,
            "errors": ["Label record must be a mapping."],
            "warnings": [],
        }

    identifier = record.get("id")
    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier <= 0
    ):
        errors.append("id must be a positive integer.")

    actual_outcome = record.get("actual_outcome")
    if actual_outcome not in ("TP", "SL"):
        errors.append("actual_outcome must be TP or SL.")

    label_value = record.get("label")
    if (
        isinstance(label_value, bool)
        or not isinstance(label_value, int)
        or label_value not in (OUTCOME_SL_FIRST, OUTCOME_TP_FIRST)
    ):
        errors.append("label must be integer 0 or 1.")
    elif actual_outcome in ("TP", "SL"):
        expected_label = (
            OUTCOME_TP_FIRST if actual_outcome == "TP" else OUTCOME_SL_FIRST
        )
        if label_value != expected_label:
            errors.append("label does not match actual_outcome.")

    required_strings = ("label_target", "label_source")
    for field_name in required_strings:
        if not isinstance(record.get(field_name), str) or not record[field_name].strip():
            errors.append(f"{field_name} must be a non-empty string.")

    entry_time = parse_timestamp(record.get("entry_time"))
    end_time = parse_timestamp(record.get(END_TIME_FIELD))

    if entry_time is None:
        errors.append("entry_time is missing or invalid.")

    if end_time is None:
        errors.append("end_time is missing or invalid.")

    if entry_time is not None and end_time is not None and end_time < entry_time:
        errors.append("end_time cannot precede entry_time.")

    identity = {name: record.get(name) for name in IDENTITY_FIELDS}
    identity_presence = [
        value is not None and str(value).strip()
        for value in identity.values()
    ]

    if any(identity_presence) and not all(identity_presence):
        errors.append("Customer identity must contain all four identity fields or none.")

    if record.get("linked_pattern_ids") is not None:
        pattern_ids = record.get("linked_pattern_ids")

        if not isinstance(pattern_ids, Sequence) or isinstance(pattern_ids, (str, bytes)):
            errors.append("linked_pattern_ids must be a sequence.")
        else:
            for pattern_id in pattern_ids:
                if not isinstance(pattern_id, str) or not _PATTERN_ID_PATTERN.fullmatch(pattern_id):
                    errors.append("Every linked pattern ID must contain exactly four digits.")
                    break

    prediction_context = record.get("prediction_context")

    if prediction_context is not None:
        if not isinstance(prediction_context, Mapping):
            errors.append("prediction_context must be a mapping.")
        else:
            probability = normalize_probability(
                prediction_context.get("predicted_probability_at_entry")
            )
            predicted_at = parse_timestamp(
                prediction_context.get("predicted_at")
            )

            if probability is None:
                warnings.append("prediction_context probability is invalid.")

            if predicted_at is None:
                warnings.append("prediction_context predicted_at is invalid.")
            elif entry_time is not None and predicted_at > entry_time:
                warnings.append("prediction_context was not available at entry_time.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# COLLECTION AND DATA-BUNDLE HELPERS
# ============================================================================

def _record_key(
    record: Mapping[str, Any],
    inherited_identity: Optional[Mapping[str, Any]],
) -> Tuple[str, str]:
    setup_id = _setup_identifier(record)
    identifier_key = str(setup_id) if setup_id is not None else ""

    try:
        identity_values = _identity_values(record, inherited_identity)
    except LabelValidationError:
        identity_values = {name: None for name in IDENTITY_FIELDS}

    identity_key = "|".join(
        str(identity_values[name] or "") for name in IDENTITY_FIELDS
    )

    return identity_key, identifier_key

def _merge_prediction_metadata(
    record: Mapping[str, Any],
    predictions_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    inherited_identity: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(record)
    key = _record_key(record, inherited_identity)
    prediction = predictions_by_key.get(key)
    if prediction is None:
        return merged

    if merged.get("predicted_probability_at_entry") is None:
        merged["predicted_probability_at_entry"] = prediction.get(
            "predicted_probability_at_entry"
        )
    if merged.get("predicted_at") is None:
        merged["predicted_at"] = prediction.get("predicted_at")
    return merged


def build_trade_labels(
    trades: Iterable[Mapping[str, Any]],
    *,
    identity: Optional[Mapping[str, Any]] = None,
    label_source: Optional[str] = None,
    include_report: bool = False,
) -> Any:
    labels: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicate_count = 0
    conflict_count = 0
    input_count = 0

    for index, trade in enumerate(trades):
        input_count += 1

        if not isinstance(trade, Mapping):
            excluded.append(
                {
                    "index": index,
                    "status": "INVALID",
                    "reason": "Trade record must be a mapping.",
                }
            )
            continue

        result = build_trade_label(
            trade,
            identity=identity,
            label_source=label_source,
        )

        if result.label is None:
            excluded.append(
                {
                    "id": result.id,
                    "status": result.status,
                    "reason": result.reason,
                }
            )
            continue

        key = (
            "|".join(
                str(result.label.get(name) or "")
                for name in IDENTITY_FIELDS
            ),
            str(result.label["id"]),
        )

        previous = seen.get(key)

        if previous is not None:
            duplicate_count += 1

            if (
                previous.get("actual_outcome") != result.label.get("actual_outcome")
                or previous.get("entry_time") != result.label.get("entry_time")
                or previous.get("end_time") != result.label.get("end_time")
            ):
                conflict_count += 1
                excluded.append(
                    {
                        "id": result.id,
                        "status": "CONFLICT",
                        "reason": "A duplicate setup id contains conflicting label data.",
                    }
                )
            continue

        seen[key] = result.label
        labels.append(result.label)

    labels.sort(
        key=lambda item: (
            int(item["end_time"]),
            int(item["entry_time"]),
            int(item["id"]),
        )
    )

    if not include_report:
        return labels

    report = {
        "input_trade_count": input_count,
        "label_count": len(labels),
        "excluded_count": len(excluded),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_count,
        "excluded": excluded,
        "target": LABEL_TARGET_NAME,
        "label_mapping": {
            "SL": OUTCOME_SL_FIRST,
            "TP": OUTCOME_TP_FIRST,
        },
    }

    return {"labels": labels, "report": report}

def build_labels_from_bundle(
    data_bundle: Mapping[str, Any],
    *,
    include_report: bool = False,
) -> Any:
    """Build supervised labels from an engine.data bundle."""
    if not isinstance(data_bundle, Mapping):
        raise TypeError("data_bundle must be a mapping.")

    identity = data_bundle.get("identity")
    if identity is not None and not isinstance(identity, Mapping):
        raise LabelValidationError("data_bundle.identity must be a mapping.")

    predictions = data_bundle.get("setup_prediction_records", [])
    predictions_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
        for prediction in predictions:
            if isinstance(prediction, Mapping):
                predictions_by_key[_record_key(prediction, identity)] = prediction

    candidates: List[Dict[str, Any]] = []
    seen_sources = set()

    for source_name in ("setup_results",):
        source_records = data_bundle.get(source_name, [])
        if not isinstance(source_records, Sequence) or isinstance(source_records, (str, bytes)):
            continue
        for record in source_records:
            if not isinstance(record, Mapping):
                continue
            merged = _merge_prediction_metadata(record, predictions_by_key, identity)
            merged.setdefault("label_source", source_name.upper())
            key = _record_key(merged, identity)
            fingerprint = (
                key,
                parse_timestamp(merged.get("entry_time")),
                parse_timestamp(merged.get(END_TIME_FIELD)),
                normalize_outcome(merged.get("actual_outcome")),
            )
            if fingerprint in seen_sources:
                continue
            seen_sources.add(fingerprint)
            candidates.append(merged)

    result = build_trade_labels(
        candidates,
        identity=identity,
        include_report=True,
    )
    report = dict(result["report"])
    report["source_record_count"] = len(candidates)
    report["bundle_status"] = data_bundle.get("status")
    if include_report:
        return {"labels": result["labels"], "report": report}
    return result["labels"]


def validate_label_collection(
    labels: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    count = 0

    for index, label in enumerate(labels):
        count += 1
        validation = validate_label_record(label)

        if not validation["valid"]:
            errors.append(
                {
                    "index": index,
                    "id": label.get("id") if isinstance(label, Mapping) else None,
                    "errors": validation["errors"],
                }
            )
            continue

        key = _record_key(label, None)
        previous = seen.get(key)

        if previous is not None and previous != dict(label):
            errors.append(
                {
                    "index": index,
                    "id": label.get("id"),
                    "errors": ["Duplicate setup id contains different data."],
                }
            )
        else:
            seen[key] = dict(label)

    return {
        "valid": not errors,
        "record_count": count,
        "unique_label_count": len(seen),
        "error_count": len(errors),
        "errors": errors,
    }

# ============================================================================
# TEMPORAL FILTERING AND TRAINING TARGETS
# ============================================================================

def labels_available_as_of(
    labels: Iterable[Mapping[str, Any]],
    cutoff_time: Any,
) -> List[Dict[str, Any]]:
    """Return only labels whose outcomes were already known at the cutoff."""
    cutoff = parse_timestamp(cutoff_time)
    if cutoff is None:
        raise ValueError("cutoff_time must be a valid timestamp.")

    available: List[Dict[str, Any]] = []
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        known_at = parse_timestamp(label.get(LABEL_AVAILABLE_TIME_FIELD))
        if known_at is not None and known_at <= cutoff:
            available.append(dict(label))
    return available


def labels_to_targets(
    labels: Iterable[Mapping[str, Any]],
) -> List[int]:
    """Extract validated binary targets without adding metadata to model inputs."""
    targets: List[int] = []
    for label in labels:
        validation = validate_label_record(label)
        if not validation["valid"]:
            raise LabelValidationError(
                "Cannot create targets from an invalid label: "
                + "; ".join(validation["errors"])
            )
        targets.append(int(label["label"]))
    return targets


def split_feature_records_and_labels(
    feature_records: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "id",
) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    normalized_labels: Dict[str, Dict[str, Any]] = {}

    for label in labels:
        validation = validate_label_record(label)

        if not validation["valid"]:
            raise LabelValidationError(
                "Invalid label record: " + "; ".join(validation["errors"])
            )

        normalized_labels[str(label["id"])] = dict(label)

    aligned_features: List[Dict[str, Any]] = []
    targets: List[int] = []
    aligned_metadata: List[Dict[str, Any]] = []

    for feature_record in feature_records:
        if not isinstance(feature_record, Mapping):
            continue

        identifier = feature_record.get(id_field)

        if identifier is None:
            continue

        label = normalized_labels.get(str(identifier))

        if label is None:
            continue

        feature_copy = dict(feature_record)

        forbidden_fields = (
            id_field,
            "id",
            "trade_id",
            "setup_id",
            "trade_record_id",
            "runtime_trade_id",
            "user_id",
            "market",
            "strategy_id",
            "tradingbot_id",
            "actual_outcome",
            "label",
            "label_id",
            "label_target",
            "label_source",
            "label_schema_version",
            "outcome_name",
            "outcome_known_at",
            "outcome_status",
            "end_time",
            "exit_time",
            "exit_price",
            "exit",
            "result",
            "training_eligible",
            "predicted_probability_at_entry",
            "predicted_at",
            "prediction_context",
        )

        for forbidden_field in forbidden_fields:
            feature_copy.pop(forbidden_field, None)

        aligned_features.append(feature_copy)
        targets.append(int(label["label"]))

        aligned_metadata.append(
            {
                "id": label["id"],
                "entry_time": label["entry_time"],
                "end_time": label["end_time"],
                "actual_outcome": label["actual_outcome"],
                "label_target": label["label_target"],
            }
        )

    return aligned_features, targets, aligned_metadata

# ============================================================================
# PUBLIC COMPATIBILITY ALIASES
# ============================================================================

create_trade_label = build_trade_label
create_trade_labels = build_trade_labels
build_labels = build_trade_labels
filter_labels_as_of = labels_available_as_of
