"""Point-in-time discovery and recording of historical market situations."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from client import normalize_market
    from engine.catalog import (
        PatternCatalog,
        PatternDefinition,
        load_catalog,
        save_catalog,
    )
    from engine.features import (
        PIVOT_FEATURE_FIELDS,
        detect_pivots,
        pattern_geometry,
    )
    from engine.settings import (
        MARKET_EVENT_TYPES,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )
    from engine.storage import (
        append_market_event,
        append_pattern_occurrence,
    )
except ImportError:
    from ..client import normalize_market
    from .catalog import (
        PatternCatalog,
        PatternDefinition,
        load_catalog,
        save_catalog,
    )
    from .features import (
        PIVOT_FEATURE_FIELDS,
        detect_pivots,
        pattern_geometry,
    )
    from .settings import (
        MARKET_EVENT_TYPES,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_SECONDS,
    )
    from .storage import (
        append_market_event,
        append_pattern_occurrence,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

SITUATION_DETECTOR_VERSION = "1.0"
SITUATION_SCHEMA_VERSION = 1
DEFAULT_LEFT_PIVOT_BARS = 3
DEFAULT_RIGHT_PIVOT_BARS = 3
DEFAULT_MINIMUM_PIVOTS = 3


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class SituationError(RuntimeError):
    """Base exception for historical situation processing."""


class SituationValidationError(SituationError, ValueError):
    """Raised when a scan request or situation record is invalid."""


class SituationCatalogError(SituationError):
    """Raised when the supplied pattern catalog is incompatible with a scan."""


@dataclass(frozen=True)
class SituationScanConfig:
    """Explicit scan settings for each timeframe.

    ``window_bars_by_timeframe`` is required because the project specification
    has not yet fixed one universal formation window for every timeframe.
    Approximate catalog matching is disabled unless a similarity threshold is
    explicitly supplied.
    """

    window_bars_by_timeframe: Mapping[str, int]
    left_bars: int = DEFAULT_LEFT_PIVOT_BARS
    right_bars: int = DEFAULT_RIGHT_PIVOT_BARS
    min_pivots: int = DEFAULT_MINIMUM_PIVOTS
    similarity_threshold: Optional[float] = None
    persist_matched_occurrences: bool = True
    max_candidates_per_timeframe: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.window_bars_by_timeframe, Mapping):
            raise SituationValidationError(
                "window_bars_by_timeframe must be a timeframe-to-window mapping."
            )

        normalized: Dict[str, int] = {}
        for raw_timeframe, raw_window in self.window_bars_by_timeframe.items():
            timeframe = str(raw_timeframe).strip().upper()
            if timeframe not in TIMEFRAME_SECONDS:
                raise SituationValidationError(
                    f"Unknown timeframe in scan configuration: {timeframe!r}."
                )
            if timeframe not in SUPPORTED_TIMEFRAMES:
                raise SituationValidationError(
                    f"Timeframe {timeframe!r} is unavailable in the current runtime."
                )
            try:
                window = int(raw_window)
            except (TypeError, ValueError, OverflowError) as error:
                raise SituationValidationError(
                    f"Window size for {timeframe} must be an integer."
                ) from error
            if window <= 0:
                raise SituationValidationError(
                    f"Window size for {timeframe} must be positive."
                )
            normalized[timeframe] = window

        if not normalized:
            raise SituationValidationError(
                "At least one timeframe window must be configured."
            )

        if int(self.left_bars) < 1 or int(self.right_bars) < 1:
            raise SituationValidationError(
                "left_bars and right_bars must both be at least one."
            )
        if int(self.min_pivots) < 2:
            raise SituationValidationError("min_pivots must be at least two.")
        if self.similarity_threshold is not None:
            threshold = _finite_float(self.similarity_threshold)
            if threshold is None or not 0.0 <= threshold <= 1.0:
                raise SituationValidationError(
                    "similarity_threshold must be between zero and one."
                )
        if self.max_candidates_per_timeframe is not None:
            if int(self.max_candidates_per_timeframe) < 1:
                raise SituationValidationError(
                    "max_candidates_per_timeframe must be positive when supplied."
                )

        object.__setattr__(self, "window_bars_by_timeframe", normalized)
        object.__setattr__(self, "left_bars", int(self.left_bars))
        object.__setattr__(self, "right_bars", int(self.right_bars))
        object.__setattr__(self, "min_pivots", int(self.min_pivots))


@dataclass(frozen=True)
class SituationCandidate:
    """One numerically described formation candidate at its recognition time."""

    candidate_id: str
    market: str
    timeframe: str
    occurrence_start: int
    occurrence_end: int
    recognized_at: int
    window_start: int
    window_end: int
    pivot_count: int
    direction: str
    geometry: Mapping[str, Any]
    features: Mapping[str, float]
    latest_pivot: Mapping[str, Any]
    matched_pattern_id: Optional[str] = None
    similarity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "market": self.market,
            "timeframe": self.timeframe,
            "occurrence_start": self.occurrence_start,
            "occurrence_end": self.occurrence_end,
            "recognized_at": self.recognized_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "pivot_count": self.pivot_count,
            "direction": self.direction,
            "geometry": _json_safe(dict(self.geometry)),
            "features": _json_safe(dict(self.features)),
            "latest_pivot": _json_safe(dict(self.latest_pivot)),
            "matched_pattern_id": self.matched_pattern_id,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class SituationScanResult:
    """Summary of one timeframe scan."""

    market: str
    timeframe: str
    catalog_version: str
    candle_count: int
    confirmed_pivot_count: int
    candidate_count: int
    matched_count: int
    inserted_count: int
    duplicate_count: int
    unmatched_candidates: Tuple[SituationCandidate, ...] = field(default_factory=tuple)
    matched_candidates: Tuple[SituationCandidate, ...] = field(default_factory=tuple)

    def to_dict(self, include_candidates: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "market": self.market,
            "timeframe": self.timeframe,
            "catalog_version": self.catalog_version,
            "candle_count": self.candle_count,
            "confirmed_pivot_count": self.confirmed_pivot_count,
            "candidate_count": self.candidate_count,
            "matched_count": self.matched_count,
            "inserted_count": self.inserted_count,
            "duplicate_count": self.duplicate_count,
        }
        if include_candidates:
            result["unmatched_candidates"] = [
                item.to_dict() for item in self.unmatched_candidates
            ]
            result["matched_candidates"] = [
                item.to_dict() for item in self.matched_candidates
            ]
        return result


@dataclass(frozen=True)
class MarketSituationScanResult:
    """Combined result across the configured market timeframes."""

    market: str
    catalog_version: str
    timeframe_results: Mapping[str, SituationScanResult]

    def to_dict(self, include_candidates: bool = True) -> Dict[str, Any]:
        values = list(self.timeframe_results.values())
        return {
            "market": self.market,
            "catalog_version": self.catalog_version,
            "timeframes": {
                timeframe: result.to_dict(include_candidates=include_candidates)
                for timeframe, result in self.timeframe_results.items()
            },
            "totals": {
                "candle_count": sum(item.candle_count for item in values),
                "confirmed_pivot_count": sum(
                    item.confirmed_pivot_count for item in values
                ),
                "candidate_count": sum(item.candidate_count for item in values),
                "matched_count": sum(item.matched_count for item in values),
                "inserted_count": sum(item.inserted_count for item in values),
                "duplicate_count": sum(item.duplicate_count for item in values),
            },
        }


# ============================================================================
# BASIC VALIDATION AND SERIALIZATION
# ============================================================================

def _finite_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> Optional[int]:
    """Convert a value to an integer or return None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _json_safe(value: Any) -> Any:
    """Convert numerical and path-like values into deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """Build a stable identifier from a JSON-safe mapping."""
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_candle_value(candle: Any, field_name: str) -> Any:
    """Read a field from a Candle instance or a mapping."""
    if isinstance(candle, Mapping):
        return candle.get(field_name)
    return getattr(candle, field_name, None)


def _normalize_candles(candles: Sequence[Any]) -> List[Dict[str, Any]]:
    """Validate candles, sort them by opening time and keep the last duplicate."""
    by_time: Dict[int, Dict[str, Any]] = {}
    for candle in candles:
        timestamp = _finite_int(_read_candle_value(candle, "time"))
        open_price = _finite_float(_read_candle_value(candle, "open"))
        high_price = _finite_float(_read_candle_value(candle, "high"))
        low_price = _finite_float(_read_candle_value(candle, "low"))
        close_price = _finite_float(_read_candle_value(candle, "close"))
        volume = _finite_float(_read_candle_value(candle, "volume"))

        if timestamp is None or timestamp < 0:
            continue
        if any(
            value is None
            for value in (open_price, high_price, low_price, close_price)
        ):
            continue

        open_value = float(open_price)
        high_value = float(high_price)
        low_value = float(low_price)
        close_value = float(close_price)
        if high_value < low_value:
            continue
        if high_value < max(open_value, close_value):
            continue
        if low_value > min(open_value, close_value):
            continue

        by_time[timestamp] = {
            "time": timestamp,
            "open": open_value,
            "high": high_value,
            "low": low_value,
            "close": close_value,
            "volume": max(float(volume or 0.0), 0.0),
        }

    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _normalize_market_name(market: str) -> str:
    """Normalize the market through the application's canonical helper."""
    try:
        normalized = str(normalize_market(market)).strip()
    except Exception as error:
        raise SituationValidationError(f"Invalid market: {market!r}.") from error
    if not normalized:
        raise SituationValidationError("Market must not be empty.")
    return normalized


def _normalize_timeframe(timeframe: str) -> str:
    """Normalize a timeframe and validate it for the current runtime mode."""
    normalized = str(timeframe).strip().upper()
    if normalized not in TIMEFRAME_SECONDS:
        raise SituationValidationError(f"Unknown timeframe: {normalized!r}.")
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise SituationValidationError(
            f"Timeframe {normalized!r} is unavailable in the current runtime."
        )
    return normalized


def _validate_catalog(market: str, catalog: PatternCatalog) -> None:
    """Ensure that the catalog belongs to the market being scanned."""
    if not isinstance(catalog, PatternCatalog):
        raise TypeError("catalog must be a PatternCatalog instance.")
    if catalog.market != market:
        raise SituationCatalogError(
            f"Catalog market {catalog.market!r} does not match {market!r}."
        )


def _validate_time_range(
    start_time: Optional[int],
    end_time: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """Validate optional Unix-second recognition-time bounds."""
    start = _finite_int(start_time) if start_time is not None else None
    end = _finite_int(end_time) if end_time is not None else None
    if start_time is not None and start is None:
        raise SituationValidationError("start_time must be an integer Unix timestamp.")
    if end_time is not None and end is None:
        raise SituationValidationError("end_time must be an integer Unix timestamp.")
    if start is not None and start < 0:
        raise SituationValidationError("start_time cannot be negative.")
    if end is not None and end < 0:
        raise SituationValidationError("end_time cannot be negative.")
    if start is not None and end is not None and end < start:
        raise SituationValidationError("end_time cannot precede start_time.")
    return start, end


# ============================================================================
# NUMERICAL PATTERN MATCHING
# ============================================================================

def _definition_feature_map(
    definition: PatternDefinition,
) -> Optional[Dict[str, float]]:
    """Extract a compatible pivot-feature template from a catalog definition."""
    numeric = definition.numeric_definition
    if not isinstance(numeric, Mapping):
        return None

    candidates: List[Any] = [
        numeric.get("pivot_features"),
        numeric.get("features"),
    ]
    pivot_geometry = numeric.get("pivot_geometry")
    if isinstance(pivot_geometry, Mapping):
        candidates.append(pivot_geometry.get("features"))
    geometry = numeric.get("geometry")
    if isinstance(geometry, Mapping):
        nested_pivot = geometry.get("pivot_geometry")
        if isinstance(nested_pivot, Mapping):
            candidates.append(nested_pivot.get("features"))
        candidates.append(geometry.get("features"))

    candidates.append(numeric)

    required = set(PIVOT_FEATURE_FIELDS)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if not required.issubset(candidate.keys()):
            continue
        normalized: Dict[str, float] = {}
        for field_name in PIVOT_FEATURE_FIELDS:
            number = _finite_float(candidate.get(field_name))
            if number is None:
                break
            normalized[field_name] = number
        if len(normalized) == len(PIVOT_FEATURE_FIELDS):
            return normalized

    return None


def geometry_similarity(
    candidate_features: Mapping[str, Any],
    reference_features: Mapping[str, Any],
) -> Optional[float]:
    """Return a scale-normalized similarity score for the fixed pivot schema.

    The score is a transparent baseline metric, not a final learned similarity
    model. All twenty configured pivot features must be present in both maps.
    Its threshold must be selected and validated separately before production
    assignments are enabled.
    """
    differences: List[float] = []
    for field_name in PIVOT_FEATURE_FIELDS:
        candidate = _finite_float(candidate_features.get(field_name))
        reference = _finite_float(reference_features.get(field_name))
        if candidate is None or reference is None:
            return None
        scale = max(abs(candidate), abs(reference), 1.0)
        differences.append(min(abs(candidate - reference) / scale, 2.0))

    if not differences:
        return None
    mean_difference = sum(differences) / len(differences)
    return float(max(0.0, min(1.0, 1.0 - mean_difference / 2.0)))


def match_geometry_to_catalog(
    features: Mapping[str, Any],
    catalog: PatternCatalog,
    *,
    similarity_threshold: Optional[float] = None,
) -> Tuple[Optional[PatternDefinition], Optional[float]]:
    """Match a geometry only to active definitions with compatible features.

    With no threshold, only a numerically identical template is accepted. A
    permissive approximate threshold is never selected implicitly.
    """
    if similarity_threshold is None:
        threshold = 1.0 - 1e-12
    else:
        threshold_value = _finite_float(similarity_threshold)
        if threshold_value is None or not 0.0 <= threshold_value <= 1.0:
            raise SituationValidationError(
                "similarity_threshold must be between zero and one."
            )
        threshold = threshold_value

    best_definition: Optional[PatternDefinition] = None
    best_score: Optional[float] = None

    for definition in catalog.active_patterns:
        reference = _definition_feature_map(definition)
        if reference is None:
            continue
        score = geometry_similarity(features, reference)
        if score is None or score < threshold:
            continue
        if best_score is None or score > best_score:
            best_definition = definition
            best_score = score
        elif score == best_score and best_definition is not None:
            if definition.pattern_id < best_definition.pattern_id:
                best_definition = definition
                best_score = score

    return best_definition, best_score


# ============================================================================
# STORAGE ADAPTERS
# ============================================================================

def _append_occurrence(
    market: str,
    record: Mapping[str, Any],
    storage: Any = None,
) -> Any:
    """Append an immutable pattern occurrence through the storage interface."""
    if storage is None:
        return append_pattern_occurrence(market, record)

    method = getattr(storage, "append_pattern_occurrence", None)
    if callable(method):
        return method(market, record)

    function = getattr(storage, "append_pattern_occurrence", None)
    if callable(function):
        return function(market, record)

    raise TypeError("storage must provide append_pattern_occurrence().")


def _append_event(
    market: str,
    record: Mapping[str, Any],
    storage: Any = None,
) -> Any:
    """Append one market event through the storage interface."""
    if storage is None:
        return append_market_event(market, record)
    method = getattr(storage, "append_market_event", None)
    if callable(method):
        return method(market, record)
    raise TypeError("storage must provide append_market_event().")


def _storage_result_inserted(result: Any) -> bool:
    """Read the idempotent insertion status from a storage result."""
    return bool(getattr(result, "inserted", False))


# ============================================================================
# SITUATION SCANNER
# ============================================================================

class SituationScanner:
    """Scan closed historical candles and persist only approved pattern matches.

    The scanner does not invent a category, pattern ID, similarity threshold,
    or candidate approval decision. Unmatched geometries are returned as
    discovery candidates for a later clustering and catalog-review process.
    """

    def __init__(
        self,
        market: str,
        catalog: PatternCatalog,
        config: SituationScanConfig,
        *,
        storage: Any = None,
    ) -> None:
        self.market = _normalize_market_name(market)
        _validate_catalog(self.market, catalog)
        if not isinstance(config, SituationScanConfig):
            raise TypeError("config must be a SituationScanConfig instance.")
        self.catalog = catalog
        self.config = config
        self.storage = storage

    def scan_timeframe(
        self,
        candles: Sequence[Any],
        timeframe: str,
        *,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> SituationScanResult:
        """Scan one timeframe at each confirmed-pivot recognition point.

        Each candidate uses only a fixed number of candles ending at the candle
        that confirms the latest pivot. The event start/end are the first and
        last pivot timestamps; recognized_at is the confirmation time. This
        keeps the formation interval separate from the time it became knowable.
        """
        normalized_timeframe = _normalize_timeframe(timeframe)
        if normalized_timeframe not in self.config.window_bars_by_timeframe:
            raise SituationValidationError(
                f"No window size configured for {normalized_timeframe}."
            )

        start_bound, end_bound = _validate_time_range(start_time, end_time)
        ordered = _normalize_candles(candles)
        window_bars = int(
            self.config.window_bars_by_timeframe[normalized_timeframe]
        )
        duration = TIMEFRAME_SECONDS[normalized_timeframe]

        if len(ordered) < window_bars:
            return SituationScanResult(
                market=self.market,
                timeframe=normalized_timeframe,
                catalog_version=self.catalog.catalog_version,
                candle_count=len(ordered),
                confirmed_pivot_count=0,
                candidate_count=0,
                matched_count=0,
                inserted_count=0,
                duplicate_count=0,
            )

        confirmed_pivots = detect_pivots(
            ordered,
            timeframe=normalized_timeframe,
            left_bars=self.config.left_bars,
            right_bars=self.config.right_bars,
        )
        anchors: Dict[int, List[Dict[str, Any]]] = {}
        for pivot in confirmed_pivots:
            confirmation_index = _finite_int(pivot.get("confirmation_index"))
            if confirmation_index is None:
                continue
            anchors.setdefault(confirmation_index, []).append(dict(pivot))

        unmatched: List[SituationCandidate] = []
        matched: List[SituationCandidate] = []
        inserted_count = 0
        duplicate_count = 0
        seen_candidate_ids: set[str] = set()
        candidate_limit = self.config.max_candidates_per_timeframe

        for confirmation_index in sorted(anchors):
            if candidate_limit is not None and len(unmatched) + len(matched) >= candidate_limit:
                break

            recognition_time = (
                int(ordered[confirmation_index]["time"]) + duration
            )
            if start_bound is not None and recognition_time < start_bound:
                continue
            if end_bound is not None and recognition_time > end_bound:
                continue

            first_index = max(0, confirmation_index - window_bars + 1)
            if confirmation_index - first_index + 1 < window_bars:
                continue

            segment = ordered[first_index:confirmation_index + 1]
            geometry = pattern_geometry(
                segment,
                timeframe=normalized_timeframe,
                calculated_at=recognition_time,
                left_bars=self.config.left_bars,
                right_bars=self.config.right_bars,
            )
            pivot_geometry = geometry.get("pivot_geometry", {})
            if not isinstance(pivot_geometry, Mapping):
                continue

            raw_features = pivot_geometry.get("features", {})
            if not isinstance(raw_features, Mapping):
                continue
            features = {
                name: float(raw_features.get(name, 0.0))
                for name in PIVOT_FEATURE_FIELDS
            }
            pivot_count = _finite_int(pivot_geometry.get("pivot_count")) or 0
            if pivot_count < self.config.min_pivots:
                continue

            pivot_times = pivot_geometry.get("pivot_times", [])
            pivot_types = pivot_geometry.get("pivot_types", [])
            pivot_prices = pivot_geometry.get("pivot_prices", [])
            recognition_times = pivot_geometry.get("recognition_times", [])
            if not pivot_times or not pivot_types or not pivot_prices:
                continue

            occurrence_start = int(pivot_times[0])
            occurrence_end = int(pivot_times[-1])
            recognized_at = max(
                [recognition_time]
                + [
                    int(value)
                    for value in recognition_times
                    if _finite_int(value) is not None
                ]
            )
            if recognized_at < occurrence_end:
                raise SituationError(
                    "A candidate recognition timestamp precedes its final pivot."
                )

            net_change = _finite_float(
                pivot_geometry.get("net_price_change")
            )
            direction = (
                "UP"
                if net_change is not None and net_change > 0.0
                else "DOWN"
                if net_change is not None and net_change < 0.0
                else "FLAT"
            )

            latest_pivot = {
                "pivot_type": str(pivot_types[-1]),
                "price": float(pivot_prices[-1]),
                "pivot_time": int(pivot_times[-1]),
                "recognition_time": int(recognition_times[-1]),
            }
            identity_payload = {
                "market": self.market,
                "timeframe": normalized_timeframe,
                "occurrence_start": occurrence_start,
                "occurrence_end": occurrence_end,
                "recognized_at": recognized_at,
                "window_bars": window_bars,
                "left_bars": self.config.left_bars,
                "right_bars": self.config.right_bars,
                "pivot_types": list(pivot_types),
                "pivot_times": list(pivot_times),
            }
            candidate_id = _canonical_hash(identity_payload)
            if candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)

            matched_definition, similarity = match_geometry_to_catalog(
                features,
                self.catalog,
                similarity_threshold=self.config.similarity_threshold,
            )

            candidate = SituationCandidate(
                candidate_id=candidate_id,
                market=self.market,
                timeframe=normalized_timeframe,
                occurrence_start=occurrence_start,
                occurrence_end=occurrence_end,
                recognized_at=recognized_at,
                window_start=int(segment[0]["time"]),
                window_end=int(segment[-1]["time"]) + duration,
                pivot_count=pivot_count,
                direction=direction,
                geometry={
                    "schema_version": SITUATION_SCHEMA_VERSION,
                    "detector_version": SITUATION_DETECTOR_VERSION,
                    "candle_geometry": geometry,
                    "pivot_geometry": pivot_geometry,
                    "window_bars": window_bars,
                    "left_bars": self.config.left_bars,
                    "right_bars": self.config.right_bars,
                },
                features=features,
                latest_pivot=latest_pivot,
                matched_pattern_id=(
                    matched_definition.pattern_id if matched_definition else None
                ),
                similarity=similarity,
            )

            if matched_definition is None:
                unmatched.append(candidate)
                continue

            matched.append(candidate)
            if not self.config.persist_matched_occurrences:
                continue

            occurrence_record = self._build_occurrence_record(
                candidate,
                matched_definition,
            )
            write_result = _append_occurrence(
                self.market,
                occurrence_record,
                self.storage,
            )
            if _storage_result_inserted(write_result):
                inserted_count += 1
            else:
                duplicate_count += 1

        return SituationScanResult(
            market=self.market,
            timeframe=normalized_timeframe,
            catalog_version=self.catalog.catalog_version,
            candle_count=len(ordered),
            confirmed_pivot_count=len(confirmed_pivots),
            candidate_count=len(unmatched) + len(matched),
            matched_count=len(matched),
            inserted_count=inserted_count,
            duplicate_count=duplicate_count,
            unmatched_candidates=tuple(unmatched),
            matched_candidates=tuple(matched),
        )

    def _build_occurrence_record(
        self,
        candidate: SituationCandidate,
        definition: PatternDefinition,
    ) -> Dict[str, Any]:
        """Build the immutable storage record for one active pattern match."""
        record_payload = {
            "candidate_id": candidate.candidate_id,
            "market": candidate.market,
            "timeframe": candidate.timeframe,
            "pattern_id": definition.pattern_id,
            "catalog_version": self.catalog.catalog_version,
            "occurrence_start": candidate.occurrence_start,
            "occurrence_end": candidate.occurrence_end,
            "recognized_at": candidate.recognized_at,
            "direction": candidate.direction,
            "status": "CONFIRMED",
            "geometry": _json_safe(dict(candidate.geometry)),
            "features": _json_safe(dict(candidate.features)),
            "similarity": candidate.similarity,
            "pattern_status": definition.status,
            "detector_version": SITUATION_DETECTOR_VERSION,
        }
        record_payload["occurrence_id"] = _canonical_hash(record_payload)
        return record_payload


# ============================================================================
# PUBLIC SCAN FUNCTIONS
# ============================================================================

def scan_timeframe(
    market: str,
    timeframe: str,
    candles: Sequence[Any],
    catalog: PatternCatalog,
    config: SituationScanConfig,
    *,
    storage: Any = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> SituationScanResult:
    """Scan one market timeframe using an explicit versioned catalog."""
    scanner = SituationScanner(
        market,
        catalog,
        config,
        storage=storage,
    )
    return scanner.scan_timeframe(
        candles,
        timeframe,
        start_time=start_time,
        end_time=end_time,
    )


def scan_market_history(
    market: str,
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    catalog: PatternCatalog,
    config: SituationScanConfig,
    *,
    storage: Any = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> MarketSituationScanResult:
    """Scan all supplied configured timeframes in a deterministic order."""
    normalized_market = _normalize_market_name(market)
    _validate_catalog(normalized_market, catalog)
    if not isinstance(candles_by_timeframe, Mapping):
        raise TypeError("candles_by_timeframe must be a mapping.")

    normalized_inputs: Dict[str, Sequence[Any]] = {}
    for raw_timeframe, candles in candles_by_timeframe.items():
        timeframe = _normalize_timeframe(str(raw_timeframe))
        if timeframe in normalized_inputs:
            raise SituationValidationError(
                f"Duplicate timeframe after normalization: {timeframe}."
            )
        normalized_inputs[timeframe] = candles

    missing_configuration = set(normalized_inputs) - set(
        config.window_bars_by_timeframe
    )
    if missing_configuration:
        raise SituationValidationError(
            "Missing scan window configuration for: "
            + ", ".join(sorted(missing_configuration))
        )

    results: Dict[str, SituationScanResult] = {}
    for timeframe in SUPPORTED_TIMEFRAMES:
        if timeframe not in normalized_inputs:
            continue
        results[timeframe] = scan_timeframe(
            normalized_market,
            timeframe,
            normalized_inputs[timeframe],
            catalog,
            config,
            storage=storage,
            start_time=start_time,
            end_time=end_time,
        )

    return MarketSituationScanResult(
        market=normalized_market,
        catalog_version=catalog.catalog_version,
        timeframe_results=results,
    )


# ============================================================================
# CATALOG CANDIDATE LIFECYCLE
# ============================================================================

def register_recurrent_pattern_candidate(
    catalog: PatternCatalog,
    category_id: int,
    candidate_geometries: Sequence[Mapping[str, Any]],
    *,
    min_occurrences: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    storage: Any = None,
) -> Tuple[PatternCatalog, PatternDefinition]:
    """Create a catalog candidate from a pre-grouped recurrent geometry cluster.

    The caller must provide the category and recurrence requirement explicitly.
    This function does not invent a similarity threshold or decide that one
    isolated geometry deserves a permanent pattern identifier.
    """
    if not isinstance(catalog, PatternCatalog):
        raise TypeError("catalog must be a PatternCatalog instance.")
    required_occurrences = _finite_int(min_occurrences)
    if required_occurrences is None or required_occurrences < 2:
        raise SituationValidationError("min_occurrences must be at least two.")
    if not isinstance(candidate_geometries, Sequence) or isinstance(
        candidate_geometries, (str, bytes)
    ):
        raise TypeError("candidate_geometries must be a sequence of geometry records.")
    if len(candidate_geometries) < required_occurrences:
        raise SituationValidationError(
            f"At least {required_occurrences} recurrent geometries are required."
        )

    feature_maps: List[Dict[str, float]] = []
    for item in candidate_geometries:
        if not isinstance(item, Mapping):
            raise SituationValidationError("Every candidate geometry must be a mapping.")
        source = item.get("features")
        if source is None:
            pivot_geometry = item.get("pivot_geometry")
            if isinstance(pivot_geometry, Mapping):
                source = pivot_geometry.get("features")
        if not isinstance(source, Mapping):
            raise SituationValidationError("A candidate geometry has no feature map.")

        normalized_features: Dict[str, float] = {}
        for field_name in PIVOT_FEATURE_FIELDS:
            value = _finite_float(source.get(field_name))
            if value is None:
                raise SituationValidationError(
                    f"Candidate geometry is missing finite feature {field_name!r}."
                )
            normalized_features[field_name] = value
        feature_maps.append(normalized_features)

    mean_features = {
        field_name: sum(item[field_name] for item in feature_maps) / len(feature_maps)
        for field_name in PIVOT_FEATURE_FIELDS
    }
    feature_std = {
        field_name: math.sqrt(
            sum(
                (item[field_name] - mean_features[field_name]) ** 2
                for item in feature_maps
            )
            / len(feature_maps)
        )
        for field_name in PIVOT_FEATURE_FIELDS
    }

    new_catalog = catalog.fork()
    definition = new_catalog.add_candidate(
        category_id,
        name=name,
        description=description,
        numeric_definition={
            "features": mean_features,
            "feature_standard_deviation": feature_std,
            "discovery_occurrence_count": len(feature_maps),
            "discovery_min_occurrences": required_occurrences,
        },
    )
    if storage is None:
        save_catalog(new_catalog)
    else:
        saver = getattr(storage, "save_pattern_catalog", None)
        if not callable(saver):
            raise TypeError("storage must provide save_pattern_catalog().")
        saver(new_catalog.market, new_catalog.catalog_version, new_catalog.to_dict())

    return new_catalog, definition


def approve_pattern_candidate(
    catalog: PatternCatalog,
    pattern_id: str,
    *,
    storage: Any = None,
) -> Tuple[PatternCatalog, PatternDefinition]:
    """Approve a candidate in a new immutable catalog version."""
    if not isinstance(catalog, PatternCatalog):
        raise TypeError("catalog must be a PatternCatalog instance.")

    new_catalog = catalog.fork()
    definition = new_catalog.approve_candidate(pattern_id)
    if storage is None:
        save_catalog(new_catalog)
    else:
        saver = getattr(storage, "save_pattern_catalog", None)
        if not callable(saver):
            raise TypeError("storage must provide save_pattern_catalog().")
        saver(new_catalog.market, new_catalog.catalog_version, new_catalog.to_dict())

    return new_catalog, definition


def load_situation_catalog(
    market: str,
    catalog_version: str,
    *,
    storage: Any = None,
) -> PatternCatalog:
    """Load one immutable catalog version for historical situation scanning."""
    normalized_market = _normalize_market_name(market)
    catalog = load_catalog(normalized_market, catalog_version, storage=storage)
    if catalog is None:
        raise SituationCatalogError(
            f"Catalog version {catalog_version!r} does not exist for {normalized_market}."
        )
    _validate_catalog(normalized_market, catalog)
    return catalog


# ============================================================================
# MARKET EVENT RECORDING
# ============================================================================

def record_market_event(
    market: str,
    *,
    timeframe: str,
    event_type: str,
    event_time: int,
    recognized_at: Optional[int] = None,
    features: Optional[Mapping[str, Any]] = None,
    event_id: Optional[str] = None,
    storage: Any = None,
) -> Any:
    """Persist one separately classified market observation.

    Event detection rules are deliberately kept outside this function because
    the project specification has not yet fixed their price-distance or
    volatility thresholds. This helper records an event once another detector
    has established its type and recognition timestamp.
    """
    normalized_market = _normalize_market_name(market)
    normalized_timeframe = _normalize_timeframe(timeframe)
    normalized_type = str(event_type).strip().upper()
    if normalized_type not in set(MARKET_EVENT_TYPES):
        raise SituationValidationError(
            f"Unsupported market event type: {normalized_type!r}."
        )

    event_timestamp = _finite_int(event_time)
    recognition_timestamp = (
        event_timestamp
        if recognized_at is None
        else _finite_int(recognized_at)
    )
    if event_timestamp is None or event_timestamp < 0:
        raise SituationValidationError("event_time must be a non-negative Unix timestamp.")
    if recognition_timestamp is None or recognition_timestamp < event_timestamp:
        raise SituationValidationError(
            "recognized_at must be greater than or equal to event_time."
        )
    if features is not None and not isinstance(features, Mapping):
        raise TypeError("features must be a mapping when supplied.")

    payload: Dict[str, Any] = {
        "event_type": normalized_type,
        "timeframe": normalized_timeframe,
        "event_time": event_timestamp,
        "recognized_at": recognition_timestamp,
        "features": _json_safe(dict(features or {})),
        "detector_version": SITUATION_DETECTOR_VERSION,
    }
    if event_id is not None:
        payload["event_id"] = str(event_id)
    return _append_event(normalized_market, payload, storage)


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "SITUATION_DETECTOR_VERSION",
    "SITUATION_SCHEMA_VERSION",
    "SituationError",
    "SituationValidationError",
    "SituationCatalogError",
    "SituationScanConfig",
    "SituationCandidate",
    "SituationScanResult",
    "MarketSituationScanResult",
    "SituationScanner",
    "geometry_similarity",
    "match_geometry_to_catalog",
    "scan_timeframe",
    "scan_market_history",
    "register_recurrent_pattern_candidate",
    "approve_pattern_candidate",
    "load_situation_catalog",
    "record_market_event",
]
