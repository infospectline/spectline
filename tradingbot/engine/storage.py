"""Append-only historical storage for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import dataclasses
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

try:
    from client import normalize_market
    from engine.settings import (
        MARKET_EVENT_ROOT,
        MARKET_PATTERN_ROOT,
        PATTERN_CATALOG_ROOT,
        SUPPORTED_TIMEFRAMES,
        ALL_TIMEFRAMES,
        MARKET_EVENT_TYPES,
    )
except ImportError:
    from ..client import normalize_market
    from .settings import (
        MARKET_EVENT_ROOT,
        MARKET_PATTERN_ROOT,
        PATTERN_CATALOG_ROOT,
        SUPPORTED_TIMEFRAMES,
        ALL_TIMEFRAMES,
        MARKET_EVENT_TYPES,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

STORAGE_SCHEMA_VERSION = 1
PATTERN_OCCURRENCES_DIRECTORY_NAME = "OCCURRENCES"
MARKET_EVENTS_DIRECTORY_NAME = "OBSERVATIONS"
CATALOG_FILENAME_PREFIX = "catalog_"
CATALOG_FILENAME_SUFFIX = ".json"
JSONL_SUFFIX = ".jsonl"
LOCK_SUFFIX = ".lock"

_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PATTERN_ID_PATTERN = re.compile(r"^[0-9]{4}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")

_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class StorageError(RuntimeError):
    """Base exception for historical storage failures."""


class StorageValidationError(StorageError, ValueError):
    """Raised when a record does not satisfy the storage contract."""


class StorageConflictError(StorageError):
    """Raised when an immutable identifier already contains different data."""


class StorageCorruptionError(StorageError):
    """Raised when a stored JSON or JSONL record cannot be decoded safely."""


@dataclass(frozen=True)
class StorageWriteResult:
    """Describe the result of an immutable append or snapshot write."""

    record_id: str
    path: Path
    inserted: bool
    checksum: str


@dataclass(frozen=True)
class StorageArchiveInfo:
    """Describe one JSONL archive without loading all records into memory."""

    path: Path
    record_count: int
    first_timestamp: Optional[int]
    last_timestamp: Optional[int]
    file_size_bytes: int


# ============================================================================
# NORMALIZATION AND SERIALIZATION
# ============================================================================

def _safe_component(value: Any, field_name: str) -> str:
    """Validate one identifier before using it as a path component."""
    normalized = str(value).strip()
    if not normalized or not _SAFE_COMPONENT_PATTERN.fullmatch(normalized):
        raise StorageValidationError(f"Invalid {field_name}.")
    if normalized in {".", ".."}:
        raise StorageValidationError(f"Invalid {field_name}.")
    return normalized


def _normalize_market(market: str) -> str:
    """Normalize a market using the application's canonical market helper."""
    try:
        normalized = str(normalize_market(market)).strip()
    except Exception as error:
        raise StorageValidationError(f"Invalid market: {market!r}.") from error
    return _safe_component(normalized, "market")


def _normalize_timeframe(timeframe: str) -> str:
    """Validate a timeframe against the complete configured catalog."""
    normalized = str(timeframe).strip().upper()
    available = set(ALL_TIMEFRAMES) if ALL_TIMEFRAMES else set(SUPPORTED_TIMEFRAMES)
    if normalized not in available:
        raise StorageValidationError(f"Unsupported timeframe: {normalized!r}.")
    return normalized


def _timestamp(value: Any, field_name: str) -> int:
    """Convert a timestamp to integer Unix seconds without guessing its unit."""
    if isinstance(value, bool):
        raise StorageValidationError(f"{field_name} must be an integer Unix timestamp.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise StorageValidationError(
            f"{field_name} must be an integer Unix timestamp."
        ) from error
    if number < 0:
        raise StorageValidationError(f"{field_name} cannot be negative.")
    return number

def _json_default(value: Any) -> Any:
    """Convert supported Python and numerical values into JSON-safe values."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach().cpu()
        return tensor.tolist()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TypeError("Naive datetime values are not supported in storage records.")
        return int(value.astimezone(timezone.utc).timestamp())
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}.")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Serialize a mapping deterministically and reject non-finite JSON values."""
    try:
        encoded = json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise StorageValidationError(f"Record cannot be serialized: {error}") from error
    return encoded.encode("utf-8")


def _sha256(payload: bytes) -> str:
    """Return a SHA-256 checksum for a serialized payload."""
    return hashlib.sha256(payload).hexdigest()


def _utc_now_timestamp() -> int:
    """Return the current UTC time as integer Unix seconds."""
    return int(datetime.now(timezone.utc).timestamp())


def _record_timestamp(record: Mapping[str, Any], candidates: Sequence[str]) -> Optional[int]:
    """Read the first available timestamp from a record."""
    for name in candidates:
        if record.get(name) is not None:
            return _timestamp(record[name], name)
    return None

# ============================================================================
# FILE LOCKING AND IMMUTABLE WRITES
# ============================================================================

@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire a process and thread lock for one archive or catalog file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())

    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock

    with lock:
        with lock_path.open("a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)

            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _append_immutable_jsonl(
    path: Path,
    record: Mapping[str, Any],
    record_kind: str,
    supplied_id_field: Optional[str] = None,
) -> StorageWriteResult:
    """Append one immutable JSONL record with idempotency and checksums."""
    normalized = dict(record)
    content_bytes = _canonical_json(normalized)
    checksum = _sha256(content_bytes)

    supplied_id = None
    if supplied_id_field:
        supplied_id = normalized.get(supplied_id_field)
    if supplied_id is None:
        supplied_id = normalized.get("record_id")

    if supplied_id is None:
        record_id = f"{record_kind}_{checksum[:32]}"
    else:
        record_id = _safe_component(supplied_id, "record identifier")

    stored_record = {
        **normalized,
        "record_id": record_id,
        "record_kind": record_kind,
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "stored_at": _utc_now_timestamp(),
        "content_sha256": checksum,
    }
    line = _canonical_json(stored_record) + b"\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + LOCK_SUFFIX)

    with _exclusive_file_lock(lock_path):
        if path.is_file():
            with path.open("r", encoding="utf-8") as existing_file:
                for line_number, raw_line in enumerate(existing_file, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        existing = json.loads(raw_line)
                    except json.JSONDecodeError as error:
                        raise StorageCorruptionError(
                            f"Invalid JSONL record in {path} at line {line_number}."
                        ) from error
                    if existing.get("record_id") != record_id:
                        continue
                    if existing.get("content_sha256") != checksum:
                        raise StorageConflictError(
                            f"Immutable {record_kind} ID {record_id!r} already exists "
                            "with different content."
                        )
                    return StorageWriteResult(
                        record_id=record_id,
                        path=path,
                        inserted=False,
                        checksum=checksum,
                    )

        try:
            with path.open("ab") as archive_file:
                archive_file.write(line)
                archive_file.flush()
                os.fsync(archive_file.fileno())
        except OSError as error:
            raise StorageError(f"Could not append record to {path}: {error}") from error

    return StorageWriteResult(
        record_id=record_id,
        path=path,
        inserted=True,
        checksum=checksum,
    )


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
) -> StorageWriteResult:
    """Write an immutable JSON snapshot and reject version replacement."""
    normalized = dict(payload)
    content_bytes = _canonical_json(normalized)
    checksum = _sha256(content_bytes)
    record_id = str(normalized.get("catalog_version") or checksum[:32])
    stored_payload = {
        **normalized,
        "storage_schema_version": STORAGE_SCHEMA_VERSION,
        "stored_at": _utc_now_timestamp(),
        "content_sha256": checksum,
    }
    encoded = _canonical_json(stored_payload) + b"\n"
    lock_path = path.with_name(path.name + LOCK_SUFFIX)

    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(lock_path):
        if path.exists():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise StorageCorruptionError(
                    f"Could not read existing immutable catalog {path}: {error}"
                ) from error
            if existing_payload.get("content_sha256") != checksum:
                raise StorageConflictError(
                    f"Immutable catalog already exists at {path} with different content."
                )
            return StorageWriteResult(
                record_id=record_id,
                path=path,
                inserted=False,
                checksum=checksum,
            )

        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise StorageError(f"Could not write immutable catalog {path}: {error}") from error

    return StorageWriteResult(
        record_id=record_id,
        path=path,
        inserted=True,
        checksum=checksum,
    )


# ============================================================================
# RECORD VALIDATION
# ============================================================================

def _validate_pattern_occurrence(
    market: str,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize one market-wide pattern occurrence."""
    normalized = dict(record)
    pattern_id = str(normalized.get("pattern_id", "")).strip()
    if not _PATTERN_ID_PATTERN.fullmatch(pattern_id):
        raise StorageValidationError("pattern_id must be a four-digit string.")

    timeframe = _normalize_timeframe(str(normalized.get("timeframe", "")))
    occurrence_start = _timestamp(normalized.get("occurrence_start"), "occurrence_start")
    occurrence_end = _timestamp(normalized.get("occurrence_end"), "occurrence_end")
    recognized_at = _timestamp(normalized.get("recognized_at"), "recognized_at")

    if occurrence_end < occurrence_start:
        raise StorageValidationError("occurrence_end cannot precede occurrence_start.")
    if recognized_at < occurrence_start:
        raise StorageValidationError("recognized_at cannot precede occurrence_start.")

    catalog_version = _safe_component(
        normalized.get("catalog_version", ""),
        "catalog_version",
    )
    geometry = normalized.get("geometry", {})
    features = normalized.get("features", {})
    if not isinstance(geometry, Mapping) or not isinstance(features, Mapping):
        raise StorageValidationError("geometry and features must be mappings.")

    normalized.update(
        {
            "market": market,
            "timeframe": timeframe,
            "pattern_id": pattern_id,
            "occurrence_start": occurrence_start,
            "occurrence_end": occurrence_end,
            "recognized_at": recognized_at,
            "catalog_version": catalog_version,
            "geometry": dict(geometry),
            "features": dict(features),
        }
    )
    return normalized


def _validate_market_event(
    market: str,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize one non-pattern market event."""
    normalized = dict(record)
    event_type = str(normalized.get("event_type", "")).strip().upper()
    if event_type not in set(MARKET_EVENT_TYPES):
        raise StorageValidationError(
            f"Unsupported market event type: {event_type!r}."
        )

    timeframe = _normalize_timeframe(str(normalized.get("timeframe", "")))
    event_time = _timestamp(normalized.get("event_time"), "event_time")
    recognized_at = _timestamp(
        normalized.get("recognized_at", event_time),
        "recognized_at",
    )
    if recognized_at < event_time:
        raise StorageValidationError("recognized_at cannot precede event_time.")

    features = normalized.get("features", {})
    if not isinstance(features, Mapping):
        raise StorageValidationError("features must be a mapping.")

    normalized.update(
        {
            "market": market,
            "timeframe": timeframe,
            "event_type": event_type,
            "event_time": event_time,
            "recognized_at": recognized_at,
            "features": dict(features),
        }
    )
    return normalized

# ============================================================================
# HISTORICAL STORAGE SERVICE
# ============================================================================

class HistoricalStorage:
    """Store shared market-wide pattern occurrences, market events, and catalog snapshots."""

    def __init__(
        self,
        market_pattern_root: Optional[Union[str, Path]] = None,
        market_event_root: Optional[Union[str, Path]] = None,
        pattern_catalog_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.market_pattern_root = Path(
            market_pattern_root if market_pattern_root is not None else MARKET_PATTERN_ROOT
        ).expanduser().resolve()
        self.market_event_root = Path(
            market_event_root if market_event_root is not None else MARKET_EVENT_ROOT
        ).expanduser().resolve()
        self.pattern_catalog_root = Path(
            pattern_catalog_root if pattern_catalog_root is not None else PATTERN_CATALOG_ROOT
        ).expanduser().resolve()

    def pattern_occurrence_path(
        self,
        market: str,
        timeframe: str,
        year: int,
    ) -> Path:
        """Return the market-wide annual pattern occurrence archive path."""
        normalized_market = _normalize_market(market)
        normalized_timeframe = _normalize_timeframe(timeframe)
        resolved_year = int(year)
        if resolved_year < 1970 or resolved_year > 9999:
            raise StorageValidationError("year is outside the supported range.")
        return (
            self.market_pattern_root
            / normalized_market
            / normalized_timeframe
            / PATTERN_OCCURRENCES_DIRECTORY_NAME
            / f"{resolved_year:04d}{JSONL_SUFFIX}"
        )

    def market_event_path(
        self,
        market: str,
        timeframe: str,
        year: int,
    ) -> Path:
        """Return the market-wide annual observation archive path."""
        normalized_market = _normalize_market(market)
        normalized_timeframe = _normalize_timeframe(timeframe)
        resolved_year = int(year)
        if resolved_year < 1970 or resolved_year > 9999:
            raise StorageValidationError("year is outside the supported range.")
        return (
            self.market_event_root
            / normalized_market
            / normalized_timeframe
            / MARKET_EVENTS_DIRECTORY_NAME
            / f"{resolved_year:04d}{JSONL_SUFFIX}"
        )

    def pattern_catalog_path(self, market: str, catalog_version: str) -> Path:
        """Return the immutable catalog snapshot path for one market/version."""
        normalized_market = _normalize_market(market)
        version = str(catalog_version).strip()
        if not _VERSION_PATTERN.fullmatch(version):
            raise StorageValidationError("Invalid catalog_version.")
        return (
            self.pattern_catalog_root
            / normalized_market
            / f"{CATALOG_FILENAME_PREFIX}{version}{CATALOG_FILENAME_SUFFIX}"
        )

    def save_pattern_catalog(
        self,
        market: str,
        catalog_version: str,
        catalog: Mapping[str, Any],
    ) -> StorageWriteResult:
        """Save an immutable pattern catalog version for one market."""
        normalized_market = _normalize_market(market)
        if not isinstance(catalog, Mapping):
            raise StorageValidationError("catalog must be a mapping.")

        payload = dict(catalog)
        declared_version = payload.get("catalog_version", catalog_version)
        if str(declared_version) != str(catalog_version):
            raise StorageValidationError(
                "The catalog payload version does not match catalog_version."
            )
        payload["market"] = normalized_market
        payload["catalog_version"] = str(catalog_version)
        payload.setdefault("storage_schema_version", STORAGE_SCHEMA_VERSION)

        path = self.pattern_catalog_path(normalized_market, catalog_version)
        return _write_immutable_json(path, payload)

    def load_pattern_catalog(
        self,
        market: str,
        catalog_version: str,
    ) -> Optional[Dict[str, Any]]:
        """Load one immutable pattern catalog version, if it exists."""
        path = self.pattern_catalog_path(market, catalog_version)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageCorruptionError(
                f"Could not read pattern catalog {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise StorageCorruptionError(f"Pattern catalog {path} is not a JSON object.")
        return payload

    def append_pattern_occurrence(
        self,
        market: str,
        record: Mapping[str, Any],
    ) -> StorageWriteResult:
        """Append one recognized pattern occurrence to the market memory."""
        normalized_market = _normalize_market(market)
        if not isinstance(record, Mapping):
            raise StorageValidationError("Pattern occurrence must be a mapping.")
        normalized = _validate_pattern_occurrence(normalized_market, record)
        year = datetime.fromtimestamp(
            normalized["occurrence_start"],
            tz=timezone.utc,
        ).year
        path = self.pattern_occurrence_path(
            normalized_market,
            normalized["timeframe"],
            year,
        )
        return _append_immutable_jsonl(
            path,
            normalized,
            record_kind="pattern_occurrence",
            supplied_id_field="occurrence_id",
        )

    def iter_pattern_occurrences(
        self,
        market: str,
        timeframe: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        pattern_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate pattern occurrences in chronological order across year shards."""
        normalized_market = _normalize_market(market)
        normalized_timeframe = (
            _normalize_timeframe(timeframe) if timeframe is not None else None
        )
        start = _timestamp(start_time, "start_time") if start_time is not None else None
        end = _timestamp(end_time, "end_time") if end_time is not None else None
        if start is not None and end is not None and end < start:
            raise StorageValidationError("end_time cannot precede start_time.")
        selected_pattern = None
        if pattern_id is not None:
            selected_pattern = str(pattern_id).strip()
            if not _PATTERN_ID_PATTERN.fullmatch(selected_pattern):
                raise StorageValidationError("pattern_id must contain four digits.")

        base = self.market_pattern_root / normalized_market
        if not base.is_dir():
            return

        timeframe_paths = (
            [base / normalized_timeframe]
            if normalized_timeframe is not None
            else [path for path in base.iterdir() if path.is_dir()]
        )
        records: list[Dict[str, Any]] = []
        for timeframe_path in timeframe_paths:
            archive_root = timeframe_path / PATTERN_OCCURRENCES_DIRECTORY_NAME
            if not archive_root.is_dir():
                continue
            for path in sorted(archive_root.glob(f"*{JSONL_SUFFIX}")):
                for record in _read_jsonl(path):
                    occurrence_time = _timestamp(
                        record.get("occurrence_start"),
                        "occurrence_start",
                    )
                    if start is not None and occurrence_time < start:
                        continue
                    if end is not None and occurrence_time > end:
                        continue
                    if selected_pattern is not None and record.get("pattern_id") != selected_pattern:
                        continue
                    records.append(record)

        records.sort(
            key=lambda item: (
                int(item.get("occurrence_start", 0)),
                str(item.get("timeframe", "")),
                str(item.get("record_id", "")),
            )
        )
        yield from records

    def append_market_event(
        self,
        market: str,
        record: Mapping[str, Any],
    ) -> StorageWriteResult:
        """Append one breakout, retest, rejection or other market observation."""
        normalized_market = _normalize_market(market)
        if not isinstance(record, Mapping):
            raise StorageValidationError("Market event must be a mapping.")
        normalized = _validate_market_event(normalized_market, record)
        year = datetime.fromtimestamp(normalized["event_time"], tz=timezone.utc).year
        path = self.market_event_path(normalized_market, normalized["timeframe"], year)
        return _append_immutable_jsonl(
            path,
            normalized,
            record_kind="market_event",
            supplied_id_field="event_id",
        )

    def iter_market_events(
        self,
        market: str,
        timeframe: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        event_types: Optional[Sequence[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate market events chronologically with optional filters."""
        normalized_market = _normalize_market(market)
        normalized_timeframe = (
            _normalize_timeframe(timeframe) if timeframe is not None else None
        )
        start = _timestamp(start_time, "start_time") if start_time is not None else None
        end = _timestamp(end_time, "end_time") if end_time is not None else None
        if start is not None and end is not None and end < start:
            raise StorageValidationError("end_time cannot precede start_time.")
        selected_types = None
        if event_types is not None:
            selected_types = {str(value).strip().upper() for value in event_types}
            unknown = selected_types - set(MARKET_EVENT_TYPES)
            if unknown:
                raise StorageValidationError(
                    "Unsupported market event types: " + ", ".join(sorted(unknown))
                )

        base = self.market_event_root / normalized_market
        if not base.is_dir():
            return
        timeframe_paths = (
            [base / normalized_timeframe]
            if normalized_timeframe is not None
            else [path for path in base.iterdir() if path.is_dir()]
        )
        records: list[Dict[str, Any]] = []
        for timeframe_path in timeframe_paths:
            archive_root = timeframe_path / MARKET_EVENTS_DIRECTORY_NAME
            if not archive_root.is_dir():
                continue
            for path in sorted(archive_root.glob(f"*{JSONL_SUFFIX}")):
                for record in _read_jsonl(path):
                    event_time = _timestamp(record.get("event_time"), "event_time")
                    if start is not None and event_time < start:
                        continue
                    if end is not None and event_time > end:
                        continue
                    if selected_types is not None and record.get("event_type") not in selected_types:
                        continue
                    records.append(record)

        records.sort(
            key=lambda item: (
                int(item.get("event_time", 0)),
                str(item.get("timeframe", "")),
                str(item.get("record_id", "")),
            )
        )
        yield from records

    def inspect_archive(self, path: Union[str, Path], timestamp_field: str) -> StorageArchiveInfo:
        """Return count and chronological bounds for one JSONL archive."""
        archive_path = Path(path).expanduser().resolve()
        if not archive_path.is_file():
            return StorageArchiveInfo(
                path=archive_path,
                record_count=0,
                first_timestamp=None,
                last_timestamp=None,
                file_size_bytes=0,
            )

        count = 0
        first_timestamp: Optional[int] = None
        last_timestamp: Optional[int] = None
        for record in _read_jsonl(archive_path):
            current = _timestamp(record.get(timestamp_field), timestamp_field)
            count += 1
            first_timestamp = current if first_timestamp is None else min(first_timestamp, current)
            last_timestamp = current if last_timestamp is None else max(last_timestamp, current)

        return StorageArchiveInfo(
            path=archive_path,
            record_count=count,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            file_size_bytes=archive_path.stat().st_size,
        )


# ============================================================================
# JSONL READING
# ============================================================================

def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield validated JSON objects from an append-only JSONL archive."""
    try:
        with path.open("r", encoding="utf-8") as archive_file:
            for line_number, raw_line in enumerate(archive_file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise StorageCorruptionError(
                        f"Invalid JSONL record in {path} at line {line_number}."
                    ) from error
                if not isinstance(record, dict):
                    raise StorageCorruptionError(
                        f"JSONL record in {path} at line {line_number} is not an object."
                    )
                yield record
    except OSError as error:
        raise StorageError(f"Could not read archive {path}: {error}") from error


# ============================================================================
# DEFAULT SERVICE AND PUBLIC API
# ============================================================================

_default_storage = HistoricalStorage()


def append_pattern_occurrence(market: str, record: Mapping[str, Any]) -> StorageWriteResult:
    """Append one market-wide pattern occurrence using the default storage."""
    return _default_storage.append_pattern_occurrence(market, record)


def iter_pattern_occurrences(
    market: str,
    timeframe: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    pattern_id: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate market-wide pattern occurrences using the default storage."""
    return _default_storage.iter_pattern_occurrences(
        market,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        pattern_id=pattern_id,
    )


def append_market_event(market: str, record: Mapping[str, Any]) -> StorageWriteResult:
    """Append one market observation using the default storage."""
    return _default_storage.append_market_event(market, record)


def iter_market_events(
    market: str,
    timeframe: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    event_types: Optional[Sequence[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate market observations using the default storage."""
    return _default_storage.iter_market_events(
        market,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        event_types=event_types,
    )


def save_pattern_catalog(
    market: str,
    catalog_version: str,
    catalog: Mapping[str, Any],
) -> StorageWriteResult:
    """Save one immutable pattern catalog version using the default storage."""
    return _default_storage.save_pattern_catalog(market, catalog_version, catalog)


def load_pattern_catalog(
    market: str,
    catalog_version: str,
) -> Optional[Dict[str, Any]]:
    """Load one immutable pattern catalog version using the default storage."""
    return _default_storage.load_pattern_catalog(market, catalog_version)

# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "STORAGE_SCHEMA_VERSION",
    "StorageError",
    "StorageValidationError",
    "StorageConflictError",
    "StorageCorruptionError",
    "StorageWriteResult",
    "StorageArchiveInfo",
    "HistoricalStorage",
    "append_pattern_occurrence",
    "iter_pattern_occurrences",
    "append_market_event",
    "iter_market_events",
    "save_pattern_catalog",
    "load_pattern_catalog",
]
