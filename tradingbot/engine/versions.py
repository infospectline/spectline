"""Immutable artifact version registry for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

import numpy as np


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

VERSION_REGISTRY_SCHEMA = 1
VERSION_EVENT_SCHEMA = 1
DEFAULT_REGISTRY_DIRECTORY = "VERSIONS"
DEFAULT_ARTIFACT_DIRECTORY = "ARTIFACTS"
DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_LOCK_STALE_SECONDS = 300.0

ARTIFACT_PATTERN_CATALOG = "pattern_catalog"
ARTIFACT_DATASET = "dataset"
ARTIFACT_SCALER = "scaler"
ARTIFACT_MODEL = "model"
ARTIFACT_TRAINING_RUN = "training_run"
ARTIFACT_EVALUATION = "evaluation"
ARTIFACT_TYPES = frozenset(
    {
        ARTIFACT_PATTERN_CATALOG,
        ARTIFACT_DATASET,
        ARTIFACT_SCALER,
        ARTIFACT_MODEL,
        ARTIFACT_TRAINING_RUN,
        ARTIFACT_EVALUATION,
    }
)

SCOPE_MARKET = "MARKET"
SCOPE_CUSTOMER = "CUSTOMER"
SCOPE_GLOBAL = "GLOBAL"

STATUS_DRAFT = "DRAFT"
STATUS_VALIDATED = "VALIDATED"
STATUS_APPROVED = "APPROVED"
STATUS_DEPRECATED = "DEPRECATED"
STATUS_RETIRED = "RETIRED"
STATUS_REJECTED = "REJECTED"
VERSION_STATUSES = frozenset(
    {
        STATUS_DRAFT,
        STATUS_VALIDATED,
        STATUS_APPROVED,
        STATUS_DEPRECATED,
        STATUS_RETIRED,
        STATUS_REJECTED,
    }
)

_STATUS_TRANSITIONS = {
    STATUS_DRAFT: frozenset({STATUS_VALIDATED, STATUS_REJECTED}),
    STATUS_VALIDATED: frozenset({STATUS_APPROVED, STATUS_REJECTED}),
    STATUS_APPROVED: frozenset({STATUS_DEPRECATED}),
    STATUS_DEPRECATED: frozenset({STATUS_RETIRED}),
    STATUS_RETIRED: frozenset(),
    STATUS_REJECTED: frozenset(),
}

_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_SAFE_ARTIFACT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class VersionError(RuntimeError):
    """Base exception for version registry failures."""


class VersionValidationError(VersionError, ValueError):
    """Raised when a version record or its metadata is invalid."""


class VersionConflictError(VersionError):
    """Raised when an immutable version identifier already has other content."""


class VersionNotFoundError(VersionError, LookupError):
    """Raised when a requested artifact version does not exist."""


class VersionStateError(VersionError):
    """Raised when a version state transition is not permitted."""


class VersionIntegrityError(VersionError):
    """Raised when an artifact no longer matches its registered checksum."""


@dataclass(frozen=True)
class VersionDependency:
    """Identify one immutable artifact version required by another artifact."""

    artifact_type: str
    version_id: str
    scope_type: str
    scope_key: str
    market: Optional[str] = None
    identity: Optional[Mapping[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "version_id": self.version_id,
            "scope_type": self.scope_type,
            "scope_key": self.scope_key,
            "market": self.market,
            "identity": dict(self.identity) if self.identity is not None else None,
        }


@dataclass(frozen=True)
class VersionRecord:
    """Immutable artifact manifest together with its current registry state."""

    version_id: str
    artifact_type: str
    scope_type: str
    scope_key: str
    market: Optional[str]
    identity: Optional[Mapping[str, str]]
    artifact_uri: str
    artifact_sha256: str
    artifact_size_bytes: Optional[int]
    metadata: Mapping[str, Any]
    dependencies: Tuple[Mapping[str, Any], ...]
    parent_version_id: Optional[str]
    created_at: str
    status: str = STATUS_DRAFT
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "artifact_type": self.artifact_type,
            "scope_type": self.scope_type,
            "scope_key": self.scope_key,
            "market": self.market,
            "identity": dict(self.identity) if self.identity is not None else None,
            "artifact_uri": self.artifact_uri,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "metadata": _to_json_value(dict(self.metadata)),
            "dependencies": [_to_json_value(dict(item)) for item in self.dependencies],
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at,
            "status": self.status,
            "manifest_path": self.manifest_path,
        }


# ============================================================================
# NORMALIZATION AND SERIALIZATION
# ============================================================================

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_component(value: Any, field_name: str) -> str:
    normalized = str(value).strip()
    if not _SAFE_COMPONENT_PATTERN.fullmatch(normalized):
        raise VersionValidationError(f"Invalid {field_name}.")
    if normalized in {".", ".."}:
        raise VersionValidationError(f"Invalid {field_name}.")
    return normalized


def _normalize_artifact_type(value: Any) -> str:
    normalized = str(value).strip().lower()
    if not _SAFE_ARTIFACT_TYPE_PATTERN.fullmatch(normalized):
        raise VersionValidationError("Invalid artifact_type.")
    return normalized


def _normalize_market(value: Any) -> str:
    normalized = str(value).strip().upper().replace("/", "_").replace("\\", "_")
    if not normalized or len(normalized) > 128:
        raise VersionValidationError("Market must be a non-empty identifier.")
    if any(character in normalized for character in ("..", "\x00")):
        raise VersionValidationError("Invalid market identifier.")
    return normalized


def _identity_mapping(identity: Any) -> Dict[str, str]:
    if identity is None:
        raise VersionValidationError("A complete TradingBot identity is required.")

    if isinstance(identity, Mapping):
        raw = dict(identity)
    elif callable(getattr(identity, "as_dict", None)):
        raw = dict(identity.as_dict())
    else:
        raw = {
            "user_id": getattr(identity, "user_id", None),
            "market": getattr(identity, "market", None),
            "strategy_id": getattr(identity, "strategy_id", None),
            "tradingbot_id": getattr(identity, "tradingbot_id", None),
        }

    required = ("user_id", "market", "strategy_id", "tradingbot_id")
    missing = [name for name in required if raw.get(name) in (None, "")]
    if missing:
        raise VersionValidationError(
            "Incomplete TradingBot identity: " + ", ".join(missing)
        )

    return {
        "user_id": _safe_component(raw["user_id"], "user_id"),
        "market": _normalize_market(raw["market"]),
        "strategy_id": _safe_component(raw["strategy_id"], "strategy_id"),
        "tradingbot_id": _safe_component(raw["tradingbot_id"], "tradingbot_id"),
    }


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_json_value(value.to_dict())
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _to_json_value(value.as_dict())
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise VersionValidationError("Metadata cannot contain NaN or infinity.")
        return value
    raise VersionValidationError(
        f"Unsupported value in version metadata: {type(value).__name__}."
    )


def _canonical_json(value: Any) -> bytes:
    try:
        normalized = _to_json_value(value)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VersionValidationError("Value cannot be serialized as canonical JSON.") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> Tuple[str, int]:
    if not path.is_file():
        raise VersionValidationError(f"Artifact file does not exist: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_checksum(value: Any) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise VersionValidationError("artifact_sha256 must be a SHA-256 hexadecimal digest.")
    return normalized


def _validate_catalog_payload(payload: Mapping[str, Any], market: str, version_id: str) -> Dict[str, Any]:
    if int(payload.get("catalog_schema_version", -1)) != 1:
        raise VersionValidationError("Unsupported pattern catalog schema version.")
    if _normalize_market(payload.get("market", "")) != _normalize_market(market):
        raise VersionValidationError("Pattern catalog market does not match the requested market.")
    if str(payload.get("catalog_version", "")) != str(version_id):
        raise VersionValidationError("Pattern catalog version does not match the requested version.")

    categories = payload.get("categories")
    patterns = payload.get("patterns")
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise VersionValidationError("Pattern catalog categories must be a sequence.")
    if not isinstance(patterns, Sequence) or isinstance(patterns, (str, bytes)):
        raise VersionValidationError("Pattern catalog patterns must be a sequence.")
    if len(categories) != 10:
        raise VersionValidationError("Pattern catalog must contain exactly ten stable categories.")

    category_ids = set()
    for category in categories:
        if not isinstance(category, Mapping):
            raise VersionValidationError("Pattern category entries must be mappings.")
        category_id = int(category.get("category_id", -1))
        if category_id < 0 or category_id > 9 or category_id in category_ids:
            raise VersionValidationError("Pattern category identifiers must be unique values from 0 to 9.")
        category_ids.add(category_id)
    if category_ids != set(range(10)):
        raise VersionValidationError("Pattern catalog category identifiers must cover 0 through 9.")

    pattern_ids = set()
    allowed_pattern_statuses = {"CANDIDATE", "ACTIVE", "DEPRECATED", "RETIRED", "REJECTED"}
    for pattern in patterns:
        if not isinstance(pattern, Mapping):
            raise VersionValidationError("Pattern entries must be mappings.")
        pattern_id = str(pattern.get("pattern_id", ""))
        if not re.fullmatch(r"[0-9]{4}", pattern_id) or pattern_id in pattern_ids:
            raise VersionValidationError("Pattern IDs must be unique four-digit categorical identifiers.")
        pattern_ids.add(pattern_id)
        category_id = int(pattern.get("category_id", -1))
        if category_id not in category_ids or int(pattern_id[0]) != category_id:
            raise VersionValidationError("Pattern ID category digit does not match category_id.")
        status = str(pattern.get("status", "CANDIDATE")).upper()
        if status not in allowed_pattern_statuses:
            raise VersionValidationError(f"Unsupported pattern status: {status}.")
        _to_json_value(pattern.get("numeric_definition", {}))

    return {
        "valid": True,
        "market": _normalize_market(market),
        "catalog_version": str(version_id),
        "category_count": len(categories),
        "pattern_count": len(patterns),
    }


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return _utc_now()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(
                microsecond=0
            ).isoformat()
        except (OverflowError, OSError, ValueError) as error:
            raise VersionValidationError("Invalid created_at timestamp.") from error
    text = str(value).strip()
    if not text:
        raise VersionValidationError("created_at cannot be empty.")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise VersionValidationError("created_at must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise VersionValidationError("created_at must include a timezone.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


# ============================================================================
# FILE LOCKING AND IMMUTABLE FILE WRITES
# ============================================================================

@contextmanager
def _file_lock(
    lock_path: Path,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    started = time.monotonic()
    locked = False
    try:
        if os.name == "nt" and msvcrt is not None:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            while not locked:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
                    time.sleep(0.05)
        elif fcntl is not None:
            while not locked:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
                    time.sleep(0.05)
        else:
            key = str(lock_path.resolve())
            with _THREAD_LOCKS_GUARD:
                thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
            remaining = max(0.0, timeout - (time.monotonic() - started))
            locked = thread_lock.acquire(timeout=remaining)
            if not locked:
                raise TimeoutError(f"Timed out acquiring lock: {lock_path}")
            try:
                yield
            finally:
                thread_lock.release()
            return

        yield
    finally:
        if locked and os.name == "nt" and msvcrt is not None:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        elif locked and fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lock_file.close()


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> bool:
    encoded = _canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")

    with _file_lock(lock_path):
        if path.exists():
            existing = path.read_bytes()
            if existing == encoded:
                return False
            raise VersionConflictError(f"Immutable version file already exists: {path}")

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
            temporary_path = None
            try:
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return True


def _copy_immutable_artifact(source: Path, destination: Path) -> Tuple[str, int, bool]:
    if not source.is_file():
        raise VersionValidationError(f"Artifact file does not exist: {source}")
    source_hash, source_size = _sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")

    with _file_lock(lock_path):
        if destination.exists():
            existing_hash, existing_size = _sha256_file(destination)
            if existing_hash != source_hash or existing_size != source_size:
                raise VersionConflictError(
                    f"Immutable artifact snapshot already exists: {destination}"
                )
            return existing_hash, existing_size, False

        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                with source.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temporary_file, length=1024 * 1024)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    final_hash, final_size = _sha256_file(destination)
    if final_hash != source_hash or final_size != source_size:
        raise VersionIntegrityError("Copied artifact failed checksum verification.")
    return final_hash, final_size, True


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical_json(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path.with_name(path.name + ".lock")):
        with path.open("ab") as file_handle:
            file_handle.write(encoded)
            file_handle.flush()
            os.fsync(file_handle.fileno())


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VersionNotFoundError(f"Version manifest does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise VersionIntegrityError(f"Could not read version manifest: {path}") from error
    if not isinstance(payload, dict):
        raise VersionIntegrityError(f"Version manifest is not a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise VersionIntegrityError(
                    f"Invalid event log JSON at {path}:{line_number}."
                ) from error
            if not isinstance(item, dict):
                raise VersionIntegrityError(
                    f"Invalid event log record at {path}:{line_number}."
                )
            yield item


# ============================================================================
# REGISTRY IMPLEMENTATION
# ============================================================================

class VersionRegistry:
    """Store immutable artifact manifests and append-only lifecycle events.

    Market-wide pattern catalogs are scoped by market. Datasets, scalers,
    checkpoints and training artefacts are scoped by the full customer identity.
    The manifest never changes after creation; state changes are recorded as
    separate append-only events.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        *,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if root is None:
            project_root = Path(__file__).resolve().parents[1]
            try:
                from engine.settings import BASE_DIR
                project_root = Path(BASE_DIR).resolve()
            except (ImportError, AttributeError):
                try:
                    from .settings import BASE_DIR
                    project_root = Path(BASE_DIR).resolve()
                except (ImportError, AttributeError):
                    pass
            root = project_root / DEFAULT_REGISTRY_DIRECTORY
        self.root = Path(root).expanduser().resolve()
        self.artifact_root = self.root / DEFAULT_ARTIFACT_DIRECTORY
        self.lock_timeout = float(lock_timeout)
        if self.lock_timeout <= 0.0:
            raise VersionValidationError("lock_timeout must be positive.")

    def _scope(
        self,
        artifact_type: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str], Optional[Dict[str, str]]]:
        kind = _normalize_artifact_type(artifact_type)
        if scope_type is not None:
            requested_scope = str(scope_type).strip().upper()
            if requested_scope not in {SCOPE_MARKET, SCOPE_CUSTOMER, SCOPE_GLOBAL}:
                raise VersionValidationError("Unsupported scope_type.")
        else:
            requested_scope = (
                SCOPE_MARKET
                if kind == ARTIFACT_PATTERN_CATALOG
                else SCOPE_CUSTOMER
            )

        if requested_scope == SCOPE_MARKET:
            if market is None and identity is not None:
                market = _identity_mapping(identity)["market"]
            if market is None:
                raise VersionValidationError("market is required for market-scoped artifacts.")
            normalized_market = _normalize_market(market)
            safe_market = re.sub(r"[^A-Za-z0-9._+-]+", "_", normalized_market)
            scope_key = f"MARKET_{safe_market}"
            return requested_scope, scope_key, normalized_market, None

        if requested_scope == SCOPE_CUSTOMER:
            normalized_identity = _identity_mapping(identity)
            if market is not None and _normalize_market(market) != normalized_identity["market"]:
                raise VersionValidationError("market does not match the supplied TradingBot identity.")
            identity_json = _canonical_json(normalized_identity)
            identity_hash = _sha256_bytes(identity_json)[:24]
            scope_key = f"CUSTOMER_{identity_hash}"
            return requested_scope, scope_key, normalized_identity["market"], normalized_identity

        if identity is not None or market is not None:
            raise VersionValidationError("Global-scoped artifacts cannot include customer or market identity.")
        return requested_scope, "GLOBAL", None, None

    def _manifest_directory(self, scope_key: str, artifact_type: str) -> Path:
        return self.root / scope_key / artifact_type / "MANIFESTS"

    def _manifest_path(self, scope_key: str, artifact_type: str, version_id: str) -> Path:
        safe_id = _safe_component(version_id, "version_id")
        return self._manifest_directory(scope_key, artifact_type) / f"{safe_id}.json"

    def _event_path(self, scope_key: str, artifact_type: str) -> Path:
        return self.root / scope_key / artifact_type / "EVENTS.jsonl"

    def _artifact_snapshot_path(self, scope_key: str, artifact_type: str, version_id: str) -> Path:
        safe_id = _safe_component(version_id, "version_id")
        return self.artifact_root / scope_key / artifact_type / f"{safe_id}.json"

    def _normalize_dependencies(
        self,
        dependencies: Optional[Sequence[Any]],
        *,
        default_identity: Optional[Mapping[str, str]],
        default_market: Optional[str],
    ) -> Tuple[Dict[str, Any], ...]:
        normalized: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        for dependency in dependencies or ():
            if isinstance(dependency, VersionDependency):
                raw = dependency.to_dict()
            elif isinstance(dependency, Mapping):
                raw = dict(dependency)
            else:
                raise VersionValidationError("Dependencies must be mappings or VersionDependency records.")

            kind = _normalize_artifact_type(raw.get("artifact_type", ""))
            version_id = _safe_component(raw.get("version_id", ""), "dependency version_id")
            dep_identity = raw.get("identity", default_identity)
            dep_market = raw.get("market", default_market)
            explicit_scope = raw.get("scope_type")

            dep_scope, dep_scope_key, normalized_dep_market, normalized_dep_identity = self._scope(
                kind,
                identity=dep_identity,
                market=dep_market,
                scope_type=explicit_scope,
            )
            key = (kind, dep_scope_key, version_id)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "artifact_type": kind,
                    "version_id": version_id,
                    "scope_type": dep_scope,
                    "scope_key": dep_scope_key,
                    "market": normalized_dep_market,
                    "identity": normalized_dep_identity,
                }
            )

        return tuple(normalized)

    def register_version(
        self,
        artifact_type: str,
        version_id: str,
        *,
        artifact_path: Optional[Union[str, Path]] = None,
        payload: Any = None,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        parent_version_id: Optional[str] = None,
        created_at: Any = None,
    ) -> VersionRecord:
        kind = _normalize_artifact_type(artifact_type)
        safe_version_id = _safe_component(version_id, "version_id")
        if (artifact_path is None) == (payload is None):
            raise VersionValidationError("Provide exactly one of artifact_path or payload.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise VersionValidationError("metadata must be a mapping.")

        resolved_scope, scope_key, normalized_market, normalized_identity = self._scope(
            kind,
            identity=identity,
            market=market,
            scope_type=scope_type,
        )

        if artifact_path is not None:
            resolved_artifact_path = Path(artifact_path).expanduser().resolve()
            if not resolved_artifact_path.is_file():
                raise VersionValidationError(f"Artifact file does not exist: {resolved_artifact_path}")
            suffix = resolved_artifact_path.suffix.lower()
            if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
                suffix = ".artifact"
            snapshot_path = (
                self.artifact_root
                / scope_key
                / kind
                / f"{safe_version_id}{suffix}"
            )
            artifact_hash, artifact_size, _ = _copy_immutable_artifact(
                resolved_artifact_path,
                snapshot_path,
            )
            artifact_uri = str(snapshot_path.resolve())
        else:
            encoded_payload = _canonical_json(payload)
            snapshot_path = self._artifact_snapshot_path(scope_key, kind, safe_version_id)
            _write_immutable_json(snapshot_path, {"artifact_payload": _to_json_value(payload)})
            artifact_uri = str(snapshot_path.resolve())
            artifact_hash, artifact_size = _sha256_file(snapshot_path)

        normalized_metadata = _to_json_value(dict(metadata or {}))
        parent_id = (
            _safe_component(parent_version_id, "parent_version_id")
            if parent_version_id is not None
            else None
        )
        dependency_inputs = list(dependencies or ())
        if parent_id is not None:
            dependency_inputs.append(
                {
                    "artifact_type": kind,
                    "version_id": parent_id,
                    "scope_type": resolved_scope,
                    "identity": normalized_identity,
                    "market": normalized_market,
                }
            )
        normalized_dependencies = self._normalize_dependencies(
            dependency_inputs,
            default_identity=normalized_identity,
            default_market=normalized_market,
        )
        timestamp = _normalize_timestamp(created_at)

        manifest = {
            "registry_schema_version": VERSION_REGISTRY_SCHEMA,
            "version_id": safe_version_id,
            "artifact_type": kind,
            "scope_type": resolved_scope,
            "scope_key": scope_key,
            "market": normalized_market,
            "identity": normalized_identity,
            "artifact_uri": artifact_uri,
            "artifact_sha256": artifact_hash,
            "artifact_size_bytes": artifact_size,
            "metadata": normalized_metadata,
            "dependencies": list(normalized_dependencies),
            "parent_version_id": parent_id,
            "created_at": timestamp,
            "initial_status": STATUS_DRAFT,
        }

        manifest_path = self._manifest_path(scope_key, kind, safe_version_id)
        if manifest_path.exists():
            existing = _read_json(manifest_path)
            comparable_fields = (
                "version_id",
                "artifact_type",
                "scope_type",
                "scope_key",
                "market",
                "identity",
                "artifact_sha256",
                "artifact_size_bytes",
                "metadata",
                "dependencies",
                "parent_version_id",
            )
            if any(existing.get(field) != manifest.get(field) for field in comparable_fields):
                raise VersionConflictError(
                    f"Version {safe_version_id!r} already exists with different content."
                )
            return self._record_from_manifest(existing, self.get_status(kind, safe_version_id, scope_key=scope_key))

        _write_immutable_json(manifest_path, manifest)
        self._append_event(
            scope_key,
            kind,
            {
                "event_type": "CREATED",
                "version_id": safe_version_id,
                "from_status": None,
                "to_status": STATUS_DRAFT,
                "reason": "Version registered.",
            },
        )
        return self._record_from_manifest(manifest, STATUS_DRAFT)

    def _append_event(
        self,
        scope_key: str,
        artifact_type: str,
        event: Mapping[str, Any],
    ) -> None:
        record = {
            "event_schema_version": VERSION_EVENT_SCHEMA,
            "event_id": uuid.uuid4().hex,
            "event_timestamp": _utc_now(),
            **dict(event),
        }
        _append_jsonl(self._event_path(scope_key, artifact_type), record)

    def _status_events(self, scope_key: str, artifact_type: str) -> Iterator[Dict[str, Any]]:
        return _read_jsonl(self._event_path(scope_key, artifact_type))

    def get_status(
        self,
        artifact_type: str,
        version_id: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
    ) -> str:
        kind = _normalize_artifact_type(artifact_type)
        safe_version_id = _safe_component(version_id, "version_id")
        if scope_key is None:
            _, resolved_scope_key, _, _ = self._scope(
                kind,
                identity=identity,
                market=market,
                scope_type=scope_type,
            )
        else:
            resolved_scope_key = _safe_component(scope_key, "scope_key")

        manifest_path = self._manifest_path(resolved_scope_key, kind, safe_version_id)
        manifest = _read_json(manifest_path)
        event_path = self._event_path(resolved_scope_key, kind)
        with _file_lock(event_path.with_name(event_path.name + ".lock"), timeout=self.lock_timeout):
            current_status = self._status_unlocked(
                resolved_scope_key,
                kind,
                safe_version_id,
                manifest,
            )
        return current_status

    def _status_unlocked(
        self,
        scope_key: str,
        artifact_type: str,
        version_id: str,
        manifest: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if manifest is None:
            manifest = _read_json(self._manifest_path(scope_key, artifact_type, version_id))
        current_status = str(manifest.get("initial_status", STATUS_DRAFT)).upper()
        for event in self._status_events(scope_key, artifact_type):
            if event.get("version_id") == version_id and event.get("to_status"):
                current_status = str(event["to_status"]).upper()
        if current_status not in VERSION_STATUSES:
            raise VersionIntegrityError("Version event log contains an unknown status.")
        return current_status

    def _append_transition_event(
        self,
        scope_key: str,
        artifact_type: str,
        version_id: str,
        target_status: str,
        *,
        expected_statuses: Iterable[str],
        event_type: str,
        reason: str,
        actor: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        event_path = self._event_path(scope_key, artifact_type)
        with _file_lock(event_path.with_name(event_path.name + ".lock"), timeout=self.lock_timeout):
            manifest = _read_json(self._manifest_path(scope_key, artifact_type, version_id))
            current_status = self._status_unlocked(
                scope_key,
                artifact_type,
                version_id,
                manifest,
            )
            if current_status == target_status:
                return current_status
            if current_status not in set(expected_statuses):
                raise VersionStateError(
                    f"Version state changed concurrently or transition is invalid: "
                    f"{current_status} -> {target_status}."
                )
            event = {
                "event_schema_version": VERSION_EVENT_SCHEMA,
                "event_id": uuid.uuid4().hex,
                "event_timestamp": _utc_now(),
                "event_type": event_type,
                "version_id": version_id,
                "from_status": current_status,
                "to_status": target_status,
                "actor": actor,
                "reason": reason,
            }
            if extra:
                event.update(_to_json_value(dict(extra)))
            with event_path.open("ab") as file_handle:
                file_handle.write(_canonical_json(event) + b"\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
            return target_status

    def _record_from_manifest(self, manifest: Mapping[str, Any], status: str) -> VersionRecord:
        return VersionRecord(
            version_id=str(manifest["version_id"]),
            artifact_type=str(manifest["artifact_type"]),
            scope_type=str(manifest["scope_type"]),
            scope_key=str(manifest["scope_key"]),
            market=manifest.get("market"),
            identity=copy.deepcopy(manifest.get("identity")),
            artifact_uri=str(manifest["artifact_uri"]),
            artifact_sha256=str(manifest["artifact_sha256"]),
            artifact_size_bytes=(
                int(manifest["artifact_size_bytes"])
                if manifest.get("artifact_size_bytes") is not None
                else None
            ),
            metadata=copy.deepcopy(manifest.get("metadata", {})),
            dependencies=tuple(copy.deepcopy(manifest.get("dependencies", []))),
            parent_version_id=manifest.get("parent_version_id"),
            created_at=str(manifest["created_at"]),
            status=status,
            manifest_path=str(
                manifest.get("manifest_path")
                or self._manifest_path(
                    str(manifest["scope_key"]),
                    str(manifest["artifact_type"]),
                    str(manifest["version_id"]),
                )
            ),
        )

    def get_version(
        self,
        artifact_type: str,
        version_id: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
    ) -> VersionRecord:
        kind = _normalize_artifact_type(artifact_type)
        if scope_key is None:
            _, resolved_scope_key, _, _ = self._scope(
                kind,
                identity=identity,
                market=market,
                scope_type=scope_type,
            )
        else:
            resolved_scope_key = _safe_component(scope_key, "scope_key")

        path = self._manifest_path(resolved_scope_key, kind, version_id)
        manifest = _read_json(path)
        if int(manifest.get("registry_schema_version", -1)) != VERSION_REGISTRY_SCHEMA:
            raise VersionIntegrityError("Unsupported version registry schema.")
        current_status = self.get_status(
            kind,
            version_id,
            scope_key=resolved_scope_key,
        )
        record = self._record_from_manifest(manifest, current_status)
        return VersionRecord(
            **{
                **record.__dict__,
                "manifest_path": str(path),
            }
        )

    def list_versions(
        self,
        artifact_type: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[VersionRecord]:
        kind = _normalize_artifact_type(artifact_type)
        if scope_key is None:
            _, resolved_scope_key, _, _ = self._scope(
                kind,
                identity=identity,
                market=market,
                scope_type=scope_type,
            )
        else:
            resolved_scope_key = _safe_component(scope_key, "scope_key")

        expected_status = str(status).upper() if status is not None else None
        if expected_status is not None and expected_status not in VERSION_STATUSES:
            raise VersionValidationError("Unknown status filter.")

        directory = self._manifest_directory(resolved_scope_key, kind)
        records: List[VersionRecord] = []
        if not directory.is_dir():
            return records
        for path in sorted(directory.glob("*.json")):
            manifest = _read_json(path)
            version_id = str(manifest.get("version_id", path.stem))
            current_status = self.get_status(kind, version_id, scope_key=resolved_scope_key)
            if expected_status is not None and current_status != expected_status:
                continue
            record = self._record_from_manifest(manifest, current_status)
            records.append(
                VersionRecord(
                    **{
                        **record.__dict__,
                        "manifest_path": str(path),
                    }
                )
            )
        records.sort(key=lambda item: (item.created_at, item.version_id))
        return records

    def verify_artifact(
        self,
        artifact_type: str,
        version_id: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = self.get_version(
            artifact_type,
            version_id,
            identity=identity,
            market=market,
            scope_type=scope_type,
            scope_key=scope_key,
        )
        path = Path(record.artifact_uri)
        actual_hash, actual_size = _sha256_file(path)
        valid = actual_hash == record.artifact_sha256 and (
            record.artifact_size_bytes is None or actual_size == record.artifact_size_bytes
        )
        if not valid:
            raise VersionIntegrityError(
                f"Artifact checksum mismatch for {record.artifact_type}:{record.version_id}."
            )
        return {
            "valid": True,
            "artifact_type": record.artifact_type,
            "version_id": record.version_id,
            "sha256": actual_hash,
            "size_bytes": actual_size,
            "artifact_uri": record.artifact_uri,
        }

    def validate_version(
        self,
        artifact_type: str,
        version_id: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        validation_metadata: Optional[Mapping[str, Any]] = None,
    ) -> VersionRecord:
        record = self.get_version(
            artifact_type,
            version_id,
            identity=identity,
            market=market,
            scope_type=scope_type,
            scope_key=scope_key,
        )
        if record.status == STATUS_VALIDATED or record.status in {
            STATUS_APPROVED,
            STATUS_DEPRECATED,
            STATUS_RETIRED,
        }:
            self.verify_artifact(
                record.artifact_type,
                record.version_id,
                scope_key=record.scope_key,
            )
            return record
        if record.status != STATUS_DRAFT:
            raise VersionStateError(f"Cannot validate a version in state {record.status}.")

        self.verify_artifact(
            record.artifact_type,
            record.version_id,
            scope_key=record.scope_key,
        )
        dependency_reports = self._check_dependencies(record, minimum_states={
            STATUS_VALIDATED,
            STATUS_APPROVED,
            STATUS_DEPRECATED,
        })
        details = _to_json_value(dict(validation_metadata or {}))
        self._append_transition_event(
            record.scope_key,
            record.artifact_type,
            record.version_id,
            STATUS_VALIDATED,
            expected_statuses={STATUS_DRAFT},
            event_type="VALIDATED",
            reason="Artifact integrity and dependencies validated.",
            extra={
                "validation_metadata": details,
                "dependency_reports": dependency_reports,
            },
        )
        return self.get_version(
            record.artifact_type,
            record.version_id,
            scope_key=record.scope_key,
        )

    def _check_dependencies(
        self,
        record: VersionRecord,
        *,
        minimum_states: Iterable[str],
    ) -> List[Dict[str, Any]]:
        allowed_states = set(minimum_states)
        reports: List[Dict[str, Any]] = []
        for dependency in record.dependencies:
            dependency_kind = str(dependency["artifact_type"])
            dependency_version = str(dependency["version_id"])
            dependency_scope_key = str(dependency["scope_key"])
            try:
                dependency_record = self.get_version(
                    dependency_kind,
                    dependency_version,
                    scope_key=dependency_scope_key,
                )
            except VersionNotFoundError as error:
                raise VersionValidationError(
                    f"Missing dependency {dependency_kind}:{dependency_version}."
                ) from error
            if dependency_record.status not in allowed_states:
                raise VersionValidationError(
                    f"Dependency {dependency_kind}:{dependency_version} is in state "
                    f"{dependency_record.status}; expected a validated or approved version."
                )
            self.verify_artifact(
                dependency_kind,
                dependency_version,
                scope_key=dependency_scope_key,
            )
            reports.append(
                {
                    "artifact_type": dependency_kind,
                    "version_id": dependency_version,
                    "status": dependency_record.status,
                    "sha256": dependency_record.artifact_sha256,
                }
            )
        return reports

    def approve_version(
        self,
        artifact_type: str,
        version_id: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> VersionRecord:
        record = self.get_version(
            artifact_type,
            version_id,
            identity=identity,
            market=market,
            scope_type=scope_type,
            scope_key=scope_key,
        )
        if record.status == STATUS_APPROVED:
            return record
        if record.status != STATUS_VALIDATED:
            raise VersionStateError("Only a validated version can be approved.")

        self.verify_artifact(record.artifact_type, record.version_id, scope_key=record.scope_key)
        dependency_reports = self._check_dependencies(
            record,
            minimum_states={STATUS_VALIDATED, STATUS_APPROVED, STATUS_DEPRECATED},
        )
        self._append_transition_event(
            record.scope_key,
            record.artifact_type,
            record.version_id,
            STATUS_APPROVED,
            expected_statuses={STATUS_VALIDATED},
            event_type="APPROVED",
            actor=_safe_component(actor, "actor") if actor else None,
            reason=str(reason or "Version approved.").strip(),
            extra={"dependency_reports": dependency_reports},
        )
        return self.get_version(record.artifact_type, record.version_id, scope_key=record.scope_key)

    def _transition(
        self,
        artifact_type: str,
        version_id: str,
        target_status: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> VersionRecord:
        target = str(target_status).strip().upper()
        if target not in VERSION_STATUSES:
            raise VersionValidationError("Unknown target version status.")
        record = self.get_version(
            artifact_type,
            version_id,
            identity=identity,
            market=market,
            scope_type=scope_type,
            scope_key=scope_key,
        )
        if record.status == target:
            return record
        if target not in _STATUS_TRANSITIONS[record.status]:
            raise VersionStateError(
                f"Transition {record.status} -> {target} is not permitted."
            )
        self._append_transition_event(
            record.scope_key,
            record.artifact_type,
            record.version_id,
            target,
            expected_statuses={record.status},
            event_type=target,
            actor=_safe_component(actor, "actor") if actor else None,
            reason=str(reason or f"Version moved to {target}.").strip(),
        )
        return self.get_version(record.artifact_type, record.version_id, scope_key=record.scope_key)

    def reject_version(self, artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
        return self._transition(artifact_type, version_id, STATUS_REJECTED, **kwargs)

    def deprecate_version(self, artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
        return self._transition(artifact_type, version_id, STATUS_DEPRECATED, **kwargs)

    def retire_version(self, artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
        return self._transition(artifact_type, version_id, STATUS_RETIRED, **kwargs)

    def latest_approved(
        self,
        artifact_type: str,
        *,
        identity: Any = None,
        market: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
    ) -> Optional[VersionRecord]:
        records = self.list_versions(
            artifact_type,
            identity=identity,
            market=market,
            scope_type=scope_type,
            scope_key=scope_key,
            status=STATUS_APPROVED,
        )
        if not records:
            return None
        return records[-1]

    def register_pattern_catalog(
        self,
        catalog: Any,
        *,
        storage: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        validate: bool = True,
        approve: bool = False,
    ) -> VersionRecord:
        if callable(getattr(catalog, "to_dict", None)):
            payload = catalog.to_dict()
            catalog_market = getattr(catalog, "market", payload.get("market"))
            version_id = str(getattr(catalog, "catalog_version", payload.get("catalog_version", "")))
            checksum_method = getattr(catalog, "checksum", None)
            catalog_checksum = checksum_method() if callable(checksum_method) else None
        elif isinstance(catalog, Mapping):
            payload = dict(catalog)
            catalog_market = payload.get("market")
            version_id = str(payload.get("catalog_version", ""))
            catalog_checksum = None
        else:
            raise VersionValidationError("catalog must be a PatternCatalog or mapping.")

        if not catalog_market or not version_id:
            raise VersionValidationError("Catalog is missing market or catalog_version.")
        _validate_catalog_payload(payload, str(catalog_market), version_id)

        artifact_path: Optional[Path] = None
        if storage is not None:
            saver = getattr(storage, "save_pattern_catalog", None)
            if not callable(saver):
                raise VersionValidationError("storage must provide save_pattern_catalog().")
            write_result = saver(str(catalog_market), version_id, payload)
            artifact_path = Path(getattr(write_result, "path", write_result)).resolve()
        else:
            try:
                from engine.storage import save_pattern_catalog
                write_result = save_pattern_catalog(str(catalog_market), version_id, payload)
                artifact_path = Path(getattr(write_result, "path", write_result)).resolve()
            except (ImportError, ModuleNotFoundError):
                artifact_path = None

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "catalog_checksum": catalog_checksum,
                "catalog_schema_version": payload.get("catalog_schema_version"),
                "pattern_count": len(payload.get("patterns", [])),
                "category_count": len(payload.get("categories", [])),
            }
        )
        register_kwargs: Dict[str, Any] = {
            "metadata": merged_metadata,
            "dependencies": dependencies,
        }
        if artifact_path is not None and artifact_path.is_file():
            register_kwargs["artifact_path"] = artifact_path
        else:
            register_kwargs["payload"] = payload

        record = self.register_version(
            ARTIFACT_PATTERN_CATALOG,
            version_id,
            market=str(catalog_market),
            parent_version_id=payload.get("parent_version"),
            **register_kwargs,
        )
        if validate and record.status == STATUS_DRAFT:
            record = self.validate_version(
                ARTIFACT_PATTERN_CATALOG,
                version_id,
                market=str(catalog_market),
                validation_metadata={"catalog_structure_valid": True},
            )
        if approve and record.status == STATUS_VALIDATED:
            record = self.approve_version(
                ARTIFACT_PATTERN_CATALOG,
                version_id,
                market=str(catalog_market),
                reason="Catalog snapshot registered and validated.",
            )
        return record

    def register_dataset(
        self,
        dataset: Mapping[str, Any],
        *,
        identity: Any = None,
        artifact_path: Optional[Union[str, Path]] = None,
        version_id: Optional[str] = None,
        dependencies: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VersionRecord:
        if not isinstance(dataset, Mapping):
            raise VersionValidationError("dataset must be a mapping.")
        dataset_identity = dataset.get("identity")
        resolved_identity = _identity_mapping(identity if identity is not None else dataset_identity)
        if dataset_identity is not None and _identity_mapping(dataset_identity) != resolved_identity:
            raise VersionValidationError("Dataset identity does not match the requested TradingBot.")

        dataset_id = str(version_id or dataset.get("dataset_id") or "").strip()
        if not dataset_id:
            raise VersionValidationError("Dataset must include dataset_id or version_id.")
        dataset_metadata = dict(metadata or {})
        dataset_metadata.update(
            {
                "dataset_id": str(dataset.get("dataset_id", dataset_id)),
                "target": dataset.get("target"),
                "feature_schema_hash": (
                    dataset.get("feature_schema", {}).get("sha256")
                    if isinstance(dataset.get("feature_schema"), Mapping)
                    else dataset.get("feature_schema_hash")
                ),
                "feature_count": len(dataset.get("feature_names", [])),
                "sample_count": int(len(dataset.get("targets", []))) if dataset.get("targets") is not None else None,
                "schema_version": dataset.get("schema_version"),
                "model_target_schema": dataset.get("model_target_schema"),
            }
        )
        kwargs: Dict[str, Any] = {
            "identity": resolved_identity,
            "dependencies": dependencies,
            "metadata": dataset_metadata,
        }
        if artifact_path is not None:
            kwargs["artifact_path"] = artifact_path
        else:
            kwargs["payload"] = dataset
        return self.register_version(ARTIFACT_DATASET, dataset_id, **kwargs)

    def register_scaler(
        self,
        scaler: Any,
        *,
        identity: Any,
        version_id: Optional[str] = None,
        artifact_path: Optional[Union[str, Path]] = None,
        dependencies: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VersionRecord:
        if callable(getattr(scaler, "to_dict", None)):
            payload = scaler.to_dict()
        elif isinstance(scaler, Mapping):
            payload = dict(scaler)
        else:
            raise VersionValidationError("scaler must provide to_dict() or be a mapping.")

        scaler_version = str(version_id or _generated_version_id("scaler"))
        scaler_metadata = dict(metadata or {})
        scaler_metadata.update(
            {
                "feature_count": len(payload.get("mean", [])),
                "feature_schema_hash": payload.get("feature_schema_hash"),
                "method": payload.get("method"),
                "fit_dataset_id": payload.get("fit_dataset_id"),
                "fit_sample_count": payload.get("fit_sample_count"),
                "fit_scope": payload.get("fit_scope"),
            }
        )
        kwargs: Dict[str, Any] = {
            "identity": identity,
            "dependencies": dependencies,
            "metadata": scaler_metadata,
        }
        if artifact_path is not None:
            kwargs["artifact_path"] = artifact_path
        else:
            kwargs["payload"] = payload
        return self.register_version(ARTIFACT_SCALER, scaler_version, **kwargs)

    def register_model(
        self,
        artifact_path: Union[str, Path],
        *,
        identity: Any,
        version_id: Optional[str] = None,
        dependencies: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VersionRecord:
        model_version = str(version_id or _generated_version_id("model"))
        return self.register_version(
            ARTIFACT_MODEL,
            model_version,
            artifact_path=artifact_path,
            identity=identity,
            dependencies=dependencies,
            metadata=metadata,
        )


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def _generated_version_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


_DEFAULT_REGISTRY: Optional[VersionRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def get_default_registry() -> VersionRegistry:
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = VersionRegistry()
        return _DEFAULT_REGISTRY


def register_version(artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
    return get_default_registry().register_version(artifact_type, version_id, **kwargs)


def get_version(artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
    return get_default_registry().get_version(artifact_type, version_id, **kwargs)


def list_versions(artifact_type: str, **kwargs: Any) -> List[VersionRecord]:
    return get_default_registry().list_versions(artifact_type, **kwargs)


def validate_version(artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
    return get_default_registry().validate_version(artifact_type, version_id, **kwargs)


def approve_version(artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
    return get_default_registry().approve_version(artifact_type, version_id, **kwargs)


def deprecate_version(artifact_type: str, version_id: str, **kwargs: Any) -> VersionRecord:
    return get_default_registry().deprecate_version(artifact_type, version_id, **kwargs)


def latest_approved(artifact_type: str, **kwargs: Any) -> Optional[VersionRecord]:
    return get_default_registry().latest_approved(artifact_type, **kwargs)


def register_pattern_catalog(catalog: Any, **kwargs: Any) -> VersionRecord:
    return get_default_registry().register_pattern_catalog(catalog, **kwargs)


def register_dataset(dataset: Mapping[str, Any], **kwargs: Any) -> VersionRecord:
    return get_default_registry().register_dataset(dataset, **kwargs)


def register_scaler(scaler: Any, **kwargs: Any) -> VersionRecord:
    return get_default_registry().register_scaler(scaler, **kwargs)


def register_model(artifact_path: Union[str, Path], **kwargs: Any) -> VersionRecord:
    return get_default_registry().register_model(artifact_path, **kwargs)


def verify_artifact(artifact_type: str, version_id: str, **kwargs: Any) -> Dict[str, Any]:
    return get_default_registry().verify_artifact(artifact_type, version_id, **kwargs)


def validate_version_status(value: Any) -> str:
    normalized = str(value).strip().upper()
    if normalized not in VERSION_STATUSES:
        raise VersionValidationError("Unknown version status.")
    return normalized


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "VERSION_REGISTRY_SCHEMA",
    "VERSION_EVENT_SCHEMA",
    "ARTIFACT_PATTERN_CATALOG",
    "ARTIFACT_DATASET",
    "ARTIFACT_SCALER",
    "ARTIFACT_MODEL",
    "ARTIFACT_TRAINING_RUN",
    "ARTIFACT_EVALUATION",
    "ARTIFACT_TYPES",
    "SCOPE_MARKET",
    "SCOPE_CUSTOMER",
    "SCOPE_GLOBAL",
    "STATUS_DRAFT",
    "STATUS_VALIDATED",
    "STATUS_APPROVED",
    "STATUS_DEPRECATED",
    "STATUS_RETIRED",
    "STATUS_REJECTED",
    "VERSION_STATUSES",
    "VersionError",
    "VersionValidationError",
    "VersionConflictError",
    "VersionNotFoundError",
    "VersionStateError",
    "VersionIntegrityError",
    "VersionDependency",
    "VersionRecord",
    "VersionRegistry",
    "get_default_registry",
    "register_version",
    "get_version",
    "list_versions",
    "validate_version",
    "approve_version",
    "deprecate_version",
    "latest_approved",
    "register_pattern_catalog",
    "register_dataset",
    "register_scaler",
    "register_model",
    "verify_artifact",
    "validate_version_status",
]
