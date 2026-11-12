"""Versioned numerical pattern catalog for the Spectline AI engine."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from client import normalize_market
    from engine.settings import (
        PATTERN_CATEGORY_ID_TO_NAME,
        PATTERN_CATEGORY_NAME_TO_ID,
        PATTERN_ID_DIGITS,
        PATTERN_ID_STORAGE_TYPE,
    )
except ImportError:
    from ..client import normalize_market
    from .settings import (
        PATTERN_CATEGORY_ID_TO_NAME,
        PATTERN_CATEGORY_NAME_TO_ID,
        PATTERN_ID_DIGITS,
        PATTERN_ID_STORAGE_TYPE,
    )


# ============================================================================
# MODULE CONFIGURATION
# ============================================================================

CATALOG_SCHEMA_VERSION = 1
INITIAL_CATALOG_VERSION = "1"
PATTERN_ID_SLOT_STEP = 10
PATTERN_ID_MIN_SLOT = 10
PATTERN_ID_MAX_SLOT = 990

PATTERN_STATUSES = frozenset(
    {
        "CANDIDATE",
        "ACTIVE",
        "DEPRECATED",
        "RETIRED",
        "REJECTED",
    }
)

_ALLOWED_STATUS_TRANSITIONS = {
    "CANDIDATE": frozenset({"ACTIVE", "REJECTED"}),
    "ACTIVE": frozenset({"DEPRECATED"}),
    "DEPRECATED": frozenset({"RETIRED"}),
    "RETIRED": frozenset(),
    "REJECTED": frozenset(),
}

_PATTERN_ID_RE = re.compile(r"^[0-9]{4}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*$")
_MAX_LABEL_LENGTH = 256


# ============================================================================
# ERRORS AND DATA TYPES
# ============================================================================

class CatalogError(RuntimeError):
    """Base exception for pattern catalog operations."""


class CatalogValidationError(CatalogError, ValueError):
    """Raised when a catalog definition or identifier is invalid."""


class CatalogNotFoundError(CatalogError, LookupError):
    """Raised when a requested pattern is not present in the catalog."""


class CatalogStateError(CatalogError):
    """Raised when a requested catalog state transition is not allowed."""


@dataclass(frozen=True)
class PatternCategory:
    """Stable top-level family encoded by the first digit of a pattern ID."""

    category_id: int
    category_code: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternDefinition:
    """Stable meaning and numerical definition of one concrete pattern ID."""

    pattern_id: str
    category_id: int
    category_name: str
    status: str = "CANDIDATE"
    name: Optional[str] = None
    description: Optional[str] = None
    numeric_definition: Mapping[str, Any] = field(default_factory=dict)
    created_at: int = 0

    @property
    def is_assignable(self) -> bool:
        """Return whether this pattern can be assigned to new occurrences."""
        return self.status == "ACTIVE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "status": self.status,
            "name": self.name,
            "description": self.description,
            "numeric_definition": copy.deepcopy(dict(self.numeric_definition)),
            "created_at": self.created_at,
        }


# ============================================================================
# CATEGORY AND IDENTIFIER HELPERS
# ============================================================================

def _normalize_market_name(market: str) -> str:
    """Normalize and validate a market using the application convention."""
    try:
        normalized = str(normalize_market(market)).strip()
    except Exception as error:
        raise CatalogValidationError(f"Invalid market: {market!r}.") from error
    if not normalized:
        raise CatalogValidationError("Market must not be empty.")
    return normalized


def list_categories() -> Tuple[PatternCategory, ...]:
    """Return the ten stable pattern categories in numeric order."""
    return tuple(
        PatternCategory(
            category_id=int(category_id),
            category_code=f"{int(category_id)}000",
            name=str(category_name),
        )
        for category_id, category_name in sorted(
            PATTERN_CATEGORY_ID_TO_NAME.items()
        )
    )


def get_category(category_id: int) -> PatternCategory:
    """Return one category by its first-digit category ID."""
    try:
        normalized_id = int(category_id)
    except (TypeError, ValueError, OverflowError) as error:
        raise CatalogValidationError("category_id must be an integer from 0 to 9.") from error

    name = PATTERN_CATEGORY_ID_TO_NAME.get(normalized_id)
    if name is None:
        raise CatalogValidationError(f"Unknown pattern category: {normalized_id}.")

    return PatternCategory(
        category_id=normalized_id,
        category_code=f"{normalized_id}000",
        name=str(name),
    )


def validate_pattern_id(
    pattern_id: Any,
    *,
    allow_category_code: bool = False,
) -> str:
    """Validate a four-digit categorical pattern identifier.

    Category codes end in 000 and are accepted only when explicitly requested.
    Concrete pattern IDs use the reserved ten-step slots X010, X020, ..., X990.
    """
    normalized = str(pattern_id).strip()
    if not _PATTERN_ID_RE.fullmatch(normalized):
        raise CatalogValidationError("pattern_id must contain exactly four digits.")

    category_id = int(normalized[0])
    if category_id not in PATTERN_CATEGORY_ID_TO_NAME:
        raise CatalogValidationError("pattern_id has an unknown category prefix.")

    slot = int(normalized[1:])
    if slot == 0:
        if allow_category_code:
            return normalized
        raise CatalogValidationError("Category codes cannot identify concrete patterns.")

    if slot < PATTERN_ID_MIN_SLOT or slot > PATTERN_ID_MAX_SLOT:
        raise CatalogValidationError("Pattern ID slot is outside the reserved range.")

    if slot % PATTERN_ID_SLOT_STEP != 0:
        raise CatalogValidationError(
            "Concrete pattern IDs must use ten-step slots such as X010 or X020."
        )

    return normalized


def build_pattern_id(category_id: int, slot: int) -> str:
    """Build a concrete pattern ID from a category and a reserved ten-step slot."""
    category = get_category(category_id)
    try:
        normalized_slot = int(slot)
    except (TypeError, ValueError, OverflowError) as error:
        raise CatalogValidationError("Pattern slot must be an integer.") from error

    if (
        normalized_slot < PATTERN_ID_MIN_SLOT
        or normalized_slot > PATTERN_ID_MAX_SLOT
        or normalized_slot % PATTERN_ID_SLOT_STEP != 0
    ):
        raise CatalogValidationError(
            "Pattern slot must be one of 10, 20, ..., 990."
        )

    return f"{category.category_id}{normalized_slot:03d}"


def category_for_pattern_id(pattern_id: Any) -> PatternCategory:
    """Resolve a concrete pattern ID to its stable top-level category."""
    normalized = validate_pattern_id(pattern_id, allow_category_code=True)
    return get_category(int(normalized[0]))


def is_category_code(pattern_id: Any) -> bool:
    """Return whether a valid four-digit code identifies a category header."""
    normalized = validate_pattern_id(pattern_id, allow_category_code=True)
    return normalized.endswith("000")


# ============================================================================
# NUMERICAL DEFINITION VALIDATION
# ============================================================================

def _normalize_numeric_tree(value: Any, path: str = "numeric_definition") -> Any:
    """Validate and copy a nested structure containing numeric values only."""
    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise CatalogValidationError(f"{path} contains a non-finite number.")
        if isinstance(value, int):
            return int(value)
        return number

    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                raise CatalogValidationError(f"{path} contains an empty feature name.")
            normalized[normalized_key] = _normalize_numeric_tree(
                item,
                f"{path}.{normalized_key}",
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_numeric_tree(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    raise CatalogValidationError(
        f"{path} must contain only numeric values, mappings and sequences."
    )


def _optional_label(value: Any, field_name: str) -> Optional[str]:
    """Normalize an optional human-readable catalog label."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_LABEL_LENGTH:
        raise CatalogValidationError(
            f"{field_name} cannot exceed {_MAX_LABEL_LENGTH} characters."
        )
    return normalized


def _timestamp(value: Any, field_name: str = "created_at") -> int:
    """Validate a non-negative Unix timestamp in seconds."""
    if isinstance(value, bool):
        raise CatalogValidationError(f"{field_name} must be a Unix timestamp.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CatalogValidationError(f"{field_name} must be a Unix timestamp.") from error
    if normalized < 0:
        raise CatalogValidationError(f"{field_name} cannot be negative.")
    return normalized


def _current_timestamp() -> int:
    """Return the current UTC time as Unix seconds."""
    return int(datetime.now(timezone.utc).timestamp())


def _normalize_pattern_definition(
    raw: Mapping[str, Any],
) -> PatternDefinition:
    """Validate one persisted pattern definition against the category registry."""
    if not isinstance(raw, Mapping):
        raise CatalogValidationError("Each pattern definition must be a mapping.")

    pattern_id = validate_pattern_id(raw.get("pattern_id"))
    category = category_for_pattern_id(pattern_id)

    try:
        declared_category_id = int(raw.get("category_id", category.category_id))
    except (TypeError, ValueError, OverflowError) as error:
        raise CatalogValidationError("Pattern category_id must be an integer.") from error

    if declared_category_id != category.category_id:
        raise CatalogValidationError(
            f"Pattern {pattern_id} does not belong to category {declared_category_id}."
        )

    declared_category_name = str(raw.get("category_name", category.name)).strip()
    if declared_category_name != category.name:
        raise CatalogValidationError(
            f"Pattern {pattern_id} has a category name inconsistent with settings.py."
        )

    status = str(raw.get("status", "CANDIDATE")).strip().upper()
    if status not in PATTERN_STATUSES:
        raise CatalogValidationError(f"Unknown pattern status: {status}.")

    numeric_definition = _normalize_numeric_tree(
        raw.get("numeric_definition", {}),
    )
    if not isinstance(numeric_definition, dict):
        raise CatalogValidationError("numeric_definition must be a mapping.")

    return PatternDefinition(
        pattern_id=pattern_id,
        category_id=category.category_id,
        category_name=category.name,
        status=status,
        name=_optional_label(raw.get("name"), "name"),
        description=_optional_label(raw.get("description"), "description"),
        numeric_definition=numeric_definition,
        created_at=_timestamp(raw.get("created_at", 0)),
    )


# ============================================================================
# PATTERN CATALOG
# ============================================================================

class PatternCatalog:
    """Manage one immutable-version snapshot of a market pattern registry.

    A catalog instance is an in-memory working snapshot. Persisted snapshots are
    immutable. Changes should be made on a fork with a new catalog version and
    then saved as a new snapshot.
    """

    def __init__(
        self,
        market: str,
        catalog_version: str = INITIAL_CATALOG_VERSION,
        *,
        parent_version: Optional[str] = None,
        patterns: Optional[Iterable[Mapping[str, Any] | PatternDefinition]] = None,
        created_at: Optional[int] = None,
    ) -> None:
        self.market = _normalize_market_name(market)
        self.catalog_version = str(catalog_version).strip()
        if not _VERSION_RE.fullmatch(self.catalog_version):
            raise CatalogValidationError(
                "catalog_version must be a positive integer string."
            )

        self.parent_version = (
            str(parent_version).strip() if parent_version is not None else None
        )
        if self.parent_version is not None and not _VERSION_RE.fullmatch(self.parent_version):
            raise CatalogValidationError("parent_version must be a positive integer string.")

        self.created_at = (
            _current_timestamp() if created_at is None else _timestamp(created_at)
        )
        self._categories = {item.category_id: item for item in list_categories()}
        self._patterns: Dict[str, PatternDefinition] = {}

        for item in patterns or ():
            raw = item.to_dict() if isinstance(item, PatternDefinition) else item
            definition = _normalize_pattern_definition(raw)
            if definition.pattern_id in self._patterns:
                raise CatalogValidationError(
                    f"Duplicate pattern ID in catalog: {definition.pattern_id}."
                )
            self._patterns[definition.pattern_id] = definition

    @property
    def categories(self) -> Tuple[PatternCategory, ...]:
        """Return all stable categories in numeric order."""
        return tuple(self._categories[key] for key in sorted(self._categories))

    @property
    def patterns(self) -> Tuple[PatternDefinition, ...]:
        """Return all concrete patterns in stable ID order, including candidates."""
        return tuple(self._patterns[key] for key in sorted(self._patterns))

    @property
    def active_patterns(self) -> Tuple[PatternDefinition, ...]:
        """Return only patterns currently approved for occurrence assignment."""
        return tuple(
            item for item in self.patterns if item.status == "ACTIVE"
        )

    def __len__(self) -> int:
        return len(self._patterns)

    def get_pattern(self, pattern_id: Any) -> Optional[PatternDefinition]:
        """Return a pattern definition by its exact categorical ID."""
        normalized = validate_pattern_id(pattern_id)
        return self._patterns.get(normalized)

    def require_pattern(self, pattern_id: Any) -> PatternDefinition:
        """Return a pattern definition or raise when it is not registered."""
        normalized = validate_pattern_id(pattern_id)
        definition = self._patterns.get(normalized)
        if definition is None:
            raise CatalogNotFoundError(f"Pattern {normalized} is not registered.")
        return definition

    def require_active_pattern(self, pattern_id: Any) -> PatternDefinition:
        """Return a pattern approved for new occurrence assignments."""
        definition = self.require_pattern(pattern_id)
        if not definition.is_assignable:
            raise CatalogStateError(
                f"Pattern {definition.pattern_id} is not active; status is {definition.status}."
            )
        return definition

    def patterns_for_category(
        self,
        category_id: int,
        *,
        active_only: bool = False,
    ) -> Tuple[PatternDefinition, ...]:
        """Return registered patterns belonging to one category."""
        category = get_category(category_id)
        return tuple(
            item
            for item in self.patterns
            if item.category_id == category.category_id
            and (not active_only or item.status == "ACTIVE")
        )

    def next_pattern_id(self, category_id: int) -> str:
        """Find the first unused reserved ID without recycling older identifiers."""
        category = get_category(category_id)
        used_slots = {
            int(pattern_id[1:])
            for pattern_id in self._patterns
            if int(pattern_id[0]) == category.category_id
        }

        for slot in range(
            PATTERN_ID_MIN_SLOT,
            PATTERN_ID_MAX_SLOT + 1,
            PATTERN_ID_SLOT_STEP,
        ):
            if slot not in used_slots:
                return build_pattern_id(category.category_id, slot)

        raise CatalogStateError(
            f"No reserved pattern IDs remain in category {category.name}."
        )

    def add_candidate(
        self,
        category_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        numeric_definition: Optional[Mapping[str, Any]] = None,
        created_at: Optional[int] = None,
    ) -> PatternDefinition:
        """Register a new candidate without making it assignable to occurrences."""
        pattern_id = self.next_pattern_id(category_id)
        category = get_category(category_id)
        definition = PatternDefinition(
            pattern_id=pattern_id,
            category_id=category.category_id,
            category_name=category.name,
            status="CANDIDATE",
            name=_optional_label(name, "name"),
            description=_optional_label(description, "description"),
            numeric_definition=_normalize_numeric_tree(
                numeric_definition if numeric_definition is not None else {}
            ),
            created_at=(
                _current_timestamp() if created_at is None else _timestamp(created_at)
            ),
        )
        if not isinstance(definition.numeric_definition, dict):
            raise CatalogValidationError("numeric_definition must be a mapping.")
        self._patterns[pattern_id] = definition
        return definition

    def set_status(self, pattern_id: Any, new_status: str) -> PatternDefinition:
        """Apply an allowed lifecycle transition without changing pattern meaning."""
        normalized_id = validate_pattern_id(pattern_id)
        current = self.require_pattern(normalized_id)
        target_status = str(new_status).strip().upper()

        if target_status not in PATTERN_STATUSES:
            raise CatalogValidationError(f"Unknown pattern status: {target_status}.")
        if target_status == current.status:
            return current
        if target_status not in _ALLOWED_STATUS_TRANSITIONS[current.status]:
            raise CatalogStateError(
                f"Invalid pattern status transition: {current.status} -> {target_status}."
            )

        updated = PatternDefinition(
            pattern_id=current.pattern_id,
            category_id=current.category_id,
            category_name=current.category_name,
            status=target_status,
            name=current.name,
            description=current.description,
            numeric_definition=copy.deepcopy(dict(current.numeric_definition)),
            created_at=current.created_at,
        )
        self._patterns[normalized_id] = updated
        return updated

    def approve_candidate(self, pattern_id: Any) -> PatternDefinition:
        """Approve a reviewed candidate in this new catalog snapshot."""
        current = self.require_pattern(pattern_id)
        if current.status != "CANDIDATE":
            raise CatalogStateError(
                f"Pattern {current.pattern_id} is not awaiting approval."
            )
        return self.set_status(current.pattern_id, "ACTIVE")

    def reject_candidate(self, pattern_id: Any) -> PatternDefinition:
        """Reject a candidate while permanently reserving its identifier."""
        current = self.require_pattern(pattern_id)
        if current.status != "CANDIDATE":
            raise CatalogStateError(
                f"Pattern {current.pattern_id} is not awaiting review."
            )
        return self.set_status(current.pattern_id, "REJECTED")

    def fork(self, catalog_version: Optional[str] = None) -> "PatternCatalog":
        """Create a new editable snapshot while retaining all existing IDs."""
        if catalog_version is None:
            next_version = str(int(self.catalog_version) + 1)
        else:
            next_version = str(catalog_version).strip()

        if not _VERSION_RE.fullmatch(next_version):
            raise CatalogValidationError(
                "catalog_version must be a positive integer string."
            )
        if int(next_version) <= int(self.catalog_version):
            raise CatalogValidationError(
                "A fork must use a catalog version greater than its parent."
            )

        return PatternCatalog(
            market=self.market,
            catalog_version=next_version,
            parent_version=self.catalog_version,
            patterns=[item.to_dict() for item in self.patterns],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe immutable-version catalog snapshot."""
        return {
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "catalog_version": self.catalog_version,
            "parent_version": self.parent_version,
            "market": self.market,
            "created_at": self.created_at,
            "pattern_id_digits": PATTERN_ID_DIGITS,
            "pattern_id_storage_type": PATTERN_ID_STORAGE_TYPE,
            "pattern_id_encoding": "categorical",
            "categories": [item.to_dict() for item in self.categories],
            "patterns": [item.to_dict() for item in self.patterns],
        }

    def checksum(self) -> str:
        """Return a deterministic SHA-256 digest of this catalog snapshot."""
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, storage: Any = None) -> Any:
        """Persist this catalog as an immutable market-wide version snapshot."""
        if storage is None:
            try:
                from engine.storage import save_pattern_catalog
            except ImportError:
                from .storage import save_pattern_catalog
            return save_pattern_catalog(
                self.market,
                self.catalog_version,
                self.to_dict(),
            )

        saver = getattr(storage, "save_pattern_catalog", None)
        if not callable(saver):
            raise TypeError("storage must provide save_pattern_catalog().")
        return saver(self.market, self.catalog_version, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatternCatalog":
        """Create a validated catalog snapshot from a stored mapping."""
        if not isinstance(payload, Mapping):
            raise CatalogValidationError("Catalog payload must be a mapping.")

        schema_version = int(payload.get("catalog_schema_version", CATALOG_SCHEMA_VERSION))
        if schema_version != CATALOG_SCHEMA_VERSION:
            raise CatalogValidationError(
                f"Unsupported catalog schema version: {schema_version}."
            )

        raw_categories = payload.get("categories")
        if raw_categories is not None:
            expected = [item.to_dict() for item in list_categories()]
            normalized_categories = [dict(item) for item in raw_categories]
            if normalized_categories != expected:
                raise CatalogValidationError(
                    "Stored pattern categories differ from the stable category registry."
                )

        raw_patterns = payload.get("patterns", [])
        if not isinstance(raw_patterns, Sequence) or isinstance(raw_patterns, (str, bytes)):
            raise CatalogValidationError("Catalog patterns must be a sequence.")

        return cls(
            market=str(payload.get("market", "")),
            catalog_version=str(payload.get("catalog_version", INITIAL_CATALOG_VERSION)),
            parent_version=payload.get("parent_version"),
            patterns=raw_patterns,
            created_at=payload.get("created_at", _current_timestamp()),
        )

    @classmethod
    def load(
        cls,
        market: str,
        catalog_version: str,
        storage: Any = None,
    ) -> Optional["PatternCatalog"]:
        """Load and validate one immutable catalog version."""
        if storage is None:
            try:
                from engine.storage import load_pattern_catalog
            except ImportError:
                from .storage import load_pattern_catalog
            payload = load_pattern_catalog(market, catalog_version)
        else:
            loader = getattr(storage, "load_pattern_catalog", None)
            if not callable(loader):
                raise TypeError("storage must provide load_pattern_catalog().")
            payload = loader(market, catalog_version)

        if payload is None:
            return None
        catalog = cls.from_dict(payload)
        if catalog.market != _normalize_market_name(market):
            raise CatalogValidationError("Loaded catalog market does not match request.")
        if catalog.catalog_version != str(catalog_version):
            raise CatalogValidationError("Loaded catalog version does not match request.")
        return catalog


# ============================================================================
# PUBLIC CATALOG HELPERS
# ============================================================================

def create_catalog(
    market: str,
    catalog_version: str = INITIAL_CATALOG_VERSION,
) -> PatternCatalog:
    """Create an empty catalog containing the stable ten-category registry."""
    return PatternCatalog(market=market, catalog_version=catalog_version)


def load_catalog(
    market: str,
    catalog_version: str,
    storage: Any = None,
) -> Optional[PatternCatalog]:
    """Load one catalog version using the default or supplied storage layer."""
    return PatternCatalog.load(market, catalog_version, storage=storage)


def save_catalog(catalog: PatternCatalog, storage: Any = None) -> Any:
    """Save a catalog snapshot using the default or supplied storage layer."""
    if not isinstance(catalog, PatternCatalog):
        raise TypeError("catalog must be a PatternCatalog instance.")
    return catalog.save(storage=storage)


def validate_catalog(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a serialized catalog and return a concise integrity report."""
    catalog = PatternCatalog.from_dict(payload)
    active_count = sum(1 for item in catalog.patterns if item.status == "ACTIVE")
    candidate_count = sum(1 for item in catalog.patterns if item.status == "CANDIDATE")
    return {
        "valid": True,
        "market": catalog.market,
        "catalog_version": catalog.catalog_version,
        "category_count": len(catalog.categories),
        "pattern_count": len(catalog.patterns),
        "active_pattern_count": active_count,
        "candidate_pattern_count": candidate_count,
        "checksum": catalog.checksum(),
    }


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "INITIAL_CATALOG_VERSION",
    "PATTERN_ID_SLOT_STEP",
    "PATTERN_ID_MIN_SLOT",
    "PATTERN_ID_MAX_SLOT",
    "PATTERN_STATUSES",
    "CatalogError",
    "CatalogValidationError",
    "CatalogNotFoundError",
    "CatalogStateError",
    "PatternCategory",
    "PatternDefinition",
    "PatternCatalog",
    "list_categories",
    "get_category",
    "validate_pattern_id",
    "build_pattern_id",
    "category_for_pattern_id",
    "is_category_code",
    "create_catalog",
    "load_catalog",
    "save_catalog",
    "validate_catalog",
]
