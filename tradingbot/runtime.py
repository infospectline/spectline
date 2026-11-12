"""Identity-isolated runtime state management for Spectline TradingBots."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, cast

try:
    from client import ClientIdentity, normalize_timeframe
except ImportError:
    from .client import ClientIdentity, normalize_timeframe


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_TIMEFRAME = "H1"
DEFAULT_STARTING_BALANCE = 100_000.0
DEFAULT_LAST_MODEL_ENTRY_INDEX = -10_000_000
DEFAULT_RUNTIME_STATUS = "idle"
MAX_STARTUP_PROGRESS = 100


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Return a finite float or the supplied default."""
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _timestamp_seconds(value: Any) -> Optional[int]:
    """Normalize supported numeric timestamp values to integer seconds."""
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    return int(number)


def _candle_timestamp(candle: Any) -> Optional[int]:
    """Read a candle timestamp from a mapping or an object."""
    if isinstance(candle, Mapping):
        value = candle.get("time", candle.get("timestamp"))
    else:
        value = getattr(candle, "time", getattr(candle, "timestamp", None))
    return _timestamp_seconds(value)


def _copy_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    """Validate and copy a mapping without retaining caller-owned containers."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    return copy.deepcopy(dict(value))


def _validate_identity_match(
    candidate: Mapping[str, Any],
    identity: ClientIdentity,
) -> None:
    """Reject candidate metadata that explicitly belongs to another identity."""
    expected = identity.as_dict()
    aliases = {
        "user_id": ("user_id", "customer_id", "customer"),
        "market": ("market", "symbol"),
        "strategy_id": ("strategy_id", "strategy"),
        "tradingbot_id": ("tradingbot_id", "tradingbot", "bot_id"),
    }
    for canonical_name, names in aliases.items():
        supplied = next(
            (candidate[name] for name in names if name in candidate and candidate[name] is not None),
            None,
        )
        if supplied is None:
            continue
        if str(supplied).strip() != str(expected[canonical_name]).strip():
            raise ValueError(
                f"Candidate {canonical_name} does not match the runtime identity."
            )


def _normalize_candle_sequence(
    candles: Sequence[Any],
    *,
    require_chronological: bool,
) -> List[Any]:
    """Copy a candle sequence and optionally validate chronological ordering."""
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise TypeError("candles must be a sequence.")

    result = list(candles)
    if not require_chronological:
        return result

    previous_timestamp: Optional[int] = None
    for index, candle in enumerate(result):
        timestamp = _candle_timestamp(candle)
        if timestamp is None:
            raise ValueError(f"Candle at index {index} has no valid timestamp.")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Candle timestamps must be strictly increasing.")
        previous_timestamp = timestamp
    return result


# ============================================================================
# TIMEFRAME SESSION STATE
# ============================================================================

@dataclass
class TradingSessionState:
    """Trading, analysis and replay state for one TradingBot timeframe."""

    current_index: int = 0
    running: bool = False
    trading_enabled: bool = True

    balance: float = DEFAULT_STARTING_BALANCE
    peak_equity: float = DEFAULT_STARTING_BALANCE

    open_trade: Optional[Dict[str, Any]] = None
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    last_trade_result: Optional[str] = None

    major_levels: List[Dict[str, Any]] = field(default_factory=list)
    cached_strategy_levels: List[Dict[str, Any]] = field(default_factory=list)
    last_level_calculation_index: int = -1
    blocked_trade_zone: Optional[Dict[str, Any]] = None
    zigzag_filter: Optional[Dict[str, Any]] = None
    blocked_trades: int = 0
    blocked_trade_signal: bool = False

    analysis_status: str = DEFAULT_RUNTIME_STATUS
    analysis_started: bool = False
    analysis_completed: bool = False
    analysis_phase: str = DEFAULT_RUNTIME_STATUS

    model_available: bool = False
    model_info: Dict[str, Any] = field(default_factory=dict)
    last_training_info: Dict[str, Any] = field(default_factory=dict)
    last_model_signal: Optional[Dict[str, Any]] = None
    last_model_entry_index: int = DEFAULT_LAST_MODEL_ENTRY_INDEX

    pending_setup: Optional[Dict[str, Any]] = None
    last_prediction: Optional[Dict[str, Any]] = None
    prediction_history: List[Dict[str, Any]] = field(default_factory=list)
    last_prediction_time: Optional[int] = None

    def reset(self, starting_balance: float = DEFAULT_STARTING_BALANCE) -> None:
        """Reset session trading and replay state while preserving model metadata."""
        starting = _finite_float(starting_balance)
        if starting is None or starting <= 0:
            raise ValueError("starting_balance must be a positive finite number.")

        self.current_index = 0
        self.running = False
        self.trading_enabled = True
        self.balance = float(starting)
        self.peak_equity = float(starting)
        self.open_trade = None
        self.trade_history = []
        self.last_trade_result = None
        self.major_levels = []
        self.cached_strategy_levels = []
        self.last_level_calculation_index = -1
        self.blocked_trade_zone = None
        self.zigzag_filter = None
        self.blocked_trades = 0
        self.blocked_trade_signal = False
        self.analysis_status = DEFAULT_RUNTIME_STATUS
        self.analysis_started = False
        self.analysis_completed = False
        self.analysis_phase = DEFAULT_RUNTIME_STATUS
        self.last_model_signal = None
        self.last_model_entry_index = DEFAULT_LAST_MODEL_ENTRY_INDEX
        self.pending_setup = None
        self.last_prediction = None
        self.prediction_history = []
        self.last_prediction_time = None

    def pause_if_running(self) -> None:
        """Pause an interrupted replay while preserving its completed state."""
        if self.running:
            self.running = False
        if self.analysis_status == "running":
            self.analysis_status = "paused"
            self.analysis_phase = "paused"

    def set_analysis_state(
        self,
        status: str,
        phase: Optional[str] = None,
        *,
        started: Optional[bool] = None,
        completed: Optional[bool] = None,
    ) -> None:
        """Update analysis lifecycle fields together."""
        normalized_status = str(status).strip().lower()
        if not normalized_status:
            raise ValueError("Analysis status must not be empty.")
        self.analysis_status = normalized_status
        self.analysis_phase = str(phase if phase is not None else normalized_status).strip().lower()
        if started is not None:
            self.analysis_started = bool(started)
        if completed is not None:
            self.analysis_completed = bool(completed)

    def record_prediction(self, prediction: Mapping[str, Any]) -> None:
        """Store a prediction snapshot without modifying its supplied values."""
        payload = _copy_mapping(prediction, "prediction")
        decision_time = _timestamp_seconds(
            payload.get("predicted_at", payload.get("decision_time", payload.get("prediction_time")))
        )
        if decision_time is None:
            raise ValueError("A prediction must contain a valid prediction timestamp.")
        if self.last_prediction_time is not None and decision_time < self.last_prediction_time:
            raise ValueError("Prediction timestamps cannot move backwards within a session.")
        self.last_prediction = payload
        self.last_prediction_time = decision_time
        self.prediction_history.append(payload)


# ============================================================================
# MANUAL ANALYSIS STATE
# ============================================================================

@dataclass
class ManualAnalysisState:
    """Manual-analysis status and results for one TradingBot."""

    analysis_status: str = DEFAULT_RUNTIME_STATUS
    analysis_started: bool = False
    analysis_completed: bool = False
    analysis_phase: str = DEFAULT_RUNTIME_STATUS
    progress: int = 0
    results_source: str = "testing"
    results: Dict[str, Any] = field(default_factory=dict)

    startup_stage: str = "Waiting for analysis"
    startup_progress: int = 0
    startup_message: str = ""

    def reset(self) -> None:
        """Clear manual-analysis state."""
        self.analysis_status = DEFAULT_RUNTIME_STATUS
        self.analysis_started = False
        self.analysis_completed = False
        self.analysis_phase = DEFAULT_RUNTIME_STATUS
        self.progress = 0
        self.results_source = "testing"
        self.results = {}
        self.startup_stage = "Waiting for analysis"
        self.startup_progress = 0
        self.startup_message = ""

    def update_progress(
        self,
        progress: int,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update progress with bounds suitable for frontend reporting."""
        self.progress = max(0, min(MAX_STARTUP_PROGRESS, int(progress)))
        if stage is not None:
            self.startup_stage = str(stage)
        if message is not None:
            self.startup_message = str(message)

    def as_dict(self) -> Dict[str, Any]:
        """Return an isolated dictionary suitable for a frontend snapshot."""
        return {
            "analysis_status": self.analysis_status,
            "analysis_started": self.analysis_started,
            "analysis_completed": self.analysis_completed,
            "analysis_phase": self.analysis_phase,
            "manual_analysis_progress": self.progress,
            "results_source": self.results_source,
            "manual_analysis_results": copy.deepcopy(self.results),
            "startup_stage": self.startup_stage,
            "startup_progress": self.startup_progress,
            "startup_message": self.startup_message,
        }


# ============================================================================
# TRADINGBOT RUNTIME CONTEXT
# ============================================================================

@dataclass
class RuntimeContext:
    """All in-memory state belonging to one complete TradingBot identity.

    Market candle archives may be shared as read-only inputs. Mutable session,
    candidate, trade and task state is isolated by the full customer identity.
    """

    identity: ClientIdentity
    active_timeframe: str = DEFAULT_TIMEFRAME

    chart_data_by_timeframe: Dict[str, List[Any]] = field(default_factory=dict)
    m1_data: List[Any] = field(default_factory=list)
    timeframe_sessions: Dict[str, TradingSessionState] = field(default_factory=dict)

    ai_model: Any = None
    ai_scaler: Any = None
    ai_checkpoint: Dict[str, Any] = field(default_factory=dict)
    ai_market_candles: Any = None
    ai_error: Optional[str] = None
    ai_model_version: Optional[str] = None
    ai_scaler_version: Optional[str] = None
    ai_dataset_version: Optional[str] = None
    ai_loaded_at: Optional[float] = None
    ai_status: str = "unavailable"

    pending_setup_by_timeframe: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    startup_stage: str = "Starting"
    startup_progress: int = 0
    startup_message: str = "Starting backend..."

    results_source: str = "testing"
    manual_analysis: ManualAnalysisState = field(default_factory=ManualAnalysisState)

    websocket_clients: Set[Any] = field(default_factory=set)
    tasks: Dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)

    lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        self.active_timeframe = normalize_timeframe(self.active_timeframe)
        if not isinstance(self.ai_checkpoint, dict):
            raise TypeError("ai_checkpoint must be a dictionary.")
        self.startup_progress = max(0, min(MAX_STARTUP_PROGRESS, int(self.startup_progress)))
        self.get_session(self.active_timeframe)

    @property
    def user_id(self) -> str:
        return self.identity.user_id

    @property
    def market(self) -> str:
        return self.identity.market

    @property
    def strategy_id(self) -> str:
        return self.identity.strategy_id

    @property
    def tradingbot_id(self) -> str:
        return self.identity.tradingbot_id

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return self.identity.key

    @property
    def session_id(self) -> str:
        """Return a stable non-sensitive identifier for the active runtime."""
        raw_key = "|".join((*self.identity.key, self.active_timeframe))
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]

    @property
    def current_session(self) -> TradingSessionState:
        """Return the state for the currently active timeframe."""
        return self.get_session(self.active_timeframe)

    @property
    def chart_data(self) -> List[Any]:
        """Compatibility alias for active timeframe chart data."""
        return self.get_chart_data()

    @chart_data.setter
    def chart_data(self, candles: Sequence[Any]) -> None:
        self.set_chart_data(candles)

    @property
    def h1_data(self) -> List[Any]:
        """Compatibility alias retained for the original H1 chart variable."""
        return self.chart_data

    @h1_data.setter
    def h1_data(self, candles: Sequence[Any]) -> None:
        self.chart_data = candles

    @property
    def pending_setup(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the candidate for the active timeframe."""
        return self.get_pending_setup()

    @pending_setup.setter
    def pending_setup(self, setup: Optional[Mapping[str, Any]]) -> None:
        if setup is None:
            self.clear_pending_setup()
        else:
            self.set_pending_setup(setup)

    def touch(self) -> None:
        """Update runtime activity metadata."""
        with self.lock:
            self.last_activity_at = time.time()

    def get_session(self, timeframe: Optional[str] = None) -> TradingSessionState:
        """Get or create independent state for a normalized timeframe."""
        selected = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )
        with self.lock:
            session = self.timeframe_sessions.get(selected)
            if session is None:
                session = TradingSessionState(
                    model_available=self.ai_model is not None and self.ai_scaler is not None,
                    model_info=copy.deepcopy(self.ai_checkpoint) if self.ai_checkpoint else {},
                )
                self.timeframe_sessions[selected] = session
            self.last_activity_at = time.time()
            return session

    def set_active_timeframe(self, timeframe: str) -> TradingSessionState:
        """Switch the active chart and trading state without losing other sessions."""
        normalized = normalize_timeframe(timeframe)
        with self.lock:
            self.active_timeframe = normalized
            self.last_activity_at = time.time()
        return self.get_session(normalized)

    def reset_active_session(
        self,
        starting_balance: float = DEFAULT_STARTING_BALANCE,
    ) -> None:
        """Reset the active timeframe session without removing market data or model."""
        with self.lock:
            self.current_session.reset(starting_balance)
            self.pending_setup_by_timeframe.pop(self.active_timeframe, None)
            self.last_activity_at = time.time()

    def set_chart_data(
        self,
        candles: Sequence[Any],
        timeframe: Optional[str] = None,
        *,
        require_chronological: bool = False,
    ) -> None:
        """Attach an independent candle list to this runtime and timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )
        prepared = _normalize_candle_sequence(
            candles,
            require_chronological=require_chronological,
        )
        with self.lock:
            self.chart_data_by_timeframe[normalized] = prepared
            if normalized == "M1":
                self.m1_data = prepared
            self.last_activity_at = time.time()

    def get_chart_data(self, timeframe: Optional[str] = None) -> List[Any]:
        """Return the current candle list for the requested timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )
        with self.lock:
            self.last_activity_at = time.time()
            return self.chart_data_by_timeframe.get(normalized, [])

    def clear_chart_data(self, timeframe: Optional[str] = None) -> None:
        """Remove one timeframe archive or all chart archives from this runtime."""
        with self.lock:
            if timeframe is None:
                self.chart_data_by_timeframe.clear()
                self.m1_data = []
            else:
                normalized = normalize_timeframe(timeframe)
                self.chart_data_by_timeframe.pop(normalized, None)
                if normalized == "M1":
                    self.m1_data = []
            self.last_activity_at = time.time()

    def set_pending_setup(
        self,
        setup: Mapping[str, Any],
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a candidate setup only within its runtime identity and timeframe."""
        payload = _copy_mapping(setup, "setup")
        _validate_identity_match(payload, self.identity)
        normalized = normalize_timeframe(
            timeframe
            or payload.get("entry_timeframe")
            or payload.get("timeframe")
            or self.active_timeframe
        )
        payload.setdefault("market", self.market)
        payload.setdefault("user_id", self.user_id)
        payload.setdefault("strategy_id", self.strategy_id)
        payload.setdefault("tradingbot_id", self.tradingbot_id)
        payload.setdefault("entry_timeframe", normalized)
        with self.lock:
            self.pending_setup_by_timeframe[normalized] = payload
            self.get_session(normalized).pending_setup = copy.deepcopy(payload)
            self.last_activity_at = time.time()
        return copy.deepcopy(payload)

    def get_pending_setup(
        self,
        timeframe: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return an isolated copy of the pending candidate for one timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )
        with self.lock:
            candidate = self.pending_setup_by_timeframe.get(normalized)
            if candidate is None:
                candidate = self.get_session(normalized).pending_setup
            self.last_activity_at = time.time()
            return copy.deepcopy(candidate) if isinstance(candidate, Mapping) else None

    def clear_pending_setup(self, timeframe: Optional[str] = None) -> None:
        """Remove the pending candidate for one timeframe or all timeframes."""
        with self.lock:
            if timeframe is None:
                self.pending_setup_by_timeframe.clear()
                for session in self.timeframe_sessions.values():
                    session.pending_setup = None
            else:
                normalized = normalize_timeframe(timeframe)
                self.pending_setup_by_timeframe.pop(normalized, None)
                self.get_session(normalized).pending_setup = None
            self.last_activity_at = time.time()

    def set_ai_artifacts(
        self,
        model: Any,
        scaler: Any,
        checkpoint: Mapping[str, Any],
        market_candles: Any,
        *,
        model_version: Optional[str] = None,
        scaler_version: Optional[str] = None,
        dataset_version: Optional[str] = None,
    ) -> None:
        """Attach a complete model bundle after its caller verifies integrity."""
        if model is None or scaler is None:
            raise ValueError("Both model and scaler are required.")
        checkpoint_payload = _copy_mapping(checkpoint, "checkpoint")
        if not isinstance(market_candles, Mapping) or not market_candles:
            raise ValueError("market_candles must be a non-empty timeframe mapping.")
        with self.lock:
            self.ai_model = model
            self.ai_scaler = scaler
            self.ai_checkpoint = checkpoint_payload
            self.ai_market_candles = market_candles
            self.ai_model_version = model_version or checkpoint_payload.get("model_version")
            self.ai_scaler_version = scaler_version or checkpoint_payload.get("scaler_version")
            self.ai_dataset_version = dataset_version or checkpoint_payload.get("dataset_version")
            self.ai_loaded_at = time.time()
            self.ai_error = None
            self.ai_status = "loaded"
            for session in self.timeframe_sessions.values():
                session.model_available = True
                session.model_info = copy.deepcopy(checkpoint_payload)
            self.last_activity_at = time.time()

    def clear_ai_artifacts(self, error: Optional[str] = None) -> None:
        """Clear model objects and metadata after load failure or identity change."""
        with self.lock:
            self.ai_model = None
            self.ai_scaler = None
            self.ai_checkpoint = {}
            self.ai_market_candles = None
            self.ai_model_version = None
            self.ai_scaler_version = None
            self.ai_dataset_version = None
            self.ai_loaded_at = None
            self.ai_error = str(error) if error else None
            self.ai_status = "error" if error else "unavailable"
            for session in self.timeframe_sessions.values():
                session.model_available = False
                session.model_info = {}
                session.last_model_signal = None
            self.last_activity_at = time.time()

    def set_startup_progress(
        self,
        stage: str,
        progress: int,
        message: str,
    ) -> None:
        """Set a bounded startup status for the frontend."""
        with self.lock:
            self.startup_stage = str(stage)
            self.startup_progress = max(0, min(MAX_STARTUP_PROGRESS, int(progress)))
            self.startup_message = str(message)
            self.last_activity_at = time.time()

    def set_task(self, task_name: str, task: Any) -> None:
        """Register or remove one named asynchronous task."""
        name = str(task_name).strip()
        if not name:
            raise ValueError("task_name must not be empty.")
        with self.lock:
            if task is None:
                self.tasks.pop(name, None)
            else:
                self.tasks[name] = task
            self.last_activity_at = time.time()

    def register_task(self, task_name: str, task: Any) -> bool:
        """Register a task only if the name is not already occupied."""
        name = str(task_name).strip()
        if not name:
            raise ValueError("task_name must not be empty.")
        if task is None:
            raise ValueError("task must not be None.")
        with self.lock:
            current = self.tasks.get(name)
            if current is not None:
                done_method = getattr(current, "done", None)
                if not callable(done_method) or not done_method():
                    return False
            self.tasks[name] = task
            self.last_activity_at = time.time()
            return True

    def get_task(self, task_name: str) -> Any:
        """Return a registered task or None."""
        with self.lock:
            return self.tasks.get(str(task_name).strip())

    def clear_task(self, task_name: str, expected_task: Any = None) -> None:
        """Clear a named task, optionally only if it is the expected instance."""
        name = str(task_name).strip()
        with self.lock:
            current = self.tasks.get(name)
            if current is None:
                return
            if expected_task is not None and current is not expected_task:
                return
            self.tasks.pop(name, None)
            self.last_activity_at = time.time()

    def add_websocket_client(self, websocket: Any) -> None:
        """Register a connected WebSocket client for this runtime."""
        with self.lock:
            self.websocket_clients.add(websocket)
            self.last_activity_at = time.time()

    def remove_websocket_client(self, websocket: Any) -> None:
        """Remove a disconnected WebSocket client."""
        with self.lock:
            self.websocket_clients.discard(websocket)
            self.last_activity_at = time.time()

    def get_websocket_clients(self) -> List[Any]:
        """Return a snapshot of connected WebSocket clients."""
        with self.lock:
            return list(self.websocket_clients)

    def as_dict(self) -> Dict[str, Any]:
        """Return frontend-safe runtime metadata without live model or task objects."""
        with self.lock:
            session = self.current_session
            return {
                "identity": self.identity.as_dict(),
                "market": self.market,
                "timeframe": self.active_timeframe,
                "session_id": self.session_id,
                "running": session.running,
                "trading_enabled": session.trading_enabled,
                "analysis_status": session.analysis_status,
                "analysis_phase": session.analysis_phase,
                "model_available": session.model_available,
                "model_status": self.ai_status,
                "model_version": self.ai_model_version,
                "scaler_version": self.ai_scaler_version,
                "dataset_version": self.ai_dataset_version,
                "startup_stage": self.startup_stage,
                "startup_progress": self.startup_progress,
                "startup_message": self.startup_message,
                "connected_clients": len(self.websocket_clients),
                "active_tasks": sorted(self.tasks),
                "candles_loaded": len(self.get_chart_data()),
                "pending_setup": self.get_pending_setup() is not None,
                "created_at": self.created_at,
                "last_activity_at": self.last_activity_at,
            }


# ============================================================================
# RUNTIME REGISTRY
# ============================================================================

class RuntimeRegistry:
    """Thread-safe registry containing one runtime per complete TradingBot identity."""

    def __init__(self) -> None:
        self._runtimes: Dict[Tuple[str, str, str, str], RuntimeContext] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_identity(identity: ClientIdentity) -> Tuple[str, str, str, str]:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        key = identity.key
        if not isinstance(key, tuple) or len(key) != 4:
            raise ValueError("ClientIdentity.key must contain four identity fields.")
        normalized = tuple(str(value).strip() for value in key)
        if any(not value for value in normalized):
            raise ValueError("Every ClientIdentity field must be non-empty.")
        return cast(Tuple[str, str, str, str], normalized)

    def get_or_create(
        self,
        identity: ClientIdentity,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> RuntimeContext:
        """Return the existing runtime or create one for this exact identity."""
        key = self._validate_identity(identity)
        normalized_timeframe = normalize_timeframe(timeframe)
        with self._lock:
            runtime = self._runtimes.get(key)
            if runtime is None:
                runtime = RuntimeContext(
                    identity=identity,
                    active_timeframe=normalized_timeframe,
                )
                self._runtimes[key] = runtime
            else:
                runtime.touch()
            return runtime

    def get(self, identity: ClientIdentity) -> Optional[RuntimeContext]:
        """Return an existing runtime without creating one."""
        key = self._validate_identity(identity)
        with self._lock:
            runtime = self._runtimes.get(key)
            if runtime is not None:
                runtime.touch()
            return runtime

    def remove(self, identity: ClientIdentity) -> Optional[RuntimeContext]:
        """Remove and return one runtime without implicitly cancelling its tasks."""
        key = self._validate_identity(identity)
        with self._lock:
            return self._runtimes.pop(key, None)

    def list_runtimes(self, user_id: Optional[str] = None) -> List[RuntimeContext]:
        """Return registered runtimes, optionally restricted to one customer."""
        with self._lock:
            runtimes = list(self._runtimes.values())
        if user_id is None:
            return runtimes
        normalized_user_id = str(user_id).strip()
        return [runtime for runtime in runtimes if runtime.user_id == normalized_user_id]

    def get_by_key(
        self,
        key: Tuple[str, str, str, str],
    ) -> Optional[RuntimeContext]:
        """Return a runtime by its four-part identity tuple."""
        if not isinstance(key, tuple) or len(key) != 4:
            raise ValueError("Runtime key must contain four identity fields.")
        normalized = tuple(str(value).strip() for value in key)
        if any(not value for value in normalized):
            raise ValueError("Runtime key fields must be non-empty.")
        with self._lock:
            return self._runtimes.get(normalized)

    def clear(self) -> None:
        """Remove all runtime references without cancelling their tasks."""
        with self._lock:
            self._runtimes.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._runtimes)


# ============================================================================
# ASYNC TASK CLEANUP
# ============================================================================

async def cancel_runtime_tasks(runtime: RuntimeContext) -> None:
    """Cancel and await all active asyncio tasks registered for one TradingBot."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    with runtime.lock:
        task_items = list(runtime.tasks.items())

    unique_tasks: List[Any] = []
    seen_ids: Set[int] = set()
    for _, task in task_items:
        if task is None or id(task) in seen_ids:
            continue
        seen_ids.add(id(task))
        done_method = getattr(task, "done", None)
        cancel_method = getattr(task, "cancel", None)
        if callable(cancel_method) and (not callable(done_method) or not done_method()):
            unique_tasks.append(task)

    for task in unique_tasks:
        try:
            task.cancel()
        except Exception:
            continue

    awaitables = [task for task in unique_tasks if asyncio.isfuture(task) or asyncio.iscoroutine(task)]
    if awaitables:
        await asyncio.gather(*awaitables, return_exceptions=True)

    with runtime.lock:
        for name, task in task_items:
            current = runtime.tasks.get(name)
            if current is task:
                runtime.tasks.pop(name, None)
        for session in runtime.timeframe_sessions.values():
            session.running = False
            if session.analysis_status == "running":
                session.pause_if_running()
        runtime.last_activity_at = time.time()


# ============================================================================
# DEFAULT REGISTRY AND HELPERS
# ============================================================================

RUNTIME_REGISTRY = RuntimeRegistry()


def get_runtime(
    identity: ClientIdentity,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> RuntimeContext:
    """Get or create the default runtime for one TradingBot identity."""
    return RUNTIME_REGISTRY.get_or_create(identity, timeframe)


def find_runtime(identity: ClientIdentity) -> Optional[RuntimeContext]:
    """Find one runtime without creating it."""
    return RUNTIME_REGISTRY.get(identity)


def remove_runtime(identity: ClientIdentity) -> Optional[RuntimeContext]:
    """Remove one runtime from the default registry."""
    return RUNTIME_REGISTRY.remove(identity)


def list_runtimes(user_id: Optional[str] = None) -> List[RuntimeContext]:
    """List all default runtimes or only those belonging to one customer."""
    return RUNTIME_REGISTRY.list_runtimes(user_id)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DEFAULT_TIMEFRAME",
    "DEFAULT_STARTING_BALANCE",
    "DEFAULT_LAST_MODEL_ENTRY_INDEX",
    "TradingSessionState",
    "ManualAnalysisState",
    "RuntimeContext",
    "RuntimeRegistry",
    "RUNTIME_REGISTRY",
    "get_runtime",
    "find_runtime",
    "remove_runtime",
    "list_runtimes",
    "cancel_runtime_tasks",
]
