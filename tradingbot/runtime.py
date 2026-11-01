"""Per-TradingBot runtime state management for Spectline."""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from client import ClientIdentity, normalize_timeframe


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_TIMEFRAME = "H1"
DEFAULT_STARTING_BALANCE = 100_000.0
DEFAULT_LAST_MODEL_ENTRY_INDEX = -10_000_000


# ============================================================================
# TIMEFRAME SESSION STATE
# ============================================================================

@dataclass
class TradingSessionState:
    """Trading and replay state for one TradingBot timeframe."""

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

    analysis_status: str = "idle"
    analysis_started: bool = False
    analysis_completed: bool = False
    analysis_phase: str = "idle"

    model_available: bool = False
    model_info: Dict[str, Any] = field(default_factory=dict)
    last_training_info: Dict[str, Any] = field(default_factory=dict)
    last_model_signal: Optional[Dict[str, Any]] = None
    last_model_entry_index: int = DEFAULT_LAST_MODEL_ENTRY_INDEX

    def reset(self, starting_balance: float = DEFAULT_STARTING_BALANCE) -> None:
        """Reset trading and replay values without changing the timeframe."""
        self.current_index = 0
        self.running = False
        self.trading_enabled = True

        self.balance = float(starting_balance)
        self.peak_equity = float(starting_balance)

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

        self.analysis_status = "idle"
        self.analysis_started = False
        self.analysis_completed = False
        self.analysis_phase = "idle"

        self.last_model_signal = None
        self.last_model_entry_index = DEFAULT_LAST_MODEL_ENTRY_INDEX

    def pause_if_running(self) -> None:
        """Mark an interrupted replay as paused while preserving its state."""
        self.running = False
        if self.analysis_status == "running":
            self.analysis_status = "paused"
            self.analysis_phase = "paused"


# ============================================================================
# MANUAL ANALYSIS STATE
# ============================================================================

@dataclass
class ManualAnalysisState:
    """Manual-analysis status and results for one TradingBot."""

    analysis_status: str = "idle"
    analysis_started: bool = False
    analysis_completed: bool = False
    analysis_phase: str = "idle"
    progress: int = 0
    results_source: str = "testing"
    results: Dict[str, Any] = field(default_factory=dict)

    startup_stage: str = "Waiting for analysis"
    startup_progress: int = 0
    startup_message: str = ""

    def reset(self) -> None:
        """Clear manual-analysis state."""
        self.analysis_status = "idle"
        self.analysis_started = False
        self.analysis_completed = False
        self.analysis_phase = "idle"
        self.progress = 0
        self.results_source = "testing"
        self.results = {}
        self.startup_stage = "Waiting for analysis"
        self.startup_progress = 0
        self.startup_message = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary suitable for a frontend snapshot."""
        return {
            "analysis_status": self.analysis_status,
            "analysis_started": self.analysis_started,
            "analysis_completed": self.analysis_completed,
            "analysis_phase": self.analysis_phase,
            "manual_analysis_progress": self.progress,
            "results_source": self.results_source,
            "manual_analysis_results": self.results,
            "startup_stage": self.startup_stage,
            "startup_progress": self.startup_progress,
            "startup_message": self.startup_message,
        }


# ============================================================================
# TRADINGBOT RUNTIME CONTEXT
# ============================================================================

@dataclass
class RuntimeContext:
    """All in-memory runtime state belonging to one TradingBot identity.

    The identity fixes the customer, market, strategy and TradingBot.
    Timeframe-specific trading state is held separately in timeframe_sessions.
    Historical chart arrays may reference shared, read-only market data.
    """

    identity: ClientIdentity
    active_timeframe: str = DEFAULT_TIMEFRAME

    chart_data_by_timeframe: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )
    m1_data: List[Dict[str, Any]] = field(default_factory=list)

    timeframe_sessions: Dict[str, TradingSessionState] = field(
        default_factory=dict
    )

    ai_model: Any = None
    ai_scaler: Any = None
    ai_checkpoint: Dict[str, Any] = field(default_factory=dict)
    ai_market_candles: Any = None

    startup_stage: str = "Starting"
    startup_progress: int = 0
    startup_message: str = "Starting backend..."

    results_source: str = "testing"
    manual_analysis: ManualAnalysisState = field(
        default_factory=ManualAnalysisState
    )

    websocket_clients: Set[Any] = field(default_factory=set)
    tasks: Dict[str, Any] = field(default_factory=dict)

    lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")

        self.active_timeframe = normalize_timeframe(self.active_timeframe)
        self.get_session(self.active_timeframe)

    # ------------------------------------------------------------------------
    # Identity properties
    # ------------------------------------------------------------------------

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
        """Return a stable non-sensitive identifier for the active session."""
        raw_key = "|".join((*self.identity.key, self.active_timeframe))
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]

    # ------------------------------------------------------------------------
    # Timeframe session management
    # ------------------------------------------------------------------------

    def get_session(
        self,
        timeframe: Optional[str] = None,
    ) -> TradingSessionState:
        """Get or create the state for the requested timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )

        with self.lock:
            if normalized not in self.timeframe_sessions:
                self.timeframe_sessions[normalized] = TradingSessionState()

            return self.timeframe_sessions[normalized]

    @property
    def current_session(self) -> TradingSessionState:
        """Return the state for the currently selected timeframe."""
        return self.get_session(self.active_timeframe)

    def set_active_timeframe(self, timeframe: str) -> TradingSessionState:
        """Select a timeframe and return its independent session state."""
        normalized = normalize_timeframe(timeframe)

        with self.lock:
            self.active_timeframe = normalized

        return self.get_session(normalized)

    def reset_active_session(
        self,
        starting_balance: float = DEFAULT_STARTING_BALANCE,
    ) -> None:
        """Reset only the active timeframe session."""
        with self.lock:
            self.current_session.reset(starting_balance)

    # ------------------------------------------------------------------------
    # Chart data management
    # ------------------------------------------------------------------------

    def set_chart_data(
        self,
        candles: List[Dict[str, Any]],
        timeframe: Optional[str] = None,
    ) -> None:
        """Attach chart history to this runtime for one timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )

        with self.lock:
            self.chart_data_by_timeframe[normalized] = candles

    def get_chart_data(
        self,
        timeframe: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return chart history for the requested or active timeframe."""
        normalized = normalize_timeframe(
            timeframe if timeframe is not None else self.active_timeframe
        )

        with self.lock:
            return self.chart_data_by_timeframe.get(normalized, [])

    @property
    def chart_data(self) -> List[Dict[str, Any]]:
        """Return chart data for the active timeframe."""
        return self.get_chart_data()

    @chart_data.setter
    def chart_data(self, candles: List[Dict[str, Any]]) -> None:
        self.set_chart_data(candles)

    @property
    def h1_data(self) -> List[Dict[str, Any]]:
        """Compatibility alias for the backend's active chart data variable."""
        return self.chart_data

    @h1_data.setter
    def h1_data(self, candles: List[Dict[str, Any]]) -> None:
        self.chart_data = candles

    # ------------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------------

    def set_task(self, task_name: str, task: Any) -> None:
        """Register an asynchronous task under a name for this TradingBot."""
        name = str(task_name).strip()
        if not name:
            raise ValueError("task_name must not be empty.")

        with self.lock:
            if task is None:
                self.tasks.pop(name, None)
            else:
                self.tasks[name] = task

    def get_task(self, task_name: str) -> Any:
        """Return a registered task, or None when it does not exist."""
        with self.lock:
            return self.tasks.get(str(task_name).strip())

    def clear_task(
        self,
        task_name: str,
        expected_task: Any = None,
    ) -> None:
        """Clear a task, optionally only when it matches expected_task."""
        name = str(task_name).strip()

        with self.lock:
            current = self.tasks.get(name)
            if current is None:
                return

            if expected_task is not None and current is not expected_task:
                return

            self.tasks.pop(name, None)

    # ------------------------------------------------------------------------
    # WebSocket client management
    # ------------------------------------------------------------------------

    def add_websocket_client(self, websocket: Any) -> None:
        with self.lock:
            self.websocket_clients.add(websocket)

    def remove_websocket_client(self, websocket: Any) -> None:
        with self.lock:
            self.websocket_clients.discard(websocket)

    def get_websocket_clients(self) -> List[Any]:
        with self.lock:
            return list(self.websocket_clients)

    # ------------------------------------------------------------------------
    # Runtime metadata
    # ------------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Return safe runtime metadata without model objects or live tasks."""
        with self.lock:
            session = self.current_session
            return {
                "identity": self.identity.as_dict(),
                "market": self.market,
                "timeframe": self.active_timeframe,
                "session_id": self.session_id,
                "running": session.running,
                "analysis_status": session.analysis_status,
                "analysis_phase": session.analysis_phase,
                "model_available": session.model_available,
                "startup_stage": self.startup_stage,
                "startup_progress": self.startup_progress,
                "startup_message": self.startup_message,
                "connected_clients": len(self.websocket_clients),
            }


# ============================================================================
# RUNTIME REGISTRY
# ============================================================================

class RuntimeRegistry:
    """Thread-safe registry containing one runtime per TradingBot identity."""

    def __init__(self) -> None:
        self._runtimes: Dict[Tuple[str, str, str, str], RuntimeContext] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        identity: ClientIdentity,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> RuntimeContext:
        """Return the existing runtime or create it for this identity."""
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")

        normalized_timeframe = normalize_timeframe(timeframe)
        key = identity.key

        with self._lock:
            runtime = self._runtimes.get(key)

            if runtime is None:
                runtime = RuntimeContext(
                    identity=identity,
                    active_timeframe=normalized_timeframe,
                )
                self._runtimes[key] = runtime

            return runtime

    def get(
        self,
        identity: ClientIdentity,
    ) -> Optional[RuntimeContext]:
        """Return an existing runtime without creating a new one."""
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")

        with self._lock:
            return self._runtimes.get(identity.key)

    def remove(
        self,
        identity: ClientIdentity,
    ) -> Optional[RuntimeContext]:
        """Remove and return a runtime from the registry."""
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")

        with self._lock:
            return self._runtimes.pop(identity.key, None)

    def list_runtimes(
        self,
        user_id: Optional[str] = None,
    ) -> List[RuntimeContext]:
        """Return registered runtimes, optionally filtered by customer."""
        with self._lock:
            runtimes = list(self._runtimes.values())

        if user_id is None:
            return runtimes

        normalized_user_id = str(user_id).strip()
        return [
            runtime
            for runtime in runtimes
            if runtime.user_id == normalized_user_id
        ]

    def clear(self) -> None:
        """Remove all runtime references from the registry."""
        with self._lock:
            self._runtimes.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._runtimes)


# ============================================================================
# ASYNC TASK CLEANUP
# ============================================================================

async def cancel_runtime_tasks(runtime: RuntimeContext) -> None:
    """Cancel and await all active tasks registered for one TradingBot."""
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext instance.")

    with runtime.lock:
        tasks = list(runtime.tasks.values())

    unique_tasks: List[Any] = []
    seen_ids: Set[int] = set()

    for task in tasks:
        if task is None or id(task) in seen_ids:
            continue
        seen_ids.add(id(task))

        done_method = getattr(task, "done", None)
        if callable(done_method) and not done_method():
            unique_tasks.append(task)

    for task in unique_tasks:
        task.cancel()

    if unique_tasks:
        await asyncio.gather(*unique_tasks, return_exceptions=True)

    with runtime.lock:
        runtime.tasks.clear()
        for session in runtime.timeframe_sessions.values():
            session.running = False


# ============================================================================
# DEFAULT REGISTRY AND HELPERS
# ============================================================================

RUNTIME_REGISTRY = RuntimeRegistry()


def get_runtime(
    identity: ClientIdentity,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> RuntimeContext:
    """Get or create the runtime for one TradingBot identity."""
    return RUNTIME_REGISTRY.get_or_create(identity, timeframe)


def find_runtime(identity: ClientIdentity) -> Optional[RuntimeContext]:
    """Find an existing runtime without creating it."""
    return RUNTIME_REGISTRY.get(identity)


def remove_runtime(identity: ClientIdentity) -> Optional[RuntimeContext]:
    """Remove one runtime from the default registry."""
    return RUNTIME_REGISTRY.remove(identity)


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
    "cancel_runtime_tasks",
]
