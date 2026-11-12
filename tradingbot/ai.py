"""Public application facade for the Spectline AI engine.

The facade routes application requests to focused engine modules. It keeps
identity-aware entry points stable while the implementation is split across
data, market memory, pattern discovery, model training, evaluation and
inference components.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from client import ClientIdentity
except ImportError:
    try:
        from .client import ClientIdentity
    except ImportError:
        ClientIdentity = Any


# ============================================================================
# PUBLIC MODULE INFORMATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent


class AIEngineError(RuntimeError):
    """Base exception for public AI facade errors."""


class AIEngineNotReadyError(AIEngineError):
    """Raised when a required internal engine component is unavailable."""


class AIEngineValidationError(AIEngineError, ValueError):
    """Raised when an application request violates the AI facade contract."""


class AIEngineCompatibilityError(AIEngineError):
    """Raised when a legacy call cannot be mapped safely to the new AI task."""


# ============================================================================
# INTERNAL MODULE ROUTING
# ============================================================================

_ENGINE_MODULES = (
    "settings",
    "data",
    "history",
    "features",
    "storage",
    "catalog",
    "situations",
    "stages",
    "replay",
    "timeline",
    "labels",
    "relevance",
    "probability",
    "dataset",
    "scaling",
    "versions",
    "memory",
    "model",
    "training",
    "evaluation",
    "testing",
    "finetune",
    "inference",
)

_ENGINE_EXPORTS: Dict[str, str] = {
    "WEB_MODE": "settings",
    "TRADING_SETUP_ROOT": "settings",
    "CHARTS_ROOT": "settings",
    "MODELS_ROOT": "settings",
    "USER_DATA_ROOT": "settings",
    "CUSTOMERS_ROOT": "settings",
    "MARKET_MEMORY_ROOT": "settings",
    "MARKET_PATTERN_ROOT": "settings",
    "MARKET_EVENT_ROOT": "settings",
    "PATTERN_CATALOG_ROOT": "settings",
    "SETUP_FILENAME": "settings",
    "CANDLE_FILENAME": "settings",
    "MODEL_FILENAME": "settings",
    "ALL_TIMEFRAMES": "settings",
    "SUPPORTED_TIMEFRAMES": "settings",
    "WEB_MODE_TIMEFRAMES": "settings",
    "LOCAL_TIMEFRAMES": "settings",
    "TIMEFRAME_SECONDS": "settings",
    "LOOKBACK_BY_TIMEFRAME": "settings",
    "RESOLUTION_TIMEFRAME_PRIORITY": "settings",
    "PATTERN_CATEGORY_ID_TO_NAME": "settings",
    "PATTERN_CATEGORY_NAME_TO_ID": "settings",
    "MARKET_EVENT_TYPES": "settings",
    "OUTCOME_SL_FIRST": "settings",
    "OUTCOME_TP_FIRST": "settings",
    "TRADE_OUTCOME_TO_LABEL": "settings",
    "LABEL_TO_TRADE_OUTCOME": "settings",
    "PROBABILITY_TARGET_NAME": "settings",
    "PREDICTION_TIME_FIELD": "settings",
    "PREDICTED_PROBABILITY_FIELD": "settings",
    "ACTUAL_OUTCOME_FIELD": "settings",
    "OUTCOME_KNOWN_AT_FIELD": "settings",
    "SEED": "settings",
    "DEFAULT_EPOCHS": "settings",
    "DEFAULT_BATCH_SIZE": "settings",
    "DEFAULT_LEARNING_RATE": "settings",
    "DEFAULT_WEIGHT_DECAY": "settings",
    "HARD_NEGATIVE_OFFSETS": "settings",
    "RANDOM_NEGATIVE_MULTIPLIER": "settings",
    "POSITIVE_EXCLUSION_SECONDS": "settings",
    "DIRECTION_TO_INDEX": "settings",
    "TIMEFRAME_TO_INDEX": "settings",
    "set_seed": "settings",
    "normalize_known_timeframe": "settings",
    "normalize_timeframe": "settings",
    "timeframe_seconds": "settings",
    "known_timeframe_seconds": "settings",
    "timeframe_lookback": "settings",
    "available_timeframes": "settings",
    "validate_configuration": "settings",
    "Candle": "data",
    "ManualZone": "data",
    "UserSetup": "data",
    "DataBundleError": "data",
    "safe_market_folder": "data",
    "safe_user_folder": "data",
    "_resolve_ai_identity": "data",
    "setup_path": "data",
    "to_float": "data",
    "parse_timestamp": "data",
    "normalize_learning_setup": "data",
    "normalize_learning_setups": "data",
    "setup_time": "data",
    "load_learning_setup_records": "data",
    "load_learning_setups": "data",
    "closest_setup": "data",
    "collect_required_timeframes": "data",
    "extract_manual_levels": "data",
    "setup_input_record": "data",
    "setup_prediction_record": "data",
    "setup_label_record": "data",
    "validate_training_data": "data",
    "prepare_training_data": "data",
    "chart_path": "history",
    "load_candles": "history",
    "load_candles_with_report": "history",
    "load_candles_for_timeframes": "history",
    "load_candles_with_reports": "history",
    "load_all_market_candles": "history",
    "is_chronological": "history",
    "require_chronological": "history",
    "history_coverage": "history",
    "candle_return": "history",
    "true_range": "history",
    "rolling_mean": "history",
    "rolling_std": "history",
    "rolling_slope": "history",
    "last_closed_index": "history",
    "slice_closed_history": "history",
    "align_timeframes_at": "history",
    "slice_timeframes_at": "history",
    "sigmoid": "features",
    "timeframe_features": "features",
    "market_state_at": "features",
    "market_state_metadata_at": "features",
    "current_price_at": "features",
    "atr_at": "features",
    "setup_target_vector": "features",
    "detect_pivots": "features",
    "detect_swings": "features",
    "build_zigzag": "features",
    "calculate_pivot_geometry": "features",
    "pattern_geometry": "features",
    "pattern_geometry_vector": "features",
    "candle_geometry_vector": "features",
    "find_ai_major_levels": "features",
    "build_auto_levels_for_setups": "features",
    "StorageError": "storage",
    "StorageValidationError": "storage",
    "StorageConflictError": "storage",
    "StorageCorruptionError": "storage",
    "StorageWriteResult": "storage",
    "StorageArchiveInfo": "storage",
    "HistoricalStorage": "storage",
    "append_pattern_occurrence": "storage",
    "iter_pattern_occurrences": "storage",
    "append_market_event": "storage",
    "iter_market_events": "storage",
    "save_pattern_catalog": "storage",
    "load_pattern_catalog": "storage",
    "CatalogError": "catalog",
    "CatalogValidationError": "catalog",
    "CatalogNotFoundError": "catalog",
    "CatalogStateError": "catalog",
    "PatternCategory": "catalog",
    "PatternDefinition": "catalog",
    "PatternCatalog": "catalog",
    "list_categories": "catalog",
    "get_category": "catalog",
    "validate_pattern_id": "catalog",
    "build_pattern_id": "catalog",
    "category_for_pattern_id": "catalog",
    "is_category_code": "catalog",
    "create_catalog": "catalog",
    "load_catalog": "catalog",
    "save_catalog": "catalog",
    "validate_catalog": "catalog",
    "SituationError": "situations",
    "SituationValidationError": "situations",
    "SituationCatalogError": "situations",
    "SituationScanConfig": "situations",
    "SituationCandidate": "situations",
    "SituationScanResult": "situations",
    "MarketSituationScanResult": "situations",
    "SituationScanner": "situations",
    "geometry_similarity": "situations",
    "match_geometry_to_catalog": "situations",
    "scan_timeframe": "situations",
    "scan_market_history": "situations",
    "register_recurrent_pattern_candidate": "situations",
    "approve_pattern_candidate": "situations",
    "load_situation_catalog": "situations",
    "record_market_event": "situations",
    "StageError": "stages",
    "StageValidationError": "stages",
    "HistoryStage": "stages",
    "HistoryStagePlan": "stages",
    "StagePartition": "stages",
    "build_stage_plan": "stages",
    "build_history_stages": "stages",
    "build_stage_plan_from_candles": "stages",
    "find_stage": "stages",
    "stage_id_for_timestamp": "stages",
    "partition_records_by_stage": "stages",
    "partition_situations_by_recognition_time": "stages",
    "partition_trades_by_entry_time": "stages",
    "slice_candles_for_stage": "stages",
    "iter_stage_candle_windows": "stages",
    "summarize_stage_coverage": "stages",
    "iter_stages": "stages",
    "stage_sequence_metadata": "stages",
    "ReplayError": "replay",
    "ReplayValidationError": "replay",
    "ReplayCallbackError": "replay",
    "ReplayConfig": "replay",
    "ReplayEvent": "replay",
    "ReplayStep": "replay",
    "ReplaySummary": "replay",
    "HistoricalReplay": "replay",
    "iter_market_replay": "replay",
    "replay_market_history": "replay",
    "load_market_replay": "replay",
    "TimelineError": "timeline",
    "TimelineValidationError": "timeline",
    "TimelineConflictError": "timeline",
    "TimelineConfig": "timeline",
    "TimelineEvent": "timeline",
    "PatternTradeLink": "timeline",
    "TimelineResult": "timeline",
    "link_patterns_to_trades": "timeline",
    "build_timeline": "timeline",
    "build_timeline_from_storage": "timeline",
    "append_customer_timeline_event": "timeline",
    "LabelError": "labels",
    "LabelValidationError": "labels",
    "LabelBuildResult": "labels",
    "classify_trade_record": "labels",
    "build_trade_label": "labels",
    "validate_label_record": "labels",
    "build_trade_labels": "labels",
    "build_labels_from_bundle": "labels",
    "validate_label_collection": "labels",
    "labels_available_as_of": "labels",
    "labels_to_targets": "labels",
    "split_feature_records_and_labels": "labels",
    "load_customer_trade_labels": "labels",
    "RelevanceError": "relevance",
    "RelevanceValidationError": "relevance",
    "RelevanceConfig": "relevance",
    "SetupRelevance": "relevance",
    "build_setup_relevance": "relevance",
    "find_relevant_patterns": "relevance",
    "find_relevant_events": "relevance",
    "build_relevance_bundle": "relevance",
    "attach_relevance_to_bundle": "relevance",
    "load_market_history": "relevance",
    "build_relevance_from_storage": "relevance",
    "validate_relevance_bundle": "relevance",
    "relevance_records_for_setup": "relevance",
    "ProbabilityError": "probability",
    "ProbabilityValidationError": "probability",
    "ProbabilityConfig": "probability",
    "estimate_tp_before_sl": "probability",
    "build_probability_bundle": "probability",
    "attach_probability_to_bundle": "probability",
    "append_probability_timeline_event": "probability",
    "validate_probability_estimate": "probability",
    "validate_probability_bundle": "probability",
    "probability_for_setup": "probability",
    "DatasetError": "dataset",
    "DatasetValidationError": "dataset",
    "DatasetDependencyError": "dataset",
    "build_training_dataset": "dataset",
    "make_negative_times": "dataset",
    "validate_training_dataset": "dataset",
    "dataset_to_serializable": "dataset",
    "ScalingError": "scaling",
    "ScalingValidationError": "scaling",
    "FeatureScaler": "scaling",
    "fit_scaler": "scaling",
    "fit_dataset_scaler": "scaling",
    "apply_scaler": "scaling",
    "inverse_scaler": "scaling",
    "transform_dataset": "scaling",
    "fit_transform_dataset": "scaling",
    "transform_splits": "scaling",
    "validate_scaler": "scaling",
    "scaler_to_serializable": "scaling",
    "scaler_from_serializable": "scaling",
    "dataset_scaling_report": "scaling",
    "VersionError": "versions",
    "VersionValidationError": "versions",
    "VersionConflictError": "versions",
    "VersionNotFoundError": "versions",
    "VersionStateError": "versions",
    "VersionIntegrityError": "versions",
    "VersionDependency": "versions",
    "VersionRecord": "versions",
    "VersionRegistry": "versions",
    "get_default_registry": "versions",
    "register_version": "versions",
    "get_version": "versions",
    "list_versions": "versions",
    "validate_version": "versions",
    "approve_version": "versions",
    "deprecate_version": "versions",
    "latest_approved": "versions",
    "register_pattern_catalog": "versions",
    "register_dataset": "versions",
    "register_scaler": "versions",
    "register_model": "versions",
    "verify_artifact": "versions",
    "validate_version_status": "versions",
    "MemoryErrorBase": "memory",
    "MemoryValidationError": "memory",
    "MemoryIntegrityError": "memory",
    "MemoryVersionError": "memory",
    "model_dir": "memory",
    "model_path": "memory",
    "save_training_data_bundle": "memory",
    "load_training_data_bundle": "memory",
    "list_training_data_bundles": "memory",
    "save_training_dataset": "memory",
    "load_training_dataset": "memory",
    "list_dataset_versions": "memory",
    "save_scaler_artifact": "memory",
    "load_scaler_artifact": "memory",
    "save_model": "memory",
    "validate_artifact_version": "memory",
    "approve_artifact_version": "memory",
    "deprecate_artifact_version": "memory",
    "approve_model_version": "memory",
    "list_model_versions": "memory",
    "latest_approved_model_version": "memory",
    "verify_model_version": "memory",
    "ModelError": "model",
    "ModelValidationError": "model",
    "ModelDependencyError": "model",
    "ModelSchemaError": "model",
    "TradingMicroModel": "model",
    "compute_loss": "model",
    "train_model": "model",
    "TrainingError": "training",
    "TrainingValidationError": "training",
    "TrainingDependencyError": "training",
    "TrainingInProgressError": "training",
    "TrainingConfig": "training",
    "run_market_training": "training",
    "run_training": "training",
    "list_training_runs": "training",
    "get_training_run": "training",
    "verify_training_run": "training",
    "EvaluationError": "evaluation",
    "EvaluationValidationError": "evaluation",
    "EvaluationDependencyError": "evaluation",
    "EvaluationSchemaError": "evaluation",
    "EvaluationPolicy": "evaluation",
    "evaluate_predictions": "evaluation",
    "evaluate_model": "evaluation",
    "evaluate_model_version": "evaluation",
    "approve_evaluated_model": "evaluation",
    "TestingError": "testing",
    "TestingValidationError": "testing",
    "TestingDependencyError": "testing",
    "TestingIntegrityError": "testing",
    "TestingSchemaError": "testing",
    "run_historical_test": "testing",
    "run_model_version_test": "testing",
    "run_testing": "testing",
    "list_testing_runs": "testing",
    "load_test_report": "testing",
    "FinetuneError": "finetune",
    "FinetuneValidationError": "finetune",
    "FinetuneDependencyError": "finetune",
    "FinetuneIntegrityError": "finetune",
    "FinetuneInProgressError": "finetune",
    "FinetuneConfig": "finetune",
    "finetune_model_version": "finetune",
    "list_finetuning_runs": "finetune",
    "get_finetuning_run": "finetune",
    "run_finetuning": "finetune",
    "run_finetune": "finetune",
    "InferenceError": "inference",
    "InferenceValidationError": "inference",
    "InferenceDependencyError": "inference",
    "InferenceModelError": "inference",
    "InferenceChronologyError": "inference",
    "InferencePersistenceError": "inference",
    "load_approved_model": "inference",
    "build_setup_feature_row": "inference",
    "build_setup_feature_row_from_context": "inference",
    "predict_feature_row": "inference",
    "predict_setup": "inference",
    "predict_setup_from_context": "inference",
    "get_inference_model_info": "inference",
}


# ============================================================================
# ENGINE IMPORT AND DISPATCH HELPERS
# ============================================================================

def _qualified_engine_name(module_name: str) -> str:
    normalized = str(module_name).strip().replace("/", ".")
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    if normalized not in _ENGINE_MODULES:
        raise ValueError(f"Unknown Spectline engine module: {module_name!r}.")
    if __package__:
        return f"{__package__}.engine.{normalized}"
    return f"engine.{normalized}"


def _import_engine_module(module_name: str) -> Any:
    """Import one engine module while preserving package and script execution."""
    qualified_name = _qualified_engine_name(module_name)
    try:
        return importlib.import_module(qualified_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or ""
        missing_component = (
            missing_name == qualified_name
            or missing_name == "engine"
            or missing_name.endswith(f".engine.{module_name}")
        )
        if not missing_component:
            raise
        raise AIEngineNotReadyError(
            f"The Spectline engine component '{module_name}.py' is not available."
        ) from error


def _get_engine_export(name: str) -> Any:
    module_name = _ENGINE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ai' has no attribute {name!r}.")
    module = _import_engine_module(module_name)
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise AIEngineNotReadyError(
            f"The symbol '{name}' is not available in 'engine/{module_name}.py'."
        ) from error


def _call_engine(
    module_name: str,
    function_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    module = _import_engine_module(module_name)
    try:
        function = getattr(module, function_name)
    except AttributeError as error:
        raise AIEngineNotReadyError(
            f"The function '{function_name}' is not available in 'engine/{module_name}.py'."
        ) from error
    if not callable(function):
        raise TypeError(f"'engine/{module_name}.py:{function_name}' is not callable.")
    return function(*args, **kwargs)


def _resolve_identity(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Any:
    return _call_engine(
        "data",
        "_resolve_ai_identity",
        market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


# ============================================================================
# IDENTITY AND PATH API
# ============================================================================

def setup_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    return _call_engine(
        "data",
        "setup_path",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def model_dir(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    return _call_engine(
        "memory",
        "model_dir",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def model_path(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Path:
    return _call_engine(
        "memory",
        "model_path",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


# ============================================================================
# CUSTOMER DATA API
# ============================================================================

def load_learning_setups(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Any:
    return _call_engine(
        "data",
        "load_learning_setups",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )


def prepare_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    point_in_time_candle_limit: Optional[int] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "strategy_id": strategy_id,
        "tradingbot_id": tradingbot_id,
        "identity": identity,
    }
    if point_in_time_candle_limit is not None:
        kwargs["point_in_time_candle_limit"] = point_in_time_candle_limit
    return _call_engine("data", "prepare_training_data", market, user_id, **kwargs)


def validate_training_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return _call_engine("data", "validate_training_data", bundle)


def prepare_and_store_training_data(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    bundle_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    return _call_engine(
        "memory",
        "prepare_and_store_training_data",
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
        bundle_id=bundle_id,
    )


def build_training_dataset(bundle: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return _call_engine("dataset", "build_training_dataset", bundle, **kwargs)


def validate_training_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return _call_engine("dataset", "validate_training_dataset", dataset)


# ============================================================================
# MARKET HISTORY AND FEATURE API
# ============================================================================

def load_all_market_candles(
    market: str,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    return _call_engine("history", "load_all_market_candles", market, identity=identity)


def market_state_at(candles_by_timeframe: Mapping[str, Any], target_time: int) -> Any:
    return _call_engine("features", "market_state_at", candles_by_timeframe, target_time)


def market_state_metadata_at(
    candles_by_timeframe: Mapping[str, Any],
    target_time: int,
) -> Any:
    return _call_engine(
        "features",
        "market_state_metadata_at",
        candles_by_timeframe,
        target_time,
    )


def current_price_at(candles_by_timeframe: Mapping[str, Any], target_time: int) -> Optional[float]:
    return _call_engine("features", "current_price_at", candles_by_timeframe, target_time)


def atr_at(candles_by_timeframe: Mapping[str, Any], target_time: int) -> float:
    return _call_engine("features", "atr_at", candles_by_timeframe, target_time)


def predict_from_features(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Predict TP-before-SL probability using either supported feature API.

    The runtime compatibility form accepts ``model``, ``scaler`` and
    ``features``. The preferred point-in-time form accepts market, features,
    ordered feature names, identity and decision time and routes through the
    inference component that verifies the approved model version.
    """
    legacy_runtime_call = (
        "model" in kwargs
        or "scaler" in kwargs
        or (len(args) >= 1 and not isinstance(args[0], str))
    )
    if legacy_runtime_call:
        if "model" in kwargs:
            model = kwargs.pop("model")
        elif args:
            model, *args = args
        else:
            model = None
        if "scaler" in kwargs:
            scaler = kwargs.pop("scaler")
        elif args:
            scaler, *args = args
        else:
            scaler = None
        if "features" in kwargs:
            features = kwargs.pop("features")
        elif args:
            features, *args = args
        else:
            features = None
        if model is None or scaler is None or features is None:
            raise AIEngineValidationError(
                "The runtime feature prediction call requires model, scaler and features."
            )
        return _call_engine(
            "model",
            "predict_from_features",
            model,
            scaler,
            features,
            **kwargs,
        )
    return _call_engine("inference", "predict_from_features", *args, **kwargs)


# ============================================================================
# MODEL LOADING, TRAINING AND PREDICTION API
# ============================================================================

def load_model(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    model_version_id: Optional[str] = None,
    version_id: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    resolved = _resolve_identity(
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )
    selected_version = model_version_id or version_id
    return _call_engine(
        "inference",
        "load_approved_model",
        market,
        identity=resolved,
        model_version_id=selected_version,
    )


def save_model(
    market: str,
    model: Any,
    scaler: Any,
    training_summary: Dict[str, Any],
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    version_id: Optional[str] = None,
    dependencies: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    register_version: bool = True,
    scaler_version_id: Optional[str] = None,
    dataset_version_id: Optional[str] = None,
) -> Path:
    return _call_engine(
        "memory",
        "save_model",
        market,
        model,
        scaler,
        training_summary,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
        version_id=version_id,
        dependencies=dependencies,
        metadata=metadata,
        register_version=register_version,
        scaler_version_id=scaler_version_id,
        dataset_version_id=dataset_version_id,
    )


def train_market(
    market: str,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    run_id: Optional[str] = None,
    config: Any = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "strategy_id": strategy_id,
        "tradingbot_id": tradingbot_id,
        "identity": identity,
        "run_id": run_id,
    }
    if config is not None:
        kwargs["config"] = config
    return _call_engine(
        "training",
        "train_market",
        market,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        user_id,
        **kwargs,
    )


def predict_setup(
    market: str,
    setup_id: Any,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    data_bundle: Optional[Mapping[str, Any]] = None,
    model_version_id: Optional[str] = None,
    prediction_time: Any = None,
    threshold: float = 0.50,
    persist: bool = True,
    allow_post_entry_analysis: bool = True,
    point_in_time_candle_limit: int = 240,
) -> Dict[str, Any]:
    return _call_engine(
        "inference",
        "predict_setup",
        market,
        setup_id,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
        data_bundle=data_bundle,
        model_version_id=model_version_id,
        prediction_time=prediction_time,
        threshold=threshold,
        persist=persist,
        allow_post_entry_analysis=allow_post_entry_analysis,
        point_in_time_candle_limit=point_in_time_candle_limit,
    )


def predict_setup_from_context(
    setup_record: Mapping[str, Any],
    candles_by_timeframe: Mapping[str, Sequence[Any]],
    *,
    market: str,
    identity: ClientIdentity,
    **kwargs: Any,
) -> Dict[str, Any]:
    return _call_engine(
        "inference",
        "predict_setup_from_context",
        setup_record,
        candles_by_timeframe,
        market=market,
        identity=identity,
        **kwargs,
    )


def predict_market(
    market: str,
    target_time: Optional[int] = None,
    user_id: Optional[str] = None,
    *,
    setup_id: Optional[Any] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if setup_id is None:
        raise AIEngineCompatibilityError(
            "TP-before-SL prediction requires a concrete setup_id. "
            "A market timestamp alone does not define an entry, stop-loss and take-profit event."
        )
    if target_time is not None and "prediction_time" not in kwargs:
        kwargs["prediction_time"] = target_time
    return predict_setup(
        market,
        setup_id,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
        **kwargs,
    )


def show_model_info(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
    model_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = _resolve_identity(
        market,
        user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
        identity=identity,
    )
    return _call_engine(
        "inference",
        "get_inference_model_info",
        market,
        identity=resolved,
        model_version_id=model_version_id,
    )


# ============================================================================
# PATTERN DISCOVERY, TIMELINE AND HISTORICAL MEMORY API
# ============================================================================

def scan_market_history(market: str, candles_by_timeframe: Mapping[str, Sequence[Any]], **kwargs: Any) -> Any:
    return _call_engine("situations", "scan_market_history", market, candles_by_timeframe, **kwargs)


def build_timeline_from_storage(market: str, *, identity: ClientIdentity, **kwargs: Any) -> Any:
    return _call_engine(
        "timeline",
        "build_timeline_from_storage",
        market,
        identity=identity,
        **kwargs,
    )


def build_relevance_bundle(bundle: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return _call_engine("relevance", "build_relevance_bundle", bundle, **kwargs)


def build_probability_bundle(bundle: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    return _call_engine("probability", "build_probability_bundle", bundle, **kwargs)


def get_engine_capabilities(*, import_modules: bool = False) -> Dict[str, Any]:
    """Return the available engine module map without forcing imports by default."""
    result: Dict[str, Any] = {
        "facade_schema_version": 2,
        "modules": {},
        "target": "TP_BEFORE_SL",
    }
    for module_name in _ENGINE_MODULES:
        if not import_modules:
            result["modules"][module_name] = {"status": "LAZY"}
            continue
        try:
            _import_engine_module(module_name)
            result["modules"][module_name] = {"status": "AVAILABLE"}
        except Exception as error:
            result["modules"][module_name] = {
                "status": "UNAVAILABLE",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    return result


# ============================================================================
# LAZY COMPATIBILITY EXPORTS
# ============================================================================

def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        return _get_engine_export(name)
    raise AttributeError(f"module 'ai' has no attribute {name!r}.")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ENGINE_EXPORTS))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def build_parser() -> Any:
    return _call_engine("model", "build_parser")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return int(_call_engine("model", "main", argv))


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "AIEngineError",
    "AIEngineNotReadyError",
    "AIEngineValidationError",
    "AIEngineCompatibilityError",
    "ClientIdentity",
    "BASE_DIR",
    "setup_path",
    "model_dir",
    "model_path",
    "load_learning_setups",
    "prepare_training_data",
    "validate_training_data",
    "prepare_and_store_training_data",
    "build_training_dataset",
    "validate_training_dataset",
    "load_all_market_candles",
    "market_state_at",
    "market_state_metadata_at",
    "current_price_at",
    "atr_at",
    "predict_from_features",
    "predict_setup",
    "predict_setup_from_context",
    "predict_market",
    "load_model",
    "save_model",
    "train_market",
    "show_model_info",
    "scan_market_history",
    "build_timeline_from_storage",
    "build_relevance_bundle",
    "build_probability_bundle",
    "get_engine_capabilities",
    "build_parser",
    "main",
] + list(_ENGINE_EXPORTS)


if __name__ == "__main__":
    raise SystemExit(main())
