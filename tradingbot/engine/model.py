"""PyTorch model architecture, training and inference for Spectline AI."""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from client import ClientIdentity, normalize_market, resolve_identity
    from engine.data import build_training_dataset, prepare_training_data
    from engine.features import (
        SETUP_PRICE_TARGET_FIELDS,
        atr_at,
        current_price_at,
        market_state_at,
    )
    from engine.history import load_all_market_candles, load_candles_for_timeframes
    from engine.memory import (
        FeatureScaler,
        apply_scaler,
        fit_scaler,
        load_model,
        model_path,
        save_model,
    )
    from engine.settings import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_WEIGHT_DECAY,
        DIRECTION_TO_INDEX,
        SEED,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_TO_INDEX,
        set_seed,
    )
except ImportError:
    from ..client import ClientIdentity, normalize_market, resolve_identity
    from .data import build_training_dataset, prepare_training_data
    from .features import (
        SETUP_PRICE_TARGET_FIELDS,
        atr_at,
        current_price_at,
        market_state_at,
    )
    from .history import load_all_market_candles, load_candles_for_timeframes
    from .memory import (
        FeatureScaler,
        apply_scaler,
        fit_scaler,
        load_model,
        model_path,
        save_model,
    )
    from .settings import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LEARNING_RATE,
        DEFAULT_WEIGHT_DECAY,
        DIRECTION_TO_INDEX,
        SEED,
        SUPPORTED_TIMEFRAMES,
        TIMEFRAME_TO_INDEX,
        set_seed,
    )


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_FORMAT_VERSION = 3
DEFAULT_HIDDEN_SIZES = (256, 128, 64)
DEFAULT_DROPOUT_1 = 0.08
DEFAULT_DROPOUT_2 = 0.06
PRICE_TARGET_SIZE = len(SETUP_PRICE_TARGET_FIELDS)
PRICE_TARGET_INDEX = {
    field_name: index
    for index, field_name in enumerate(SETUP_PRICE_TARGET_FIELDS)
}

LOSS_WEIGHTS = {
    "direction": 1.00,
    "quality": 0.75,
    "setup_timeframe": 0.20,
    "entry_timeframe": 0.20,
    "price": 0.70,
}


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TradingMicroModel(nn.Module):
    """Multi-task numerical network for market direction and setup structure.

    Output heads:
        direction: WAIT / LONG / SHORT logits
        setup_probability: setup-quality logit
        setup_timeframe: configured setup-timeframe logits
        entry_timeframe: configured entry-timeframe logits
        price_offsets: ATR-normalized Anchor, Breakout, Entry, SL and TP offsets
    """

    def __init__(
        self,
        input_size: int,
        hidden_1: int = DEFAULT_HIDDEN_SIZES[0],
        hidden_2: int = DEFAULT_HIDDEN_SIZES[1],
        hidden_3: int = DEFAULT_HIDDEN_SIZES[2],
    ) -> None:
        super().__init__()

        dimensions = (input_size, hidden_1, hidden_2, hidden_3)
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Model input and hidden dimensions must be positive.")

        self.input_size = int(input_size)
        self.hidden_sizes = (int(hidden_1), int(hidden_2), int(hidden_3))

        self.backbone = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.LayerNorm(self.hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(DEFAULT_DROPOUT_1),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.LayerNorm(self.hidden_sizes[1]),
            nn.GELU(),
            nn.Dropout(DEFAULT_DROPOUT_2),
            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
            nn.LayerNorm(self.hidden_sizes[2]),
            nn.GELU(),
        )

        latent_size = self.hidden_sizes[2]
        self.direction_head = nn.Linear(latent_size, len(DIRECTION_TO_INDEX))
        self.setup_probability_head = nn.Linear(latent_size, 1)
        self.setup_timeframe_head = nn.Linear(latent_size, len(SUPPORTED_TIMEFRAMES))
        self.entry_timeframe_head = nn.Linear(latent_size, len(SUPPORTED_TIMEFRAMES))
        self.price_offset_head = nn.Linear(latent_size, PRICE_TARGET_SIZE)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """Return raw logits and normalized price-offset predictions."""
        if features.ndim != 2:
            raise ValueError("Model input must be a two-dimensional tensor.")
        if features.shape[1] != self.input_size:
            raise ValueError(
                f"Model expects {self.input_size} features, "
                f"received {features.shape[1]}."
            )

        latent = self.backbone(features)
        return {
            "direction": self.direction_head(latent),
            "setup_probability": self.setup_probability_head(latent),
            "setup_timeframe": self.setup_timeframe_head(latent),
            "entry_timeframe": self.entry_timeframe_head(latent),
            "price_offsets": self.price_offset_head(latent),
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Calculate Smooth L1 loss only for target values that are present."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Price prediction, target and mask shapes must match.")

    raw_loss = nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    safe_mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_loss = raw_loss * safe_mask
    return weighted_loss.sum() / safe_mask.sum().clamp_min(1.0)


def compute_loss(
    outputs: Mapping[str, Tensor],
    direction: Tensor,
    quality: Tensor,
    price_targets: Tensor,
    price_masks: Tensor,
    setup_timeframe: Tensor,
    entry_timeframe: Tensor,
) -> Tuple[Tensor, Dict[str, float]]:
    """Calculate the weighted multi-task objective and detached metrics."""
    required = {
        "direction",
        "setup_probability",
        "setup_timeframe",
        "entry_timeframe",
        "price_offsets",
    }
    missing = required - set(outputs)
    if missing:
        raise KeyError("Model output is missing: " + ", ".join(sorted(missing)))

    direction_loss = nn.functional.cross_entropy(
        outputs["direction"],
        direction,
        label_smoothing=0.04,
    )
    quality_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["setup_probability"],
        quality,
    )
    setup_timeframe_loss = nn.functional.cross_entropy(
        outputs["setup_timeframe"],
        setup_timeframe,
    )
    entry_timeframe_loss = nn.functional.cross_entropy(
        outputs["entry_timeframe"],
        entry_timeframe,
    )
    price_loss = masked_smooth_l1(
        outputs["price_offsets"],
        price_targets,
        price_masks,
    )

    total = (
        LOSS_WEIGHTS["direction"] * direction_loss
        + LOSS_WEIGHTS["quality"] * quality_loss
        + LOSS_WEIGHTS["setup_timeframe"] * setup_timeframe_loss
        + LOSS_WEIGHTS["entry_timeframe"] * entry_timeframe_loss
        + LOSS_WEIGHTS["price"] * price_loss
    )

    metrics = {
        "total": float(total.detach().cpu().item()),
        "direction": float(direction_loss.detach().cpu().item()),
        "quality": float(quality_loss.detach().cpu().item()),
        "setup_timeframe": float(setup_timeframe_loss.detach().cpu().item()),
        "entry_timeframe": float(entry_timeframe_loss.detach().cpu().item()),
        "price": float(price_loss.detach().cpu().item()),
    }
    return total, metrics


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def _normalize_training_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the arrays required by the multi-task model."""
    required = (
        "features",
        "direction",
        "quality",
        "price_targets",
        "price_masks",
        "setup_timeframe",
        "entry_timeframe",
    )
    missing = [key for key in required if key not in dataset]
    if missing:
        raise ValueError("Training dataset is missing: " + ", ".join(missing))

    features = np.asarray(dataset["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Training features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Training features contain NaN or infinite values.")

    sample_count = features.shape[0]
    direction = np.asarray(dataset["direction"], dtype=np.int64).reshape(-1)
    quality = np.asarray(dataset["quality"], dtype=np.float32).reshape(-1, 1)
    price_targets = np.asarray(dataset["price_targets"], dtype=np.float32)
    price_masks = np.asarray(dataset["price_masks"], dtype=np.float32)
    setup_timeframe = np.asarray(dataset["setup_timeframe"], dtype=np.int64).reshape(-1)
    entry_timeframe = np.asarray(dataset["entry_timeframe"], dtype=np.int64).reshape(-1)

    arrays = {
        "direction": direction,
        "quality": quality,
        "price_targets": price_targets,
        "price_masks": price_masks,
        "setup_timeframe": setup_timeframe,
        "entry_timeframe": entry_timeframe,
    }
    for name, values in arrays.items():
        if values.shape[0] != sample_count:
            raise ValueError(
                f"Training array '{name}' has {values.shape[0]} rows; "
                f"expected {sample_count}."
            )

    if price_targets.shape != (sample_count, PRICE_TARGET_SIZE):
        raise ValueError(
            f"price_targets must have shape ({sample_count}, {PRICE_TARGET_SIZE})."
        )
    if price_masks.shape != (sample_count, PRICE_TARGET_SIZE):
        raise ValueError(
            f"price_masks must have shape ({sample_count}, {PRICE_TARGET_SIZE})."
        )
    if not np.all(np.isfinite(quality)):
        raise ValueError("Quality labels contain NaN or infinite values.")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("Quality labels must be between 0 and 1.")
    if not np.all(np.isfinite(price_targets)) or not np.all(np.isfinite(price_masks)):
        raise ValueError("Price targets and masks must contain finite values.")
    if np.any(price_masks < 0.0):
        raise ValueError("Price masks cannot contain negative values.")
    if np.any((direction < 0) | (direction >= len(DIRECTION_TO_INDEX))):
        raise ValueError("Direction labels contain an unsupported class index.")
    timeframe_count = len(SUPPORTED_TIMEFRAMES)
    if np.any((setup_timeframe < 0) | (setup_timeframe >= timeframe_count)):
        raise ValueError("Setup timeframe labels contain an unsupported index.")
    if np.any((entry_timeframe < 0) | (entry_timeframe >= timeframe_count)):
        raise ValueError("Entry timeframe labels contain an unsupported index.")

    return {
        "features": features,
        "direction": direction,
        "quality": quality,
        "price_targets": price_targets,
        "price_masks": price_masks,
        "setup_timeframe": setup_timeframe,
        "entry_timeframe": entry_timeframe,
        "stats": dict(dataset.get("stats", {})),
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    dataset: Mapping[str, Any],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> Tuple[TradingMicroModel, FeatureScaler, Dict[str, Any]]:
    """Train from a fully labelled dataset supplied by the dataset pipeline.

    This function is not called by Training Data Bundle preparation. It runs
    only when a separate explicit training request reaches ``train_market``.
    """
    normalized = _normalize_training_dataset(dataset)

    if int(epochs) <= 0:
        raise ValueError("epochs must be greater than zero.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be a finite positive number.")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be a finite non-negative number.")

    set_seed(SEED)
    features = normalized["features"]
    scaler = fit_scaler(features)
    scaled_features = apply_scaler(features, scaler)

    tensor_dataset = TensorDataset(
        torch.from_numpy(scaled_features),
        torch.from_numpy(normalized["direction"]),
        torch.from_numpy(normalized["quality"]),
        torch.from_numpy(normalized["price_targets"]),
        torch.from_numpy(normalized["price_masks"]),
        torch.from_numpy(normalized["setup_timeframe"]),
        torch.from_numpy(normalized["entry_timeframe"]),
    )
    effective_batch_size = max(1, min(int(batch_size), len(tensor_dataset)))
    loader = DataLoader(
        tensor_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = TradingMicroModel(input_size=int(scaled_features.shape[1]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
    )

    model.train()
    last_metrics: Dict[str, float] = {}
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        batch_count = 0
        for batch in loader:
            (
                batch_features,
                batch_direction,
                batch_quality,
                batch_price_targets,
                batch_price_masks,
                batch_setup_timeframe,
                batch_entry_timeframe,
            ) = batch

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss, metrics = compute_loss(
                outputs=outputs,
                direction=batch_direction,
                quality=batch_quality,
                price_targets=batch_price_targets,
                price_masks=batch_price_masks,
                setup_timeframe=batch_setup_timeframe,
                entry_timeframe=batch_entry_timeframe,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss detected at epoch {epoch}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1
            last_metrics = metrics

        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            average_loss = epoch_loss / max(batch_count, 1)
            print(
                f"[TRAIN] epoch={epoch:04d} "
                f"loss={average_loss:.6f} "
                f"direction={last_metrics.get('direction', 0.0):.6f} "
                f"quality={last_metrics.get('quality', 0.0):.6f} "
                f"price={last_metrics.get('price', 0.0):.6f}"
            )

    model.eval()
    summary: Dict[str, Any] = {
        **normalized["stats"],
        "input_features": int(scaled_features.shape[1]),
        "price_target_fields": list(SETUP_PRICE_TARGET_FIELDS),
        "epochs": int(epochs),
        "batch_size": effective_batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "final_metrics": last_metrics,
    }
    summary.setdefault("total_examples", int(features.shape[0]))
    return model, scaler, summary


# ============================================================================
# IDENTITY AND DATASET HELPERS
# ============================================================================

def _resolve_model_identity(
    market: str,
    identity: Optional[ClientIdentity] = None,
    user_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
) -> ClientIdentity:
    """Resolve and validate the complete TradingBot identity."""
    normalized_market = normalize_market(market)
    if identity is not None:
        if not isinstance(identity, ClientIdentity):
            raise TypeError("identity must be a ClientIdentity instance.")
        if identity.market != normalized_market:
            raise ValueError(
                f"Identity market {identity.market} does not match {normalized_market}."
            )
        return identity

    return resolve_identity(
        market=normalized_market,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )


def _dataset_timeframes(dataset_bundle: Mapping[str, Any]) -> list[str]:
    """Read the stable timeframe list from a prepared bundle."""
    historical = dataset_bundle.get("historical_data", {})
    timeframes = historical.get("timeframes", []) if isinstance(historical, Mapping) else []
    selected = [
        str(value).strip().upper()
        for value in timeframes
        if str(value).strip().upper() in SUPPORTED_TIMEFRAMES
    ]
    return [timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe in set(selected)]


# ============================================================================
# MARKET TRAINING ENTRY POINT
# ============================================================================

def train_market(
    market: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Train and save one model after a labelled dataset is available.

    Preparing or storing a Training Data Bundle never calls this function.
    The current data module deliberately delegates label construction to the
    future dataset module, so this entry point reports that dependency clearly
    until that stage is implemented.
    """
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )

    bundle = prepare_training_data(
        normalized_market,
        identity=resolved,
    )
    validation = bundle.get("validation", {})
    if not isinstance(validation, Mapping) or not validation.get("valid", False):
        errors = validation.get("errors", []) if isinstance(validation, Mapping) else []
        raise RuntimeError(
            "Training Data Bundle is not valid: " + "; ".join(str(item) for item in errors)
        )

    report = bundle.get("report", {})
    if not isinstance(report, Mapping) or not report.get("ready_for_next_stage", False):
        issues = report.get("issues", []) if isinstance(report, Mapping) else []
        raise RuntimeError(
            "Training Data Bundle is not ready for labelled dataset generation: "
            + "; ".join(str(item) for item in issues)
        )

    setups = bundle["learning_setups"]["normalized"]
    candles_by_timeframe = bundle["historical_data"]["candles_by_timeframe"]
    try:
        labelled_dataset = build_training_dataset(
            market=normalized_market,
            setups=setups,
            candles_by_timeframe=candles_by_timeframe,
        )
    except NotImplementedError as error:
        raise RuntimeError(
            "Model training is not available yet. The labelled dataset pipeline "
            "in engine/dataset.py must be completed first. Training Data Bundle "
            "preparation remains available and does not train model weights."
        ) from error

    model, scaler, training_summary = train_model(
        dataset=labelled_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    training_summary["identity"] = resolved.as_dict()
    training_summary["input_timeframes"] = _dataset_timeframes(bundle)
    training_summary["model_output_schema"] = {
        "price_targets": list(SETUP_PRICE_TARGET_FIELDS),
        "direction_classes": list(DIRECTION_TO_INDEX),
        "timeframe_classes": list(SUPPORTED_TIMEFRAMES),
    }

    saved_path = save_model(
        normalized_market,
        model,
        scaler,
        training_summary,
        identity=resolved,
    )

    return {
        "training": {
            **training_summary,
            "input_setup_count": len(setups),
        },
        "model": {
            **resolved.as_dict(),
            "market": normalized_market,
            "model_path": str(saved_path),
        },
    }


# ============================================================================
# INFERENCE
# ============================================================================

def predict_from_features(
    model: TradingMicroModel,
    scaler: FeatureScaler,
    features: np.ndarray,
) -> Dict[str, Any]:
    """Predict direction, setup context and five ATR-normalized price targets."""
    if not isinstance(model, TradingMicroModel):
        raise TypeError("model must be a TradingMicroModel instance.")

    one = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if one.shape[1] != model.input_size:
        raise ValueError(
            f"Received {one.shape[1]} input features, but this model expects "
            f"{model.input_size}. The TradingBot timeframe configuration may "
            "have changed since this model was trained."
        )
    if not np.all(np.isfinite(one)):
        raise ValueError("Inference features contain NaN or infinite values.")

    scaled = apply_scaler(one, scaler)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(scaled))

    direction_probabilities = torch.softmax(outputs["direction"], dim=1)[0].cpu().numpy()
    setup_probability = float(torch.sigmoid(outputs["setup_probability"])[0, 0].cpu().item())
    setup_tf_probabilities = torch.softmax(outputs["setup_timeframe"], dim=1)[0].cpu().numpy()
    entry_tf_probabilities = torch.softmax(outputs["entry_timeframe"], dim=1)[0].cpu().numpy()
    price_offsets = outputs["price_offsets"][0].cpu().numpy()

    direction_by_index = {index: name for name, index in DIRECTION_TO_INDEX.items()}
    direction_index = int(np.argmax(direction_probabilities))
    setup_tf_index = int(np.argmax(setup_tf_probabilities))
    entry_tf_index = int(np.argmax(entry_tf_probabilities))

    offsets_by_name = {
        field_name: float(price_offsets[index])
        for index, field_name in enumerate(SETUP_PRICE_TARGET_FIELDS)
    }

    return {
        "direction": direction_by_index[direction_index],
        "direction_probabilities": {
            name: float(direction_probabilities[index])
            for name, index in DIRECTION_TO_INDEX.items()
        },
        "setup_probability": setup_probability,
        "setup_timeframe": SUPPORTED_TIMEFRAMES[setup_tf_index],
        "setup_timeframe_probability": float(setup_tf_probabilities[setup_tf_index]),
        "entry_timeframe": SUPPORTED_TIMEFRAMES[entry_tf_index],
        "entry_timeframe_probability": float(entry_tf_probabilities[entry_tf_index]),
        "price_offsets_atr": offsets_by_name,
    }


def predict_market(
    market: str,
    target_time: int,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Load one TradingBot model and predict from point-in-time candle data."""
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    model, scaler, checkpoint = load_model(normalized_market, identity=resolved)

    training_summary = checkpoint.get("training_summary", {})
    trained_timeframes = (
        training_summary.get("input_timeframes")
        if isinstance(training_summary, Mapping)
        else None
    )
    if isinstance(trained_timeframes, Sequence) and not isinstance(trained_timeframes, (str, bytes)) and trained_timeframes:
        candles_by_timeframe = load_candles_for_timeframes(
            normalized_market,
            trained_timeframes,
            identity=resolved,
        )
    else:
        candles_by_timeframe = load_all_market_candles(
            normalized_market,
            identity=resolved,
        )

    features = market_state_at(candles_by_timeframe, int(target_time))
    prediction = predict_from_features(model, scaler, features)
    current_price = current_price_at(candles_by_timeframe, int(target_time))
    atr = atr_at(candles_by_timeframe, int(target_time))

    if current_price is not None:
        prediction["price_candidates"] = {
            field_name: float(current_price + offset * atr)
            for field_name, offset in prediction["price_offsets_atr"].items()
        }

    prediction["market"] = normalized_market
    prediction["target_time"] = int(target_time)
    prediction["identity"] = resolved.as_dict()
    return prediction


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def show_model_info(
    market: str,
    user_id: Optional[str] = None,
    *,
    strategy_id: Optional[str] = None,
    tradingbot_id: Optional[str] = None,
    identity: Optional[ClientIdentity] = None,
) -> Dict[str, Any]:
    """Print and return checkpoint metadata for one TradingBot model."""
    normalized_market = normalize_market(market)
    resolved = _resolve_model_identity(
        normalized_market,
        identity=identity,
        user_id=user_id,
        strategy_id=strategy_id,
        tradingbot_id=tradingbot_id,
    )
    model, _, checkpoint = load_model(normalized_market, identity=resolved)
    summary = checkpoint.get("training_summary", {})

    result = {
        "identity": resolved.as_dict(),
        "market": normalized_market,
        "path": str(model_path(normalized_market, identity=resolved)),
        "input_features": int(model.input_size),
        "hidden_sizes": list(model.hidden_sizes),
        "price_target_fields": list(SETUP_PRICE_TARGET_FIELDS),
        "timeframes": list(
            summary.get("input_timeframes", checkpoint.get("timeframes", SUPPORTED_TIMEFRAMES))
            if isinstance(summary, Mapping)
            else checkpoint.get("timeframes", SUPPORTED_TIMEFRAMES)
        ),
        "training_summary": dict(summary) if isinstance(summary, Mapping) else {},
        "trained_at": checkpoint.get("trained_at"),
        "format_version": checkpoint.get("format_version"),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional local or authenticated identity arguments to a command."""
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--strategy-id", type=str, default=None)
    parser.add_argument("--tradingbot-id", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the internal AI model command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train, inspect and query Spectline numerical trading models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a TradingBot model.")
    train_parser.add_argument("market", type=str)
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    _add_identity_arguments(train_parser)

    info_parser = subparsers.add_parser("info", help="Show a saved model checkpoint.")
    info_parser.add_argument("market", type=str)
    _add_identity_arguments(info_parser)

    predict_parser = subparsers.add_parser("predict", help="Predict a historical market state.")
    predict_parser.add_argument("market", type=str)
    predict_parser.add_argument("time", type=str)
    _add_identity_arguments(predict_parser)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the model command-line interface."""
    set_seed(SEED)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        identity_args = {
            "user_id": args.user_id,
            "strategy_id": args.strategy_id,
            "tradingbot_id": args.tradingbot_id,
        }

        if args.command == "train":
            result = train_market(
                market=args.market,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                **identity_args,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0

        if args.command == "info":
            show_model_info(args.market, **identity_args)
            return 0

        if args.command == "predict":
            try:
                target_time = int(float(args.time))
            except (TypeError, ValueError, OverflowError):
                parsed = datetime.fromisoformat(str(args.time).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                target_time = int(parsed.timestamp())

            result = predict_market(
                market=args.market,
                target_time=target_time,
                **identity_args,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return 0

        parser.error(f"Unknown command: {args.command}")
    except Exception as error:
        print(f"[AI ERROR] {error}")
        return 1

    return 0


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "MODEL_FORMAT_VERSION",
    "DEFAULT_HIDDEN_SIZES",
    "PRICE_TARGET_SIZE",
    "PRICE_TARGET_INDEX",
    "TradingMicroModel",
    "masked_smooth_l1",
    "compute_loss",
    "train_model",
    "train_market",
    "predict_from_features",
    "predict_market",
    "show_model_info",
    "build_parser",
    "main",
]
