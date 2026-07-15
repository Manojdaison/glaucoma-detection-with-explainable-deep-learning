"""
model_builder.py — MobileNetV2 model construction, compilation, and training.
Glaucoma Detection Framework | Manoj | VIT Chennai

Architecture:
  MobileNetV2 (ImageNet pretrained)
  → GlobalAveragePooling2D
  → Dense(256, ReLU) → BatchNormalization → Dropout(0.5)
  → Dense(1, sigmoid)

Two-phase training:
  Phase 1: frozen backbone, LR=1e-3, up to 20 epochs
  Phase 2: unfreeze last 30 layers, LR=1e-5, up to 20 epochs

FIXES FROM ORIGINAL:
  - compile_model: Added label_smoothing=config.LABEL_SMOOTHING (0.1) to
    BinaryCrossentropy. Plain cross-entropy with hard labels (0/1) drives
    the sigmoid towards 0 and 1 with no resistance, producing overconfident
    predictions. Label smoothing sets targets to [0.1, 0.9] instead of [0, 1],
    regularising confidence and improving calibration — especially important
    for the OD/PP models which are trained on noisier cropped inputs.

  - Removed duplicate plot_training_curves function. It also exists in
    train.py. Having two versions meant one was always out of sync.
    train.py's version is kept since it's the one called by main().
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)

import config

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_mobilenetv2(input_shape: tuple = (config.IMG_HEIGHT,
                                             config.IMG_WIDTH,
                                             config.IMG_CHANNELS)) -> Model:
    """
    Build MobileNetV2-based binary classifier for glaucoma detection.

    Args:
        input_shape: (H, W, C) — default (224, 224, 3)

    Returns:
        Uncompiled Keras Model (backbone frozen for Phase 1)
    """
    backbone = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False   # Phase 1: frozen

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x      = backbone(inputs, training=False)
    x      = layers.GlobalAveragePooling2D(name="gap")(x)
    x      = layers.Dense(256, activation="relu", name="fc_256")(x)
    x      = layers.BatchNormalization(name="bn_256")(x)
    x      = layers.Dropout(0.5, name="dropout_0_5",
                             seed=config.SEED)(x)
    out    = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inputs, outputs=out)


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def compile_model(model: Model, learning_rate: float) -> None:
    """
    Compile model with Adam + label-smoothed BinaryCrossentropy.

    Label smoothing (config.LABEL_SMOOTHING = 0.1) replaces hard targets
    [0, 1] with soft targets [0.1, 0.9]. This:
      • Prevents sigmoid from saturating towards 0/1 during training.
      • Reduces overconfidence in final predictions.
      • Improves calibration — especially important for the OD/PP models
        trained on noisier, cropped inputs.

    Args:
        model:         Keras model (modified in-place)
        learning_rate: Adam learning rate
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def get_callbacks(model_name: str, phase: int) -> list:
    """
    Build training callbacks for one training phase.

    Args:
        model_name: Name used for checkpoint and log filenames
        phase:      1 (feature extraction) or 2 (fine-tuning)

    Returns:
        List of Keras Callback objects
    """
    os.makedirs(config.MODELS_DIR,  exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(config.MODELS_DIR,   f"{model_name}.h5")
    log_path        = os.path.join(config.REPORTS_DIR,
                                   f"{model_name}_phase{phase}_log.csv")

    return [
        # Save best model by validation AUC
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            save_best_only=True,
            verbose=1,
        ),
        # Stop early and restore best weights
        EarlyStopping(
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve LR when val_loss stalls
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN,
            verbose=1,
        ),
        # CSV log — append on Phase 2 so the file covers both phases
        CSVLogger(log_path, append=(phase == 2)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PHASE TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_two_phase(model: Model,
                    train_gen,
                    val_gen,
                    model_name: str,
                    class_weights: dict = None) -> dict:
    """
    Train model in two phases (feature extraction → fine-tuning).

    Phase 1 — Frozen backbone:
      LR = config.LR_PHASE1 (1e-3), up to config.EPOCHS_PHASE1 (20) epochs.
      Only the Dense head is trained. EarlyStopping with restore_best_weights
      will stop earlier if val_auc plateaus.

    Phase 2 — Fine-tuning:
      LR = config.LR_PHASE2 (1e-5), last config.UNFREEZE_LAYERS (30) unfrozen.
      Very small LR prevents catastrophic forgetting of ImageNet features.

    Args:
        model:         Freshly built (or Phase-1-trained) Keras Model
        train_gen:     Training generator
        val_gen:       Validation generator
        model_name:    Identifier for checkpoint and log filenames
        class_weights: Optional {0: w, 1: w} to rebalance loss

    Returns:
        {'phase1_history': History, 'phase2_history': History}
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    # ── Phase 1: Feature Extraction ──────────────────────────────────────────
    print(f"\n[TRAIN] Phase 1 — LR={config.LR_PHASE1}, "
          f"frozen backbone, up to {config.EPOCHS_PHASE1} epochs")

    compile_model(model, config.LR_PHASE1)
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_PHASE1,
        callbacks=get_callbacks(model_name, phase=1),
        class_weight=class_weights,
        verbose=1,
    )

    # ── Phase 2: Fine-Tuning ─────────────────────────────────────────────────
    print(f"\n[TRAIN] Phase 2 — LR={config.LR_PHASE2}, "
          f"unfreeze last {config.UNFREEZE_LAYERS} layers, "
          f"up to {config.EPOCHS_PHASE2} epochs")

    # Find and partially unfreeze backbone
    backbone = next(
        (lyr for lyr in model.layers if isinstance(lyr, Model)),
        None
    )
    if backbone is not None:
        backbone.trainable = True
        for lyr in backbone.layers[:-config.UNFREEZE_LAYERS]:
            lyr.trainable = False
        n_trainable = sum(l.trainable for l in backbone.layers)
        print(f"[TRAIN] Backbone trainable layers: "
              f"{n_trainable} / {len(backbone.layers)}")
    else:
        print("[TRAIN] WARNING: Backbone not identified — training full model.")

    compile_model(model, config.LR_PHASE2)
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_PHASE2,
        callbacks=get_callbacks(model_name, phase=2),
        class_weight=class_weights,
        verbose=1,
    )

    saved_path = os.path.join(config.MODELS_DIR, f"{model_name}.h5")
    print(f"\n[TRAIN] {model_name} complete. Best model → {saved_path}")
    return {"phase1_history": history1, "phase2_history": history2}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str) -> Model:
    """
    Load a saved model from disk by registry key.

    Args:
        model_name: Key in config.MODEL_REGISTRY ('fullimage', 'od', 'pp')

    Returns:
        Loaded Keras Model, or None if file missing
    """
    if model_name not in config.MODEL_REGISTRY:
        print(f"[MODEL] Unknown model key: '{model_name}'")
        return None

    model_path = config.MODEL_REGISTRY[model_name]["model_path"]
    if not os.path.exists(model_path):
        print(f"[MODEL] Model file not found: {model_path}")
        return None

    print(f"[MODEL] Loading '{model_name}' from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"[MODEL] ERROR loading {model_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_mobilenetv2()
    compile_model(m, config.LR_PHASE1)
    m.summary()
    print(f"\n[MODEL] Total parameters : {m.count_params():,}")
    print(f"[MODEL] Label smoothing  : {config.LABEL_SMOOTHING}")