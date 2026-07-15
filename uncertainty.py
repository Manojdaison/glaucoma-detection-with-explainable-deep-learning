"""
uncertainty.py — Monte Carlo Dropout and Test-Time Augmentation uncertainty estimation.
Glaucoma Detection Framework | Manoj | VIT Chennai

MC Dropout:
  Run inference N=50 times with dropout ENABLED (training=True).
  Variance across predictions → epistemic uncertainty.

TTA (Test-Time Augmentation):
  Predict on 5 augmented versions of same image → aleatory uncertainty.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO DROPOUT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def mc_dropout_predict(model: tf.keras.Model,
                       img_array: np.ndarray,
                       n_passes: int = config.MC_DROPOUT_PASSES) -> dict:
    """
    Run Monte Carlo Dropout inference.

    Calls model with training=True to keep Dropout layers active,
    collecting N stochastic forward passes.

    Args:
        model:     Keras model with Dropout layer(s)
        img_array: Preprocessed image (1, H, W, 3)
        n_passes:  Number of stochastic passes (default: 50)

    Returns:
        dict {
            mean:        Mean prediction probability
            std:         Standard deviation (uncertainty)
            ci_lower:    2.5th percentile (95% CI lower bound)
            ci_upper:    97.5th percentile (95% CI upper bound)
            uncertainty: std (alias)
            all_preds:   Array of all N predictions
        }
    """
    preds = []
    img_tensor = tf.cast(img_array, tf.float32)

    for _ in range(n_passes):
        # training=True activates Dropout during inference
        pred = model(img_tensor, training=True)
        preds.append(float(pred[0][0]))

    preds = np.array(preds)

    return {
        "mean":      float(np.mean(preds)),
        "std":       float(np.std(preds)),
        "ci_lower":  float(np.percentile(preds, 2.5)),
        "ci_upper":  float(np.percentile(preds, 97.5)),
        "uncertainty": float(np.std(preds)),
        "all_preds": preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def tta_predict(model: tf.keras.Model,
                img_array: np.ndarray,
                n_versions: int = config.TTA_VERSIONS) -> dict:
    """
    Predict on N augmented versions of the input image.

    Augmentations applied:
      - Original (no augmentation)
      - Horizontal flip
      - Vertical flip
      - 90° rotation
      - Brightness increase (+20%)

    Args:
        model:      Keras model
        img_array:  Preprocessed image (1, H, W, 3)
        n_versions: Number of TTA versions (default: 5)

    Returns:
        dict {
            mean:        Mean prediction
            std:         Standard deviation
            ci_lower:    2.5th percentile
            ci_upper:    97.5th percentile
            all_preds:   All predictions
        }
    """
    img = img_array[0]  # (H, W, 3)
    augmented_versions = [img]   # Original

    # Horizontal flip
    augmented_versions.append(np.fliplr(img))

    # Vertical flip
    augmented_versions.append(np.flipud(img))

    # 90° rotation
    augmented_versions.append(np.rot90(img, k=1))

    # Brightness increase (+20%)
    bright = np.clip(img * 1.2, -1.0, 1.0)
    augmented_versions.append(bright)

    # Use only n_versions
    augmented_versions = augmented_versions[:n_versions]

    preds = []
    for aug_img in augmented_versions:
        batch = np.expand_dims(aug_img, axis=0)
        pred  = float(model.predict(batch, verbose=0)[0][0])
        preds.append(pred)

    preds = np.array(preds)
    return {
        "mean":      float(np.mean(preds)),
        "std":       float(np.std(preds)),
        "ci_lower":  float(np.percentile(preds, 2.5)),
        "ci_upper":  float(np.percentile(preds, 97.5)),
        "all_preds": preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY REPORT FOR DATASET
# ─────────────────────────────────────────────────────────────────────────────
def generate_uncertainty_report(model: tf.keras.Model,
                                val_paths: list,
                                val_labels: list,
                                model_name: str = "fullimage") -> pd.DataFrame:
    """
    Compute MC Dropout and TTA uncertainty for all validation images.

    Args:
        model:      Trained Keras model
        val_paths:  Validation image file paths
        val_labels: Ground truth labels
        model_name: For output naming

    Returns:
        DataFrame with per-image uncertainty metrics
    """
    from gradcam import preprocess_image

    records = []

    print(f"[UNCERTAINTY] Running MC Dropout ({config.MC_DROPOUT_PASSES} passes) "
          f"+ TTA ({config.TTA_VERSIONS} versions) on {len(val_paths)} images ...")

    for img_path, label in tqdm(zip(val_paths, val_labels),
                                total=len(val_paths),
                                desc="  Uncertainty"):
        try:
            _, img_array = preprocess_image(img_path)

            mc  = mc_dropout_predict(model, img_array)
            tta = tta_predict(model, img_array)

            records.append({
                "filename":      os.path.basename(img_path),
                "label":         int(label),
                "diagnosis":     "Glaucoma" if label == 1 else "Normal",
                "mc_mean":       round(mc["mean"], 4),
                "mc_std":        round(mc["std"], 4),
                "mc_ci_lower":   round(mc["ci_lower"], 4),
                "mc_ci_upper":   round(mc["ci_upper"], 4),
                "tta_mean":      round(tta["mean"], 4),
                "tta_std":       round(tta["std"], 4),
                "high_uncertainty": mc["std"] > 0.15,
            })
        except Exception as e:
            print(f"[UNCERTAINTY] Error {img_path}: {e}")

    df = pd.DataFrame(records)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.REPORTS_DIR,
                            f"uncertainty_{model_name}.csv")
    df.to_csv(csv_path, index=False)

    # Text summary
    report_lines = [
        "═" * 60,
        f"  UNCERTAINTY REPORT — {model_name}",
        "═" * 60,
        f"\nMC Dropout ({config.MC_DROPOUT_PASSES} passes):",
        f"  Mean uncertainty (std): {df['mc_std'].mean():.4f}",
        f"  High uncertainty cases (std>0.15): "
        f"{df['high_uncertainty'].sum()} / {len(df)}",
        f"\nTTA ({config.TTA_VERSIONS} versions):",
        f"  Mean TTA std: {df['tta_std'].mean():.4f}",
        "\n" + "═" * 60,
    ]
    report = "\n".join(report_lines)
    txt_path = os.path.join(config.REPORTS_DIR,
                            f"uncertainty_{model_name}.txt")
    with open(txt_path, "w") as f:
        f.write(report)

    print(report)
    print(f"[UNCERTAINTY] Saved → {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY CALIBRATION PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_uncertainty_distribution(df: pd.DataFrame,
                                  model_name: str = "fullimage") -> str:
    """
    Plot distribution of MC Dropout uncertainty per diagnosis group.

    Returns:
        Path to saved figure
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for diag, color in [("Glaucoma", "#E53935"), ("Normal", "#1E88E5")]:
        subset = df[df["diagnosis"] == diag]["mc_std"]
        ax.hist(subset, bins=20, alpha=0.6, label=diag, color=color,
                edgecolor="white")

    ax.set_xlabel("MC Dropout Uncertainty (std)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Prediction Uncertainty Distribution — {model_name}",
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR,
                             f"uncertainty_{model_name}.png")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[UNCERTAINTY] Plot saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[UNCERTAINTY] Module ready.")
    print(f"  MC Dropout passes : {config.MC_DROPOUT_PASSES}")
    print(f"  TTA versions      : {config.TTA_VERSIONS}")
    print("  Use generate_uncertainty_report(model, val_paths, val_labels)")