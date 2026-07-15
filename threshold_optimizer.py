"""
threshold_optimizer.py — Per-model optimal classification threshold finder.
Glaucoma Detection Framework | Manoj | VIT Chennai

The default 0.5 threshold is rarely optimal. This script:
  1. Loads each trained model.
  2. Runs inference on the validation set.
  3. Computes the ROC curve.
  4. Finds the threshold that maximises Youden's J statistic
     (= Sensitivity + Specificity - 1) — balances FP and FN rate.
  5. Saves all thresholds to outputs/reports/thresholds.json
     (which app.py and ensemble.py load automatically).

Usage:
    python threshold_optimizer.py

Run this AFTER training all three models.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import collect_filepaths, stratified_split
from model_builder import load_model

# ─────────────────────────────────────────────────────────────────────────────
# BUILD VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────
def build_val_set(data_dir: str) -> tuple:
    """
    Return (val_paths, val_labels) using the SAME split seed as training.

    Args:
        data_dir: Root directory with glaucoma/ and normal/ sub-folders

    Returns:
        (val_paths: list, val_labels: list)
    """
    paths, labels = collect_filepaths(data_dir)
    _, val_paths, _, val_labels = stratified_split(paths, labels)
    return val_paths, val_labels


def predict_on_val(model: tf.keras.Model, val_paths: list) -> np.ndarray:
    """
    Run inference on a list of image file paths.

    Args:
        model:     Loaded Keras model
        val_paths: List of absolute file paths

    Returns:
        Array of glaucoma probabilities, shape (N,)
    """
    preds = []
    for path in val_paths:
        import cv2
        img = cv2.imread(path)
        if img is None:
            preds.append(0.5)   # fallback for missing file
            continue
        img   = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        x     = np.expand_dims(preprocess_input(img), 0)
        prob  = float(model.predict(x, verbose=0)[0, 0])
        preds.append(prob)
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# YOUDEN'S J THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
def youden_threshold(y_true: np.ndarray,
                     y_prob: np.ndarray) -> tuple:
    """
    Find the threshold that maximises Youden's J (sensitivity + specificity - 1).

    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted glaucoma probabilities

    Returns:
        (best_threshold, roc_auc, fpr_array, tpr_array, thresholds_array)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc              = auc(fpr, tpr)
    j_scores             = tpr - fpr             # = Sensitivity + Specificity - 1
    best_idx             = int(np.argmax(j_scores))
    best_threshold       = float(thresholds[best_idx])
    return best_threshold, roc_auc, fpr, tpr, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# PLOT ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, save_path: str) -> None:
    """
    Plot ROC curves for all models with their optimal thresholds marked.

    Args:
        results:   Dict from optimize_all_thresholds()
        save_path: Output PNG path
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = {"fullimage": "#2196F3", "od": "#FF5722", "pp": "#4CAF50"}

    for key, info in results.items():
        fpr = info["fpr"]
        tpr = info["tpr"]
        auc_val = info["auc"]
        thr     = info["threshold"]
        c       = colors.get(key, "#999")

        ax.plot(fpr, tpr, color=c, linewidth=2,
                label=f"{key} (AUC={auc_val:.3f}, thr={thr:.3f})")
        # Mark the Youden point
        best_fpr = info["best_fpr"]
        best_tpr = info["best_tpr"]
        ax.scatter([best_fpr], [best_tpr], color=c, s=120, zorder=5,
                   edgecolors="white", linewidths=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=12)
    ax.set_title("ROC Curves — Optimal Thresholds (Youden's J)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[THRESHOLD] ROC curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: OPTIMIZE ALL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
def optimize_all_thresholds() -> dict:
    """
    Find optimal thresholds for all three models and save to JSON.

    Returns:
        Dict mapping model_key → optimal threshold float
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR,   exist_ok=True)

    saved_thresholds = {}
    plot_data        = {}

    for key, info in config.MODEL_REGISTRY.items():
        print(f"\n[THRESHOLD] Optimizing: {key}")

        model = load_model(key)
        if model is None:
            print(f"[THRESHOLD]   Skipping — model file missing.")
            saved_thresholds[key] = 0.5
            continue

        val_paths, val_labels = build_val_set(info["data_dir"])
        y_true = np.array(val_labels)
        y_prob = predict_on_val(model, val_paths)

        thr, roc_auc, fpr, tpr, thresholds_arr = youden_threshold(y_true, y_prob)

        # Find the FPR/TPR at the optimal threshold for plotting
        best_idx = int(np.argmax(tpr - fpr))
        best_fpr = float(fpr[best_idx])
        best_tpr = float(tpr[best_idx])

        # Classification report at optimal threshold
        y_pred = (y_prob >= thr).astype(int)
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Glaucoma"]
        )

        print(f"[THRESHOLD]   AUC          : {roc_auc:.4f}")
        print(f"[THRESHOLD]   Youden thr   : {thr:.4f}")
        print(f"[THRESHOLD]   Sensitivity  : {best_tpr:.3f}")
        print(f"[THRESHOLD]   Specificity  : {1 - best_fpr:.3f}")
        print(f"[THRESHOLD]   Report:\n{report}")

        saved_thresholds[key] = round(thr, 4)
        plot_data[key] = {
            "fpr":      fpr,
            "tpr":      tpr,
            "threshold": thr,
            "auc":       roc_auc,
            "best_fpr":  best_fpr,
            "best_tpr":  best_tpr,
        }

        # Save per-model report
        report_path = os.path.join(
            config.REPORTS_DIR, f"{key}_threshold_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(f"Model: {key}\n")
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Optimal threshold (Youden's J): {thr:.4f}\n")
            f.write(f"Sensitivity at threshold: {best_tpr:.4f}\n")
            f.write(f"Specificity at threshold: {1 - best_fpr:.4f}\n\n")
            f.write(report)
        print(f"[THRESHOLD]   Report saved → {report_path}")

    # Save all thresholds to JSON (loaded by app.py on startup)
    with open(config.THRESHOLDS_FILE, "w") as f:
        json.dump(saved_thresholds, f, indent=2)
    print(f"\n[THRESHOLD] Thresholds saved → {config.THRESHOLDS_FILE}")
    print(f"[THRESHOLD] Contents: {saved_thresholds}")

    # ROC plot
    if plot_data:
        plot_roc_curves(
            plot_data,
            os.path.join(config.PLOTS_DIR, "threshold_roc_curves.png")
        )

    return saved_thresholds


if __name__ == "__main__":
    optimize_all_thresholds()