"""
train.py — Master training script.
Glaucoma Detection Framework | Manoj | VIT Chennai

Orchestrates:
  1. (Optional) Data preparation from ZIP
  2. Region cropping: generates data_od/ and data_pp/ from data/
  3. Trains 3 models: fullimage, od, pp

Usage:
  # Prepare data from ZIP first (one-time):
  python train.py --prepare_only --zip_path /path/to/ACRIMA.zip

  # Then train all models:
  python train.py

  # Train a single model:
  python train.py --model fullimage

FIXES FROM ORIGINAL:
  - Removed hardcoded Windows zip path (C:\\Users\\manoj\\...).
    zip_path is now a required argument when --prepare_only is used.
  - Removed duplicate plot_training_curves function (also existed in
    model_builder.py with identical logic). Kept one version here.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from config import (
    SEED, PLOTS_DIR, MODEL_REGISTRY, FIGURE_DPI
)
from dataset import (
    prepare_from_zip, collect_filepaths,
    stratified_split, get_class_weights, build_generators
)
from model_builder import build_mobilenetv2, train_two_phase
from region_crop import precrop_all

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history_dict: dict, model_name: str) -> None:
    """
    Save training vs validation loss and accuracy curves (Phase 1 + Phase 2).

    Args:
        history_dict: Output of train_two_phase() with keys 'phase1_history',
                      'phase2_history'
        model_name:   Used for figure title and filename
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    h1 = history_dict["phase1_history"].history
    h2 = history_dict["phase2_history"].history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    phase_boundary = len(h1.get("loss", []))

    for ax, metric, ylabel in [
        (axes[0], "loss",     "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        train_vals = h1.get(metric, []) + h2.get(metric, [])
        val_vals   = h1.get(f"val_{metric}", []) + h2.get(f"val_{metric}", [])
        epochs     = range(1, len(train_vals) + 1)

        ax.plot(epochs, train_vals, linewidth=2,
                label="Train", color="#1F77B4")
        ax.plot(epochs, val_vals,   linewidth=2,
                label="Val",   color="#D62728", linestyle="--")
        if phase_boundary > 0:
            ax.axvline(x=phase_boundary, color="gray",
                       linestyle=":", alpha=0.7, label="Fine-tune start")
        ax.set_title(f"{model_name} — {ylabel}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Training curves → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_single_model(model_key: str) -> None:
    """
    Run the full training pipeline for one model variant.

    Args:
        model_key: 'fullimage', 'od', or 'pp'
    """
    cfg        = MODEL_REGISTRY[model_key]
    data_dir   = cfg["data_dir"]
    model_name = f"{model_key}_model"

    print(f"\n{'='*55}")
    print(f"  {model_key.upper()} MODEL")
    print(f"{'='*55}")

    paths, labels         = collect_filepaths(data_dir)
    X_tr, X_vl, y_tr, y_vl = stratified_split(paths, labels)
    cw                    = get_class_weights(y_tr)

    train_gen, val_gen = build_generators(
        X_tr, X_vl, y_tr, y_vl,
        preprocess_fn=preprocess_input
    )

    model        = build_mobilenetv2()
    history_dict = train_two_phase(model, train_gen, val_gen, model_name, cw)
    plot_training_curves(history_dict, model_name)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Glaucoma Detection Training Script"
    )
    parser.add_argument(
        "--zip_path", default=None,
        help="Path to ACRIMA.zip (required when using --prepare_only)"
    )
    parser.add_argument(
        "--data_dir", default="data/",
        help="Root data directory (default: data/)"
    )
    parser.add_argument(
        "--prepare_only", action="store_true",
        help="Only extract and sort the dataset ZIP, then exit"
    )
    parser.add_argument(
        "--model", default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Train only one model variant (default: train all three)"
    )
    parser.add_argument(
        "--skip_crop", action="store_true",
        help="Skip region cropping step (use if data_od/ and data_pp/ already exist)"
    )
    args = parser.parse_args()

    # ── Optional: extract and sort dataset ───────────────────────────────────
    if args.prepare_only:
        if args.zip_path is None:
            parser.error("--zip_path is required when using --prepare_only")
        prepare_from_zip(args.zip_path, args.data_dir)
        return

    # ── Region cropping ──────────────────────────────────────────────────────
    if not args.skip_crop:
        print("\n[STEP 1] Generating region crops (OD + PP) ...")
        print("[INFO]  This re-runs with the FIXED detect_disc_centre and")
        print("[INFO]  get_peripapillary_crop. Delete data_od/ and data_pp/")
        print("[INFO]  first if they contain crops from the old broken version.")
        precrop_all(source_dir=args.data_dir, quality_check=True)
    else:
        print("[STEP 1] Skipping region crop (--skip_crop set).")

    # ── Train model(s) ───────────────────────────────────────────────────────
    models_to_train = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    print(f"\n[STEP 2] Training {len(models_to_train)} model(s): "
          f"{models_to_train}")

    for key in models_to_train:
        train_single_model(key)

    print("\n[DONE] All training complete.")
    print(f"[DONE] Run threshold_optimizer.py next to find per-model "
          f"optimal thresholds.")


if __name__ == "__main__":
    main()