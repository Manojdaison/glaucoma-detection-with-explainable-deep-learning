"""
ensemble.py — Weighted ensemble prediction across all three region models.
Glaucoma Detection Framework | Manoj | VIT Chennai

Combines fullimage, OD, and PP model predictions using configurable weights.
Weights are defined in config.ENSEMBLE_WEIGHTS and should be re-tuned after
retraining the OD/PP models with the fixed region_crop.py.

Usage:
    from ensemble import EnsemblePredictor
    ep  = EnsemblePredictor()
    ep.load_all()
    result = ep.predict(img_rgb_numpy)   # (H, W, 3) uint8
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config
from model_builder import load_model


class EnsemblePredictor:
    """
    Weighted average ensemble of the three glaucoma detection models.

    Args:
        weights: Dict mapping model key → weight. Must sum to ~1.0.
                 Defaults to config.ENSEMBLE_WEIGHTS.
        thresholds: Dict mapping model key → decision threshold.
                    Defaults to config.THRESHOLDS.
    """

    def __init__(self,
                 weights:    dict = None,
                 thresholds: dict = None):
        self.weights    = weights    or dict(config.ENSEMBLE_WEIGHTS)
        self.thresholds = thresholds or dict(config.THRESHOLDS)
        self.models     = {}   # key → loaded Keras Model

        # Normalise weights so they sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-4:
            print(f"[ENSEMBLE] Normalising weights (sum was {total:.4f})")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def load_all(self) -> None:
        """Load all models whose weights are > 0."""
        for key in self.weights:
            m = load_model(key)
            if m is not None:
                self.models[key] = m
                print(f"[ENSEMBLE] Loaded '{key}' (weight={self.weights[key]:.2f})")
            else:
                print(f"[ENSEMBLE] WARNING: '{key}' could not be loaded — "
                      "it will be excluded from the ensemble.")

        if not self.models:
            raise RuntimeError("[ENSEMBLE] No models loaded. "
                               "Run train.py first.")

    def predict(self, img_rgb: np.ndarray) -> dict:
        """
        Run ensemble prediction on a single image.

        Args:
            img_rgb: (H, W, 3) uint8 RGB image — does NOT need to be
                     pre-resized; this function handles resizing and preprocessing.

        Returns:
            dict with keys:
              'ensemble_prob'  — weighted average probability
              'ensemble_label' — 'Glaucoma' or 'Normal'
              'per_model'      — {key: {'prob': float, 'label': str}}
        """
        if not self.models:
            raise RuntimeError("[ENSEMBLE] Call load_all() first.")

        # Resize + preprocess once (shared input shape for all models)
        from PIL import Image as PILImage
        img_resized = np.array(
            PILImage.fromarray(img_rgb).resize(
                (config.IMG_WIDTH, config.IMG_HEIGHT)
            ),
            dtype=np.float32
        )
        x = np.expand_dims(
            preprocess_input(img_resized), axis=0
        )   # (1, 224, 224, 3)

        ensemble_prob = 0.0
        per_model     = {}
        weight_used   = 0.0

        for key, model in self.models.items():
            w    = self.weights.get(key, 0.0)
            prob = float(model.predict(x, verbose=0)[0, 0])
            thr  = self.thresholds.get(key, 0.5)
            lbl  = "Glaucoma" if prob >= thr else "Normal"
            per_model[key]  = {"prob": prob, "label": lbl}
            ensemble_prob  += w * prob
            weight_used    += w

        # Renormalise if some models were missing
        if weight_used > 1e-6 and abs(weight_used - 1.0) > 1e-4:
            ensemble_prob /= weight_used

        ens_thr   = self.thresholds.get("ensemble", 0.5)
        ens_label = "Glaucoma" if ensemble_prob >= ens_thr else "Normal"

        return {
            "ensemble_prob":  ensemble_prob,
            "ensemble_label": ens_label,
            "ensemble_threshold": ens_thr,
            "per_model":      per_model,
        }

    def predict_batch(self, img_list: list) -> list:
        """
        Run ensemble prediction on a list of images.

        Args:
            img_list: List of (H, W, 3) uint8 RGB arrays

        Returns:
            List of dicts (same structure as predict())
        """
        return [self.predict(img) for img in img_list]


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2

    if len(sys.argv) < 2:
        print("Usage: python ensemble.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ep = EnsemblePredictor()
    ep.load_all()
    result = ep.predict(img_rgb)

    print(f"\n[ENSEMBLE] Result for: {img_path}")
    print(f"  Ensemble score : {result['ensemble_prob']:.4f}")
    print(f"  Ensemble label : {result['ensemble_label']}")
    print(f"  Threshold used : {result['ensemble_threshold']:.3f}")
    print("\n  Per-model breakdown:")
    for key, info in result["per_model"].items():
        print(f"    {key:12s} → {info['prob']:.4f}  ({info['label']})")