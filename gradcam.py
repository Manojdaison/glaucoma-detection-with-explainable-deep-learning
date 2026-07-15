"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).
Glaucoma Detection Framework | Manoj | VIT Chennai

ROOT CAUSE OF ValueError "Graph disconnected":
  tf.keras.Model(inputs=model.inputs, outputs=nested_layer.output) fails
  when the model was loaded from .h5 because the nested backbone's input
  tensor is NOT the same object as the outer model's input tensor.

FIX:
  Use the monkey-patch / call-capture approach:
    1. Temporarily patch target_layer.call() to store its output tensor.
    2. Run a full forward pass through model(inp, training=False).
    3. Use GradientTape watching the captured tensor to get gradients.
  This NEVER builds a new tf.keras.Model, so graph connectivity is irrelevant.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import config


# ─────────────────────────────────────────────────────────────────────────────
# LAYER SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def _find_layer(model, name):
    """Recursively find a layer by name (searches nested sub-models too)."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer(layer, name)
            if found:
                return found
    return None


def _last_conv(model):
    """Return the last Conv2D found (outer model first, then nested)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM  (graph-safe — no tf.keras.Model reconstruction)
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam(model: tf.keras.Model,
                img_array: np.ndarray,
                layer_name: str = config.GRADCAM_LAYER_NAME) -> tuple:
    """
    Compute Grad-CAM heatmap without rebuilding the Keras graph.

    Works with:
      • Models built in-session
      • Models loaded from .h5 (tf.keras.models.load_model)
      • MobileNetV2 wrapped in an outer functional model

    Args:
        model:      Any Keras model
        img_array:  (1, H, W, 3) float32, MobileNetV2-preprocessed
        layer_name: Target conv layer name (default "Conv_1")

    Returns:
        heatmap_small — (h, w) raw output, values in [0,1]
        heatmap_large — (224, 224) bilinearly upsampled
    """
    # Locate target layer
    target = _find_layer(model, layer_name)
    if target is None:
        target = _last_conv(model)
        if target is None:
            raise ValueError("[GradCAM] No conv layer found.")
        print(f"[GradCAM] '{layer_name}' not found — using fallback: {target.name}")

    # ── Capture conv output by patching target_layer.call ────────────────────
    captured_output = []
    original_call   = target.call

    def capturing_call(inputs, **kwargs):
        out = original_call(inputs, **kwargs)
        captured_output.append(out)
        return out

    target.call = capturing_call

    # ── Forward pass + GradientTape ───────────────────────────────────────────
    inp = tf.cast(img_array, tf.float32)

    try:
        with tf.GradientTape() as tape:
            # Run full model (capturing_call stores conv output as a side effect)
            preds = model(inp, training=False)

            if not captured_output:
                raise RuntimeError("[GradCAM] Conv layer output was not captured. "
                                   "Check layer name or model structure.")

            conv_out  = captured_output[-1]   # (1, h, w, C)
            tape.watch(conv_out)

            # Re-run only the remaining layers AFTER the conv output
            # We need gradients of the prediction w.r.t. conv_out.
            # Since we already have preds, re-compute from conv_out:
            score = preds[:, 0]              # glaucoma probability

        grads = tape.gradient(score, conv_out)   # (1, h, w, C)

    finally:
        target.call = original_call   # always restore, even on error

    # ── Fallback: if tape didn't capture gradient, use second-pass approach ───
    if grads is None:
        grads = _gradients_second_pass(model, inp, target, original_call)

    if grads is None:
        print("[GradCAM] WARNING: Zero gradients — returning blank heatmap.")
        return np.zeros(config.GRADCAM_FEATURE_SIZE), \
               np.zeros(config.IMG_SIZE)

    # ── Compute weighted heatmap ──────────────────────────────────────────────
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()   # (C,)
    cam    = conv_out.numpy()[0]                              # (h, w, C)
    heatmap = np.dot(cam, pooled)                             # (h, w)

    heatmap = np.maximum(heatmap, 0)   # ReLU
    mn, mx  = heatmap.min(), heatmap.max()
    if mx - mn > 1e-8:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap, resize_heatmap(heatmap)


def _gradients_second_pass(model, inp, target_layer, original_call):
    """
    Second-pass gradient computation using a tf.Variable proxy.
    Used when the first tape doesn't connect to conv_out.
    """
    captured = []

    def cap_call(inputs, **kwargs):
        out = original_call(inputs, **kwargs)
        captured.append(out)
        return out

    target_layer.call = cap_call
    try:
        with tf.GradientTape() as tape:
            _ = model(inp, training=False)
            if not captured:
                return None
            conv_out = tf.identity(captured[-1])
            tape.watch(conv_out)
            # Recompute score using captured conv output as proxy
            preds = model(inp, training=False)
            score = preds[:, 0]
        grads = tape.gradient(score, conv_out)
    finally:
        target_layer.call = original_call

    return grads


# ─────────────────────────────────────────────────────────────────────────────
# RESIZE
# ─────────────────────────────────────────────────────────────────────────────
def resize_heatmap(heatmap: np.ndarray,
                   target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """Bilinear upsample heatmap → target_size (H, W), values in [0,1]."""
    resized = cv2.resize(heatmap.astype(np.float32),
                         (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    mn, mx = resized.min(), resized.max()
    if mx - mn > 1e-8:
        resized = (resized - mn) / (mx - mn)
    return resized


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(img_rgb: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = config.GRADCAM_ALPHA) -> np.ndarray:
    """Blend jet-coloured heatmap onto RGB image (alpha = heatmap opacity)."""
    jet     = cm.get_cmap("jet")
    colored = (jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * img_rgb).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS FROM FILE PATH
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> tuple:
    """
    Load image from disk and preprocess for MobileNetV2.

    Returns:
        img_rgb   — (224, 224, 3) uint8, for display
        img_array — (1, 224, 224, 3) float32, for model
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"[GradCAM] Cannot load: {img_path}")
    bgr     = cv2.resize(bgr, config.IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pre     = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
    return rgb, np.expand_dims(pre, 0)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS FROM PIL (Streamlit uploads)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_pil(pil_image) -> tuple:
    """
    Preprocess a PIL Image for MobileNetV2 inference.

    Returns:
        img_rgb   — (224, 224, 3) uint8
        img_array — (1, 224, 224, 3) float32
    """
    from PIL import Image
    img     = pil_image.convert("RGB").resize(
                  (config.IMG_WIDTH, config.IMG_HEIGHT), Image.LANCZOS)
    rgb     = np.array(img, dtype=np.uint8)
    pre     = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
    return rgb, np.expand_dims(pre, 0)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def save_gradcam_figure(img_rgb, heatmap_large, overlay_img,
                        filename: str, prediction: float) -> str:
    """Save 3-panel Grad-CAM figure: Original | Overlay | Heatmap."""
    os.makedirs(config.GRADCAM_DIR, exist_ok=True)
    label = "Glaucoma" if prediction >= 0.5 else "Normal"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Grad-CAM | {label} (p={prediction:.3f})", fontsize=13)
    axes[0].imshow(img_rgb);            axes[0].set_title("Original");         axes[0].axis("off")
    axes[1].imshow(overlay_img);        axes[1].set_title("Grad-CAM Overlay"); axes[1].axis("off")
    im = axes[2].imshow(heatmap_large, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Heatmap");       axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(config.GRADCAM_DIR, f"{filename}_gradcam.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    print(f"[GradCAM] Target layer : {config.GRADCAM_LAYER_NAME}")
    print(f"[GradCAM] Overlay alpha: {config.GRADCAM_ALPHA}")
    print("[GradCAM] Ready — call get_gradcam(model, img_array).")