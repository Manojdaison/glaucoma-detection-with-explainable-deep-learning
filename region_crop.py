"""
region_crop.py — Optic disc detection and region-based image cropping.
Glaucoma Detection Framework | Manoj | VIT Chennai

Clinical regions:
  OD  (Optic Disc)    : radius ≤ 33px  (bright central disc)
  PP  (Peripapillary) : 33px < r ≤ 67px (RNFL assessment zone)
  BG  (Background)    : r > 67px        (non-clinical)

FIXES FROM ORIGINAL:
─────────────────────────────────────────────────────────────────────────────
1. detect_disc_centre:
   OLD: cv2.minMaxLoc → single brightest PIXEL.
        Fragile to camera flare, vessel reflections, and noise specks.
        A single hot pixel can pull the "centre" 30–40px off the real disc.
   NEW: threshold top-2% brightness → largest connected blob → CENTROID.
        Centroid of a region is much more stable than a single pixel max,
        especially since the optic disc is a 60–100px-wide bright blob, not a point.

2. get_optic_disc_crop:
   OLD: DISC_CROP_RADIUS_FRACTION = 0.15 → 66px native crop → 3.4× upsample.
        Fine disc/cup/rim detail destroyed before the model ever sees it.
   NEW: DISC_CROP_RADIUS_FRACTION = 0.22 → 98px native crop → 2.3× upsample.
        Meaningful improvement; ablation study (ablation_study.py) should
        confirm the optimal radius for ACRIMA.

3. get_peripapillary_crop:
   OLD: Filled both the disc hole AND everything outside the ring with one
        FLAT UNIFORM COLOUR. This produced an almost-identical hard-edged
        synthetic disc shape burned into every single training image — both
        Glaucoma and Normal. That's a classic shortcut-learning trap: the model
        keyed off the artificial edge rather than genuine RNFL tissue, causing
        high-confidence false-positive glaucoma predictions on new Normal images.
   NEW: Feathered (soft-transition) alpha window + per-image Gaussian blur
        for the suppressed regions. No hard edge, no dataset-wide identical
        shape, no shortcut. The model now sees real peripapillary texture
        in the ring and gradually-blurred content on both sides.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# OPTIC DISC DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_disc_centre(img_bgr: np.ndarray) -> tuple:
    """
    Detect the optic disc centre using green-channel blob detection.

    Method:
      1. Extract green channel (highest contrast for OD in fundus images).
      2. Apply Gaussian blur (51×51) to suppress noise.
      3. Threshold to the top 2% brightness — isolates the disc as a region.
      4. Take the largest connected bright component (guards against flare).
      5. Return the CENTROID of that component, not a single pixel.

    Why centroid-of-blob instead of the original single brightest pixel:
      cv2.minMaxLoc returns ONE pixel coordinate. A tiny camera flare, vessel
      reflection, or noise spike anywhere in the frame can outshine the disc
      momentarily and pull the "centre" 20–40px off. The optic disc is a wide
      (~60–100px) bright blob; averaging over it produces a stable centre that
      is unaffected by single-pixel hot spots.

    Args:
        img_bgr: BGR image (H, W, 3) — should already be resized to target size.

    Returns:
        (cx, cy): integer pixel coordinates of the optic disc centre.
    """
    h, w = img_bgr.shape[:2]

    # Green channel — optic disc has highest reflectance here
    green   = img_bgr[:, :, 1].copy()
    blurred = cv2.GaussianBlur(green, (51, 51), 0)

    # Threshold to brightest 2%: isolates disc as a region, not a point
    thresh_val  = np.percentile(blurred, 98)
    bright_mask = (blurred >= thresh_val).astype(np.uint8)

    # Largest connected component = the disc
    # (multiple disjoint specks can appear from vessel glare elsewhere)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )

    if num_labels > 1:
        # Label 0 is background; pick the biggest non-background component
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx = int(round(centroids[largest_label][0]))
        cy = int(round(centroids[largest_label][1]))
    else:
        # Fallback (degenerate image): use image centre
        cx, cy = w // 2, h // 2

    # Clamp to image bounds
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# OPTIC DISC CROP
# ─────────────────────────────────────────────────────────────────────────────
def get_optic_disc_crop(img_bgr: np.ndarray,
                        target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """
    Extract and resize the optic disc region.

    Crop radius = config.DISC_CROP_RADIUS_FRACTION × image width.
    Now 0.22 (≈98px at 224px input) instead of the original 0.15 (≈66px),
    which required a 3.4× upsample and destroyed fine disc/cup/rim detail.

    Args:
        img_bgr:     BGR input image (H, W, 3)
        target_size: (height, width) to resize result to

    Returns:
        Resized OD crop (H, W, 3) in BGR
    """
    h, w = img_bgr.shape[:2]
    cx, cy = detect_disc_centre(img_bgr)

    radius = int(w * config.DISC_CROP_RADIUS_FRACTION)

    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr   # fallback

    return cv2.resize(crop, (target_size[1], target_size[0]),
                      interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# PERIPAPILLARY CROP
# ─────────────────────────────────────────────────────────────────────────────
def get_peripapillary_crop(img_bgr: np.ndarray,
                           target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """
    Extract the peripapillary annular region with a SOFT alpha window.

    Method:
      1. Detect OD centre.
      2. Build a smooth radial alpha window:
           - ~0  at the disc centre (suppresses OD detail)
           - Ramps up across a 'feather' band to 1 at the inner PP boundary
           - Stays 1 through the PP ring (sharp — full detail here)
           - Ramps back down to 0 past the outer PP boundary
      3. Blend the original image with a per-image Gaussian blur using the
         alpha: alpha=1 → original; alpha=0 → blurred background.
      4. Crop bounding box of the PP annulus, resize to target_size.

    Why this replaces the original flat-fill approach:
      The original code filled both the disc hole AND everything outside the
      ring with one single flat background colour. This created a near-identical
      hard-edged synthetic disc shape in EVERY training image, regardless of
      class. The model could trivially key off this artificial artifact instead
      of learning genuine peripapillary RNFL thinning, producing high-confidence
      false-positive glaucoma predictions on new Normal images. This version:
        • Uses a feathered (gradual) transition — no crisp memorisable edge.
        • Blurs using each image's OWN content — no dataset-wide identical shape.
        • Preserves genuine PP tissue sharply in the ring itself.

    Args:
        img_bgr:     BGR input image (should be 224×224 already)
        target_size: (H, W) output size

    Returns:
        PP region image (H, W, 3) in BGR
    """
    h, w = img_bgr.shape[:2]
    cx, cy = detect_disc_centre(img_bgr)

    # Scale radii to current image size (config values are for 224px)
    scale   = ((w / config.IMG_WIDTH) + (h / config.IMG_HEIGHT)) / 2.0
    inner_r = int(config.PP_INNER_RADIUS_PX * scale)
    outer_r = int(config.PP_OUTER_RADIUS_PX * scale)
    feather = max(4, int(0.25 * inner_r))   # soft transition width in px

    # Distance map from detected centre
    Y, X = np.ogrid[:h, :w]
    dist  = np.sqrt((X - cx).astype(np.float32) ** 2 +
                    (Y - cy).astype(np.float32) ** 2)

    # Inner alpha ramp: 0 at centre, reaches 1 at (inner_r + feather)
    inner_alpha = np.clip((dist - (inner_r - feather)) / (2.0 * feather), 0.0, 1.0)
    # Outer alpha ramp: 1 inside ring, 0 beyond (outer_r + feather)
    outer_alpha = np.clip((outer_r + feather - dist) / (2.0 * feather), 0.0, 1.0)
    # Combined: soft ring window, ~1 in PP zone, ~0 elsewhere
    alpha     = (inner_alpha * outer_alpha).astype(np.float32)
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    # Per-image blurred background — suppresses disc/background detail without
    # introducing any dataset-wide identical shape to memorise
    blur_sigma = max(8, int(outer_r * 0.6))
    blurred_bg = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=blur_sigma)

    result = (img_bgr.astype(np.float32)  * alpha_3ch +
              blurred_bg.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)

    # Crop bounding box around the PP annulus
    x1 = max(0, cx - outer_r)
    y1 = max(0, cy - outer_r)
    x2 = min(w, cx + outer_r)
    y2 = min(h, cy + outer_r)

    crop = result[y1:y2, x1:x2]
    if crop.size == 0:
        crop = result   # fallback

    return cv2.resize(crop, (target_size[1], target_size[0]),
                      interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# CROP QUALITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_crop_quality(crop_bgr: np.ndarray,
                       min_mean: float = 20.0,
                       min_std:  float = 5.0,
                       min_laplacian: float = 5.0) -> dict:
    """
    Assess whether a crop is usable or should be flagged/rejected.

    Checks:
      - Mean brightness (too dark → disc not captured)
      - Std brightness  (too low  → blank/uniform → disc not in frame)
      - Laplacian variance (too low → excessively blurry)

    Args:
        crop_bgr:      Cropped BGR image
        min_mean:      Minimum acceptable mean pixel value
        min_std:       Minimum acceptable std pixel value
        min_laplacian: Minimum Laplacian variance (sharpness)

    Returns:
        dict with keys: 'ok' (bool), 'mean', 'std', 'sharpness', 'reason'
    """
    gray      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_val  = float(np.mean(gray))
    std_val   = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    reason = []
    if mean_val < min_mean:
        reason.append(f"too_dark (mean={mean_val:.1f}<{min_mean})")
    if std_val < min_std:
        reason.append(f"low_contrast (std={std_val:.1f}<{min_std})")
    if sharpness < min_laplacian:
        reason.append(f"too_blurry (lap_var={sharpness:.1f}<{min_laplacian})")

    return {
        "ok":        len(reason) == 0,
        "mean":      mean_val,
        "std":       std_val,
        "sharpness": sharpness,
        "reason":    ", ".join(reason) if reason else "OK",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH CROP ALL IMAGES
# ─────────────────────────────────────────────────────────────────────────────
def precrop_all(source_dir: str = None,
                od_dir: str = None,
                pp_dir: str = None,
                quality_check: bool = True) -> dict:
    """
    Batch-crop all images into OD and PP regions.

    Reads from:   source_dir/{glaucoma,normal}/
    Writes to:    od_dir/{glaucoma,normal}/
                  pp_dir/{glaucoma,normal}/

    Args:
        source_dir:    Root data directory (default: config.DATA_DIR)
        od_dir:        Output root for OD crops (default: config.DATA_OD_DIR)
        pp_dir:        Output root for PP crops (default: config.DATA_PP_DIR)
        quality_check: Whether to log crops that fail quality thresholds

    Returns:
        dict: {'processed': int, 'skipped': int, 'low_quality': list}
    """
    source_dir = source_dir or config.DATA_DIR
    od_dir     = od_dir     or config.DATA_OD_DIR
    pp_dir     = pp_dir     or config.DATA_PP_DIR

    classes        = ["glaucoma", "normal"]
    total_processed = 0
    total_skipped   = 0
    low_quality     = []

    for cls in classes:
        src_cls = os.path.join(source_dir, cls)
        od_cls  = os.path.join(od_dir, cls)
        pp_cls  = os.path.join(pp_dir, cls)

        os.makedirs(od_cls, exist_ok=True)
        os.makedirs(pp_cls, exist_ok=True)

        if not os.path.exists(src_cls):
            print(f"[CROP] Skipping missing directory: {src_cls}")
            continue

        image_files = [
            f for f in sorted(os.listdir(src_cls))
            if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]
        print(f"[CROP] {cls}: {len(image_files)} images to process ...")

        for fname in tqdm(image_files, desc=f"  {cls}"):
            src_path = os.path.join(src_cls, fname)
            od_dst   = os.path.join(od_cls,  fname)
            pp_dst   = os.path.join(pp_cls,  fname)

            # Skip already-cropped files (re-run safe)
            if os.path.exists(od_dst) and os.path.exists(pp_dst):
                total_skipped += 1
                continue

            img = cv2.imread(src_path)
            if img is None:
                print(f"[CROP] WARNING: Cannot read {src_path}, skipping.")
                total_skipped += 1
                continue

            # Resize to standard size first (ensures pixel radii match config)
            img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)

            try:
                od_crop = get_optic_disc_crop(img)
                pp_crop = get_peripapillary_crop(img)

                if quality_check:
                    qc = check_crop_quality(od_crop)
                    if not qc["ok"]:
                        low_quality.append(
                            {"file": fname, "class": cls,
                             "type": "od", "reason": qc["reason"]}
                        )

                cv2.imwrite(od_dst, od_crop)
                cv2.imwrite(pp_dst, pp_crop)
                total_processed += 1

            except Exception as e:
                print(f"[CROP] ERROR processing {fname}: {e}")
                total_skipped += 1
                continue

    print(f"\n[CROP] Done.")
    print(f"[CROP]   Processed : {total_processed}")
    print(f"[CROP]   Skipped   : {total_skipped}")
    print(f"[CROP]   Low quality OD crops flagged: {len(low_quality)}")
    if low_quality:
        print("[CROP]   Low quality examples (first 5):")
        for item in low_quality[:5]:
            print(f"         {item['class']}/{item['file']} → {item['reason']}")
    print(f"[CROP]   OD crops → {od_dir}")
    print(f"[CROP]   PP crops → {pp_dir}")

    return {
        "processed":   total_processed,
        "skipped":     total_skipped,
        "low_quality": low_quality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[CROP] Starting batch crop (with quality check) ...")
    results = precrop_all(quality_check=True)
    print(f"[CROP] Complete. {results['processed']} crops written.")