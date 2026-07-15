"""
dataset.py — Data loading, splitting, and augmentation utilities.
Glaucoma Detection Framework | Manoj | VIT Chennai

FIXES FROM ORIGINAL:
  1. build_generators: There was a complete second implementation of the
     function sitting AFTER the return statement — unreachable dead code.
     The live (reachable) first implementation used HARDCODED augmentation
     values (rotation=15, zoom=0.1) instead of reading from config.py, so
     changes to AUG_* settings in config.py had zero effect on training.
     Fixed: removed dead code, live implementation now reads all augmentation
     settings from config.AUG_*.

  2. AUG_VERTICAL_FLIP removed from augmentation.
     Vertical flip is NOT medically valid for fundus images — the retinal
     anatomy has a defined superior/inferior orientation. Flipping vertically
     produces anatomically impossible training samples. (AUG_VERTICAL_FLIP
     is now False in config.py.)

  3. Duplicate import of ImageDataGenerator (was at module level AND inside
     build_generators). Kept only the module-level import.
"""

import os
import zipfile
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

# ─────────────────────────────────────────────────────────────────────────────
# SEED EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION FROM ZIP
# ─────────────────────────────────────────────────────────────────────────────
def prepare_from_zip(zip_path: str, output_dir: str = None) -> dict:
    """
    Extract ACRIMA.zip and sort images into glaucoma/ and normal/ sub-folders.

    ACRIMA naming convention:
      - Filenames containing "_g_" → Glaucoma
      - All other image files      → Normal

    Args:
        zip_path:   Path to ACRIMA.zip
        output_dir: Destination folder (default: config.DATA_DIR)

    Returns:
        dict with counts: {glaucoma, normal, total}
    """
    if output_dir is None:
        output_dir = config.DATA_DIR

    glaucoma_dir = os.path.join(output_dir, "glaucoma")
    normal_dir   = os.path.join(output_dir, "normal")
    os.makedirs(glaucoma_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[DATASET] ZIP not found: {zip_path}")

    print(f"[DATASET] Extracting {zip_path} ...")
    extract_tmp = os.path.join(config.BASE_DIR, "_acrima_extract_tmp")
    os.makedirs(extract_tmp, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_tmp)

    all_images = []
    for root, _, files in os.walk(extract_tmp):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                all_images.append(os.path.join(root, f))

    all_images.sort()
    counts = {"glaucoma": 0, "normal": 0}

    for src_path in all_images:
        fname = os.path.basename(src_path)
        if "_g_" in fname.lower():
            dst = os.path.join(glaucoma_dir, fname)
            counts["glaucoma"] += 1
        else:
            dst = os.path.join(normal_dir, fname)
            counts["normal"] += 1
        shutil.copy2(src_path, dst)

    shutil.rmtree(extract_tmp, ignore_errors=True)

    counts["total"] = counts["glaucoma"] + counts["normal"]
    print(f"[DATASET] Sorted {counts['total']} images  "
          f"(Glaucoma: {counts['glaucoma']}, Normal: {counts['normal']})")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT FILEPATHS
# ─────────────────────────────────────────────────────────────────────────────
def collect_filepaths(data_dir: str) -> tuple:
    """
    Scan data_dir/glaucoma/ and data_dir/normal/ for image files.

    Returns:
        (filepaths, labels) — parallel lists
        labels: 1 = glaucoma, 0 = normal
    """
    filepaths = []
    labels    = []

    class_map = {"glaucoma": 1, "normal": 0}
    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[DATASET] WARNING: {class_dir} not found, skipping.")
            continue
        for f in sorted(os.listdir(class_dir)):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                filepaths.append(os.path.join(class_dir, f))
                labels.append(label)

    print(f"[DATASET] Collected {len(filepaths)} images from {data_dir}")
    n_glaucoma = sum(labels)
    n_normal   = len(labels) - n_glaucoma
    print(f"[DATASET]   Glaucoma: {n_glaucoma} | Normal: {n_normal}")
    return filepaths, labels


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(filepaths: list, labels: list,
                     val_size: float = config.VAL_SPLIT) -> tuple:
    """
    Stratified 80/20 train-validation split preserving class proportions.

    Returns:
        (train_paths, val_paths, train_labels, val_labels)
    """
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filepaths, labels,
        test_size=val_size,
        stratify=labels,
        random_state=config.SEED
    )
    print(f"[DATASET] Train: {len(train_paths)} | Val: {len(val_paths)}")
    g_tr = sum(train_labels)
    g_vl = sum(val_labels)
    print(f"[DATASET]   Train → Glaucoma: {g_tr}, Normal: {len(train_labels)-g_tr}")
    print(f"[DATASET]   Val   → Glaucoma: {g_vl}, Normal: {len(val_labels)-g_vl}")
    return train_paths, val_paths, train_labels, val_labels


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def get_class_weights(labels: list) -> dict:
    """
    Compute balanced inverse-frequency class weights.

    Returns:
        {0: weight_normal, 1: weight_glaucoma}
    """
    labels_arr = np.array(labels)
    classes    = np.unique(labels_arr)
    weights    = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[DATASET] Class weights: {weight_dict}")
    return weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def build_generators(X_tr, X_vl, y_tr, y_vl,
                     preprocess_fn=None,
                     batch_size: int = config.BATCH_SIZE,
                     target_size: tuple = config.IMG_SIZE):
    """
    Build Keras ImageDataGenerators for training and validation.

    Augmentation settings are read from config.AUG_* — NOT hardcoded here.
    Changes to config.py immediately take effect without touching this file.

    IMPORTANT: AUG_VERTICAL_FLIP is intentionally False.
      Vertical flip is not medically valid for fundus images.

    Args:
        X_tr, X_vl:     File path lists for train/validation
        y_tr, y_vl:     Label lists (0/1) for train/validation
        preprocess_fn:  Preprocessing function (e.g. mobilenet_v2.preprocess_input)
        batch_size:     Mini-batch size
        target_size:    (H, W) to resize images to

    Returns:
        (train_gen, val_gen)
    """
    # Training generator — WITH augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=config.AUG_ROTATION_RANGE,
        width_shift_range=config.AUG_WIDTH_SHIFT,
        height_shift_range=config.AUG_HEIGHT_SHIFT,
        shear_range=config.AUG_SHEAR_RANGE,
        zoom_range=config.AUG_ZOOM_RANGE,
        horizontal_flip=config.AUG_HORIZONTAL_FLIP,
        vertical_flip=config.AUG_VERTICAL_FLIP,   # Always False (medically invalid)
        brightness_range=config.AUG_BRIGHTNESS_RANGE,
        fill_mode=config.AUG_FILL_MODE,
    )

    # Validation generator — NO augmentation, just preprocessing
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({
            "filename": X_tr,
            "class":    [str(lbl) for lbl in y_tr]
        }),
        x_col="filename",
        y_col="class",
        target_size=target_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=config.SEED,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({
            "filename": X_vl,
            "class":    [str(lbl) for lbl in y_vl]
        }),
        x_col="filename",
        y_col="class",
        target_size=target_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    return train_gen, val_gen


# ─────────────────────────────────────────────────────────────────────────────
# K-FOLD SPLITS (for cross-validation)
# ─────────────────────────────────────────────────────────────────────────────
def kfold_splits(filepaths: list, labels: list,
                 n_splits: int = config.KFOLD_N_SPLITS):
    """
    Generate stratified K-fold cross-validation splits.

    Yields:
        (fold_idx, (train_paths, val_paths, train_labels, val_labels))
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.SEED)
    fps = np.array(filepaths)
    lbs = np.array(labels)

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(fps, lbs)):
        yield (
            fold_idx,
            (
                list(fps[tr_idx]),
                list(fps[vl_idx]),
                list(lbs[tr_idx]),
                list(lbs[vl_idx]),
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fps, lbls = collect_filepaths(config.DATA_DIR)
    if fps:
        tr_p, vl_p, tr_l, vl_l = stratified_split(fps, lbls)
        cw = get_class_weights(tr_l)
        print("[DATASET] Sanity check passed.")
    else:
        print("[DATASET] No images found — run data preparation first.")