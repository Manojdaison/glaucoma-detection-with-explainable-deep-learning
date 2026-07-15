"""
config.py — Single source of truth for ALL project settings.
Glaucoma Detection Framework | Manoj | VIT Chennai

CHANGES FROM ORIGINAL:
  - DISC_CROP_RADIUS_FRACTION: 0.15 → 0.22
      Old 0.15 produced a 66px native crop that required 3.4x upsampling to
      reach 224px, destroying fine disc/cup/rim detail. 0.22 gives a ~98px
      native crop (2.3x upsample) — still imperfect but meaningfully better.
  - AUG_VERTICAL_FLIP: True → False
      Vertical flip is NOT medically valid for fundus images. The optic disc,
      blood vessel branching pattern, and fovea have a defined superior/inferior
      orientation. Flipping vertically produces anatomically impossible images
      that teach the model the wrong spatial priors.
  - Added LABEL_SMOOTHING = 0.1
      Prevents overconfident soft-max outputs that hurt calibration.
  - Added ENSEMBLE_WEIGHTS — configurable per-model weights.
  - Added THRESHOLDS — per-model slots populated by threshold_optimizer.py.
  - Added LOGS_DIR and MISCLASSIFIED_DIR paths.
  - Removed intermediate hardcoded Windows model_path strings (were overwritten
      anyway; the dynamic os.path.join block is the correct approach).
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (all relative to the project root — no hardcoded Windows paths)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

DATA_DIR          = os.path.join(BASE_DIR, "data")
DATA_GLAUCOMA     = os.path.join(DATA_DIR, "glaucoma")
DATA_NORMAL       = os.path.join(DATA_DIR, "normal")

DATA_OD_DIR       = os.path.join(BASE_DIR, "data_od")
DATA_OD_GLAUCOMA  = os.path.join(DATA_OD_DIR, "glaucoma")
DATA_OD_NORMAL    = os.path.join(DATA_OD_DIR, "normal")

DATA_PP_DIR       = os.path.join(BASE_DIR, "data_pp")
DATA_PP_GLAUCOMA  = os.path.join(DATA_PP_DIR, "glaucoma")
DATA_PP_NORMAL    = os.path.join(DATA_PP_DIR, "normal")

MODELS_DIR        = os.path.join(BASE_DIR, "models")

OUTPUTS_DIR       = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR         = os.path.join(OUTPUTS_DIR, "plots")
RCS_DIR           = os.path.join(OUTPUTS_DIR, "rcs")
REPORTS_DIR       = os.path.join(OUTPUTS_DIR, "reports")
GRADCAM_DIR       = os.path.join(OUTPUTS_DIR, "gradcam")

# New directories
LOGS_DIR             = os.path.join(OUTPUTS_DIR, "logs")
MISCLASSIFIED_DIR    = os.path.join(OUTPUTS_DIR, "misclassified")
MISCLASSIFIED_FP_DIR = os.path.join(MISCLASSIFIED_DIR, "false_positive")
MISCLASSIFIED_FN_DIR = os.path.join(MISCLASSIFIED_DIR, "false_negative")
THRESHOLDS_FILE      = os.path.join(OUTPUTS_DIR, "reports", "thresholds.json")

# ACRIMA CSV for CDR correlation (optional)
ACRIMA_CSV        = os.path.join(DATA_DIR, "ACRIMA.csv")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
IMG_CHANNELS = 3
IMG_SIZE     = (IMG_HEIGHT, IMG_WIDTH)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 16
EPOCHS_PHASE1     = 20       # Frozen backbone (feature extraction)
EPOCHS_PHASE2     = 20       # Fine-tuning last N layers
UNFREEZE_LAYERS   = 30       # Layers to unfreeze in Phase 2

LR_PHASE1         = 1e-3
LR_PHASE2         = 1e-5

# Label smoothing: reduces overconfidence, improves calibration.
# 0.1 means the target for a "glaucoma" label becomes 0.9, not 1.0.
# This is especially useful here because our OD/PP crops are noisy.
LABEL_SMOOTHING   = 0.1

# Callbacks
EARLY_STOPPING_PATIENCE   = 12
REDUCE_LR_PATIENCE        = 6
REDUCE_LR_FACTOR          = 0.5
REDUCE_LR_MIN             = 1e-7

MONITOR_METRIC    = "val_auc"
MONITOR_MODE      = "max"

# Validation split fraction
VAL_SPLIT         = 0.20     # 80/20 stratified

# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
AUG_ROTATION_RANGE    = 20          # degrees — fundus cameras can be slightly rotated
AUG_BRIGHTNESS_RANGE  = [0.85, 1.15]
AUG_ZOOM_RANGE        = 0.10
AUG_HORIZONTAL_FLIP   = True        # LEFT eye vs RIGHT eye → valid flip axis
AUG_VERTICAL_FLIP     = False       # REMOVED: NOT medically valid for fundus images.
                                     # The optic disc and fovea have a defined
                                     # superior/inferior orientation; vertical flip
                                     # produces anatomically impossible training samples.
AUG_FILL_MODE         = "nearest"
AUG_WIDTH_SHIFT       = 0.05
AUG_HEIGHT_SHIFT      = 0.05
AUG_SHEAR_RANGE       = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# REGION DEFINITIONS  (in pixels, after resizing to 224×224)
# ─────────────────────────────────────────────────────────────────────────────
# Optic Disc (OD): clinically the bright central region
OD_RADIUS_PX      = 33       # radius ≤ 33px from disc centre

# Peripapillary (PP): RNFL assessment zone
PP_INNER_RADIUS_PX = 33      # inner boundary (exclusive)
PP_OUTER_RADIUS_PX = 67      # outer boundary (inclusive)

# Background: r > 67px
BG_INNER_RADIUS_PX = 67

# Optic disc crop radius as fraction of image width.
# CHANGED from 0.15 to 0.22:
#   0.15 → 66px native crop → 3.4x upsample → blurry, no detail
#   0.22 → 98px native crop → 2.3x upsample → meaningfully better
# Ablation study (planned ablation_study.py) should sweep 0.10/0.15/0.20/0.25
# to confirm this is optimal for ACRIMA; 0.22 is a reasoned starting point.
DISC_CROP_RADIUS_FRACTION = 0.22

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_LAYER_NAME     = "Conv_1"     # Last conv layer in MobileNetV2
GRADCAM_ALPHA          = 0.4          # Heatmap overlay transparency
GRADCAM_COLORMAP       = "jet"
GRADCAM_FEATURE_SIZE   = (7, 7)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY  (paths built dynamically — no hardcoded Windows strings)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "fullimage": {
        "name":        "fullimage_model",
        "description": "Full Fundus Image Model",
        "last_conv":   "block_16_project",
        "data_dir":    DATA_DIR,
    },
    "od": {
        "name":        "od_model",
        "description": "Optic Disc Model",
        "last_conv":   "block_16_project",
        "data_dir":    DATA_OD_DIR,
    },
    "pp": {
        "name":        "pp_model",
        "description": "Peripapillary Model",
        "last_conv":   "block_16_project",
        "data_dir":    DATA_PP_DIR,
    },
}

# Populate model paths dynamically
for _key in MODEL_REGISTRY:
    MODEL_REGISTRY[_key]["model_path"] = os.path.join(
        MODELS_DIR, f"{MODEL_REGISTRY[_key]['name']}.h5"
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE WEIGHTS
# The full-image model currently outperforms the cropped models.
# These weights are used by ensemble.py for weighted average prediction.
# Adjust after retraining OD/PP with fixed crops — OD and PP weights should
# increase as their false-positive rate drops.
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "fullimage": 0.6,
    "od":        0.2,
    "pp":        0.2,
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION THRESHOLDS (per model)
# Default = 0.5 until threshold_optimizer.py runs and writes thresholds.json.
# threshold_optimizer.py will overwrite these from the validation ROC curve
# using the Youden's J criterion (maximises sensitivity + specificity - 1).
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "fullimage": 0.5,
    "od":        0.5,
    "pp":        0.5,
    "ensemble":  0.5,
}

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD_METHOD  = "youden"   # Options: "youden", "f1", "sensitivity95"
FIGURE_DPI        = 300
FIGURE_FORMAT     = "png"
KFOLD_N_SPLITS    = 5
MC_DROPOUT_PASSES = 50
TTA_VERSIONS      = 5

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — create all required output directories
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required output directories if they don't exist."""
    dirs = [
        MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, RCS_DIR,
        REPORTS_DIR, GRADCAM_DIR, LOGS_DIR,
        MISCLASSIFIED_FP_DIR, MISCLASSIFIED_FN_DIR,
        DATA_GLAUCOMA, DATA_NORMAL,
        DATA_OD_GLAUCOMA, DATA_OD_NORMAL,
        DATA_PP_GLAUCOMA, DATA_PP_NORMAL,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("[CONFIG] All directories ensured.")
    print(f"[CONFIG] SEED={SEED}  BATCH={BATCH_SIZE}  "
          f"EPOCHS={EPOCHS_PHASE1}+{EPOCHS_PHASE2}")
    print(f"[CONFIG] OD crop radius fraction = {DISC_CROP_RADIUS_FRACTION} "
          f"(native px at 224 = {int(224*DISC_CROP_RADIUS_FRACTION)*2}px square)")
    print(f"[CONFIG] LABEL_SMOOTHING = {LABEL_SMOOTHING}")
    print(f"[CONFIG] AUG_VERTICAL_FLIP = {AUG_VERTICAL_FLIP}  (must be False)")
    print("[CONFIG] Model registry:")
    for k, v in MODEL_REGISTRY.items():
        print(f"  [{k}] → {v['model_path']}")