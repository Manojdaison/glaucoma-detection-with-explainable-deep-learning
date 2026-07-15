"""
robustness.py
=============
Test model stability under noise, blur, illumination changes.
"""
import os, cv2
import numpy as np
from config import *

def add_gaussian_noise(image, mean=0, std=0.1):
    """Add Gaussian noise to image."""
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 1).astype(np.float32)

def add_gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0).astype(np.float32)

def change_brightness(image, factor=0.7):
    """Adjust brightness."""
    return np.clip(image * factor, 0, 1).astype(np.float32)

---

"""
ablation.py
===========
Ablation studies: impact of RCS, region cropping, fine-tuning.
"""
import os
import pandas as pd
from config import *

def create_ablation_report():
    """Document ablation study design."""
    ablations = [
        {
            "study": "Remove fine-tuning",
            "description": "Train only Phase 1 (frozen backbone)",
            "expected_impact": "Lower AUC (5-8% drop)",
        },
        {
            "study": "Remove augmentation",
            "description": "No data augmentation during training",
            "expected_impact": "Higher overfitting, lower val AUC",
        },
        {
            "study": "Remove dropout",
            "description": "Set dropout to 0",
            "expected_impact": "Severe overfitting",
        },
        {
            "study": "Full image only",
            "description": "Use only full image model",
            "expected_impact": "Baseline performance",
        },
    ]
    df = pd.DataFrame(ablations)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORTS_DIR, "ablation_study_design.csv"), index=False)
    print(f"[ABLATION] Design saved → {os.path.join(REPORTS_DIR, 'ablation_study_design.csv')}")