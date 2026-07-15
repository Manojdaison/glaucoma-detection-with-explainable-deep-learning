"""
evaluate.py
===========
Comprehensive evaluation: metrics, ROC, confusion matrix, threshold tuning.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_fscore_support, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from dataset import collect_filepaths
from model_builder import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_val_predictions(model, data_dir):
    """Get validation predictions on exact train-test split."""
    paths, labels = collect_filepaths(data_dir)
    _, X_val, _, y_val = train_test_split(paths, labels, test_size=VAL_SPLIT,
                                          stratify=labels, random_state=SEED)
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    df = pd.DataFrame({"filename": X_val, "label": [str(l) for l in y_val]})
    val_gen = gen.flow_from_dataframe(df, x_col="filename", y_col="label",
                                      target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                      class_mode="binary", shuffle=False, seed=SEED)
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=1).ravel()
    return np.array(y_val), y_prob

def find_optimal_threshold(y_true, y_prob):
    """Youden's J statistic: J = Sensitivity + Specificity - 1"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    youden = tpr + specificity - 1
    best_idx = int(np.argmax(youden))
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(specificity[best_idx]),
        "fpr": fpr, "tpr": tpr,
    }

def evaluate_model(model_key):
    """Full evaluation of single model."""
    cfg = MODEL_REGISTRY[model_key]
    model_name = f"{model_key}_model"
    data_dir = cfg["data_dir"]
    
    print(f"\n[EVAL] {model_name}")
    model = load_model(model_name)
    y_true, y_prob = get_val_predictions(model, data_dir)
    
    thresh_info = find_optimal_threshold(y_true, y_prob)
    opt_thresh = thresh_info["optimal_threshold"]
    y_pred = (y_prob >= opt_thresh).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    
    metrics = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auc": round(float(auc), 4),
        "optimal_threshold": round(opt_thresh, 4),
    }
    
    print(f"  Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    
    # Save metrics
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresh_info["fpr"], thresh_info["tpr"], linewidth=2.5, color="#1F77B4",
            label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{model_name} — ROC Curve", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    return metrics, thresh_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--model", default="fullimage")
    args = parser.parse_args()
    
    if args.model == "all":
        models = list(MODEL_REGISTRY.keys())
    else:
        models = [args.model]
    
    all_metrics = []
    for key in models:
        if os.path.isfile(MODEL_REGISTRY[key]["path"]):
            m, _ = evaluate_model(key)
            all_metrics.append(m)
    
    # Save comparison table
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(REPORTS_DIR, "model_comparison_table.csv"), index=False)
        print("\n[RESULTS]")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()