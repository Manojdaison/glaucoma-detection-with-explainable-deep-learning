"""
rcs.py — Region Contribution Score (RCS): novel quantitative explainability metric.
Glaucoma Detection Framework | Manoj | VIT Chennai

RCS Formula:
  RCS(R) = Σ_{(i,j) ∈ R} H(i,j) / Σ_{all (i,j)} H(i,j)

where H is the Grad-CAM heatmap normalized to [0,1].

Properties:
  - RCS_OD + RCS_PP + RCS_BG = 1.0  (completeness)
  - RCS_OD > 0 for glaucoma (higher optic disc attention)
  - Validated against expert Cup-to-Disc Ratio (ExpCDR)

Regions (at 224×224):
  OD  : radius ≤ 33px from disc centre
  PP  : 33px < r ≤ 67px  (RNFL zone)
  BG  : r > 67px
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config

# ─────────────────────────────────────────────────────────────────────────────
# BUILD REGION MASKS
# ─────────────────────────────────────────────────────────────────────────────
def build_region_masks(image_size: tuple = config.IMG_SIZE,
                       centre: tuple = None) -> dict:
    """
    Build circular binary masks for OD, PP, and BG regions.

    Masks are at image_size resolution. Centre defaults to image centre.

    Args:
        image_size: (H, W) — default (224, 224)
        centre:     (cx, cy) pixel coordinates — default image centre

    Returns:
        dict with keys "od", "peripapillary", "background" — each (H, W) bool array
    """
    h, w = image_size
    if centre is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = centre

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    od_mask = dist <= config.OD_RADIUS_PX
    pp_mask = (dist > config.PP_INNER_RADIUS_PX) & (dist <= config.PP_OUTER_RADIUS_PX)
    bg_mask = dist > config.BG_INNER_RADIUS_PX

    return {
        "od":            od_mask,
        "peripapillary": pp_mask,
        "background":    bg_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RCS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rcs(heatmap: np.ndarray,
                centre: tuple = None,
                image_size: tuple = config.IMG_SIZE) -> dict:
    """
    Compute Region Contribution Score for all three anatomical regions.

    Args:
        heatmap:    Grad-CAM heatmap (H, W), normalized to [0, 1]
        centre:     Disc centre (cx, cy); defaults to image centre
        image_size: (H, W) of the heatmap

    Returns:
        dict {
            "od":            RCS_OD   (float in [0, 1]),
            "peripapillary": RCS_PP   (float in [0, 1]),
            "background":    RCS_BG   (float in [0, 1]),
            "total_sum":     Σ heatmap (for verification)
        }
        Note: od + peripapillary + background ≈ 1.0
    """
    masks = build_region_masks(image_size=image_size, centre=centre)
    total = heatmap.sum()

    if total < 1e-8:
        # Degenerate heatmap — equal contribution
        return {
            "od": 1/3, "peripapillary": 1/3, "background": 1/3,
            "total_sum": 0.0
        }

    rcs = {}
    for region_name, mask in masks.items():
        rcs[region_name] = float(heatmap[mask].sum() / total)

    rcs["total_sum"] = float(total)
    return rcs


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RCS FOR ENTIRE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def compute_rcs_for_dataset(model,
                            val_paths: list,
                            val_labels: list,
                            model_name: str = "fullimage") -> pd.DataFrame:
    """
    Compute RCS scores for all validation images.

    Args:
        model:      Trained Keras model
        val_paths:  List of image file paths
        val_labels: Corresponding labels (0=normal, 1=glaucoma)
        model_name: Used for output CSV naming

    Returns:
        DataFrame with columns:
          filename, label, diagnosis, prediction, rcs_od, rcs_pp, rcs_bg
    """
    from gradcam import get_gradcam, preprocess_image
    from tqdm import tqdm

    records = []

    for img_path, label in tqdm(zip(val_paths, val_labels),
                                total=len(val_paths),
                                desc=f"  RCS [{model_name}]"):
        try:
            img_rgb, img_array = preprocess_image(img_path)

            # Model prediction
            pred = float(model.predict(img_array, verbose=0)[0][0])

            # Grad-CAM heatmap
            _, heatmap = get_gradcam(model, img_array)

            # RCS
            rcs = compute_rcs(heatmap)

            records.append({
                "filename":      os.path.basename(img_path),
                "label":         int(label),
                "diagnosis":     "Glaucoma" if label == 1 else "Normal",
                "prediction":    round(pred, 4),
                "rcs_od":        round(rcs["od"], 4),
                "rcs_pp":        round(rcs["peripapillary"], 4),
                "rcs_bg":        round(rcs["background"], 4),
            })

        except Exception as e:
            print(f"[RCS] Error processing {img_path}: {e}")
            continue

    df = pd.DataFrame(records)

    # Save CSV
    os.makedirs(config.RCS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RCS_DIR, f"rcs_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[RCS] Saved {len(df)} RCS scores → {csv_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Run Mann-Whitney U test: H0: RCS_OD(glaucoma) > RCS_OD(normal).

    Args:
        df: DataFrame with columns "rcs_od" and "diagnosis"

    Returns:
        dict with {statistic, p_value, significant}
    """
    glaucoma_rcs = df[df["diagnosis"] == "Glaucoma"]["rcs_od"].values
    normal_rcs   = df[df["diagnosis"] == "Normal"]["rcs_od"].values

    if len(glaucoma_rcs) == 0 or len(normal_rcs) == 0:
        return {"statistic": None, "p_value": None, "significant": False}

    stat, p_val = mannwhitneyu(glaucoma_rcs, normal_rcs, alternative="greater")

    result = {
        "statistic":   float(stat),
        "p_value":     float(p_val),
        "significant": bool(p_val < 0.05),
    }
    print(f"[RCS] Mann-Whitney U: stat={stat:.2f}, p={p_val:.4f} "
          f"({'significant' if result['significant'] else 'not significant'})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLOT RCS BOXPLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_rcs_comparison(df: pd.DataFrame, model_name: str = "fullimage") -> str:
    """
    Box plots of RCS values per region, split by diagnosis.

    Args:
        df:         RCS DataFrame (output of compute_rcs_for_dataset)
        model_name: For title and filename

    Returns:
        Path to saved figure
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Melt to long form
    df_long = df.melt(
        id_vars=["diagnosis"],
        value_vars=["rcs_od", "rcs_pp", "rcs_bg"],
        var_name="Region",
        value_name="RCS"
    )
    region_labels = {"rcs_od": "Optic Disc (OD)",
                     "rcs_pp": "Peripapillary (PP)",
                     "rcs_bg": "Background (BG)"}
    df_long["Region"] = df_long["Region"].map(region_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"Glaucoma": "#E53935", "Normal": "#1E88E5"}

    sns.boxplot(
        data=df_long, x="Region", y="RCS", hue="diagnosis",
        palette=palette, ax=ax, width=0.5
    )

    ax.set_title(f"Region Contribution Score (RCS) — {model_name}",
                 fontsize=13)
    ax.set_xlabel("Anatomical Region", fontsize=11)
    ax.set_ylabel("RCS (fraction of total heatmap)", fontsize=11)
    ax.legend(title="Diagnosis")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "rcs_boxplots.png")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[RCS] Box plots saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STACKED BAR CHART — MEAN RCS PER REGION
# ─────────────────────────────────────────────────────────────────────────────
def plot_rcs_stacked_bar(df: pd.DataFrame, model_name: str = "fullimage") -> str:
    """
    Stacked bar chart: mean RCS_OD, RCS_PP, RCS_BG per diagnosis group.

    Returns:
        Path to saved figure
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    summary = df.groupby("diagnosis")[["rcs_od", "rcs_pp", "rcs_bg"]].mean()
    summary.columns = ["Optic Disc", "Peripapillary", "Background"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E53935", "#FB8C00", "#43A047"]

    summary.plot(kind="bar", stacked=True, ax=ax,
                 color=colors, edgecolor="white", linewidth=0.5)

    ax.set_title(f"Mean RCS Distribution by Diagnosis — {model_name}",
                 fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Mean RCS")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, f"rcs_stacked_{model_name}.png")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[RCS] Stacked bar saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
def generate_rcs_summary(df: pd.DataFrame, model_name: str = "fullimage") -> str:
    """
    Generate and save a text summary of RCS statistics.

    Returns:
        Path to saved report
    """
    os.makedirs(config.RCS_DIR, exist_ok=True)

    lines = []
    lines.append("═" * 60)
    lines.append(f"  RCS SUMMARY REPORT — {model_name}")
    lines.append("═" * 60)

    for diag in ["Glaucoma", "Normal"]:
        grp = df[df["diagnosis"] == diag]
        lines.append(f"\n{diag} (n={len(grp)}):")
        for col, label in [("rcs_od", "OD "), ("rcs_pp", "PP "),
                            ("rcs_bg", "BG ")]:
            lines.append(f"  RCS_{label}: mean={grp[col].mean():.3f}  "
                         f"std={grp[col].std():.3f}  "
                         f"median={grp[col].median():.3f}")

    mw = statistical_analysis(df)
    lines.append(f"\nMann-Whitney U (RCS_OD: Glaucoma > Normal):")
    lines.append(f"  U={mw['statistic']:.1f}  p={mw['p_value']:.4f}  "
                 f"{'*** SIGNIFICANT ***' if mw['significant'] else 'not significant'}")
    lines.append("\n" + "═" * 60)

    report = "\n".join(lines)
    save_path = os.path.join(config.RCS_DIR, "rcs_summary.txt")
    with open(save_path, "w") as f:
        f.write(report)
    print(f"[RCS] Summary report → {save_path}")
    print(report)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
def compute_rcs_single(heatmap, image_size=config.IMG_SIZE, centre=None):
    """
    Wrapper for single-image RCS computation.
    Args:
        heatmap: Grad-CAM heatmap (H, W), normalized [0,1]
        image_size: (H, W)
        centre: optional disc centre (cx, cy), defaults to image centre
    Returns:
        dict with od, peripapillary, background scores
    """
    return compute_rcs(heatmap, centre=centre, image_size=image_size)

if __name__ == "__main__":
    # Demonstrate mask completeness property
    masks = build_region_masks()
    total_pixels = config.IMG_HEIGHT * config.IMG_WIDTH
    od_px  = masks["od"].sum()
    pp_px  = masks["peripapillary"].sum()
    bg_px  = masks["background"].sum()
    print(f"[RCS] Mask pixel counts at 224×224:")
    print(f"  OD  ({config.OD_RADIUS_PX}px radius): {od_px} px")
    print(f"  PP  ({config.PP_INNER_RADIUS_PX}–{config.PP_OUTER_RADIUS_PX}px): {pp_px} px")
    print(f"  BG  (>{config.BG_INNER_RADIUS_PX}px): {bg_px} px")
    print(f"  Coverage: {(od_px+pp_px+bg_px)/total_pixels*100:.1f}% of image")

