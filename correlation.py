"""
correlation.py — Clinical validation of RCS against expert Cup-to-Disc Ratio (ExpCDR).
Glaucoma Detection Framework | Manoj | VIT Chennai

Hypotheses validated:
  H1: Pearson r(RCS_OD, ExpCDR) > 0       — positive linear correlation
  H2: Spearman ρ(RCS_OD, ExpCDR) > 0      — positive rank correlation
  H3: Point-biserial r(RCS_OD, Glaucoma)   — discrimination ability
  H4: Mann-Whitney U: RCS_OD(G) > RCS_OD(N)— diagnostic separation

ACRIMA.csv columns: Filename, ExpCDR, Eye, Set, Glaucoma
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND MERGE
# ─────────────────────────────────────────────────────────────────────────────
def load_and_merge(acrima_csv: str = None,
                   rcs_csv: str = None) -> pd.DataFrame:
    """
    Load ACRIMA.csv and RCS scores, merge on filename.

    Args:
        acrima_csv: Path to ACRIMA.csv (default: config.ACRIMA_CSV)
        rcs_csv:    Path to RCS scores CSV (default: rcs/rcs_fullimage.csv)

    Returns:
        Merged DataFrame with [Filename, ExpCDR, Glaucoma, rcs_od, rcs_pp, rcs_bg]
        Returns None if files are missing.
    """
    if acrima_csv is None:
        acrima_csv = config.ACRIMA_CSV
    if rcs_csv is None:
        rcs_csv = os.path.join(config.RCS_DIR, "rcs_fullimage.csv")

    if not os.path.exists(acrima_csv):
        print(f"[CORR] ACRIMA.csv not found at {acrima_csv}")
        print("[CORR] Place ACRIMA.csv in the data/ folder to enable CDR correlation.")
        return None

    if not os.path.exists(rcs_csv):
        print(f"[CORR] RCS CSV not found at {rcs_csv}")
        print("[CORR] Run training first to generate RCS scores.")
        return None

    df_acrima = pd.read_csv(acrima_csv)
    df_rcs    = pd.read_csv(rcs_csv)

    # Normalise filename column
    if "Filename" in df_acrima.columns:
        df_acrima["filename"] = df_acrima["Filename"].apply(
            lambda x: os.path.basename(str(x))
        )
    elif "filename" in df_acrima.columns:
        df_acrima["filename"] = df_acrima["filename"].apply(
            lambda x: os.path.basename(str(x))
        )

    # Merge
    merged = pd.merge(df_rcs, df_acrima, on="filename", how="inner")
    print(f"[CORR] Merged {len(merged)} records (ACRIMA.csv ∩ RCS scores)")

    if len(merged) == 0:
        print("[CORR] WARNING: No matching filenames. Check ACRIMA.csv filename format.")
        return None

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Run all four statistical tests on RCS_OD vs ExpCDR.

    Tests:
      - Pearson r    (linear correlation)
      - Spearman ρ   (rank correlation)
      - Point-biserial r  (RCS_OD vs binary Glaucoma label)
      - Mann-Whitney U    (RCS_OD distribution: Glaucoma vs Normal)

    Args:
        df: Merged DataFrame with "rcs_od", "ExpCDR", "Glaucoma" columns

    Returns:
        dict with all test results
    """
    results = {}

    # Drop rows with NaN ExpCDR
    df_valid = df.dropna(subset=["ExpCDR", "rcs_od"])
    n = len(df_valid)
    print(f"[CORR] Valid pairs for correlation: {n}")

    if n < 5:
        print("[CORR] Too few samples for reliable correlation.")
        return {}

    rcs_od  = df_valid["rcs_od"].values
    exp_cdr = df_valid["ExpCDR"].values
    labels  = df_valid["Glaucoma"].values if "Glaucoma" in df_valid.columns \
              else df_valid["label"].values

    # ── Pearson r ─────────────────────────────────────────────────────────────
    r_val, r_p = pearsonr(rcs_od, exp_cdr)
    results["pearson"] = {
        "r":       round(float(r_val), 4),
        "p_value": round(float(r_p), 4),
        "significant": bool(r_p < 0.05),
    }

    # ── Spearman ρ ────────────────────────────────────────────────────────────
    rho_val, rho_p = spearmanr(rcs_od, exp_cdr)
    results["spearman"] = {
        "rho":     round(float(rho_val), 4),
        "p_value": round(float(rho_p), 4),
        "significant": bool(rho_p < 0.05),
    }

    # ── Point-biserial r ─────────────────────────────────────────────────────
    pb_val, pb_p = pointbiserialr(labels, rcs_od)
    results["point_biserial"] = {
        "r":       round(float(pb_val), 4),
        "p_value": round(float(pb_p), 4),
        "significant": bool(pb_p < 0.05),
    }

    # ── Mann-Whitney U ────────────────────────────────────────────────────────
    g_rcs = df_valid[df_valid["Glaucoma"] == 1]["rcs_od"].values \
            if "Glaucoma" in df_valid.columns \
            else df_valid[df_valid["label"] == 1]["rcs_od"].values
    n_rcs = df_valid[df_valid["Glaucoma"] == 0]["rcs_od"].values \
            if "Glaucoma" in df_valid.columns \
            else df_valid[df_valid["label"] == 0]["rcs_od"].values

    if len(g_rcs) > 0 and len(n_rcs) > 0:
        u_stat, u_p = mannwhitneyu(g_rcs, n_rcs, alternative="greater")
        results["mann_whitney"] = {
            "U":       round(float(u_stat), 2),
            "p_value": round(float(u_p), 4),
            "significant": bool(u_p < 0.05),
        }

    # Print summary
    print("\n[CORR] Results:")
    print(f"  Pearson r      = {results['pearson']['r']:.4f}  "
          f"(p={results['pearson']['p_value']:.4f})")
    print(f"  Spearman ρ     = {results['spearman']['rho']:.4f}  "
          f"(p={results['spearman']['p_value']:.4f})")
    print(f"  Point-biserial = {results['point_biserial']['r']:.4f}  "
          f"(p={results['point_biserial']['p_value']:.4f})")
    if "mann_whitney" in results:
        print(f"  Mann-Whitney U = {results['mann_whitney']['U']:.1f}  "
              f"(p={results['mann_whitney']['p_value']:.4f})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CORRELATION REPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_correlation_report(results: dict) -> str:
    """
    Save correlation analysis results as a formatted text report.

    Returns:
        Path to saved report
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    lines = [
        "═" * 60,
        "  CDR–RCS CORRELATION REPORT",
        "  (RCS_OD vs Expert Cup-to-Disc Ratio)",
        "═" * 60,
        "",
        "Hypothesis: Higher ExpCDR → Higher RCS_OD (glaucoma attention)",
        "",
        "── Pearson Correlation (linear) ─────────────────────────",
        f"  r       = {results.get('pearson', {}).get('r', 'N/A')}",
        f"  p-value = {results.get('pearson', {}).get('p_value', 'N/A')}",
        f"  Result  : {'SIGNIFICANT' if results.get('pearson', {}).get('significant') else 'not significant'}",
        "",
        "── Spearman Correlation (rank) ──────────────────────────",
        f"  ρ       = {results.get('spearman', {}).get('rho', 'N/A')}",
        f"  p-value = {results.get('spearman', {}).get('p_value', 'N/A')}",
        f"  Result  : {'SIGNIFICANT' if results.get('spearman', {}).get('significant') else 'not significant'}",
        "",
        "── Point-biserial Correlation ───────────────────────────",
        f"  r       = {results.get('point_biserial', {}).get('r', 'N/A')}",
        f"  p-value = {results.get('point_biserial', {}).get('p_value', 'N/A')}",
        f"  Result  : {'SIGNIFICANT' if results.get('point_biserial', {}).get('significant') else 'not significant'}",
        "",
        "── Mann-Whitney U Test ───────────────────────────────────",
        f"  U       = {results.get('mann_whitney', {}).get('U', 'N/A')}",
        f"  p-value = {results.get('mann_whitney', {}).get('p_value', 'N/A')}",
        f"  Result  : {'SIGNIFICANT' if results.get('mann_whitney', {}).get('significant') else 'not significant'}",
        "",
        "═" * 60,
    ]

    report = "\n".join(lines)
    save_path = os.path.join(config.REPORTS_DIR, "cdr_correlation_report.txt")
    with open(save_path, "w") as f:
        f.write(report)
    print(f"[CORR] Correlation report → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, results: dict = None) -> str:
    """
    Scatter plot of RCS_OD vs ExpCDR with regression line.

    Args:
        df:      Merged DataFrame (must have rcs_od and ExpCDR columns)
        results: Optional correlation results dict for annotation

    Returns:
        Path to saved figure
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    df_valid = df.dropna(subset=["ExpCDR", "rcs_od"])
    label_col = "Glaucoma" if "Glaucoma" in df_valid.columns else "label"

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {0: "#1E88E5", 1: "#E53935"}

    for lbl, color, name in [(0, "#1E88E5", "Normal"),
                              (1, "#E53935", "Glaucoma")]:
        subset = df_valid[df_valid[label_col] == lbl]
        ax.scatter(subset["ExpCDR"], subset["rcs_od"],
                   c=color, label=name, alpha=0.6, s=30, edgecolors="none")

    # Regression line
    z = np.polyfit(df_valid["ExpCDR"], df_valid["rcs_od"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_valid["ExpCDR"].min(), df_valid["ExpCDR"].max(), 100)
    ax.plot(x_range, p(x_range), "k--", linewidth=1.5, alpha=0.7,
            label="Linear fit")

    # Annotation
    if results and "pearson" in results:
        r_val = results["pearson"]["r"]
        p_val = results["pearson"]["p_value"]
        ax.text(0.05, 0.95, f"Pearson r = {r_val:.3f}\np = {p_val:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Expert Cup-to-Disc Ratio (ExpCDR)", fontsize=11)
    ax.set_ylabel("RCS_OD (Optic Disc Attention Score)", fontsize=11)
    ax.set_title("RCS_OD vs Expert CDR — Clinical Validation", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "cdr_rcs_scatter.png")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[CORR] Scatter plot saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_full_correlation_pipeline() -> None:
    """
    Load data, run all correlation tests, save report and figures.
    Safe to call even if ACRIMA.csv is missing (will skip gracefully).
    """
    df = load_and_merge()
    if df is None:
        print("[CORR] Skipping CDR correlation (data not available).")
        return

    results = run_correlation_analysis(df)
    if results:
        save_correlation_report(results)
        plot_correlation(df, results)
    print("[CORR] Correlation pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_correlation_pipeline()