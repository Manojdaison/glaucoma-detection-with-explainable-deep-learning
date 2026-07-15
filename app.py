"""
app.py  ·  GlaucomaNet Research Dashboard
Manoj · VIT Chennai · M.Sc. Data Science

Pages
-----
  🔬 Diagnosis  – single-model prediction + Grad-CAM + RCS
  🔵 Ensemble   – weighted average across all three models
  📋 History    – persistent record of every prediction

Storage
-------
  outputs/history/predictions.json     ← all records (JSON)
  outputs/history/images/<id>_*.png    ← per-record images
  outputs/logs/app.log                 ← prediction + error log
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, json, uuid, logging
from datetime import datetime
from io import BytesIO

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── project ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from gradcam import get_gradcam, overlay_heatmap
from rcs import compute_rcs_single
from model_builder import load_model

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_WEIGHTS: dict = {"fullimage": 0.60, "od": 0.20, "pp": 0.20}

THRESHOLDS: dict = {"fullimage": 0.50, "od": 0.50, "pp": 0.50, "ensemble": 0.50}

HISTORY_DIR    = os.path.join(config.OUTPUTS_DIR, "history")
HISTORY_IMAGES = os.path.join(HISTORY_DIR, "images")
HISTORY_FILE   = os.path.join(HISTORY_DIR, "predictions.json")
LOGS_DIR       = os.path.join(config.OUTPUTS_DIR, "logs")

MODEL_LABELS = {
    k: config.MODEL_REGISTRY[k]["description"]
    for k in config.MODEL_REGISTRY
}

# ── directory bootstrap ───────────────────────────────────────────────────────
for _d in (HISTORY_DIR, HISTORY_IMAGES, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("glaucomanet")

# ── load optimised thresholds (from threshold_optimizer.py if run) ────────────
_thr_file = os.path.join(config.OUTPUTS_DIR, "reports", "thresholds.json")
if os.path.isfile(_thr_file):
    try:
        with open(_thr_file) as _f:
            THRESHOLDS.update(json.load(_f))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY  (disk-backed — survives restarts)
# ═══════════════════════════════════════════════════════════════════════════════
def load_history() -> list:
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"load_history: {e}")
        return []

def save_history(records: list) -> None:
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        log.error(f"save_history: {e}")

def append_record(rec: dict) -> None:
    recs = load_history()
    recs.append(rec)
    save_history(recs)

def save_img(arr: np.ndarray, rid: str, tag: str) -> str:
    try:
        p = os.path.join(HISTORY_IMAGES, f"{rid}_{tag}.png")
        Image.fromarray(arr.astype(np.uint8)).save(p)
        return p
    except Exception as e:
        log.warning(f"save_img {tag}: {e}")
        return ""

def load_img(path: str) -> np.ndarray | None:
    if path and os.path.isfile(path):
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None
    return None

def history_stats(records: list) -> dict:
    n_g    = sum(1 for r in records if r.get("prediction") == "Glaucoma")
    n_diag = sum(1 for r in records if r.get("page")       == "diagnosis")
    n_ens  = sum(1 for r in records if r.get("page")       == "ensemble")
    return {"total": len(records), "glaucoma": n_g,
            "normal": len(records) - n_g,
            "diagnosis": n_diag, "ensemble": n_ens}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE  (one load per session)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model(key: str):
    return load_model(key)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_img(img_rgb: np.ndarray) -> np.ndarray:
    h, w = config.IMG_SIZE
    r = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(preprocess_input(r.astype(np.float32)), 0)

def run_gradcam(model, x: np.ndarray, img_rgb: np.ndarray) -> dict:
    try:
        _, heatmap  = get_gradcam(model, x)
        overlay     = overlay_heatmap(img_rgb, heatmap)
        heat_u8     = np.uint8(255 * heatmap)
        heat_rgb    = cv2.cvtColor(
            cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )
        rcs         = compute_rcs_single(heatmap, config.IMG_SIZE)
        return {"ok": True, "overlay": overlay, "heatmap": heat_rgb, "rcs": rcs}
    except Exception as e:
        log.warning(f"Grad-CAM failed: {e}")
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS  (reusable — called from every page)
# ═══════════════════════════════════════════════════════════════════════════════
def page_header(title: str, subtitle: str = "") -> None:
    """Full-width page title with optional subtitle."""
    sub = (f"<p style='margin:6px 0 0;font-size:.875rem;"
           f"color:var(--text-muted);font-weight:400'>{subtitle}</p>"
           if subtitle else "")
    st.markdown(
        f"<div class='page-header'><h1>{title}</h1>{sub}</div>",
        unsafe_allow_html=True,
    )

def section_header(title: str) -> None:
    """Blue-accented sub-section title."""
    st.markdown(f"<div class='section-header'>{title}</div>",
                unsafe_allow_html=True)

def stat_card(value, label: str, color: str = "blue") -> str:
    """Return HTML for a single metric card (used in st.columns)."""
    accent = {"blue": "#3b82f6", "red": "#ef4444",
              "green": "#22c55e", "amber": "#f59e0b"}.get(color, "#3b82f6")
    return (
        f"<div class='stat-card' style='border-top:3px solid {accent}'>"
        f"<div class='stat-val' style='color:{accent}'>{value}</div>"
        f"<div class='stat-lbl'>{label}</div>"
        f"</div>"
    )

def prediction_badge(label: str, prob: float, threshold: float) -> None:
    """Large prominent prediction result card."""
    is_g   = label == "Glaucoma"
    color  = "#ef4444" if is_g else "#22c55e"
    bg     = "rgba(239,68,68,.1)" if is_g else "rgba(34,197,94,.1)"
    border = "rgba(239,68,68,.35)" if is_g else "rgba(34,197,94,.35)"
    icon   = "⚠️" if is_g else "✅"
    pct    = f"{prob * 100:.1f}%"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
         padding:24px 28px;margin:12px 0">
      <div style="display:flex;align-items:center;gap:14px">
        <span style="font-size:2.2rem">{icon}</span>
        <div>
          <div style="font-size:1.5rem;font-weight:700;color:{color}">{label}</div>
          <div style="font-size:.8rem;color:var(--text-muted);margin-top:2px">
            AI Screening Result</div>
        </div>
        <div style="margin-left:auto;text-align:right">
          <div style="font-size:2.4rem;font-weight:800;color:{color}">{pct}</div>
          <div style="font-size:.75rem;color:var(--text-muted)">
            probability · threshold&nbsp;{threshold:.2f}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.progress(float(min(prob, 1.0)))

def model_prob_card(desc: str, prob: float, weight: float, label: str) -> str:
    """Return HTML for one model's result inside the ensemble breakdown."""
    is_g   = label == "Glaucoma"
    color  = "#ef4444" if is_g else "#22c55e"
    bg     = "rgba(239,68,68,.08)" if is_g else "rgba(34,197,94,.08)"
    border = "rgba(239,68,68,.25)" if is_g else "rgba(34,197,94,.25)"
    return (
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-radius:10px;padding:18px 16px;text-align:center'>"
        f"<div style='font-size:.78rem;color:var(--text-muted);margin-bottom:8px;"
        f"font-weight:600;text-transform:uppercase;letter-spacing:.06em'>{desc}</div>"
        f"<div style='font-size:1.9rem;font-weight:800;color:{color}'>"
        f"{prob*100:.1f}%</div>"
        f"<div style='font-size:.8rem;color:{color};font-weight:600;margin:4px 0'>"
        f"{label}</div>"
        f"<div style='font-size:.73rem;color:var(--text-muted)'>"
        f"weight&nbsp;{weight:.0%}</div>"
        f"</div>"
    )

def alert(msg: str, kind: str = "info") -> None:
    """Inline alert box — kind: info | warn | success | error"""
    cfg = {
        "info":    ("#3b82f6", "rgba(59,130,246,.1)",  "rgba(59,130,246,.3)",  "ℹ️"),
        "warn":    ("#f59e0b", "rgba(245,158,11,.1)",  "rgba(245,158,11,.3)",  "⚠️"),
        "success": ("#22c55e", "rgba(34,197,94,.1)",   "rgba(34,197,94,.3)",   "✅"),
        "error":   ("#ef4444", "rgba(239,68,68,.1)",   "rgba(239,68,68,.3)",   "❌"),
    }.get(kind, ("#3b82f6", "rgba(59,130,246,.1)", "rgba(59,130,246,.3)", "ℹ️"))
    color, bg, border, ico = cfg
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-radius:8px;padding:10px 14px;margin:8px 0;"
        f"font-size:.875rem;color:var(--text-secondary)'>"
        f"{ico}&nbsp;&nbsp;{msg}</div>",
        unsafe_allow_html=True,
    )

def empty_state(icon: str, title: str, body: str = "") -> None:
    """Centred empty-state placeholder."""
    sub = (f"<p style='margin:8px 0 0;font-size:.85rem;"
           f"color:var(--text-muted)'>{body}</p>" if body else "")
    st.markdown(f"""
    <div style="border:2px dashed var(--border);border-radius:16px;
         padding:64px 32px;text-align:center;margin:24px 0">
      <div style="font-size:3rem;margin-bottom:12px">{icon}</div>
      <p style="font-size:1rem;font-weight:600;color:var(--accent);margin:0">
        {title}</p>
      {sub}
    </div>""", unsafe_allow_html=True)

def image_panel(img: np.ndarray | None, caption: str,
                placeholder: str = "Not available") -> None:
    """Image with styled caption; shows placeholder text if img is None."""
    st.markdown(f"<p class='img-caption'>{caption}</p>",
                unsafe_allow_html=True)
    if img is not None:
        st.image(img, width=350)
    else:
        st.markdown(
            f"<div class='img-placeholder'>{placeholder}</div>",
            unsafe_allow_html=True,
        )

def rcs_chart(rcs: dict) -> plt.Figure:
    """Horizontal bar chart for Region Contribution Scores."""
    regions = ["Optic Disc", "Peripapillary", "Background"]
    values  = [rcs.get("od", 0), rcs.get("peripapillary", 0),
               rcs.get("background", 0)]
    colors  = ["#ef4444", "#3b82f6", "#64748b"]

    fig, ax = plt.subplots(figsize=(6, 1.9))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    bars = ax.barh(regions, values, color=colors,
                   edgecolor="none", height=0.42)
    ax.set_xlim(0, 1.18)
    ax.tick_params(colors="#94a3b8", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.set_xlabel("Attention proportion", color="#64748b", fontsize=8.5)
    ax.xaxis.label.set_color("#64748b")

    for bar, v in zip(bars, values):
        ax.text(v + 0.025, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left",
                color="#f1f5f9", fontsize=9.5, fontweight="600")

    fig.tight_layout(pad=0.3)
    return fig

def divider() -> None:
    st.markdown("<div style='margin:20px 0;border-top:1px solid var(--border)'></div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GlaucomaNet · Research Dashboard",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  (CSS custom properties + component styles)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Design tokens ── */
:root {
  --bg:           #080d18;
  --bg-card:      #0f172a;
  --bg-card-alt:  #111827;
  --border:       #1e293b;
  --border-light: #263347;
  --accent:       #3b82f6;
  --accent-dark:  #2563eb;
  --accent-glow:  rgba(59,130,246,.15);
  --success:      #22c55e;
  --danger:       #ef4444;
  --warning:      #f59e0b;
  --text-primary:   #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted:     #64748b;
  --radius-sm:  6px;
  --radius-md:  10px;
  --radius-lg:  14px;
  --shadow:     0 4px 24px rgba(0,0,0,.45);
}

/* ── Global reset ── */
*, *::before, *::after {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  box-sizing: border-box;
}
.stApp {
  background: var(--bg) !important;
  color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #070c16 !important;
  border-right: 1px solid var(--border) !important;
  min-width: 240px !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem !important; }
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] b { color: var(--text-primary) !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
  text-align: center;
  padding: 20px 0 16px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.sidebar-logo-icon  { font-size: 2.8rem; line-height: 1; }
.sidebar-logo-name  { font-size: 1.15rem; font-weight: 700; color: var(--accent) !important;
                      margin: 8px 0 2px; letter-spacing: -.01em; }
.sidebar-logo-tag   { font-size: .62rem; color: var(--text-muted) !important;
                      letter-spacing: .18em; text-transform: uppercase; }

/* ── Sidebar stats ── */
.sidebar-stat-box {
  background: rgba(59,130,246,.07);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  font-size: .82rem;
  line-height: 1.9;
}

/* ── Page header ── */
.page-header {
  padding: 0 0 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}
.page-header h1 {
  font-size: 1.65rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
  letter-spacing: -.02em;
}

/* ── Section header ── */
.section-header {
  font-size: .95rem;
  font-weight: 600;
  color: var(--accent);
  border-left: 3px solid var(--accent);
  padding: 2px 0 2px 10px;
  margin: 20px 0 12px;
  letter-spacing: .01em;
}

/* ── Stat cards (history metrics row) ── */
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px 12px;
  text-align: center;
}
.stat-val { font-size: 1.9rem; font-weight: 800; line-height: 1.1; }
.stat-lbl { font-size: .68rem; color: var(--text-muted); text-transform: uppercase;
             letter-spacing: .1em; margin-top: 5px; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
  background: rgba(59,130,246,.04) !important;
  border: 2px dashed var(--border-light) !important;
  border-radius: var(--radius-lg) !important;
  padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background: var(--accent-glow) !important;
}

/* ── Streamlit select / radio / slider overrides ── */
.stSelectbox > div > div,
.stSlider > div {
  background: var(--bg-card) !important;
}
.stRadio > div { gap: 4px !important; }
.stRadio label {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 7px 14px !important;
  font-size: .875rem;
  transition: all .15s;
  cursor: pointer;
}
.stRadio label:hover { border-color: var(--accent); }
.stRadio [data-checked="true"] label {
  background: var(--accent-glow);
  border-color: var(--accent);
  color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-light) !important;
  color: var(--text-secondary) !important;
  border-radius: var(--radius-sm) !important;
  font-size: .85rem !important;
  padding: 6px 16px !important;
  transition: all .15s !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
  background: rgba(239,68,68,.12) !important;
  border-color: rgba(239,68,68,.4) !important;
  color: #fca5a5 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
  background: rgba(59,130,246,.1) !important;
  border: 1px solid rgba(59,130,246,.35) !important;
  color: var(--accent) !important;
  border-radius: var(--radius-sm) !important;
  font-size: .85rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div { background: var(--accent) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary {
  font-size: .875rem !important;
  color: var(--text-secondary) !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text-primary) !important; }

/* ── Info / success / warning / error native boxes ── */
.stAlert { border-radius: var(--radius-md) !important; font-size: .875rem !important; }

/* ── Image caption ── */
.img-caption {
  font-size: .75rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin: 0 0 6px;
}
.img-placeholder {
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: .82rem;
}

/* ── Control panel card (sidebar-like left col) ── */
.control-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 18px;
}
.control-label {
  font-size: .72rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: 6px;
}
.threshold-pill {
  display: inline-block;
  background: rgba(59,130,246,.1);
  border: 1px solid rgba(59,130,246,.25);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: .78rem;
  color: var(--accent);
  font-weight: 600;
}

/* ── History record row ── */
.history-meta {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 14px;
}
.history-meta-item { font-size: .82rem; }
.history-meta-label {
  font-size: .68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 3px;
}

/* ── Misc ── */
code {
  background: rgba(59,130,246,.12) !important;
  color: #93c5fd !important;
  border-radius: 4px !important;
  padding: 1px 6px !important;
  font-size: .8rem !important;
}
hr { border-color: var(--border) !important; }
h1,h2,h3,h4,h5 { color: var(--text-primary) !important; }
p { color: var(--text-secondary); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">👁</div>
      <div class="sidebar-logo-name">GlaucomaNet</div>
      <div class="sidebar-logo-tag">Research Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔬  Diagnosis", "🔵  Ensemble", "📋  History"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin:16px 0 8px;font-size:.7rem;color:var(--text-muted);"
                "text-transform:uppercase;letter-spacing:.1em'>Session Stats</div>",
                unsafe_allow_html=True)

    # Load once for sidebar — don't call again until needed
    _hist_all = load_history()
    _s        = history_stats(_hist_all)
    st.markdown(f"""
    <div class="sidebar-stat-box">
      <div>Total predictions&nbsp;&nbsp;<b>{_s['total']}</b></div>
      <div>Glaucoma&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
           <b style='color:#f87171'>{_s['glaucoma']}</b></div>
      <div>Normal&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
           <b style='color:#4ade80'>{_s['normal']}</b></div>
      <div>Diagnosis runs&nbsp;&nbsp;&nbsp;
           <b>{_s['diagnosis']}</b></div>
      <div>Ensemble runs&nbsp;&nbsp;&nbsp;&nbsp;
           <b>{_s['ensemble']}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:auto;padding-top:20px;"
                "font-size:.72rem;color:var(--text-muted)'>"
                "VIT Chennai &nbsp;·&nbsp; M.Sc. Data Science<br>"
                "Manoj &nbsp;·&nbsp; 2024</div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════
if "Diagnosis" in page:
    page_header(
        "🔬 Single-Model Diagnosis",
        "Upload a retinal fundus image and select a model to generate a prediction with Grad-CAM explainability.",
    )

    left, right = st.columns([1, 2.2], gap="large")

    # ── Left column: controls ─────────────────────────────────────────────────
    with left:
        with st.container():
            st.markdown("<div class='control-card'>", unsafe_allow_html=True)

            st.markdown("<div class='control-label'>Select Model</div>",
                        unsafe_allow_html=True)
            model_key = st.selectbox(
                "Model",
                list(config.MODEL_REGISTRY.keys()),
                format_func=lambda k: MODEL_LABELS[k],
                label_visibility="collapsed",
            )

            thr = THRESHOLDS.get(model_key, 0.50)
            st.markdown(
                f"<div style='margin:10px 0 16px'>"
                f"<div class='control-label'>Decision Threshold</div>"
                f"<span class='threshold-pill'>{thr:.3f}</span>"
                f"<span style='font-size:.73rem;color:var(--text-muted);"
                f"margin-left:8px'>Youden-optimised</span></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='control-label'>Upload Image</div>",
                        unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed", key="diag_up",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            st.markdown("<div style='margin-top:12px;font-size:.78rem;"
                        "color:var(--text-muted);text-align:center'>"
                        f"📄 {uploaded.name}</div>", unsafe_allow_html=True)

    # ── Right column: results ─────────────────────────────────────────────────
    with right:
        if uploaded is None:
            empty_state(
                "🏥",
                "No image uploaded",
                "Use the panel on the left to select a model and upload a fundus image.",
            )
        else:
            # Load image
            try:
                img_rgb = np.array(
                    Image.open(uploaded).convert("RGB").resize(config.IMG_SIZE[::-1]),
                    dtype=np.uint8,
                )
                x = preprocess_img(img_rgb)
            except Exception as e:
                st.error(f"Cannot open image: {e}")
                log.error(f"Image load: {e}")
                st.stop()

            # Load model
            with st.spinner("Loading model …"):
                try:
                    model = get_model(model_key)
                    if model is None:
                        st.error("Model file not found. Run `python train.py` first.")
                        st.stop()
                except Exception as e:
                    st.error(f"Model load failed: {e}")
                    log.error(f"Model load ({model_key}): {e}")
                    st.stop()

            # Predict
            try:
                pred_prob  = float(model.predict(x, verbose=0)[0, 0])
                pred_label = "Glaucoma" if pred_prob >= thr else "Normal"
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                log.error(f"Prediction ({model_key}): {e}")
                st.stop()

            # Grad-CAM
            with st.spinner("Computing Grad-CAM …"):
                gc = run_gradcam(model, x, img_rgb)

            # ── Result badge ──────────────────────────────────────────────────
            prediction_badge(pred_label, pred_prob, thr)

            # ── Image trio ───────────────────────────────────────────────────
            divider()
            section_header("Visual Analysis")
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                image_panel(img_rgb, "Original Fundus")
            with c2:
                image_panel(
                    gc.get("overlay") if gc["ok"] else None,
                    "Grad-CAM Overlay",
                    "Grad-CAM unavailable",
                )
            with c3:
                image_panel(
                    gc.get("heatmap") if gc["ok"] else None,
                    "Saliency Heatmap",
                    "Grad-CAM unavailable",
                )

            if not gc["ok"]:
                alert(f"Grad-CAM failed — {gc.get('error','unknown error')}", "warn")

            # ── RCS ───────────────────────────────────────────────────────────
            if gc["ok"]:
                divider()
                section_header("Region Contribution Score (RCS)")
                rcs = gc["rcs"]

                rcs_left, rcs_right = st.columns([2, 1], gap="large")
                with rcs_left:
                    fig = rcs_chart(rcs)
                    st.pyplot(fig)
                    plt.close(fig)
                with rcs_right:
                    st.markdown(f"""
                    <div style='font-size:.82rem;color:var(--text-secondary);
                         line-height:1.8'>
                      <div><span style='color:#ef4444'>■</span>
                           &nbsp;Optic Disc&emsp;<b>{rcs.get('od',0):.3f}</b></div>
                      <div><span style='color:#3b82f6'>■</span>
                           &nbsp;Peripapillary&emsp;<b>{rcs.get('peripapillary',0):.3f}</b></div>
                      <div><span style='color:#64748b'>■</span>
                           &nbsp;Background&emsp;<b>{rcs.get('background',0):.3f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                if rcs.get("background", 0) > 0.60:
                    alert("More than 60 % of model attention falls on the background — "
                          "the model may be keying off image artifacts rather than "
                          "clinical structures. Retrain with fixed region crops.", "warn")
                elif rcs.get("od", 0) > 0.40 and pred_label == "Glaucoma":
                    alert("High optic-disc attention ✓ — model is focusing on a "
                          "clinically relevant region (cup enlargement / rim thinning).",
                          "success")

            # ── Persist record ────────────────────────────────────────────────
            rid    = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6]
            rcs_d  = gc.get("rcs", {}) if gc["ok"] else {}
            record = {
                "id": rid,
                "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename":    uploaded.name,
                "page":        "diagnosis",
                "model":       model_key,
                "model_label": MODEL_LABELS[model_key],
                "prediction":  pred_label,
                "probability": round(pred_prob, 6),
                "threshold":   thr,
                "gradcam_ok":  gc["ok"],
                "rcs_od":  round(rcs_d.get("od",           0), 4) if gc["ok"] else None,
                "rcs_pp":  round(rcs_d.get("peripapillary",0), 4) if gc["ok"] else None,
                "rcs_bg":  round(rcs_d.get("background",   0), 4) if gc["ok"] else None,
                "original_img": save_img(img_rgb, rid, "original"),
                "overlay_img":  save_img(gc["overlay"], rid, "overlay") if gc["ok"] else "",
                "heatmap_img":  save_img(gc["heatmap"], rid, "heatmap") if gc["ok"] else "",
            }
            append_record(record)
            log.info(f"Diagnosis | {rid} | {model_key} | {pred_label} | {pred_prob:.4f}")

            alert(f"Prediction saved to History — ID: <code>{rid}</code>", "success")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Ensemble" in page:
    page_header(
        "🔵 Ensemble Prediction",
        "Combine all three region models with configurable weights for a robust consensus prediction.",
    )

    # ── Weight controls ───────────────────────────────────────────────────────
    with st.expander("⚙️  Configure model weights", expanded=False):
        wc1, wc2, wc3 = st.columns(3, gap="medium")
        w_full = wc1.slider("Full Image Model",    0.0, 1.0,
                            ENSEMBLE_WEIGHTS["fullimage"], 0.05, key="w_full")
        w_od   = wc2.slider("Optic Disc Model",    0.0, 1.0,
                            ENSEMBLE_WEIGHTS["od"],        0.05, key="w_od")
        w_pp   = wc3.slider("Peripapillary Model", 0.0, 1.0,
                            ENSEMBLE_WEIGHTS["pp"],        0.05, key="w_pp")
        wt_sum = w_full + w_od + w_pp
        if abs(wt_sum - 1.0) > 0.01:
            alert(f"Weights sum to {wt_sum:.2f} — will be auto-normalised to 1.0.", "warn")
        else:
            alert(f"Weights sum to {wt_sum:.2f} — valid. ✓", "success")

    live_w  = {"fullimage": w_full, "od": w_od, "pp": w_pp}
    ens_thr = THRESHOLDS.get("ensemble", 0.50)

    # ── Upload ────────────────────────────────────────────────────────────────
    up_ens = st.file_uploader(
        "Upload Fundus Image",
        type=["jpg", "jpeg", "png"],
        key="ens_up",
    )

    if up_ens is None:
        empty_state(
            "🔵",
            "No image uploaded",
            "Upload a retinal fundus image above to run the ensemble.",
        )
    else:
        # Load image
        try:
            img_ens = np.array(
                Image.open(up_ens).convert("RGB").resize(config.IMG_SIZE[::-1]),
                dtype=np.uint8,
            )
            x_ens = preprocess_img(img_ens)
        except Exception as e:
            st.error(f"Cannot open image: {e}")
            st.stop()

        # Normalise weights
        nw_sum = sum(live_w.values())
        nw     = {k: v / nw_sum if nw_sum > 0 else 1/3 for k, v in live_w.items()}

        per_model: dict     = {}
        ensemble_score: float = 0.0
        weight_used: float    = 0.0
        ens_gc                = None

        with st.spinner("Running all three models …"):
            for key in config.MODEL_REGISTRY:
                thr_k = THRESHOLDS.get(key, 0.50)
                try:
                    m = get_model(key)
                    if m is None:
                        per_model[key] = {"ok": False, "error": "Model not found"}
                        continue
                    prob   = float(m.predict(x_ens, verbose=0)[0, 0])
                    w      = nw.get(key, 0.0)
                    lbl    = "Glaucoma" if prob >= thr_k else "Normal"
                    gc_k   = run_gradcam(m, x_ens, img_ens)
                    per_model[key]  = {"ok": True, "prob": prob, "weight": w,
                                       "label": lbl, "threshold": thr_k, "gc": gc_k}
                    ensemble_score += w * prob
                    weight_used    += w
                    if key == "fullimage" and gc_k["ok"]:
                        ens_gc = gc_k
                except Exception as e:
                    per_model[key] = {"ok": False, "error": str(e)}
                    log.error(f"Ensemble model ({key}): {e}")

        if weight_used > 1e-6:
            ensemble_score /= weight_used

        ens_label = "Glaucoma" if ensemble_score >= ens_thr else "Normal"

        # ── Ensemble result ───────────────────────────────────────────────────
        prediction_badge(ens_label, ensemble_score, ens_thr)

        # ── Per-model cards ───────────────────────────────────────────────────
        divider()
        section_header("Per-Model Breakdown")
        mc1, mc2, mc3 = st.columns(3, gap="medium")
        for col, key in zip((mc1, mc2, mc3), config.MODEL_REGISTRY):
            info = per_model.get(key, {})
            desc = MODEL_LABELS[key]
            with col:
                if info.get("ok"):
                    st.markdown(
                        model_prob_card(desc, info["prob"], info["weight"],
                                        info["label"]),
                        unsafe_allow_html=True,
                    )
                    st.progress(float(min(info["prob"], 1.0)))
                else:
                    alert(f"{desc}<br>{info.get('error','Failed')}", "error")

        # ── Visual analysis ───────────────────────────────────────────────────
        divider()
        section_header("Explainability · Full Image Model")
        v1, v2, v3 = st.columns(3, gap="medium")
        with v1:
            image_panel(img_ens, "Original Fundus")
        with v2:
            image_panel(
                ens_gc.get("overlay") if ens_gc and ens_gc["ok"] else None,
                "Grad-CAM Overlay", "Not available",
            )
        with v3:
            image_panel(
                ens_gc.get("heatmap") if ens_gc and ens_gc["ok"] else None,
                "Saliency Heatmap", "Not available",
            )

        if ens_gc and ens_gc["ok"]:
            divider()
            section_header("Region Contribution Score")
            rc1, rc2 = st.columns([2, 1], gap="large")
            with rc1:
                fig_e = rcs_chart(ens_gc["rcs"])
                st.pyplot(fig_e)
                plt.close(fig_e)
            with rc2:
                _rcs = ens_gc["rcs"]
                st.markdown(f"""
                <div style='font-size:.82rem;color:var(--text-secondary);line-height:1.8'>
                  <div><span style='color:#ef4444'>■</span>&nbsp;
                       Optic Disc &emsp;<b>{_rcs.get('od',0):.3f}</b></div>
                  <div><span style='color:#3b82f6'>■</span>&nbsp;
                       Peripapillary &emsp;<b>{_rcs.get('peripapillary',0):.3f}</b></div>
                  <div><span style='color:#64748b'>■</span>&nbsp;
                       Background &emsp;<b>{_rcs.get('background',0):.3f}</b></div>
                </div>""", unsafe_allow_html=True)

        # ── Persist ───────────────────────────────────────────────────────────
        rid     = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6]
        ens_rcs = ens_gc["rcs"] if ens_gc and ens_gc["ok"] else {}
        bd_log  = {k: {"prob": round(v["prob"], 6),
                        "weight": round(v["weight"], 4),
                        "label": v["label"]}
                   for k, v in per_model.items() if v.get("ok")}
        record = {
            "id":           rid,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":     up_ens.name,
            "page":         "ensemble",
            "model":        "ensemble",
            "model_label":  "Ensemble (3 Models)",
            "prediction":   ens_label,
            "probability":  round(ensemble_score, 6),
            "threshold":    ens_thr,
            "gradcam_ok":   bool(ens_gc and ens_gc["ok"]),
            "rcs_od":  round(ens_rcs.get("od",           0), 4) if ens_rcs else None,
            "rcs_pp":  round(ens_rcs.get("peripapillary",0), 4) if ens_rcs else None,
            "rcs_bg":  round(ens_rcs.get("background",   0), 4) if ens_rcs else None,
            "original_img": save_img(img_ens, rid, "original"),
            "overlay_img":  save_img(ens_gc["overlay"], rid, "overlay")
                            if ens_gc and ens_gc["ok"] else "",
            "heatmap_img":  save_img(ens_gc["heatmap"], rid, "heatmap")
                            if ens_gc and ens_gc["ok"] else "",
            "ensemble_breakdown": bd_log,
        }
        append_record(record)
        log.info(f"Ensemble | {rid} | {ensemble_score:.4f} | {ens_label}")
        alert(f"Ensemble prediction saved to History — ID: <code>{rid}</code>", "success")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif "History" in page:
    page_header(
        "📋 Prediction History",
        "Every prediction is stored permanently. Search, filter, export, or inspect individual records.",
    )

    records = load_history()

    if not records:
        empty_state(
            "📂",
            "No predictions recorded yet",
            "Run a Diagnosis or Ensemble prediction first.",
        )
        st.stop()

    # ── Summary row ───────────────────────────────────────────────────────────
    s = history_stats(records)
    sc = st.columns(5, gap="small")
    for col, val, lbl, color in [
        (sc[0], s["total"],     "Total",    "blue"),
        (sc[1], s["glaucoma"],  "Glaucoma", "red"),
        (sc[2], s["normal"],    "Normal",   "green"),
        (sc[3], s["diagnosis"], "Diagnosis","blue"),
        (sc[4], s["ensemble"],  "Ensemble", "amber"),
    ]:
        col.markdown(stat_card(val, lbl, color), unsafe_allow_html=True)

    divider()

    # ── Filter bar ────────────────────────────────────────────────────────────
    section_header("Filter & Search")
    fa, fb, fc, fd = st.columns([3, 1.4, 1.4, 1.2], gap="small")
    with fa:
        q   = st.text_input("Search", placeholder="filename or record ID …",
                            label_visibility="collapsed")
    with fb:
        fp  = st.selectbox("Prediction", ["All", "Glaucoma", "Normal"],
                           label_visibility="collapsed")
    with fc:
        fs  = st.selectbox("Source", ["All", "diagnosis", "ensemble"],
                           label_visibility="collapsed")
    with fd:
        so  = st.selectbox("Sort", ["Newest first", "Oldest first"],
                           label_visibility="collapsed")

    filtered = list(records)
    if q.strip():
        ql = q.strip().lower()
        filtered = [r for r in filtered
                    if ql in r.get("filename","").lower()
                    or ql in r.get("id","").lower()]
    if fp != "All":
        filtered = [r for r in filtered if r.get("prediction") == fp]
    if fs != "All":
        filtered = [r for r in filtered if r.get("page") == fs]
    if so == "Newest first":
        filtered = list(reversed(filtered))

    # ── Toolbar: export + count + clear ──────────────────────────────────────
    tb1, tb2, tb3 = st.columns([2, 3, 1], gap="small")
    with tb1:
        if filtered:
            rows = [{"ID": r.get("id"), "Timestamp": r.get("timestamp"),
                     "Filename": r.get("filename"), "Source": r.get("page"),
                     "Model": r.get("model_label"), "Prediction": r.get("prediction"),
                     "Probability": r.get("probability"), "Threshold": r.get("threshold"),
                     "RCS_OD": r.get("rcs_od"), "RCS_PP": r.get("rcs_pp"),
                     "RCS_BG": r.get("rcs_bg")} for r in filtered]
            st.download_button(
                "⬇️  Export as CSV",
                data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                file_name=f"glaucomanet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    with tb2:
        st.markdown(
            f"<div style='padding:8px 0;font-size:.83rem;color:var(--text-muted)'>"
            f"Showing <b style='color:var(--text-secondary)'>{len(filtered)}</b> "
            f"of <b style='color:var(--text-secondary)'>{len(records)}</b> records"
            f"</div>",
            unsafe_allow_html=True,
        )
    with tb3:
        with st.expander("⚠️ Danger"):
            if st.button("Clear all history", type="primary"):
                save_history([])
                st.success("Cleared. Refresh the page.")
                st.stop()

    divider()

    if not filtered:
        alert("No records match the current filter.", "info")
        st.stop()

    # ── Record cards ──────────────────────────────────────────────────────────
    for rec in filtered:
        pred   = rec.get("prediction", "Unknown")
        prob   = rec.get("probability", 0.0)
        ts     = rec.get("timestamp",  "—")
        fname  = rec.get("filename",   "unknown")
        source = rec.get("page",       "—")
        mlabel = rec.get("model_label", rec.get("model", "—"))
        rid    = rec.get("id", "—")
        is_g   = pred == "Glaucoma"
        ico    = "🔴" if is_g else "🟢"
        pred_color = "#f87171" if is_g else "#4ade80"

        # Expander header: icon + prediction + filename + timestamp
        with st.expander(
            f"{ico}  {pred} · {prob*100:.1f}%  ·  {fname}  ·  {ts}",
            expanded=False,
        ):
            # ── Meta row ─────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4, gap="small")
            m1.markdown(f"<div class='history-meta-label'>Record ID</div>"
                        f"<code>{rid}</code>", unsafe_allow_html=True)
            m2.markdown(f"<div class='history-meta-label'>Timestamp</div>"
                        f"<span style='font-size:.83rem'>{ts}</span>",
                        unsafe_allow_html=True)
            m3.markdown(f"<div class='history-meta-label'>Source</div>"
                        f"<span style='font-size:.83rem'>{source.capitalize()}</span>",
                        unsafe_allow_html=True)
            m4.markdown(f"<div class='history-meta-label'>Model</div>"
                        f"<span style='font-size:.83rem'>{mlabel}</span>",
                        unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # ── Prediction result ─────────────────────────────────────────────
            st.markdown(
                f"<div style='background:rgba(0,0,0,.2);border:1px solid var(--border);"
                f"border-left:4px solid {pred_color};border-radius:var(--radius-md);"
                f"padding:12px 18px;display:flex;align-items:center;gap:12px'>"
                f"<span style='font-size:1.5rem'>{ico}</span>"
                f"<div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{pred_color}'>"
                f"{pred}</div>"
                f"<div style='font-size:.78rem;color:var(--text-muted)'>"
                f"Probability {prob:.4f} &nbsp;·&nbsp; "
                f"Threshold {rec.get('threshold',0.5):.3f} &nbsp;·&nbsp; "
                f"File: {fname}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            # ── Images ───────────────────────────────────────────────────────
            i1, i2, i3 = st.columns(3, gap="small")
            orig = load_img(rec.get("original_img", ""))
            ov   = load_img(rec.get("overlay_img",  ""))
            hm   = load_img(rec.get("heatmap_img",  ""))

            with i1:
                image_panel(orig, "Original Fundus",
                            "Image file missing from disk")
            with i2:
                image_panel(ov, "Grad-CAM Overlay",
                            "Grad-CAM failed at prediction time"
                            if not rec.get("gradcam_ok") else "File missing")
            with i3:
                image_panel(hm, "Saliency Heatmap",
                            "Grad-CAM failed at prediction time"
                            if not rec.get("gradcam_ok") else "File missing")

            # ── RCS chart ─────────────────────────────────────────────────────
            if rec.get("rcs_od") is not None:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                section_header("Region Contribution Score")
                r1, r2 = st.columns([2, 1], gap="large")
                with r1:
                    fig_r = rcs_chart({
                        "od":            rec.get("rcs_od", 0),
                        "peripapillary": rec.get("rcs_pp", 0),
                        "background":    rec.get("rcs_bg", 0),
                    })
                    st.pyplot(fig_r)
                    plt.close(fig_r)
                with r2:
                    st.markdown(
                        f"<div style='font-size:.81rem;color:var(--text-secondary);"
                        f"line-height:2'>"
                        f"<div><span style='color:#ef4444'>■</span> OD &emsp;"
                        f"<b>{rec.get('rcs_od',0):.3f}</b></div>"
                        f"<div><span style='color:#3b82f6'>■</span> PP &emsp;"
                        f"<b>{rec.get('rcs_pp',0):.3f}</b></div>"
                        f"<div><span style='color:#64748b'>■</span> BG &emsp;"
                        f"<b>{rec.get('rcs_bg',0):.3f}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Ensemble breakdown ────────────────────────────────────────────
            if source == "ensemble" and rec.get("ensemble_breakdown"):
                st.markdown("<div style='height:6px'></div>",
                            unsafe_allow_html=True)
                section_header("Ensemble Breakdown")
                bd = rec["ensemble_breakdown"]
                b_cols = st.columns(len(bd), gap="small")
                for col, (k, info) in zip(b_cols, bd.items()):
                    desc_k = config.MODEL_REGISTRY.get(k, {}).get("description", k)
                    col.markdown(
                        model_prob_card(
                            desc_k,
                            info.get("prob", 0),
                            info.get("weight", 0),
                            info.get("label", "?"),
                        ),
                        unsafe_allow_html=True,
                    )

            # ── Delete ────────────────────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🗑️  Delete this record", key=f"del_{rid}"):
                updated = [r for r in load_history() if r.get("id") != rid]
                save_history(updated)
                for ip in (rec.get("original_img"), rec.get("overlay_img"),
                           rec.get("heatmap_img")):
                    if ip and os.path.isfile(ip):
                        try:
                            os.remove(ip)
                        except Exception:
                            pass
                st.success("Record deleted — refresh the page to update the list.")
                log.info(f"Deleted record {rid}")