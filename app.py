import os
import math
from datetime import datetime

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# REVIEWGUARD AI ANALYTICS — SIGNATURE LIVE DASHBOARD
# ============================================================

BASE = r"D:\Trained Dataset files\Ahad Data Set"

DEPLOY_MODEL = os.path.join(BASE, "final_multigenerator_model.joblib")
DEPLOY_TFIDF = os.path.join(BASE, "final_multigenerator_tfidf.joblib")
BASELINE_MODEL = os.path.join(BASE, "best_fake_review_model.joblib")
BASELINE_TFIDF = os.path.join(BASE, "tfidf_vectorizer.joblib")

TRAIN_CSV = os.path.join(BASE, "modern_fake_reviews_train.csv")
VAL_CSV = os.path.join(BASE, "modern_fake_reviews_validation.csv")
TEST_CSV = os.path.join(BASE, "modern_fake_reviews_test.csv")

BASELINE_RESULTS = os.path.join(BASE, "baseline_model_results.csv")
CONTROLLED_RESULTS = os.path.join(BASE, "controlled_cross_generator_comparison.csv")
LOGO_RESULTS = os.path.join(BASE, "leave_one_generator_out_results.csv")
PERF_CATEGORY = os.path.join(BASE, "performance_by_category.csv")
PERF_RATING = os.path.join(BASE, "performance_by_rating.csv")
PERF_LENGTH = os.path.join(BASE, "performance_by_review_length.csv")
MISCLASSIFIED = os.path.join(BASE, "misclassified_reviews.csv")

st.set_page_config(
    page_title="ReviewGuard AI Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

# ============================================================
# LOADERS
# ============================================================

@st.cache_resource
def load_joblib(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def exists(path):
    return os.path.exists(path)

required = [
    DEPLOY_MODEL,
    DEPLOY_TFIDF,
    BASELINE_MODEL,
    BASELINE_TFIDF,
    TEST_CSV,
]

missing = [p for p in required if not exists(p)]
if missing:
    st.error(
        "ReviewGuard AI cannot start because these required files are missing:\n\n"
        + "\n".join(missing)
    )
    st.stop()

deploy_model = load_joblib(DEPLOY_MODEL)
deploy_tfidf = load_joblib(DEPLOY_TFIDF)
baseline_model = load_joblib(BASELINE_MODEL)
baseline_tfidf = load_joblib(BASELINE_TFIDF)
test_df_all = load_csv(TEST_CSV)

# ============================================================
# HELPERS
# ============================================================

def pct(v):
    return f"{float(v) * 100:.2f}%"

def fmt_int(v):
    return f"{int(v):,}"

def go_to(page_name):
    st.session_state.page = page_name

def confidence_band(v):
    if v >= 0.85:
        return "HIGH", "#00A979", "#E8F8F1"
    if v >= 0.65:
        return "MODERATE", "#D97706", "#FFF4DF"
    return "LOW", "#D92D4A", "#FDECEF"

# ============================================================
# BRAND THEME
# ============================================================

dark = st.session_state.dark_mode

C = {
    "bg": "#F3F6FA" if not dark else "#06111E",
    "surface": "#FFFFFF" if not dark else "#0C1E31",
    "surface2": "#F8FAFD" if not dark else "#10263C",
    "text": "#07182E" if not dark else "#F4F8FD",
    "muted": "#66758A" if not dark else "#A9B7C8",
    "border": "#DFE6EE" if not dark else "#1B3A57",
    "navy": "#04182E",
    "navy2": "#073154",
    "cyan": "#13C6F3",
    "blue": "#1478F2",
    "teal": "#00A979",
    "violet": "#7A50F9",
    "pink": "#E52B70",
    "orange": "#F47521",
    "red": "#D92D4A",
}

# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    f"""
<style>
#MainMenu, footer, header {{visibility:hidden;}}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display:none !important;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background:linear-gradient(180deg,{C["bg"]} 0%,{"#EEF3F8" if not dark else "#071522"} 100%);
    color:{C["text"]};
}}

.block-container {{
    max-width:1600px;
    padding:0.45rem 1.0rem 2rem 1.0rem;
}}

[data-testid="stSidebar"] {{
    min-width:258px !important;
    max-width:258px !important;
    background:
        radial-gradient(circle at 18% 4%, rgba(19,198,243,.14), transparent 24%),
        linear-gradient(180deg,#03172C 0%,#052641 54%,#073150 100%);
    border-right:1px solid rgba(255,255,255,.07);
}}

[data-testid="stSidebar"] > div:first-child {{
    width:258px !important;
}}

/* FORCE REVIEWGUARD SIDEBAR TO REMAIN VISIBLE ON DESKTOP */
section[data-testid="stSidebar"],
[data-testid="stSidebar"] {{
    display:block !important;
    visibility:visible !important;
    opacity:1 !important;
    transform:translateX(0px) !important;
    left:0 !important;
    min-width:258px !important;
    width:258px !important;
    max-width:258px !important;
    position:relative !important;
    flex-shrink:0 !important;
    z-index:999 !important;
    pointer-events:auto !important;
}}

section[data-testid="stSidebar"][aria-expanded="false"] {{
    transform:translateX(0px) !important;
    margin-left:0 !important;
    min-width:258px !important;
    width:258px !important;
    max-width:258px !important;
}}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {{
    display:block !important;
    visibility:visible !important;
    opacity:1 !important;
    width:258px !important;
}}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] {{
    display:none !important;
}}

[data-testid="stSidebar"] * {{
    color:#F4F8FD;
}}

[data-testid="stSidebar"] .stButton > button {{
    width:100%;
    min-height:42px;
    justify-content:flex-start;
    border:0;
    border-radius:11px;
    font-size:.84rem;
    font-weight:700;
    padding:.58rem .76rem;
    transition:all .16s ease;
}}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background:linear-gradient(135deg,#137CF6,#0B60D2);
    box-shadow:0 9px 24px rgba(16,112,238,.28);
    color:white;
}}

[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    background:transparent;
    color:#DDE9F6;
}}

[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
    background:rgba(255,255,255,.075);
    transform:translateX(3px);
}}

/* BRAND */
.brand-lockup {{
    display:flex;
    align-items:center;
    gap:.82rem;
    margin:.20rem 0 .25rem;
}}
.rg-shield {{
    width:62px;
    height:70px;
    display:grid;
    place-items:center;
    background:linear-gradient(145deg,#0C8DFF,#18D1F4);
    clip-path:polygon(50% 0,93% 14%,88% 67%,50% 100%,12% 67%,7% 14%);
    box-shadow:0 12px 30px rgba(18,174,242,.28);
}}
.rg-monogram {{
    color:white;
    font-size:1.10rem;
    font-weight:900;
    letter-spacing:-.06em;
}}
.brand-name {{
    font-size:1.52rem;
    font-weight:900;
    letter-spacing:-.035em;
    line-height:1;
}}
.brand-ai {{
    color:#19D5F5;
    font-size:.94rem;
    font-weight:850;
    letter-spacing:.04em;
    margin-top:.26rem;
}}
.brand-tagline {{
    color:#B9CADE !important;
    font-size:.75rem;
    line-height:1.48;
    margin:.55rem 0 .75rem;
}}
.live-pill {{
    display:inline-flex;
    align-items:center;
    gap:.42rem;
    padding:.30rem .56rem;
    border-radius:999px;
    color:#B9F5DF !important;
    background:rgba(0,169,121,.13);
    border:1px solid rgba(0,169,121,.20);
    font-size:.66rem;
    font-weight:800;
    letter-spacing:.05em;
}}
.live-dot {{
    width:7px;height:7px;border-radius:50%;
    background:#18D99B;
    box-shadow:0 0 0 4px rgba(24,217,155,.09);
}}

/* TOP HEADER */
.topbar {{
    position:relative;
    overflow:hidden;
    min-height:112px;
    border-radius:0 0 18px 18px;
    padding:1.20rem 1.55rem;
    margin-bottom:.90rem;
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:
        radial-gradient(circle at 92% 22%,rgba(19,198,243,.12),transparent 22%),
        radial-gradient(circle at 76% 120%,rgba(20,120,242,.18),transparent 30%),
        linear-gradient(100deg,#03162B 0%,#052642 63%,#073B62 100%);
    box-shadow:0 14px 32px rgba(4,24,46,.16);
}}
.topbar:after {{
    content:"";
    position:absolute;
    width:260px;height:260px;border-radius:50%;
    right:-110px;top:-170px;
    border:1px solid rgba(255,255,255,.05);
}}
.eyebrow {{
    font-size:.66rem;
    color:#28D5F5;
    font-weight:850;
    letter-spacing:.14em;
    margin-bottom:.28rem;
}}
.page-title {{
    color:white;
    font-size:1.73rem;
    font-weight:900;
    letter-spacing:-.025em;
    line-height:1.15;
}}
.page-sub {{
    color:#C7D6E7;
    font-size:.84rem;
    margin-top:.40rem;
}}
.header-pills {{
    display:flex;
    gap:.45rem;
    margin-top:.62rem;
}}
.header-pill {{
    border-radius:999px;
    padding:.28rem .55rem;
    font-size:.64rem;
    font-weight:750;
    color:#D6E8F7;
    border:1px solid rgba(255,255,255,.12);
    background:rgba(255,255,255,.055);
}}
.header-right {{
    display:flex;
    gap:.85rem;
    align-items:center;
    position:relative;
    z-index:2;
}}
.header-util {{
    min-width:132px;
    display:flex;
    align-items:center;
    gap:.55rem;
}}
.util-bubble {{
    width:42px;height:42px;border-radius:50%;
    display:grid;place-items:center;
    background:rgba(255,255,255,.065);
    color:#EAF5FF;
    font-size:1.05rem;
}}
.util-k {{
    color:white;font-size:.72rem;font-weight:800;
}}
.util-v {{
    color:#C9D6E5;font-size:.69rem;margin-top:.10rem;
}}
.util-divider {{
    width:1px;height:45px;background:rgba(255,255,255,.18);
}}

/* KPI */
.kpi-grid {{
    display:grid;
    grid-template-columns:repeat(6,minmax(0,1fr));
    gap:.82rem;
    margin:.30rem 0 .92rem;
}}
.kpi {{
    position:relative;
    overflow:hidden;
    min-height:112px;
    padding:.92rem .92rem;
    border-radius:14px;
    background:{C["surface"]};
    border:1px solid {C["border"]};
    border-top:3px solid var(--accent);
    box-shadow:0 7px 18px rgba(18,35,58,.065);
    display:flex;
    align-items:center;
    gap:.72rem;
    transition:.16s ease;
}}
.kpi:hover {{
    transform:translateY(-3px);
    box-shadow:0 13px 28px rgba(18,35,58,.105);
}}
.kpi:after {{
    content:"";
    position:absolute;
    width:70px;height:70px;border-radius:50%;
    right:-27px;bottom:-31px;
    background:var(--soft);
}}
.kpi-icon {{
    width:46px;height:46px;min-width:46px;
    display:grid;place-items:center;border-radius:50%;
    color:var(--accent);background:var(--soft);
    font-size:1.03rem;font-weight:900;
}}
.kpi-value {{
    font-size:1.48rem;
    line-height:1.04;
    color:var(--accent);
    font-weight:900;
    letter-spacing:-.035em;
}}
.kpi-label {{
    margin-top:.28rem;
    color:{C["muted"]};
    font-size:.72rem;
    font-weight:700;
    line-height:1.25;
}}

/* STATUS RIBBON */
.status-grid {{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:.62rem;
    margin:.10rem 0 .80rem;
}}
.status-item {{
    background:{C["surface"]};
    border:1px solid {C["border"]};
    border-radius:12px;
    padding:.66rem .72rem;
    display:flex;
    align-items:center;
    gap:.58rem;
    box-shadow:0 5px 14px rgba(18,35,58,.04);
}}
.status-icon {{
    width:34px;height:34px;border-radius:10px;
    display:grid;place-items:center;
    background:var(--soft);color:var(--accent);
    font-size:.90rem;
}}
.status-title {{
    color:{C["text"]};font-size:.70rem;font-weight:850;
}}
.status-sub {{
    color:{C["muted"]};font-size:.60rem;margin-top:.10rem;
}}

/* PANELS */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color:{C["border"]} !important;
    background:{C["surface"]} !important;
    border-radius:14px !important;
    box-shadow:0 7px 18px rgba(18,35,58,.05);
}}
.panel-head {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:.60rem;
}}
.panel-title {{
    color:{C["text"]};
    font-size:1.00rem;
    font-weight:900;
    letter-spacing:-.015em;
}}
.info-dot {{
    width:22px;height:22px;border-radius:50%;
    display:grid;place-items:center;
    border:1px solid {C["border"]};
    color:{C["muted"]};
    font-size:.68rem;
}}
.mini-card {{
    min-height:108px;
    padding:.74rem;
    border-radius:11px;
    border:1px solid {C["border"]};
    border-left:4px solid var(--accent);
    background:{C["surface2"]};
    display:flex;gap:.64rem;align-items:flex-start;
}}
.mini-icon {{
    width:42px;height:42px;min-width:42px;
    border-radius:50%;
    display:grid;place-items:center;
    color:var(--accent);
    background:var(--soft);
    font-size:1.15rem;font-weight:900;
}}
.mini-value {{
    color:var(--accent);
    font-size:1.38rem;font-weight:900;line-height:1;
}}
.mini-label {{
    color:{C["text"]};font-size:.76rem;font-weight:850;margin-top:.28rem;
}}
.mini-sub {{
    color:{C["muted"]};font-size:.64rem;line-height:1.32;margin-top:.14rem;
}}

/* QUICK ACTIONS */
.action-head {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin:.12rem 0 .42rem;
}}
.action-title {{
    color:{C["text"]};font-size:.92rem;font-weight:900;
}}
.action-sub {{
    color:{C["muted"]};font-size:.66rem;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {{
    min-height:48px;
    border-radius:11px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
    color:{C["text"]};
    box-shadow:0 5px 14px rgba(18,35,58,.045);
    transition:.15s ease;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {{
    transform:translateY(-2px);
    border-color:#88BDF9;
    box-shadow:0 10px 22px rgba(20,120,242,.11);
}}

/* WORKFLOW */
.flow-shell {{
    border-radius:14px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
    padding:.86rem .95rem;
    box-shadow:0 6px 16px rgba(18,35,58,.045);
    margin:.75rem 0;
}}
.flow-grid {{
    display:grid;
    grid-template-columns:1fr 26px 1fr 26px 1fr 26px 1fr 26px 1fr;
    align-items:center;
    gap:.22rem;
}}
.flow-node {{
    min-height:84px;
    padding:.52rem;
    border-radius:12px;
    border:1px solid {C["border"]};
    background:{C["surface2"]};
    text-align:center;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    transition:.16s ease;
}}
.flow-node:hover {{
    transform:translateY(-3px);
    border-color:#8CC0FC;
    box-shadow:0 8px 18px rgba(20,120,242,.09);
}}
.flow-symbol {{
    width:35px;height:35px;border-radius:10px;
    display:grid;place-items:center;
    color:#156FE2;background:#EAF3FF;
    font-size:.95rem;font-weight:900;
    margin-bottom:.30rem;
}}
.flow-title {{
    color:{C["text"]};font-size:.70rem;font-weight:850;
}}
.flow-sub {{
    color:{C["muted"]};font-size:.59rem;margin-top:.10rem;
}}
.flow-arrow {{
    color:#9BB0C6;text-align:center;font-size:1.0rem;
    animation:pulse 1.7s ease-in-out infinite;
}}
@keyframes pulse {{
    0%,100% {{opacity:.35;transform:translateX(-1px);}}
    50% {{opacity:1;transform:translateX(2px);}}
}}

/* CONFIG */
.config-grid {{
    display:grid;grid-template-columns:repeat(4,1fr);gap:.58rem;
}}
.config {{
    min-height:121px;
    padding:.72rem .48rem;
    border-radius:11px;
    border:1px solid {C["border"]};
    background:{C["surface2"]};
    text-align:center;
}}
.config-symbol {{
    width:40px;height:40px;margin:0 auto .38rem;
    border-radius:12px;display:grid;place-items:center;
    background:#EAF3FF;color:#1478F2;font-weight:900;
}}
.config-k {{
    color:{C["muted"]};font-size:.63rem;font-weight:750;
}}
.config-v {{
    color:{C["text"]};font-size:.76rem;font-weight:850;line-height:1.30;margin-top:.32rem;
}}

/* INSIGHT */
.insight {{
    min-height:214px;
    padding:.98rem 1.0rem;
    border-radius:14px;
    color:white;
    background:
        radial-gradient(circle at 91% 72%,rgba(19,198,243,.17),transparent 31%),
        linear-gradient(117deg,#031A34,#07385F);
    box-shadow:0 10px 25px rgba(4,25,51,.18);
}}
.insight-label {{
    color:#29D3F3;font-size:.68rem;font-weight:850;letter-spacing:.10em;
}}
.insight-title {{
    font-size:1.14rem;font-weight:900;margin:.28rem 0 .42rem;
}}
.insight p {{
    color:#DCE8F5;font-size:.78rem;line-height:1.48;
}}

/* SUMMARY */
.summary {{
    display:grid;grid-template-columns:repeat(5,1fr);gap:.65rem;
    margin-top:.78rem;
    padding:.80rem .92rem;
    border-radius:13px;
    background:linear-gradient(100deg,#03172E,#073253);
    box-shadow:0 9px 23px rgba(3,23,46,.17);
}}
.summary-item {{
    padding-right:.65rem;border-right:1px solid rgba(255,255,255,.16);
}}
.summary-item:last-child {{border-right:0;}}
.summary-k {{color:#AABDD2;font-size:.63rem;}}
.summary-v {{color:#42ACFF;font-size:1.02rem;font-weight:900;margin-top:.15rem;}}

/* REVIEW ANALYSIS */
.result-card {{
    padding:1rem;
    border-radius:15px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
    box-shadow:0 8px 20px rgba(18,35,58,.06);
}}
.result-banner {{
    border-radius:12px;
    padding:.78rem .88rem;
    border-left:5px solid var(--accent);
    background:var(--soft);
}}
.result-main {{color:var(--accent);font-size:1.03rem;font-weight:900;}}
.result-sub {{color:{C["muted"]};font-size:.67rem;margin-top:.14rem;}}
.gauge {{
    --value:50;
    --gcolor:#1478F2;
    width:150px;height:150px;border-radius:50%;
    margin:.75rem auto;
    display:grid;place-items:center;
    position:relative;
    background:conic-gradient(var(--gcolor) calc(var(--value)*1%), #E7EDF4 0);
}}
.gauge:before {{
    content:"";position:absolute;width:112px;height:112px;border-radius:50%;
    background:{C["surface"]};
}}
.gauge-inner {{position:relative;z-index:2;text-align:center;}}
.gauge-num {{color:{C["text"]};font-size:1.50rem;font-weight:900;}}
.gauge-cap {{color:{C["muted"]};font-size:.62rem;font-weight:750;}}
.prob-head {{
    display:flex;justify-content:space-between;
    font-size:.72rem;font-weight:750;color:{C["text"]};
    margin-bottom:.24rem;
}}
.prob-track {{
    height:9px;border-radius:999px;background:#E7EDF4;overflow:hidden;margin-bottom:.62rem;
}}
.prob-fill {{height:100%;border-radius:999px;}}

/* POPOVERS */
div[data-testid="stPopover"] > button {{
    border-radius:999px !important;
    min-height:35px !important;
    border:1px solid {C["border"]} !important;
    background:{C["surface"]} !important;
    color:{C["text"]} !important;
    font-size:.74rem !important;
    font-weight:750 !important;
}}

/* METRIC */
div[data-testid="stMetric"] {{
    border:1px solid {C["border"]};
    background:{C["surface"]};
    border-radius:12px;
    padding:.68rem .80rem;
    box-shadow:0 5px 14px rgba(18,35,58,.04);
}}


/* PREMIUM MODEL PERFORMANCE */
.performance-hero {{
    display:grid;
    grid-template-columns:1.15fr .85fr;
    gap:.75rem;
    margin:.20rem 0 .82rem;
}}
.best-model-card {{
    min-height:142px;
    padding:1.05rem 1.15rem;
    border-radius:15px;
    background:
      radial-gradient(circle at 92% 18%, rgba(19,198,243,.16), transparent 28%),
      linear-gradient(125deg,#041A33,#083A61);
    color:white;
    box-shadow:0 12px 30px rgba(4,26,51,.16);
}}
.best-model-kicker {{
    color:#34D8F4;
    font-size:.66rem;
    font-weight:900;
    letter-spacing:.12em;
}}
.best-model-title {{
    font-size:1.62rem;
    font-weight:900;
    margin:.32rem 0 .18rem;
}}
.best-model-sub {{
    color:#CBD9E8;
    font-size:.76rem;
}}
.best-badges {{
    display:flex;
    gap:.42rem;
    flex-wrap:wrap;
    margin-top:.78rem;
}}
.best-badge {{
    border-radius:999px;
    padding:.28rem .50rem;
    font-size:.62rem;
    font-weight:800;
    color:#DDEBFA;
    border:1px solid rgba(255,255,255,.13);
    background:rgba(255,255,255,.055);
}}
.rank-card {{
    min-height:142px;
    padding:.95rem 1rem;
    border-radius:15px;
    background:{C["surface"]};
    border:1px solid {C["border"]};
    box-shadow:0 8px 20px rgba(18,35,58,.055);
}}
.rank-title {{
    color:{C["text"]};
    font-size:.84rem;
    font-weight:900;
    margin-bottom:.55rem;
}}
.rank-row {{
    display:grid;
    grid-template-columns:24px 1fr auto;
    align-items:center;
    gap:.50rem;
    padding:.35rem 0;
    border-bottom:1px dashed {C["border"]};
}}
.rank-row:last-child {{border-bottom:0;}}
.rank-num {{
    width:22px;height:22px;border-radius:7px;
    display:grid;place-items:center;
    font-size:.65rem;font-weight:900;
    background:#EAF3FF;color:#1478F2;
}}
.rank-name {{
    color:{C["text"]};
    font-size:.70rem;
    font-weight:760;
}}
.rank-score {{
    color:#1478F2;
    font-size:.72rem;
    font-weight:900;
}}

.metric-panel-grid {{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:.78rem;
    margin:.30rem 0 .85rem;
}}
.metric-panel {{
    padding:.92rem 1rem;
    border-radius:14px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
    box-shadow:0 7px 18px rgba(18,35,58,.045);
}}
.metric-panel-title {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:.60rem;
}}
.metric-panel-title span:first-child {{
    color:{C["text"]};
    font-size:.92rem;
    font-weight:900;
}}
.metric-chip {{
    font-size:.60rem;
    font-weight:850;
    padding:.20rem .42rem;
    border-radius:999px;
    color:var(--accent);
    background:var(--soft);
}}
.model-bar-row {{
    display:grid;
    grid-template-columns:125px 1fr 58px;
    gap:.55rem;
    align-items:center;
    margin:.47rem 0;
}}
.model-bar-name {{
    color:{C["muted"]};
    font-size:.67rem;
    font-weight:720;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}}
.model-bar-track {{
    height:11px;
    border-radius:999px;
    background:#E7EDF4;
    overflow:hidden;
}}
.model-bar-fill {{
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
}}
.model-bar-score {{
    color:{C["text"]};
    text-align:right;
    font-size:.68rem;
    font-weight:900;
}}
.efficiency-panel {{
    padding:.95rem 1rem;
    border-radius:14px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
    box-shadow:0 7px 18px rgba(18,35,58,.045);
    margin-bottom:.82rem;
}}
.time-row {{
    display:grid;
    grid-template-columns:145px 1fr 70px;
    gap:.55rem;
    align-items:center;
    margin:.50rem 0;
}}
.time-name {{
    color:{C["muted"]};
    font-size:.68rem;
    font-weight:730;
}}
.time-track {{
    height:13px;
    border-radius:999px;
    background:#E7EDF4;
    overflow:hidden;
}}
.time-fill {{
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#F47521,#F8B14A);
}}
.time-value {{
    color:{C["text"]};
    text-align:right;
    font-size:.68rem;
    font-weight:900;
}}
.decision-strip {{
    display:grid;
    grid-template-columns:1.15fr .85fr;
    gap:.72rem;
    margin-bottom:.7rem;
}}
.decision-card {{
    padding:.92rem 1rem;
    border-radius:14px;
    border:1px solid {C["border"]};
    background:{C["surface"]};
}}
.decision-card strong {{
    color:{C["text"]};
}}
.decision-card p {{
    color:{C["muted"]};
    font-size:.74rem;
    line-height:1.5;
    margin:.28rem 0 0;
}}
.table-label {{
    color:{C["text"]};
    font-size:.90rem;
    font-weight:900;
    margin:.25rem 0 .45rem;
}}


/* LEAVE-ONE-GENERATOR-OUT PROFESSIONAL CARDS */
.logo-grid-card {{
    border:1px solid {C["border"]};
    background:{C["surface"]};
    border-radius:14px;
    padding:.92rem .95rem;
    box-shadow:0 7px 18px rgba(18,35,58,.045);
    min-height:225px;
}}
.logo-card-title {{
    color:{C["muted"]};
    font-size:.74rem;
    font-weight:760;
    line-height:1.35;
    margin-bottom:.72rem;
}}
.logo-metrics {{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:.65rem;
}}
.logo-metric {{
    border-radius:12px;
    border:1px solid {C["border"]};
    background:{C["surface2"]};
    padding:.72rem .72rem;
    min-height:104px;
    display:flex;
    flex-direction:column;
    justify-content:center;
}}
.logo-metric-k {{
    color:{C["muted"]};
    font-size:.66rem;
    font-weight:760;
    margin-bottom:.22rem;
}}
.logo-metric-v {{
    color:{C["text"]};
    font-size:1.65rem;
    font-weight:900;
    letter-spacing:-.035em;
    line-height:1.05;
    white-space:nowrap;
}}
.logo-metric-v.accuracy {{ color:#1478F2; }}
.logo-metric-v.f1 {{ color:#00A979; }}
.logo-foot {{
    margin-top:.72rem;
    padding-top:.62rem;
    border-top:1px dashed {C["border"]};
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:.5rem;
}}
.logo-foot-label {{
    color:{C["muted"]};
    font-size:.67rem;
    font-weight:720;
}}
.logo-foot-value {{
    color:#D92D4A;
    font-size:.78rem;
    font-weight:900;
}}

@media(max-width:1260px) {{
    .kpi-grid {{grid-template-columns:repeat(3,1fr);}}
    .header-right {{display:none;}}
    .status-grid {{grid-template-columns:repeat(2,1fr);}}
}}
@media(max-width:800px) {{
    .kpi-grid {{grid-template-columns:repeat(2,1fr);}}
    .flow-grid {{grid-template-columns:1fr;}}
    .flow-arrow {{transform:rotate(90deg);}}
    .summary {{grid-template-columns:1fr 1fr;}}
}}
</style>
"""
)

# ============================================================
# COMPONENTS
# ============================================================

def brand():
    st.html(
        """
<div class="brand-lockup">
    <div class="rg-shield">
        <svg viewBox="0 0 64 72" width="46" height="52" fill="none">
            <path d="M32 3L57 12V30C57 48 47 60 32 68C17 60 7 48 7 30V12L32 3Z"
                  stroke="white" stroke-width="3.2" fill="rgba(255,255,255,.08)"/>
            <path d="M21 35L28 42L43 25" stroke="white" stroke-width="4"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M18 18H24M40 18H46M15 25H21M43 25H49"
                  stroke="#BDF6FF" stroke-width="2" stroke-linecap="round"/>
        </svg>
    </div>
    <div>
        <div class="brand-name">ReviewGuard</div>
        <div class="brand-ai">AI ANALYTICS</div>
    </div>
</div>
<div class="brand-tagline">
    Explainable authenticity intelligence for e-commerce review screening.
</div>
<div class="live-pill"><span class="live-dot"></span> LIVE RESEARCH BUILD</div>
"""
    )

def topbar(title, subtitle, scope):
    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    st.html(
        f"""
<div class="topbar">
    <div>
        <div class="eyebrow">REVIEWGUARD AI • AUTHENTICITY INTELLIGENCE</div>
        <div class="page-title">{title}</div>
        <div class="page-sub">{subtitle}</div>
        <div class="header-pills">
            <span class="header-pill">LIVE MODEL</span>
            <span class="header-pill">LOCAL INFERENCE</span>
            <span class="header-pill">3 GENERATORS</span>
        </div>
    </div>
    <div class="header-right">
        <div class="header-util">
            <div class="util-bubble">◷</div>
            <div><div class="util-k">Last refreshed</div><div class="util-v">Today at {now}</div></div>
        </div>
        <div class="util-divider"></div>
        <div class="header-util">
            <div class="util-bubble">◇</div>
            <div><div class="util-k">Scope</div><div class="util-v">{scope}</div></div>
        </div>
    </div>
</div>
"""
    )

def kpis(items):
    palette = [
        ("#1478F2", "#EAF3FF"),
        ("#00A979", "#E8F8F1"),
        ("#7A50F9", "#F1EBFF"),
        ("#E52B70", "#FDEAF2"),
        ("#F47521", "#FFF0E7"),
        ("#0F6FD9", "#EAF3FD"),
    ]
    symbols = ["◎", "⊕", "↻", "↗", "◇", "▤"]
    html = ['<div class="kpi-grid">']
    for i, (value, label) in enumerate(items):
        accent, soft = palette[i % len(palette)]
        sym = symbols[i % len(symbols)]
        html.append(
            f'<div class="kpi" style="--accent:{accent};--soft:{soft};">'
            f'<div class="kpi-icon">{sym}</div>'
            f'<div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div></div>'
        )
    html.append("</div>")
    st.html("".join(html))

def status_ribbon():
    st.html(
        """
<div class="status-grid">
    <div class="status-item" style="--accent:#00A979;--soft:#E8F8F1;">
        <div class="status-icon">●</div><div><div class="status-title">Model Service</div><div class="status-sub">Online & ready</div></div>
    </div>
    <div class="status-item" style="--accent:#1478F2;--soft:#EAF3FF;">
        <div class="status-icon">50K</div><div><div class="status-title">Vectoriser</div><div class="status-sub">TF-IDF feature space loaded</div></div>
    </div>
    <div class="status-item" style="--accent:#7A50F9;--soft:#F1EBFF;">
        <div class="status-icon">3×</div><div><div class="status-title">Robustness Suite</div><div class="status-sub">DeepSeek • LFM • GLM</div></div>
    </div>
    <div class="status-item" style="--accent:#F47521;--soft:#FFF0E7;">
        <div class="status-icon">↯</div><div><div class="status-title">Inference Mode</div><div class="status-sub">Local deployment</div></div>
    </div>
</div>
"""
    )

def panel_head(title):
    st.html(
        f'<div class="panel-head"><div class="panel-title">{title}</div><div class="info-dot">i</div></div>'
    )

def mini(value, label, sub, accent, soft, symbol):
    st.html(
        f"""
<div class="mini-card" style="--accent:{accent};--soft:{soft};">
    <div class="mini-icon">{symbol}</div>
    <div>
        <div class="mini-value">{value}</div>
        <div class="mini-label">{label}</div>
        <div class="mini-sub">{sub}</div>
    </div>
</div>
"""
    )

def quick_actions():
    st.html(
        """
<div class="action-head">
    <div><div class="action-title">Operational Shortcuts</div>
    <div class="action-sub">Launch the live tools directly.</div></div>
</div>
"""
    )
    a, b, c, d = st.columns(4)
    with a:
        st.button("🔎  Analyse Review", key="qa1", use_container_width=True, on_click=go_to, args=("Review Analysis",))
    with b:
        st.button("🧪  Generator Lab", key="qa2", use_container_width=True, on_click=go_to, args=("Generator Analysis",))
    with c:
        st.button("🧠  Explain Model", key="qa3", use_container_width=True, on_click=go_to, args=("Explainability",))
    with d:
        st.button("📦  Batch Screen", key="qa4", use_container_width=True, on_click=go_to, args=("Batch Screening",))

def info_popovers():
    a, b, c, d = st.columns(4)
    with a:
        with st.popover("🧠 Model Card", use_container_width=True):
            st.markdown("**Classifier:** Multinomial Naive Bayes")
            st.markdown("**Representation:** TF-IDF")
            st.markdown("**Feature space:** 50,000")
    with b:
        with st.popover("🧪 Evaluation Design", use_container_width=True):
            st.markdown("Validation data was used for model selection.")
            st.markdown("The test split remained unseen until final baseline evaluation.")
    with c:
        with st.popover("🛡️ Responsible AI", use_container_width=True):
            st.markdown("A CG prediction is a screening output, not proof of fraud.")
    with d:
        with st.popover("📊 Metrics Guide", use_container_width=True):
            st.markdown("**Precision:** correctness of CG flags.")
            st.markdown("**Recall:** proportion of CG detected.")
            st.markdown("**F1:** balance between precision and recall.")

def workflow():
    st.html(
        """
<div class="flow-shell">
    <div class="panel-title" style="margin-bottom:.62rem;">Live Detection Architecture</div>
    <div class="flow-grid">
        <div class="flow-node"><div class="flow-symbol">IN</div><div class="flow-title">Review Input</div><div class="flow-sub">Text or CSV</div></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><div class="flow-symbol">TX</div><div class="flow-title">Text Pipeline</div><div class="flow-sub">Consistent handling</div></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><div class="flow-symbol">TF</div><div class="flow-title">TF-IDF</div><div class="flow-sub">50K features</div></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><div class="flow-symbol">NB</div><div class="flow-title">Naive Bayes</div><div class="flow-sub">Classifier</div></div>
        <div class="flow-arrow">→</div>
        <div class="flow-node"><div class="flow-symbol">AI</div><div class="flow-title">OR / CG</div><div class="flow-sub">Class + probability</div></div>
    </div>
</div>
"""
    )

def chart_style(chart):
    return (
        chart
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=C["muted"],
            titleColor=C["muted"],
            gridColor=C["border"],
            domainColor=C["border"],
            tickColor=C["border"],
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=C["muted"],
            titleColor=C["text"],
        )
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    brand()

    nav = [
        ("⌂  Overview", "Overview"),
        ("◎  Review Analysis", "Review Analysis"),
        ("✣  Generator Analysis", "Generator Analysis"),
        ("⌘  Explainability", "Explainability"),
        ("⌁  Model Performance", "Model Performance"),
        ("▱  Data Quality", "Data Quality"),
        ("▣  Batch Screening", "Batch Screening"),
        ("▤  Reports", "Reports"),
        ("⚙  Settings", "Settings"),
    ]

    for label, value in nav:
        st.button(
            label,
            key=f"nav_{value}",
            type="primary" if st.session_state.page == value else "secondary",
            use_container_width=True,
            on_click=go_to,
            args=(value,),
        )

    st.markdown("---")
    new_dark = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if new_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown("---")
    st.caption("RESEARCH BUILD 1.0")
    st.caption("TF-IDF • Multinomial Naive Bayes")
    st.caption("DeepSeek • LFM • GLM")

page = st.session_state.page

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    categories = ["All categories"] + sorted(
        test_df_all["category"].astype(str).unique().tolist()
    )
    selected = st.sidebar.selectbox("Analysis scope", categories, index=0)

    test_df = (
        test_df_all.copy()
        if selected == "All categories"
        else test_df_all[test_df_all["category"].astype(str) == selected].copy()
    )

    topbar(
        "Explainable Fake Review Detection Dashboard",
        "Final Multinomial Naive Bayes Model • TF-IDF Pipeline • Held-out Test Evaluation",
        selected,
    )

    X = baseline_tfidf.transform(test_df["text_"].fillna("").astype(str))
    y = test_df["label"]
    pred = baseline_model.predict(X)

    acc = accuracy_score(y, pred)
    pre = precision_score(y, pred, pos_label="CG", zero_division=0)
    rec = recall_score(y, pred, pos_label="CG", zero_division=0)
    f1 = f1_score(y, pred, pos_label="CG", zero_division=0)
    cm = confusion_matrix(y, pred, labels=["CG", "OR"])

    controlled = load_csv(CONTROLLED_RESULTS) if exists(CONTROLLED_RESULTS) else None
    robust_avg = float(controlled["F1_CG"].mean()) if controlled is not None else np.nan

    kpis([
        (pct(acc), "Accuracy"),
        (pct(pre), "Precision (CG)"),
        (pct(rec), "Recall (CG)"),
        (pct(f1), "F1-Score (CG)"),
        (pct(robust_avg) if not np.isnan(robust_avg) else "—", "Cross-Generator Avg F1"),
        (fmt_int(len(test_df)), "Test Reviews"),
    ])

    status_ribbon()
    quick_actions()
    info_popovers()

    st.markdown("")
    left, right = st.columns([1, 1.58], gap="medium")

    with left:
        with st.container(border=True):
            panel_head("Final Test Confusion Matrix")
            a, b = st.columns(2)
            with a:
                mini(fmt_int(cm[0,0]), "Correct CG", "AI-generated reviews correctly detected", "#00A979", "#E8F8F1", "✓")
            with b:
                mini(fmt_int(cm[0,1]), "Missed CG", "AI-generated reviews classified as OR", "#D92D4A", "#FDECEF", "×")
            c, d = st.columns(2)
            with c:
                mini(fmt_int(cm[1,0]), "False Alarms", "Human reviews incorrectly flagged as CG", "#F47521", "#FFF0E7", "!")
            with d:
                mini(fmt_int(cm[1,1]), "Correct OR", "Original reviews correctly identified", "#1478F2", "#EAF3FF", "●")

    with right:
        with st.container(border=True):
            panel_head("Classification Outcomes")
            outcomes = pd.DataFrame(
                {
                    "Outcome": ["Correct OR", "Correct CG", "Missed CG", "False Alarms"],
                    "Count": [cm[1,1], cm[0,0], cm[0,1], cm[1,0]],
                    "Color": ["#1478F2", "#00A979", "#D92D4A", "#F47521"],
                }
            )

            bars = (
                alt.Chart(outcomes)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("Count:Q", title="Number of Reviews"),
                    y=alt.Y(
                        "Outcome:N",
                        title=None,
                        sort=["Correct OR", "Correct CG", "Missed CG", "False Alarms"],
                    ),
                    color=alt.Color("Color:N", scale=None, legend=None),
                    tooltip=["Outcome", "Count"],
                )
            )
            labels = bars.mark_text(
                align="left",
                baseline="middle",
                dx=6,
                color=C["text"],
                fontWeight=700,
            ).encode(text="Count:Q")

            st.altair_chart(
                chart_style((bars + labels).properties(height=286)),
                use_container_width=True,
            )

    st.markdown("")
    l1, l2, l3 = st.columns([1.05, .88, 1.05], gap="medium")

    with l1:
        with st.container(border=True):
            panel_head("Model Configuration")
            st.html(
                """
<div class="config-grid">
    <div class="config"><div class="config-symbol">NB</div><div class="config-k">Classifier</div><div class="config-v">Multinomial<br>Naive Bayes</div></div>
    <div class="config"><div class="config-symbol">TF</div><div class="config-k">Representation</div><div class="config-v">TF-IDF</div></div>
    <div class="config"><div class="config-symbol">50K</div><div class="config-k">Feature Space</div><div class="config-v">50,000</div></div>
    <div class="config"><div class="config-symbol">3×</div><div class="config-k">CG Sources</div><div class="config-v">DeepSeek<br>LFM • GLM</div></div>
</div>
"""
            )

    with l2:
        with st.container(border=True):
            panel_head("Cross-Generator Snapshot")
            if controlled is not None:
                snap = controlled[["Generator", "F1_CG"]].copy()
                snap["F1"] = snap["F1_CG"] * 100
                chart = (
                    alt.Chart(snap)
                    .mark_bar(cornerRadiusEnd=5)
                    .encode(
                        x=alt.X("F1:Q", title="F1 (%)", scale=alt.Scale(domain=[0,100])),
                        y=alt.Y("Generator:N", title=None, sort=None),
                        color=alt.Color(
                            "Generator:N",
                            scale=alt.Scale(
                                domain=["DeepSeek","GLM","LFM"],
                                range=["#1478F2","#00A979","#F47521"],
                            ),
                            legend=None,
                        ),
                        tooltip=["Generator", alt.Tooltip("F1:Q", format=".2f")],
                    )
                    .properties(height=184)
                )
                st.altair_chart(chart_style(chart), use_container_width=True)

    with l3:
        st.html(
            """
<div class="insight">
    <div class="insight-label">KEY RESEARCH INSIGHT</div>
    <div class="insight-title">Generalisation varies by AI generator.</div>
    <p>The detector is strongest on DeepSeek-style content and remains comparatively robust on GLM.</p>
    <p><b>Primary weakness:</b> LFM produces the largest loss in CG recall, demonstrating generator-specific variation.</p>
    <p><b>Deployment implication:</b> multi-generator training improves resilience, but adaptation is still required.</p>
</div>
"""
        )

    workflow()

    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    st.html(
        f"""
<div class="summary">
    <div class="summary-item"><div class="summary-k">Accuracy</div><div class="summary-v">{pct(acc)}</div></div>
    <div class="summary-item"><div class="summary-k">F1-Score</div><div class="summary-v">{pct(f1)}</div></div>
    <div class="summary-item"><div class="summary-k">Test Reviews</div><div class="summary-v">{fmt_int(len(test_df))}</div></div>
    <div class="summary-item"><div class="summary-k">Cross-Generator Avg F1</div><div class="summary-v">{pct(robust_avg) if not np.isnan(robust_avg) else "—"}</div></div>
    <div class="summary-item"><div class="summary-k">Last Refreshed</div><div class="summary-v">{now}</div></div>
</div>
"""
    )

# ============================================================
# REVIEW ANALYSIS
# ============================================================

elif page == "Review Analysis":
    topbar(
        "Live Review Analysis",
        "Interactive product-review screening • Final multi-generator deployment pipeline",
        "Manual input",
    )

    kpis([
        ("34,706", "Deployment Training Reviews"),
        ("3", "AI Generator Sources"),
        ("50,000", "TF-IDF Features"),
        ("Naive Bayes", "Classifier"),
        ("OR / CG", "Output Classes"),
        ("LOCAL", "Inference Mode"),
    ])

    info_popovers()

    left, right = st.columns([1.08, 1], gap="large")

    with left:
        with st.container(border=True):
            panel_head("Review Input")

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("👤 Human-style", key="sample_or", use_container_width=True):
                    st.session_state.review_input = (
                        "I bought this charger two weeks ago. It charges quickly, but the cable is shorter "
                        "than I expected. It has worked reliably so far and the plug does not get hot."
                    )
                    st.rerun()
            with b2:
                if st.button("🤖 AI-style", key="sample_cg", use_container_width=True):
                    st.session_state.review_input = (
                        "This exceptional product delivers outstanding quality, remarkable performance and "
                        "impressive reliability, making it an excellent choice for anyone seeking superior value."
                    )
                    st.rerun()
            with b3:
                if st.button("🧹 Clear", key="clear_review", use_container_width=True):
                    st.session_state.review_input = ""
                    st.rerun()

            review_text = st.text_area(
                "Product review text",
                height=255,
                key="review_input",
                placeholder="Paste or type the product review here...",
            )

            c1, c2 = st.columns(2)
            c1.caption(f"Words: {len(review_text.split()) if review_text.strip() else 0}")
            c2.caption(f"Characters: {len(review_text)}")

            run = st.button(
                "🛡️  Run ReviewGuard Analysis",
                type="primary",
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            panel_head("Prediction Intelligence")

            if not run:
                st.info("Enter a review and run ReviewGuard AI to see the classification profile.")
            else:
                cleaned = review_text.strip()
                if not cleaned:
                    st.warning("Please enter review text first.")
                else:
                    X = deploy_tfidf.transform([cleaned])
                    prediction = deploy_model.predict(X)[0]
                    raw_probs = deploy_model.predict_proba(X)[0]
                    probs = dict(zip(deploy_model.classes_, raw_probs))

                    cg = float(probs.get("CG", 0))
                    orp = float(probs.get("OR", 0))
                    conf = max(cg, orp)
                    band, accent, soft = confidence_band(conf)

                    label = (
                        "AI-generated / suspicious (CG)"
                        if prediction == "CG"
                        else "Original / human (OR)"
                    )

                    st.html(
                        f"""
<div class="result-card">
    <div class="result-banner" style="--accent:{accent};--soft:{soft};">
        <div class="result-main">{label}</div>
        <div class="result-sub">{band} CONFIDENCE • ReviewGuard AI output</div>
    </div>
    <div class="gauge" style="--value:{conf*100:.1f};--gcolor:{accent};">
        <div class="gauge-inner"><div class="gauge-num">{conf*100:.1f}%</div><div class="gauge-cap">CONFIDENCE</div></div>
    </div>
    <div class="prob-head"><span>AI-generated / suspicious (CG)</span><span>{cg*100:.2f}%</span></div>
    <div class="prob-track"><div class="prob-fill" style="width:{cg*100:.2f}%;background:#D92D4A;"></div></div>
    <div class="prob-head"><span>Original / human (OR)</span><span>{orp*100:.2f}%</span></div>
    <div class="prob-track"><div class="prob-fill" style="width:{orp*100:.2f}%;background:#00A979;"></div></div>
</div>
"""
                    )

    workflow()
    st.warning(
        "Responsible-use note: a CG prediction is a machine-learning screening result. "
        "It does not establish fraudulent intent, reviewer identity or legal deception."
    )

# ============================================================
# GENERATOR ANALYSIS
# ============================================================

elif page == "Generator Analysis":
    topbar(
        "Generator Robustness Lab",
        "Controlled cross-generator evaluation • Leave-one-generator-out testing",
        "DeepSeek • LFM • GLM",
    )

    if not exists(CONTROLLED_RESULTS) or not exists(LOGO_RESULTS):
        st.error("Generator-analysis files were not found.")
        st.stop()

    controlled = load_csv(CONTROLLED_RESULTS)
    logo = load_csv(LOGO_RESULTS)

    avg_f1 = float(logo["F1_CG"].mean())
    best = logo.loc[logo["F1_CG"].idxmax()]
    hardest = logo.loc[logo["F1_CG"].idxmin()]

    kpis([
        (pct(avg_f1), "Average Held-out F1"),
        (str(hardest["Held_Out_Generator"]), "Hardest Generator"),
        (pct(best["F1_CG"]), "Best Held-out F1"),
        ("2,858", "Common OR Test Reviews"),
        (pct(controlled["F1_CG"].mean()), "Controlled Avg F1"),
        ("PASS", "Leakage Control"),
    ])

    quick_actions()

    l, r = st.columns(2, gap="medium")

    with l:
        with st.container(border=True):
            panel_head("Held-out Generator F1")
            df = logo[["Held_Out_Generator","F1_CG"]].copy()
            df["F1"] = df["F1_CG"] * 100
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("F1:Q", title="F1-Score (%)", scale=alt.Scale(domain=[0,100])),
                    y=alt.Y("Held_Out_Generator:N", title=None, sort=None),
                    color=alt.Color(
                        "Held_Out_Generator:N",
                        scale=alt.Scale(
                            domain=["DeepSeek","GLM","LFM"],
                            range=["#1478F2","#00A979","#F47521"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Held_Out_Generator", alt.Tooltip("F1:Q", format=".2f")],
                )
                .properties(height=285)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    with r:
        with st.container(border=True):
            panel_head("CG Reviews Missed as OR")
            df = logo[["Held_Out_Generator","CG_Missed_as_OR"]].copy()
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusEnd=5, color="#E52B70")
                .encode(
                    x=alt.X("CG_Missed_as_OR:Q", title="Missed CG Reviews"),
                    y=alt.Y("Held_Out_Generator:N", title=None, sort=None),
                    tooltip=["Held_Out_Generator","CG_Missed_as_OR"],
                )
                .properties(height=285)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    st.subheader("Leave-One-Generator-Out Experiments")
    cards = st.columns(3)
    for i, (_, row) in enumerate(logo.iterrows()):
        with cards[i]:
            st.html(
                f"""
<div class="logo-grid-card">
    <div class="logo-card-title">
        {row['Training_Generators']} → Held-out <b>{row['Held_Out_Generator']}</b>
    </div>
    <div class="logo-metrics">
        <div class="logo-metric">
            <div class="logo-metric-k">Accuracy</div>
            <div class="logo-metric-v accuracy">{row['Accuracy']*100:.2f}%</div>
        </div>
        <div class="logo-metric">
            <div class="logo-metric-k">F1-Score (CG)</div>
            <div class="logo-metric-v f1">{row['F1_CG']*100:.2f}%</div>
        </div>
    </div>
    <div class="logo-foot">
        <span class="logo-foot-label">CG reviews missed as OR</span>
        <span class="logo-foot-value">{int(row['CG_Missed_as_OR']):,}</span>
    </div>
</div>
"""
            )

    st.info(
        "Key finding: LFM is the most difficult held-out generator, while DeepSeek is the strongest. "
        "The result supports generator-diverse training for more robust deployment."
    )

# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "Explainability":
    topbar(
        "Explainability & Error Intelligence",
        "Model-derived TF-IDF evidence • Post-test error analysis",
        "Baseline test set",
    )

    X = baseline_tfidf.transform(test_df_all["text_"].fillna("").astype(str))
    y = test_df_all["label"]
    pred = baseline_model.predict(X)
    cm = confusion_matrix(y, pred, labels=["CG","OR"])

    pre = precision_score(y, pred, pos_label="CG", zero_division=0)
    rec = recall_score(y, pred, pos_label="CG", zero_division=0)
    fnr = cm[0,1] / cm[0].sum()
    fpr = cm[1,0] / cm[1].sum()

    kpis([
        (pct(pre), "Precision (CG)"),
        (pct(rec), "Recall (CG)"),
        (pct(fnr), "False Negative Rate"),
        (pct(fpr), "False Positive Rate"),
        (fmt_int(cm[0,1] + cm[1,0]), "Total Errors"),
        ("0.61 s", "NB Training Time"),
    ])

    feature_names = np.array(baseline_tfidf.get_feature_names_out())
    classes = list(baseline_model.classes_)
    cg_i, or_i = classes.index("CG"), classes.index("OR")
    diff = baseline_model.feature_log_prob_[cg_i] - baseline_model.feature_log_prob_[or_i]

    top_cg = np.argsort(diff)[-12:][::-1]
    top_or = np.argsort(diff)[:12]

    cg_df = pd.DataFrame({"Feature": feature_names[top_cg], "Association": diff[top_cg]})
    or_df = pd.DataFrame({"Feature": feature_names[top_or], "Association": -diff[top_or]})

    l, r = st.columns(2, gap="medium")

    with l:
        with st.container(border=True):
            panel_head("Top CG-Indicative TF-IDF Signals")
            chart = (
                alt.Chart(cg_df)
                .mark_bar(cornerRadiusEnd=4, color="#1478F2")
                .encode(
                    x=alt.X("Association:Q", title="Relative NB association"),
                    y=alt.Y("Feature:N", title=None, sort="-x"),
                    tooltip=["Feature", alt.Tooltip("Association:Q", format=".3f")],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    with r:
        with st.container(border=True):
            panel_head("Top OR-Indicative TF-IDF Signals")
            chart = (
                alt.Chart(or_df)
                .mark_bar(cornerRadiusEnd=4, color="#00A979")
                .encode(
                    x=alt.X("Association:Q", title="Relative NB association"),
                    y=alt.Y("Feature:N", title=None, sort="-x"),
                    tooltip=["Feature", alt.Tooltip("Association:Q", format=".3f")],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    l2, r2 = st.columns([.8,1.2], gap="medium")
    with l2:
        with st.container(border=True):
            panel_head("Error Profile")
            a,b = st.columns(2)
            with a:
                mini(fmt_int(cm[1,0]), "False Alarms", "OR predicted as CG", "#F47521", "#FFF0E7", "!")
            with b:
                mini(fmt_int(cm[0,1]), "Missed CG", "CG predicted as OR", "#D92D4A", "#FDECEF", "×")

    with r2:
        with st.container(border=True):
            panel_head("Performance by Review Length")
            if exists(PERF_LENGTH):
                df = load_csv(PERF_LENGTH).copy()
                for col in ["Accuracy","F1_CG"]:
                    if col in df.columns:
                        df[col] = df[col].map(pct)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Review-length performance file was not found.")

    st.warning(
        "Interpretability note: these feature associations come directly from the fitted Multinomial Naive Bayes model. "
        "They are statistical associations, not causal explanations or proof of authorship."
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    topbar(
        "Model Performance Studio",
        "Four-model validation benchmark • Predictive quality • Computational efficiency",
        "Baseline model suite",
    )

    if not exists(BASELINE_RESULTS):
        st.error("baseline_model_results.csv was not found.")
        st.stop()

    results = load_csv(BASELINE_RESULTS).copy()
    best = results.loc[results["F1_CG"].idxmax()]

    kpis([
        (str(best["Model"]), "Best Validation Model"),
        (pct(best["Accuracy"]), "Best Accuracy"),
        (pct(best["Precision_CG"]), "Precision (CG)"),
        (pct(best["Recall_CG"]), "Recall (CG)"),
        (pct(best["F1_CG"]), "F1-Score (CG)"),
        (f"{best['Training_Time_Seconds']:.2f}s", "Training Time"),
    ])

    # Premium best-model + leaderboard area
    ranked = results.sort_values("F1_CG", ascending=False).reset_index(drop=True)
    rank_rows = []
    for i, row in ranked.iterrows():
        rank_rows.append(
            f'<div class="rank-row">'
            f'<div class="rank-num">{i+1}</div>'
            f'<div class="rank-name">{row["Model"]}</div>'
            f'<div class="rank-score">{row["F1_CG"]*100:.2f}%</div>'
            f'</div>'
        )

    st.html(
        f"""
<div class="performance-hero">
    <div class="best-model-card">
        <div class="best-model-kicker">SELECTED BASELINE CLASSIFIER</div>
        <div class="best-model-title">{best["Model"]}</div>
        <div class="best-model-sub">
            Highest validation F1 with extremely low training cost, making it the strongest
            accuracy–efficiency trade-off for the baseline text-classification pipeline.
        </div>
        <div class="best-badges">
            <span class="best-badge">F1 {best["F1_CG"]*100:.2f}%</span>
            <span class="best-badge">Accuracy {best["Accuracy"]*100:.2f}%</span>
            <span class="best-badge">{best["Training_Time_Seconds"]:.2f}s training</span>
        </div>
    </div>
    <div class="rank-card">
        <div class="rank-title">Model Leaderboard · F1 (CG)</div>
        {"".join(rank_rows)}
    </div>
</div>
"""
    )

    # Reliable custom HTML bars rather than Vega charts.
    metrics = [
        ("Accuracy", "Accuracy", "#1478F2", "#54B4FF"),
        ("Precision_CG", "Precision (CG)", "#7A50F9", "#B48CFF"),
        ("Recall_CG", "Recall (CG)", "#E52B70", "#FF72A7"),
        ("F1_CG", "F1-Score (CG)", "#00A979", "#37D5A7"),
    ]

    panels = []
    for col, title, accent, accent2 in metrics:
        rows = []
        best_val = results[col].max()
        for _, row in results.sort_values(col, ascending=False).iterrows():
            score = float(row[col]) * 100
            width = max(8, min(100, (score - 90) / 10 * 100))
            chip = "BEST" if abs(float(row[col]) - best_val) < 1e-12 else ""
            rows.append(
                f'<div class="model-bar-row">'
                f'<div class="model-bar-name">{row["Model"]}</div>'
                f'<div class="model-bar-track"><div class="model-bar-fill" '
                f'style="width:{width:.1f}%;--accent:{accent};--accent2:{accent2};"></div></div>'
                f'<div class="model-bar-score">{score:.2f}%</div>'
                f'</div>'
            )
        panels.append(
            f'<div class="metric-panel" style="--accent:{accent};--soft:{accent}14;">'
            f'<div class="metric-panel-title"><span>{title}</span>'
            f'<span class="metric-chip">{max(results[col])*100:.2f}% TOP</span></div>'
            f'{"".join(rows)}</div>'
        )

    st.html(
        '<div class="metric-panel-grid">' + "".join(panels) + '</div>'
    )

    # Training time with log-scaled custom bars
    positive_times = results["Training_Time_Seconds"].clip(lower=1e-6)
    min_log = math.log10(float(positive_times.min()))
    max_log = math.log10(float(positive_times.max()))
    span = max(max_log - min_log, 1e-9)

    time_rows = []
    for _, row in results.sort_values("Training_Time_Seconds").iterrows():
        t = float(row["Training_Time_Seconds"])
        w = 14 + 86 * ((math.log10(max(t, 1e-6)) - min_log) / span)
        time_rows.append(
            f'<div class="time-row">'
            f'<div class="time-name">{row["Model"]}</div>'
            f'<div class="time-track"><div class="time-fill" style="width:{w:.1f}%"></div></div>'
            f'<div class="time-value">{t:.2f}s</div>'
            f'</div>'
        )

    st.html(
        f"""
<div class="efficiency-panel">
    <div class="metric-panel-title">
        <span>Computational Efficiency</span>
        <span class="metric-chip" style="--accent:#F47521;--soft:#FFF0E7;">LOG-SCALED</span>
    </div>
    {"".join(time_rows)}
</div>
"""
    )

    # Decision summary
    rf = results.loc[results["Model"] == "Random Forest"]
    rf_time = float(rf["Training_Time_Seconds"].iloc[0]) if not rf.empty else float(results["Training_Time_Seconds"].max())
    nb_time = float(best["Training_Time_Seconds"])
    speed_ratio = rf_time / max(nb_time, 1e-9)

    st.html(
        f"""
<div class="decision-strip">
    <div class="decision-card">
        <strong>Why Naive Bayes was selected</strong>
        <p>
            It achieved the strongest validation F1 ({best["F1_CG"]*100:.2f}%) while training
            in only {nb_time:.2f} seconds. This provides a strong accuracy–efficiency balance
            for a deployable text-classification artefact.
        </p>
    </div>
    <div class="decision-card">
        <strong>Efficiency contrast</strong>
        <p>
            Random Forest required approximately {rf_time:.2f} seconds to train—about
            {speed_ratio:,.0f}× longer than the selected Naive Bayes model—while producing
            weaker validation performance.
        </p>
    </div>
</div>
"""
    )

    display = results.copy()
    for col in ["Accuracy", "Precision_CG", "Recall_CG", "F1_CG"]:
        display[col] = display[col].map(pct)
    display["Training_Time_Seconds"] = display["Training_Time_Seconds"].round(2)

    st.html('<div class="table-label">Detailed Validation Results</div>')
    st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":
    topbar(
        "Dataset Quality & Coverage",
        "Completeness • Balance • Ratings • Category coverage • Review length",
        "DeepSeek source dataset",
    )

    if not exists(TRAIN_CSV) or not exists(VAL_CSV):
        st.error("Training or validation CSV file was not found.")
        st.stop()

    train = load_csv(TRAIN_CSV)
    val = load_csv(VAL_CSV)
    test = test_df_all.copy()

    total = len(train) + len(val) + len(test)
    missing_total = int(
        train.isna().sum().sum()
        + val.isna().sum().sum()
        + test.isna().sum().sum()
    )
    duplicate_train = int(train.duplicated(subset=["text_"]).sum())

    kpis([
        (fmt_int(total), "Total Reviews"),
        (fmt_int(len(train)), "Training Reviews"),
        (fmt_int(len(val)), "Validation Reviews"),
        (fmt_int(len(test)), "Test Reviews"),
        (fmt_int(missing_total), "Missing Values"),
        (fmt_int(duplicate_train), "Training Duplicates"),
    ])

    status_ribbon()

    l, r = st.columns(2, gap="medium")

    with l:
        with st.container(border=True):
            panel_head("Training Label Balance")
            df = train["label"].value_counts().rename_axis("Label").reset_index(name="Reviews")
            chart = (
                alt.Chart(df)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("Reviews:Q", title="Reviews"),
                    y=alt.Y("Label:N", title=None),
                    color=alt.Color(
                        "Label:N",
                        scale=alt.Scale(domain=["CG","OR"], range=["#1478F2","#00A979"]),
                        legend=None,
                    ),
                    tooltip=["Label","Reviews"],
                )
                .properties(height=230)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    with r:
        with st.container(border=True):
            panel_head("Training Rating Distribution")
            df = train["rating"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Reviews")
            chart = (
                alt.Chart(df)
                .mark_bar(color="#7A50F9", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Rating:O", title="Rating"),
                    y=alt.Y("Reviews:Q", title="Reviews"),
                    tooltip=["Rating","Reviews"],
                )
                .properties(height=230)
            )
            st.altair_chart(chart_style(chart), use_container_width=True)

    # Review length by label — computed live from actual training data.
    with st.container(border=True):
        panel_head("Review Length Profile")
        length_df = train[["label","text_"]].copy()
        length_df["Words"] = length_df["text_"].fillna("").astype(str).str.split().str.len()
        summary = (
            length_df.groupby("label")["Words"]
            .agg(["count","mean","median","min","max"])
            .round(2)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.container(border=True):
        panel_head("Category Distribution")
        df = train["category"].value_counts().rename_axis("Category").reset_index(name="Reviews")
        chart = (
            alt.Chart(df)
            .mark_bar(color="#0F6FD9", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Reviews:Q", title="Reviews"),
                y=alt.Y("Category:N", title=None, sort="-x"),
                tooltip=["Category","Reviews"],
            )
            .properties(height=340)
        )
        st.altair_chart(chart_style(chart), use_container_width=True)

# ============================================================
# BATCH SCREENING
# ============================================================

elif page == "Batch Screening":
    topbar(
        "Batch Review Screening",
        "CSV upload • Multi-review classification • Confidence analytics • Export",
        "CSV input",
    )

    uploaded = st.file_uploader(
        "Upload a CSV containing a `text_` column",
        type=["csv"],
    )

    if uploaded is None:
        st.info("Upload a CSV file to begin batch screening.")
        quick_actions()
    else:
        work = pd.read_csv(uploaded)

        if "text_" not in work.columns:
            st.error("Required column missing: `text_`.")
        else:
            work = work.copy()
            work["text_"] = work["text_"].fillna("").astype(str)

            X = deploy_tfidf.transform(work["text_"])
            predictions = deploy_model.predict(X)
            probabilities = deploy_model.predict_proba(X)

            classes = list(deploy_model.classes_)
            cg_i, or_i = classes.index("CG"), classes.index("OR")

            work["prediction"] = predictions
            work["CG_probability"] = probabilities[:, cg_i]
            work["OR_probability"] = probabilities[:, or_i]
            work["confidence"] = work[["CG_probability","OR_probability"]].max(axis=1)

            cg_count = int((work["prediction"]=="CG").sum())
            or_count = int((work["prediction"]=="OR").sum())

            kpis([
                (fmt_int(len(work)), "Reviews Analysed"),
                (fmt_int(cg_count), "Predicted CG"),
                (fmt_int(or_count), "Predicted OR"),
                (pct(work["confidence"].mean()), "Average Confidence"),
                (fmt_int((work["confidence"]<.65).sum()), "Low-Confidence Cases"),
                ("READY", "Export Status"),
            ])

            if "label" in work.columns and work["label"].isin(["CG","OR"]).all():
                work["correct"] = work["label"] == work["prediction"]
                a = accuracy_score(work["label"],work["prediction"])
                p = precision_score(work["label"],work["prediction"],pos_label="CG",zero_division=0)
                r = recall_score(work["label"],work["prediction"],pos_label="CG",zero_division=0)
                f = f1_score(work["label"],work["prediction"],pos_label="CG",zero_division=0)

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Accuracy",pct(a))
                m2.metric("Precision (CG)",pct(p))
                m3.metric("Recall (CG)",pct(r))
                m4.metric("F1 (CG)",pct(f))

            st.dataframe(work, use_container_width=True, height=430)
            st.download_button(
                "Download ReviewGuard Results",
                data=work.to_csv(index=False).encode("utf-8"),
                file_name="reviewguard_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":
    topbar(
        "Research Report Library",
        "Download evaluation evidence and implementation outputs",
        "Available CSV evidence",
    )

    report_files = [
        ("Baseline Model Results", BASELINE_RESULTS),
        ("Controlled Cross-Generator Comparison", CONTROLLED_RESULTS),
        ("Leave-One-Generator-Out Results", LOGO_RESULTS),
        ("Performance by Category", PERF_CATEGORY),
        ("Performance by Rating", PERF_RATING),
        ("Performance by Review Length", PERF_LENGTH),
        ("Misclassified Reviews", MISCLASSIFIED),
    ]

    cols = st.columns(2)
    visible = 0
    for label, path in report_files:
        if exists(path):
            with open(path, "rb") as f:
                payload = f.read()
            with cols[visible % 2]:
                st.download_button(
                    f"⬇  {label}",
                    data=payload,
                    file_name=os.path.basename(path),
                    mime="text/csv",
                    use_container_width=True,
                )
            visible += 1

# ============================================================
# SETTINGS
# ============================================================

else:
    topbar(
        "Settings & Research Scope",
        "Interface preferences • Model boundaries • Responsible-use guidance",
        "Local research session",
    )

    st.subheader("Interface")
    st.write("Use the **Dark mode** switch in the sidebar to change the display theme.")

    st.subheader("Experimental vs Deployment Models")
    st.info(
        "Overview, Generator Analysis, Explainability and Model Performance use held-out experimental evidence. "
        "Review Analysis and Batch Screening use the separate final multi-generator deployment model."
    )

    st.subheader("Responsible Interpretation")
    st.warning(
        "ReviewGuard AI detects statistical text patterns. It does not determine fraudulent intent, "
        "reviewer identity, legal liability or platform-policy violations."
    )
