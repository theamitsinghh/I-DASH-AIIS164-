"""
╔══════════════════════════════════════════════════════════════╗
║     HYROX — AI Hybrid Warfare Intelligence System           ║
║     Team I-DASH  |  Streamlit Dashboard  |  Clean Build     ║
╚══════════════════════════════════════════════════════════════╝

Install:
    pip install streamlit==1.32.0 plotly==5.19.0 folium==0.16.0
    pip install streamlit-folium==0.18.0 pandas numpy scikit-learn

Run:
    streamlit run app.py
"""

# ── TensorFlow / Metal guards ─────────────────────────────────
# MUST be set before ANY import that could pull in tensorflow/keras.
# Prevents segfault on Apple Silicon (M1/M2/M3/M4) from Metal GPU plugin.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",     "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",      "")
os.environ.setdefault("TF_METAL_DEVICE_PLACEMENT", "0")

# ── Imports ───────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
# streamlit.components.v1 replaced by st.iframe
import time
import random
from datetime import datetime, timedelta
import requests

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HYROX — Hybrid Warfare Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HYROX Bridge — start once per server session ──────────────
import bridge

@st.cache_resource
def _start_bridge():
    """st.cache_resource runs this exactly once across all reruns."""
    return bridge.start(5050)

BRIDGE_PORT = _start_bridge()
BRIDGE_URL  = f"http://localhost:{BRIDGE_PORT}"   # always http://localhost:5050

# ─────────────────────────────────────────────────────────────
# CSS — fixed: removed the broad [class*="css"] selector
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── App background ─────────────────────────────────────── */
.stApp {
    background-color: #070C18 !important;
    color: #CBD5E1 !important;
    font-family: 'Courier New', monospace !important;
}
.block-container {
    background-color: #070C18 !important;
    padding-top: 3.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── Refresh button — high-visibility override ──────────── */
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button {
    background: linear-gradient(90deg, #0C2A1A, #071A26) !important;
    color: #00FFB0 !important;
    border: 1px solid #00FFB0 !important;
    box-shadow: 0 0 8px rgba(0,255,176,0.35), inset 0 0 8px rgba(0,255,176,0.06) !important;
    letter-spacing: 1.5px !important;
    font-weight: bold !important;
}

/* ── Hide Streamlit chrome ──────────────────────────────── */
#MainMenu  { visibility: visible; }
footer     { visibility: hidden; }
header     { visibility: visible; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0C1828 !important;
    border-right: 1px solid #1A3358 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, {
    color: #94A3B8 !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #0EA5E9 !important;
    color: #070C18 !important;
    border: none !important;
    font-weight: bold !important;
    width: 100% !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Metric cards ───────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #0C1828 !important;
    border: 1px solid #1A3358 !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricValue"] {
    color: #EFF6FF !important;
    font-size: 1.4rem !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: #0EA5E9 !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    font-weight: bold !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0C1828 !important;
    border-bottom: 2px solid #1A3358 !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: #070C18 !important;
    color: #475569 !important;
    border: 1px solid #1A3358 !important;
    border-radius: 4px 4px 0 0 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.8px !important;
    padding: 7px 14px !important;
}
.stTabs [aria-selected="true"] {
    background-color: #0C1828 !important;
    color: #22D3EE !important;
    border-top: 2px solid #0EA5E9 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #070C18 !important;
    padding-top: 0.8rem !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background-color: #0C1828 !important;
    color: #22D3EE !important;
    border: 1px solid #0EA5E9 !important;
    border-radius: 3px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.73rem !important;
    letter-spacing: 0.8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background-color: #0EA5E9 !important;
    color: #070C18 !important;
}

/* ── Selectbox ──────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #0C1828 !important;
    border: 1px solid #1A3358 !important;
    color: #CBD5E1 !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Slider ─────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #0EA5E9 !important;
}

/* ── Dataframe ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1A3358 !important;
    border-radius: 3px !important;
}

/* ── Download button ────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #022C22 !important;
    color: #10B981 !important;
    border: 1px solid #10B981 !important;
    border-radius: 3px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.73rem !important;
    width: 100% !important;
}

/* ── Spinner ────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #22D3EE !important;
}

/* ── Custom component classes ───────────────────────────── */
.top-header {
    background: linear-gradient(90deg, #0C1828, #070C18);
    border-bottom: 2px solid #0EA5E9;
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    border-radius: 3px;
}
.panel-box {
    background-color: #0C1828;
    border: 1px solid #1A3358;
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 8px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
}
.panel-title {
    color: #0EA5E9;
    font-size: 0.7rem;
    letter-spacing: 2px;
    font-weight: bold;
    border-bottom: 1px solid #1A3358;
    padding-bottom: 5px;
    margin-bottom: 8px;
    font-family: 'Courier New', monospace;
}
.terminal-box {
    background-color: #020804;
    border: 1px solid #10B981;
    border-radius: 3px;
    padding: 14px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #10B981;
    line-height: 1.9;
    white-space: pre-wrap;
    overflow-x: auto;
}
.alert-critical {
    background-color: #1A0606;
    border: 1px solid #EF4444;
    border-left: 4px solid #EF4444;
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #FCA5A5;
}
.alert-high {
    background-color: #1A0E00;
    border: 1px solid #F97316;
    border-left: 4px solid #F97316;
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #FDBA74;
}
.alert-medium {
    background-color: #141200;
    border: 1px solid #F59E0B;
    border-left: 4px solid #F59E0B;
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #FCD34D;
}
.alert-low {
    background-color: #001A0D;
    border: 1px solid #10B981;
    border-left: 4px solid #10B981;
    border-radius: 3px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #6EE7B7;
}
.score-badge-critical {
    display:inline-block; background-color:#7F1D1D;
    border:1px solid #EF4444; color:#FCA5A5;
    padding:3px 10px; border-radius:3px;
    font-size:0.75rem; font-weight:bold;
    font-family:'Courier New',monospace;
}
.score-badge-high {
    display:inline-block; background-color:#431407;
    border:1px solid #F97316; color:#FDBA74;
    padding:3px 10px; border-radius:3px;
    font-size:0.75rem; font-weight:bold;
    font-family:'Courier New',monospace;
}
.score-badge-medium {
    display:inline-block; background-color:#422006;
    border:1px solid #F59E0B; color:#FCD34D;
    padding:3px 10px; border-radius:3px;
    font-size:0.75rem; font-weight:bold;
    font-family:'Courier New',monospace;
}
.score-badge-low {
    display:inline-block; background-color:#022C22;
    border:1px solid #10B981; color:#6EE7B7;
    padding:3px 10px; border-radius:3px;
    font-size:0.75rem; font-weight:bold;
    font-family:'Courier New',monospace;
}
hr { border-color: #1A3358 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

ZONES = {
    "India–Pakistan LoC":     (34.10, 74.80),
    "India–China LAC":        (33.50, 79.50),
    "India–Bangladesh Border":(23.80, 91.00),
    "India–Arunachal Pradesh":(27.80, 93.60),
}

# Map simulator border_region strings → ZONES key
BORDER_REGION_MAP = {
    # simulator internal keys
    "india-pak":        "India–Pakistan LoC",
    "india-china":      "India–China LAC",
    "india-bangladesh": "India–Bangladesh Border",
    "india-arunachal":  "India–Arunachal Pradesh",
    # simulator payload .name strings (current after our fix)
    "India–Pakistan Line of Control":   "India–Pakistan LoC",
    "India–China LAC":                   "India–China LAC",
    "India–Bangladesh Border":           "India–Bangladesh Border",
    "India–Arunachal Pradesh":           "India–Arunachal Pradesh",
    # simulator payload .name strings (old / alternate forms)
    "India–China Line of Actual Control": "India–China LAC",
    "India–China Arunachal Border":       "India–Arunachal Pradesh",
    "India-Pakistan Line of Control":     "India–Pakistan LoC",
    "India-China LAC":                    "India–China LAC",
    "India-Bangladesh Border":            "India–Bangladesh Border",
    "India-Arunachal Pradesh":            "India–Arunachal Pradesh",
    # legacy app.py zone names that may linger in session state
    "Northern Sector": "India–Pakistan LoC",
    "Eastern Sector":  "India–Arunachal Pradesh",
    "Western Sector":  "India–Pakistan LoC",
    "Southern Sector": "India–Bangladesh Border",
    "Coastal Zone":    "India–Bangladesh Border",
    "Mountain Pass":   "India–China LAC",
}

PHASES = ["MONITORING", "RECONNAISSANCE", "STAGING", "MOBILISATION", "CRITICAL"]

PHASE_COLOR = {
    "MONITORING":     "#10B981",
    "RECONNAISSANCE": "#F59E0B",
    "STAGING":        "#F97316",
    "MOBILISATION":   "#EF4444",
    "CRITICAL":       "#DC2626",
}

PHASE_MULT = {
    "MONITORING": 0.3, "RECONNAISSANCE": 0.5,
    "STAGING": 0.7, "MOBILISATION": 0.85, "CRITICAL": 0.95,
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_threat_level(score):
    if score >= 0.85: return "CRITICAL", "#EF4444", "critical"
    if score >= 0.65: return "HIGH",     "#F97316", "high"
    if score >= 0.40: return "MEDIUM",   "#F59E0B", "medium"
    return "LOW", "#10B981", "low"


def generate_scores():
    """
    Try to fetch live inference results from the bridge server.

    Flow:
      1. Health-check GET /health  →  mark bridge online/offline in session state.
      2. If online, GET /state     →  read latest model inference result.
      3. Returns all zeros if no dispatch has arrived yet, keeping the
         dashboard clean and NOT showing random placeholder values.

    Simulator posts to:  POST http://localhost:5050/api/hyrox/ingest
    Dashboard reads from: GET  http://localhost:5050/state
    """
    # ── Step 1: health-check ───────────────────────────────────────────
    try:
        hc = requests.get(f"{BRIDGE_URL}/health", timeout=0.5)
        bridge_online = hc.status_code == 200
    except Exception:
        bridge_online = False

    st.session_state["bridge_online"] = bridge_online

    if not bridge_online:
        # Do NOT attempt /state if bridge is unreachable
        return 0.0, 0.0, 0.0, 0.0

    # ── Step 2: fetch latest inference state ───────────────────────────
    try:
        r = requests.get(f"{BRIDGE_URL}/state", timeout=0.4)
        if r.status_code == 200:
            data = r.json()
            if data.get("ready"):
                # ── Live data received from simulator ──────────────────
                st.session_state["bridge_state"] = data
                return (
                    round(data.get("cyber",   0.0), 3),
                    round(data.get("socmint", 0.0), 3),
                    round(data.get("geoint",  0.0), 3),
                    round(data.get("fused",   0.0), 3),
                )
    except Exception:
        pass

    # ── Fallback: ZEROS (not random) ──────────────────────────────────
    # Returns (0.0, 0.0, 0.0, 0.0) until the simulator sends its first
    # payload via POST /api/hyrox/ingest.
    return 0.0, 0.0, 0.0, 0.0


def _append_timeline(fused_score: float):
    """Append fused score + timestamp to rolling timeline (max 300 pts)."""
    times_list, tl_list = st.session_state.get("timeline", ([], []))
    label = datetime.now().strftime("%H:%M:%S")
    times_list = (times_list + [label])[-300:]
    tl_list    = (tl_list    + [fused_score])[-300:]
    st.session_state.timeline = (times_list, tl_list)


def generate_timeline(n=120):
    t = np.linspace(0, n, n)
    scores = np.clip(
        0.12 + 0.004 * t
        + 0.25 * np.exp((t - 85) / 18)
        + np.random.normal(0, 0.012, n),
        0.05, 1.0
    )
    times = [
        (datetime.now() - timedelta(minutes=n - i)).strftime("%H:%M")
        for i in range(n)
    ]
    return times, scores.tolist()


def generate_geo_points(zone_name, n_threat=40, n_normal=120):
    clat, clon = ZONES[zone_name]
    pts = []
    for _ in range(n_normal):
        pts.append({
            "lat": clat + np.random.uniform(-1.5, 1.5),
            "lon": clon + np.random.uniform(-1.5, 1.5),
            "threat": 0,
            "count": random.randint(1, 4),
            "speed": round(random.uniform(30, 90), 1),
        })
    for _ in range(n_threat):
        pts.append({
            "lat": clat + np.random.normal(0, 0.05),
            "lon": clon + np.random.normal(0, 0.05),
            "threat": 1,
            "count": random.randint(8, 25),
            "speed": round(random.uniform(3, 15), 1),
        })
    return pts


def generate_cyber_logs(n=150):
    rows = []
    for i in range(n):
        anom = random.random() < 0.12
        rows.append({
            "Time":       (datetime.now() - timedelta(minutes=n - i)).strftime("%H:%M:%S"),
            "IP":         f"{random.randint(10,220)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "Logins":     random.randint(200, 800) if anom else random.randint(1, 15),
            "Traffic MB": round(random.uniform(500, 2000), 1) if anom else round(random.uniform(0.1, 50), 1),
            "Port Scan":  "YES ⚠" if (anom and random.random() < 0.6) else "—",
            "Status":     "ANOMALY ⚠" if anom else "normal",
        })
    return pd.DataFrame(rows)


def generate_posts(n=40):
    threat_posts = [
        "Government hiding truth about border situation #exposed",
        "Military buildup confirmed near northern sector — share NOW",
        "Troops spotted crossing border — mainstream media silent",
        "Cyber attack on infrastructure COVERED UP by authorities",
        "Border violations happening daily — wake up people",
        "Coordinated disinformation campaign targeting our region",
        "Foreign agents infiltrating social networks — be aware",
        "Military movements confirmed by satellite imagery leaked",
    ]
    normal_posts = [
        "Great weather today in the city, perfect for a walk",
        "New restaurant opened downtown, food was amazing",
        "Local sports team wins championship tonight",
        "Traffic update: highway clear, no delays reported",
        "Community event this weekend, all welcome to join",
    ]
    rows = []
    for _ in range(n):
        threat = random.random() < 0.45
        post   = random.choice(threat_posts if threat else normal_posts)
        rows.append({
            "Time":       (datetime.now() - timedelta(minutes=random.randint(1, 120))).strftime("%H:%M"),
            "Post":       post[:58] + "..." if len(post) > 58 else post,
            "Confidence": round(random.uniform(0.70, 0.96), 2) if threat else round(random.uniform(0.05, 0.28), 2),
            "Label":      "PROPAGANDA ⚠" if threat else "normal",
            "Bots":       random.randint(50, 400) if threat else random.randint(0, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────

def _safe_zone(z: str) -> str:
    """Return z if it's a valid ZONES key, otherwise map it or fall back to first key."""
    if z in ZONES:
        return z
    mapped = BORDER_REGION_MAP.get(z, "")
    if mapped and mapped in ZONES:
        return mapped
    return list(ZONES.keys())[0]

if "scores" not in st.session_state:
    # Start at absolute zero — no random values on first load.
    # generate_scores() will return zeros until the first simulator dispatch.
    st.session_state.scores = (0.0, 0.0, 0.0, 0.0)

if "timeline" not in st.session_state:
    # Empty timeline — no fabricated historical data before real data arrives.
    st.session_state.timeline = ([], [])

if "zone" not in st.session_state:
    st.session_state.zone = "India–Pakistan LoC"

if "phase" not in st.session_state:
    # Start at MONITORING (least alarming) rather than STAGING
    st.session_state.phase = "MONITORING"

if "alert_count" not in st.session_state:
    # Zero alerts until real data arrives
    st.session_state.alert_count = 0

if "boot_time" not in st.session_state:
    st.session_state.boot_time = datetime.now()

if "bridge_online" not in st.session_state:
    st.session_state.bridge_online = True   # assume online until first check

if "bridge_state" not in st.session_state:
    st.session_state.bridge_state = {}

if "last_dispatch" not in st.session_state:
    st.session_state.last_dispatch = None


# ─────────────────────────────────────────────────────────────
# UNPACK SCORES
# ─────────────────────────────────────────────────────────────

cyber, socmint, geoint, fused = st.session_state.scores
level, lc, lcls = get_threat_level(fused)
cl,  cc,  _     = get_threat_level(cyber)
sl,  sc,  _     = get_threat_level(socmint)
gl,  gc,  _     = get_threat_level(geoint)

# ── Pull all real model outputs from bridge_state ─────────────────────────
_bstate = st.session_state.get("bridge_state", {})
_live   = _bstate.get("ready", False)

# ── When live, override level with simulator's declared threat_level ───────
# The XGBoost model compresses outputs; the simulator's threat_level field
# directly reflects the simulation state and is more reliable for display.
_sim_threat_level = _bstate.get("raw_payload", {}).get("threat_level", "") if _live else ""
_SIM_LEVEL_MAP = {
    "LOW":      ("LOW",      "#10B981", "low"),
    "MODERATE": ("MEDIUM",   "#F59E0B", "medium"),
    "HIGH":     ("HIGH",     "#F97316", "high"),
    "CRITICAL": ("CRITICAL", "#EF4444", "critical"),
}
if _live and _sim_threat_level in _SIM_LEVEL_MAP:
    level, lc, lcls = _SIM_LEVEL_MAP[_sim_threat_level]

# Gauge display value — map simulator level to minimum gauge position
# so needle always reaches the correct zone even if XGBoost compresses scores
_level_to_min = {"LOW": 0.0, "MODERATE": 0.45, "HIGH": 0.68, "CRITICAL": 0.87}
_gauge_fused = max(fused, _level_to_min.get(_sim_threat_level, 0.0)) if _live and _sim_threat_level else fused

# SOCMINT — Model 3 + Model 4 + SBERT
_socmint_label      = _bstate.get("socmint_label",       sl)
_bot_flag           = _bstate.get("iso_anomaly",         False)
_coordination_idx   = _bstate.get("coordination_index",  None)
_n_clusters         = _bstate.get("n_clusters",          None)
_narrative_drift    = _bstate.get("narrative_drift",      None)
_sbert_score        = _bstate.get("sbert_score",         None)
_n_total_alerts     = _bstate.get("raw_payload", {}).get("social_intel", {}).get("total_alerts", None)

# CYBINT — Model 5 + Model 6
_cybint_label       = _bstate.get("cybint_label",        cl)
_zeroday_flag       = _bstate.get("ae_anomaly",          False)
_ae_mse             = _bstate.get("ae_mse",              None)
_model5_label       = _bstate.get("model5_label",        cl)
_n_cyber_events     = _bstate.get("raw_payload", {}).get("cyber_intel", {}).get("total_events", None)
_n_intrusions       = sum(1 for e in _bstate.get("raw_payload",{}).get("cyber_intel",{}).get("alerts",[]) if e.get("type")=="intrusion")
_n_jamming          = sum(1 for e in _bstate.get("raw_payload",{}).get("cyber_intel",{}).get("alerts",[]) if e.get("type")=="jamming")
_n_malware          = sum(1 for e in _bstate.get("raw_payload",{}).get("cyber_intel",{}).get("alerts",[]) if e.get("type")=="malware")

# GEOINT — Model 1 + Model 2
_geoint_label       = _bstate.get("geoint_label",        gl)
_lstm_trajectory    = _bstate.get("lstm_trajectory",     None)   # e.g. "ESCALATING"
_model1_label       = _bstate.get("model1_label",        gl)
_n_total_units      = _bstate.get("raw_payload", {}).get("geo_intel", {}).get("total_units", None)
_border_dist        = _bstate.get("border_dist_km",      None)
_mean_velocity      = _bstate.get("mean_velocity_kmh",   None)

# Fusion — Model 7
_fusion_method      = _bstate.get("fusion_method",       "Weighted (0.40/0.30/0.30)")
_escalation_pattern = _bstate.get("escalation_pattern",  None)

# Auto-sync zone from live simulator payload border_region
_sim_border = _bstate.get("raw_payload", {}).get("border_region", "")
if _live and _sim_border:
    _mapped_zone = BORDER_REGION_MAP.get(_sim_border, "")
    if _mapped_zone and _mapped_zone in ZONES and _mapped_zone != st.session_state.zone:
        st.session_state.zone = _mapped_zone

# Always guarantee session zone is a valid ZONES key
st.session_state.zone = _safe_zone(st.session_state.zone)


# ─────────────────────────────────────────────────────────────
# PLOTLY LAYOUT DEFAULTS
# ─────────────────────────────────────────────────────────────

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="#070C18",
        plot_bgcolor="#0C1828",
        font=dict(family="Courier New", color="#94A3B8", size=9),
        margin=dict(l=40, r=20, t=14, b=30),
        showlegend=False,
    )
    base.update(kwargs)
    return base


def dark_axes():
    return dict(showgrid=True, gridcolor="#0D1F33",
                zeroline=False, tickfont=dict(size=8))


# ─────────────────────────────────────────────────────────────
# BRIDGE STATUS BANNER  (shown only when bridge is unreachable)
# ─────────────────────────────────────────────────────────────

if not st.session_state.get("bridge_online", True):
    st.markdown("""
    <div style='background:#1A0606; border:1px solid #EF4444; border-left:4px solid #EF4444;
                border-radius:3px; padding:8px 14px; margin-bottom:8px;
                font-family:Courier New,monospace; font-size:0.8rem; color:#FCA5A5;'>
      ⚠  BRIDGE OFFLINE — Cannot reach <b>http://localhost:5050</b>.
      Ensure <code>bridge.py</code> is running and the port is not blocked.
      Dashboard is showing last-known or zero values.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP HEADER
# ─────────────────────────────────────────────────────────────

now_str  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S  UTC")
uptime   = str(datetime.now() - st.session_state.boot_time).split(".")[0]

st.markdown(f"""
<div class="top-header">
  <div>
    <div style="color:#EFF6FF; font-size:1.05rem; font-weight:bold;
                letter-spacing:2px; font-family:'Courier New',monospace;">
      🛡  HYROX — AI HYBRID WARFARE INTELLIGENCE SYSTEM
    </div>
    <div style="color:#3D5A7A; font-size:0.68rem; letter-spacing:1px; margin-top:3px;">
      MULTI-DOMAIN THREAT ANALYSIS  ·  REAL-TIME MONITORING  ·  UPTIME: {uptime}
    </div>
  </div>
  <div style="text-align:center;">
    <div style="color:{lc}; font-size:1.05rem; font-weight:bold;
                letter-spacing:3px; font-family:'Courier New',monospace;">
      ⚠  THREAT LEVEL: {level}
    </div>
    <div style="color:#475569; font-size:0.68rem; margin-top:3px;">
      FUSED SCORE: {fused:.3f}  ·  ALERTS: {st.session_state.alert_count}
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:#F59E0B; font-size:0.82rem; font-weight:bold;
                letter-spacing:3px; font-family:'Courier New',monospace;">
      TEAM  I-DASH  ·  PROJECT HYROX
    </div>
    <div style="color:#475569; font-size:0.68rem; margin-top:3px;">{now_str}</div>
    <div style="color:#475569; font-size:0.68rem;">
      ZONE: {st.session_state.zone}  ·  PHASE: {st.session_state.phase}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:8px 0 14px 0;'>
      <div style='color:#F59E0B; font-size:1.05rem; font-weight:bold;
                  letter-spacing:3px; font-family:Courier New,monospace;'>
        TEAM  I-DASH
      </div>
      <div style='color:#22D3EE; font-size:0.72rem; letter-spacing:2px; margin-top:3px;'>
        PROJECT  HYROX
      </div>
    </div>
    <hr style='border-color:#1A3358; margin:8px 0 14px 0;'/>
    """, unsafe_allow_html=True)

    # Refresh
    st.markdown("<div style='color:#0EA5E9; font-size:0.68rem; letter-spacing:2px; font-weight:bold; margin-bottom:6px;'>SYSTEM CONTROLS</div>", unsafe_allow_html=True)
    if st.button("⟳  REFRESH INTELLIGENCE FEED", width='stretch'):
        st.session_state.scores = generate_scores()
        # Only bump alert count if live data actually arrived
        _live_on_refresh = st.session_state.get("bridge_state", {}).get("ready", False)
        if _live_on_refresh:
            n_alerts = (
                st.session_state.bridge_state
                .get("raw_payload", {})
                .get("social_intel", {})
                .get("total_alerts", 0)
            )
            st.session_state.alert_count = n_alerts
            _append_timeline(st.session_state.scores[3])
        st.rerun()

    # Auto-refresh toggle ────────────────────────────────────────────────────
    st.markdown("<div style='color:#0EA5E9;font-size:0.68rem;letter-spacing:2px;font-weight:bold;margin-bottom:4px;'>LIVE POLLING</div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh (1s)", value=False, key="auto_refresh")
    st.markdown("<div style='color:#3D5A7A;font-size:0.62rem;'>Pull latest simulation scores automatically</div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div style='color:#0EA5E9; font-size:0.68rem; letter-spacing:2px; font-weight:bold; margin-bottom:4px;'>MONITORING ZONE</div>", unsafe_allow_html=True)
    _zone_keys = list(ZONES.keys())
    _zone_idx  = _zone_keys.index(st.session_state.zone) if st.session_state.zone in _zone_keys else 0
    new_zone = st.selectbox("zone_sel", _zone_keys,
        index=_zone_idx,
        label_visibility="collapsed")
    if new_zone != st.session_state.zone:
        st.session_state.zone = new_zone
        st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # Phase simulator
    st.markdown("<div style='color:#0EA5E9; font-size:0.68rem; letter-spacing:2px; font-weight:bold; margin-bottom:4px;'>THREAT SIMULATION PHASE</div>", unsafe_allow_html=True)
    new_phase = st.selectbox("phase_sel", PHASES,
        index=PHASES.index(st.session_state.phase),
        label_visibility="collapsed")
    if new_phase != st.session_state.phase:
        st.session_state.phase = new_phase

    pc = PHASE_COLOR[st.session_state.phase]
    st.markdown(f"""
    <div style='background:#0C1828; border:1px solid {pc}; border-radius:3px;
                padding:7px; margin-top:5px; text-align:center;'>
      <span style='color:{pc}; font-size:0.78rem; font-weight:bold;'>
        ●  {st.session_state.phase}
      </span>
      <div style='color:#475569; font-size:0.68rem; margin-top:2px;'>
        Multiplier: ×{PHASE_MULT[st.session_state.phase]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Fusion weights — fixed defaults (sliders removed)
    w_c, w_s, w_g = 0.40, 0.30, 0.30

    # System status
    # ── Live Model Status ──────────────────────────────────────────────────────
    bs = st.session_state.get("bridge_state", {})
    model_status  = bs.get("model_status", {})
    fusion_method = bs.get("fusion_method", "—")
    models_loaded = bs.get("models_loaded", "0/6")
    ae_anom  = bs.get("ae_anomaly",  False)
    iso_anom = bs.get("iso_anomaly", False)
    sim_ts   = bs.get("timestamp", None)
    bridge_ready = bs.get("ready", False)

    def _dot(ok):
        c = "#10B981" if ok else "#EF4444"
        s = "ACTIVE" if ok else "OFFLINE"
        return f"<span style='color:{c};'>●</span>  {{name:<22}}  {s}"

    status_rows = "".join(
        f"<span style='color:{'#10B981' if v else '#EF4444'};'>●</span>"
        f"&nbsp;&nbsp;{k}&nbsp;&nbsp;<span style='color:{'#10B981' if v else '#475569'};font-size:0.67rem;'>{'ACTIVE' if v else 'OFFLINE'}</span><br/>"
        for k, v in model_status.items()
    ) if model_status else "<span style='color:#F59E0B;font-size:0.72rem;'>No dispatch received yet.<br/>Open simulator and press SEND.</span><br/>"

    dispatch_info = f"Last: {sim_ts[11:19]} UTC" if sim_ts else "Awaiting first dispatch"

    st.markdown(f"""
    <div class='panel-box'>
      <div class='panel-title'>MODEL STATUS  ·  {models_loaded} LOADED</div>
      <div style='line-height:2.0; font-size:0.70rem;'>
        {status_rows}
        <span style='color:#0EA5E9;'>●</span>&nbsp;&nbsp;
        FUSION ENGINE&nbsp;&nbsp;<span style='color:#0EA5E9;font-size:0.67rem;'>{fusion_method}</span><br/>
        {"<span style='color:#EF4444;'>⚠ AE ANOMALY DETECTED</span><br/>" if ae_anom else ""}
        {"<span style='color:#F97316;'>⚠ ISO ANOMALY DETECTED</span><br/>" if iso_anom else ""}
      </div>
    </div>
    <div style='text-align:center;color:#3D5A7A;font-size:0.62rem;padding:4px 0;'>
      BRIDGE :5050  ·  {dispatch_info}
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric("CYBINT SCORE",   f"{cyber:.3f}",   delta=cl,  delta_color="off")
with k2: st.metric("SOCMINT SCORE",  f"{socmint:.3f}", delta=sl,  delta_color="off")
with k3: st.metric("GEOINT SCORE",   f"{geoint:.3f}",  delta=gl,  delta_color="off")
with k4: st.metric("FUSED SCORE",    f"{fused:.3f}",   delta=level, delta_color="off")
with k5: st.metric("ACTIVE ALERTS",  st.session_state.alert_count, delta="↑ LIVE")
with k6: st.metric("ACTIVE ZONE",    st.session_state.zone[:12],   delta=st.session_state.phase)

# ── Live anomaly banner ───────────────────────────────────────────────────────
_bs = st.session_state.get("bridge_state", {})
_ae  = _bs.get("ae_anomaly",  False)
_iso = _bs.get("iso_anomaly", False)
_src = _bs.get("ready", False)
if _src:
    _counts = _bs.get("raw_payload", {})
    _parts  = []
    if _ae:  _parts.append("⚠ AUTOENCODER ANOMALY — CYBINT")
    if _iso: _parts.append("⚠ ISOLATION FOREST ANOMALY — SOCINT")
    _sim_bdr = _bs.get("raw_payload", {}).get("border_region", "")
    _fm = _bs.get("fusion_method", "—")
    _disp_line = f"  ·  Region: {_sim_bdr}  ·  Fusion: {_fm}" if _sim_bdr else f"  ·  Fusion: {_fm}"
    if _parts:
        st.markdown(
            f"<div style='background:#1A0606;border:1px solid #EF4444;border-left:4px solid #EF4444;"
            f"border-radius:3px;padding:7px 14px;font-family:Courier New,monospace;font-size:0.75rem;"
            f"color:#FCA5A5;margin-bottom:4px;'>{'   ·   '.join(_parts)}{_disp_line}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:#001A0D;border:1px solid #10B981;border-left:4px solid #10B981;"
            f"border-radius:3px;padding:5px 14px;font-family:Courier New,monospace;font-size:0.72rem;"
            f"color:#6EE7B7;margin-bottom:4px;'>✓ No anomalies detected — All domains nominal{_disp_line}</div>",
            unsafe_allow_html=True
        )

st.markdown("<hr style='border-color:#1A3358; margin:6px 0 10px 0;'/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺  OVERVIEW",
    "📡  SOCMINT",
    "🖥  CYBINT",
    "🛰  GEOINT",
    "⚙  FUSION ENGINE",
    "📋  INTEL REPORT",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════

with tab1:
    col_map, col_right = st.columns([6, 4], gap="small")

    with col_map:
        st.markdown("<div class='panel-title'>◈  LIVE TACTICAL THREAT MAP</div>", unsafe_allow_html=True)

        clat, clon = ZONES[st.session_state.zone]

        # ── Pull real unit/alert/event positions from payload when live ────────
        _raw_payload  = _bstate.get("raw_payload", {})
        _geo_units    = _raw_payload.get("geo_intel",    {}).get("units",   [])
        _soc_alerts   = _raw_payload.get("social_intel", {}).get("alerts",  [])
        _cyb_events   = _raw_payload.get("cyber_intel",  {}).get("alerts",  [])

        # When live, re-center on mean unit lat/lon if units exist
        if _live and _geo_units:
            _lats = [u["lat"] for u in _geo_units if "lat" in u]
            _lons = [u["lng"] for u in _geo_units if "lng" in u]
            if _lats:
                clat = sum(_lats) / len(_lats)
                clon = sum(_lons) / len(_lons)

        m = folium.Map(location=[clat, clon], zoom_start=8,
                       tiles="CartoDB dark_matter", prefer_canvas=True)

        if _live and (_geo_units or _soc_alerts or _cyb_events):
            # ── LIVE path: plot real simulator positions ───────────────────────

            # Heatmap from all three domains
            heat_pts = []
            for u in _geo_units:
                heat_pts.append([u["lat"], u["lng"], 0.9])
            for a in _soc_alerts:
                heat_pts.append([a["lat"], a["lng"], 0.55])
            for e in _cyb_events:
                heat_pts.append([e["lat"], e["lng"], 0.4])
            if heat_pts:
                HeatMap(heat_pts, radius=28, blur=20, max_zoom=10,
                    gradient={"0.2":"#22D3EE","0.5":"#F59E0B","0.8":"#F97316","1.0":"#EF4444"},
                ).add_to(m)

            # GEOINT unit markers
            _UNIT_COLOR = {"tank":"#EF4444","troops":"#F97316","drone":"#F59E0B","vehicle":"#22D3EE"}
            _UNIT_ICON  = {"tank":"▲","troops":"●","drone":"◆","vehicle":"■"}
            geo_cluster = MarkerCluster(name="GEO Units").add_to(m)
            for u in _geo_units:
                _ut  = u.get("type","unit")
                _uc  = _UNIT_COLOR.get(_ut, "#EF4444")
                _ui  = _UNIT_ICON.get(_ut, "●")
                _vel = u.get("velocity_kmh", 0)
                folium.CircleMarker(
                    [u["lat"], u["lng"]], radius=7,
                    color=_uc, fill=True, fill_opacity=0.85,
                    popup=folium.Popup(
                        f"<div style='background:#120606;color:{_uc};"
                        f"font-family:monospace;font-size:11px;padding:8px;"
                        f"border:1px solid {_uc};'>"
                        f"<b>{_ui} {_ut.upper()}</b><br/>"
                        f"ID: {u.get('id','—')}<br/>"
                        f"Velocity: {_vel} km/h<br/>"
                        f"Lat/Lon: {u['lat']:.4f}, {u['lng']:.4f}</div>",
                        max_width=200),
                    tooltip=f"{_ui} {_ut} · {_vel} km/h",
                ).add_to(geo_cluster)

            # SOCINT alert markers (orange diamonds)
            for a in _soc_alerts:
                _sev  = a.get("severity","medium")
                _sc_  = "#EF4444" if _sev == "high" else "#F59E0B" if _sev == "medium" else "#10B981"
                folium.CircleMarker(
                    [a["lat"], a["lng"]], radius=5,
                    color=_sc_, fill=True, fill_opacity=0.75,
                    popup=folium.Popup(
                        f"<div style='background:#1A0E00;color:{_sc_};"
                        f"font-family:monospace;font-size:11px;padding:8px;"
                        f"border:1px solid {_sc_};'>"
                        f"<b>SOCINT {a.get('type','alert').upper()}</b><br/>"
                        f"Severity: {_sev}<br/>"
                        f"Msg: {str(a.get('msg',''))[:60]}</div>",
                        max_width=220),
                    tooltip=f"SOCINT · {a.get('type','alert')} · {_sev}",
                ).add_to(m)

            # CYBINT event markers (cyan squares)
            for e in _cyb_events:
                folium.CircleMarker(
                    [e["lat"], e["lng"]], radius=4,
                    color="#22D3EE", fill=True, fill_opacity=0.60,
                    tooltip=f"CYBINT · {e.get('type','event')}",
                ).add_to(m)

        else:
            # ── Fallback: random points ────────────────────────────────────────
            pts = generate_geo_points(st.session_state.zone)
            HeatMap(
                [[p["lat"], p["lon"], 0.9 if p["threat"] else 0.15] for p in pts],
                radius=28, blur=20, max_zoom=10,
                gradient={"0.2":"#22D3EE","0.5":"#F59E0B","0.8":"#F97316","1.0":"#EF4444"},
            ).add_to(m)
            for p in pts:
                if p["threat"] == 0:
                    folium.CircleMarker(
                        [p["lat"], p["lon"]], radius=3,
                        color="#22D3EE", fill=True, fill_opacity=0.35,
                        tooltip="Normal Activity",
                    ).add_to(m)
            cluster = MarkerCluster(name="Threat Clusters",
                options={"disableClusteringAtZoom": 10}).add_to(m)
            for p in pts:
                if p["threat"] == 1:
                    folium.CircleMarker(
                        [p["lat"], p["lon"]], radius=8,
                        color="#EF4444", fill=True, fill_opacity=0.85,
                        popup=folium.Popup(
                            f"<div style='background:#120606;color:#FCA5A5;"
                            f"font-family:monospace;font-size:11px;padding:8px;"
                            f"border:1px solid #EF4444;'>"
                            f"<b>⚠ THREAT</b><br/>Count: {p['count']}<br/>"
                            f"Speed: {p['speed']} km/h<br/>Pattern: STAGING</div>",
                            max_width=180),
                        tooltip="⚠ Threat",
                    ).add_to(cluster)

        # Always-on decorations
        folium.Circle([clat, clon], radius=8000,
            color="#EF4444", fill=True, fill_opacity=0.04,
            weight=1.5, dash_array="8").add_to(m)
        folium.Marker([clat, clon],
            icon=folium.DivIcon(html=(
                "<div style='color:#EF4444;font-size:22px;font-weight:bold;"
                "text-shadow:0 0 8px #EF4444;margin-left:-11px;margin-top:-11px;'>✛</div>"
            )),
            tooltip=f"⚠ {st.session_state.zone}",
        ).add_to(m)
        folium.PolyLine(
            [[clat + 0.3, clon - 2.2], [clat + 0.3, clon + 2.2]],
            color="#EF4444", weight=2, dash_array="10",
            opacity=0.7, tooltip="Border Zone — Restricted",
        ).add_to(m)

        MiniMap(tile_layer="CartoDB dark_matter",
                position="bottomright", width=110, height=75).add_to(m)
        folium.LayerControl().add_to(m)

        st.iframe(m._repr_html_(), height=450)



    with col_right:
        # Threat timeline
        st.markdown("<div class='panel-title'>◈  THREAT SCORE TIMELINE</div>", unsafe_allow_html=True)
        times, tl = st.session_state.timeline
        fig_tl = go.Figure()
        for y0, y1, col in [(0,.40,"rgba(16,185,129,0.07)"),(.40,.65,"rgba(245,158,11,0.07)"),
                             (.65,.85,"rgba(249,115,22,0.07)"),(.85,1,"rgba(239,68,68,0.09)")]:
            fig_tl.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0)
        for thresh, col, lbl in [(.40,"#F59E0B","MED"),(.65,"#F97316","HIGH"),(.85,"#EF4444","CRIT")]:
            fig_tl.add_hline(y=thresh, line_dash="dot", line_color=col,
                             line_width=0.8, opacity=0.5)
            if times:
                fig_tl.add_annotation(x=times[-1], y=thresh, text=lbl, showarrow=False,
                    font=dict(size=7, color=col), xanchor="right", yanchor="bottom")
        if times:
            fig_tl.add_trace(go.Scatter(
                x=times, y=tl, mode="lines",
                line=dict(color="#EF4444", width=2),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
            ))
            fig_tl.add_trace(go.Scatter(
                x=[times[-1]], y=[tl[-1]], mode="markers",
                marker=dict(size=9, color="#EF4444", line=dict(width=2, color="white")),
            ))
            _tl_tickvals = [times[i] for i in [0, 30, 60, 90] if i < len(times)] + [times[-1]]
            _tl_xaxis = {**dark_axes(), "showgrid": False, "tickvals": _tl_tickvals}
        else:
            _tl_xaxis = {**dark_axes(), "showgrid": False}
        fig_tl.update_layout(**dark_layout(height=190),
            xaxis=_tl_xaxis,
            yaxis={**dark_axes(), "range": [0, 1.05]},
        )
        if times:
            st.plotly_chart(fig_tl, width='stretch', config={"displayModeBar": False})
        else:
            st.markdown(
                "<div style='height:190px; display:flex; align-items:center; justify-content:center;"
                " color:#475569; font-size:0.75rem; font-family:Courier New,monospace; border:1px solid #1A3358;"
                " border-radius:4px;'>AWAITING FIRST DISPATCH — timeline will populate on live data</div>",
                unsafe_allow_html=True,
            )

        # Domain bar
        st.markdown("<div class='panel-title'>◈  DOMAIN SCORE COMPARISON</div>", unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            y=["CYBINT","SOCMINT","GEOINT","FUSED"],
            x=[cyber, socmint, geoint, fused],
            orientation="h",
            marker=dict(color=["#EF4444","#F97316","#F59E0B","#22D3EE"]),
            text=[f"{v:.3f}" for v in [cyber, socmint, geoint, fused]],
            textposition="outside", textfont=dict(color="#CBD5E1", size=9),
        ))
        fig_bar.add_vline(x=0.65, line_dash="dot", line_color="#F97316",
                          line_width=1, opacity=0.6)
        fig_bar.update_layout(**dark_layout(height=155, margin=dict(l=10, r=55, t=10, b=20)),
            xaxis={**dark_axes(), "range": [0, 1.15]},
            yaxis={**dark_axes(), "showgrid": False, "tickfont": dict(size=9)},
        )
        st.plotly_chart(fig_bar, width='stretch', config={"displayModeBar": False})

        # Alert feed — real when live, canned otherwise
        st.markdown("<div class='panel-title'>◈  ACTIVE ALERT FEED</div>", unsafe_allow_html=True)
        if _live:
            _alerts_live = []
            if fused >= 0.85 or _sim_threat_level == "CRITICAL":
                _alerts_live.append(("critical", f"⚠ CRITICAL THREAT  |  {_sim_border or st.session_state.zone}  |  Score {fused:.3f}  |  SIM: {_sim_threat_level}"))
            elif fused >= 0.65 or _sim_threat_level == "HIGH":
                _alerts_live.append(("high",     f"↑ HIGH THREAT  |  {_sim_border or st.session_state.zone}  |  Score {fused:.3f}  |  SIM: {_sim_threat_level}"))
            if _zeroday_flag:
                _alerts_live.append(("critical", f"⚠ ZERO-DAY ANOMALY  |  AE MSE={_ae_mse:.4f}  |  CYBINT"))
            if _bot_flag:
                _alerts_live.append(("high",     f"↑ BOT CLUSTER DETECTED  |  IsoForest M4  |  SOCMINT"))
            if _coordination_idx is not None and _coordination_idx > 0.5:
                _alerts_live.append(("high",     f"↑ HIGH COORDINATION IDX={_coordination_idx:.3f}  |  SBERT  |  SOCMINT"))
            if _lstm_trajectory in ("ESCALATING","CRITICAL"):
                _alerts_live.append(("high",     f"↑ TRAJECTORY: {_lstm_trajectory}  |  BiLSTM M2  |  GEOINT"))
            if _border_dist is not None and _border_dist < 2.0:
                _alerts_live.append(("high",     f"↑ UNITS {_border_dist:.2f}km FROM BORDER  |  GEOINT"))
            _active_sc2 = _bstate.get("raw_payload", {}).get("active_scenarios", [])
            for sc in _active_sc2:
                _alerts_live.append(("medium", f"→ SCENARIO: {sc.upper()}"))
            if not _alerts_live:
                _alerts_live.append(("low", "✓ All domains nominal — no alerts"))
            for cls, msg in _alerts_live:
                st.markdown(f'<div class="alert-{cls}">{msg}</div>', unsafe_allow_html=True)
        else:
            alerts = [
                ("critical", f"⚠ CLUSTER ALPHA-7  |  {st.session_state.zone}  |  43 objects"),
                ("high",     f"↑ CYBINT  |  Port scan 4,200/min  |  {datetime.now().strftime('%H:%M')}"),
                ("high",     f"↑ SOCMINT  |  1,240 bot accounts  |  Propaganda surge"),
                ("medium",   f"→ GEOINT  |  Vehicles 1.2km from border"),
                ("low",      f"ℹ SOCMINT  |  Keyword spike +847%"),
            ]
            for cls, msg in alerts:
                st.markdown(f'<div class="alert-{cls}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — SOCMINT
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div class='panel-title'>◈  SOCIAL MEDIA INTELLIGENCE — XGBoost M3 · Isolation Forest M4 · SBERT</div>", unsafe_allow_html=True)
    m1, m2_, m3, m4_ = st.columns(4)
    with m1:  st.metric("SOCMINT SCORE",    f"{socmint:.3f}", delta=_socmint_label)
    with m2_: st.metric("ALERTS INGESTED",  str(_n_total_alerts) if _n_total_alerts is not None else "—",
                         delta="LIVE" if _live else "NO FEED")
    with m3:  st.metric("BOT / ANOMALY",    "⚠ DETECTED" if _bot_flag else "NORMAL",
                         delta="IsoForest M4")
    with m4_: st.metric("COORDINATION IDX", f"{_coordination_idx:.3f}" if _coordination_idx is not None else "—",
                         delta="SBERT" if _coordination_idx is not None else "Awaiting")

    # ── Real model output panel ───────────────────────────────────────────────
    if _live:
        _sbert_html = ""
        if _coordination_idx is not None:
            _sbert_html = f"""
            <span style='color:#0EA5E9;'>Coordination Index :</span> {_coordination_idx:.4f}<br/>
            <span style='color:#0EA5E9;'>Clusters Detected  :</span> {_n_clusters if _n_clusters is not None else "—"}<br/>
            <span style='color:#0EA5E9;'>Narrative Drift    :</span> {f"{_narrative_drift:.4f}" if _narrative_drift is not None else "—"}<br/>
            <span style='color:#0EA5E9;'>SBERT Score        :</span> {f"{_sbert_score:.4f}" if _sbert_score is not None else "—"}
            """
        _bot_color = "#EF4444" if _bot_flag else "#10B981"
        _bot_text  = "⚠ BOT / ANOMALY DETECTED" if _bot_flag else "✓ NORMAL — No bot cluster detected"
        st.markdown(f"""
        <div class='panel-box' style='border-left:3px solid #F97316; margin-bottom:10px;'>
          <div class='panel-title'>◈  MODEL OUTPUTS  ·  M3 XGBoost + M4 IsoForest + SBERT</div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;font-size:0.73rem;margin-top:8px;'>
            <div>
              <div style='color:#F97316;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 3 — XGBOOST CLASSIFIER</div>
              <span style='color:#0EA5E9;'>SOCMINT Label  :</span> <b style='color:#EFF6FF;'>{_socmint_label}</b><br/>
              <span style='color:#0EA5E9;'>Raw Score      :</span> {socmint:.4f}
            </div>
            <div>
              <div style='color:#F97316;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 4 — ISOLATION FOREST</div>
              <span style='color:{_bot_color};font-weight:bold;'>{_bot_text}</span>
            </div>
            <div>
              <div style='color:#F97316;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>SBERT — SEMANTIC ANALYSIS</div>
              {_sbert_html if _sbert_html else "<span style='color:#475569;'>SBERT outputs not yet received.</span>"}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='panel-box' style='border-left:3px solid #1A3358; margin-bottom:10px; text-align:center;'>
          <div style='color:#475569;font-size:0.73rem;padding:8px;'>
            No bridge data yet. Open the Tactical Simulator in the <b>OVERVIEW</b> tab,
            set URL to <code style='color:#0EA5E9;'>http://localhost:5050/ingest</code> and press <b>SEND</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2, gap="small")

    with c_left:
        # Propaganda gauge
        st.markdown("<div class='panel-title'>◈  PROPAGANDA CONFIDENCE GAUGE</div>", unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=socmint * 100,
            delta={"reference": 40, "valueformat": ".1f", "font": {"size": 11}},
            number={"suffix": "%", "font": {"size": 26, "family": "Courier New", "color": "#EFF6FF"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 8}},
                "bar": {"color": "#F97316", "thickness": 0.25},
                "bgcolor": "#0C1828", "borderwidth": 1, "bordercolor": "#1A3358",
                "steps": [{"range": [0,40], "color": "#022C22"},
                           {"range": [40,65], "color": "#1C1A00"},
                           {"range": [65,85], "color": "#1C0A00"},
                           {"range": [85,100], "color": "#1A0606"}],
                "threshold": {"line": {"color": "#EF4444", "width": 3},
                              "thickness": 0.75, "value": 65},
            },
            title={"text": "Propaganda Probability",
                   "font": {"size": 10, "color": "#22D3EE", "family": "Courier New"}},
        ))
        fig_g.update_layout(paper_bgcolor="#070C18", font_color="#94A3B8",
                            height=210, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_g, width='stretch', config={"displayModeBar": False})

        # Keyword spike
        st.markdown("<div class='panel-title'>◈  KEYWORD SPIKE DETECTION</div>", unsafe_allow_html=True)
        kws = ["border","military","cover-up","attack","infiltration","government","troops","crisis"]
        spk = [847, 620, 480, 390, 310, 280, 210, 180]
        fig_kw = go.Figure(go.Bar(
            y=kws, x=spk, orientation="h",
            marker=dict(color=[f"rgba(239,68,68,{min(v/900,1.0):.2f})" for v in spk]),
            text=[f"+{v}%" for v in spk], textposition="outside",
            textfont=dict(color="#CBD5E1", size=8),
        ))
        fig_kw.update_layout(**dark_layout(height=215, margin=dict(l=10, r=55, t=10, b=20)),
            xaxis={**dark_axes(), "title": "% above baseline"},
            yaxis={**dark_axes(), "showgrid": False},
        )
        st.plotly_chart(fig_kw, width='stretch', config={"displayModeBar": False})

    with c_right:
        # Bot scatter
        st.markdown("<div class='panel-title'>◈  BOT NETWORK PATTERN</div>", unsafe_allow_html=True)
        t_ = np.linspace(0, 4 * np.pi, 200)
        fig_bot = go.Figure()
        fig_bot.add_trace(go.Scatter(
            x=np.random.normal(0, 0.3, 80).tolist(),
            y=np.random.normal(0, 0.3, 80).tolist(),
            mode="markers", marker=dict(size=5, color="#22D3EE", opacity=0.5),
            name="Normal",
        ))
        fig_bot.add_trace(go.Scatter(
            x=(1.5 * np.cos(t_) + np.random.normal(0, 0.15, 200)).tolist(),
            y=(1.5 * np.sin(t_) + np.random.normal(0, 0.15, 200)).tolist(),
            mode="markers",
            marker=dict(size=6, color="#EF4444", opacity=0.8, symbol="diamond"),
            name="Bot ⚠",
        ))
        fig_bot.update_layout(**dark_layout(height=215, showlegend=True,
            legend=dict(bgcolor="#0C1828", bordercolor="#1A3358", font=dict(size=8))),
            xaxis=dark_axes(), yaxis=dark_axes(),
        )
        st.plotly_chart(fig_bot, width='stretch', config={"displayModeBar": False})

        # Flagged posts table
        st.markdown("<div class='panel-title'>◈  FLAGGED POST FEED</div>", unsafe_allow_html=True)
        df_p = generate_posts()
        st.dataframe(
            df_p[df_p["Label"] != "normal"][["Time","Post","Confidence","Bots"]].head(8),
            width='stretch', height=215, hide_index=True,
        )

    # Sentiment timeline
    st.markdown("<div class='panel-title'>◈  SENTIMENT TIMELINE — 24H</div>", unsafe_allow_html=True)
    hrs24 = [f"{h:02d}:00" for h in range(24)]
    prop  = np.clip([.10 + .04*i + .3*np.exp((i-18)/4) + np.random.normal(0,.02) for i in range(24)], 0, 1).tolist()
    norm_ = [max(0, 1-p-.1) for p in prop]
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=hrs24, y=prop, name="Propaganda",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line=dict(color="#EF4444", width=2)))
    fig_s.add_trace(go.Scatter(x=hrs24, y=norm_, name="Normal",
        fill="tozeroy", fillcolor="rgba(34,211,238,0.06)", line=dict(color="#22D3EE", width=1.5)))
    fig_s.update_layout(**dark_layout(height=155, showlegend=True,
        legend=dict(bgcolor="#0C1828", bordercolor="#1A3358", font=dict(size=8),
                    orientation="h", yanchor="bottom", y=1.02)),
        xaxis={**dark_axes(), "showgrid": False},
        yaxis={**dark_axes(), "range": [0, 1.1]},
    )
    st.plotly_chart(fig_s, width='stretch', config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# TAB 3 — CYBINT
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='panel-title'>◈  CYBER INTELLIGENCE — LightGBM M5 · Autoencoder M6 (Zero-Day)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CYBINT SCORE",   f"{cyber:.3f}",  delta=_cybint_label)
    with c2: st.metric("CYBER EVENTS",   str(_n_cyber_events) if _n_cyber_events is not None else "—",
                        delta="LIVE" if _live else "NO FEED")
    with c3: st.metric("ZERO-DAY FLAG",  "⚠ DETECTED" if _zeroday_flag else "CLEAR",
                        delta=f"AE MSE: {_ae_mse:.4f}" if _ae_mse is not None else "AE MSE: —")
    with c4: st.metric("M5 LABEL",       _model5_label,   delta="LightGBM")

    # ── Real model output panel ───────────────────────────────────────────────
    if _live:
        _zd_color = "#EF4444" if _zeroday_flag else "#10B981"
        _zd_text  = "⚠ ZERO-DAY ANOMALY — MSE EXCEEDS THRESHOLD (0.0846)" if _zeroday_flag else "✓ NORMAL — MSE within threshold (0.0846)"
        st.markdown(f"""
        <div class='panel-box' style='border-left:3px solid #EF4444; margin-bottom:10px;'>
          <div class='panel-title'>◈  MODEL OUTPUTS  ·  M5 LightGBM + M6 Autoencoder</div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;font-size:0.73rem;margin-top:8px;'>
            <div>
              <div style='color:#EF4444;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 5 — LIGHTGBM (60 FEATURES)</div>
              <span style='color:#0EA5E9;'>Threat Label   :</span> <b style='color:#EFF6FF;'>{_model5_label}</b><br/>
              <span style='color:#0EA5E9;'>Raw Score      :</span> {cyber:.4f}<br/>
              <span style='color:#0EA5E9;'>Intrusions     :</span> {_n_intrusions}&nbsp;&nbsp;
              <span style='color:#0EA5E9;'>Jamming :</span> {_n_jamming}&nbsp;&nbsp;
              <span style='color:#0EA5E9;'>Malware :</span> {_n_malware}
            </div>
            <div>
              <div style='color:#EF4444;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 6 — AUTOENCODER (ZERO-DAY)</div>
              <span style='color:{_zd_color};font-weight:bold;'>{_zd_text}</span><br/><br/>
              <span style='color:#0EA5E9;'>AE MSE         :</span> {f"{_ae_mse:.6f}" if _ae_mse is not None else "—"}<br/>
              <span style='color:#0EA5E9;'>Threshold      :</span> 0.084596
            </div>
            <div>
              <div style='color:#EF4444;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>RAW CYBINT COUNTS</div>
              <span style='color:#0EA5E9;'>Total Events   :</span> {_n_cyber_events if _n_cyber_events is not None else "—"}<br/>
              <span style='color:#0EA5E9;'>Intrusions     :</span> {_n_intrusions}<br/>
              <span style='color:#0EA5E9;'>Jamming        :</span> {_n_jamming}<br/>
              <span style='color:#0EA5E9;'>Malware        :</span> {_n_malware}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='panel-box' style='border-left:3px solid #1A3358; margin-bottom:10px; text-align:center;'>
          <div style='color:#475569;font-size:0.73rem;padding:8px;'>
            No bridge data yet. Open the Tactical Simulator and send a payload to see live model outputs.
          </div>
        </div>
        """, unsafe_allow_html=True)

    cl_, cr_ = st.columns(2, gap="small")

    with cl_:
        # Isolation Forest scatter
        st.markdown("<div class='panel-title'>◈  ISOLATION FOREST — ANOMALY SCATTER</div>", unsafe_allow_html=True)
        nx = np.random.normal(0, 0.4, 120).tolist()
        ny = np.random.normal(0, 0.4, 120).tolist()
        ax_ = [x + (1.5 if x > 0 else -1.5) for x in np.random.normal(0, 0.3, 12).tolist()]
        ay_ = [y + (1.5 if y > 0 else -1.5) for y in np.random.normal(0, 0.3, 12).tolist()]
        fig_iso = go.Figure()
        fig_iso.add_trace(go.Scatter(x=nx, y=ny, mode="markers",
            marker=dict(size=5, color="#22D3EE", opacity=0.5), name="Normal"))
        fig_iso.add_trace(go.Scatter(x=ax_, y=ay_, mode="markers",
            marker=dict(size=9, color="#EF4444", opacity=0.9, symbol="x",
                        line=dict(width=1, color="white")), name="Anomaly ⚠"))
        fig_iso.update_layout(**dark_layout(height=255, showlegend=True,
            legend=dict(bgcolor="#0C1828", bordercolor="#1A3358", font=dict(size=8))),
            xaxis={**dark_axes(), "title": "Feature 1"},
            yaxis={**dark_axes(), "title": "Feature 2"},
        )
        st.plotly_chart(fig_iso, width='stretch', config={"displayModeBar": False})

        # Traffic timeline
        st.markdown("<div class='panel-title'>◈  NETWORK TRAFFIC ANOMALY TIMELINE</div>", unsafe_allow_html=True)
        hrs = list(range(24))
        trf = [random.randint(10, 60) for _ in range(20)] + [random.randint(400, 900) for _ in range(4)]
        random.shuffle(trf)
        fig_tr = go.Figure(go.Bar(
            x=hrs, y=trf,
            marker=dict(color=["#EF4444" if v > 200 else "#22D3EE" for v in trf], opacity=0.8),
        ))
        fig_tr.add_hline(y=200, line_dash="dot", line_color="#F59E0B", line_width=1.2,
                         annotation_text="Threshold", annotation_font_size=8,
                         annotation_font_color="#F59E0B")
        fig_tr.update_layout(**dark_layout(height=195),
            xaxis={**dark_axes(), "showgrid": False, "title": "Hour (UTC)"},
            yaxis={**dark_axes(), "title": "MB"},
        )
        st.plotly_chart(fig_tr, width='stretch', config={"displayModeBar": False})

    with cr_:
        # Login heatmap
        st.markdown("<div class='panel-title'>◈  LOGIN ANOMALY HEATMAP  (Hour × Day)</div>", unsafe_allow_html=True)
        days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        hrs2  = [f"{h:02d}h" for h in range(0, 24, 2)]
        z_    = np.random.randint(2, 30, (7, 12))
        z_[2,9] = 340; z_[3,10] = 280; z_[4,8] = 210
        fig_hm = go.Figure(go.Heatmap(
            z=z_, x=hrs2, y=days,
            colorscale=[[0,"#070C18"],[0.3,"#0C1828"],[0.6,"#F59E0B"],[0.8,"#F97316"],[1,"#EF4444"]],
            showscale=True,
            colorbar=dict(tickfont=dict(size=7, color="#94A3B8"),
                          outlinecolor="#1A3358", outlinewidth=1, thickness=10),
            hovertemplate="Day:%{y} Hour:%{x} Attempts:%{z}<extra></extra>",
        ))
        fig_hm.update_layout(**dark_layout(height=215, margin=dict(l=30, r=10, t=10, b=30)),
            xaxis=dict(tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_hm, width='stretch', config={"displayModeBar": False})

        # Cyber log table
        st.markdown("<div class='panel-title'>◈  LIVE ANOMALY LOG FEED</div>", unsafe_allow_html=True)
        df_l = generate_cyber_logs()
        st.dataframe(
            df_l[df_l["Status"] == "ANOMALY ⚠"].head(10),
            width='stretch', height=235, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════
# TAB 4 — GEOINT
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='panel-title'>◈  GEOSPATIAL INTELLIGENCE — XGBoost M1 (Classifier) · BiLSTM M2 (Temporal)</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("GEOINT SCORE",    f"{geoint:.3f}",  delta=_geoint_label)
    with c2: st.metric("TOTAL UNITS",     str(_n_total_units) if _n_total_units is not None else "—",
                        delta="LIVE" if _live else "NO FEED")
    with c3: st.metric("M1 LABEL",        _model1_label,    delta="XGBoost 99.97%")
    with c4: st.metric("LSTM TRAJECTORY", _lstm_trajectory if _lstm_trajectory else "—",
                        delta="BiLSTM M2" if _lstm_trajectory else "Async / Pending")

    # ── Real model output panel ───────────────────────────────────────────────
    if _live:
        _traj_color = "#EF4444" if _lstm_trajectory in ("ESCALATING","CRITICAL") else "#F59E0B" if _lstm_trajectory == "RISING" else "#10B981"
        st.markdown(f"""
        <div class='panel-box' style='border-left:3px solid #F59E0B; margin-bottom:10px;'>
          <div class='panel-title'>◈  MODEL OUTPUTS  ·  M1 XGBoost + M2 BiLSTM Temporal</div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;font-size:0.73rem;margin-top:8px;'>
            <div>
              <div style='color:#F59E0B;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 1 — XGBOOST (21 FEATURES)</div>
              <span style='color:#0EA5E9;'>Geo Label      :</span> <b style='color:#EFF6FF;'>{_model1_label}</b><br/>
              <span style='color:#0EA5E9;'>Raw Score      :</span> {geoint:.4f}<br/>
              <span style='color:#0EA5E9;'>Total Units    :</span> {_n_total_units if _n_total_units is not None else "—"}
            </div>
            <div>
              <div style='color:#F59E0B;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>MODEL 2 — BiLSTM TEMPORAL</div>
              <span style='color:#0EA5E9;'>Trajectory     :</span>
              <b style='color:{_traj_color};'>{_lstm_trajectory if _lstm_trajectory else "Awaiting 20-step buffer"}</b><br/>
              <span style='color:#475569;font-size:0.68rem;'>Input: 10-step × 58-feature window  ·  TemporalAttention(64)</span>
            </div>
            <div>
              <div style='color:#F59E0B;font-size:0.67rem;letter-spacing:2px;margin-bottom:6px;'>UNIT STATS</div>
              <span style='color:#0EA5E9;'>Border Dist    :</span> {f"{_border_dist:.2f} km" if _border_dist is not None else "—"}<br/>
              <span style='color:#0EA5E9;'>Mean Velocity  :</span> {f"{_mean_velocity:.1f} km/h" if _mean_velocity is not None else "—"}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='panel-box' style='border-left:3px solid #1A3358; margin-bottom:10px; text-align:center;'>
          <div style='color:#475569;font-size:0.73rem;padding:8px;'>
            No bridge data yet. Open the Tactical Simulator and send a payload to see live model outputs.
          </div>
        </div>
        """, unsafe_allow_html=True)

    gm_col, gr_col = st.columns([6, 4], gap="small")

    with gm_col:
        st.markdown("<div class='panel-title'>◈  GEOSPATIAL THREAT MAP — LIVE UNIT POSITIONS + CLUSTERS</div>", unsafe_allow_html=True)
        clat, clon = ZONES[st.session_state.zone]

        # Re-center on real units when live
        if _live and _geo_units:
            _lats2 = [u["lat"] for u in _geo_units if "lat" in u]
            _lons2 = [u["lng"] for u in _geo_units if "lng" in u]
            if _lats2:
                clat = sum(_lats2) / len(_lats2)
                clon = sum(_lons2) / len(_lons2)

        m2 = folium.Map(location=[clat, clon], zoom_start=9,
                        tiles="CartoDB dark_matter", prefer_canvas=True)

        if _live and _geo_units:
            # ── LIVE path ─────────────────────────────────────────────────────

            # Heatmap weighted by velocity (faster = hotter)
            _heat2 = []
            for u in _geo_units:
                _vel2 = u.get("velocity_kmh", 5)
                _w    = min(0.3 + _vel2 / 20.0, 1.0)
                _heat2.append([u["lat"], u["lng"], _w])
            if _heat2:
                HeatMap(_heat2, radius=30, blur=22, max_zoom=12,
                    gradient={"0.12":"#22D3EE","0.4":"#F59E0B","0.7":"#F97316","1.0":"#EF4444"},
                    name="Threat Heatmap").add_to(m2)

            # Per-type clusters
            _UNIT_COLOR2 = {"tank":"#EF4444","troops":"#F97316","drone":"#F59E0B","vehicle":"#22D3EE"}
            _UNIT_ICON2  = {"tank":"▲","troops":"●","drone":"◆","vehicle":"■"}
            mc = MarkerCluster(name="Unit Clusters").add_to(m2)
            for u in _geo_units:
                _ut2  = u.get("type","unit")
                _uc2  = _UNIT_COLOR2.get(_ut2, "#EF4444")
                _ui2  = _UNIT_ICON2.get(_ut2, "●")
                _vel2 = u.get("velocity_kmh", 0)
                folium.CircleMarker(
                    [u["lat"], u["lng"]], radius=8,
                    color=_uc2, fill=True, fill_opacity=0.88,
                    popup=folium.Popup(
                        f"<div style='background:#120606;color:{_uc2};"
                        f"font-family:monospace;font-size:11px;padding:8px;"
                        f"border:1px solid {_uc2};'>"
                        f"<b>{_ui2} {_ut2.upper()}</b><br/>"
                        f"ID: {u.get('id','—')}<br/>"
                        f"Velocity: {_vel2} km/h<br/>"
                        f"Lat: {u['lat']:.4f}  Lon: {u['lng']:.4f}<br/>"
                        f"M1 Label: {_model1_label}</div>",
                        max_width=210),
                    tooltip=f"{_ui2} {_ut2} · {_vel2} km/h",
                ).add_to(mc)

            # SOCINT alert overlays on GEOINT map
            for a in _soc_alerts:
                _sev2  = a.get("severity","medium")
                _sc2_  = "#EF4444" if _sev2 == "high" else "#F59E0B" if _sev2 == "medium" else "#10B981"
                folium.CircleMarker(
                    [a["lat"], a["lng"]], radius=5,
                    color=_sc2_, fill=True, fill_opacity=0.55,
                    tooltip=f"SOCINT · {a.get('type','alert')} · {_sev2}",
                ).add_to(m2)

        else:
            # ── Fallback: random geo points ────────────────────────────────────
            geo_pts2 = generate_geo_points(st.session_state.zone, n_threat=45, n_normal=150)
            HeatMap(
                [[p["lat"], p["lon"], 1.0 if p["threat"] else 0.12] for p in geo_pts2],
                radius=30, blur=22, max_zoom=12,
                gradient={"0.12":"#22D3EE","0.4":"#F59E0B","0.7":"#F97316","1.0":"#EF4444"},
                name="Threat Heatmap").add_to(m2)
            mc = MarkerCluster(name="Threat Clusters").add_to(m2)
            for p in geo_pts2:
                if p["threat"] == 1:
                    folium.CircleMarker(
                        [p["lat"], p["lon"]], radius=7,
                        color="#EF4444", fill=True, fill_opacity=0.9,
                        popup=folium.Popup(
                            f"<div style='background:#120606;color:#FCA5A5;"
                            f"font-family:monospace;font-size:11px;padding:8px;"
                            f"border:1px solid #EF4444;'>"
                            f"<b>⚠ CLUSTER ALPHA-7</b><br/>"
                            f"Count: {p['count']}<br/>Speed: {p['speed']} km/h<br/>"
                            f"Pattern: STAGING<br/>Distance: 1.2 km</div>",
                            max_width=200),
                        tooltip="⚠ Military Cluster",
                    ).add_to(mc)
            for p in geo_pts2:
                if p["threat"] == 0:
                    folium.CircleMarker(
                        [p["lat"], p["lon"]], radius=3,
                        color="#22D3EE", fill=True, fill_opacity=0.3,
                        tooltip="Normal",
                    ).add_to(m2)
            # Fallback vehicle icons
            for i in range(5):
                folium.Marker(
                    [clat - 0.15 + i*0.07, clon - 0.4 + i*0.15],
                    icon=folium.DivIcon(html="<div style='color:#F59E0B;font-size:14px;'>▲</div>"),
                    tooltip="Military Vehicle — Moving",
                ).add_to(m2)

        # Always-on decorations
        for r_, op_ in [(2000,0.45),(4000,0.28),(6000,0.14),(8000,0.07)]:
            folium.Circle([clat, clon], radius=r_, color="#EF4444",
                          fill=False, opacity=op_, weight=1.5).add_to(m2)
        folium.Circle([clat, clon], radius=5500, color="#EF4444",
                      fill=True, fill_opacity=0.04, weight=2,
                      dash_array="6", name="DBSCAN Boundary").add_to(m2)
        folium.Marker([clat, clon], icon=folium.DivIcon(html=(
            "<div style='color:#EF4444;font-size:26px;font-weight:bold;"
            "text-shadow:0 0 10px #EF4444;margin-left:-13px;margin-top:-13px;'>✛</div>"
        )), tooltip=f"⚠ {st.session_state.zone}").add_to(m2)
        folium.PolyLine(
            [[clat+0.25, clon-2.5], [clat+0.25, clon+2.5]],
            color="#EF4444", weight=2.5, dash_array="12",
            opacity=0.8, name="Border Line",
        ).add_to(m2)
        MiniMap(tile_layer="CartoDB dark_matter",
                position="bottomright", width=110, height=75).add_to(m2)
        folium.LayerControl().add_to(m2)

        st.iframe(m2._repr_html_(), height=490)

    with gr_col:
        # Cluster formation timeline
        st.markdown("<div class='panel-title'>◈  CLUSTER FORMATION TIMELINE</div>", unsafe_allow_html=True)
        px_ = ["06:00","08:00","10:00","12:00","14:00","16:00","18:00","20:00","22:00","00:00"]
        oc_ = [3, 5, 8, 12, 18, 28, 38, 43, 43, 43]
        sp_ = [1.8, 1.5, 1.2, 0.9, 0.7, 0.5, 0.3, 0.15, 0.12, 0.10]
        fig_cl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        fig_cl.add_trace(go.Scatter(x=px_, y=oc_, mode="lines+markers",
            line=dict(color="#EF4444", width=2), marker=dict(size=5, color="#EF4444"),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"), row=1, col=1)
        fig_cl.add_trace(go.Scatter(x=px_, y=sp_, mode="lines+markers",
            line=dict(color="#F59E0B", width=2), marker=dict(size=5, color="#F59E0B")), row=2, col=1)
        fig_cl.update_layout(paper_bgcolor="#070C18", plot_bgcolor="#0C1828",
            font=dict(family="Courier New", color="#94A3B8", size=8),
            margin=dict(l=40, r=10, t=10, b=20), height=215, showlegend=False)
        for r_ in [1, 2]:
            fig_cl.update_xaxes(showgrid=False, tickfont=dict(size=7), row=r_, col=1)
            fig_cl.update_yaxes(showgrid=True, gridcolor="#0D1F33", tickfont=dict(size=7), row=r_, col=1)
        st.plotly_chart(fig_cl, width='stretch', config={"displayModeBar": False})

        # Zone risk summary
        st.markdown("<div class='panel-title'>◈  MULTI-ZONE RISK SUMMARY</div>", unsafe_allow_html=True)
        zone_rows = []
        for zn in ZONES:
            zs_ = round(np.random.uniform(0.1, 0.95), 2)
            zl_, _, _ = get_threat_level(zs_)
            zone_rows.append({"Zone": zn, "Score": zs_, "Level": zl_,
                              "Objects": random.randint(2, 50),
                              "Dist (km)": round(random.uniform(0.5, 12), 1)})
        st.dataframe(pd.DataFrame(zone_rows), width='stretch',
                     height=195, hide_index=True)

        # Border proximity
        st.markdown("<div class='panel-title'>◈  BORDER PROXIMITY TREND</div>", unsafe_allow_html=True)
        fig_d = go.Figure(go.Scatter(
            x=["22:00","21:00","20:00","19:00","18:00","17:00","16:00"],
            y=[1.2, 2.5, 3.8, 5.1, 7.4, 9.2, 11.6],
            mode="lines+markers",
            line=dict(color="#EF4444", width=2.5),
            marker=dict(size=6, color="#EF4444"),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
        ))
        fig_d.add_hline(y=2.0, line_dash="dot", line_color="#F59E0B", line_width=1.5,
                        annotation_text="⚠ 2km", annotation_font_size=8,
                        annotation_font_color="#F59E0B")
        fig_d.update_layout(**dark_layout(height=155, margin=dict(l=40, r=20, t=10, b=30)),
            xaxis={**dark_axes(), "showgrid": False},
            yaxis={**dark_axes(), "title": "km", "autorange": "reversed"},
        )
        st.plotly_chart(fig_d, width='stretch', config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# TAB 5 — FUSION ENGINE
# ══════════════════════════════════════════════════════════════

with tab5:
    st.markdown("<div class='panel-title'>◈  THREAT FUSION ENGINE — MODEL 7 XGBoost REGRESSOR (229 TREES)</div>", unsafe_allow_html=True)

    _bst   = st.session_state.get("bridge_state", {})
    _ready = _bst.get("ready", False)
    _ts    = _bst.get("timestamp", None)

    f5_left, f5_right = st.columns([5, 3], gap="small")

    with f5_left:
        st.markdown(f"""
        <div class="terminal-box" style="font-size:0.82rem; line-height:2.2; padding:28px 36px;">
THREAT FUSION  ·  MODEL 7 XGBoost Regressor (229 trees)
─────────────────────────────────────────────────────────────────
{"LIVE — XGBoost prediction from bridge" if _live else "OFFLINE — Weighted fallback (no dispatch yet)"}
{"Last dispatch : " + _ts[11:19] + " UTC" if _ts else "Awaiting first dispatch from simulator"}

INPUT FEATURES  ➜  cyber={cyber:.3f}   socmint={socmint:.3f}   geoint={geoint:.3f}

{"XGBoost OUTPUT  =  " + f"{fused:.3f}" if _live else f"0.40×{cyber:.3f} + 0.30×{socmint:.3f} + 0.30×{geoint:.3f}  =  {fused:.3f}"}

FUSED THREAT SCORE  =  {fused:.3f}
{"SIM THREAT LEVEL   =  " + _sim_threat_level if _sim_threat_level else ""}

THREAT LEVEL  ➜  {level}

METHOD            ➜  {_fusion_method}
ESCALATION        ➜  {_escalation_pattern if _escalation_pattern else "—"}
        </div>
        """, unsafe_allow_html=True)

    with f5_right:
        st.markdown("<div class='panel-title'>◈  FUSED THREAT GAUGE</div>", unsafe_allow_html=True)
        fig_mg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_gauge_fused,
            number={"font": {"size": 28, "family": "Courier New", "color": lc},
                    "valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1], "nticks": 6, "tickfont": {"size": 8}},
                "bar": {"color": lc, "thickness": 0.30},
                "bgcolor": "#0C1828", "borderwidth": 1, "bordercolor": "#1A3358",
                "steps": [
                    {"range": [0.00, 0.40], "color": "#022C22"},
                    {"range": [0.40, 0.65], "color": "#1C1500"},
                    {"range": [0.65, 0.85], "color": "#1C0800"},
                    {"range": [0.85, 1.00], "color": "#1A0000"},
                ],
                "threshold": {"line": {"color": lc, "width": 4},
                              "thickness": 0.85, "value": _gauge_fused},
            },
            title={"text": f"FUSED SCORE<br><span style='font-size:0.9em;color:{lc}'>{level}</span>"
                           + (f"<br><span style='font-size:0.75em;color:#475569'>XGB:{fused:.3f}  SIM:{_sim_threat_level}</span>" if _live and _sim_threat_level else ""),
                   "font": {"size": 10, "color": "#22D3EE", "family": "Courier New"}},
        ))
        fig_mg.update_layout(paper_bgcolor="#070C18", font_color="#94A3B8",
                             height=300, margin=dict(l=20, r=20, t=55, b=10))
        st.plotly_chart(fig_mg, width='stretch', config={"displayModeBar": False})

        # Threat level classification bands
        for lo, hi, lbl, col, act in [
            (0.00, 0.39, "LOW",      "#10B981", "Routine monitoring"),
            (0.40, 0.64, "MEDIUM",   "#F59E0B", "Heightened surveillance"),
            (0.65, 0.84, "HIGH",     "#F97316", "Analyst investigation"),
            (0.85, 1.00, "CRITICAL", "#EF4444", "Immediate escalation"),
        ]:
            active = lbl == level
            st.markdown(f"""
            <div style='background:#0C1828;
                        border-left:{("4px" if active else "1px")} solid {col};
                        border:1px solid {"" if active else "#1A3358"};
                        border-left-color:{col};
                        border-radius:2px; padding:5px 10px; margin-bottom:3px;'>
              <span style='color:{col};font-weight:bold;font-family:Courier New,monospace;
                           font-size:0.72rem;'>{"→ " if active else "  "}{lbl}</span>
              <span style='color:#475569;font-size:0.68rem;'> [{lo:.2f}–{hi:.2f}]</span>
              <span style='color:#64748B;font-size:0.67rem;float:right;'>{act}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — INTEL REPORT
# ══════════════════════════════════════════════════════════════

with tab6:
    st.markdown("<div class='panel-title'>◈  AI INTELLIGENCE REPORT GENERATOR — LLaMA 3 (LOCAL) / GPT-4</div>", unsafe_allow_html=True)
    rp_l, rp_r = st.columns([4, 6], gap="small")

    with rp_l:
        st.markdown("<div class='panel-title'>◈  REPORT PARAMETERS</div>", unsafe_allow_html=True)
        region   = st.selectbox("Region",         list(ZONES.keys()))
        priority = st.selectbox("Priority",       ["URGENT", "HIGH", "ROUTINE"])
        rtype    = st.selectbox("Report Type",    ["THREAT ASSESSMENT","SITUATION REPORT","INTELLIGENCE BRIEF"])
        classify = st.selectbox("Classification", ["CONFIDENTIAL","SECRET","TOP SECRET"])

        st.markdown("<br/>", unsafe_allow_html=True)
        gen_btn = st.button("◈  GENERATE INTELLIGENCE REPORT", width='stretch')
        if gen_btn:
            with st.spinner("LLM generating intelligence brief..."):
                time.sleep(1.2)
                st.session_state["report_ready"] = True

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="panel-box">
          <div class="panel-title">LLM CONFIGURATION</div>
          <div style="line-height:2.1; font-size:0.73rem;">
            <span style="color:#0EA5E9;">Model :</span> LLaMA 3 (Ollama local)<br/>
            <span style="color:#0EA5E9;">Alt   :</span> GPT-4 via OpenAI API<br/>
            <span style="color:#0EA5E9;">Temp  :</span> 0.3  ·  Tokens: 512<br/>
            <span style="color:#0EA5E9;">Cyber :</span> {cyber:.3f}
            <span style="color:#0EA5E9;">  Info:</span> {socmint:.3f}
            <span style="color:#0EA5E9;">  Geo:</span> {geoint:.3f}<br/>
            <span style="color:#0EA5E9;">Score :</span> {fused:.3f}
            <span style="color:{lc};">  {level}</span><br/>
            <span style="color:#0EA5E9;">Feed  :</span> {"LIVE — bridge data" if _live else "FALLBACK — random scores"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='panel-title'>◈  OLLAMA SNIPPET</div>", unsafe_allow_html=True)
        st.code("""
import ollama

response = ollama.chat(
    model="llama3",
    messages=[{
        "role": "user",
        "content": prompt
    }]
)
report = response["message"]["content"]
        """, language="python")

    with rp_r:
        st.markdown("<div class='panel-title'>◈  GENERATED INTELLIGENCE BRIEF</div>", unsafe_allow_html=True)

        # ── Build report — use real model outputs when live ───────────────────
        _report_region = _bstate.get("raw_payload", {}).get("border_region", region) if _live else region
        _active_sc     = _bstate.get("raw_payload", {}).get("active_scenarios", [])
        _sc_line       = ", ".join(_active_sc) if _active_sc else "None detected"

        _cybint_detail = (
            f"Intrusions: {_n_intrusions}  Jamming: {_n_jamming}  Malware: {_n_malware}\n"
            f"{'Zero-day anomaly flagged (AE MSE=' + f'{_ae_mse:.6f}' + ' > 0.0846)' if _zeroday_flag else 'No zero-day anomaly (AE MSE within threshold)'}\n"
            f"LightGBM M5 classification: {_model5_label}"
        ) if _live else (
            "Abnormal network activity detected. Port scanning at\n"
            "4,200 attempts/min. Login spike: +340%. Anomaly\nratio: 9.2% of total network events flagged."
        )

        _socmint_detail = (
            f"Total alerts: {_n_total_alerts}   Bot cluster: {'DETECTED ⚠' if _bot_flag else 'CLEAR'}\n"
            f"Coordination Index (SBERT): {f'{_coordination_idx:.4f}' if _coordination_idx is not None else '—'}\n"
            f"Clusters: {_n_clusters if _n_clusters is not None else '—'}   "
            f"Narrative Drift: {f'{_narrative_drift:.4f}' if _narrative_drift is not None else '—'}\n"
            f"XGBoost M3 label: {_socmint_label}"
        ) if _live else (
            "Coordinated propaganda campaign active. 1,240 bot\n"
            "accounts in synchronized posting. Keyword surge\n+847% above baseline in border-related narratives."
        )

        _geoint_detail = (
            f"Total units: {_n_total_units}   "
            f"Border distance: {f'{_border_dist:.2f} km' if _border_dist is not None else '—'}\n"
            f"Mean velocity: {f'{_mean_velocity:.1f} km/h' if _mean_velocity is not None else '—'}\n"
            f"XGBoost M1 label: {_model1_label}   "
            f"BiLSTM M2 trajectory: {_lstm_trajectory if _lstm_trajectory else 'Pending async'}"
        ) if _live else (
            "Cluster Alpha-7 confirmed. 43 military vehicles\n"
            "within 1.2km of restricted border zone. Speed:\n8.2 km/h — staging pattern. Cluster tightening."
        )

        report = f"""INTELLIGENCE BRIEF — {"LIVE MODEL OUTPUTS" if _live else "SIMULATED DATA"}
════════════════════════════════════════════════════
CLASSIFICATION  :  {classify}
PRIORITY        :  {priority}
REPORT TYPE     :  {rtype}
REGION          :  {_report_region}
DATE / TIME     :  {datetime.now().strftime("%Y-%m-%d  %H:%M:%S  UTC")}
DATA SOURCE     :  {"HYROX Bridge — Real model inference" if _live else "Fallback — Random scores"}
PREPARED BY     :  HYROX AI SYSTEM  |  TEAM I-DASH
════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────
{"Coordinated hybrid warfare activity confirmed across" if fused >= 0.65 else "Elevated activity detected across"}
three intelligence domains in {_report_region}.
Active scenarios: {_sc_line}
Fusion method: {_fusion_method}
Escalation pattern: {_escalation_pattern if _escalation_pattern else "—"}

DOMAIN FINDINGS
───────────────
[CYBINT  {cyber:.3f}  ·  {_model5_label}]
{_cybint_detail}

[SOCMINT {socmint:.3f}  ·  {_socmint_label}]
{_socmint_detail}

[GEOINT  {geoint:.3f}  ·  {_model1_label}]
{_geoint_detail}

THREAT ASSESSMENT
─────────────────
FUSED SCORE    :  {fused:.3f}   (Model 7 XGBoost — 229 trees)
THREAT LEVEL   :  {level}
WEIGHTS        :  {"XGBoost M7 — learned feature importance" if _live else "Fallback weighted 0.40/0.30/0.30"}
ANOMALY FLAGS  :  AE Zero-Day: {"YES ⚠" if _zeroday_flag else "NO"}  ·  IsoForest Bot: {"YES ⚠" if _bot_flag else "NO"}

RECOMMENDED ACTIONS
───────────────────
{"→  ESCALATE TO SENIOR COMMAND IMMEDIATELY" if level == "CRITICAL" else "→  HEIGHTENED SURVEILLANCE PROTOCOL"}
{"→  Activate border surveillance protocols" if level in ("CRITICAL","HIGH") else "→  Continue monitoring all domains"}
{"→  Raise cyber defense posture immediately" if _zeroday_flag else "→  Maintain standard cyber posture"}
→  Monitor all three domains at 5-min intervals

════════════════════════════════════════════════════
END OF REPORT  |  HYROX AI  |  TEAM I-DASH
════════════════════════════════════════════════════"""

        st.markdown(f'<div class="terminal-box">{report}</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

        dl_col, rg_col = st.columns(2)
        with dl_col:
            st.download_button(
                "⬇  DOWNLOAD REPORT (.txt)",
                data=report,
                file_name=f"hyrox_intel_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                width='stretch',
            )
        with rg_col:
            if st.button("⟳  REGENERATE", width='stretch'):
                st.rerun()

# ─────────────────────────────────────────────────────────────
# AUTO-REFRESH ENGINE
# Must be at module level — outside every `with` block so that
# st.rerun() propagates cleanly without being swallowed by a
# context manager's __exit__.
# ─────────────────────────────────────────────────────────────
if st.session_state.get("auto_refresh", False):
    new_scores = generate_scores()
    if new_scores != st.session_state.scores:
        st.session_state.scores = new_scores
        _bs_ar = st.session_state.get("bridge_state", {})
        if _bs_ar.get("ready", False):
            st.session_state.alert_count = (
                _bs_ar.get("raw_payload", {})
                .get("social_intel", {})
                .get("total_alerts", st.session_state.alert_count)
            )
            _append_timeline(new_scores[3])
    time.sleep(1)
    st.rerun()