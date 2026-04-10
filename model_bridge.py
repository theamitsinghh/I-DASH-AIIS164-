"""
╔══════════════════════════════════════════════════════════════════╗
║  model_bridge.py  —  HYROX Full Inference Pipeline              ║
║                                                                  ║
║  Loads all 7 trained models once at startup, then runs the      ║
║  complete inference chain on every incoming simulator payload.   ║
║                                                                  ║
║  INFERENCE CHAIN (per payload, ~50–100 ms sync path)            ║
║    payload → feature_extraction()                               ║
║      → Model 1  (GEOINT XGBoost classifier)                     ║
║      → Model 3  (SOCINT XGBoost classifier)                     ║
║      → Model 4  (SOCINT Isolation Forest anomaly)               ║
║      → Model 5  (CYBINT LightGBM classifier)                    ║
║      → Model 6  (CYBINT Autoencoder zero-day)                   ║
║      → Model 7  (Fusion XGBoost regressor)                      ║
║      [async] Model 2  (BiLSTM temporal trajectory)              ║
║      [async] SBERT embedding pipeline                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── TensorFlow / Metal guards — MUST be set before any tf/keras import ────────
# Prevents segfault on Apple Silicon (M1/M2/M3/M4) caused by the Metal GPU
# plugin initializing before the process is fully ready.
# TF_CPP_MIN_LOG_LEVEL=3  → suppresses all C++ INFO/WARNING/ERROR logs (incl. NUMA)
# CUDA_VISIBLE_DEVICES=""  → disables CUDA (not present on Apple Silicon anyway)
# TF_METAL_DEVICE_PLACEMENT=0 → stops Metal plugin from registering as a device
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",       "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "")
os.environ.setdefault("TF_METAL_DEVICE_PLACEMENT",   "0")
import math
import pickle
import threading
import traceback
import warnings
from collections import deque
from datetime import datetime

# Suppress sklearn "X has feature names but scaler was fitted without" warning.
# Caused by StandardScaler fitted on raw numpy arrays; safe to ignore here
# since we always call .values before transform().
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but .* was fitted without feature names",
    category=UserWarning,
)

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# All paths are relative to the hyrox/ project root.
# Adjust MODEL_ROOT if your working directory differs.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ROOT = os.path.join(os.path.dirname(__file__), "models")

_P = {
    # GEOINT
    "geo_xgb":      os.path.join(MODEL_ROOT, "geoint",         "geoint_xgb_model.pkl"),
    # GEOINT temporal (BiLSTM)
    "lstm":         os.path.join(MODEL_ROOT, "temporal",        "best_model.keras"),
    # SOCINT
    "soc_xgb":      os.path.join(MODEL_ROOT, "socint",         "model3_xgb_socint.json"),
    "soc_feat":     os.path.join(MODEL_ROOT, "socint",         "feature_columns.pkl"),
    "soc_enc":      os.path.join(MODEL_ROOT, "socint",         "label_encoder.pkl"),
    "iso_forest":   os.path.join(MODEL_ROOT, "socint",         "model4_isolation_forest.joblib"),
    # CYBINT
    "cyber_lgbm":   os.path.join(MODEL_ROOT, "cybint",         "model5_lgbm.txt"),
    "ae_model":     os.path.join(MODEL_ROOT, "cybint",         "model6_autoencoder.h5"),
    "ae_scaler":    os.path.join(MODEL_ROOT, "cybint",         "model6_scaler.pkl"),
    "ae_threshold": os.path.join(MODEL_ROOT, "cybint",         "model6_threshold.npy"),
    # Fusion
    "fusion_xgb":   os.path.join(MODEL_ROOT, "fusion engine",  "model7_xgb_fusion.json"),
    # CYBINT feature reference
    "cyber_feat":   os.path.join(os.path.dirname(__file__),
                                 "reference", "feature_columns.txt"),
}

# ── Region encoding: used across all domain feature extractors ────────────────
REGION_ENC = {
    "india-arunachal":  0,
    "india-bangladesh": 1,
    "india-china":      2,
    "india-pak":        3,
    # payload .name strings (long form)
    "India–Arunachal Pradesh":            0,
    "India–Bangladesh Border":            1,
    "India–China LAC":                    2,
    "India–China Line of Actual Control": 2,
    "India–Pakistan Line of Control":     3,
}

# ── Unit-history buffer for BiLSTM (per session_id, 10 steps × 58 features) ──
# Using a plain dict of deques — no Redis needed for single-device demo.
_unit_history: dict[str, deque] = {}
_unit_history_lock = threading.Lock()

# ── Model handles (populated once by load_all_models()) ──────────────────────
_models: dict = {}
_models_loaded: bool = False
_load_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_all_models() -> None:
    """
    Load every model into the _models dict exactly once.
    Called by bridge.start() at server startup.
    Thread-safe via _load_lock.
    """
    global _models_loaded
    with _load_lock:
        if _models_loaded:
            return

        status = {}

        # ── Model 1: GEOINT XGBoost ───────────────────────────────────────────
        # FIX: File was saved with joblib (zlib-compressed), not plain pickle.
        # pickle.load() fails with "invalid load key, 'x'" on zlib streams.
        # Solution: use joblib.load() which handles all compression formats.
        try:
            import joblib  # type: ignore
            bundle = joblib.load(_P["geo_xgb"])
            _models["geo_xgb_model"]   = bundle["model"]
            _models["geo_xgb_scaler"]  = bundle["scaler"]
            _models["geo_xgb_feats"]   = bundle["feature_cols"]
            _models["geo_label_order"] = bundle["label_order"]    # ['LOW','MODERATE','HIGH']
            _models["geo_label_int"]   = bundle["label_int_map"]  # {'LOW':0,'MODERATE':1,'HIGH':2}
            status["M1-GEOINT-XGB"] = True
            print("[HYROX] ✓ Model 1 (GEOINT XGBoost) loaded")
        except Exception as e:
            status["M1-GEOINT-XGB"] = False
            print(f"[HYROX] ✗ Model 1 failed: {e}")

        # ── Model 2: BiLSTM (Temporal GEOINT) ────────────────────────────────
        # CRITICAL: TemporalAttention must be registered BEFORE keras.load_model().
        #
        # ROOT CAUSE of "expected 3 variables, received 0":
        #   Previous implementation used add_weight() → variables named W_a/b_a/v_a.
        #   Training notebook uses two Dense sublayers (self.W, self.v) → variables
        #   named "W/kernel" and "v/kernel". Keras matches weights by variable
        #   structure; mismatched implementation = zero variables restored.
        #   compile=False alone cannot fix an architecture mismatch.
        #
        # FIX: mirror the training notebook exactly —
        #   • Dense sublayers instead of add_weight
        #   • call() returns tuple (context, alpha) — the saved computation graph
        #     was built with this signature; changing it breaks downstream connections
        #   • compile=False still required to skip AdamW variable-count mismatch
        try:
            import tensorflow as tf  # type: ignore
            import keras             # type: ignore

            @keras.saving.register_keras_serializable()
            class TemporalAttention(keras.layers.Layer):
                """
                Additive (Bahdanau-style) temporal attention.
                Mirrors training notebook exactly: Dense sublayers, tuple return.
                """
                def __init__(self, units: int = 64, **kwargs):
                    super().__init__(**kwargs)
                    self.units = units
                    # Dense sublayers — NOT add_weight — to match variable layout
                    # stored in best_model.keras ("W/kernel", "v/kernel").
                    self.W = keras.layers.Dense(units, use_bias=False)
                    self.v = keras.layers.Dense(1,     use_bias=False)

                def build(self, input_shape):
                    # input_shape: (batch, timesteps, hidden_dim)
                    self.W.build(input_shape)
                    # Output of W is (batch, timesteps, self.units)
                    self.v.build((*input_shape[:-1], self.units))
                    super().build(input_shape)

                def call(self, h):
                    # h : (batch, timesteps, hidden_dim)
                    score   = self.v(tf.nn.tanh(self.W(h)))    # (B, T, 1)
                    alpha   = tf.nn.softmax(score, axis=1)     # (B, T, 1)
                    context = tf.reduce_sum(alpha * h, axis=1) # (B, H)
                    # Tuple return matches the saved computation graph signature.
                    return context, tf.squeeze(alpha, -1)

                def get_config(self):
                    cfg = super().get_config()
                    cfg.update({"units": self.units})
                    return cfg

            # Rebuild with tf.keras + load_weights to bypass Keras 3 config mismatch.
            # Architecture mirrors the saved graph exactly (from error log config).
            # TemporalAttention is a TWO-OUTPUT layer: (context, alpha).
            # Keras weight loading matches by layer index — order must be exact.

            class _TA(tf.keras.layers.Layer):
                def __init__(self, units=64, **kw):
                    super().__init__(**kw)
                    self.units = units
                    self.W = tf.keras.layers.Dense(units, use_bias=False, name="W")
                    self.v = tf.keras.layers.Dense(1, use_bias=False, name="v")
                def build(self, s):
                    self.W.build(s)
                    self.v.build((*s[:-1], self.units))
                    super().build(s)
                def call(self, h):
                    score = self.v(tf.nn.tanh(self.W(h)))        # (B,T,1)
                    alpha = tf.nn.softmax(score, axis=1)          # (B,T,1)
                    ctx   = tf.reduce_sum(alpha * h, axis=1)      # (B,H)
                    return ctx, tf.squeeze(alpha, -1)
                def get_config(self):
                    c = super().get_config(); c["units"] = self.units; return c

            def _build_bilstm():
                inp  = tf.keras.Input(shape=(10, 58), name="input")
                x    = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(inp)
                x    = tf.keras.layers.Bidirectional(
                           tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.1,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               recurrent_regularizer=tf.keras.regularizers.l2(1e-4)),
                           name="bilstm_1")(x)
                x    = tf.keras.layers.SpatialDropout1D(0.2, name="sdrop_1")(x)
                x    = tf.keras.layers.LayerNormalization(epsilon=0.001, name="ln_1")(x)
                x    = tf.keras.layers.Bidirectional(
                           tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.1,
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                               recurrent_regularizer=tf.keras.regularizers.l2(1e-4)),
                           name="bilstm_2")(x)
                x    = tf.keras.layers.SpatialDropout1D(0.2, name="sdrop_2")(x)
                ln2  = tf.keras.layers.LayerNormalization(epsilon=0.001, name="ln_2")(x)
                # TemporalAttention returns (context, alpha) — only context used downstream
                ta_out = _TA(units=64, name="temporal_attn")(ln2)
                ctx  = tf.keras.layers.Lambda(lambda t: t[0], name="attn_ctx")(ta_out)
                gap  = tf.keras.layers.GlobalAveragePooling1D(name="gap")(ln2)
                mrg  = tf.keras.layers.Concatenate(name="merge")([ctx, gap])
                x    = tf.keras.layers.Dense(128, activation="gelu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           name="dense_1")(mrg)
                x    = tf.keras.layers.Dropout(0.4, name="drop_3")(x)
                x    = tf.keras.layers.BatchNormalization(momentum=0.99,
                           epsilon=0.001, name="bn_1")(x)
                x    = tf.keras.layers.Dense(64, activation="gelu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           name="dense_2")(x)
                x    = tf.keras.layers.Dropout(0.2, name="drop_4")(x)
                out  = tf.keras.layers.Dense(4, activation="softmax", name="output")(x)
                return tf.keras.Model(inp, out, name="BiLSTM_Attention")

            m2 = _build_bilstm()
            m2.load_weights(_P["lstm"])
            _models["lstm"] = m2
            _models["TemporalAttention"] = _TA
            status["M2-LSTM"] = True
            print("[HYROX] ✓ Model 2 (BiLSTM Temporal) loaded")
        except Exception as e:
            status["M2-LSTM"] = False
            print(f"[HYROX] ✗ Model 2 failed: {e}")

        # ── Model 3: SOCINT XGBoost ───────────────────────────────────────────
        try:
            import xgboost as xgb  # type: ignore
            bst = xgb.Booster()
            bst.load_model(_P["soc_xgb"])
            _models["soc_xgb"] = bst
            with open(_P["soc_feat"], "rb") as f:
                _models["soc_feat_cols"] = pickle.load(f)   # list of 23 feature names
            with open(_P["soc_enc"], "rb") as f:
                _models["soc_label_enc"] = pickle.load(f)   # dict {0:'LOW',...,3:'CRITICAL'}
            status["M3-SOCINT-XGB"] = True
            print("[HYROX] ✓ Model 3 (SOCINT XGBoost) loaded")
        except Exception as e:
            status["M3-SOCINT-XGB"] = False
            print(f"[HYROX] ✗ Model 3 failed: {e}")

        # ── Model 4: SOCINT Isolation Forest ─────────────────────────────────
        try:
            import joblib  # type: ignore
            iso_bundle = joblib.load(_P["iso_forest"])
            # bundle is dict: {'pipeline': Pipeline(StandardScaler→IsoForest),
            #                  'feature_cols': list[41]}
            _models["iso_pipeline"]  = iso_bundle["pipeline"]
            _models["iso_feat_cols"] = iso_bundle["feature_cols"]  # list[41]
            status["M4-ISOFOREST"] = True
            print("[HYROX] ✓ Model 4 (Isolation Forest) loaded")
        except Exception as e:
            status["M4-ISOFOREST"] = False
            print(f"[HYROX] ✗ Model 4 failed: {e}")

        # ── Model 5: CYBINT LightGBM ──────────────────────────────────────────
        try:
            import lightgbm as lgb  # type: ignore
            bst5 = lgb.Booster(model_file=_P["cyber_lgbm"])
            _models["cyber_lgbm"] = bst5
            # Read the 60 feature names from the reference file
            with open(_P["cyber_feat"]) as f:
                _models["cyber_feat_cols"] = [ln.strip() for ln in f if ln.strip()]
            status["M5-CYBINT-LGBM"] = True
            print("[HYROX] ✓ Model 5 (CYBINT LightGBM) loaded")
        except Exception as e:
            status["M5-CYBINT-LGBM"] = False
            print(f"[HYROX] ✗ Model 5 failed: {e}")

        # ── Model 6: CYBINT Autoencoder ───────────────────────────────────────
        # TWO fixes required:
        #
        # FIX A — compile=False (was already in place):
        #   Model compiled with loss="mse" metrics=["mae"]. The bare string
        #   "mse" is no longer a valid metric alias in Keras 3; it was removed.
        #   compile=False skips metric/optimizer deserialization entirely — safe
        #   because this model is inference-only (.predict, never .fit).
        #
        # FIX B — joblib.load for the scaler (root cause of '\x07' error):
        #   model6_scaler.pkl was saved with joblib.dump() in the training notebook.
        #   joblib files start with a protocol-2 zlib header; the first byte is
        #   0x80 (pickle protocol 2 marker) when uncompressed, or a zlib stream
        #   byte ('\x07' with certain compression settings) when compressed.
        #   pickle.load() cannot read joblib-compressed files — use joblib.load().
        try:
            import joblib  # type: ignore
            import keras   # type: ignore
            _models["ae_model"] = keras.models.load_model(
                _P["ae_model"],
                compile=False,   # skip 'mse' metric — removed in Keras 3
            )
            _models["ae_scaler"]    = joblib.load(_P["ae_scaler"])          # FIX B
            _models["ae_threshold"] = float(np.load(_P["ae_threshold"]))
            status["M6-AUTOENCODER"] = True
            print(f"[HYROX] ✓ Model 6 (Autoencoder) loaded — threshold={_models['ae_threshold']:.6f}")
        except Exception as e:
            status["M6-AUTOENCODER"] = False
            print(f"[HYROX] ✗ Model 6 failed: {e}")

        # ── Model 7: Fusion XGBoost Regressor ────────────────────────────────
        try:
            import xgboost as xgb  # type: ignore
            bst7 = xgb.Booster()
            bst7.load_model(_P["fusion_xgb"])
            _models["fusion_xgb"] = bst7
            status["M7-FUSION-XGB"] = True
            print("[HYROX] ✓ Model 7 (Fusion XGBoost) loaded")
        except Exception as e:
            status["M7-FUSION-XGB"] = False
            print(f"[HYROX] ✗ Model 7 failed: {e}")

        # ── Summary ───────────────────────────────────────────────────────────
        loaded = sum(1 for v in status.values() if v)
        _models["model_status"]  = status
        _models["models_loaded"] = f"{loaded}/{len(status)}"
        _models_loaded = True
        print(f"[HYROX] Models ready: {loaded}/{len(status)}")


# ═════════════════════════════════════════════════════════════════════════════
#  TEMPORAL HELPERS (sin/cos encoding)
# ═════════════════════════════════════════════════════════════════════════════

def _sin_cos(value: float, period: float) -> tuple[float, float]:
    """Return (sin, cos) cyclical encoding of value within period."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _temporal_features(ts_str: str) -> dict:
    """
    Parse ISO timestamp and return raw + cyclical temporal features
    used across all domain feature extractors.
    """
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()

    h, dow, mo = dt.hour, dt.weekday(), dt.month
    h_sin,   h_cos   = _sin_cos(h,   24)
    mo_sin,  mo_cos  = _sin_cos(mo,  12)
    dow_sin, dow_cos = _sin_cos(dow,  7)

    return {
        "hour":        h,
        "day_of_week": dow,
        "month":       mo,
        "is_weekend":  int(dow >= 5),
        "hour_sin":    h_sin,
        "hour_cos":    h_cos,
        "month_sin":   mo_sin,
        "month_cos":   mo_cos,
        "dow_sin":     dow_sin,
        "dow_cos":     dow_cos,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — GEOINT (Model 1 · 21 features)
# ═════════════════════════════════════════════════════════════════════════════

# Reference lat/lon for each border region (used for border-distance proxy)
_REGION_REF = {
    0: (27.80, 93.60),   # india-arunachal
    1: (23.80, 91.00),   # india-bangladesh
    2: (33.50, 79.50),   # india-china
    3: (34.10, 74.80),   # india-pak
}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _extract_geoint_features(payload: dict, region_code: int, n_scenarios: int) -> dict:
    """
    Build the 21-feature dict expected by Model 1 (GEOINT XGBoost).
    All 7 engineered features are computed here.
    """
    units = payload.get("geo_intel", {}).get("units", [])

    # Type counts
    type_count: dict[str, int] = {}
    for u in units:
        t = u.get("type", "unknown")
        type_count[t] = type_count.get(t, 0) + 1

    n_tanks    = type_count.get("tank",    0)
    n_troops   = type_count.get("troops",  0)
    n_drones   = type_count.get("drone",   0)
    n_vehicles = type_count.get("vehicle", 0)
    n_total    = len(units)

    # Velocities
    vels     = [u.get("velocity_kmh", 0.0) for u in units] or [0.0]
    mean_vel = float(np.mean(vels))
    max_vel  = float(np.max(vels))
    vel_std  = float(np.std(vels))

    # Spatial: cluster radius and border distance
    ref_lat, ref_lon = _REGION_REF.get(region_code, (30.0, 77.0))
    lats = [u.get("lat", ref_lat) for u in units] or [ref_lat]
    lons = [u.get("lng", ref_lon) for u in units] or [ref_lon]
    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))

    # Cluster radius: mean distance from centroid
    cluster_radius = float(np.mean([
        _haversine_km(lat, lon, mean_lat, mean_lon)
        for lat, lon in zip(lats, lons)
    ])) if len(units) > 1 else 0.0

    # Border distance: distance of cluster centroid from region reference point
    dist_from_border = _haversine_km(mean_lat, mean_lon, ref_lat, ref_lon)

    # Units in 5 km strip around border
    units_in_5km = sum(
        1 for lat, lon in zip(lats, lons)
        if _haversine_km(lat, lon, ref_lat, ref_lon) <= 5.0
    )

    # Ratios
    heavy_armor_ratio = n_tanks  / max(n_total, 1)
    air_ratio         = n_drones / max(n_total, 1)

    # ── 7 engineered features ────────────────────────────────────────────────
    proximity_score     = 1.0 / (1.0 + dist_from_border)
    speed_border_index  = mean_vel * proximity_score
    armor_border_index  = heavy_armor_ratio * proximity_score
    total_threat_signal = (
        n_tanks * 3 + n_troops * 1 + n_drones * 2 + n_vehicles * 1.5
    ) * proximity_score
    velocity_spread     = vel_std / max(mean_vel, 1.0)
    strip_density       = units_in_5km / max(n_total, 1)
    # scenario_multiplier: 1.0 when no scenario, scales up to ~3.0
    scenario_multiplier = 1.0 + (n_scenarios * 0.2)

    return {
        "n_tanks":             n_tanks,
        "n_troops":            n_troops,
        "n_drones":            n_drones,
        "n_vehicles":          n_vehicles,
        "n_total_units":       n_total,
        "mean_velocity_kmh":   mean_vel,
        "max_velocity_kmh":    max_vel,
        "velocity_std":        vel_std,
        "cluster_radius_km":   cluster_radius,
        "dist_from_border_km": dist_from_border,
        "units_in_5km_strip":  units_in_5km,
        "heavy_armor_ratio":   heavy_armor_ratio,
        "air_ratio":           air_ratio,
        "n_active_scenarios":  n_scenarios,
        # 7 engineered
        "proximity_score":     proximity_score,
        "speed_border_index":  speed_border_index,
        "armor_border_index":  armor_border_index,
        "total_threat_signal": total_threat_signal,
        "velocity_spread":     velocity_spread,
        "strip_density":       strip_density,
        "scenario_multiplier": scenario_multiplier,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — SOCINT (Model 3 · 23 features)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_socint_features(
    payload: dict, region_code: int, n_scenarios: int, temp: dict
) -> dict:
    """
    Build the 23-feature dict expected by Model 3 (SOCINT XGBoost).
    """
    ref_lat, ref_lon = _REGION_REF.get(region_code, (30.0, 77.0))
    alerts = payload.get("social_intel", {}).get("alerts", [])

    # Type counts
    n_protests   = sum(1 for a in alerts if a.get("type") == "protest")
    n_propaganda = sum(1 for a in alerts if a.get("type") == "propaganda")
    n_informants = sum(1 for a in alerts if a.get("type") == "informant")
    n_total      = len(alerts)

    # Severity counts (severity field: 'low', 'medium', 'high', 'critical')
    SEV_SCORE = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    sev_scores = [SEV_SCORE.get(a.get("severity", "low").lower(), 1) for a in alerts]

    n_high_sev = sum(1 for s in sev_scores if s >= 3)
    n_med_sev  = sum(1 for s in sev_scores if s == 2)
    n_low_sev  = sum(1 for s in sev_scores if s == 1)

    high_sev_ratio   = n_high_sev   / max(n_total, 1)
    propaganda_ratio = n_propaganda / max(n_total, 1)
    informant_ratio  = n_informants / max(n_total, 1)
    mean_sev_score   = float(np.mean(sev_scores)) if sev_scores else 1.0

    # Alerts near border: within 30 km of region reference point
    alerts_near_border = sum(
        1 for a in alerts
        if _haversine_km(
            a.get("lat", ref_lat), a.get("lng", ref_lon), ref_lat, ref_lon
        ) <= 30.0
    )

    # ── Engineered interaction features ──────────────────────────────────────
    weighted_sev_ratio = (n_high_sev * 3 + n_med_sev * 2 + n_low_sev) / max(n_total, 1)
    alert_pressure     = n_total * mean_sev_score
    high_sev_x_border  = high_sev_ratio * (alerts_near_border / max(n_total, 1))
    prop_x_informant   = propaganda_ratio * informant_ratio
    # scenario_amplifier: boosts score in presence of active scenarios
    scenario_amplifier = 1.0 + (n_scenarios * 0.25)

    return {
        "n_protests":              n_protests,
        "n_propaganda":            n_propaganda,
        "n_informants":            n_informants,
        "n_total_alerts":          n_total,
        "n_high_sev":              n_high_sev,
        "n_med_sev":               n_med_sev,
        "n_low_sev":               n_low_sev,
        "high_sev_ratio":          high_sev_ratio,
        "propaganda_ratio":        propaganda_ratio,
        "informant_ratio":         informant_ratio,
        "mean_severity_score":     mean_sev_score,
        "alerts_near_border":      alerts_near_border,
        "n_active_scenarios":      n_scenarios,
        # temporal (from shared _temporal_features)
        "hour":                    temp["hour"],
        "day_of_week":             temp["day_of_week"],
        "is_weekend":              temp["is_weekend"],
        "month":                   temp["month"],
        # region
        "region_enc":              region_code,
        # engineered
        "weighted_sev_ratio":      weighted_sev_ratio,
        "alert_pressure":          alert_pressure,
        "high_sev_x_border":       high_sev_x_border,
        "propaganda_x_informant":  prop_x_informant,
        "scenario_amplifier":      scenario_amplifier,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — ISO FOREST (Model 4 · 41 features)
#  NOTE: socint_score from Model 3 must be included as feature 41.
# ═════════════════════════════════════════════════════════════════════════════

def _extract_isoforest_features(
    socint_feats: dict, temp: dict, region_code: int, socint_score: float
) -> pd.DataFrame:
    """
    Build the 41-feature DataFrame for Model 4 (Isolation Forest).
    The last feature is socint_score computed from Model 3.
    """
    # Derived entropy-like features
    n_total = max(socint_feats.get("n_total_alerts", 0), 1)
    prots   = socint_feats.get("n_protests",   0) / n_total
    props   = socint_feats.get("n_propaganda", 0) / n_total
    infs    = socint_feats.get("n_informants", 0) / n_total

    def _entropy(p_list):
        """Shannon entropy of a probability distribution."""
        total = sum(p_list)
        if total == 0:
            return 0.0
        ps = [p / total for p in p_list if p > 0]
        return -sum(p * math.log2(p) for p in ps)

    type_entropy     = _entropy([prots, props, infs])
    severity_entropy = _entropy([
        socint_feats.get("n_low_sev",  0),
        socint_feats.get("n_med_sev",  0),
        socint_feats.get("n_high_sev", 0),
    ])

    border_density = socint_feats.get("alerts_near_border", 0) / n_total

    row = {
        # Cyclical temporal (6)
        "hour_sin":          temp.get("hour_sin",    0.0),
        "hour_cos":          temp.get("hour_cos",    1.0),
        "month_sin":         temp.get("month_sin",   0.0),
        "month_cos":         temp.get("month_cos",   1.0),
        "dow_sin":           temp.get("dow_sin",     0.0),
        "dow_cos":           temp.get("dow_cos",     1.0),
        # Raw temporal (3)
        "day_of_week":       temp.get("day_of_week", 0),
        "is_weekend":        temp.get("is_weekend",  0),
        # Region (2)
        "region_enc":        region_code,
        "alerts_near_border":socint_feats.get("alerts_near_border", 0),
        "border_density":    border_density,
        # Type counts (3)
        "n_protests":        socint_feats.get("n_protests",   0),
        "n_propaganda":      socint_feats.get("n_propaganda", 0),
        "n_informants":      socint_feats.get("n_informants", 0),
        "n_total_alerts":    socint_feats.get("n_total_alerts", 0),
        # Severity (5)
        "n_high_sev":        socint_feats.get("n_high_sev",         0),
        "n_med_sev":         socint_feats.get("n_med_sev",          0),
        "n_low_sev":         socint_feats.get("n_low_sev",          0),
        "high_sev_ratio":    socint_feats.get("high_sev_ratio",     0.0),
        "mean_severity_score":socint_feats.get("mean_severity_score", 1.0),
        # Ratios (3)
        "propaganda_ratio":  socint_feats.get("propaganda_ratio",   0.0),
        "informant_ratio":   socint_feats.get("informant_ratio",    0.0),
        "weighted_sev_ratio":socint_feats.get("weighted_sev_ratio", 0.0),
        # Entropy (2)
        "type_entropy":      type_entropy,
        "severity_entropy":  severity_entropy,
        # Interactions (6)
        "alert_pressure":          socint_feats.get("alert_pressure",         0.0),
        "high_sev_x_border":       socint_feats.get("high_sev_x_border",      0.0),
        "propaganda_x_informant":  socint_feats.get("propaganda_x_informant", 0.0),
        "scenario_amplifier":      socint_feats.get("scenario_amplifier",      1.0),
        "n_active_scenarios":      socint_feats.get("n_active_scenarios",      0),
        # Protest interaction (3)
        "protest_x_prop":    prots * props,
        "protest_x_border":  prots * border_density,
        "sev_x_scenarios":   socint_feats.get("mean_severity_score", 1.0) * socint_feats.get("n_active_scenarios", 0),
        # Month raw (1)
        "month":             temp.get("month", 1),
        # Log transforms (4)
        "log_n_total":    math.log1p(socint_feats.get("n_total_alerts",  0)),
        "log_n_prop":     math.log1p(socint_feats.get("n_propaganda",    0)),
        "log_n_high_sev": math.log1p(socint_feats.get("n_high_sev",      0)),
        "log_alert_pres": math.log1p(socint_feats.get("alert_pressure",  0.0)),
        # MUST be last — socint_score from Model 3 (feature 41)
        "socint_score":   socint_score,
    }

    df = pd.DataFrame([row])

    # Reorder to match training column order if feature_cols is available
    feat_cols = _models.get("iso_feat_cols")
    if feat_cols:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feat_cols]

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION — CYBINT (Model 5 · 60 features + Model 6 · 10 features)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_cybint_features(
    payload: dict, region_code: int, n_scenarios: int, temp: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Build the 60-feature DataFrame for Model 5 (CYBINT LightGBM).
    Also returns the 10-feature subset dict for the autoencoder (Model 6).
    """
    events   = payload.get("cyber_intel", {}).get("alerts", [])  # HTML sends 'alerts' (fixed)
    ref_lat, ref_lon = _REGION_REF.get(region_code, (30.0, 77.0))

    n_intrusions = sum(1 for e in events if e.get("type") == "intrusion")
    n_jamming    = sum(1 for e in events if e.get("type") == "jamming")
    n_malware    = sum(1 for e in events if e.get("type") == "malware")
    n_total      = len(events)

    # Weighted cyber signal (from HYROX context spec)
    weighted_cyber_signal = n_intrusions * 2.0 + n_jamming * 1.5 + n_malware * 2.5

    # Events near border (within 30 km)
    events_near_border = sum(
        1 for e in events
        if _haversine_km(
            e.get("lat", ref_lat), e.get("lng", ref_lon), ref_lat, ref_lon
        ) <= 30.0
    )

    # Ratios
    intrusion_ratio = n_intrusions / max(n_total, 1)
    jamming_ratio   = n_jamming   / max(n_total, 1)
    malware_ratio   = n_malware   / max(n_total, 1)
    border_ratio    = events_near_border / max(n_total, 1)

    # Log transforms
    log_intrusions = math.log1p(n_intrusions)
    log_jamming    = math.log1p(n_jamming)
    log_malware    = math.log1p(n_malware)
    log_signal     = math.log1p(weighted_cyber_signal)
    log_total      = math.log1p(n_total)

    # Interaction terms
    intrusion_x_malware = intrusion_ratio * malware_ratio
    jamming_x_border    = jamming_ratio   * border_ratio
    signal_x_scenarios  = weighted_cyber_signal * n_scenarios
    malware_x_border    = malware_ratio   * border_ratio
    intrusion_x_border  = intrusion_ratio * border_ratio

    # Region target encodings (pseudo-values; replace with actual TE from training if available)
    REGION_TE = {
        0: [0.10, 0.40, 0.30, 0.20],   # arunachal  cls0–3
        1: [0.15, 0.35, 0.35, 0.15],   # bangladesh
        2: [0.12, 0.38, 0.32, 0.18],   # china
        3: [0.08, 0.30, 0.38, 0.24],   # pak
    }
    rte = REGION_TE.get(region_code, [0.25, 0.25, 0.25, 0.25])

    row = {
        # Base counts (6)
        "n_intrusions":          n_intrusions,
        "n_jamming":             n_jamming,
        "n_malware":             n_malware,
        "weighted_cyber_signal": weighted_cyber_signal,
        "events_near_border":    events_near_border,
        "n_active_scenarios":    n_scenarios,
        # Temporal raw (4)
        "hour":                  temp["hour"],
        "day_of_week":           temp["day_of_week"],
        "month":                 temp["month"],
        "is_weekend":            temp["is_weekend"],
        # Temporal cyclical (4)
        "hour_sin":              temp["hour_sin"],
        "hour_cos":              temp["hour_cos"],
        "month_sin":             temp["month_sin"],
        "month_cos":             temp["month_cos"],
        # Region (1)
        "region_enc":            region_code,
        # Ratios (4)
        "intrusion_ratio":       intrusion_ratio,
        "jamming_ratio":         jamming_ratio,
        "malware_ratio":         malware_ratio,
        "border_ratio":          border_ratio,
        # Interactions (5)
        "intrusion_x_malware":   intrusion_x_malware,
        "jamming_x_border":      jamming_x_border,
        "signal_x_scenarios":    signal_x_scenarios,
        "malware_x_border":      malware_x_border,
        "intrusion_x_border":    intrusion_x_border,
        # Log transforms (5)
        "log_intrusions":        log_intrusions,
        "log_jamming":           log_jamming,
        "log_malware":           log_malware,
        "log_signal":            log_signal,
        "log_total":             log_total,
        # Region target encodings (4)
        "region_te_cls0":        rte[0],
        "region_te_cls1":        rte[1],
        "region_te_cls2":        rte[2],
        "region_te_cls3":        rte[3],
    }

    df = pd.DataFrame([row])

    # Align to training 60-column order; fill missing columns with 0
    feat_cols = _models.get("cyber_feat_cols", [])
    if feat_cols:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feat_cols]

    ae_raw = {
        # 10-feature subset for Autoencoder (Model 6)
        "n_intrusions":          n_intrusions,
        "n_jamming":             n_jamming,
        "n_malware":             n_malware,
        "weighted_cyber_signal": weighted_cyber_signal,
        "events_near_border":    events_near_border,
        "n_active_scenarios":    n_scenarios,
        "hour_sin":              temp["hour_sin"],
        "hour_cos":              temp["hour_cos"],
        "month_sin":             temp["month_sin"],
        "month_cos":             temp["month_cos"],
    }

    return df, ae_raw


# ═════════════════════════════════════════════════════════════════════════════
#  LABEL HELPERS — map raw model output to 0–1 score
# ═════════════════════════════════════════════════════════════════════════════

# Midpoint of each threat band → used to convert label → float score
LABEL_SCORE = {
    "LOW":      0.20,
    "MODERATE": 0.52,
    "HIGH":     0.75,
    "CRITICAL": 0.92,
}
LABEL_IDX = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def _prob_to_score(probs: np.ndarray, labels: list) -> float:
    """
    Convert softmax probabilities into a single 0–1 score using
    the weighted midpoints of each threat band.
    """
    score = 0.0
    for i, lbl in enumerate(labels):
        score += probs[i] * LABEL_SCORE.get(lbl, 0.5)
    return float(np.clip(score, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
#  FUSION FEATURE EXTRACTION (Model 7 · 21 features)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_fusion_features(
    cybint_score:   float,
    socint_score:   float,
    geoint_score:   float,
    n_scenarios:    int,
    region_code:    int,
    temp:           dict,
    escalation_int: int,
) -> pd.DataFrame:
    """
    Build the 21-feature DataFrame for Model 7 (Fusion XGBoost Regressor).
    Monotone constraints: cybint(+), socint(+), geoint(+).
    """
    domain_scores = [cybint_score, socint_score, geoint_score]

    row = {
        # Core domain scores (3)
        "cybint_score":  cybint_score,
        "socint_score":  socint_score,
        "geoint_score":  geoint_score,
        # Domain statistics (5)
        "domain_max":    float(np.max(domain_scores)),
        "domain_min":    float(np.min(domain_scores)),
        "domain_std":    float(np.std(domain_scores)),
        "domain_range":  float(np.max(domain_scores) - np.min(domain_scores)),
        "mean_score":    float(np.mean(domain_scores)),
        # Elevation flags (6)
        "all_domains_elevated": int(all(s >= 0.40 for s in domain_scores)),
        "cyber_leads":          int(cybint_score == max(domain_scores)),
        "soc_leads":            int(socint_score == max(domain_scores)),
        "geo_leads":            int(geoint_score == max(domain_scores)),
        "n_high_domains":       int(sum(1 for s in domain_scores if s >= 0.65)),
        # Scenarios (1)
        "n_active_scenarios":   n_scenarios,
        # Cross-domain interaction (1)
        "cyber_geo_interaction": cybint_score * geoint_score,
        # Temporal cyclical (4)
        "hour_sin":    temp["hour_sin"],
        "hour_cos":    temp["hour_cos"],
        "month_sin":   temp["month_sin"],
        "month_cos":   temp["month_cos"],
        # Region + escalation (2)
        "region_code":            region_code,
        "escalation_pattern_int": escalation_int,
    }

    return pd.DataFrame([row])


# ═════════════════════════════════════════════════════════════════════════════
#  ASYNC — Model 2 BiLSTM Temporal (runs in background thread)
# ═════════════════════════════════════════════════════════════════════════════

def _async_lstm(session_id: str, geo_feats: dict, result_dict: dict) -> None:
    """
    Append the current geo feature vector to the per-session history buffer
    (max 10 steps) then run the BiLSTM if the buffer is full.
    Writes "lstm_trajectory" into result_dict.
    """
    model = _models.get("lstm")
    if model is None:
        result_dict["lstm_trajectory"] = "MODEL_UNAVAILABLE"
        return

    if not geo_feats:
        result_dict["lstm_trajectory"] = "FEATURE_ERROR"
        return

    # Build a 58-dim feature vector from geo_feats
    # The BiLSTM expects (None, 10, 58) — we use the 21 geo features + padding
    geo_keys = list(geo_feats.keys())
    vec = np.array([float(geo_feats.get(k, 0.0)) for k in geo_keys], dtype=np.float32)
    # Pad / truncate to 58 features
    if len(vec) < 58:
        vec = np.pad(vec, (0, 58 - len(vec)))
    else:
        vec = vec[:58]

    with _unit_history_lock:
        if session_id not in _unit_history:
            _unit_history[session_id] = deque(maxlen=10)  # 10-step window
        _unit_history[session_id].append(vec)
        history = list(_unit_history[session_id])

    # Only run inference when we have a full 10-step window
    if len(history) < 10:
        result_dict["lstm_trajectory"] = f"COLLECTING ({len(history)}/10)"
        return

    try:
        X      = np.array(history, dtype=np.float32)[np.newaxis, :, :]  # (1, 10, 58)
        output = model.predict(X, verbose=0)
        # TemporalAttention.call() returns a tuple (context, alpha), so Keras
        # may expose the model as multi-output: [probs_batch, alpha_batch].
        # Handle both single-output and multi-output cases defensively.
        if isinstance(output, (list, tuple)):
            probs_batch = output[0]   # (1, 4) — class probabilities
        else:
            probs_batch = output      # (1, 4) — single output model
        probs      = probs_batch[0]   # (4,) — first (only) sample
        LSTM_LABELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        pred_label  = LSTM_LABELS[int(np.argmax(probs))]
        result_dict["lstm_trajectory"] = pred_label
    except Exception as e:
        result_dict["lstm_trajectory"] = f"ERROR: {str(e)[:40]}"


# ═════════════════════════════════════════════════════════════════════════════
#  ASYNC — SBERT Embedding Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _async_sbert(alerts: list, result_dict: dict) -> None:
    """
    Compute SBERT embeddings for SOCINT alert messages and derive:
      coordination_index  — cosine similarity across alert embeddings (0–1)
      n_clusters          — number of distinct narrative clusters
      narrative_drift     — max cosine distance between successive alerts
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        msgs = [a.get("msg", "") for a in alerts if a.get("msg", "").strip()]
        if len(msgs) < 2:
            result_dict.update({
                "coordination_index": 0.0,
                "n_clusters":         0,
                "narrative_drift":    0.0,
                "sbert_score":        0.0,
            })
            return

        # Use cached model if already loaded, else load (slow first time)
        if "sbert_model" not in _models:
            _models["sbert_model"] = SentenceTransformer(
                "paraphrase-multilingual-mpnet-base-v2"
            )
        sbert = _models["sbert_model"]

        embeds = sbert.encode(msgs, normalize_embeddings=True)  # (N, 768)

        # Coordination index: mean pairwise cosine similarity
        sim_matrix = embeds @ embeds.T
        n          = len(msgs)
        off_diag   = [sim_matrix[i, j] for i in range(n) for j in range(n) if i != j]
        coord_idx  = float(np.mean(off_diag)) if off_diag else 0.0

        # Narrative drift: max cosine distance between adjacent pairs
        drifts = [
            1.0 - float(embeds[i] @ embeds[i + 1])
            for i in range(len(embeds) - 1)
        ]
        narrative_drift = float(max(drifts)) if drifts else 0.0

        # Cluster count via simple threshold (similarity > 0.7 → same cluster)
        visited   = [False] * n
        n_clusters = 0
        for i in range(n):
            if not visited[i]:
                n_clusters += 1
                for j in range(i + 1, n):
                    if sim_matrix[i, j] > 0.70:
                        visited[j] = True

        # SBERT-derived score: high coordination + high severity → higher score
        sbert_score = float(np.clip(coord_idx * 0.7 + (1 - narrative_drift) * 0.3, 0, 1))

        result_dict.update({
            "coordination_index": round(coord_idx,       4),
            "n_clusters":         n_clusters,
            "narrative_drift":    round(narrative_drift, 4),
            "sbert_score":        round(sbert_score,     4),
        })

    except Exception as e:
        result_dict.update({
            "coordination_index": None,
            "n_clusters":         None,
            "narrative_drift":    None,
            "sbert_score":        None,
            "sbert_error":        str(e)[:80],
        })


# ═════════════════════════════════════════════════════════════════════════════
#  ESCALATION PATTERN DETECTION
# ═════════════════════════════════════════════════════════════════════════════
# Matches the int encoding in the Fusion dataset:
#   FALSE_POSITIVE=0, GEO_FIRST=1, SOC_FIRST=2, CYBER_FIRST=3, SIMULTANEOUS=4

def _detect_escalation(
    geo: float, soc: float, cyb: float, threshold: float = 0.40
) -> tuple[str, int]:
    active = {"geo": geo >= threshold, "soc": soc >= threshold, "cyb": cyb >= threshold}
    count  = sum(active.values())

    if count == 0:
        return "FALSE_POSITIVE", 0
    if count == 3:
        return "SIMULTANEOUS", 4
    if active["geo"] and count == 1:
        return "GEO_FIRST",    1
    if active["soc"] and count == 1:
        return "SOC_FIRST",    2
    if active["cyb"] and count == 1:
        return "CYBER_FIRST",  3
    # Two domains active — pick dominant
    if active["cyb"]:
        return "CYBER_FIRST",  3
    if active["soc"]:
        return "SOC_FIRST",    2
    return "GEO_FIRST", 1


# ═════════════════════════════════════════════════════════════════════════════
#  THREAT LABEL from fused score
# ═════════════════════════════════════════════════════════════════════════════

def _fused_label(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.40:
        return "MODERATE"
    return "LOW"


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — run_inference(payload)
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(payload: dict) -> dict:
    """
    Full HYROX inference chain.

    Parameters
    ----------
    payload : dict
        The raw JSON body from the simulator's buildPayload().

    Returns
    -------
    dict
        All scores, labels, flags, and metadata ready for Streamlit.
    """
    result: dict = {
        # defaults (all zero / fallback) until each model runs
        "geoint":             0.0,
        "socmint":            0.0,
        "cyber":              0.0,
        "fused":              0.0,
        "geoint_label":       "LOW",
        "socmint_label":      "LOW",
        "cybint_label":       "LOW",
        "model1_label":       "LOW",
        "model5_label":       "LOW",
        "iso_anomaly":        False,
        "ae_anomaly":         False,
        "ae_mse":             None,
        "lstm_trajectory":    "PENDING",
        "coordination_index": None,
        "n_clusters":         None,
        "narrative_drift":    None,
        "sbert_score":        None,
        "fusion_method":      "Model 7 XGBoost (229 trees) · 0.40/0.30/0.30",
        "escalation_pattern": None,
        "model_status":       _models.get("model_status", {}),
        "models_loaded":      _models.get("models_loaded", "0/7"),
        "border_dist_km":     None,
        "mean_velocity_kmh":  None,
    }

    # ── Shared pre-processing ──────────────────────────────────────────────
    ts_str      = payload.get("timestamp", datetime.utcnow().isoformat())
    temp        = _temporal_features(ts_str)
    n_scenarios = len(payload.get("active_scenarios", []))
    session_id  = payload.get("session_id", "default")

    # Resolve region encoding
    raw_region  = payload.get("border_region", "")
    region_code = REGION_ENC.get(raw_region, 3)   # default to india-pak

    # ── MODEL 1: GEOINT XGBoost ────────────────────────────────────────────
    try:
        geo_feats = _extract_geoint_features(payload, region_code, n_scenarios)
    except Exception as e:
        print(f"[HYROX] geo feature extraction error: {e}")
        geo_feats = {}
    result["border_dist_km"]    = geo_feats.get("dist_from_border_km")
    result["mean_velocity_kmh"] = geo_feats.get("mean_velocity_kmh")

    geoint_score = 0.0
    try:
        import xgboost as xgb
        model1  = _models["geo_xgb_model"]
        scaler1 = _models["geo_xgb_scaler"]
        feats1  = _models["geo_xgb_feats"]
        labels1 = _models["geo_label_order"]   # ['LOW','MODERATE','HIGH']

        df1  = pd.DataFrame([{k: geo_feats.get(k, 0.0) for k in feats1}])
        X1   = scaler1.transform(df1.values)   # .values → numpy array, avoids sklearn feature-name warning
        dm1  = xgb.DMatrix(X1, feature_names=feats1)

        probs1 = model1.predict(dm1)[0]        # shape (3,) or (1, 3)
        if probs1.ndim > 1:
            probs1 = probs1[0]
        geoint_score = _prob_to_score(probs1, labels1)
        pred_idx1    = int(np.argmax(probs1))
        label1       = labels1[pred_idx1]

        result["geoint"]       = round(geoint_score, 4)
        result["geoint_label"] = label1
        result["model1_label"] = label1

    except Exception as e:
        print(f"[HYROX] Model 1 inference error: {e}")
        traceback.print_exc()

    # ── MODEL 3: SOCINT XGBoost ────────────────────────────────────────────
    try:
        soc_feats = _extract_socint_features(payload, region_code, n_scenarios, temp)
    except Exception as e:
        print(f"[HYROX] socint feature extraction error: {e}")
        soc_feats = {}
    socint_score = 0.0
    socint_label = "LOW"

    try:
        import xgboost as xgb
        model3 = _models["soc_xgb"]
        feat3  = _models["soc_feat_cols"]    # list of 23 names
        enc3   = _models["soc_label_enc"]    # {0:'LOW',...,3:'CRITICAL'}

        df3  = pd.DataFrame([{k: soc_feats.get(k, 0.0) for k in feat3}])
        dm3  = xgb.DMatrix(df3, feature_names=feat3)  # pass DataFrame, not .values

        probs3 = model3.predict(dm3)[0]
        if probs3.ndim > 1:
            probs3 = probs3[0]
        pred_idx3    = int(np.argmax(probs3))
        socint_label = enc3.get(pred_idx3, "LOW")

        SOCINT_LABELS = [enc3.get(i, f"C{i}") for i in range(len(probs3))]
        socint_score  = _prob_to_score(probs3, SOCINT_LABELS)

        result["socmint"]       = round(socint_score, 4)
        result["socmint_label"] = socint_label

    except Exception as e:
        print(f"[HYROX] Model 3 inference error: {e}")
        traceback.print_exc()

    # ── MODEL 4: SOCINT Isolation Forest ──────────────────────────────────
    try:
        iso_pipeline = _models["iso_pipeline"]
        df4          = _extract_isoforest_features(soc_feats, temp, region_code, socint_score)
        iso_pred     = iso_pipeline.predict(df4)[0]   # -1 = anomaly, 1 = normal
        result["iso_anomaly"] = bool(iso_pred == -1)

    except Exception as e:
        print(f"[HYROX] Model 4 inference error: {e}")

    # ── MODEL 5: CYBINT LightGBM ───────────────────────────────────────────
    cybint_score = 0.0
    try:
        df5, ae_raw = _extract_cybint_features(payload, region_code, n_scenarios, temp)
    except Exception as e:
        print(f"[HYROX] cybint feature extraction error: {e}")
        df5, ae_raw = pd.DataFrame(), {}

    try:
        import lightgbm as lgb
        model5  = _models["cyber_lgbm"]
        if df5.empty:
            raise ValueError("cybint feature DataFrame is empty")
        probs5  = model5.predict(df5)[0]    # shape (4,) probabilities

        CYBER_LABELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        cybint_score = _prob_to_score(probs5, CYBER_LABELS)
        pred_idx5    = int(np.argmax(probs5))
        cyber_label5 = CYBER_LABELS[pred_idx5]

        result["cyber"]        = round(cybint_score, 4)
        result["cybint_label"] = cyber_label5
        result["model5_label"] = cyber_label5

    except Exception as e:
        print(f"[HYROX] Model 5 inference error: {e}")
        traceback.print_exc()

    # ── MODEL 6: CYBINT Autoencoder (zero-day detection) ──────────────────
    try:
        ae_model     = _models["ae_model"]
        ae_scaler    = _models["ae_scaler"]
        ae_threshold = _models["ae_threshold"]

        # 10 features in fixed order matching scaler training
        AE_FEAT_ORDER = [
            "n_intrusions", "n_jamming", "n_malware",
            "weighted_cyber_signal", "events_near_border", "n_active_scenarios",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
        ]
        ae_vec    = np.array(
            [[ae_raw.get(k, 0.0) for k in AE_FEAT_ORDER]], dtype=np.float32
        )
        ae_scaled = ae_scaler.transform(ae_vec)
        ae_recon  = ae_model.predict(ae_scaled, verbose=0)
        ae_mse    = float(np.mean((ae_scaled - ae_recon) ** 2))

        result["ae_mse"]     = round(ae_mse, 7)
        result["ae_anomaly"] = bool(ae_mse > ae_threshold)

    except Exception as e:
        print(f"[HYROX] Model 6 inference error: {e}")

    # ── MODEL 7: Fusion XGBoost Regressor ─────────────────────────────────
    esc_pattern, esc_int = _detect_escalation(geoint_score, socint_score, cybint_score)
    result["escalation_pattern"] = esc_pattern

    fused_score = 0.0
    try:
        import xgboost as xgb
        model7 = _models["fusion_xgb"]

        df7 = _extract_fusion_features(
            cybint_score, socint_score, geoint_score,
            n_scenarios, region_code, temp, esc_int,
        )
        dm7 = xgb.DMatrix(df7, feature_names=list(df7.columns))  # pass DataFrame, not .values
        fused_score = float(np.clip(model7.predict(dm7)[0], 0.0, 1.0))

    except Exception as e:
        # Fallback: weighted sum matching Model 7's monotone weights
        fused_score = float(np.clip(
            0.40 * cybint_score + 0.30 * socint_score + 0.30 * geoint_score,
            0.0, 1.0,
        ))
        result["fusion_method"] = "FALLBACK weighted sum (0.40/0.30/0.30)"
        print(f"[HYROX] Model 7 inference error: {e}")

    result["fused"] = round(fused_score, 4)

    # ── ASYNC: Model 2 (BiLSTM) ────────────────────────────────────────────
    lstm_result: dict = {}
    t_lstm = threading.Thread(
        target=_async_lstm,
        args=(session_id, geo_feats, lstm_result),
        daemon=True,
    )
    t_lstm.start()

    # ── ASYNC: SBERT embedding pipeline ───────────────────────────────────
    sbert_result: dict = {}
    alerts = payload.get("social_intel", {}).get("alerts", [])
    t_sbert = threading.Thread(
        target=_async_sbert,
        args=(alerts, sbert_result),
        daemon=True,
    )
    t_sbert.start()

    # Wait for async tasks (capped at 8 s — won't block fast UI polling)
    t_lstm.join(timeout=8.0)
    t_sbert.join(timeout=8.0)

    result.update(lstm_result)
    result.update(sbert_result)

    return result