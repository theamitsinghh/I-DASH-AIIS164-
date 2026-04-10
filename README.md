# HYROX — AI Hybrid Warfare Intelligence System

> **Team I-DASH · Defence Intelligence Hackathon 2026 · Version 2.0**  
> `UNCLASSIFIED — FOR EVALUATION PURPOSES ONLY`

Real-time, multi-domain AI intelligence fusion platform for border threat assessment. Ingests battlefield telemetry from a Leaflet simulator, routes each payload through a 7-model ML inference chain, and surfaces a live fused threat score on a tactical Streamlit dashboard within sub-100 ms.

---

## Wiring & Run Guide

### Port Map

| Port | Service |
|------|---------|
| `:8501` | Streamlit dashboard (`app.py`) |
| `:5050` | Flask bridge server (`bridge.py`, auto-started by `app.py`) |

---

### Mode A — Single Device

**Step 0 — Install dependencies (once)**

```bash
cd hyrox/
pip install -r requirements.txt
```

**Step 1 — Start the backend**

```bash
streamlit run app.py --server.port 8501
```

Wait until all 7 models confirm loaded in the terminal:

```
[HYROX] ✓ Model 1 (GEOINT XGBoost) loaded
[HYROX] ✓ Model 2 (BiLSTM Temporal) loaded
[HYROX] ✓ Model 3 (SOCINT XGBoost) loaded
[HYROX] ✓ Model 4 (Isolation Forest) loaded
[HYROX] ✓ Model 5 (CYBINT LightGBM) loaded
[HYROX] ✓ Model 6 (Autoencoder) loaded
[HYROX] ✓ Model 7 (Fusion XGBoost) loaded
[HYROX] Models ready: 7/7
[HYROX Bridge] Daemon thread started → http://0.0.0.0:5050
```

**Step 2 — Open the dashboard**

```
http://localhost:8501
```

**Step 3 — Open the simulator**

Open `final_simualtion.html` directly in your browser (File → Open File).  
The endpoint URL is pre-filled: `http://localhost:5050/api/hyrox/ingest` — do not change it.

**Step 4 — Start sending data**

- **Manual:** Click **SEND** to dispatch a single payload.  
- **Auto (recommended):** Toggle **AUTO-DISPATCH → ON**. The simulator POSTs every 5 seconds; the dashboard updates in near real-time.

**Step 5 — Verify**

The simulator's output panel should show `{ "status": "ok", "fused": 0.XXXX }` and the dashboard gauges should display non-zero scores.

---

### Mode B — Two-Device LAN

> **Device B** = backend (runs `app.py`, bridge, models)  
> **Device A** = simulator (opens `final_simualtion.html` in browser)  
> Both devices must be on the same Wi-Fi / LAN.

**Step 1 — Find Device B's LAN IP**

```bash
# macOS / Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig   # look for IPv4 Address under Wi-Fi adapter
```

Example IP used below: `10.217.234.83`

**Step 2 — Start the backend on Device B**

```bash
streamlit run app.py --server.host 0.0.0.0 --server.port 8501
```

`--server.host 0.0.0.0` is **required** — without it Device A cannot reach the dashboard.

**Step 3 — Open the dashboard from Device A (optional)**

```
http://10.217.234.83:8501
```

**Step 4 — Open the simulator on Device A**

Copy `final_simualtion.html` to Device A and open it in a browser.  
Change the endpoint URL from:

```
http://localhost:5050/api/hyrox/ingest
```

to:

```
http://10.217.234.83:5050/api/hyrox/ingest
```

**Step 5 — Open firewall ports on Device B**

```bash
# macOS
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add $(which python3)

# Linux (ufw)
sudo ufw allow 8501/tcp
sudo ufw allow 5050/tcp

# Windows
# Defender Firewall → Advanced Settings → New Inbound Rule → TCP → 8501, 5050 → Allow
```

**Step 6 — Start sending data**

Toggle **AUTO-DISPATCH → ON** on Device A. Watch the dashboard update on Device B.

---

### Data Flow

```
final_simualtion.html  (Leaflet simulator)
  └─ buildPayload() every 5 s
       geo_intel.units[]  |  social_intel.alerts[]  |  cyber_intel.alerts[]
       POST /api/hyrox/ingest
              │
              ▼
       bridge.py  —  Flask :5050
         validates payload → calls model_bridge.run_inference()
         stores result in _state dict
         GET /state  →  returns _state as JSON
              │
              ▼
       model_bridge.py
         M1 GEOINT XGBoost  →  M3 SOCINT XGBoost  →  M4 IsoForest
         M5 CYBINT LightGBM  →  M6 Autoencoder  →  M7 Fusion XGBoost
         M2 BiLSTM (async)  +  SBERT pipeline (async)
              │
              ▼
       app.py  —  Streamlit :8501
         generate_scores() → GET /state → renders dashboard
```

---

### Quick Reference — URLs

| | Single Device | Two Device (replace `10.X.X.X`) |
|---|---|---|
| Dashboard | `http://localhost:8501` | `http://10.X.X.X:8501` |
| Ingest endpoint | `http://localhost:5050/api/hyrox/ingest` | `http://10.X.X.X:5050/api/hyrox/ingest` |
| Health check | `http://localhost:5050/health` | `http://10.X.X.X:5050/health` |
| State | `http://localhost:5050/state` | `http://10.X.X.X:5050/state` |

---

### Troubleshooting

**Dashboard shows all zeros after dispatch**  
Check the simulator OUTPUT panel. `{ "status": "ok" }` means the pipeline is running — try a manual browser refresh. If it shows `CONNECTION FAILED`, the bridge URL is wrong or the bridge failed to start (check the terminal for model load errors).

**CONNECTION FAILED in simulator**  
Single device: confirm URL is `http://localhost:5050/api/hyrox/ingest`.  
Two device: confirm you replaced `localhost` with Device B's LAN IP.  
Quick test: `curl -X POST http://<Device-B-IP>:5050/health` — should return `{"status": "ok", ...}`.

**Segfault on startup (Apple Silicon)**  
TF env guards are set at the top of `model_bridge.py`. If it still crashes, run again immediately — macOS releases the Metal device between attempts and the second launch succeeds.

**Port 5050 already in use**  
```bash
lsof -ti:5050 | xargs kill -9
streamlit run app.py
```

**Model X failed to load**  
Verify the `models/` folder structure matches the layout below exactly (folder names, file names, extensions). Note: the `fusion engine` folder name contains a space — this is intentional.

---

## Repository Structure

```
hyrox/
├── app.py                          Streamlit dashboard (~97 KB)
├── bridge.py                       Flask bridge server, daemon thread
├── model_bridge.py                 7-model inference pipeline (~62 KB)
├── socint_embeddings.py            SBERT + XLM-RoBERTa embedding pipeline
├── sitecustomize.py                TF/Metal env guards + asyncio patch (macOS)
├── final_simualtion.html           Standalone Leaflet simulator (~74 KB)
├── requirements.txt
│
├── models/
│   ├── geoint/
│   │   └── geoint_xgb_model.pkl            Model 1 — GEOINT XGBoost
│   ├── temporal/
│   │   └── best_model.keras                Model 2 — BiLSTM + Attention
│   ├── socint/
│   │   ├── model3_xgb_socint.json          Model 3 — SOCINT XGBoost
│   │   ├── feature_columns.pkl             23 SOCINT feature names
│   │   ├── label_encoder.pkl               {0:'LOW'..3:'CRITICAL'}
│   │   └── model4_isolation_forest.joblib  Model 4 — Isolation Forest
│   ├── cybint/
│   │   ├── model5_lgbm.txt                 Model 5 — CYBINT LightGBM
│   │   ├── model6_autoencoder.h5           Model 6 — Autoencoder
│   │   ├── model6_scaler.pkl               StandardScaler
│   │   └── model6_threshold.npy            MSE anomaly threshold
│   └── fusion engine/
│       └── model7_xgb_fusion.json          Model 7 — Fusion XGBoost
│
├── datasets/
│   ├── geoint_cluster_v2.csv       52,480 rows, 21 features, 3-class labels
│   ├── cybint_events.csv           ~150K raw cyber events
│   ├── cybint_clean_train.csv      40K processed, 60 features, 4 classes
│   ├── socint_alerts.csv           ~100K raw SOCINT alerts
│   ├── fusion_clean_train.csv      ~80K pre-fused training set for M7
│   ├── SOCMINT/model4_*.{csv,joblib}
│   └── LSTM/X_*.npy  y_*.npy      BiLSTM sequences, shape (N, 10, 58)
│
├── notebook/                       Training notebooks for all 4 domains
├── reference/
│   └── feature_columns.txt         60 CYBINT feature names for M5
├── .streamlit/
│   └── config.toml                 Disables file watcher; log level = error
└── logs/geoint/train/              TensorBoard event files (BiLSTM)
```

---

## System Overview

HYROX (Hybrid Reconnaissance & Operations eXecution) fuses three intelligence domains in real time to detect pre-attack hybrid warfare patterns across four Indian border theatres.

| Theatre | Coordinates | Key Threats |
|---------|------------|-------------|
| India–Pakistan LoC | 34.10°N, 74.80°E | Armour density, cyber probing, disinformation |
| India–China LAC | 33.50°N, 79.50°E | Drone surveillance, rapid mobilisation, propaganda |
| India–Bangladesh | 23.80°N, 91.00°E | Informant networks, smuggling-linked movement |
| India–Arunachal Pradesh | 27.80°N, 93.60°E | Drone sightings, mountainous terrain constraints |

### Three-Tier Architecture

| Tier | Component | Role |
|------|-----------|------|
| 1 — Simulation | `final_simualtion.html` | Generates GEOINT/SOCINT/CYBINT JSON payloads via Leaflet UI; dispatches every 5 s |
| 2 — Inference | `bridge.py` + `model_bridge.py` | Validates payloads, runs 7-model chain (~50–100 ms), stores result in thread-safe state |
| 3 — Dashboard | `app.py` (Streamlit) | Polls `/state` on every refresh; renders live scores, heat-maps, CYBINT logs, SOCINT alerts |

---

## Machine Learning Models

Seven models run in a structured sequential inference chain. Models 1 and 3–7 are synchronous on the critical path; Model 2 (BiLSTM) and the SBERT pipeline execute asynchronously.

| # | Model | Algorithm | Key Metric | Score |
|---|-------|-----------|------------|-------|
| M1 | GEOINT XGBoost | XGBoost | ROC-AUC (3-class) | 0.617 |
| M2 | BiLSTM Temporal | BiLSTM + Attention | Architecture | 2×BiLSTM + Attention |
| M3 | SOCINT XGBoost | XGBoost | Macro F1 | 0.9325 |
| M4 | SOCINT Isolation Forest | IsolationForest | AUC vs CRITICAL | 0.9552 |
| M5 | CYBINT LightGBM | LightGBM | AUC-OvR | 0.9861 |
| M6 | CYBINT Autoencoder | Autoencoder | Threshold-based MSE | — |
| M7 | Fusion XGBoost | XGBoost Regressor | Output range | 0.0 – 1.0 |

### Inference Chain

```
Payload → Feature Extraction
  → M1 GEOINT XGBoost      (geo_probs[3], geo_score)
  → M3 SOCINT XGBoost      (soc_class, soc_probs[4])
  → M4 Isolation Forest    (iso_score, iso_anomaly)
  → M5 CYBINT LightGBM     (cyber_probs[4], cyber_class)
  → M6 Autoencoder         (ae_anomaly, ae_mse)
  → M7 Fusion XGBoost      (fused_score ∈ [0,1])
  ↗ M2 BiLSTM (async)      (temporal_probs[4])
  ↗ SBERT pipeline (async) (coordination_index, n_clusters)
```

### Threat Level Thresholds

| Level | Fused Score |
|-------|------------|
| CRITICAL | ≥ 0.85 |
| HIGH | 0.65 – 0.84 |
| MEDIUM | 0.40 – 0.64 |
| LOW | < 0.40 |

---

## Installation

### Requirements

| Requirement | Specification |
|-------------|--------------|
| Python | 3.10 or 3.11 (CPython) |
| RAM | Minimum 8 GB, recommended 16 GB |
| OS | macOS 12+, Ubuntu 20.04+, Windows 10/11 (WSL2 recommended) |
| Browser | Chrome 90+, Firefox 90+, Safari 15+ |
| Ports | `localhost:5050` (bridge) + `localhost:8501` (Streamlit) |

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `streamlit==1.32.0`, `plotly==5.19.0`, `folium==0.16.0`, `flask`, `xgboost`, `lightgbm`, `scikit-learn`, `keras`, `tensorflow`, `sentence-transformers`, `torch`, `pandas`, `numpy`.

### Apple Silicon (M1/M2/M3/M4)

TensorFlow env guards are already applied at the top of `model_bridge.py` and `sitecustomize.py`. No additional configuration is required.

---

## Bridge API Reference

### `POST /api/hyrox/ingest`

Accepts the simulator JSON payload. Returns `{ "status": "ok", "fused": 0.723 }` on success, `HTTP 400` on validation failure.

### `GET /state`

Returns the latest inference result when data is available:

```json
{
  "ready": true,
  "cyber": 0.61, "socmint": 0.78, "geoint": 0.54, "fused": 0.72,
  "cybint_label": "HIGH", "socmint_label": "MEDIUM", "geoint_label": "HIGH",
  "ae_anomaly": false, "iso_anomaly": false,
  "ae_mse": 0.0031, "iso_score": 0.52,
  "fusion_method": "Weighted (0.40/0.30/0.30)",
  "models_loaded": "7/7",
  "timestamp": "2026-04-10T08:22:11Z"
}
```

Returns `{ "ready": false }` before the first dispatch.

### `GET /health`

Returns `{ "status": "ok", "uptime": "<ISO timestamp>" }`.

---

## Known Limitations

- **GEOINT accuracy (M1):** 0.486 accuracy / 0.220 macro F1 due to class imbalance. M1 functions as a probabilistic signal supplier to M7, not a standalone classifier.
- **Synthetic data:** All datasets are synthetically generated. Operational deployment requires retraining on verified intelligence data.
- **Single-device architecture:** The in-process Flask bridge is not horizontally scalable. Multi-instance deployments require Redis Pub/Sub or a message queue.
- **No persistent storage:** Timeline data lives in Streamlit `session_state` only; server restart clears history.
- **Static fusion weights:** Retraining any upstream model (M1–M6) without retraining M7 may degrade fusion accuracy.

---

## Roadmap

- Persistent Redis state backend for distributed deployment and durable history
- SIGINT as a fourth domain (Model 8) feeding the fusion layer
- Bayesian online learning for M7 adaptive fusion weights
- Live SHAP explainability panel (pre-computed M5 SHAP values available)
- JWT / API-key authentication on `/api/hyrox/ingest`
- Docker Compose containerisation (bridge + dashboard + model server)
- LSTM-Fusion integration: feed `temporal_probs` from M2 into M7 on the critical path

---

`CLASSIFICATION: UNCLASSIFIED — FOR EVALUATION PURPOSES ONLY`
