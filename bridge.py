"""
╔══════════════════════════════════════════════════════════════════╗
║  bridge.py  —  HYROX In-Process Bridge Server                   ║
║                                                                  ║
║  Runs a Flask HTTP server inside a daemon thread so it shares   ║
║  the same Python process as Streamlit (app.py).                 ║
║                                                                  ║
║  ENDPOINTS                                                       ║
║    POST /api/hyrox/ingest   ← receives JSON from simulator       ║
║    GET  /state              ← polled by app.py every refresh     ║
║    GET  /health             ← simple liveness check              ║
║                                                                  ║
║  Called from app.py:                                             ║
║    import bridge                                                 ║
║    BRIDGE_PORT = bridge.start(5050)   # starts once via cache   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import threading
import logging
import time
import traceback
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS            # pip install flask-cors

import model_bridge                    # our inference module

# ── Silence Flask's noisy request logger ──────────────────────────────────────
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# ── Shared state dict — written by /ingest, read by /state ────────────────────
# Starts as NOT-ready so app.py shows all zeros until first dispatch.
_state: dict = {"ready": False}
_state_lock = threading.Lock()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)          # allow simulator on a different origin / LAN device


# ─────────────────────────────────────────────────────────────────────────────
# _validate_payload(payload) — pre-inference structural check
# Returns (True, None) on success or (False, "reason string") on failure.
# ─────────────────────────────────────────────────────────────────────────────
def _validate_payload(payload: dict):
    """
    Verify that the three top-level intelligence blocks are present and that
    the list-typed sub-fields are actually lists.  Does NOT check values —
    model_bridge is responsible for safe extraction.

    Returns
    -------
    (bool, str | None)
        (True, None)           → payload is acceptable
        (False, reason_string) → caller should return 400
    """
    required_top_keys = ("geo_intel", "social_intel", "cyber_intel")
    for key in required_top_keys:
        if key not in payload:
            return False, f"Missing required top-level key: '{key}'"
        if not isinstance(payload[key], dict):
            return False, f"'{key}' must be a JSON object, got {type(payload[key]).__name__}"

    # Each block has list-typed sub-fields that the models iterate over.
    list_fields = {
        "geo_intel":    ("units",),       # HTML sends units[] only
        "social_intel": ("alerts",),      # HTML sends alerts[]
        "cyber_intel":  ("alerts",),      # HTML sends alerts[] (was 'events' — fixed in simulator)
    }
    for block, fields in list_fields.items():
        block_data = payload[block]
        for field in fields:
            value = block_data.get(field)
            if value is not None and not isinstance(value, list):
                return False, (
                    f"'{block}.{field}' must be a list, "
                    f"got {type(value).__name__}"
                )

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/hyrox/ingest
# Receives the full JSON payload from the Leaflet simulator every 5 seconds.
# Runs full model inference chain and stores result in _state.
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/hyrox/ingest", methods=["POST"])
def ingest():
    """
    Accepts the simulator's buildPayload() JSON, validates structure,
    runs all 7 model inference steps, and persists the result for
    Streamlit to poll.

    Always returns a JSON response — never raises an unhandled exception.
    """
    # ── 1. Parse body ─────────────────────────────────────────────────────────
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as parse_exc:
        return jsonify({"error": f"JSON parse error: {parse_exc}"}), 400

    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Payload root must be a JSON object"}), 400

    # ── 2. Structural validation BEFORE inference ─────────────────────────────
    valid, reason = _validate_payload(payload)
    if not valid:
        return jsonify({"error": f"Payload validation failed: {reason}"}), 400

    # ── 3. Run inference (all exceptions caught — server stays alive) ─────────
    try:
        result = model_bridge.run_inference(payload)

        # Stamp result with raw payload for UI detail panels
        result["raw_payload"] = payload
        result["timestamp"]   = datetime.utcnow().isoformat() + "Z"
        result["ready"]       = True

        # Thread-safe write
        with _state_lock:
            _state.clear()
            _state.update(result)

        return jsonify({"status": "ok", "fused": result.get("fused", 0.0)}), 200

    except Exception as exc:
        # Full traceback to server logs; structured error to caller.
        traceback.print_exc()
        return jsonify({
            "error":  "Inference pipeline error",
            "detail": str(exc),
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /state
# Polled by app.py's generate_scores() on every Streamlit refresh.
# Returns the latest inference result or {"ready": False} if no dispatch yet.
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/state", methods=["GET"])
def state():
    try:
        with _state_lock:
            return jsonify(dict(_state)), 200
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": "State read error", "detail": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "uptime": datetime.utcnow().isoformat()}), 200


# ─────────────────────────────────────────────────────────────────────────────
# start(port) — called exactly once from app.py via @st.cache_resource
# Launches Flask in a self-restarting background daemon thread and returns
# the port number.
# ─────────────────────────────────────────────────────────────────────────────
def start(port: int = 5050) -> int:
    """
    Start the Flask bridge server in a background thread.

    The server loop is wrapped in ``while True / try-except`` so that any
    unexpected crash inside Flask itself causes an automatic restart after a
    short back-off delay — the daemon thread never exits permanently.

    Parameters
    ----------
    port : int
        The TCP port to listen on (must match BRIDGE_URL in app.py).

    Returns
    -------
    int
        The port the server is listening on (passed back for BRIDGE_URL).
    """
    # Pre-load all ML models once at startup so the first dispatch is fast.
    model_bridge.load_all_models()

    def _run():
        restart_delay = 2  # seconds to wait between crash-restarts
        attempt = 0

        while True:
            attempt += 1
            try:
                print(
                    f"[HYROX Bridge] Starting Flask server on port {port} "
                    f"(attempt #{attempt})"
                )
                # use_reloader=False is critical — reloader spawns child
                # processes which break the shared-memory _state dict.
                app.run(
                    host="0.0.0.0",
                    port=port,
                    use_reloader=False,
                    threaded=True,   # handle concurrent requests properly
                    debug=False,
                )
                # app.run() normally blocks forever; reaching this line means
                # Flask returned cleanly — still restart for safety.
                print("[HYROX Bridge] Flask exited cleanly — restarting.")
            except Exception as server_exc:
                print(
                    f"[HYROX Bridge] Server crashed (attempt #{attempt}): "
                    f"{server_exc}"
                )
                traceback.print_exc()

            # Back-off before retry so we don't spin-burn on a hard error
            # (e.g. port already in use on first boot — stops after OS frees it).
            print(f"[HYROX Bridge] Restarting in {restart_delay}s …")
            time.sleep(restart_delay)

    thread = threading.Thread(target=_run, daemon=True, name="hyrox-bridge")
    thread.start()

    print(f"[HYROX Bridge] Daemon thread started → http://0.0.0.0:{port}")
    return port
