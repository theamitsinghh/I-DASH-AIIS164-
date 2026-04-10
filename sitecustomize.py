"""
sitecustomize.py — HYROX Apple Silicon segfault fix
====================================================
Python loads this file automatically before anything else runs —
before Streamlit, before TensorFlow, before every import in app.py.

ROOT CAUSE
----------
Streamlit 1.32 on macOS/Apple Silicon uses asyncio internally for
cache expiry. On Python 3.10 + macOS, the default event loop policy
(WindowsSelectorEventLoopPolicy on Win / default on Mac) leaves
coroutines unawaited when threads call async functions without a
running loop. The specific culprit is:

    RuntimeWarning: coroutine 'expire_cache' was never awaited

This unawaited coroutine causes a fatal signal (SIGSEGV) when
Streamlit's internal thread tries to clean up — hence the segfault.

FIX
---
1. Set the event loop policy to use a new loop per thread (safe on macOS).
2. Patch asyncio.get_event_loop() so threads that don't have a running
   loop automatically get a new one instead of raising RuntimeError.
3. Set TF / Metal env vars here too — absolute earliest possible point.
"""

import asyncio
import os
import sys

# ── 1. TensorFlow / Metal guards (earliest possible) ──────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",     "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",      "")
os.environ.setdefault("TF_METAL_DEVICE_PLACEMENT", "0")

# ── 2. Event loop policy fix for macOS + Python 3.10 ──────────────────────────
if sys.platform == "darwin":
    # Use the default policy but ensure every thread can get a loop
    _original_get_event_loop = asyncio.get_event_loop

    def _safe_get_event_loop():
        try:
            loop = _original_get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
            return loop
        except RuntimeError:
            # No running loop in this thread — create and set one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    asyncio.get_event_loop = _safe_get_event_loop
