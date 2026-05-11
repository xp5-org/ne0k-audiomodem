"""
tests/testlib.py  –  shared utilities for all AudioTool test scripts.

Centralises the boilerplate that was copy-pasted into every wrapper:
  · environment-variable parsing  (CLIENT_ID, RUN_DIR, PARAMS, …)
  · logging helper
  · coordination HTTP helpers     (coord_signal / coord_check / coord_wait_any)
  · device coercion               (to_int_dev)
  · WAV capture utility           (save_wav)

Usage in any test script::

    from testlib import (
        CLIENT_ID, INPUT_DEV, OUTPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
        log, to_int_dev,
        coord_signal, coord_check, coord_wait_any, coord_wait,
        save_wav,
    )
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Environment variables  (read once at import time)
# ---------------------------------------------------------------------------

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID",     "unknown")
INPUT_DEV  = os.environ.get("AUDIOTOOL_INPUT_DEVICE",  "")
OUTPUT_DEV = os.environ.get("AUDIOTOOL_OUTPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR",  "."))
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS",  "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID",        "unknown")
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL",    "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def to_int_dev(val) -> Optional[int]:
    """Coerce a device value to int, returning None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Coordination helpers
# ---------------------------------------------------------------------------

def coord_signal(signal: str) -> list:
    """POST this client's arrival on *signal*.  Returns list of all signalers."""
    import urllib.request
    url  = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
    body = json.dumps({"client_id": CLIENT_ID}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception as exc:
        log(f"  [coord] {signal} failed: {exc}")
        return []


def coord_check(signal: str) -> list:
    """GET current signaler list for *signal* (non-blocking)."""
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception:
        return []


def coord_wait_any(signal: str,
                   timeout: float = 30.0,
                   poll: float = 0.3) -> list:
    """Block until ≥ 1 client signals *signal* or *timeout* expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        clients = coord_check(signal)
        if clients:
            return clients
        time.sleep(poll)
    return []


def coord_wait(signal: str,
               timeout: float = 60.0,
               poll: float = 0.3) -> bool:
    """Block until *signal* appears or *timeout* expires.  Returns True on success."""
    return bool(coord_wait_any(signal, timeout=timeout, poll=poll))


# ---------------------------------------------------------------------------
# WAV capture
# ---------------------------------------------------------------------------

def save_wav(audio: np.ndarray, fs: int, path: "str | Path") -> None:
    """
    Save a float32 mono array (values in [-1, 1]) as a 16-bit WAV.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 samples.  Multi-channel arrays are mixed to mono by
        taking the first channel.
    fs    : int
        Sample rate in Hz.
    path  : str or Path
        Destination file path (will be created / overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = audio[:, 0] if (audio.ndim > 1) else audio.flatten()
    pcm  = (np.clip(flat.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

    duration = len(pcm) / max(fs, 1)
    log(f"  WAV capture -> {path.name}  ({duration:.1f}s @ {fs} Hz, {len(pcm)} samples)")
