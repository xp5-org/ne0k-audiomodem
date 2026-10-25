# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

"""micprobe.py -- capture-path diagnostic (single node).

Opens the selected input device (or PARAMS device_id) under a matrix of
configurations and reports what each actually delivers: frames seen, max
|sample|, RMS.  Distinguishes a dead device (zeros/no frames everywhere)
from a config-sensitive one (alive under some configs only).

Params: {"device_id": 5, "seconds": 1.5}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).parent

# --- category-aware sys.path bootstrap (auto-inserted) ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TESTS_ROOT = _HERE.parent if _HERE.name in ("CSS", "MESH", "OFDM", "QAM") else _HERE
_REPO_ROOT = _TESTS_ROOT.parent
for _p in (_TESTS_ROOT, _REPO_ROOT, _TESTS_ROOT / "CSS", _TESTS_ROOT / "MESH",
           _TESTS_ROOT / "OFDM", _TESTS_ROOT / "QAM", _TESTS_ROOT / "regression"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# --- end bootstrap ---

from testlib import CLIENT_ID, INPUT_DEV, PARAMS, log, to_int_dev  # noqa: E402


def grab(dev, fs, dtype, latency, seconds):
    import sounddevice as sd
    chunks = []

    def cb(indata, frames, t, status):
        chunks.append(np.asarray(indata[:, 0], dtype=np.float64).copy())

    kw = dict(samplerate=fs, channels=1, dtype=dtype, device=dev, callback=cb)
    if latency is not None:
        kw["latency"] = latency
    with sd.InputStream(**kw):
        time.sleep(seconds)
    if not chunks:
        return {"frames": 0, "max": None, "rms": None}
    x = np.concatenate(chunks)
    if dtype == "int16":
        x = x / 32768.0
    return {"frames": int(len(x)), "max": round(float(np.max(np.abs(x))), 6),
            "rms": round(float(np.sqrt(np.mean(x ** 2))), 6)}


def main():
    import sounddevice as sd
    seconds = float(PARAMS.get("seconds", 1.5))
    dev = PARAMS.get("device_id")
    dev = to_int_dev(INPUT_DEV) if dev is None else int(dev)

    info = sd.query_devices(dev)
    log(f"[{CLIENT_ID}] micprobe device {dev}: {info['name']!r} "
        f"native={info['default_samplerate']} in_ch={info['max_input_channels']}")

    results = {}
    for fs in (48000, 44100, 16000):
        for dtype in ("float32", "int16"):
            for latency in (None, "low", "high", 0.5):
                key = f"fs={fs} dtype={dtype} latency={latency}"
                try:
                    r = grab(dev, fs, dtype, latency, seconds)
                except Exception as exc:
                    r = {"error": str(exc)[:90]}
                results[key] = r
                log(f"  {key:44s} -> {r}")

    alive = [k for k, r in results.items()
             if r.get("max") not in (None, 0.0) and r.get("max", 0) > 0]
    verdict = ("ALIVE under: " + "; ".join(alive)) if alive else \
        "DEAD under every configuration (device/firmware/USB issue)"
    log(f"[{CLIENT_ID}] VERDICT: {verdict}")
    print(json.dumps({"client_id": CLIENT_ID, "status": "ok",
                      "device": int(dev), "alive_configs": alive,
                      "results": results}), flush=True)


if __name__ == "__main__":
    main()
