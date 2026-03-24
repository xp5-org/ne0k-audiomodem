"""
css_listen.py  -  Framework wrapper for css_decoder.py (role: listen).

Handles audiotool coordination (waits for css_ready, signals css_armed),
spawns css_decoder.py as a subprocess, streams its structured stdout to the
server log, and collects the result.

This file is NOT meant to be run standalone - use css_decoder.py directly.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CLIENT_ID  = os.environ.get("AUDIOTOOL_CLIENT_ID", "unknown")
INPUT_DEV  = os.environ.get("AUDIOTOOL_INPUT_DEVICE", "")
RUN_DIR    = Path(os.environ.get("AUDIOTOOL_RUN_DIR", "."))
PARAMS     = json.loads(os.environ.get("AUDIOTOOL_PARAMS", "{}"))
RUN_ID     = os.environ.get("AUDIOTOOL_RUN_ID", "unknown")
SERVER_URL = os.environ.get("AUDIOTOOL_SERVER_URL", "http://127.0.0.1:5000")
TESTS_DIR  = Path(__file__).parent


def log(msg):
    print(msg, flush=True)


def coord_signal(signal):
    import urllib.request
    url  = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
    body = json.dumps({"client_id": CLIENT_ID}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log(f"  [coord] {signal} failed: {e}")


def coord_wait_any(signal, timeout=30, poll=0.3):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
            with urllib.request.urlopen(url, timeout=5) as r:
                clients = json.loads(r.read()).get("clients", [])
            if clients:
                return clients
        except Exception:
            pass
        time.sleep(poll)
    return []


def coord_check(signal):
    import urllib.request
    try:
        url = f"{SERVER_URL}/api/coord/{RUN_ID}/{signal}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("clients", [])
    except Exception:
        return []


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log(f"[{CLIENT_ID}] css_listen: waiting for player css_ready signal...")
    ready = coord_wait_any("css_ready", timeout=30)
    if not ready:
        log(f"  ERROR: no css_ready signal within 30 s")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "failure_reason": "no_ready_signal", "run_id": RUN_ID}), flush=True)
        return

    log(f"  player ready: {ready}")

    # Wait for css_info.json to appear (single stat() call, no glob)
    info_path = RUN_DIR / "css_info.json"
    deadline  = time.time() + 10
    while time.time() < deadline:
        if info_path.exists():
            break
        time.sleep(0.2)
    else:
        log(f"  ERROR: css_info.json not found after ready signal")
        print(json.dumps({"client_id": CLIENT_ID, "status": "error",
                          "failure_reason": "no_info_file", "run_id": RUN_ID}), flush=True)
        return

    # Build config for css_decoder.py
    try:
        in_dev = int(INPUT_DEV)
    except (TypeError, ValueError):
        in_dev = None

    # Derive timeout from the actual encoded audio duration so we never cut off early.
    # css_info.json has encoding.duration_s written by css_encoder.py.
    # Add 60 s of headroom for sync preamble + network/processing jitter.
    try:
        info_data    = json.loads(info_path.read_text())
        # duration_s is written directly into css_info.json by css_encoder.py
        audio_dur    = float(info_data.get("duration_s", 0))
        auto_timeout = audio_dur + 60 if audio_dur > 0 else 360
    except Exception:
        audio_dur    = 0
        auto_timeout = 360   # safe fallback if info unreadable

    # User can still override via params, but default is now duration-aware
    timeout_sec = float(PARAMS.get("timeout_sec", auto_timeout))
    log(f"  audio duration: {audio_dur:.1f}s  decode timeout: {timeout_sec:.0f}s")

    decode_cfg = {
        "info":              str(info_path),
        "result_out":        str(RUN_DIR / f"decode_result_{CLIENT_ID}.json"),
        "device":            in_dev,
        "timeout_sec":       timeout_sec,
        "tail_secs":         PARAMS.get("tail_secs",          5),
        "silence_threshold": PARAMS.get("silence_threshold",  0.002),
        "sync_confidence":   PARAMS.get("sync_confidence",    1.5),
        # DSP overrides - if set in params, decoder uses these instead of css_info.json values
        # Useful for changing symbol_dur without re-encoding
        **({k: PARAMS[k] for k in ("fs","f_min","f_max","symbol_dur","chunk_size","preamble","rs_nsym","sync_threshold")
            if k in PARAMS}),
    }
    cfg_path = RUN_DIR / f"css_decode_config_{CLIENT_ID}.json"
    cfg_path.write_text(json.dumps(decode_cfg, indent=2))

    # Signal armed
    coord_signal("css_armed")
    log(f"  signalled css_armed - starting css_decoder.py...")

    # Spawn css_decoder.py
    decode_script = TESTS_DIR / "css_decoder.py"
    proc = subprocess.Popen(
        [sys.executable, str(decode_script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Read stdout in a background thread so we can watch for css_played in parallel
    import threading
    last_result_json = None
    output_lines = []

    def _read_stdout():
        nonlocal last_result_json
        for line in proc.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if line.startswith("{"):
                last_result_json = line
            else:
                log(f"  [decode] {line}")

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    # Main thread: poll for css_played signal then stop decoder after tail
    tail_secs   = float(PARAMS.get("tail_secs", 5))
    stop_after  = None
    poll_start  = time.time()

    while proc.poll() is None:
        if stop_after is None:
            played = coord_check("css_played")
            if played:
                stop_after = time.time() + tail_secs
                log(f"  css_played received - stopping decoder after {tail_secs}s tail")
        elif time.time() >= stop_after:
            log(f"  tail elapsed - terminating decoder")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        # Also respect hard timeout
        if time.time() - poll_start > timeout_sec + 30:
            log(f"  hard timeout - terminating decoder")
            proc.terminate()
            break
        time.sleep(0.5)

    reader.join(timeout=5)
    proc.wait()

    # Load result file (more reliable than parsing last stdout line)
    result_path = RUN_DIR / f"decode_result_{CLIENT_ID}.json"
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        # Fall back to last JSON line from stdout
        try:
            result = json.loads(last_result_json) if last_result_json else {}
        except Exception:
            result = {"status": "error", "error": "could not read result"}

    result.update({"client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID})
    result_path.write_text(json.dumps(result, indent=2))

    log(f"  status={result.get('status')} frames={result.get('frames_decoded', '?')}")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()