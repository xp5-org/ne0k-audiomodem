"""
css_listen.py  -  Framework wrapper for css_decoder.py (role: listen).

Handles audiotool coordination (waits for css_ready, signals css_armed),
spawns css_decoder.py as a subprocess, streams its structured stdout to the
server log, and collects the result.

WAV capture
-----------
The raw mic input is saved to  <RUN_DIR>/<CLIENT_ID>_input.wav  via the
css_decoder.py ``wav_out`` config key.  This WAV can be replayed offline with
csstest.py for post-hoc analysis or spectrograph generation.

This file is NOT meant to be run standalone - use css_decoder.py directly.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from analysis import make_css_spectrograph_png
from testlib import (
    CLIENT_ID, INPUT_DEV, RUN_DIR, PARAMS, RUN_ID, SERVER_URL,
    log, to_int_dev, coord_signal, coord_check, coord_wait_any,
)

TESTS_DIR = Path(__file__).parent


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

    # Wait for css_info.json to appear (written by the player)
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

    in_dev = to_int_dev(INPUT_DEV)

    # Derive timeout from the encoded audio duration
    try:
        info_data    = json.loads(info_path.read_text())
        audio_dur    = float(info_data.get("duration_s", 0))
        auto_timeout = audio_dur + 60 if audio_dur > 0 else 360
    except Exception:
        audio_dur    = 0
        auto_timeout = 360

    try:
        timeout_sec = float(PARAMS.get("timeout_sec", auto_timeout))
    except (TypeError, ValueError):
        timeout_sec = float(auto_timeout)
    log(f"  audio duration: {audio_dur:.1f}s  decode timeout: {timeout_sec:.0f}s")

    # WAV capture path — css_decoder.py writes the raw mic input here
    wav_capture_path = RUN_DIR / f"{CLIENT_ID}_input.wav"

    decode_cfg = {
        "info":              str(info_path),
        "result_out":        str(RUN_DIR / f"decode_result_{CLIENT_ID}.json"),
        "device":            in_dev,
        "timeout_sec":       timeout_sec,
        "tail_secs":         PARAMS.get("tail_secs",         5),
        "silence_threshold": PARAMS.get("silence_threshold", 0.002),
        "sync_threshold":    PARAMS.get("sync_threshold",    0.50),
        # Raw mic input WAV capture (for offline replay / spectrograph)
        "wav_out":           str(wav_capture_path),
        # DSP overrides – if set in params the decoder uses these over css_info.json
        **({k: PARAMS[k] for k in ("fs", "f_min", "f_max", "symbol_dur",
                                   "chunk_size", "preamble", "rs_nsym")
            if k in PARAMS}),
    }
    cfg_path = RUN_DIR / f"css_decode_config_{CLIENT_ID}.json"
    cfg_path.write_text(json.dumps(decode_cfg, indent=2))

    # Signal armed
    coord_signal("css_armed")
    log(f"  signalled css_armed - starting css_decoder.py...")

    decode_script = TESTS_DIR / "css_decoder.py"
    proc = subprocess.Popen(
        [sys.executable, str(decode_script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

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

    # Main thread: poll for css_played then stop decoder after tail
    tail_secs  = float(PARAMS.get("tail_secs", 5))
    stop_after = None
    poll_start = time.time()

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
        if time.time() - poll_start > timeout_sec + 30:
            log(f"  hard timeout - terminating decoder")
            proc.terminate()
            break
        time.sleep(0.5)

    reader.join(timeout=5)
    proc.wait()

    # Log WAV capture outcome
    if wav_capture_path.exists():
        size_kb = wav_capture_path.stat().st_size // 1024
        log(f"  WAV capture: {wav_capture_path.name}  ({size_kb} KB)")
    else:
        log(f"  WAV capture: not written (decoder may have exited early)")

    # Load result file
    result_path = RUN_DIR / f"decode_result_{CLIENT_ID}.json"
    try:
        result = json.loads(result_path.read_text())
    except Exception:
        try:
            result = json.loads(last_result_json) if last_result_json else {}
        except Exception:
            result = {"status": "error", "error": "could not read result"}

    # CSS spectrograph from the WAV capture
    spectrograph_path = RUN_DIR / f"{CLIENT_ID}_spectrograph.png"
    if wav_capture_path.exists():
        # Load css_info.json and result for overlays
        _info = None
        if info_path.exists():
            try:
                _info = json.loads(info_path.read_text())
            except Exception:
                pass
        make_css_spectrograph_png(
            wav_capture_path, spectrograph_path,
            info=_info, result=result,
        )

    result.update({"client_id": CLIENT_ID, "role": "listen", "run_id": RUN_ID})
    if wav_capture_path.exists():
        result["wav_capture"] = str(wav_capture_path)
    if spectrograph_path.exists():
        result["spectrograph_png"] = str(spectrograph_path)
    result_path.write_text(json.dumps(result, indent=2))

    log(f"  status={result.get('status')} frames={result.get('frames_decoded', '?')}")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()